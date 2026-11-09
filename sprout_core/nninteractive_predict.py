import numpy as np
import pandas as pd
import torch
import tifffile
from pathlib import Path
from typing import TypedDict, Literal, Optional, Union, Dict, Tuple
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from sprout_core.sprout_prompt_core import seed_to_point_prompts_nninteractive
from sprout_core.nni_engine import make_session
from sprout_core.sprout_core import label_dtype
import os, sys
import yaml
import sprout_core.config_core as config_core
from sprout_core.nni_prob_session import nnInteractiveSessionWithProb
from sprout_core.seed_roi import seed_rois, inner_face_fraction, roi_log

# inner_face_fraction above this -> the prediction touches the ROI edge
ROI_EDGE_WARN_FRACTION = 0.02


def parse_scribble_config(optional_params):
    scribble_config = {
        'use_negative_scribble': optional_params.get('use_negative_scribble', False)    
    }
    return scribble_config

def parse_point_config(optional_params):
    point_config = {
        'default_n_pos': optional_params['default_n_pos'],
        'default_n_neg': optional_params['default_n_neg'],
        'default_method': optional_params['default_method'],
        'negative_from_bg': optional_params['negative_from_bg'],
        'negative_from_other_classes': optional_params['negative_from_other_classes'],
        'negative_per_other_class': optional_params['negative_per_other_class'],
        "class_config": optional_params.get('class_config', None),
    }
    return point_config

def nninter_main_from_prompts(model_path, img_path,output_folder,
                 df_pt_path = None, scribble_mask_path = None,
                 
                 return_per_class_masks: bool = False,
                 device = None,
                 seed: int = 0):
    """
    Main function for nnInteractive prediction with point prompts.
    
    Args:
        model_path: Path to trained model folder
        img_path: Path to input TIFF image
        df_pt_path: Path to CSV file with point prompts
        scribble_mask_path: Path to scribble mask image
        output_folder: Folder to save output segmentation
        return_per_class_masks: Whether to save individual class masks
        device: "cuda" / "cpu" / torch.device; None = cuda if available else cpu
        seed: random seed: np.random.seed(seed + class_id) before each class (see nninter_predict)
    
    Returns:
        total_mask or (total_mask, per_class_masks)
    """
    print("=" * 60)
    print("nnInteractive Prediction")
    print("=" * 60)
    
    # if both df_pt_path and scribble_mask_path are None, raise error
    if df_pt_path is None and scribble_mask_path is None:
        raise ValueError("Must provide at least one of: df_pt_path or scribble_mask_path")
    
    # Validate paths
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not Path(img_path).exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    if df_pt_path is not None and not Path(df_pt_path).exists():
        raise FileNotFoundError(f"Prompt CSV not found: {df_pt_path}")
    if scribble_mask_path is not None and not Path(scribble_mask_path).exists():
        raise FileNotFoundError(f"Scribble mask not found: {scribble_mask_path}")
    
    # Create output directory
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # 1. Initialize session
    print("\n[1/4] Initializing model...")
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"      Device: {device}")
    
    # keyword arguments filtered for the installed nnInteractive (2.x has no use_pinned_memory)
    session = make_session(device, use_pinned_memory=True)
    
    try:
        session.initialize_from_trained_model_folder(model_path)
        print("      ✓ Model loaded")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
    # 2. Load image
    print("\n[2/4] Loading image...")
    try:
        img = tifffile.imread(img_path)
        print(f"      Shape: {img.shape}, dtype: {img.dtype}")
        
        # Ensure correct dimensions
        if img.ndim == 3:
            img = img[None]  # Shape: (1, x, y, z)
        elif img.ndim == 2:
            img = img[None, None]  # Shape: (1, 1, y, x)
        elif img.ndim != 4:
            raise ValueError(f"Unsupported image dimensions: {img.ndim}D")
        
        print(f"      Final shape: {img.shape}")
    except Exception as e:
        raise RuntimeError(f"Failed to load image: {e}")
    
    # 3. Load prompts
    print("\n[3/4] Loading prompts...")
    if df_pt_path is not None:
        df_pt = pd.read_csv(df_pt_path)
        
        # Validate CSV columns
        expected_columns = {'x', 'y', 'z', 'label', 'class_id'}
        if not expected_columns.issubset(set(df_pt.columns)):
            missing = expected_columns - set(df_pt.columns)
            raise ValueError(f"CSV missing columns: {missing}")
        
        # Validate coordinates
        _validate_coordinates(df_pt, img.shape[1:])
    else:
        df_pt = None
    
    if scribble_mask_path is not None:
        scribble_mask = tifffile.imread(scribble_mask_path)
        
        # Validate scribble shape
        expected_shape = img.shape[1:]
        if scribble_mask.shape != expected_shape:
            raise ValueError(
                f"Scribble mask shape {scribble_mask.shape} doesn't match "
                f"image shape {expected_shape}"
            )
        
        print(f"      Scribble mask shape: {scribble_mask.shape}, unique values: {np.unique(scribble_mask)}")
    else:
        scribble_mask = None
    
    print(f"      Total points: {len(df_pt) if df_pt is not None else 0}")
    print(f"      Total classes: {len(df_pt['class_id'].unique()) if df_pt is not None else 0}")
    print(f"      Total points Classes: {sorted(df_pt['class_id'].unique()) if df_pt is not None else []}")

    
    # 4. Run prediction
    print("\n[4/4] Running prediction...")
    per_class_masks = {} if return_per_class_masks else None

    def collect_class_mask(class_id, mask):
        per_class_masks[int(class_id)] = mask

    total_mask = nninter_predict(
        session, img, df_pt=df_pt, scribble_mask=scribble_mask,
        on_class_predicted=collect_class_mask if return_per_class_masks else None,
        seed=seed
    )
    
    # Save results
    print("\n      Saving results...")
    output_path = Path(output_folder) / f"{Path(img_path).stem}_segmentation.tif"
    tifffile.imwrite(output_path, total_mask, compression='zlib')
    print(f"      ✓ Saved: {output_path}")
    
    if return_per_class_masks:
        for class_id, mask in per_class_masks.items():
            output_path = Path(output_folder) / f"{Path(img_path).stem}_class_{class_id}.tif"
            tifffile.imwrite(output_path, mask.astype(np.uint8) * 255, compression='zlib')
            print(f"      ✓ Saved class {class_id}: {mask.sum():,} pixels")
    
    # Cleanup
    del session
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("✓ Completed successfully!")
    print("=" * 60 + "\n")
    
    if return_per_class_masks:
        return total_mask, per_class_masks
    else:
        return total_mask


class PointPromptConfig(TypedDict, total=False):
    """Configuration for point prompt generation."""
    class_config: Optional[dict]
    default_n_pos: int
    default_n_neg: int
    default_method: Literal['random', 'kmeans', 'center_edge', 'grid']
    negative_from_bg: bool
    negative_from_other_classes: bool
    negative_per_other_class: bool
    seed_pattern: str

def nninter_main(model_path, img_path, seg_path ,output_folder,device,
                 init_seg_path = None,
                prompt_type: Literal['point', 'scribble'] = "point",
                point_config: Optional[PointPromptConfig] = None,
                 return_per_class_masks: bool = True,
                 scribble_config= None,
                 save_prob_maps: bool = False,
                 do_autozoom: bool = True,
                 seed_roi: Literal['off', 'postmask', 'crop'] = "off",
                 seed_roi_margin=32,
                 seed_roi_margin_unit: Literal['voxel', 'percent'] = "voxel",
                 seed: int = 0):
    """
    Main function for nnInteractive prediction with point prompts.
    
    Args:
        model_path: Path to trained model folder
        img_path: Path to input TIFF image
        seg_path: Path to input segmentation image (seed)
        output_folder: Folder to save output segmentation
        device: Device to run inference on ("cuda" or "cpu")
        init_seg_path: Optional path to initial segmentation mask for iterative refinement
        prompt_type: Type of prompt to use ("point" or "scribble")
        point_config: Configuration for point prompt generation
        return_per_class_masks: Whether to save individual class masks
        scribble_config: Configuration for scribble prompt generation
        save_prob_maps: Whether to save per-class foreground probability maps
        do_autozoom: Let nnInteractive zoom out when the prediction reaches the patch border
        seed_roi: Per-class ROI = the class's seed bbox + seed_roi_margin.
                  'off' (default): not used.
                  'postmask': predict on the full image, then clear each class's prediction
                              outside its ROI (removes leaks into other parts).
                  'crop': nnInteractive only sees the ROI crop (faster, less memory, but
                          less accurate with scribble prompts). Outputs are pasted back to
                          full size, zero outside the ROI.
        seed_roi_margin: Margin around the seed bbox, a number or [z, y, x]
        seed_roi_margin_unit: 'voxel', or 'percent' of the seed bbox size per axis
        seed: random seed (default 0): np.random.seed(seed) before the point sampling and
              np.random.seed(seed + class_id) before each class, so a run is reproducible and a
              class's result does not depend on which other classes run
    Returns:
        total_mask or (total_mask, per_class_masks)
    """
    print("=" * 60)
    print("nnInteractive Prediction")
    # Print all parameters
    print(f"    Model path: {model_path}")
    print(f"    Image path: {img_path}")
    print(f"    Segmentation path: {seg_path}")
    print(f"    Output folder: {output_folder}")
    print(f"    Device: {device}")
    print(f"    Prompt type: {prompt_type}")
    print(f"    Point config: {point_config}")
    print(f"    Return per class masks: {return_per_class_masks}")
    print(f"    Scribble config: {scribble_config}")
    print(f"    Save probability maps: {save_prob_maps}")
    print(f"    Do autozoom: {do_autozoom}")
    if seed_roi not in ("off", "postmask", "crop"):
        raise ValueError(f"seed_roi must be 'off', 'postmask' or 'crop', got {seed_roi!r}")
    print(f"    Seed ROI: {seed_roi}" + (f", margin: {seed_roi_margin} {seed_roi_margin_unit}"
                                         if seed_roi != "off" else ""))
    print("=" * 60)

    log_dict = {
        "img_path": img_path,
        "seg_path": seg_path,
        "output_folder": output_folder,
        "init_seg_path": init_seg_path,
        "do_autozoom": do_autozoom,
        "seed_roi": seed_roi,
        "seed_roi_margin": list(seed_roi_margin) if isinstance(seed_roi_margin, (list, tuple)) else seed_roi_margin,
        "seed_roi_margin_unit": seed_roi_margin_unit,
        "seed": seed}
  
    if torch.cuda.is_available() and ("cuda" in str(device)):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(torch.device("cuda"))
            
    # Validate paths
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not Path(img_path).exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not Path(seg_path).exists():
        raise FileNotFoundError(f"Segmentation image not found: {seg_path}")
    if init_seg_path is not None and not Path(init_seg_path).exists():
        raise FileNotFoundError(f"Initial segmentation not found: {init_seg_path}")
    
    # Create output directory
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # 1. Initialize session
    print("\n[1/4] Initializing model...")
    device = torch.device(device)
    print(f"      Device: {device}")
    
    # Use probability-tracking session if save_prob_maps is enabled
    SessionClass = nnInteractiveSessionWithProb if save_prob_maps else nnInteractiveInferenceSession
    # keyword arguments filtered for the installed nnInteractive (2.x has no use_pinned_memory)
    session = make_session(device, session_cls=SessionClass, do_autozoom=do_autozoom,
                           use_pinned_memory=False)
    
    try:
        session.initialize_from_trained_model_folder(model_path)
        print("      ✓ Model loaded")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
    # 2. Load image
    print("\n[2/4] Loading image...")
    try:
        img = tifffile.imread(img_path)
        print(f"      Shape: {img.shape}, dtype: {img.dtype}")
        
        # Ensure correct dimensions
        if img.ndim == 3:
            img = img[None]  # Shape: (1, x, y, z)
        elif img.ndim == 2:
            img = img[None, None]  # Shape: (1, 1, y, x)
        elif img.ndim != 4:
            raise ValueError(f"Unsupported image dimensions: {img.ndim}D")
        
        print(f"      Final shape: {img.shape}")
    except Exception as e:
        raise RuntimeError(f"Failed to load image: {e}")

    # Seed ROIs: one per class, seed bbox + margin
    seed_img, rois, roi_stats = None, None, None
    if seed_roi != "off":
        seed_img = tifffile.imread(seg_path)
        if seed_img.shape != img.shape[1:]:
            raise ValueError(f"Seed shape {seed_img.shape} doesn't match image shape {img.shape[1:]}")
        rois = seed_rois(seed_img, mode="class", margin=seed_roi_margin, unit=seed_roi_margin_unit)
        roi_stats = {}
        print(f"      Seed ROIs ({seed_roi}):")
        for class_id, roi in sorted(rois.items()):
            print(f"        Class {class_id}: {_roi_text(roi)}")

    # 3. Load prompts
    print("\n[3/4] Loading prompts...")
    if prompt_type == "point":

        default_point_config = {
            'default_n_pos': 1,
            'default_n_neg': 0,
            'default_method': 'grid',
            'negative_from_bg': True,
            'negative_from_other_classes': True,
            'negative_per_other_class': False
        }
        # merge user config
        if point_config is not None:
            default_point_config.update(point_config)

        pt_csv_path = Path(output_folder) / f"{Path(img_path).stem}_point_prompts.csv"
        
        if rois and seed_roi == "crop":
            # negatives outside a class's ROI would be dropped: sample them inside it
            default_point_config['neg_regions'] = {c: roi.bbox() for c, roi in rois.items()}

        if seed is not None:
            np.random.seed(int(seed))          # the same points for the same seed
        df_pt= seed_to_point_prompts_nninteractive(seg_path,img_path=img_path,
                                    output_csv = pt_csv_path,
                                    **default_point_config
                                    )
        
        # Validate CSV columns
        expected_columns = {'x', 'y', 'z', 'label', 'class_id'}
        if not expected_columns.issubset(set(df_pt.columns)):
            missing = expected_columns - set(df_pt.columns)
            raise ValueError(f"CSV missing columns: {missing}")
        
        # Validate coordinates
        _validate_coordinates(df_pt, img.shape[1:])
        print(f"      Total points: {len(df_pt) if df_pt is not None else 0}")
        print(f"      Total points Classes: {sorted(df_pt['class_id'].unique()) if df_pt is not None else []}")

    else:
        df_pt = None
    
    if prompt_type == "scribble":
        scribble_mask = seed_img if seed_img is not None else tifffile.imread(seg_path)
        
        # Validate scribble shape
        expected_shape = img.shape[1:]
        if scribble_mask.shape != expected_shape:
            raise ValueError(
                f"Scribble mask shape {scribble_mask.shape} doesn't match "
                f"image shape {expected_shape}"
            )
        
        print(f"      Scribble mask shape: {scribble_mask.shape}, unique values: {np.unique(scribble_mask)}")
        print(f"      Total classes: {len(np.unique(scribble_mask))}")
   
    else:
        scribble_mask = None
    
    # read initial segmentation if provided
    if init_seg_path is not None:
        init_mask = tifffile.imread(init_seg_path)
    else:
        init_mask = None
    
    # 4. Run prediction
    print("\n[4/4] Running prediction...")

    segs_folder = Path(output_folder) / "segs"
    prob_folder = Path(output_folder) / "prob"

    def save_class_mask(class_id, mask):
        segs_folder.mkdir(parents=True, exist_ok=True)
        out = segs_folder / f"{Path(img_path).stem}_class_{class_id}.tif"
        tifffile.imwrite(out, mask.astype(np.uint8) * 255, compression='zlib')
        print(f"      ✓ Saved class {class_id}: {mask.sum():,} pixels")

    def save_prob_map(class_id, prob_map):
        # Save foreground probability map as float32 TIFF (values in [0.0, 1.0])
        prob_folder.mkdir(parents=True, exist_ok=True)
        out = prob_folder / f"{Path(img_path).stem}_class_{class_id}_prob.tif"
        tifffile.imwrite(out, prob_map, compression='zlib')
        print(f"      Saved prob map class {class_id}: min={prob_map.min():.3f}, max={prob_map.max():.3f}")

    total_mask = nninter_predict(
        session, img, df_pt,
        scribble_mask=scribble_mask,
        init_mask=init_mask,
        scribble_config=scribble_config,
        on_class_predicted=save_class_mask if return_per_class_masks else None,
        on_class_prob_predicted=save_prob_map if save_prob_maps else None,
        rois=rois,
        roi_apply="crop" if seed_roi == "crop" else "postmask",
        roi_stats=roi_stats,
        seed=seed
    )
    if roi_stats is not None:
        log_dict["roi_per_class"] = roi_stats

    
    # Save results
    print("\n      Saving results...")
    output_path = Path(output_folder) / f"{Path(img_path).stem}_segmentation.tif"
    tifffile.imwrite(output_path, total_mask , compression='zlib')
    print(f"      ✓ Saved: {output_path}")
    
    # if return_per_class_masks:
    #     for class_id, mask in per_class_masks.items():
    #         output_path = Path(output_folder) / f"{Path(img_path).stem}_class_{class_id}.tif"
    #         tifffile.imwrite(output_path, mask.astype(np.uint8) * 255, compression='zlib')
    #         print(f"      ✓ Saved class {class_id}: {mask.sum():,} pixels")

    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)      # bytes
        peak_reserved  = torch.cuda.max_memory_reserved(device)       # bytes
        log_dict["gpu_peak_allocated_bytes"] = int(peak_allocated)
        log_dict["gpu_peak_reserved_bytes"]  = int(peak_reserved)
        log_dict["gpu_peak_allocated_MB"]    = round(peak_allocated / 1024**2, 2)
        log_dict["gpu_peak_reserved_MB"]     = round(peak_reserved  / 1024**2, 2)

        print(
            f"\n[GPU] Peak allocated: {log_dict['gpu_peak_allocated_MB']} MB, "
            f"Peak reserved: {log_dict['gpu_peak_reserved_MB']} MB"
        )
    else:
        log_dict["gpu_peak_allocated_bytes"] = 0
        log_dict["gpu_peak_reserved_bytes"]  = 0
        log_dict["gpu_peak_allocated_MB"]    = 0.0
        log_dict["gpu_peak_reserved_MB"]     = 0.0

    
    # Cleanup
    del session
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("✓ Completed successfully!")
    print("=" * 60 + "\n")
    
    # write log_dict to yaml
    with open(Path(output_folder) / "nninteractive_log.yaml", 'w') as f:
        yaml.dump(log_dict, f)
    
    # if return_per_class_masks:
    #     return total_mask, per_class_masks , log_dict
    # else:
    return total_mask, log_dict


def nninter_predict(
    session, 
    img, 
    df_pt: Optional[pd.DataFrame] = None,
    init_mask: Optional[np.ndarray] = None,
    scribble_mask: Optional[np.ndarray] = None,
    scribble_config = None,
    on_class_predicted=None,
    on_class_prob_predicted=None,
    rois: Optional[dict] = None,
    roi_apply: Literal['crop', 'postmask'] = "crop",
    roi_stats: Optional[dict] = None,
    seed: Optional[int] = 0
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[int, np.ndarray]]]:
    """
    Perform prediction with point and/or scribble prompts for each class.
    
    Args:
        session: nnInteractive inference session
        img: Input image array with shape (1, z, y, x)
        df_pt: DataFrame with point prompts (columns: x, y, z, label, class_id)
                   Can be None if only using scribble
        init_mask: Optional initial mask with shape (z, y, x)
                   Values: 0=background, 1,2,3...=class IDs
                   Can be None if not using an initial mask
        scribble_mask: Optional scribble mask with shape (z, y, x)
                       Values: 0=background, 1,2,3...=class IDs
                       Can be None if only using points
        return_per_class_masks: Whether to return individual masks per class
        scribble_config: Configuration for scribble prompt generation
        on_class_predicted: Optional callback function with signature (class_id, mask) called after each class is predicted. Useful for saving masks without keeping all in memory.
        on_class_prob_predicted: Optional callback function with signature (class_id, prob_map_float32) called after each class probability map is predicted.
        rois: Optional {class_id: Roi} from seed_roi.seed_rois. The mask and prob map of a
              class with an ROI are zero outside the ROI. Classes without an ROI are
              predicted on the full image and not masked.
        roi_apply: How rois are used. 'crop': a class with an ROI is predicted on
                   img[:, roi.bbox()] only (prompts cropped / shifted, points outside
                   dropped) and pasted back to full size. 'postmask': every class is
                   predicted on the full image, then cleared outside its ROI.
        roi_stats: Optional dict, filled with {class_id: ROI log} when rois is given
        seed: np.random.seed(seed + class_id) after each class's reset (default 0; None = no
              seeding). nnInteractive's refinement draws from np.random: with a seed per class the
              result of a class does not depend on the classes run before it (the napari Prompt
              tab does the same).
    
    Returns:
        total_mask: Combined segmentation with class IDs
        per_class_masks: (Optional) Dictionary {class_id: binary_mask}
    
    Examples:
        # Only points
        total_mask = nninter_predict(session, img, df_pt=df_points)
        
        # Only scribble
        total_mask = nninter_predict(session, img, scribble_mask=scribble)
        
        # Points + Scribble
        total_mask = nninter_predict(session, img, df_pt=df_points, scribble_mask=scribble)
    """
    # Validate inputs
    if df_pt is None and scribble_mask is None:
        raise ValueError("Must provide at least one of: df_pt or scribble_mask")
    
    # Validate scribble shape if provided
    if scribble_mask is not None:
        expected_shape = img.shape[1:]
        if scribble_mask.shape != expected_shape:
            raise ValueError(
                f"Scribble mask shape {scribble_mask.shape} doesn't match "
                f"image shape {expected_shape}"
            )
    
    if roi_apply not in ("crop", "postmask"):
        raise ValueError(f"roi_apply must be 'crop' or 'postmask', got {roi_apply!r}")
    crop_mode = bool(rois) and roi_apply == "crop"

    # Set image (once; in crop mode it is set per class crop in the loop, and the
    # full image is never converted to float32 unless a class has no ROI)
    full_window = tuple(slice(0, n) for n in img.shape[1:])
    if not crop_mode:
        session.set_image(img.astype(np.float32))
        current_window = full_window
    else:
        current_window = None  # window of img currently set in the session
    # per_class_masks = {}
    
    # Collect all class IDs from both sources
    class_ids = set()
    
    if df_pt is not None and len(df_pt) > 0:
        class_ids.update(df_pt['class_id'].unique())
    
    if scribble_mask is not None:
        scribble_classes = np.unique(scribble_mask)
        scribble_classes = scribble_classes[scribble_classes > 0]  # Exclude background (0)
        class_ids.update(scribble_classes)
    
    # Sort class IDs (reversed order for better results)
    class_ids = sorted(class_ids, reverse=True)

    # Initialize output (dtype by the largest class id: ids > 255 must not wrap)
    total_mask = np.zeros(img.shape[1:], dtype=label_dtype(int(class_ids[0]) if class_ids else 0))
    
    print(f"      Processing {len(class_ids)} classes: {class_ids}")
    
    # Process each class
    for idx, class_id in enumerate(class_ids, 1):
        print(f"      [{idx}/{len(class_ids)}] Class {class_id}...", end=" ")

        # ROI of this class (None: no seed ROI, full image, not masked); crop_roi is the
        # ROI the session works on (crop mode only). 'union' ROIs share one window, so
        # the image is only set again when the window changes
        roi = rois.get(int(class_id)) if rois else None
        if rois and roi is None:
            print("no seed ROI (full image, not masked)", end=" ")
        crop_roi = roi if crop_mode else None
        window = crop_roi.bbox() if crop_roi is not None else full_window
        if window != current_window:
            session.set_image(img[(slice(None),) + window].astype(np.float32))
            # set_image preprocesses in a background thread; wait for it so that
            # reset_interactions below sees the new image shape
            session._finish_preprocessing_and_initialize_interactions()
            current_window = window
        if crop_roi is not None:
            print(f"crop {_roi_text(crop_roi)}", end=" ")
        work_shape = tuple(s.stop - s.start for s in window)

        # Reset interactions for new object
        session.reset_interactions()
        if seed is not None:
            np.random.seed(int(seed) + int(class_id))
        target_tensor = torch.zeros(work_shape, dtype=torch.uint8)
        session.set_target_buffer(target_tensor)

        n_scribbles = 0
        n_points = 0
        n_dropped = 0

        # add initial mask if provided
        if init_mask is not None:
            class_init_mask = (_crop(init_mask, crop_roi) == class_id).astype(np.uint8)
            print(f"      Initial mask pixels: {class_init_mask.sum():,}")
            if class_init_mask.sum() > 0:
                session.add_initial_seg_interaction(
                    class_init_mask,
                    run_prediction=False   
                )

        
        # 1. Add scribble interaction if available for this class
        if scribble_mask is not None:
            # Extract binary mask for current class
            class_scribble_all = _crop(scribble_mask, crop_roi)
            class_scribble = (class_scribble_all == class_id).astype(np.uint8)
           
            
            if class_scribble.sum() > 0:  # Only add if scribble exists
  
                session.add_scribble_interaction(
                    scribble_image=class_scribble,
                    include_interaction=True,
                    run_prediction=False
                )
                session._predict()
                if scribble_config and scribble_config.get('use_negative_scribble', False):
                    negative_scribble = (class_scribble_all != class_id) & (class_scribble_all != 0)
            
                    session.add_scribble_interaction(
                        scribble_image=negative_scribble,
                        include_interaction=False,
                        run_prediction=False
                    )
                    session._predict()
                n_scribbles = class_scribble.sum()
                # session._predict()
        # 2. Add point interactions if available for this class
        if df_pt is not None:
            class_df = df_pt[df_pt['class_id'] == class_id]
            if crop_roi is not None and len(class_df) > 0:
                # to crop coordinates; points outside the ROI are dropped
                class_df, n_dropped = _points_to_roi(class_df, crop_roi)
                if n_dropped > 0:
                    print(f"\n      {n_dropped} point(s) outside the ROI dropped")

            if len(class_df) > 0:
                for _, row in class_df.iterrows():
                    # Create point coordinates for session
                    point_coords = (int(row['z']), int(row['y']), int(row['x']))
                    is_positive = bool(row['label'])
                    print("using point:", point_coords, "positive" if is_positive else "negative")
                    session.add_point_interaction(
                        coordinates=point_coords,
                        include_interaction=is_positive,
                        run_prediction=False
                    )
                    session._predict()
                    
                n_points = len(class_df)
        
        # 3. Run prediction once for all interactions
        # session._predict()
        
        # 4. Get result
        current_mask = session.target_buffer.cpu().numpy() > 0
        n_removed = 0
        if roi is not None and crop_roi is None:
            # postmask: clear the prediction outside the class's ROI
            n_full = int(current_mask.sum())
            current_mask = roi.paste(roi.crop(current_mask), region="bbox")
            n_removed = n_full - int(current_mask.sum())
        n_pixels = current_mask.sum()

        # 5. Get probability map if available and callback is provided
        if on_class_prob_predicted is not None and hasattr(session, 'prob_buffer') and session.prob_buffer is not None:
            prob_map = session.prob_buffer
            if isinstance(prob_map, torch.Tensor):
                prob_map = prob_map.cpu().numpy()
            prob_map = prob_map.copy().astype(np.float32)
            if roi is not None:
                # full size, 0 outside the ROI
                prob_map = roi.paste(prob_map if crop_roi is not None else roi.crop(prob_map), region="bbox")
            on_class_prob_predicted(class_id, prob_map)

        # 6. Save mask

        if on_class_predicted is not None:
            on_class_predicted(class_id, current_mask if crop_roi is None else crop_roi.paste(current_mask, region="bbox"))
        # 7. Update total mask (current_mask is in ROI coordinates when cropped)
        _crop(total_mask, crop_roi)[current_mask] = class_id

        # ROI log: is the prediction cut by the ROI?
        if roi_stats is not None:
            stats = {"cropped": crop_roi is not None}
            if roi is not None:
                face = inner_face_fraction(current_mask if crop_roi is not None else roi.crop(current_mask), roi)
                stats.update(roi_log(roi))
                stats["inner_face_fraction"] = round(face, 5)
                if crop_roi is not None:
                    stats["n_points"] = int(n_points)
                    stats["n_points_dropped"] = int(n_dropped)
                else:
                    stats["n_removed_outside_roi"] = int(n_removed)
                if face > ROI_EDGE_WARN_FRACTION:
                    removed = f", {n_removed:,} voxels outside it removed" if n_removed else ""
                    print(f"\n      [WARNING] Class {class_id}: {face:.1%} of the prediction lies on the ROI edge"
                          f"{removed}. Either it leaked into another part (then the ROI fixed it), or the "
                          f"part extends beyond its seed (then check the seed or raise seed_roi_margin)")
            roi_stats[int(class_id)] = stats

        # Print summary
        prompt_info = []
        if n_scribbles > 0:
            prompt_info.append(f"{n_scribbles:,} scribble pixels")
        if n_points > 0:
            prompt_info.append(f"{n_points} points")
        
        print(f"{', '.join(prompt_info)} → {n_pixels:,} pixels"
              + (f" ({n_removed:,} outside the ROI removed)" if n_removed else ""))
    
    # Print final statistics
    print(f"\n      Final mask statistics:")
    unique_values, counts = np.unique(total_mask, return_counts=True)
    for val, count in zip(unique_values, counts):
        if val == 0:
            print(f"        Background: {count:,} pixels")
        else:
            print(f"        Class {val}: {count:,} pixels")
            if init_mask is not None:
                init_count = (init_mask == val).sum()
                print(f"          Initial mask had {init_count:,} pixels")
    
    session.reset_interactions()
    del session
    torch.cuda.empty_cache()

    return total_mask


def _roi_text(roi):
    """e.g. 'z[0:96] y[10:150] x[0:122] (96, 140, 122)'"""
    rng = " ".join(f"{a}[{l}:{h}]" for a, l, h in zip("zyx", roi.lo, roi.hi))
    return f"{rng} {tuple(int(h - l) for l, h in zip(roi.lo, roi.hi))}"


def _crop(arr, roi):
    """arr on the ROI's bbox (a view), or arr itself when there is no ROI."""
    return arr if roi is None else roi.crop(arr)


def _points_to_roi(class_df: pd.DataFrame, roi):
    """Point prompts shifted to ROI crop coordinates; points outside the ROI are
    dropped. Returns (shifted DataFrame, number of dropped points)."""
    inside = roi.contains(class_df[['z', 'y', 'x']].to_numpy())
    out = class_df[inside].copy()
    oz, oy, ox = roi.offset()
    out['z'] -= oz
    out['y'] -= oy
    out['x'] -= ox
    _validate_coordinates(out, roi.crop_shape())
    return out, int((~inside).sum())


def _validate_coordinates(df: pd.DataFrame, img_shape: tuple):
    """Validate that all coordinates are within image bounds."""
    is_3d = len(img_shape) == 3
    
    if is_3d:
        z_max, y_max, x_max = img_shape
        if ((df['z'] < 0) | (df['z'] >= z_max)).any():
            raise ValueError(f"Z coordinates out of bounds [0, {z_max})")
    else:
        y_max, x_max = img_shape
    
    if ((df['y'] < 0) | (df['y'] >= y_max)).any():
        raise ValueError(f"Y coordinates out of bounds [0, {y_max})")
    
    if ((df['x'] < 0) | (df['x'] >= x_max)).any():
        raise ValueError(f"X coordinates out of bounds [0, {x_max})")


def run_nninteractive_yaml(file_path):
    _, extension = os.path.splitext(file_path)
    print(f"processing config the file {file_path}")

    if extension == '.yaml':
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            optional_params = config_core.validate_input_yaml(config, config_core.input_val_nninteractive_run)
    
    print(config)
    print(optional_params)
    
    model_path = config['model_path']
    device = optional_params['device']
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, switching to CPU")
        device = 'cpu'

    point_config = parse_point_config(optional_params)
    
    scribble_config = parse_scribble_config(optional_params)
    # use the output folder to save yaml
    output_folder = config['output_folder']
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    # save config and optional to yaml
    with open(Path(output_folder) / "used_config.yaml", 'w') as f:
        yaml.dump(config, f)
        yaml.dump(optional_params, f)

    
    _ = nninter_main(
        model_path=model_path,
        img_path=config['img_path'],
        seg_path=config['seg_path'],
        device=device,
        prompt_type=config['prompt_type'],
        init_seg_path=optional_params.get('init_seg_path', None),
        point_config=point_config,
        output_folder=output_folder,
        return_per_class_masks=optional_params['return_per_class_masks'],
        scribble_config=scribble_config,
        save_prob_maps=optional_params.get('save_prob_maps', False),
        do_autozoom=optional_params.get('do_autozoom', True),
        seed_roi=optional_params.get('seed_roi', "off"),
        seed_roi_margin=optional_params.get('seed_roi_margin', 32),
        seed_roi_margin_unit=optional_params.get('seed_roi_margin_unit', "voxel"),
        seed=optional_params['seed']
    )

# Main execution
if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(f"Reading config file from command-line argument: {sys.argv[1]}")
        file_path = sys.argv[1]
    else:
        print("No config file specified in arguments. Using default: ./template/nninteractive_predict.yaml")
        file_path = './template/nninteractive_predict.yaml' 
    
   
    
    run_nninteractive_yaml(file_path)  
