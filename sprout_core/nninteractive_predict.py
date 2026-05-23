import numpy as np
import pandas as pd
import torch
import tifffile
from pathlib import Path
from typing import List, TypedDict, Literal, Optional, Union, Dict, Tuple
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from sprout_core.sprout_prompt_core import seed_to_point_prompts_nninteractive
import os, sys
import yaml
import sprout_core.config_core as config_core


def estimate_scribble_zoom_factor(scribble_binary: np.ndarray, patch_size: list) -> Tuple[float, list, list]:
    """
    Estimate the zoom_out_factor that nnInteractive would apply to the given scribble.
    This replicates the logic in nnInteractive._generic_add_patch_from_image.

    Returns
    -------
    zoom_out_factor : float
    roi             : [[z_min, z_max], [y_min, y_max], [x_min, x_max]]
    roi_center      : [cz, cy, cx]
    """
    nonzero = np.nonzero(scribble_binary)
    if len(nonzero[0]) == 0:
        return 1.0, [], []

    roi        = [[int(ax.min()), int(ax.max()) + 1] for ax in nonzero]
    roi_center = [round((r[0] + r[1]) / 2) for r in roi]
    roi_size   = [r[1] - r[0] for r in roi]

    # Replicates the logic in nnInteractive source code
    requested_size   = [s + p // 3 for s, p in zip(roi_size, patch_size)]
    zoom_out_factor  = max(1.0, max(r / p for r, p in zip(requested_size, patch_size)))

    return zoom_out_factor, roi, roi_center


def _print_scribble_debug_info(
    label: str,
    scribble_binary: np.ndarray,
    patch_size: list,
    session=None,
):
    """
    Print debug information related to a scribble:
      - Number of non-zero pixels and the ROI
      - Estimated zoom_out_factor
      - Zoom/center queued in the session (readable after add_scribble_interaction)
    """
    zoom_est, roi, roi_center = estimate_scribble_zoom_factor(scribble_binary, patch_size)
    roi_size = [r[1] - r[0] for r in roi] if roi else []

    print(f"\n        ── [Scribble Debug | {label}] ──")
    print(f"           Non-zero pixels : {int(scribble_binary.sum()):,}")
    print(f"           ROI           : {roi}  (z/y/x order)")
    print(f"           ROI size      : {roi_size}")
    print(f"           ROI center    : {roi_center}")
    print(f"           Estimated zoom: {zoom_est:.3f}  (patch_size={patch_size})")
    print(f"           Patch size    : {patch_size}")

    # If the session has already add_scribble_interaction (run_prediction=False),
    # session.new_interaction_zoom_out_factors will have values
    if session is not None and hasattr(session, 'new_interaction_zoom_out_factors'):
        q_zooms   = session.new_interaction_zoom_out_factors
        q_centers = session.new_interaction_centers
        if q_zooms:
            print(f"           session queued zoom   : {q_zooms}")
            print(f"           session queued center : {q_centers}")
        else:
            print(f"           session queued zoom   : (empty, not yet add_interaction)")
    print(f"        ── [End Debug] ──\n")


def split_scribble_into_chunks(
    scribble_binary: np.ndarray,
    patch_size: list,
    max_zoom_factor: float = 2.0,
    _depth: int = 0,
    _max_depth: int = 6,
) -> List[np.ndarray]:
    """
    If scribble zoom_out_factor pass max_zoom_factor, split the scribble into smaller chunks along the longest axis of the ROI.

    Parameters
    ----------
    scribble_binary : np.ndarray  Binary mask (0/1), shape (z, y, x)
    patch_size      : list        Model patch size, e.g., [192, 192, 192]
    max_zoom_factor : float       Maximum allowed zoom_out_factor, split if exceeded
    _depth/_max_depth             Internal recursion protection, do not set externally

    Returns
    -------
    list of np.ndarray  Each element is a sub scribble binary mask that does not exceed the threshold
    """
    if _depth >= _max_depth:
        return [scribble_binary]  # Prevent infinite recursion

    zoom_out_factor, roi, _ = estimate_scribble_zoom_factor(scribble_binary, patch_size)

    if zoom_out_factor <= max_zoom_factor:
        return [scribble_binary]

    # Find the longest axis
    roi_sizes    = [r[1] - r[0] for r in roi]
    longest_axis = int(np.argmax(roi_sizes))

    # Split along the longest axis
    axis_start = roi[longest_axis][0]
    axis_end   = roi[longest_axis][1]
    mid        = (axis_start + axis_end) // 2

    slicer_a = [slice(None)] * scribble_binary.ndim
    slicer_b = [slice(None)] * scribble_binary.ndim
    slicer_a[longest_axis] = slice(axis_start, mid)
    slicer_b[longest_axis] = slice(mid, axis_end)

    half_a = np.zeros_like(scribble_binary)
    half_b = np.zeros_like(scribble_binary)
    half_a[tuple(slicer_a)] = scribble_binary[tuple(slicer_a)]
    half_b[tuple(slicer_b)] = scribble_binary[tuple(slicer_b)]

    chunks = []
    for half in (half_a, half_b):
        if half.sum() > 0:
            chunks.extend(
                split_scribble_into_chunks(
                    half, patch_size, max_zoom_factor,
                    _depth=_depth + 1, _max_depth=_max_depth
                )
            )
    return chunks if chunks else [scribble_binary]


def parse_scribble_config(optional_params):
    scribble_config = {
        'use_negative_scribble': optional_params.get('use_negative_scribble', False),
        
        'auto_split_scribble':   optional_params.get('auto_split_scribble', True),
        
        'max_zoom_factor':       optional_params.get('max_zoom_factor', 2.0),
        
        'verbose_scribble':      optional_params.get('verbose_scribble', False),
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
                 
                 return_per_class_masks: bool = False):
    """
    Main function for nnInteractive prediction with point prompts.
    
    Args:
        model_path: Path to trained model folder
        img_path: Path to input TIFF image
        df_pt_path: Path to CSV file with point prompts
        scribble_mask_path: Path to scribble mask image
        output_folder: Folder to save output segmentation
        return_per_class_masks: Whether to save individual class masks
    
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"      Device: {device}")
    
    session = nnInteractiveInferenceSession(
        device=device,
        verbose=False,
        use_torch_compile=False,
        do_autozoom=True,
        use_pinned_memory=True
    )
    
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
    if return_per_class_masks:
        total_mask, per_class_masks = nninter_predict(
            session, img, df_pt, scribble_mask, return_per_class_masks=True
        )
    else:
        total_mask = nninter_predict(
            session, img, df_pt, scribble_mask, return_per_class_masks=False
        )
        per_class_masks = None
    
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
                 scribble_config= None):
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
    print("=" * 60)

    log_dict = {
        "img_path": img_path,
        "seg_path": seg_path,
        "output_folder": output_folder,
        "init_seg_path": init_seg_path}
  
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
    
    session = nnInteractiveInferenceSession(
        device=device,
        verbose=False,
        use_torch_compile=False,
        do_autozoom=True,
        use_pinned_memory=False
    )
    
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
        scribble_mask = tifffile.imread(seg_path)
        
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

    def save_class_mask(class_id, mask):
        out = Path(output_folder) / f"{Path(img_path).stem}_class_{class_id}.tif"
        tifffile.imwrite(out, mask.astype(np.uint8) * 255, compression='zlib')
        print(f"      ✓ Saved class {class_id}: {mask.sum():,} pixels")

    total_mask = nninter_predict(
        session, img, df_pt,
        scribble_mask=scribble_mask,
        init_mask=init_mask,
        scribble_config=scribble_config,
        on_class_predicted=save_class_mask if return_per_class_masks else None
    )

    
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
    on_class_predicted=None
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

    # Get the scribble config if provided, or set defaults
    _cfg                = scribble_config or {}
    use_neg_scribble    = _cfg.get('use_negative_scribble', False)
    auto_split          = _cfg.get('auto_split_scribble',   False)
    max_zoom_factor     = float(_cfg.get('max_zoom_factor', 2.0))
    verbose_scribble    = _cfg.get('verbose_scribble',      False)

    
    # get the patch size
    patch_size = list(session.configuration_manager.patch_size) \
    if session.configuration_manager is not None else None
    
    # Set image
    session.set_image(img.astype(np.float32))
    
    # Initialize output
    total_mask = np.zeros(img.shape[1:], dtype=np.uint8)
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
    
    print(f"      Processing {len(class_ids)} classes: {class_ids}")
    
    # Process each class
    for idx, class_id in enumerate(class_ids, 1):
        print(f"      [{idx}/{len(class_ids)}] Class {class_id}...", end=" ")
        
        # Reset interactions for new object
        session.reset_interactions()
        target_tensor = torch.zeros(img.shape[1:], dtype=torch.uint8)
        session.set_target_buffer(target_tensor)
        
        n_scribbles = 0
        n_points = 0
        
        # add initial mask if provided
        if init_mask is not None:
            class_init_mask = (init_mask == class_id).astype(np.uint8)
            print(f"      Initial mask pixels: {class_init_mask.sum():,}")
            if class_init_mask.sum() > 0:
                session.add_initial_seg_interaction(
                    class_init_mask,
                    run_prediction=False   
                )

        
        # 1. Add scribble interaction if available for this class
        if scribble_mask is not None:
            # Extract binary mask for current class
            class_scribble = (scribble_mask == class_id).astype(np.uint8)
           
            
            if class_scribble.sum() > 0:  # Only add if scribble exists
                
                
                # zoom_est, _, _ = estimate_scribble_zoom_factor(class_scribble, patch_size)
                # print(" Working on class", class_id, end="")
                # print(" The zoom factor for this scribble is estimated to be:", zoom_est)
                
                # session.add_scribble_interaction(
                #     scribble_image=class_scribble,
                #     include_interaction=True,
                #     run_prediction=False
                # )
                # session._predict()
                
                if auto_split and patch_size:
                    zoom_est, _, _ = estimate_scribble_zoom_factor(class_scribble, patch_size)
                    if zoom_est > max_zoom_factor:
                        chunks = split_scribble_into_chunks(
                            class_scribble, patch_size, max_zoom_factor
                        )
                        print(
                            f"        ⚠  Scribble zoom_est={zoom_est:.2f} > {max_zoom_factor} → "
                            f"split into {len(chunks)} chunks"
                        )
                    else:
                        chunks = [class_scribble]
                        print(f"        ✓  Scribble zoom_est={zoom_est:.2f} ≤ {max_zoom_factor}, no split needed")
                else:
                    chunks = [class_scribble]

                # for chunk_i, chunk in enumerate(chunks):
                for chunk_i, chunk in enumerate(chunks):
                    if chunk.sum() == 0:
                        continue

                    # verbose: 每个 chunk 的信息
                    if verbose_scribble and patch_size:
                        _print_scribble_debug_info(
                            label=f"Class {class_id} - chunk {chunk_i + 1}/{len(chunks)}",
                            scribble_binary=chunk,
                            patch_size=patch_size,
                            session=None,
                        )

                    session.add_scribble_interaction(
                        scribble_image=chunk,
                        include_interaction=True,
                        run_prediction=False,
                    )
                    session._predict()
                
                # # add negative scribble if configured
                # if scribble_config and scribble_config.get('use_negative_scribble', False):
                #     negative_scribble = (scribble_mask != class_id) & (scribble_mask != 0)
                    
                #     zoom_est, _, _ = estimate_scribble_zoom_factor(negative_scribble, patch_size)
                #     print(" Adding negative scribble for class", class_id, end="")
                #     print(" The zoom factor for this scribble is estimated to be:", zoom_est)
            
                #     session.add_scribble_interaction(
                #         scribble_image=negative_scribble,
                #         include_interaction=False,
                #         run_prediction=False
                #     )
                #     session._predict()
                # ── Negative scribble ─────────────────────────────────────────
                if use_neg_scribble:
                    negative_scribble = (
                        (scribble_mask != class_id) & (scribble_mask != 0)
                    ).astype(np.float32)

                    if negative_scribble.sum() > 0:
                        if verbose_scribble and patch_size:
                            _print_scribble_debug_info(
                                label=f"Class {class_id} - negative scribble",
                                scribble_binary=negative_scribble,
                                patch_size=patch_size,
                                session=None,
                            )

                        # split for the negative scribble if zoom factor is too high
                        if auto_split and patch_size:
                            neg_zoom_est, _, _ = estimate_scribble_zoom_factor(
                                negative_scribble, patch_size
                            )
                            if neg_zoom_est > max_zoom_factor:
                                neg_chunks = split_scribble_into_chunks(
                                    negative_scribble, patch_size, max_zoom_factor
                                )
                                print(
                                    f"        ⚠  Neg scribble zoom_est={neg_zoom_est:.2f} > {max_zoom_factor} → "
                                    f"split into {len(neg_chunks)} chunks"
                                )
                            else:
                                neg_chunks = [negative_scribble]
                        else:
                            neg_chunks = [negative_scribble]

                        for neg_chunk in neg_chunks:
                            if neg_chunk.sum() == 0:
                                continue
                            session.add_scribble_interaction(
                                scribble_image=neg_chunk,
                                include_interaction=False,
                                run_prediction=False,
                            )
                            session._predict()                    
                    
                    
                n_scribbles = class_scribble.sum()
                # session._predict()
        # 2. Add point interactions if available for this class
        if df_pt is not None:
            class_df = df_pt[df_pt['class_id'] == class_id]
            
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
        n_pixels = current_mask.sum()
        
        # 5. Save mask
        
        if on_class_predicted is not None:
            on_class_predicted(class_id, current_mask)
        # 6. Update total mask
        total_mask[current_mask] = class_id
        
        # Print summary
        prompt_info = []
        if n_scribbles > 0:
            prompt_info.append(f"{n_scribbles:,} scribble pixels")
        if n_points > 0:
            prompt_info.append(f"{n_points} points")
        
        print(f"{', '.join(prompt_info)} → {n_pixels:,} pixels")
    
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
        scribble_config=scribble_config
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