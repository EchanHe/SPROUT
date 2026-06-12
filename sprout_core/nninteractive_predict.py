import numpy as np
import pandas as pd
import torch
import tifffile
from pathlib import Path
from typing import TypedDict, Literal, Optional, Union, Dict, Tuple
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from sprout_core.sprout_prompt_core import seed_to_point_prompts_nninteractive
import os, sys
import yaml
import sprout_core.config_core as config_core

class nnInteractiveSessionWithProb(nnInteractiveInferenceSession):
    """
    Extension of nnInteractiveInferenceSession that also maintains a float32
    foreground probability map (prob_buffer) in addition to the binary target_buffer.

    A forward hook captures softmax(logits)[1] (foreground probability) after each
    network forward pass. A patched paste_tensor writes those probabilities into
    prob_buffer whenever the binary prediction is written into target_buffer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prob_buffer: Optional[torch.Tensor] = None
        self._last_fg_prob: Optional[torch.Tensor] = None  # CPU float32 tensor, updated by hook

    def initialize_from_trained_model_folder(self, *args, **kwargs):
        super().initialize_from_trained_model_folder(*args, **kwargs)
        # Register hook to capture foreground probability after every forward pass
        self.network.register_forward_hook(self._capture_fg_prob_hook)

    def _capture_fg_prob_hook(self, module, input, output):
        """
        Fires after every network forward pass.
        output shape: (1, 2, *patch_size) -- two logit channels (bg, fg).
        Stores softmax fg probability as a CPU float32 tensor.
        """
        logits = output[0].float()  # (2, *patch_size) on device
        self._last_fg_prob = torch.softmax(logits, dim=0)[1].detach().cpu()

    def set_target_buffer(self, target_buffer):
        super().set_target_buffer(target_buffer)
        # Initialise a matching float32 prob_buffer
        if isinstance(target_buffer, torch.Tensor):
            self.prob_buffer = torch.zeros(target_buffer.shape, dtype=torch.float32)
        elif isinstance(target_buffer, np.ndarray):
            self.prob_buffer = np.zeros(target_buffer.shape, dtype=np.float32)

    def reset_interactions(self):
        super().reset_interactions()
        if self.prob_buffer is not None:
            if isinstance(self.prob_buffer, torch.Tensor):
                self.prob_buffer.zero_()
            else:
                self.prob_buffer.fill(0.0)
        self._last_fg_prob = None

    @torch.inference_mode()
    def _predict(self, force_full_refine: bool = False):
        """
        Override _predict to also write foreground probabilities into prob_buffer.

        We temporarily replace paste_tensor in the inference_session module's namespace
        with a wrapper. Whenever paste_tensor writes into target_buffer, the wrapper
        also writes the corresponding fg probability into prob_buffer.
        The original function is always restored in the finally block.
        """
        import nnInteractive.inference.inference_session as _sess_mod
        from nnInteractive.utils.crop import paste_tensor as _orig_paste

        session = self

        def _prob_paste(dst, src, bbox):
            _orig_paste(dst, src, bbox)
            if (
                dst is session.target_buffer
                and session._last_fg_prob is not None
                and session.prob_buffer is not None
            ):
                fg_prob = session._last_fg_prob
                # Resize fg_prob if the prediction patch was spatially resampled (autozoom)
                if list(fg_prob.shape) != list(src.shape):
                    fg_prob = torch.nn.functional.interpolate(
                        fg_prob[None, None].float(),
                        list(src.shape),
                        mode='trilinear',
                        align_corners=False,
                    )[0, 0]
                prob_dst = (
                    session.prob_buffer
                    if isinstance(session.prob_buffer, torch.Tensor)
                    else torch.from_numpy(session.prob_buffer)
                )
                _orig_paste(prob_dst, fg_prob.cpu(), bbox)
                if isinstance(session.prob_buffer, np.ndarray):
                    session.prob_buffer[:] = prob_dst.numpy()

        _sess_mod.paste_tensor = _prob_paste
        try:
            super()._predict()
        finally:
            _sess_mod.paste_tensor = _orig_paste  # always restore original

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
                 scribble_config= None,
                 save_prob_maps: bool = False):
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
    
    # Use probability-tracking session if save_prob_maps is enabled
    SessionClass = nnInteractiveSessionWithProb if save_prob_maps else nnInteractiveInferenceSession
    session = SessionClass(
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

    def save_prob_map(class_id, prob_map):
        # Save foreground probability map as float32 TIFF (values in [0.0, 1.0])
        out = Path(output_folder) / f"{Path(img_path).stem}_class_{class_id}_prob.tif"
        tifffile.imwrite(out, prob_map, compression='zlib')
        print(f"      Saved prob map class {class_id}: min={prob_map.min():.3f}, max={prob_map.max():.3f}")

    total_mask = nninter_predict(
        session, img, df_pt,
        scribble_mask=scribble_mask,
        init_mask=init_mask,
        scribble_config=scribble_config,
        on_class_predicted=save_class_mask if return_per_class_masks else None,
        on_class_prob_predicted=save_prob_map if save_prob_maps else None,
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
    on_class_predicted=None,
    on_class_prob_predicted=None,
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
  
                session.add_scribble_interaction(
                    scribble_image=class_scribble,
                    include_interaction=True,
                    run_prediction=False
                )
                session._predict()
                if scribble_config and scribble_config.get('use_negative_scribble', False):
                    negative_scribble = (scribble_mask != class_id) & (scribble_mask != 0)
            
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
        
        # 5. Get probability map if available and callback is provided
        if on_class_prob_predicted is not None and hasattr(session, 'prob_buffer') and session.prob_buffer is not None:
            prob_map = session.prob_buffer
            if isinstance(prob_map, torch.Tensor):
                prob_map = prob_map.cpu().numpy()
            on_class_prob_predicted(class_id, prob_map.copy().astype(np.float32))

        # 6. Save mask
        
        if on_class_predicted is not None:
            on_class_predicted(class_id, current_mask)
        # 7. Update total mask
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
        scribble_config=scribble_config,
        save_prob_maps=optional_params.get('save_prob_maps', False),
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
