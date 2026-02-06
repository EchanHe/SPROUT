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
        do_autozoom=True
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
    tifffile.imwrite(output_path, total_mask)
    print(f"      ✓ Saved: {output_path}")
    
    if return_per_class_masks:
        for class_id, mask in per_class_masks.items():
            output_path = Path(output_folder) / f"{Path(img_path).stem}_class_{class_id}.tif"
            tifffile.imwrite(output_path, mask.astype(np.uint8) * 255)
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
                prompt_type: Literal['point', 'scribble'] = "point",
                point_config: Optional[PointPromptConfig] = None,
                 return_per_class_masks: bool = False):
    """
    Main function for nnInteractive prediction with point prompts.
    
    Args:
        model_path: Path to trained model folder
        img_path: Path to input TIFF image
        seg_path: Path to input segmentation image (seed)
        output_folder: Folder to save output segmentation
        prompt_type: Type of prompt to use ("point" or "scribble")
        point_config: Configuration for point prompt generation
        return_per_class_masks: Whether to save individual class masks
    
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
    print(f"    evice: {device}")
    print(f"    Prompt type: {prompt_type}")
    print(f"    Point config: {point_config}")
    print(f"    Return per class masks: {return_per_class_masks}")
    
    print("=" * 60)

    log_dict = {
        "img_path": img_path,
        "seg_path": seg_path,
        "output_folder": output_folder}
  
    
    # Validate paths
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not Path(img_path).exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not Path(seg_path).exists():
        raise FileNotFoundError(f"Segmentation image not found: {seg_path}")
    
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
        do_autozoom=True
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
        
        df_pt= seed_to_point_prompts_nninteractive(seg_path,
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
    tifffile.imwrite(output_path, total_mask)
    print(f"      ✓ Saved: {output_path}")
    
    if return_per_class_masks:
        for class_id, mask in per_class_masks.items():
            output_path = Path(output_folder) / f"{Path(img_path).stem}_class_{class_id}.tif"
            tifffile.imwrite(output_path, mask.astype(np.uint8) * 255)
            print(f"      ✓ Saved class {class_id}: {mask.sum():,} pixels")
    
    # Cleanup
    del session
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("✓ Completed successfully!")
    print("=" * 60 + "\n")
    
    if return_per_class_masks:
        return total_mask, per_class_masks , log_dict
    else:
        return total_mask, log_dict


def nninter_predict(
    session, 
    img, 
    df_pt: Optional[pd.DataFrame] = None,
    scribble_mask: Optional[np.ndarray] = None,
    return_per_class_masks: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[int, np.ndarray]]]:
    """
    Perform prediction with point and/or scribble prompts for each class.
    
    Args:
        session: nnInteractive inference session
        img: Input image array with shape (1, z, y, x)
        df_pt: DataFrame with point prompts (columns: x, y, z, label, class_id)
                   Can be None if only using scribble
        scribble_mask: Optional scribble mask with shape (z, y, x)
                       Values: 0=background, 1,2,3...=class IDs
                       Can be None if only using points
        return_per_class_masks: Whether to return individual masks per class
    
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
    per_class_masks = {}
    
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
        
        # 1. Add scribble interaction if available for this class
        if scribble_mask is not None:
            # Extract binary mask for current class
            class_scribble = (scribble_mask == class_id).astype(np.float32)
            
            if class_scribble.sum() > 0:  # Only add if scribble exists
                session.add_scribble_interaction(
                    scribble_image=class_scribble,
                    include_interaction=True,
                    run_prediction=False
                )
                n_scribbles = class_scribble.sum()
        
        # 2. Add point interactions if available for this class
        if df_pt is not None:
            class_df = df_pt[df_pt['class_id'] == class_id]
            
            if len(class_df) > 0:
                for _, row in class_df.iterrows():
                    point_coords = (int(row['z']), int(row['y']), int(row['x']))
                    is_positive = bool(row['label'])
                    
                    session.add_point_interaction(
                        coordinates=point_coords,
                        include_interaction=is_positive,
                        run_prediction=False
                    )
                
                n_points = len(class_df)
        
        # 3. Run prediction once for all interactions
        session._predict()
        
        # 4. Get result
        current_mask = session.target_buffer.cpu().numpy() > 0
        n_pixels = current_mask.sum()
        
        # 5. Save mask
        if return_per_class_masks:
            per_class_masks[class_id] = current_mask
        
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
    
    if return_per_class_masks:
        return total_mask, per_class_masks
    else:
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
    if device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, switching to CPU")
        device = 'cpu'

    point_config = {
        'default_n_pos': optional_params['default_n_pos'],
        'default_n_neg': optional_params['default_n_neg'],
        'default_method': optional_params['default_method'],
        'negative_from_bg': optional_params['negative_from_bg'],
        'negative_from_other_classes': optional_params['negative_from_other_classes'],
        'negative_per_other_class': optional_params['negative_per_other_class'],
        "class_config": optional_params.get('class_config', None),
    }

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
        point_config=point_config,
        output_folder=output_folder,
        return_per_class_masks=optional_params['return_per_class_masks']
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