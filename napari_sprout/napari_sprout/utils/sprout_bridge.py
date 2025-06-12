import sys
import os
from pathlib import Path
import numpy as np
import tifffile
from typing import Dict, List, Optional, Tuple, Any

# Add SPROUT to path if needed
sprout_path = Path(__file__).parent.parent.parent.parent
if str(sprout_path) not in sys.path:
    sys.path.insert(0, str(sprout_path))

try:
    import sprout_core.sprout_core as sprout_core
    import sprout_core.config_core as config_core
    SPROUT_AVAILABLE = True
except ImportError:
    SPROUT_AVAILABLE = False
    print("Warning: SPROUT core modules not found. Please ensure SPROUT is in the Python path.")


class SPROUTBridge:
    """Bridge class to interface with SPROUT functionality."""
    
    def __init__(self):
        if not SPROUT_AVAILABLE:
            raise ImportError("SPROUT modules are not available. Please check installation.")
        
    @staticmethod
    def generate_seeds(
        volume_array: np.ndarray,
        threshold: float,
        segments: int,
        ero_iter: int,
        footprint: str = 'ball',
        upper_threshold: Optional[float] = None,
        boundary: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Generate seeds using SPROUT's seed generation algorithm.
        
        Parameters
        ----------
        volume_array : np.ndarray
            Input image array
        threshold : float
            Lower threshold for segmentation
        segments : int
            Number of largest components to keep
        ero_iter : int
            Number of erosion iterations
        footprint : str
            Erosion footprint shape
        upper_threshold : float, optional
            Upper threshold for segmentation
        boundary : np.ndarray, optional
            Boundary mask
            
        Returns
        -------
        seed : np.ndarray
            Generated seed labels
        sizes : List[int]
            Sizes of connected components
        """
        # Apply thresholding
        if upper_threshold is None:
            volume_label = volume_array >= threshold
        else:
            volume_label = (volume_array >= threshold) & (volume_array <= upper_threshold)
        
        # Apply boundary if provided
        if boundary is not None:
            boundary = sprout_core.check_and_cast_boundary(boundary)
            volume_label[boundary] = False
        
        # Apply erosion
        for i in range(ero_iter):
            volume_label = sprout_core.erosion_binary_img_on_sub(
                volume_label, footprint=footprint
            )
        
        # Get connected components
        seed, sizes = sprout_core.get_ccomps_with_size_order(volume_label, segments)
        
        return seed.astype(np.uint8), sizes.tolist()
    
    @staticmethod
    def grow_seeds(
        image: np.ndarray,
        seeds: np.ndarray,
        thresholds: List[float],
        dilate_iters: List[int],
        upper_thresholds: Optional[List[float]] = None,
        boundary: Optional[np.ndarray] = None,
        touch_rule: str = 'stop',
        to_grow_ids: Optional[List[int]] = None,
        callback: Optional[Any] = None
    ) -> np.ndarray:
        """
        Grow seeds using SPROUT's growth algorithm.
        
        Parameters
        ----------
        image : np.ndarray
            Original image
        seeds : np.ndarray
            Seed labels to grow
        thresholds : List[float]
            Growth thresholds
        dilate_iters : List[int]
            Dilation iterations for each threshold
        upper_thresholds : List[float], optional
            Upper thresholds for growth
        boundary : np.ndarray, optional
            Boundary mask
        touch_rule : str
            Rule for handling overlaps
        to_grow_ids : List[int], optional
            Specific IDs to grow
        callback : callable, optional
            Progress callback function
            
        Returns
        -------
        result : np.ndarray
            Grown segmentation
        """
        result = seeds.copy().astype(np.uint8)
        
        # Check boundary
        if boundary is not None:
            boundary = sprout_core.check_and_cast_boundary(boundary)
        
        # Iterate through thresholds
        for i, (threshold, dilate_iter) in enumerate(zip(thresholds, dilate_iters)):
            # Create threshold binary
            if upper_thresholds is not None and upper_thresholds[i] is not None:
                threshold_binary = (image >= threshold) & (image <= upper_thresholds[i])
            else:
                threshold_binary = image >= threshold
            
            # Apply boundary
            if boundary is not None:
                threshold_binary[boundary] = False
            
            # Perform dilation iterations
            for j in range(dilate_iter):
                result = sprout_core.dilation_one_iter(
                    result, threshold_binary,
                    touch_rule=touch_rule,
                    to_grow_ids=to_grow_ids
                )
                
                if callback:
                    progress = ((i * dilate_iter + j + 1) / 
                               (len(thresholds) * max(dilate_iters))) * 100
                    callback(progress)
        
        # Reorder segmentation
        result, _ = sprout_core.reorder_segmentation(result, sort_ids=True)
        
        return result
    
    @staticmethod
    def apply_threshold_preview(
        image: np.ndarray,
        threshold: float,
        upper_threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply threshold to create a binary preview.
        
        Parameters
        ----------
        image : np.ndarray
            Input image
        threshold : float
            Lower threshold
        upper_threshold : float, optional
            Upper threshold
            
        Returns
        -------
        binary : np.ndarray
            Binary threshold result
        """
        if upper_threshold is None:
            return image >= threshold
        else:
            return (image >= threshold) & (image <= upper_threshold)
    
    @staticmethod
    def get_footprint_options() -> List[str]:
        """Get available footprint options for erosion."""
        return ['ball', 'cube', 'ball_XY', 'ball_XZ', 'ball_YZ', 'X', 'Y', 'Z']
    
    @staticmethod
    def save_segmentation(
        segmentation: np.ndarray,
        filepath: str,
        compression: str = 'zlib'
    ) -> None:
        """Save segmentation to TIFF file."""
        tifffile.imwrite(filepath, segmentation.astype(np.uint8), compression=compression)
    
    @staticmethod
    def load_image(filepath: str) -> np.ndarray:
        """Load image from file."""
        return tifffile.imread(filepath)
