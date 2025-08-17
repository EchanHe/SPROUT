import yaml
import json
import os
import psutil
import shutil

import pandas as pd
import numpy as np
import tifffile as tiff
from datetime import datetime
from skimage.filters import threshold_otsu
import ast  # safer than eval

from tifffile import imread


support_footprints =['ball','cube',
                     'ball_XY','ball_XZ','ball_YZ',
                     'X','Y','Z',
                     '2XZ_1Y','2XY_1Z','2YZ_1X']

support_footprints_2d = ['ball', 'cube' ,'disk' , 'square', 'X', 'Y']


def valid_input_data(img, seg=None, boundary=None):
    """
    Validate that input image, segmentation, and boundary arrays match in shape and format.

    Parameters
    ----------
    img : np.ndarray
        Input 2D or 3D grayscale image.
    seg : np.ndarray, optional
        Optional segmentation mask. Must match shape of `img`.
    boundary : np.ndarray, optional
        Optional binary boundary mask. Must be same shape as `img` and either boolean or two-valued (e.g., 0 and 1).

    Raises
    ------
    ValueError
        If inputs are not 2D/3D, shapes do not match, or types are invalid.
    """
    
    # check if img only have one channel
    if img.ndim not in [2, 3]:
        raise ValueError(f"Image must be 2D or 3D, got {img.ndim}D.")
    # if seg is provided, check if it has the same shape as img
    if seg is not None:
        # check if seg have the same number of channels as img
        if seg.ndim not in [2, 3]:
            raise ValueError(f"Segmentation must be 2D or 3D, got {seg.ndim}D.")
        if img.shape != seg.shape:
            raise ValueError(f"Image and segmentation must have the same shape, got {img.shape} and {seg.shape}.")
    # if boundary is provided, check if it has the same shape as img
    if boundary is not None:
        if boundary.ndim not in [2, 3]:
            raise ValueError(f"Boundary must be 2D or 3D, got {boundary.ndim}D.")
        # check if boundary is True or false or just two value (0 or others)
        if not np.issubdtype(boundary.dtype, np.bool_):
            if not np.issubdtype(boundary.dtype, np.integer) or boundary.dtype != np.uint8:
                raise ValueError(f"Boundary must be boolean or integer type, got {boundary.dtype}.")
            # check if boundary only have two values
            unique_values = np.unique(boundary)
            if len(unique_values) != 2:
                raise ValueError(f"Boundary must have only two values, got {unique_values}.")
            
        
        
        if img.shape != boundary.shape:
            raise ValueError(f"Image and boundary must have the same shape, got {img.shape} and {boundary.shape}.")

def check_and_load_data(array, path, name, must_exist=True):
    """
    Load image data from file or return provided array, ensuring only one input source is used.

    Parameters
    ----------
    array : np.ndarray or None
        Directly provided image or mask array.
    path : str or None
        Path to the image or mask file.
    name : str
        Name identifier for error messages.
    must_exist : bool, default=True
        If True, raise error if both `array` and `path` are None.

    Returns
    -------
    np.ndarray
        Loaded or passed-in array.

    Raises
    ------
    ValueError
        If both `array` and `path` are provided or neither is available when `must_exist` is True.
    """
    if array is not None and path is not None:
        raise ValueError(f"Both {name} and {name}_path provided; only one is allowed.")
    if array is None and path is None and must_exist:
        raise ValueError(f"Either {name} or {name}_path must be provided.")
    if array is not None and path is None:
        return array
    if path is not None:
        return imread(path)
    return array

def check_and_cast_boundary(boundary):
    """
    Cast a 2-valued or boolean array into a binary mask (True/False).

    Parameters
    ----------
    boundary : np.ndarray or None
        Boundary mask array with dtype of bool or 2-valued uint8.

    Returns
    -------
    np.ndarray or None
        Boolean mask array, or None if input was None.

    Raises
    ------
    ValueError
        If the boundary is not a valid binary or boolean array.
    """
    if boundary is None:
        # print("No boundary provided, returning None.")
        return None
    
    # Check if the boundary is a matrix with only True and False
    if isinstance(boundary, np.ndarray) and np.issubdtype(boundary.dtype, np.bool_):
        # print("Boundary is a valid True/False matrix.")
        return boundary  # No changes needed

    # Check if the boundary has only two values: 0 and another value
    elif isinstance(boundary, np.ndarray) and len(np.unique(boundary)) == 2:
        unique_values = np.unique(boundary)
        if 0 in unique_values:
            # print("Boundary has two values: 0 and another. Casting to True/False.")
            # Cast to True/False: replace 0 with False and others with True
            return boundary != 0
        else:
            raise ValueError("Boundary is not supported. It must be a True/False matrix or have two values (0 and other).")
    
    # If neither condition is met, raise an error
    else:
        raise ValueError("Boundary is not supported. It must be a True/False matrix or have two values (0 and other).")



def check_and_assign_segment_list(segments, init_segments,  last_segments, erosion_steps =None, n_threhsolds=None):
    """
    Generate a list of segment counts per step, supporting flexible control for adaptive seed.
    For both adaptive erosion and adaptive threshold modes.
    
    Parameters
    ----------
    segments : int or list of int
        Number of connected components to retain. Can be fixed or step-specific.
    init_segments : int or None
        Custom number for the first step. Ignored if `segments` is a list.
    last_segments : int or None
        Custom number for the last step. Ignored if `segments` is a list.
    erosion_steps : int, optional
        Number of erosion steps. Mutually exclusive with `n_threhsolds`.
    n_threhsolds : int, optional
        Number of thresholds. Mutually exclusive with `erosion_steps`.

    Returns
    -------
    list of int
        List of segment counts for each step.

    Raises
    ------
    ValueError
        If configuration is ambiguous or inconsistent.
    ValueError
        If length of `segments` list does not match expected length.
        Erosion steps + 2 in adaptive erosion mode, or n_threhsolds + 1 in adaptive threshold mode.
    TypeError
        If `segments` is neither int nor list.
    """    
    # Assert that only one of erosion_steps or n_threhsolds can be not None
    if (erosion_steps is not None) and (n_threhsolds is not None):
        raise ValueError("Only one of 'erosion_steps' or 'n_threhsolds' can be not None.")
    if erosion_steps is not None:
        length = erosion_steps +2
    else:
        length = n_threhsolds +1
    
    
    # Handle segments parameter flexibly
    if isinstance(segments, int):
        segments_list = [segments] * length

        if init_segments is not None:
            segments_list[0] = init_segments

        if last_segments is not None:
            segments_list[-1] = last_segments

    elif isinstance(segments, list):
        if init_segments is not None or last_segments is not None:
            raise ValueError(
                "When 'segments' is a list, please do not use 'init_segments' or 'last_segments' to avoid ambiguity."
            )
        if len(segments) != length:
            raise ValueError(
                f"If 'segments' is a list, its length must be erosion_steps +2 in adaptive erosion mode, or n_threhsolds +1 in adaptive threshold mode."
            )
        segments_list = segments
    else:
        raise TypeError("'segments' must be either an int or a list of ints.")

    return segments_list
    
def check_and_assign_base_name(base_name, img_path, default_base_name):
    """
    Determine a base name for output files from input name or fallback.

    Parameters
    ----------
    base_name : str or None
        Optional name to use.
    img_path : str or None
        Path to input image file.
    default_base_name : str
        Fallback name if none is found.

    Returns
    -------
    str
        Selected base name.
    """
    if base_name is None:
        if img_path is None:
            base_name = default_base_name
        else:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
    return base_name

def check_and_assign_threshold(threshold, upper_threshold):
    """
    Validate and convert a single threshold pair to integers.
    Used in adaptive erosion mode.

    Parameters
    ----------
    threshold : int, float, or list of int
        Lower threshold value or list with one element.
    upper_threshold : int, float, or list of int, optional
        Optional upper threshold for range.

    Returns
    -------
    tuple of int or None
        (threshold, upper_threshold)

    Raises
    ------
    ValueError
        For invalid input types or threshold logic.
    """
    
    
    if isinstance(threshold, list):
        assert len(threshold) == 1, "If threshold is a list, it must have only one element."
        threshold = threshold[0]
    elif isinstance(threshold, float):
        threshold = int(threshold)
    elif isinstance(threshold, int):
        threshold = int(threshold)
    else:
        raise ValueError(f"Invalid type for threshold: {type(threshold)}. Must be int, float, or list.")
    
    if upper_threshold is None:
        upper_threshold = None 
    elif isinstance(upper_threshold, list):
        assert len(upper_threshold) == 1, "If threshold is a list, it must have only one element."
        upper_threshold = upper_threshold[0]
    elif isinstance(upper_threshold, float):
        upper_threshold = int(upper_threshold) 
    elif isinstance(upper_threshold, int):
        upper_threshold = int(upper_threshold)
    else:
        raise ValueError(f"Invalid type for upper_threshold: {type(upper_threshold)}. Must be int, float, or list.")

    if upper_threshold is not None and upper_threshold < threshold:
        raise ValueError(f"Upper threshold must be greater than the threshold. Found threshold={threshold} and upper_threshold={upper_threshold}.")
    return threshold, upper_threshold

def check_and_assign_thresholds(thresholds, upper_thresholds, reverse=False):
    """
    Validate and standardize lists of thresholds and upper thresholds.

    Parameters
    ----------
    thresholds : int, float, or list of int
        Lower threshold(s).
    upper_thresholds : int, float, list of int, or None
        Upper threshold(s). Can be None.
    reverse : bool, default=False
        Whether to check that thresholds are decreasing instead of increasing.

    Returns
    -------
    tuple of list[int], list[int or None]
        Processed thresholds and upper thresholds.

    Raises
    ------
    ValueError
        If type or ordering constraints are violated.
    TODO
    """
    if isinstance(thresholds, int):
        thresholds = [thresholds]
    elif isinstance(thresholds, float):
        thresholds = [int(thresholds)]
    elif isinstance(thresholds, list):
        thresholds = [int(t) for t in thresholds]
    else:
        raise ValueError(f"Invalid type for thresholds: {type(thresholds)}. Must be int, float, or list.")

    if upper_thresholds is None:
        upper_thresholds = [None] * len(thresholds)
    elif isinstance(upper_thresholds, int):
        upper_thresholds = [upper_thresholds]
    elif isinstance(upper_thresholds, float):
        upper_thresholds = [int(upper_thresholds)]
    elif isinstance(upper_thresholds, list):
        upper_thresholds = [int(t) for t in upper_thresholds]
    else:
        raise ValueError(f"Invalid type for upper_thresholds: {type(upper_thresholds)}. Must be int, float, or list.")

    if len(upper_thresholds) != len(thresholds):
        raise ValueError("Length of upper_thresholds must match length of thresholds.")
    
    # Ensure each upper_threshold is greater than the corresponding threshold
    # Only check if upper_thresholds is not all None
    if not all(up is None for up in upper_thresholds):
        for i, (th, up) in enumerate(zip(thresholds, upper_thresholds)):
            if up is not None and up < th:
                raise ValueError(f"Each upper_threshold must be greater than its corresponding threshold. Found thresholds[{i}]={th} and upper_thresholds[{i}]={up}.")
    
    if reverse:
        # Ensure thresholds are equal or strictly decreasing
        if any(thresholds[i+1] > thresholds[i] for i in range(len(thresholds)-1)):
            raise ValueError("Thresholds must be equal or strictly decreasing.")
        
        # Ensure upper_thresholds are equal or strictly decreasing, but only if not all None
        if not all(up is None for up in upper_thresholds):
            filtered_upper = [up for up in upper_thresholds if up is not None]
            if any(filtered_upper[i] > filtered_upper[i+1] for i in range(len(filtered_upper)-1)):
                raise ValueError("Upper thresholds must be equal or strictly increasing (ignoring None values).")        
    else:
        # Ensure thresholds are equal or strictly increasing
        if any(thresholds[i] > thresholds[i+1] for i in range(len(thresholds)-1)):
            raise ValueError("Thresholds must be equal or strictly increasing.")
        
        # Ensure upper_thresholds are equal or strictly increasing, but only if not all None
        if not all(up is None for up in upper_thresholds):
            filtered_upper = [up for up in upper_thresholds if up is not None]
            if any(filtered_upper[i+1] > filtered_upper[i] for i in range(len(filtered_upper)-1)):
                raise ValueError("Upper thresholds must be equal or strictly decreasing (ignoring None values).")
    
    
    return thresholds, upper_thresholds

def check_and_assign_footprint(footprints, erosion_steps , with_folder_name=False):
    # TODO check docstring
    # TODO add support for 2D footprints
    """
    Validate and construct a list of erosion footprints.

    Parameters
    ----------
    footprints : str or list of str
        Either a single named footprint or a list for each erosion step.
    erosion_steps : int
        Number of erosion steps.
    with_folder_name : bool, default=False
        If True, also return a folder name derived from the footprint.

    Returns
    -------
    list of str or (list of str, str)
        List of footprint names. If `with_folder_name`, also returns folder name.

    Raises
    ------
    ValueError
        If footprint is not valid or list length does not match erosion steps.
    """
    if footprints is None:
        footprints = "ball"
    
    if isinstance(footprints, str):

        assert footprints in support_footprints, f"footprint {footprints} is invalid, use supported footprints"
        footprint_list = [footprints]*erosion_steps
        if with_folder_name:
            folders = footprints
        
    elif isinstance(footprints, list):
        assert len(footprints) ==erosion_steps, "If input_footprints is a list, it must have the same length as erosion_steps"
        
        check_support_footprint = [footprint in support_footprints for footprint in footprints]
        if not np.all(check_support_footprint):
            raise ValueError(f"footprint {footprints} is invalid, use supported footprints")
        if with_folder_name:
            folders = "custom_footprints"
        
        footprint_list = footprints
    else:
        raise ValueError(f"Can't set the footprint list with the input footprint {footprints} ")
    
    if with_folder_name:
        return footprint_list , folders
    else:
        return footprint_list
# Function to recursively create global variables from the config dictionary
def load_config_yaml(config, parent_key=''):
    for key, value in config.items():
        if isinstance(value, dict):
            load_config_yaml(value, parent_key='')
        else:
            globals()[parent_key + key] = value
            

# Define a function to read the configuration and set variables dynamically
def load_config_json(file_path):
    with open(file_path, 'r') as config_file:
        config = json.load(config_file)

    # Dynamically set variables in the global namespace
 
    for key, value in config.items():
        globals()[key] = value
 
 
def write_json(filename, args_dict):
    """
    Write dictionary data to a JSON file, appending it if the file exists.
    Use to write seed generation log.
    Args:
        filename (str): Path to the JSON file.
        args_dict (dict): Data to be written to the file.
    """    
    # Check if the file exists and load existing data
    if os.path.exists(filename):
        with open(filename, 'r') as jsonfile:
            results = json.load(jsonfile)
    else:
        results = []
    
    results.append(args_dict)
    
    # Write the results to the JSON file
    with open(filename, 'w') as jsonfile:
        json.dump(results, jsonfile, indent=4)

def save_config_with_output(output_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    config_path = os.path.join(output_dir, f"config_{timestamp}.yaml")

    # Set 'img', 'seg', 'boundary' keys to None if they exist and value is not None
    for k in ("img", "seg", "boundary"):
        if k in output_dict['params'] and output_dict['params'][k] is not None:
            output_dict['params'][k] = None


    with open(config_path, "w") as f:
        yaml.dump(output_dict, f)

    print(f"Input config for this run has been saved to : {config_path}")
    return config_path

def check_file_extension(file_path):
    # Get the file extension
    _, extension = os.path.splitext(file_path)
    
    # Check if the extension is either .json, .yaml, or .yml
    if extension.lower() in ['.json', '.yaml', '.yml']:
        return True
    else:
        return False




###############


def validate_input_yaml(config, rules):
    errors = []
    optional_values = {}
    
    for key in config:
        if key not in rules:
            errors.append(f"Unsupported parameter '{key}' in input config.")

    for param, rule in rules.items():
        # Check if required parameters are present
        if rule.get("required", False) and param not in config:
            errors.append(f"Missing required parameter: '{param}' ({rule['description']})")
            continue  # Skip further checks for missing params
        if not(rule.get("required", False)):
            optional_values[param] = config.get(param, rule.get("default"))    
        
        if rule.get("either", False):
            either_keys = rule['either']
            either_matches = sum(1 for either_key in either_keys if either_key in config)
            if either_matches!=1:
                errors.append(f"Only one value from {either_keys} can be included in the yaml")
        
        # Use default value if not provided
        value = config.get(param, rule.get("default"))
        
        # Try parsing if rule expects a list but input is string
        if isinstance(value, str) and ((isinstance(rule["type"], str) and rule["type"]== list) or (isinstance(rule["type"], tuple) and  list in rule["type"])):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    value = parsed
                    config[param] = value

            except Exception as e:
                pass
        
        
        if rule.get("required", False) or (not(rule.get("required", False)) and (value is not None)):
            # Check data type
            if not (isinstance(value, rule["type"])):
                errors.append(f"Parameter '{param}' should be of type {rule['type'].__name__}, got {type(value).__name__}.")
                continue 
            
            # If list, check subtype
            if isinstance(value, list) and "subtype" in rule:
                if not all(isinstance(item, rule["subtype"]) for item in value):
                    errors.append(f"All elements in '{param}': {value} must be of type {rule['subtype']}.")
                else:
                    for item in value:
                        # Check min/max constraints for list value
                        if "min" in rule and item < rule["min"]:
                            errors.append(f"Parameter '{param}' must have all values >= {rule['min']}, got {item} in {value}.")
                        if "max" in rule and item > rule["max"]:
                            errors.append(f"Parameter '{param}' must have all values <= {rule['max']}, got {item} in {value}.")
            else:
                # Check min/max constraints for non list value
                if "min" in rule and value < rule["min"]:
                    errors.append(f"Parameter '{param}' must be >= {rule['min']}, got {value}.")
                if "max" in rule and value > rule["max"]:
                    errors.append(f"Parameter '{param}' must be <= {rule['max']}, got {value}.")
                
            # Check choices for categorical variables
            if "choices" in rule and value not in rule["choices"]:
                errors.append(f"Parameter '{param}' must be one of {rule['choices']}, got '{value}'.")
            if rule.get("check_exist", False):
                if 'workspace' in rules and 'workspace' in config:
                    value = os.path.join(config['workspace'] , value)
                if not os.path.exists(value):
                    errors.append(f"Parameter '{param}' points to a non-existing path: {value}")
            # Check file extension if specified
            if "check_extension" in rule:
                valid_extensions = rule["check_extension"]
                if not value.lower().endswith(valid_extensions):
                    errors.append(f"Parameter '{param}' must have extension {valid_extensions}, got '{value}'.")


    if errors:
        raise ValueError("\n".join(errors) + "\n\nPlease check README.md and templates .yaml file in the ./template/ folder")

    print("Config file is valid!\n")
    return optional_values






input_val_make_seeds = {

    "img_path": {
        "type": str,
        "required": True,
        "description": "File name",
        "check_exist": True,
        "check_extension": (".tif", ".tiff")

    },
    "num_threads": {
        "type": int,
        "min": 1,
        "max": psutil.cpu_count(),
        "required": True,
        "description": "num of threads"
    },
    "thresholds": {
        "type": (int,list),
        "subtype": (int, float),  # Each element should be int or float
        "required": True,
        "min_length": 1,
        "min": 0,
        "description": "List of thresholds, must contain at least one numeric value."
    },
    "erosion_steps": {
        "type": int,
        # "subtype": int,
        "min": 0,
        "max": 1000,
        "required": True,
        "description": "List of number of iterations, if just one value, using bracket e.g., [3]"
    },
    "segments": {
        "type": int,
        "min": 1,
        "required": True,
        "description": "The algorithm keeps top <segments> largest disconnected components"
    },
    "output_folder": {
        "type": str,
        "required": True,
        "description": "Output directory path as a string."
    },
    "footprints": {
        "type": str,
        "required": True,
        "description": "Footprints for morphological transformation"
    },
    #### Optional parameters
    "workspace": {
        "type": str,
        "required": False,
        "default":"",
        "description": "Workspace folder, default is an empty string"

    },    
    "upper_thresholds": {
        "type": (int,list),
        "subtype": (int, float),  # Each element should be int or float
        "required": False,
        "default":None,
        "min_length": 1,
        "min": 0,
        "description": "List of upper thresholds, must contain at least one numeric value."
    },

    "boundary_path": {
        "type": str,
        "required": False,
        "default":None,
        "check_exist": True,
        "description": "boundary image",
        "check_extension": (".tif", ".tiff")
    },
    
    "base_name": {
        "type": str,
        "required": False,
        'default': None,
        "description": "base_name for naming output files and folders."
    }, 
    
    "is_make_meshes": {
        "type": bool,
        "required": False,
        "default": False
    },
    "downsample_scale": {
        "type": int,
        "min": 1,
        "max": 100,
        "required": False,
        "default":10,
        "description": "Scale for downsampling"
    },
    "step_size": {
        "type": int,
        "min": 1,
        "max": 10,
        "required": False,
        "default": 1,
        "description": "Step size in Marching Cubes alogrithms, Default is 1"
    },
    "footprints": {
            "type": (str,list),
        "required": False,
        "default": None,
        "description": "Footprints for morphological transformation"
    }
}






input_val_make_grow = {
    "img_path": {
        "type": str,
        "required": True,
        "description": "Image path",
        "check_exist": True,
        "check_extension": (".tif", ".tiff")

    },
    "seg_path": {
        "type": str,
        "required": True,
        "description": "segmentation/seed path",
        "check_exist": True,
        "check_extension": (".tif", ".tiff")

    },
    "num_threads": {
        "type": int,
        "min": 1,
        "max": psutil.cpu_count(),
        "required": True,
        "description": "num of threads"
    },

    "dilation_steps": {
        "type": (int,list),
        "subtype": int,
        "min": 0,
        "required": True,
        "description": "Number of iterations, must be a non-negative integer."
    },
    "thresholds": {
        "type": (int,list),
        "subtype": (int, float),  # Each element should be int or float
        "required": True,
        "min_length": 1,
        "min": 0,
        "description": "List of thresholds, must contain at least one numeric value."
    },
    "touch_rule": {
        "type": str,
        "choices": ["stop", "overwrite"],
        "required": True,
        "description": "The rule of growing, current supports 'stop' and 'overwrite'.",
    },
    "output_folder": {
        "type": str,
        "required": True,
        "description": "Output directory path as a string."
    },

    #### Optional parameters
    "save_every_n_iters": {
        "type": int,
        "min": 1,
        "required": False,
        "default": None,
        "description": "Save the grow result every n iters"
    }, 
    
    "workspace": {
        "type": str,
        "required": False,
        "default":"",
        "description": "Workspace folder, default is an empty string"

    },    
    "upper_thresholds": {
        "type": (int,list),
        "subtype": (int, float),  # Each element should be int or float
        "required": False,
        'default': None,
        "min_length": 1,
        "min": 0,
        "description": "List of upper thresholds, must contain at least one numeric value."
    },
    "boundary_path": {
        "type": str,
        "required": False,
        'default': None,
        "check_exist": True,
        "description": "boundary image",
        "check_extension": (".tif", ".tiff")
    },
    "grow_to_end": {
        "type": bool,
        "required": False,
        'default': False
    },    
    "final_grow_output_folder": {
        "type": str,
        "required": False,
        'default': None,
        "description": "Specify final grow folder if specified"
    },    
    "base_name": {
        "type": str,
        "required": False,
        'default': None,
        "description": "The name prefix"
    },   
    "use_simple_naming": {
        "type": bool,
        "required": False,
        'default': True
    },    
    "to_grow_ids": {
        "type": list,
        "subtype": (int),  # Each element should be int or float
        "required": False,
        'default': None,
        "min_length": 1,
        "min": 0,
        "description": "ids to grow"
    },
    "is_sort": {
        "type": bool,
        "required": False,
        'default': False,
        "description": "Sort the result on the segments by size"
    },    
    "no_growth_max_iter": {
        "type": int,
        "min": 1,
        "required": False,
        'default': 3,
        "description": "limit for consecutive no-growth iterations."
    },    
    "min_growth_size": {
        "type": int,
        "min": 0,
        "required": False,
        'default': 50,
        "description": "The minimum difference to consider there is a growth in a dilation iteration"
    },    
    
    "is_make_meshes": {
        "type": bool,
        "required": False,
        "default": False
    },
    "downsample_scale": {
        "type": int,
        "min": 1,
        "max": 100,
        "required": False,
        "default":10,
        "description": "Scale for downsampling"
    },
    "step_size": {
        "type": int,
        "min": 1,
        "max": 10,
        "required": False,
        "default": 1,
        "description": "Step size in Marching Cubes alogrithms, Default is 1"
    }
}



input_val_make_adaptive_seed = {

    "img_path": {
        "type": str,
        "required": True,
        "description": "File name",
        "check_exist": True,
        "check_extension": (".tif", ".tiff")

    },
    "num_threads": {
        "type": int,
        "min": 1,
        "max": psutil.cpu_count(),
        "required": True,
        "description": "num of threads"
    },
    "thresholds": {
        "type": (int,list),
        "subtype": (int, float),  # Each element should be int or float
        "required": True,
        "min_length": 1,
        "min": 0,
        "description": "List of thresholds, must contain at least one numeric value."
    },
    "erosion_steps": {
        "type": int,
        "min": 0,
        "max": 1000,
        "required": True,
        "description": "Number of iterations, must be a non-negative integer."
    },
    "segments": {
        "type": (int,list),
        "subtype": int,
        "min": 1,
        "required": True,
        "description": "The algorithm keeps top <segments> largest disconnected components."
    },
    "output_folder": {
        "type": str,
        "required": True,
        "description": "Output directory path as a string."
    },
    #### Optional parameters

    "upper_thresholds": {
        "type": (int,list),
        "subtype": (int, float),  # Each element should be int or float
        "required": False,
        "default":None,
        "min_length": 1,
        "min": 0,
        "description": "List of upper thresholds, must contain at least one numeric value."
    },

    "boundary_path": {
        "type": str,
        "required": False,
        "default":None,
        "check_exist": True,
        "description": "boundary image",
        "check_extension": (".tif", ".tiff")
    },
    "background": {
        "type": int,
        "min": 0,
        "required": False,
        "default": 0,
        "description": "Background value. Defaults is 0."
    },
    "sort": {
        "type": bool,
        "required": False,
        'default': True
    },   
    ## Optional for early stopping
    "no_split_max_iter": {
        "type": int,
        "min": 0,
        "required": False,
        "default": 3,
        "description": "Limit for consecutive no-split iterations. Defaults is 3."
    },
    "min_size": {
        "type": int,
        "min": 1,
        "required": False,
        "default": 1,
        "description": "Minimum size for segments. Defaults is 1."
    },
    "min_split_ratio": {
        "type": (int,float),
        "min": 0,
        "max": 1,
        "required": False,
        "default": 0,
        "description": "Minimum proportion to consider a split. Defaults is 0."
    },
    "min_split_total_ratio": {
        "type": (int,float),
        "min": 0,
        "max": 1,
        "required": False,
        "default": 0,
        "description": "Minimum proportion of (sub-segments from next step)/(current segments) to consider a split"
    },   
    
    "save_every_iter": {
        "type": bool,
        "required": False,
        'default': True,
        "description": "Save results at every iteration. Defaults is False."
    },            
    "base_name": {
        "type": str,
        "required": False,
        'default': None,
        "description": "base_name for naming output files and folders."
    }, 
    
    "init_segments": {
        "type": int,
        "min": 1,
        "required": False,
        "default": None,
        "description": "Number of segments for the first seed."
    },

    "last_segments":{
        "type": int,
        "min": 1,
        "required": False,
        "default": None,
        "description": "Number of segments for the last seed."
    },   
    "footprints": {       
        "type": (str,list),
        "required": False,
        "default": None,
        "description": "Footprints for morphological transformation"
    },
    "split_size_limit": {
        "type": list,
        "subtype": (int,type(None)),  # Each element should be int or float
        "required": False,
        "default": [None,None],
        "min_length": 2,
        "description": "create a split if the region size (np.sum(mask)) is within the limit"
    },
    "split_convex_hull_limit": {
        "type": list,
        "subtype": (int,type(None)),  # Each element should be int or float
        "required": False,
        "default": [None,None],
        "min_length": 2,
        "description": "create a split if the the convex hull's area/volume is within the limit"
    },  
   
}


input_val_make_mesh = {
    "num_threads": {
        "type": int,
        "min": 1,
        "max": psutil.cpu_count(),
        "required": True,
        "description": "num of threads"
    },
    "output_folder": {
        "type": str,
        "required": True,
        "description": "Output folder"
    },
    

    #### Optional #####
    "input_folder": {
        "type": str,
        "required": False,
        "default":None,
        "either" : ("input_folder", "img_path"),
        "description": "Output folder",
        "check_exist": True
    },
    "img_path": {
        "type": str,
        "required": False,
        "default":None,
        "description": "File name",
        "either" : ("input_folder", "img_path"),
        "check_exist": True,
        "check_extension": (".tif", ".tiff")

    }, 
    
    "downsample_scale": {
        "type": int,
        "min": 1,
        "max": 100,
        "required": False,
        "default":10,
        "description": "Scale for downsampling"
    },
    "step_size": {
        "type": int,
        "min": 1,
        "max": 10,
        "required": False,
        "default": 1,
        "description": "Step size in Marching Cubes alogrithms, Default is 1"
    },
}

input_val_make_junctions =  {

    "input_folder": {
        "type": str,
        "required": False,
        "default":None,
        "either" : ("input_folder", "img_path"),
        "description": "Output folder",
        "check_exist": True
    },
    "img_path": {
        "type": str,
        "required": False,
        "default":None,
        "description": "File name",
        "either" : ("input_folder", "img_path"),
        "check_exist": True,
        "check_extension": (".tif", ".tiff")

    }, 

    # "img_path": {
    #     "type": str,
    #     "required": True,
    #     "description": "File name",
    #     "check_exist": True,
    #     "check_extension": (".tif", ".tiff")

    # },
    "num_threads": {
        "type": int,
        "min": 1,
        "max": psutil.cpu_count(),
        "required": True,
        "description": "num of threads"
    },
    "output_folder": {
        "type": str,
        "required": True,
        "description": "Output directory path as a string."
    },
    "max_width": {
        "type": int,
        "min": 1,
        "required": True,
        "description": "Maximum width of detected junctions."
    },
    #### Optional parameters
    "is_save_islands":{
        "type": bool,
        "required": False,
        'default': False,
        "description": "Save the islands of a junction as well"        
    },

    "boundary_path": {
        "type": str,
        "required": False,
        "default":None,
        "check_exist": True,
        "description": "boundary image",
        "check_extension": (".tif", ".tiff")
    },
    "background": {
        "type": int,
        "min": 0,
        "required": False,
        "default": 0,
        "description": "Background value. Defaults is 0."
    },
   
}


input_val_sam_run = {
    "img_path": {
        "type": str,
        "required": True,
        "description": "Path to input image",
        "check_exist": True,
        "check_extension": (".tif", ".tiff", ".png")
    },
    "seg_path": {
        "type": str,
        "required": True,
        "description": "Path to input segmentation",
        "check_exist": True,
        "check_extension": (".tif", ".tiff", ".png")
    },
    "n_points_per_class": {
        "type": int,
        "min": 1,
        "required": False,
        "default": 3,
        "description": "Number of positive points sampled per class"
    },
    "prompt_type": {
        "type": str,
        "required": False,
        "default": "point",
        "description": "Prompt type for SAM",
        "choices": ["point", "bbox"]
    },
    "output_folder": {
        "type": str,
        "required": True,
        "description": "Folder to save prompt and segmentation outputs"
    },
    "output_filename": {
        "type": str,
        "required": False,
        "default": None,
        "description": "Filename of final merged segmentation"
    },
    "device": {
        "type": str,
        "required": False,
        "default": "cuda",
        "description": "Device to run SAM model on (e.g., 'cuda', 'cpu')"
    },
    "sample_neg_each_class": {
        "type": bool,
        "required": False,
        "default": False,
        "description": "If True, sample negative points for each non-target class individually"
    },
    "negative_points": {
        "type": int,
        "min": 0,
        "required": False,
        "default": None,
        "description": "Number of negative points sampled per class"
    },
    "sample_method": {
        "type": str,
        "required": False,
        "default": "random",
        "description": "Method to sample points: 'kmeans', 'center_edge', 'skeleton', or 'random'"
    },
    "per_cls_mode": {
        "type": bool,
        "required": False,
        "default": True,
        "description": "Whether to perform per-class majority vote and fusion"
    },
    "which_sam": {
        "type": str,
        "required": False,
        "default": "sam1",
        "choices": ["sam1", "sam2"],
        "description": "Select whether to use SAM1 or SAM2 model"
    },
    "sam_checkpoint": {
        "type": str,
        "required": False,
        "default": None,
        "description": "Path to the SAM1 checkpoint file",
        "check_exist": True
    },
    "sam_model_type": {
        "type": str,
        "required": False,
        "default": None,
        "description": "SAM1 model architecture (e.g., vit_b, vit_l, vit_h)"
    },
    "sam2_checkpoint": {
        "type": str,
        "required": False,
        "default": None, 
        "description": "Path to the SAM2 checkpoint file",
        "check_exist": True
    },
    "sam2_model_cfg": {
        "type": str,
        "required": False,
        "default": None,
        "description": "Config file for SAM2 model",
        "check_exist": True
    },
    "custom_checkpoint": {
        "type": str,
        "required": False,
        "default": None,
        "description": "Path to the custom checkpoint file",
        "check_exist": True
    }   
}

def return_thre_value(input_config, img):
    if isinstance(input_config, (int, float)):
        return input_config
    
    method_all = input_config
    method = method_all.split("_")[0]
    try:
        percentage = int(method_all.split("_")[1])
    except:
        percentage = 100
    if not (0 <= percentage <= 100):
        raise ValueError(f"Invalid Otsu percentage: {percentage}. Must be between 0 and 100.") 

    if method == "otsu":
        otsu_val = int(threshold_otsu(img) * percentage/100)
    else:
        raise ValueError(f"{input_config} is not supported threshold method")
    return otsu_val


def process_images_with_config(input_csv, output_csv, input_config):
    """
    Reads a CSV file with image paths, processes images based on input_config, 
    and outputs a new CSV with assigned values.

    Args:
        input_csv (str): Path to the input CSV containing the "img_path" column.
        output_csv (str): Path to save the processed CSV.
        input_config (dict): Dictionary of parameters to assign to the DataFrame.
                            - If a value is a number or list, it applies to all rows.
                            - If a value is "otsu", it computes Otsu's threshold per image.

    Returns:
        None
    """
    # Load the CSV file
    df = pd.read_csv(input_csv)

    # Ensure "img_path" column exists
    if "img_path" not in df.columns:
        raise ValueError("CSV must contain an 'img_path' column with image file paths.")

    for key, value in input_config.items():
        if isinstance(value, (int, float)):  
            # Assign a constant value (or list) to all rows
            df[key] = value
        elif isinstance(value, (list)):  
            df[key] = str(value)
        elif isinstance(value, dict):
            
            type = value["type"]

            thre_values = []
            for index, row in df.iterrows():
                img_path = row["img_path"]
                if not os.path.exists(img_path):
                    print(f"Skipping {img_path}: File not found.")
                    thre_values.append(None)
                    continue

                # Read image
                img = tiff.imread(img_path)

                # Ensure it's grayscale (2D)
                if img.ndim != 2:
                    print(f"Skipping {img_path}: Not a 2D grayscale image.")
                    thre_values.append(None)
                    continue



                if type == "single":
                    # Compute Otsu's threshold
                    otsu_val = return_thre_value(value['method'], img)
                    thre_values.append(otsu_val)
                elif type == "list":
                    upper =  return_thre_value(value['upper'], img)
                    lower = return_thre_value(value['lower'], img)
                    N = value['N']
                    ascending = value["ascending"]
                    if ascending:
                        otsu_list = list(np.linspace( lower ,upper, N, dtype=int))
                    else:
                        otsu_list = list(np.linspace( lower ,upper, N, dtype=int))
                        otsu_list = sorted(otsu_list,reverse=True)
                    thre_values.append(otsu_list)

            df[key] = thre_values

        
    # Save updated CSV
    df.to_csv(output_csv, index=False)
    print(f"Processed CSV saved to: {output_csv}")


def copy_matching_files(src_folder, dest_folder, prefix, extension):
    """
    Searches for files with a specific prefix and extension in all subdirectories of src_folder 
    and copies them to dest_folder.

    Args:
        src_folder (str): The root folder to search for matching files.
        dest_folder (str): The folder where matching files will be copied.
        prefix (str): The required prefix of the files.
        extension (str): The required file extension (e.g., ".tif", ".txt").
    """
    # Ensure the destination folder exists
    os.makedirs(dest_folder, exist_ok=True)

    # Walk through all subdirectories
    for root, _, files in os.walk(src_folder):
        for file in files:
            if file.startswith(prefix) and file.endswith(extension):
                src_path = os.path.join(root, file)  # Full path of the source file
                dest_path = os.path.join(dest_folder, file)  # Destination path

                # Copy file to the destination folder
                shutil.copy2(src_path, dest_path)
                print(f"Copied: {src_path} → {dest_path}")

def merge_row_and_yaml_no_conflict(row: dict, yaml_config: dict) -> dict:
    duplicate_keys = set(row) & set(yaml_config)
    if duplicate_keys:
        raise ValueError(f"Conflict detected: keys present in both CSV row and YAML config: {list(duplicate_keys)}")
    
    # Merge with row taking precedence if needed (not used here due to check)
    merged = {**yaml_config, **row}
    return merged

if __name__ == "__main__":
    df = pd.read_csv("config/seeds_input.csv")

    import yaml
    # config = yaml.safe_load("../PipelineSeed.yaml")

    with open("PipelineSeed_v2.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    config.pop("csv_path", None)

    for idx, row in df.iterrows():
        new_config = merge_row_and_yaml_no_conflict(row, config)

        
        try:
            optional = validate_input_yaml(new_config, input_val_make_adaptive_seed)
            print(optional)
        except Exception as e:
            print("Error when valid make_adaptive_seed", e)
        
        try:
            validate_input_yaml(new_config, input_val_make_seeds)
        except Exception as e:
            print("Error when valid make_seeds", e)  


    

# try:
#     check_required_keys(yaml_path, required_keys)
# except Exception as e:
#     print(e)

# if __name__ == "__main__":           
#     # Load the YAML file
#     with open('./test.yaml', 'r') as file:
#         config = yaml.safe_load(file)
#     load_config_yaml(config)
#     print(string_value, type(int_value), list_value)
    
#     load_config_json('./make_seeds.json')
    
#     print(workspace, output_seed_folder)