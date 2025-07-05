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

support_footprints =['ball','cube',
                     'ball_XY','ball_XZ','ball_YZ',
                     'X','Y','Z',
                     '2XZ_1Y','2XY_1Z','2YZ_1X']


def check_and_assign_base_name(base_name, img_path, default_base_name):
    if base_name is None:
        if img_path is None:
            base_name = default_base_name
        else:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
    return base_name

def check_and_assign_thresholds(thresholds, upper_thresholds, reverse=False):
    """ Check and assign thresholds and upper thresholds, turn them all to int.
    Args:
        thresholds (int, float, list): Thresholds to be checked and assigned.
        upper_thresholds (int, float, list, optional): Upper thresholds to be checked and assigned. Defaults to None.
    Returns:
        tuple: A tuple containing the processed thresholds and upper thresholds as lists.
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
            if up is not None and up <= th:
                raise ValueError(f"Each upper_threshold must be greater than its corresponding threshold. Found thresholds[{i}]={th} and upper_thresholds[{i}]={up}.")
    
    if reverse:
        # Ensure thresholds are equal or strictly decreasing
        if any(thresholds[i+1] > thresholds[i] for i in range(len(thresholds)-1)):
            raise ValueError("Thresholds must be equal or strictly decreasing.")
        
        # Ensure upper_thresholds are equal or strictly decreasing, but only if not all None
        if not all(up is None for up in upper_thresholds):
            filtered_upper = [up for up in upper_thresholds if up is not None]
            if any(filtered_upper[i+1] > filtered_upper[i] for i in range(len(filtered_upper)-1)):
                raise ValueError("Upper thresholds must be equal or strictly decreasing (ignoring None values).")        
    else:
        # Ensure thresholds are equal or strictly increasing
        if any(thresholds[i] > thresholds[i+1] for i in range(len(thresholds)-1)):
            raise ValueError("Thresholds must be equal or strictly increasing.")
        
        # Ensure upper_thresholds are equal or strictly increasing, but only if not all None
        if not all(up is None for up in upper_thresholds):
            filtered_upper = [up for up in upper_thresholds if up is not None]
            if any(filtered_upper[i] > filtered_upper[i+1] for i in range(len(filtered_upper)-1)):
                raise ValueError("Upper thresholds must be equal or strictly increasing (ignoring None values).")
    
    
    return thresholds, upper_thresholds

def check_and_assign_footprint(footprints, ero_iters , with_folder_name=False):
    if isinstance(footprints, str):

        assert footprints in support_footprints, f"footprint {footprints} is invalid, use supported footprints"
        footprint_list = [footprints]*ero_iters
        if with_folder_name:
            folders = footprints
        
    elif isinstance(footprints, list):
        assert len(footprints) ==ero_iters, "If input_footprints is a list, it must have the same length as ero_iters"
        
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
        raise ValueError("\n".join(errors) + "\n\nPlease check the corresponding template .yaml file in the ./template/ folder")

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
    "ero_iters": {
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
    # "output_log_file": {
    #     "type": str,
    #     "required": True,
    #     "description": "Output log",
    #     "check_extension": ".json"
    # },
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
}



input_val_make_seeds_all = input_val_make_seeds.copy()

mesh_dict = {
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
}

input_val_make_seeds_all.update(mesh_dict)
# input_val_make_seeds_all['footprints']['required'] = False
# input_val_make_seeds_all.pop("footprints")
input_val_make_seeds_all["footprints"] =  {
        "type": (str,list),
        "required": False,
        "default": None,
        "description": "Footprints for morphological transformation"
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

    "dilate_iters": {
        "type": (int,list),
        "subtype": int,
        "min": 0,
        "max": 1000,
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
        "choices": ["stop", "no"],
        "required": True,
        "description": "The rule of growing"
    },
    "output_folder": {
        "type": str,
        "required": True,
        "description": "Output directory path as a string."
    },
    "save_interval": {
        "type": int,
        "min": 1,
        "required": True,
        "description": "Save the grow result every n iters"
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
    "simple_naming": {
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
        'default': True
    },    
    "tolerate_iters": {
        "type": int,
        "min": 1,
        "required": False,
        'default': 3,
        "description": "limit for consecutive no-growth iterations."
    },    
    "min_diff": {
        "type": int,
        "min": 0,
        "required": False,
        'default': 50,
        "description": "The minimum difference to consider there is a growth in a dilation iteration"
    },    
}

input_val_make_grow.update(mesh_dict)

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
    "ero_iters": {
        "type": int,
        "min": 0,
        "max": 1000,
        "required": True,
        "description": "Number of iterations, must be a non-negative integer."
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
    "no_split_limit": {
        "type": int,
        "min": 0,
        "required": False,
        "default": 3,
        "description": "Limit for consecutive no-split iterations. Defaults is 3."
    },
    "min_size": {
        "type": int,
        "min": 0,
        "required": False,
        "default": 5,
        "description": "Minimum size for segments. Defaults is 5."
    },
    "min_split_prop": {
        "type": (int,float),
        "min": 0,
        "max": 1,
        "required": False,
        "default": 0.01,
        "description": "Minimum proportion to consider a split. Defaults is 0.01."
    },
    "min_split_sum_prop": {
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
     "save_merged_every_iter": {
        "type": bool,
        "required": False,
        'default': False,
        "description": "Save merged results at every iteration. Defaults is False."
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
    "footprints": {       
        "type": (str,list),
        "required": False,
        "default": "ball",
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
            validate_input_yaml(new_config, input_val_make_seeds_all)
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