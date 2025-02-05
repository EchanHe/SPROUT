import yaml
import json
import os
import psutil
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
 

def check_file_extension(file_path):
    # Get the file extension
    _, extension = os.path.splitext(file_path)
    
    # Check if the extension is either .json, .yaml, or .yml
    if extension.lower() in ['.json', '.yaml', '.yml']:
        return True
    else:
        return False






def validate_input_yaml(config, rules):
    errors = []
    optional_values = {}
    for param, rule in rules.items():
        # Check if required parameters are present
        if rule.get("required", False) and param not in config:
            errors.append(f"Missing required parameter: '{param}' ({rule['description']})")
            continue  # Skip further checks for missing params
        if not(rule.get("required", False)):
            optional_values[param] = config.get(param, rule.get("default"))    
        
        # Use default value if not provided
        value = config.get(param, rule.get("default"))
        
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
                if 'workspace' in rules:
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

    print("YAML file is valid!")
    return optional_values





input_val_make_seeds = {
    "workspace": {
        "type": str,
        "required": True,
        "description": "Workspace folder, can be an empty string"

    },
    "file_name": {
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
        "type": list,
        "subtype": (int, float),  # Each element should be int or float
        "required": True,
        "min_length": 1,
        "min": 0,
        "description": "List of thresholds, must contain at least one numeric value."
    },
    "ero_iters": {
        "type": (int,list),
        "subtype": int,
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
    "output_log_file": {
        "type": str,
        "required": True,
        "description": "Output log",
        "check_extension": ".json"
    },
    "footprints": {
        "type": str,
        "required": True,
        "description": "Footprints for morphological transformation"
    },
    #### Optional parameters
    
    "upper_thresholds": {
        "type": list,
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
    }
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
input_val_make_seeds_all['footprints']['required'] = False



input_val_make_grow = {
    "workspace": {
        "type": str,
        "required": True,
        "description": "Workspace folder, can be an empty string"

    },
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
        "type": list,
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
    "upper_thresholds": {
        "type": list,
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
    "name_prefix": {
        "type": str,
        "required": False,
        'default': "final_grow",
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

input_val_make_seeds_merged = {

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
    "n_iters": {
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
        'default': False,
        "description": "Save results at every iteration. Defaults is False."
    },       
     "save_merged_every_iter": {
        "type": bool,
        "required": False,
        'default': False,
        "description": "Save merged results at every iteration. Defaults is False."
    },      
    "name_prefix": {
        "type": str,
        "required": False,
        'default': "Merged_seed",
        "description": "Prefix for output file names. Defaults is Merged_seed."
    }, 
    "init_segments": {
        "type": int,
        "min": 1,
        "required": False,
        "default": None,
        "description": "Number of segments for the first seed."
    },
    "footprints": {
        "type": str,
        "required": False,
        "default": "ball",
        "description": "Footprints for morphological transformation"
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
        "description": "Output folder",
        "check_exist": True
    },
    

    #### Optional #####
    "input_folder": {
        "type": str,
        "required": False,
        "default":None,
        "description": "Output folder",
        "check_exist": True
    },
    "img_path": {
        "type": str,
        "required": False,
        "default":None,
        "description": "File name",
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