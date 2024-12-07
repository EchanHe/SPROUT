import yaml
import json
import os

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







try:
    check_required_keys(yaml_path, required_keys)
except Exception as e:
    print(e)

if __name__ == "__main__":           
    # Load the YAML file
    with open('./test.yaml', 'r') as file:
        config = yaml.safe_load(file)
    load_config_yaml(config)
    print(string_value, type(int_value), list_value)
    
    load_config_json('./make_seeds.json')
    
    print(workspace, output_seed_folder)