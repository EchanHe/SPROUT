import make_seeds
import make_adaptive_seed

import yaml
import os,sys
import tifffile

import pandas as pd
import sprout_core.sprout_core as sprout_core
import sprout_core.config_core as config_core

from datetime import datetime

# def load_config_yaml(config, parent_key=''):
#     for key, value in config.items():
#         if isinstance(value, dict):
#             load_config_yaml(value, parent_key='')
#         else:
#             globals()[parent_key + key] = value



def run_batch_seeds(file_path):
    
    _, extension = os.path.splitext(file_path)
    print(f"processing config the file {file_path}")
    if extension == '.yaml':
        with open(file_path, 'r') as file:
            yaml_config = yaml.safe_load(file)

    print("Config for pipeline")
    for key, value in yaml_config.items():
        print(f"\t{key}: {value}")

    csv_path = yaml_config['csv_path']
    df = pd.read_csv(csv_path)
    sprout_core.check_tiff_files(df['img_path'])

    for index, row in df.iterrows():
        ## Initial the config and optional parameters for each row
        yaml_config.pop("csv_path", None)
        config = config_core.merge_row_and_yaml_no_conflict(dict(row), yaml_config)
        optional_params = config_core.validate_input_yaml(config, config_core.input_val_make_seeds_all)
        try:           
               
            _,_ = make_seeds.make_seeds(
                                    img_path= config['img_path'],
                                    
                                    num_threads = config['num_threads'] , 
                                    boundary_path=optional_params['boundary_path'],
                                    
                                    output_folder = config['output_folder'],
                                    ero_iters = config['ero_iters'],
                                    thresholds = config['thresholds'],
                                    segments = config['segments'],
                                    
                                    input_footprints = optional_params['footprints'],
                                    
                                    upper_thresholds = optional_params['upper_thresholds']
                                    )
        except Exception as e:
            print(f"Error occurs when processing {config['img_path']}")
            df.loc[index,'error'] = str(e)
            
        df.loc[index,'output_folder'] = config['output_folder']

    df.to_csv(os.path.join(config['output_folder'],
                           os.path.basename(csv_path) + f"_running_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"), index = False)    


def run_batch_adaptive_seed(file_path):
    _, extension = os.path.splitext(file_path)
    print(f"processing config the file {file_path}")
    if extension == '.yaml':
        with open(file_path, 'r') as file:
            yaml_config = yaml.safe_load(file)

    print("Config for pipeline")
    for key, value in yaml_config.items():
        print(f"\t{key}: {value}")

    csv_path = yaml_config['csv_path']
    df = pd.read_csv(csv_path)
    sprout_core.check_tiff_files(df['img_path'])

    for index, row in df.iterrows():
        ## Initial the config and optional parameters for each row
        yaml_config.pop("csv_path", None)
        config = config_core.merge_row_and_yaml_no_conflict(dict(row), yaml_config)
        optional_params = config_core.validate_input_yaml(config, config_core.input_val_make_adaptive_seed)
        
        
        try:
            sub_folder = os.path.basename(input_path)
            
            if isinstance(config['thresholds'], int):
                seed_merging_mode = "ERO"
            elif isinstance(config['thresholds'], list):
                if all(isinstance(t, int) for t in config['thresholds']):
                    if len(config['thresholds']) == 1:
                        seed_merging_mode = "ERO"
                        config['thresholds'] = config['thresholds'][0]
                        if optional_params['upper_thresholds'] is not None:
                            optional_params['upper_thresholds'] = optional_params['upper_thresholds'][0]
                        
                    elif len(config['thresholds']) > 1:
                        seed_merging_mode = "THRE"
                else:
                    raise ValueError("'thresholds' must be an int or a list of int(s).")
                    
            if seed_merging_mode == "THRE":
        
                print("Running make_adaptive_seed_thre")
                seed ,ori_combine_ids_map , output_dict=make_adaptive_seed.make_adaptive_seed_thre(                           
                                   thresholds=config['thresholds'],
                                    output_folder=config['output_folder'],
                                    n_iters=config['ero_iters'], 
                                    segments= config['segments'],
                                    
                                    num_threads = config['num_threads'],
                                    
                                    img_path = config['img_path'],
                                    boundary_path = optional_params['boundary_path'],    
                                    
                                    background = optional_params["background"],
                                    sort = optional_params["sort"],
                                    
                                    name_prefix = optional_params["name_prefix"],
                                    
                                    no_split_limit =optional_params["no_split_limit"],
                                    min_size=optional_params["min_size"],
                                    min_split_prop = optional_params["min_split_prop"],
                                    min_split_sum_prop = optional_params["min_split_sum_prop"],
                                    
                                    save_every_iter = optional_params["save_every_iter"],
                                    save_merged_every_iter = optional_params["save_merged_every_iter"],
                                                                        
                                    init_segments = optional_params["init_segments"],
                                    footprint = optional_params["footprints"],
                                    
                                    upper_thresholds = optional_params["upper_thresholds"],
                                    split_size_limit= optional_params["split_size_limit"],
                                    split_convex_hull_limit = optional_params["split_convex_hull_limit"],
                                    
                                    sub_folder = sub_folder   
                            
                            )
            
            elif seed_merging_mode=="ERO":
                print("Running make_seeds_merged")

                
                seed ,ori_combine_ids_map , output_dict=make_adaptive_seed.make_adaptive_seed(                           
                                       
                                    threshold=config['thresholds'],
                                    output_folder=config['output_folder'],
                                    n_iters=config['ero_iters'], 
                                    segments= config['segments'],
                                    num_threads = config['num_threads'],
                                    
                                    img_path = config['img_path'],
                                    boundary_path = optional_params['boundary_path'],                                            
                                    
                                    background = optional_params["background"],
                                    sort = optional_params["sort"],
                                    
                                    name_prefix = optional_params["name_prefix"],
                                    
                                    no_split_limit =optional_params["no_split_limit"],
                                    min_size=optional_params["min_size"],
                                    min_split_prop = optional_params["min_split_prop"],
                                    min_split_sum_prop = optional_params["min_split_sum_prop"],
                                    
                                    save_every_iter = optional_params["save_every_iter"],
                                    save_merged_every_iter = optional_params["save_merged_every_iter"],
                                    
                                    init_segments = optional_params["init_segments"],
                                    footprint = optional_params["footprints"],
                                    
                                    upper_threshold = optional_params["upper_thresholds"],
                                    split_size_limit= optional_params["split_size_limit"],
                                    split_convex_hull_limit = optional_params["split_convex_hull_limit"],
                                    
                                    sub_folder = sub_folder
                                        
                                        )            
        except Exception as e:
            print(f"Error occurs when processing {config['img_path']}")
            df.loc[index,'error'] = str(e)
            
        df.loc[index,'output_folder'] = config['output_folder']

    df.to_csv(os.path.join(config['output_folder'],
                           os.path.basename(csv_path) + f"_running_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"), index = False)    


if __name__ == "__main__":

    # Get the file path from the first command-line argument or use the default
    file_path = sys.argv[1] if len(sys.argv) > 1 else './batch_seeds.yaml'
    
    run_batch_seeds(file_path)
    
 
        
    