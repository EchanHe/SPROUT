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


    for index, row in df.iterrows():
        ## Initial the config and optional parameters for each row
        yaml_config.pop("csv_path", None)
        config = config_core.merge_row_and_yaml_no_conflict(dict(row), yaml_config)
        optional_params = config_core.validate_input_yaml(config, config_core.input_val_make_seeds_all)
        try:           
               
            output_dict = make_seeds.make_seeds(
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
        
                print("Running make_seeds_merged_by_thres_mp")
                seed ,ori_combine_ids_map , output_dict=make_adaptive_seed.make_seeds_merged_by_thres_mp(                           
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

                
                seed ,ori_combine_ids_map , output_dict=make_adaptive_seed.make_seeds_merged_mp(                           
                                       
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
    file_path = sys.argv[1] if len(sys.argv) > 1 else './PipelineSeed_all.yaml'
    
    run_batch_seeds(file_path)
    
    _, extension = os.path.splitext(file_path)
    print(f"processing config the file {file_path}")
    if extension == '.yaml':
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            
        # load_config_yaml(config)
        config_core.check_required_keys(config,
                                    config_core.pipeline_seed_required_keys)        

    for key, value in config.items():
        print(f"\t{key}: {value}")

    df = pd.read_csv(config['csv_path'])
    
    seed_mode = config['seed_mode']
    csv_path = config['csv_path']
    num_threads = config['num_threads']
    
    if seed_mode == "original" or seed_mode == "merge":
    
        either_keys = ["ero_iters", "segments",
                       "output_folder", "name_prefix",
                       "footprints", "seed_threshold"]
    else:
        either_keys = ["ero_iters", "segments",
                       "output_folder", "name_prefix", "seed_threshold"]
    
    
  
    config_core.check_csv_required_keys(df,
                                        config_core.csv_required_keys)  
    
    either_keys_info = config_core.check_either_csv_yaml_keys(df,
                                                              config,
                                                              either_keys)
    
    # Set the name_prefix
    
    
    print("Finish reading yaml and csv")  
    print(f"CSV: {csv_path}")
    print(f"seed_mode: {seed_mode}")
    print(f"num_threads: {num_threads}\n" )
        
        
    
    for index, row in df.iterrows():
        
        input_path = row['img_path']
        img = tifffile.imread(input_path)
        
        if "boundary_path" in df.columns and (not pd.isna(row['boundary_path'])):
            boundary_path = row['boundary_path']
            # boundary = tifffile.imread(row['boundary_path'])
            # boundary = sprout_core.check_and_cast_boundary(boundary)
        else:
            boundary_path = None
    
        if "output_folder" in either_keys_info["keys_in_df"]:
            output_folder = row['output_folder']    
        if "ero_iters" in either_keys_info["keys_in_df"]:
            ero_iters = row['ero_iters']
        if "segments" in either_keys_info["keys_in_df"]:
            segments = row['segments']
        if "name_prefix" in either_keys_info["keys_in_df"]:    
            name_prefix = row['name_prefix']
        if "footprints" in either_keys_info["keys_in_df"]:    
            footprints = row['footprints']
        if "seed_threshold" in either_keys_info["keys_in_df"]:  
            seed_threshold = eval(row['seed_threshold']) if isinstance(row['seed_threshold'], str) else row['seed_threshold'] 

 
        # Assign the optional parameters per row
        optional_params = sprout_core.assign_config_values_pipeline(config,row,
                                                  sprout_core.optional_params_default_seeds) 
        
        output_names = f"{name_prefix}_{os.path.splitext(os.path.basename(input_path))[0]}"
        
        
        # values_to_print = {
        #     "Segmentation Path": row['seg_path'],
        #     "Boundary Path": row['boundary_path'] if "boundary_path" in df.columns and not pd.isna(row['boundary_path']) else None,
        #     "Erosion Iterations": ero_iters,
        #     "Number of Segments": segments,
        #     "footprints": output_folder,
        #     "Save Interval": save_interval,
        # }
        # print(f"Growing on: {row['img_path']}")
        # for key, value in values_to_print.items():
        #     print(f"  {key}: {value}")
        
        try:
            if seed_mode == "original":
                if type(footprints) is str:
                    footprints = [footprints]   
                
                output_dict = make_seeds.make_seeds(
                                        img_path= input_path,
                                        boundary_path=boundary_path,
                                        
                                        output_folder = output_folder,
                                        ero_iters = ero_iters,
                                        thresholds = seed_threshold,
                                        segments = segments,
                                        name_prefix = output_names,
                                        num_threads = num_threads,
                                        input_footprints = footprints,
                                        
                                        upper_thresholds = optional_params['upper_thresholds']
                                        )
            elif seed_mode == "all":
                output_dict = make_seeds.make_seeds(
                                img_path= input_path,
                                boundary_path=boundary_path,
                                
                                output_folder = output_folder,
                                ero_iters = ero_iters,
                                thresholds = seed_threshold,
                                segments = segments,
                                name_prefix = output_names,
                                num_threads = num_threads,
                                upper_thresholds = optional_params['upper_thresholds']
                                )
            
            elif seed_mode == "merge":
                sub_folder = os.path.basename(input_path)
                if type(footprints) is list:
                    footprints = footprints [0]
                
                if isinstance(seed_threshold,list) and len(seed_threshold)!=1:
            
                    print("Running make_seeds_merged_by_thres_mp")
                    seed ,ori_combine_ids_map , output_dict=make_adaptive_seed.make_seeds_merged_by_thres_mp(                           
                                        img_path= input_path,
                                        thresholds= seed_threshold,
                                        output_folder=output_folder,
                                        boundary_path = boundary_path,
                                        n_iters = ero_iters,
                                        segments = segments,
                                        num_threads = num_threads,
                                        
                                        sort = optional_params['sort'],
                                        background = optional_params['background'],
                                        
                                        no_split_limit = optional_params['no_split_limit'],
                                        min_size= optional_params['min_size'],
                                        min_split_prop = optional_params['min_split_prop'],
                                        min_split_sum_prop = optional_params['min_split_sum_prop'],

                                        save_every_iter = optional_params["save_every_iter"],
                                        save_merged_every_iter = optional_params["save_merged_every_iter"],
                                        name_prefix = output_names,
                                        init_segments = optional_params["init_segments"],
                                        footprint= footprints,
                                        
                                        upper_thresholds = optional_params["upper_thresholds"],
                                        split_size_limit = optional_params["split_size_limit"] ,
                                        split_convex_hull_limit = optional_params["split_convex_hull_limit"] ,
                                        sub_folder=sub_folder
                                
                                )
                
                else:
                    if isinstance(seed_threshold,list) and len(seed_threshold)==1:
                        seed_threshold = seed_threshold[0]
                    if optional_params["upper_thresholds"] is not None:
                        if isinstance(seed_threshold,list) and len(seed_threshold)==1:
                        # assert len(optional_params["upper_thresholds"])==1, "Upper threshold should have 1 element"
                            upper_threshold = optional_params["upper_thresholds"][0]
                        else:
                            upper_threshold = optional_params["upper_thresholds"]
                    else:
                        upper_threshold = None
                    print("Running make_seeds_merged")

                    
                    seed ,ori_combine_ids_map , output_dict=make_adaptive_seed.make_seeds_merged_mp(                           
                                            img_path= input_path,
                                            threshold= seed_threshold,
                                            output_folder=output_folder,
                                            boundary_path = boundary_path,
                                            n_iters = ero_iters,
                                            segments = segments,
                                            num_threads = num_threads, 
                                            
                                            background = optional_params["background"],
                                            sort = optional_params["sort"],
                                                                                       
                                            no_split_limit =optional_params["no_split_limit"],
                                            min_size=optional_params["min_size"],
                                            min_split_prop = optional_params["min_split_prop"],
                                            min_split_sum_prop = optional_params["min_split_sum_prop"],

                                            save_every_iter = optional_params["save_every_iter"],
                                            save_merged_every_iter = optional_params["save_merged_every_iter"],
                                            name_prefix = output_names,
                                            init_segments = optional_params["init_segments"],
                                            footprint= footprints,
                                            
                                            upper_threshold = upper_threshold,
                                            split_size_limit = optional_params["split_size_limit"] ,
                                            split_convex_hull_limit = optional_params["split_convex_hull_limit"],
                                            
                                            sub_folder=sub_folder
                                            )
                    
        except Exception as e:
            df.loc[index,'error'] = str(e)
        df.loc[index,'output_folder'] = output_folder
        df.loc[index,'name_prefix'] = name_prefix

    df.to_csv(os.path.join(output_folder,
                           os.path.basename(csv_path) + f"_running_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"), index = False)    


        
    