import make_seeds
import make_seeds_all
import make_mesh
import yaml
import os,sys
import glob
import tifffile


import make_seeds_merged

import pandas as pd
import sprout_core.vis_lib as vis_lib
import sprout_core.sprout_core as sprout_core


def load_config_yaml(config, parent_key=''):
    for key, value in config.items():
        if isinstance(value, dict):
            load_config_yaml(value, parent_key='')
        else:
            globals()[parent_key + key] = value


pipeline_seed_required_keys = [
    
    "csv_path",
    "seed_mode",
    
    
    ## Must have in yaml
    "num_threads",
    # "ero_iters",
    # "segments",
]


if __name__ == "__main__":

    # Get the file path from the first command-line argument or use the default
    file_path = sys.argv[1] if len(sys.argv) > 1 else './PipelineSeed.yaml'
    
    _, extension = os.path.splitext(file_path)
    print(f"processing config the file {file_path}")
    if extension == '.yaml':
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            

            background = config.get('background', 0)
            sort = config.get('sort', True)
            
            no_split_limit = config.get('no_split_limit', 3)
            min_size = config.get('min_size', 5)
            min_split_prop = config.get('min_split_prop', 0.01)
            min_split_sum_prop = config.get('min_split_sum_prop', 0)
            
            save_every_iter = config.get('save_every_iter', False)
            save_merged_every_iter = config.get('save_merged_every_iter', False)
           
            init_segments = config.get('init_segments', None)
            
            workspace = config.get("workspace","")
        load_config_yaml(config)
    

    for key, value in config.items():
        print(f"\t{key}: {value}")

    df = pd.read_csv(csv_path)
    
    if seed_mode == "original" or seed_mode == "merge":
    
        either_keys = ["ero_iters", "segments",
                       "output_folder", "name_prefix",
                       "footprints", "seed_threshold"]
    else:
        either_keys = ["ero_iters", "segments",
                       "output_folder", "name_prefix", "seed_threshold"]
    csv_required_keys = ['img_path']
    
    sprout_core.check_required_keys(config,pipeline_seed_required_keys)          
    sprout_core.check_csv_required_keys(df,csv_required_keys)  
    
    either_keys_info = sprout_core.check_either_csv_yaml_keys(df,
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
                
                output_dict = make_seeds_all.for_pipeline_wrapper(
                                        img_path= input_path,
                                        boundary_path=boundary_path,
                                        
                                        output_folder = output_folder,
                                        ero_iters = ero_iters,
                                        target_thresholds = seed_threshold,
                                        segments = segments,
                                        name_prefix = output_names,
                                        num_threads = num_threads,
                                        footprints = footprints
                                        )
            elif seed_mode == "all":
                output_dict = make_seeds_all.for_pipeline_wrapper(
                                img_path= input_path,
                                boundary_path=boundary_path,
                                
                                output_folder = output_folder,
                                ero_iters = ero_iters,
                                target_thresholds = seed_threshold,
                                segments = segments,
                                name_prefix = output_names,
                                num_threads = num_threads,
                                )
            
            elif seed_mode == "merge":
                if type(footprints) is list:
                    footprints = footprints [0]
                
                if isinstance(seed_threshold,list) and len(seed_threshold)!=1:
            
                    print("Running make_seeds_merged_by_thres_mp")
                    seed ,ori_combine_ids_map , output_dict=make_seeds_merged.make_seeds_merged_by_thres_path_wrapper(                           
                                img_path= input_path,
                                thresholds= seed_threshold,
                                output_folder=output_folder,
                                boundary_path = boundary_path,
                                n_iters = ero_iters,
                                segments = segments,
                                num_threads = num_threads,
                                no_split_limit =no_split_limit,
                                min_size= min_size,
                                min_split_prop = min_split_prop,
                                min_split_sum_prop = min_split_sum_prop,
                                sort = sort,
                                background = background,
                                save_every_iter = save_every_iter,
                                save_merged_every_iter = save_merged_every_iter ,
                                name_prefix = output_names,
                                init_segments = init_segments,
                                footprint= footprints
                                )
                
                else:
                    if isinstance(seed_threshold,list) and len(seed_threshold)==1:
                        seed_threshold = seed_threshold[0]
                    print("Running make_seeds_merged")
                    seed ,ori_combine_ids_map , output_dict=make_seeds_merged.make_seeds_merged_path_wrapper(                           
                                            img_path= input_path,
                                            threshold= seed_threshold,
                                            output_folder=output_folder,
                                            boundary_path = boundary_path,
                                            n_iters = ero_iters,
                                            segments = segments,
                                            num_threads = num_threads,                                           
                                            no_split_limit =no_split_limit,
                                            min_size= min_size,
                                            min_split_prop = min_split_prop,
                                            min_split_sum_prop = min_split_sum_prop,
                                            sort = sort,
                                            background = background,
                                            save_every_iter = save_every_iter,
                                            save_merged_every_iter = save_merged_every_iter ,
                                            name_prefix = output_names,
                                            init_segments = init_segments,
                                            footprint= footprints
                                            )
        except Exception as e:
            df.loc[index,'error'] = str(e)
        df.loc[index,'output_folder'] = output_folder
        df.loc[index,'name_prefix'] = name_prefix

    df.to_csv(csv_path + "_running_results.csv", index = False)    


        
    
    # for csv_required_key in csv_required_keys:
    # if "file_path" not in df.columns:
    #     raise Exception("'file_path' must be present as a column in the CSV file.")
    # else:
    #     print("'file_path' is present in the CSV file.")
    
    # try:
    #     yaml_data, df
    # except Exception as e:
    #     print(e)

#     
    
#     #TODO check if This df fits the requirements
    
#     #a check to see if all files exist
#     sprout_core.check_tiff_files(df['file_name'])

    
#     output_dict_list = []
    

#     for idx, row in df.iterrows():
#         file_name = row["file_name"]
#         file_name_no_ext = os.path.splitext(os.path.basename(file_name))[0]
        
#         target_thresholds = eval(row['target_thresholds'])
        
#         if 'output_seed_folder' in df.columns:
#             output_seed_folder = row['output_seed_folder']
#         else:
#             output_seed_folder = os.path.join(output_seed_root_dir, file_name_no_ext)
        
#         print(file_name, target_thresholds)

#         output_dict = make_seeds_all.main(workspace=workspace,
#                                       file_name= file_name,
#                                       output_log_file = output_log_file,
#                                       output_seed_folder = output_seed_folder,
#                                       ero_iters = ero_iters,
#                                       target_thresholds = target_thresholds,
#                                       segments = segments
#                                       )
        
#         output_dict_list.append(output_dict)
# #         # make_seeds_all.plot(output_dict , os.path.join(output_seed_folder,"log.png"))
        
#     for idx, output_dict in enumerate(output_dict_list):
#         merged_img_name = os.path.basename(output_dict["output_seed_sub_folders"])
#         make_seeds_all.plot(output_dict , os.path.join(output_seed_root_dir,f"{merged_img_name}.png"))

# # output = make_seeds.main(workspace= workspace,
# #                 file_name= file_name,
# #                 output_log_file = output_log_file,
# #                 output_seed_folder = output_seed_folder,
# #                 num_threads = num_threads,
# #                 ero_iters = ero_iters,
# #                 target_thresholds = target_thresholds,
# #                 segments = segments,
# #                 footprints = footprints)

# # output_seed_folder = output[0]

# # tif_files = glob.glob(os.path.join(output_seed_folder, '*.tif'))

# # for tif_file in tif_files:
# #     make_mesh.make_mesh_for_tiff(tif_file,output_seed_folder,
# #                         num_threads,no_zero = True,
# #                         colormap = "color10")