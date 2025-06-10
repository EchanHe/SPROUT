import make_grow

import yaml
import os,sys
import glob
import pandas as pd

import sprout_core.config_core as config_core
import sprout_core.sprout_core as sprout_core


def load_config_yaml(config, parent_key=''):
    for key, value in config.items():
        if isinstance(value, dict):
            load_config_yaml(value, parent_key='')
        else:
            globals()[parent_key + key] = value
            


def run_batch_grow(file_path):
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
        optional_params = config_core.validate_input_yaml(config, config_core.input_val_make_grow)    

        try:
            output = make_grow.grow_mp(
                img_path = config['img_path'] ,
                seg_path = config['seg_path'],
                boundary_path = optional_params['boundary_path'],
                output_folder = config['output_folder'],

                workspace = None,               

                dilate_iters = config['dilate_iters'],
                thresholds = config['thresholds'],
                upper_thresholds = optional_params["upper_thresholds"],
                num_threads = config['num_threads'],
                
                save_interval = config['save_interval'],  
                touch_rule = config['touch_rule'],             
    
                grow_to_end = optional_params["grow_to_end"],
                to_grow_ids = optional_params["to_grow_ids"],
                
                final_grow_output_folder = config['output_folder'],
                name_prefix =  optional_params["name_prefix"],
                simple_naming =  optional_params["simple_naming"],    

                is_sort = optional_params['is_sort'],
                min_diff = optional_params['min_diff'],
                tolerate_iters = optional_params['tolerate_iters'],

   
                # For mesh making
                is_make_meshes = optional_params['is_make_meshes'],
                downsample_scale = optional_params['downsample_scale'],
                step_size  = optional_params['step_size']
                
            )

            df.loc[index,'final_output_path'] = output['final_output_path']
            df.loc[index,'output_folder'] = output['output_folder']
        
        except Exception as e:
            print(f"Error occurs when growing on {config['img_path']}")
            df.loc[index,'error'] = str(e)
    
    df.to_csv(csv_path + "_running_results.csv", index = False)



if __name__ == "__main__":
    
    # Get the file path from the first command-line argument or use the default
    file_path = sys.argv[1] if len(sys.argv) > 1 else './PipelineGrow.yaml'
    
    run_batch_grow(file_path)
    _, extension = os.path.splitext(file_path)
    print(f"processing config the file {file_path}")
    if extension == '.yaml':
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)         
            
            
        load_config_yaml(config)
        
    df = pd.read_csv(csv_path)

    sprout_core.check_tiff_files(df['img_path'])
    sprout_core.check_tiff_files(df['seg_path'])

    pipeline_grow_required_keys =["csv_path",
                                  "num_threads",
                                  "touch_rule"
                                  ]

    either_keys = ["dilate_iters", "grow_thresholds",
                "output_folder", "save_interval"]
    
    csv_required_keys = ['img_path', "seg_path"]
    
    config_core.check_required_keys(config,pipeline_grow_required_keys)          
    config_core.check_csv_required_keys(df,csv_required_keys)  
    
    either_keys_info = config_core.check_either_csv_yaml_keys(df,
                                                              config,
                                                              either_keys)
        
    print("Finish reading yaml and csv")  
    print(f"CSV: {csv_path}")
    print(f"touch_rule: {touch_rule}")
    print(f"num_threads: {num_threads}\n" )
                 

    for index, row in df.iterrows():
        
        input_path = row['img_path']
        # img = tifffile.imread(input_path)
        
        seg_path = row['seg_path']
        # s = tifffile.imread(seg_path)
    
        if "boundary_path" in df.columns and (not pd.isna(row['boundary_path'])):
            boundary_path = row['boundary_path']
            
        else:
            boundary_path = None
    
        
        if "dilate_iters" in either_keys_info["keys_in_df"]:
            dilate_iters = eval(row['dilate_iters'])
        if "grow_thresholds" in either_keys_info["keys_in_df"]:
            grow_thresholds = eval(row['grow_thresholds'])
        if "output_folder" in either_keys_info["keys_in_df"]:    
            output_folder = row['output_folder']
        if "save_interval" in either_keys_info["keys_in_df"]:    
            save_interval = row['save_interval']


        # Assign the optional parameters per row
        optional_params = sprout_core.assign_config_values_pipeline(config,row,
                                                  sprout_core.optional_params_default_grow)
        

        # Check if both are int
        if isinstance(dilate_iters, int) and isinstance(grow_thresholds, int):
            grow_thresholds = [grow_thresholds]
            dilate_iters = [dilate_iters]
        elif (
            isinstance(dilate_iters, list) and all(isinstance(i, int) for i in dilate_iters) and
            isinstance(grow_thresholds, list) and all(isinstance(i, int) for i in grow_thresholds)
        ):
             # Check if they have the same length
            if len(dilate_iters) == len(grow_thresholds):
                pass
                # print("Both are lists of integers and have the same length.")
            else:
                pass
                raise Exception("grow_thresholds and dilate_iters have different lengths")
                # print("Both are lists of integers but have different lengths.")
            # print("Both are not lists.")
        else:
            # Other cases
            raise Exception("grow_thresholds or dilate_iters are incorrectly set")


        name = os.path.splitext(os.path.basename(input_path))[0]
        # output_sub_folder = os.path.join(output_folder , name)

        try:
            output = make_grow.grow_mp(
                img_path = input_path,
                seg_path = seg_path,
                dilate_iters = dilate_iters,
                thresholds = grow_thresholds,
                save_interval = save_interval,  
                touch_rule = touch_rule, 
                num_threads = num_threads,
                workspace = None,
                boundary_path = boundary_path,
                output_folder = output_folder,
                
                grow_to_end = optional_params["grow_to_end"],
                to_grow_ids = optional_params["to_grow_ids"],
                
                final_grow_output_folder = output_folder,
                name_prefix =  optional_params["name_prefix"],
                simple_naming =  optional_params["simple_naming"],    

                is_sort = optional_params['is_sort'],
                min_diff = optional_params['min_diff'],
                tolerate_iters = optional_params['tolerate_iters'],

                # For mesh making
                is_make_meshes = optional_params['is_make_meshes'],
                downsample_scale = optional_params['downsample_scale'],
                step_size  = optional_params['step_size'],
                
                upper_thresholds = optional_params["upper_thresholds"]
            )

            df.loc[index,'final_output_path'] = output['final_output_path']
            df.loc[index,'output_folder'] = output['output_folder']
        
        except Exception as e:
            df.loc[index,'error'] = str(e)
    
    df.to_csv(csv_path + "_running_results.csv", index = False)


        # if is_make_mesh:
        #     mesh_folder = grow_dict['output_folder']
        #     tif_files = glob.glob(os.path.join(mesh_folder, '*.tif'))
        #     for tif_file in tif_files:
        #         make_mesh.make_mesh_for_tiff(tif_file,mesh_folder,
        #                             num_threads,no_zero = True,
        #                             colormap = "color10")

        # for grow_dict in grow_dict_list:
        #     vis_lib.plot_grow(pd.read_csv(grow_dict['log_path']),
        #         grow_dict['log_path'] +".png")


            
    
        
#         # TODO , check is it possible to making plot multi-thread using plt
#         grow_dict_list.append(grow_dict)

#         # if is_make_mesh:
#         #     mesh_folder = grow_dict['output_folder']
#         #     tif_files = glob.glob(os.path.join(mesh_folder, '*.tif'))
#         #     for tif_file in tif_files:
#         #         make_mesh.make_mesh_for_tiff(tif_file,mesh_folder,
#         #                             num_threads,no_zero = True,
#         #                             colormap = "color10")

#     for grow_dict in grow_dict_list:
#         vis_lib.plot_grow(pd.read_csv(grow_dict['log_path']),
#             grow_dict['log_path'] +".png")