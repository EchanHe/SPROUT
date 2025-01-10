import make_grow_result
import make_mesh
import yaml
import os,sys
import glob
import pandas as pd
import sprout_core.vis_lib as vis_lib
import sprout_core.sprout_core as sprout_core
import tifffile

def load_config_yaml(config, parent_key=''):
    for key, value in config.items():
        if isinstance(value, dict):
            load_config_yaml(value, parent_key='')
        else:
            globals()[parent_key + key] = value
            


if __name__ == "__main__":
    
    # Get the file path from the first command-line argument or use the default
    file_path = sys.argv[1] if len(sys.argv) > 1 else './PipelineGrow.yaml'
    

    _, extension = os.path.splitext(file_path)
    print(f"processing config he file {file_path}")
    if extension == '.yaml':
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            workspace = config.get("workspace","")
            
            to_grow_ids = config.get("to_grow_ids" , None)
            name_prefix = config.get("name_prefix" , "final_grow")
            grow_to_end = config.get("grow_to_end" , False)
            simple_naming = config.get("simple_naming",True)
            
            is_sort = config.get('is_sort', True) 
            min_diff = config.get('min_diff', 50) 
            tolerate_iters = config.get('tolerate_iters', 3) 
            
            # For mesh making
            is_make_meshes = config.get('is_make_meshes', False) 
            downsample_scale = config.get('downsample_scale', 10) 
            step_size  = config.get('step_size', 1)     
            
            
            
            
        load_config_yaml(config)
        
    df = pd.read_csv(csv_path)

    pipeline_seed_required_keys =["csv_path",
                                  "num_threads",
                                  "touch_rule"
                                  ]

    either_keys = ["dilate_iters", "grow_thresholds",
                "output_folder", "save_interval"]
    
    csv_required_keys = ['img_path', "seg_path"]
    
    sprout_core.check_required_keys(config,pipeline_seed_required_keys)          
    sprout_core.check_csv_required_keys(df,csv_required_keys)  
    
    either_keys_info = sprout_core.check_either_csv_yaml_keys(df,
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
        output_sub_folder = os.path.join(output_folder , name)

        try:
            output = make_grow_result.grow_mp(
                img_path = input_path,
                seg_path = seg_path,
                dilate_iters = dilate_iters,
                thresholds = grow_thresholds,
                save_interval = save_interval,  
                touch_rule = touch_rule, 
                num_threads = num_threads,
                workspace = None,
                boundary_path = boundary_path,
                output_folder = output_sub_folder,
                
                grow_to_end = grow_to_end,
                to_grow_ids = to_grow_ids,    
                
                final_grow_output_folder = output_folder,
                name_prefix = name_prefix,
                simple_naming = simple_naming ,     

                is_sort = is_sort,
                min_diff = min_diff,
                tolerate_iters = tolerate_iters,

                # For mesh making
                is_make_meshes = is_make_meshes,
                downsample_scale = downsample_scale,
                step_size  = step_size

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


            
# if __name__ == "__main__":
    
#     file_path = 'PipelineGrow.yaml'
#     _, extension = os.path.splitext(file_path)
#     print(f"processing config he file {file_path}")
#     if extension == '.yaml':
#         with open(file_path, 'r') as file:
#             config = yaml.safe_load(file)
#             workspace = config.get("workspace","")
#             to_grow_ids = config.get("to_grow_ids" , None)
#         load_config_yaml(config)
    
    
    
#     df = pd.read_csv(csv_path)

#     #TODO check if This df fits the requirements
    
#     #a check to see if all files exist
#     sprout_core.check_tiff_files(df['img_path'])
#     sprout_core.check_tiff_files(df['seg_path'])
    
    
#     grow_dict_list = []
    
#     # Or the multi thread version:
#     for idx, row in df.iterrows():
      
#         img_path = row["img_path"]
#         img_name = os.path.splitext(os.path.basename(img_path))[0]
        
#         seg_path = row["seg_path"]  
#         seg_name = os.path.splitext(os.path.basename(seg_path))[0]
        
        
#         if "boundary_path" in df.columns and (not pd.isna(row['boundary_path'])):
#             boundary = tifffile.imread(row['boundary_path'])
#         else:
#             boundary = None
        
#         grow_name = img_name + "_" + seg_name
        
        
#         if 'output_folder' in df.columns:
#             output_folder = row['output_folder']
#         else:
#             output_folder = os.path.join(output_root_dir, grow_name)
        
#         if 'save_interval' in df.columns:
#             save_interval = eval(row['save_interval'])

#         if 'thresholds' in df.columns:
#             thresholds = eval(row['thresholds'])
        
#         if 'dilate_iters' in df.columns:
#             dilate_iters = eval(row['dilate_iters'])
            
#         if 'to_grow_ids' in df.columns:
#             thresholds = eval(row['to_grow_ids'])
            
#         if 'touch_rule' in df.columns:
#             touch_rule = row['touch_rule']

        
#         grow_dict = make_grow_result.main(
#             dilate_iters = dilate_iters,
#             thresholds = thresholds,
#             save_interval = save_interval,  
#             touch_rule = touch_rule, 
            
#             workspace = workspace,
#             img_path = img_path,
#             seg_path = seg_path,
#             output_folder = output_folder,
#             to_grow_ids = to_grow_ids
#         )
        
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