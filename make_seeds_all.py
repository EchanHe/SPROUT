import tifffile

import numpy as np
import threading
import os, sys
from datetime import datetime
import itertools

import make_mesh
import glob

lock = threading.Lock()


import multiprocessing
max_threads = multiprocessing.cpu_count()

# import configparser
# import sprout_core.sprout_core as sprout_core
import make_seeds

import sprout_core.vis_lib as vis_lib
import json, yaml


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



pre_set_footprint_list = [
    ["ball"],
    ["ball_XY"],
    ["ball_YZ"] ,
    ["ball_XZ"]

]

output_seed_sub_folders = [
    "seeds_ball",
    "seeds_XY",
    "seeds_YZ",
    "seeds_XZ"
]

def main(**kwargs):
    
    
    # Input and Output
    workspace = kwargs.get('workspace', None)
    file_name = kwargs.get('file_name', None)
    output_log_file = kwargs.get('output_log_file', None) 
    output_seed_folder = kwargs.get('output_seed_folder', None) 
    
    # Seed generation related 
    ero_iters = kwargs.get('ero_iters', None)
    target_thresholds = kwargs.get('target_thresholds', None)  
    segments = kwargs.get('segments', None)  
    
    num_threads = kwargs.get('num_threads', None) 
    
    is_make_meshes = kwargs.get('is_make_meshes', False) 
    num_threads = kwargs.get('num_threads', None) 
    downsample_scale = kwargs.get('downsample_scale', 10) 
    step_size  = kwargs.get('step_size', 2) 

    if num_threads is None:
        num_threads = len(target_thresholds)

    if num_threads>=max_threads:
        num_threads = max_threads-1
    
    
    output_seed_folder =os.path.join(workspace, output_seed_folder)
    
    file_path = os.path.join(workspace, file_name)
    
    footprint_list = [footprint*ero_iters for footprint in pre_set_footprint_list]

    output_dict = {
        "output_seed_sub_folders":[],
        "output_log_files":[]
    }
    
    volume = tifffile.imread(file_path)

    
    for footprints, output_seed_sub_folder in zip(footprint_list,output_seed_sub_folders):

        # Init the folders and path for output files
        output_seed_sub_folder = os.path.join(output_seed_folder, output_seed_sub_folder)
        os.makedirs(output_seed_sub_folder , exist_ok=True)
        output_json_path = os.path.join(output_seed_sub_folder, output_log_file)

        start_time = datetime.now()
        print(f"""{start_time.strftime("%Y-%m-%d %H:%M:%S")}
        Making erosion seeds for 
            Img: {file_path}
            Threshold for Img {target_thresholds}
            Erode {ero_iters} iterations
            Keeping {segments} components
            Erosion footprints {footprints}
            Running in {num_threads} threads
                """)
        


        threshold_ero_iter_pairs = list(itertools.product(target_thresholds, [ero_iters]))



    
        sublists = [threshold_ero_iter_pairs[i::num_threads] for i in range(num_threads)]

        # Create a list to hold the threads
        threads = []


        # Start a new thread for each sublist
        for sublist in sublists:
           
            thread = threading.Thread(target=make_seeds.find_seed_by_ero_mp, args=(volume,sublist, segments,
                                                                        output_seed_sub_folder,output_json_path, footprints ))
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # print(f"All threads have completed. Log is saved at {output_json_path},seeds are saved at {output_seed_folder}")

        
        end_time = datetime.now()
        running_time = end_time - start_time
        total_seconds = running_time.total_seconds()
        minutes, _ = divmod(total_seconds, 60)
        print(f"Running time:{minutes}")
    
        output_dict["output_seed_sub_folders"].append(output_seed_sub_folder)
        output_dict["output_log_files"].append(output_json_path)
        
        # Make meshes  
        if is_make_meshes:  
            tif_files = glob.glob(os.path.join(output_seed_sub_folder, '*.tif'))

            for tif_file in tif_files:
                make_mesh.make_mesh_for_tiff(tif_file,output_seed_sub_folder,
                                    num_threads=num_threads,no_zero = True,
                                    colormap = "color10",
                                    downsample_scale=downsample_scale,
                                    step_size=step_size)
        
        
    return output_dict

def plot(output_dict, full_log_plot_path):
    output_log_files = output_dict["output_log_files"]
    output_seed_sub_folders = output_dict["output_seed_sub_folders"]
    
    plot_list = []
    for output_log_file, output_seed_sub_folder in zip(output_log_files,output_seed_sub_folders):
        
        with open(output_log_file, 'r') as config_file:
            json_data = json.load(config_file)
        
        vis_lib.plot_seeds_log_json(json_data, os.path.join(output_seed_sub_folder, "seeds.png"))
        
        # plot_data = vis_lib.seeds_json_to_plot_ready(output_log_file)
        # vis_lib.plot_seeds_log(plot_data, os.path.join(output_seed_sub_folder, "seeds.png"))
        
        plot_list.append(os.path.join(output_seed_sub_folder, "seeds.png"))

    
    vis_lib.merge_plots(plot_list, full_log_plot_path)

if __name__ == "__main__":
    file_path = 'make_seeds_all.yaml'
    _, extension = os.path.splitext(file_path)
    print(f"processing config he file {file_path}")
    if extension == '.json':
        
        load_config_json(file_path)
    elif extension == '.yaml':
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            num_threads = config.get('num_threads', None) 
        load_config_yaml(config)
        
    # ### To change your input params ####
    # GEN_WHOLE_MESH = True
    # workspace = r'C:\Users\Yichen\OneDrive\work\codes\nhm_bounti_pipeline\result\foram_james'
    # # file_path = os.path.join(workspace, 'input//procaviaH4981C_0001.tif.resampled_400_600.tif')
    # file_name = 'input/ai/final.20180802_VERSA_1905_ASB_OLK_st016_bl4_fo1_recon.tif'
    # # file_path = r"C:\Users\Yichen\OneDrive\work\codes\nhm_monai_suture_demo\data\bones_suture\procavia\procaviaH4981C_0001.tif.resampled.tif"
    # output_json_file = 'thre_ero_log.json' 
     
    # num_threads = 4
    # ### To change the thre ranges
    # # target_thresholds = list(range(3000, 4501, 100))
    # target_thresholds = [0]
    # ero_iters = [5]
    # 
    # segments = 25
    
    output_dict = main(workspace= workspace,
            file_name= file_name,
            output_log_file = output_log_file,
            output_seed_folder = output_seed_folder,
            num_threads = num_threads,
            ero_iters = ero_iters,
            target_thresholds = target_thresholds,
            segments = segments)
    

    
    # Make plot based on the seeds log json
    # Doing this after parallel/multi processing
    plot(output_dict, os.path.join(os.path.join(workspace, output_seed_folder, "full_log.png")))
                                                     

    # if make_mesh:
        
    #     output_seed_folders = output_dict["output_seed_folders"]

    #     for output_seed_folder in output_seed_folders：
        
    #         tif_files = glob.glob(os.path.join(output_seed_folder, '*.tif'))

    #         for tif_file in tif_files:
    #             make_mesh.make_mesh_for_tiff(tif_file,output_seed_folder,
    #                                 num_threads,no_zero = True,
    #                                 colormap = "color10")
    # ###################

    # output_seed_folder =os.path.join(workspace, output_seed_folder)
    
    # file_path = os.path.join(workspace, file_name)
    


    # footprint_list = [
    #     ["ball"] * ero_iters,
    #     ["ball_XY"] * ero_iters,
    #     ["ball_YZ"] * ero_iters,
    #     ["ball_XZ"] * ero_iters

    # ]

    # output_seed_sub_folders = [
    #     "seeds_ball",
    #     "seeds_XY",
    #     "seeds_YZ",
    #     "seeds_XZ"
    # ]

    
    # for footprints, output_seed_sub_folder in zip(footprint_list,output_seed_sub_folders):


    #     output_seed_sub_folder = os.path.join(output_seed_folder, output_seed_sub_folder)
    #     os.makedirs(output_seed_sub_folder , exist_ok=True)
    #     output_json_path = os.path.join(output_seed_sub_folder, output_log_file)

    #     start_time = datetime.now()
    #     print(f"""{start_time.strftime("%Y-%m-%d %H:%M:%S")}
    #     Making erosion seeds for 
    #         Img: {file_path}
    #         Threshold for Img {target_thresholds}
    #         Erode {ero_iters} iterations
    #         Keeping {segments} components
    #         Erosion footprints {footprints}
    #             """)
        


    #     threshold_ero_iter_pairs = list(itertools.product(target_thresholds, [ero_iters]))

    #     volume = tifffile.imread(file_path)
    #     volume = volume.astype("uint8")
        
        

        
    
    #     sublists = [threshold_ero_iter_pairs[i::num_threads] for i in range(num_threads)]

    #     # Create a list to hold the threads
    #     threads = []


    #     # Start a new thread for each sublist
    #     for sublist in sublists:
           
    #         thread = threading.Thread(target=make_seeds.find_seed_by_ero_mp, args=(volume,sublist, segments,
    #                                                                     output_seed_sub_folder,output_json_path, footprints ))
    #         threads.append(thread)
    #         thread.start()
            
    #     # Wait for all threads to complete
    #     for thread in threads:
    #         thread.join()

    #     # print(f"All threads have completed. Log is saved at {output_json_path},seeds are saved at {output_seed_folder}")

        
    #     end_time = datetime.now()
    #     running_time = end_time - start_time
    #     total_seconds = running_time.total_seconds()
    #     minutes, _ = divmod(total_seconds, 60)
    #     print(f"Running time:{minutes}")