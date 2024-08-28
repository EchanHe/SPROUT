import tifffile

import numpy as np
import threading
import os, sys
from datetime import datetime
import itertools

lock = threading.Lock()

import configparser
import suture_morph.suture_morpho as suture_morpho
import make_seeds


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


def write_json(filename, args_dict):
    
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


if __name__ == "__main__":
    file_path = 'make_seeds_all.yaml'
    _, extension = os.path.splitext(file_path)
    print(f"processing config he file {file_path}")
    if extension == '.json':
        
        load_config_json(file_path)
    elif extension == '.yaml':
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
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
    
    
    ###################

    output_seed_folder =os.path.join(workspace, output_seed_folder)
    
    file_path = os.path.join(workspace, file_name)
    


    footprint_list = [
        ["ball"] * ero_iters,
        ["ball_XY"] * ero_iters,
        ["ball_YZ"] * ero_iters,
        ["ball_XZ"] * ero_iters

    ]

    output_seed_sub_folders = [
        "seeds_ball",
        "seeds_XY",
        "seeds_YZ",
        "seeds_XZ"
    ]

    
    for footprints, output_seed_sub_folder in zip(footprint_list,output_seed_sub_folders):


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
                """)
        


        threshold_ero_iter_pairs = list(itertools.product(target_thresholds, [ero_iters]))

        volume = tifffile.imread(file_path)
        volume = volume.astype("uint8")
        
        

        
    
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