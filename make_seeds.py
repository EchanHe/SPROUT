import tifffile

import numpy as np
import threading
import os, sys
from datetime import datetime
import itertools

lock = threading.Lock()

import configparser

# lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'sprout_core'))
# sys.path.insert(0, lib_path)

import sprout_core.sprout_core as sprout_core
# import sprout_core.load_config as load_config
# import sprout_core
# import load_config


import json, yaml


# Function to recursively create global variables from the config dictionary
def load_config_yaml(config, parent_key=''):
    """
    Recursively load configuration values from a YAML dictionary into global variables.

    Args:
        config (dict): Configuration dictionary.
        parent_key (str): Key prefix for nested configurations (default is '').
    """
    
    for key, value in config.items():
        if isinstance(value, dict):
            load_config_yaml(value, parent_key='')
        else:
            globals()[parent_key + key] = value
            




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

def find_seg_by_ero(volume, threshold , segments, input_ero_iters):
    """
    Deprecated
    Find and save segmentation by erosion for given thresholds and iterations.

    Args:
        volume (np.ndarray): 3D volume data.
        threshold (int): Threshold value for segmentation.
        segments (int): Number of segments to keep.
        input_ero_iters (list): List of erosion iteration values.
    """    


    for ero_iter in input_ero_iters:
        seed, result, log_dict = BounTI.find_seg_by_ero(volume, threshold , segments, ero_iter)
        
        seed = seed.astype('uint8')
        result = result.astype('uint8')
        
        output_file = os.path.join(output_seg_folder ,
                            f"seg_{ero_iter}iter_thre{threshold}.tif")
        
        
        seed_file = os.path.join(output_seed_folder , 
                            f"thre_ero_{ero_iter}iter_thre{threshold}.tif")
        
        tifffile.imwrite(seed_file, 
            seed)
        
        tifffile.imwrite(output_file, 
                result)
        
        log_dict['input_file'] = file_path
        log_dict['output_file'] = seed_file
        
        with lock:
            # filename = f'output/json/Bount_ori_run_log_{init_threshold}_{target_threshold}.json'
            write_json(output_json_path, log_dict)   
            
def find_seg_by_ero_v2(volume, input_threshold_ero_iter_pairs , segments):
    """
    Version 2 of find_seg_by_ero with support for multiple threshold and erosion pairs.

    Args:
        volume (np.ndarray): 3D volume data.
        input_threshold_ero_iter_pairs (list): List of (threshold, erosion) pairs.
        segments (int): Number of segments to keep.
    """
    
    for threshold_ero_iter_pair in input_threshold_ero_iter_pairs:
        threshold = threshold_ero_iter_pair[0]
        ero_iter = threshold_ero_iter_pair[1]
        
        seed, result,log_dict = BounTI.find_seg_by_ero(volume, threshold , segments, ero_iter)
        
        seed = seed.astype('uint8')
        result = result.astype('uint8')
        
        output_file = os.path.join(output_seg_folder ,
                    f"seg_{ero_iter}iter_thre{threshold}.tif")
        seed_file = os.path.join(output_seed_folder , 
                            f"thre_ero_{ero_iter}iter_thre{threshold}.tif")
        
        tifffile.imwrite(seed_file, 
            seed)
        
        tifffile.imwrite(output_file, 
            result)
        
        log_dict['input_file'] = file_path
        log_dict['output_seed_file'] = seed_file
        log_dict['output_result_file'] = seed_file
        
        with lock:
            # filename = f'output/json/Bount_ori_run_log_{init_threshold}_{target_threshold}.json'
            write_json(output_json_path, log_dict)   

def find_seed_by_ero_mp(volume, input_threshold_ero_iter_pairs , segments , 
                        output_seed_folder, output_json_path,  footprints = 'default',):
    """
    Seed generation using erosion and thresholds for multi-threading.

    Args:
        volume (np.ndarray): 3D volume data.
        input_threshold_ero_iter_pairs (list): List of (threshold, erosion) pairs.
        segments (int): Number of segments to keep.
        output_seed_folder (str): Directory to save seed outputs.
        output_json_path (str): Path to save the log JSON.
        footprints (str): Footprint type for erosion (default is 'default').
    """
    

    
    for threshold_ero_iter_pair in input_threshold_ero_iter_pairs:
        threshold = threshold_ero_iter_pair[0]
        ero_iter = threshold_ero_iter_pair[1]
        
        if footprints == 'ball' or footprints == 'default':
            footprints = ['ball'] * ero_iter
        
        print(f"Saving every seeds for thresholds {threshold} for {ero_iter} erosion")
        
        log_dict = sprout_core.find_seed_by_ero_custom(volume, threshold , segments, ero_iter,
                                          output_dir =output_seed_folder,
                                          footprints = footprints)
        
        # seed = seed.astype('uint8')
        # seed_file = os.path.join(output_seed_folder , 
        #                     f"thre_ero_{ero_iter}iter_thre{threshold}.tif")
        
        # tifffile.imwrite(seed_file, 
        #     seed)
        
        # log_dict['input_file'] = file_path
        
        with lock:
            # filename = f'output/json/Bount_ori_run_log_{init_threshold}_{target_threshold}.json'
            write_json(output_json_path, log_dict)   


def find_seed_by_ero_mp_v2(volume, input_threshold_ero_iter_pairs , segments , 
                        output_seed_folder, output_json_path,  footprints = 'default',):
    """Version 2Seed generation using erosion and thresholds for multi-threading.

    Args:
        volume (_type_): _description_
        input_threshold_ero_iter_pairs (_type_): _description_
        segments (_type_): _description_
    """
    

    
    for threshold_ero_iter_pair in input_threshold_ero_iter_pairs:
        threshold = threshold_ero_iter_pair[0]
        ero_iter = threshold_ero_iter_pair[1]
        
        if footprints == 'ball' or footprints == 'default':
            footprints = ['ball'] * ero_iter
        
        print(f"Saving every seeds for thresholds {threshold} for {ero_iter} erosion")
        
        volume_label = volume > threshold

        
        for ero_i in ero_iter:
            seed, ccomp_sizes = sprout_core.ero_one_iter(volume_label,
                                                        segments, 
                                                        footprint = footprints[ero_i])
            
            seed = seed.astype('uint8')
            
            seed_file = os.path.join(output_seed_folder , 
                            f"seed_ero_{ero_i}_thre{threshold}_segs_{segments}.tif")
            
            tifffile.imwrite(seed_file, seed,
                    compression ='zlib')
            # log_dict = sprout_core.find_seed_by_ero_custom(volume, threshold , segments, ero_iter,
            #                                 output_dir =output_seed_folder,
            #                                 footprints = footprints)
        

        
        # seed = seed.astype('uint8')
        # seed_file = os.path.join(output_seed_folder , 
        #                     f"thre_ero_{ero_iter}iter_thre{threshold}.tif")
        
        # tifffile.imwrite(seed_file, 
        #     seed)
        
        # log_dict['input_file'] = file_path
        
        with lock:
            # filename = f'output/json/Bount_ori_run_log_{init_threshold}_{target_threshold}.json'
            write_json(output_json_path, log_dict)   

def main(**kwargs):
    """
    Main function to orchestrate erosion-based seed generation with multi-threading.

    Args:
        kwargs: Dictionary containing parameters for seed generation.

    Returns:
        tuple: Output folder and log file paths.
    """    
    ero_iters = kwargs.get('ero_iters', None)
    target_thresholds = kwargs.get('target_thresholds', None)  
    segments = kwargs.get('segments', None)  
    footprints = kwargs.get('footprints', None)  
    
    workspace = kwargs.get('workspace', None)
    file_name = kwargs.get('file_name', None)
    output_log_file = kwargs.get('output_log_file', None) 
    output_seed_folder = kwargs.get('output_seed_folder', None) 
    
    num_threads = kwargs.get('num_threads', 1) 
    
    start_time = datetime.now()
    
    
    output_seed_folder =os.path.join(workspace, output_seed_folder)
    output_json_path = os.path.join(output_seed_folder,output_log_file)
    file_path = os.path.join(workspace, file_name)
    os.makedirs(output_seed_folder , exist_ok=True)
    
    print(f"""{start_time.strftime("%Y-%m-%d %H:%M:%S")}
    Making erosion seeds for 
        Img: {file_path}
        Threshold for Img {target_thresholds}
        Erode {ero_iters} iterations
        Keeping {segments} components
        Erosion footprints {footprints}
        Using {num_threads} threads
            """)
    
    threshold_ero_iter_pairs = list(itertools.product(target_thresholds, ero_iters))

    volume = tifffile.imread(file_path)
    
   # Split pairs among threads
    sublists = [threshold_ero_iter_pairs[i::num_threads] for i in range(num_threads)]
    # Create a list to hold the threads
    threads = []


    # # Start threads
    for sublist in sublists:
        # thread = threading.Thread(target=find_seg_by_ero_v2, args=(volume,sublist, segments ))
        thread = threading.Thread(target=find_seed_by_ero_mp, args=(volume,sublist, segments,
                                                                    output_seed_folder, output_json_path, footprints ))
        threads.append(thread)
        thread.start()
        
    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    end_time = datetime.now()
    running_time = end_time - start_time
    total_seconds = running_time.total_seconds()
    minutes, second = divmod(total_seconds, 60)
    print(f"Running time:{minutes}mins {second}secs")
    
    return (output_seed_folder,output_log_file)
    
    
if __name__ == "__main__":
    
    # Get the file path from the first command-line argument or use the default
    file_path = sys.argv[1] if len(sys.argv) > 1 else './make_seeds.yaml'
    
    _, extension = os.path.splitext(file_path)
    print(f"processing config the file {file_path}")
    if extension == '.yaml':
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
    output_json_path = os.path.join(output_seed_folder,output_log_file)
    file_path = os.path.join(workspace, file_name)
    os.makedirs(output_seed_folder , exist_ok=True)


    start_time = datetime.now()
    print(f"""{start_time.strftime("%Y-%m-%d %H:%M:%S")}
    Making erosion seeds for 
        Img: {file_path}
        Threshold for Img {target_thresholds}
        Erode {ero_iters} iterations
        Keeping {segments} components
        Erosion footprints {footprints}
            """)
    


    threshold_ero_iter_pairs = list(itertools.product(target_thresholds, ero_iters))

    volume = tifffile.imread(file_path)
    
    
    
    # find_seg_by_ero(volume, 4500, 25,[2])
    # sys.exit(0)
    # find_seg_by_ero_v2(volume, threshold_ero_iter_pairs , segments)
    
   
    sublists = [threshold_ero_iter_pairs[i::num_threads] for i in range(num_threads)]

    # Create a list to hold the threads
    threads = []


    # Start a new thread for each sublist
    for sublist in sublists:
        # thread = threading.Thread(target=find_seg_by_ero_v2, args=(volume,sublist, segments ))
        thread = threading.Thread(target=find_seed_by_ero_mp, args=(volume,sublist, segments,
                                                                    output_seed_folder,output_json_path, footprints ))
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
