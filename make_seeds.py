import tifffile

import numpy as np
import threading
import os, sys
from datetime import datetime
import itertools

lock = threading.Lock()

import configparser

# lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'suture_morph'))
# sys.path.insert(0, lib_path)

import suture_morph.suture_morpho as suture_morpho
# import suture_morph.load_config as load_config
# import suture_morpho
# import load_config


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

def find_seg_by_ero(volume, threshold , segments, input_ero_iters):
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
    """The function for find_seed_by_ero using multi threads

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
        
        log_dict = suture_morpho.find_seed_by_ero_custom(volume, threshold , segments, ero_iter,
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


 
def gen_mesh(volume, thresholds, output_dir):
    for threshold in thresholds:
        output_path = os.path.join(output_dir, f"{threshold}.ply")
        if os.path.isfile(output_path):
            return
        else:
            output = BounTI.binary_stack_to_mesh(volume, threshold)
            output.export(output_path)

def main(**kwargs):
    
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
    
   
    sublists = [threshold_ero_iter_pairs[i::num_threads] for i in range(num_threads)]

    # Create a list to hold the threads
    threads = []


    # Start a new thread for each sublist
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
    minutes, _ = divmod(total_seconds, 60)
    print(f"Running time:{minutes}")
    
    return (output_seed_folder,output_log_file)
    
    
if __name__ == "__main__":
    file_path = 'make_seeds.yaml'
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

    # if GEN_WHOLE_MESH:
    #     output_whole_dir = os.path.join(workspace,"whole_mesh")
    #     os.makedirs(output_whole_dir , exist_ok=True)
    #     threads = []
    #     sub_thresholds = [target_thresholds[i::num_threads] for i in range(num_threads)]
    #     # Start a new thread for each sublist
    #     for sublist in sub_thresholds:
    #         thread = threading.Thread(target=gen_mesh, args=(volume,sublist,output_whole_dir))
    #         threads.append(thread)
    #         thread.start()
            
    #     # Wait for all threads to complete
    #     for thread in threads:
    #         thread.join()
    #     print(f"Whole meshes generated at {output_whole_dir}")
    
    end_time = datetime.now()
    running_time = end_time - start_time
    total_seconds = running_time.total_seconds()
    minutes, _ = divmod(total_seconds, 60)
    print(f"Running time:{minutes}")
