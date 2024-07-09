import tifffile
from . import BounTI
import skimage
import numpy as np
import threading
import os, json

lock = threading.Lock()

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


def bounti_with_thresholds(volume, init_thresholds, target_threshold, is_flood = True):
    
    for init_threshold in init_thresholds:
        if is_flood:
            volume_label, seed, log_dict = BounTI.segmentation(volume,init_threshold,target_threshold,bounti_segments,bounti_iters,seed_dilation=False)
        else:
            volume_label, seed, log_dict = BounTI.segmentation_flood(volume,init_threshold,target_threshold,bounti_segments,bounti_iters,seed_dilation=False)
        
        # volume_label, seed = BounTI.segmentation_ero(volume,5500,5000,45,2,seed_dilation=False)
        labeled_volume = volume_label.astype(np.uint8)
        
        seed = seed.astype(np.uint8)

        output_file = os.path.join(output_seg_folder ,
                                   f"seg_{init_threshold}_{target_threshold}.tif")
        
        seed_file = os.path.join(output_seed_folder , 
                                 f"seed_{init_threshold}_{target_threshold}.tif")
        
        tifffile.imwrite(output_file, 
                        labeled_volume)
        
        tifffile.imwrite(seed_file, 
                seed)

        log_dict['input_file'] = file_path
        log_dict['output_file'] = output_file
        
        with lock:
            # filename = f'output/json/Bount_ori_run_log_{init_threshold}_{target_threshold}.json'
            write_json(output_json_path, log_dict)   


if __name__ == "__main__":
    ### To change your input params ####
    file_path = r"C:\Users\Yichen\OneDrive\work\codes\nhm_monai_suture_demo\data\bones_suture\procavia\procaviaH4981C_0001.tif.resampled.tif"
    output_json_path = 'Bount_ori_run_log.json'
    output_seg_folder = "output_procavia/"
    output_seed_folder = "output_procavia_seed"
    num_threads = 3
    
    ### To change the thre ranges
    max_thre = 6300
    min_thre = 4500
    target_threshold = 3000
    #The interval for selecting thresholds, 1 as selecting all values from max to min
    interval = 500
    bounti_segments = 45
    bounti_iters = 2
    is_flood = True


    os.makedirs(output_seg_folder , exist_ok=True)
    os.makedirs(output_seed_folder , exist_ok=True)

    # tifffile.imread("../../nhm_monai_suture_demo/data/bones_suture/procavia/procaviaH4981C_0001.tif.resampled.tif")
    volume = BounTI.volume_import(file_path)


    all_values = np.unique(volume)
    test_values = all_values[(all_values<=max_thre) & (all_values>=min_thre)] 
    test_values = test_values[::interval]
    print(f"Testing {len(test_values)} init thresholds  from {max_thre} to {min_thre}")
    print(test_values)

    # bounti_with_thresholds(volume, test_values, target_threshold)

    sublists = [test_values[i::num_threads] for i in range(num_threads)]

    # Create a list to hold the threads
    threads = []

    # Start a new thread for each sublist
    for sublist in sublists:
        thread = threading.Thread(target=bounti_with_thresholds, args=(volume,sublist,target_threshold,is_flood))
        threads.append(thread)
        thread.start()
        
    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print(f"All threads have completed. Log is saved at {output_json_path},seeds are saved at {output_seed_folder} images are saved at {output_seg_folder}")

