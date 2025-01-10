# import nibabel as nib
import os, sys
from skimage.morphology import ball
import tifffile
from datetime import datetime
import yaml
import numpy as np
import pandas as pd
import glob
import multiprocessing
import threading

import sprout_core.sprout_core as sprout_core 
import sprout_core.vis_lib as vis_lib
import make_mesh


# Maximum threads for multiprocessing
max_threads = multiprocessing.cpu_count()
lock = threading.Lock()


# Function to recursively create global variables from the config dictionary
def load_config_yaml(config, parent_key=''):
    """
    Recursively load configuration values from a YAML dictionary into global variables.

    Args:
        config (dict): The configuration dictionary.
        parent_key (str): Key prefix for nested configurations.
    """
    for key, value in config.items():
        if isinstance(value, dict):
            load_config_yaml(value, parent_key='')
        else:
            globals()[parent_key + key] = value
            

def grow_function(result, threshold_binary, label_id_list,touch_rule,non_bg_mask):
    """
    Perform the growth operation for each label ID in the label ID list.

    Args:
        result (np.ndarray): The mask being grown.
        threshold_binary (np.ndarray): Binary mask threshold for the guide.
        label_id_list (list): List of label IDs to grow.
        touch_rule (str): Rule for handling overlaps (e.g., 'stop').
    """
    for label_id in label_id_list:
        dilated_binary_label_id = (result ==label_id)
        dilated_binary_label_id = sprout_core.dilation_binary_img_on_sub(dilated_binary_label_id, 
                                                                    margin = 2, kernal_size = 1)

        if touch_rule == 'stop':
            # This is the binary for non-label of the updated mask
            # binary_non_label = (result !=label_id) & non_bg_mask
            # See if original mask overlay with grown label_id mask
            # overlay = np.logical_and(binary_non_label, dilated_binary_label_id)
            overlay = (result !=label_id) & non_bg_mask & dilated_binary_label_id
                        
            # # Check if there are any True values in the resulting array
            # HAS_OVERLAY = np.any(overlay)
            
            # Quicker way to do intersection check
            # inter = np.sum(binary_non_label[dilated_binary_label_id])
            # HAS_OVERLAY = inter>0
            
            # print(f"""
            #     np.sum((result ==label_id)){np.sum((result ==label_id))},
            #     np.sum(dilated_binary_label_id){np.sum(dilated_binary_label_id)},
            #     np.sum(overlay){np.sum(overlay)},
            #     np.sum(binary_non_label){np.sum(binary_non_label)}
            #     """)
            
            # if HAS_OVERLAY:
            dilated_binary_label_id[overlay] = False
        # Lock result for multi-threading, and use threshold_binary as the guide
        with lock:        
            # result[dilated_binary_label_id & threshold_binary] = label_id  
            result[dilated_binary_label_id] = label_id  

def dilation_one_iter_mp(input_mask, threshold_binary, 
                            num_threads,
                             touch_rule = 'stop',
                             segments=None, ero_shape = 'ball',
                             to_grow_ids = None,
                             boundary = None):
    """
    Perform one iteration of dilation by using grow_function(...) in multi-threading.

    Args:
        result (np.ndarray): The mask being grown.
        threshold_binary (np.ndarray): Binary mask threshold for the guide.
        num_threads (int): Number of threads to use.
        touch_rule (str): Rule for handling overlaps.
        to_grow_ids (list, optional): Specific IDs to grow. Defaults to None.
        boundary (np.ndarray, optional): Boundary mask to constrain growth. Defaults to None.

    Returns:
        np.ndarray: The updated mask after one dilation iteration.
    """

    if to_grow_ids is None:
        label_id_list = np.unique(input_mask)
        label_id_list = label_id_list[label_id_list!=0]
    else:
        label_id_list = to_grow_ids
    
    result = input_mask.copy()  
    # Create sublists and start multi-threading
    sublists = [label_id_list[i::num_threads] for i in range(num_threads)]
    # Create a list to hold the threads
    threads = []
    
    if touch_rule == 'stop':
        non_bg_mask = (result != 0)
    else:
        non_bg_mask = None
    
    for sublist in sublists:
        # print(f"Processing sublist {sublist}")
        
        thread = threading.Thread(target=grow_function, args=(result, 
                                                              threshold_binary,
                                                              sublist,
                                                              touch_rule,
                                                              non_bg_mask))
        threads.append(thread)
        thread.start()
        
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
     
    
    if boundary is not None:
        result[boundary] = False
    
    result[threshold_binary==False] = 0  
    
    return result

        
def grow_mp(**kwargs):
    """
    Main function to perform multi-threaded growing on a segmentation mask.

    Args:
        kwargs: Key-value arguments containing growing parameters.

    Returns:
        dict: A dictionary containing paths to the final output and log files.
    """
    # Extract configuration values from kwargs
    dilate_iters = kwargs.get('dilate_iters', None)
    thresholds = kwargs.get('thresholds', None)  
    num_threads = kwargs.get('num_threads', None) 
    save_interval = kwargs.get('save_interval', None)  
    touch_rule = kwargs.get('touch_rule', "stop")  
    grow_to_end = kwargs.get('grow_to_end', False)  
    
    workspace = kwargs.get('workspace', None)
    img_path = kwargs.get('img_path', None)
    seg_path = kwargs.get('seg_path', None) 
    output_folder = kwargs.get('output_folder', None) 
    final_grow_output_folder = kwargs.get('final_grow_output_folder', None) 
    name_prefix = kwargs.get('name_prefix', "final_grow")  
    simple_naming = kwargs.get('simple_naming', True)  
    
    to_grow_ids = kwargs.get('to_grow_ids', None) 
    is_sort = kwargs.get('is_sort', True) 
    min_diff = kwargs.get('min_diff', 50) 
    tolerate_iters = kwargs.get('tolerate_iters', 3) 
    
    # For mesh making
    is_make_meshes = kwargs.get('is_make_meshes', False) 
    downsample_scale = kwargs.get('downsample_scale', 10) 
    step_size  = kwargs.get('step_size', 1) 
    
    
    default_grow_to_end_iter = 150
    if grow_to_end:
        dilate_iters = [default_grow_to_end_iter] * len(dilate_iters)    
    
    boundary_path  = kwargs.get('boundary_path', None)
    
    if num_threads is None:
        num_threads = max_threads-1
    
    # Test if the grown result and input's diff is more than this. Default is 10.
    # min_diff = 50
    # The number of iters for diff is less than diff_threshold

    # Ensure thresholds and dilate_iters have the same length
    assert len(thresholds) == len(dilate_iters), f"thresholds and dilate_iters must have the same length, but got {len(thresholds)} and {len(dilate_iters)}."
    if isinstance(save_interval, list):
        assert len(thresholds) == len(save_interval), f"Save interval list should have the same length as well"

    
    if workspace is not None:
        img_path = os.path.join(workspace, img_path)
        seg_path = os.path.join(workspace, seg_path)
        output_folder = os.path.join(workspace, output_folder)
    
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    input_mask = tifffile.imread(seg_path)
    ori_img = tifffile.imread(img_path)

    os.makedirs(output_folder , exist_ok=True)
    
    # Loading a boundary if it's provided
    if boundary_path is not None:
        if workspace is not None:
            boundary_path = os.path.join(workspace, boundary_path)
        boundary = tifffile.imread(boundary_path)
        boundary = sprout_core.check_and_cast_boundary(boundary)
    else:
        boundary = None
    
    # Record the start time
    start_time = datetime.now()
    # Print
    values_to_print = {   
            "Segmentation Path": seg_path,
            "Boundary Path": boundary_path,      
            "grow_to_end" : grow_to_end,
            "Dilate Iterations": dilate_iters,
            "Grow Thresholds": thresholds,
            "Output Folder": output_folder,
            "Save Interval": save_interval,
            "num_threads": num_threads,
            "Early stopping": f"min_diff = {min_diff} and tolerate_iters = {tolerate_iters}"
            }
    print("Start time: "+start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Growing on: {img_path}")
    for key, value in values_to_print.items():
        print(f"  {key}: {value}")

    
    df_log = []

    result = input_mask.copy()
    result = result.astype('uint8')
        
    # Iterate through and make growth results
    for i, (threshold,dilate_iter) in enumerate(zip(thresholds,dilate_iters)):
        # Set the count for check diff for each growing threshold
        count_below_threshold = 0
        
        # How many iterations for saving intermediate results
        if isinstance(save_interval, list):
                real_save_interval = save_interval[i]
        elif isinstance(save_interval, int):
            real_save_interval = save_interval
        
        threshold_name = "_".join(str(s) for s in thresholds[:i+1])
        dilate_name = "_".join(str(s) for s in dilate_iters[:i+1])
        
        
        threshold_binary = ori_img > threshold
        full_size = np.sum(threshold_binary)
        print(f"Size of the threshold {threshold} mask:  {full_size}")
        for i_dilate in range(1, dilate_iter+1):
            # Get the input size for the log
            input_size = np.sum(result!=0)
            
            ##
            
            ## Making grow for one iteration
            # result = sprout_core.dilation_one_iter(result, threshold_binary ,
            #                                 touch_rule = touch_rule,
            #                                 to_grow_ids=to_grow_ids)
            
        
            result = dilation_one_iter_mp(result, threshold_binary ,
                                          num_threads=num_threads,
                                            touch_rule = touch_rule,
                                            to_grow_ids=to_grow_ids,
                                            boundary=boundary)
            
            # Get the output size for the log
            output_size = np.sum(result!=0)
            
            
            
            ## Check if output size and input 's diff is bigger than min_diff
            if output_size - input_size < min_diff:
                count_below_threshold += 1
            else:
                count_below_threshold = 0
                
            # Situations to save grow results 
            # When it ends:
            # 1. Reach the final iter, 
            # 2. Not been growing for sometime
            # 3. Grow to the size of the current threshold   
            if (i_dilate%real_save_interval==0 or 
                i_dilate ==dilate_iter or 
                count_below_threshold >= tolerate_iters or
                (grow_to_end == True and abs(full_size - output_size) < 0.05) ):
                
                if simple_naming:
                    output_path = os.path.join(output_folder, f'{base_name}_{threshold}_{i_dilate}.tif')
                else:
                    output_path = os.path.join(output_folder, f'{base_name}_iter_{i_dilate}_dilate_{dilate_name}_thre_{threshold_name}.tif')
                
                # Write the log
                df_log.append({'id': (i*dilate_iter)+i_dilate, 
                    'grow_size': output_size,
                    'full_size': full_size,
                    'cur_threshold': threshold,
                    "file_name": os.path.basename(output_path),
                    'full_path': os.path.abspath(output_path),
                    'cur_dilate_step': i_dilate,
                    })
                
                
                result,_ = sprout_core.reorder_segmentation(result, sort_ids=is_sort)
                tifffile.imwrite(output_path, 
                    result,
                    compression ='zlib')
                print(f"\tGrown result has been saved {output_path}")
                print(f"\tIter:{i_dilate}. Last Input size = {input_size} and Output_size = {output_size}")

                
                if count_below_threshold >= tolerate_iters:
                    print(f"\tNot growing for {tolerate_iters} iters\nBreaking at iteration {i_dilate} with Input size = {input_size} and Output_size = {output_size}")
                    break
            
                if (grow_to_end == True and abs(full_size - output_size) < 0.05) :
                    print(f"\tGrow size is similar to the threshold size\nBreaking at iteration {i_dilate}: Input size = {input_size}, Output_size = {output_size} and size of threshold binary = {full_size}")
                    break
            
        print(f"\tFinish growing. Last Input size = {input_size} and Output_size = {output_size}")
    
    ## Save the final grow output as the final_<img_name>
    if final_grow_output_folder is not None:
        final_output_path = os.path.join(final_grow_output_folder,f"{name_prefix}_{base_name}.tiff")
    else:
        final_output_path = os.path.join(output_folder,f"{name_prefix}_{base_name}.tiff")
    tifffile.imwrite(final_output_path, 
        result,
        compression ='zlib')
    

    total_seconds = (datetime.now() - start_time).total_seconds()
    minutes, s = divmod(total_seconds, 60)
    print(f"Running time:{minutes} minutes {round(s,2)} sec")
    
    # Save the dataframe of the growing log
    df_log = pd.DataFrame(df_log)
    log_path =  os.path.join(output_folder, f'grow_log_{base_name}.csv')   
    df_log.to_csv(log_path, index = False)


    
    # Make meshes  
    if is_make_meshes:  
        tif_files = glob.glob(os.path.join(output_folder, '*.tif'))

        for tif_file in tif_files:
            make_mesh.make_mesh_for_tiff(tif_file,output_folder,
                                num_threads=num_threads,no_zero = True,
                                colormap = "color10",
                                downsample_scale=downsample_scale,
                                step_size=step_size)
    
    # Return a dict 
    grow_dict = {
        "final_output_path": final_output_path,
        "log_path":log_path,
        "output_folder": output_folder
    }
    
    return grow_dict


def main(**kwargs):
    
    dilate_iters = kwargs.get('dilate_iters', None)
    thresholds = kwargs.get('thresholds', None)  
    save_interval = kwargs.get('save_interval', None)  
    touch_rule = kwargs.get('touch_rule', "stop")  
    
    workspace = kwargs.get('workspace', None)
    img_path = kwargs.get('img_path', None)
    seg_path = kwargs.get('seg_path', None) 
    output_folder = kwargs.get('output_folder', None) 
    to_grow_ids = kwargs.get('to_grow_ids', None) 
    
    is_sort = kwargs.get('is_sort', True) 
    
    min_diff = kwargs.get('min_diff', 50) 
    tolerate_iters = kwargs.get('tolerate_iters', 3) 
    
    is_make_meshes = kwargs.get('is_make_meshes', False) 
    num_threads = kwargs.get('num_threads', None) 
    downsample_scale = kwargs.get('downsample_scale', 10) 
    step_size  = kwargs.get('step_size', 1) 
    
    
    if num_threads is None:
        num_threads = max_threads-1
    # Test if the grown result and input's diff is more than this. Default is 10.
    # min_diff = 50
    # The number of iters for diff is less than diff_threshold
    
    
    
    assert len(thresholds) == len(dilate_iters), f"thresholds and dilate_iters must have the same length, but got {len(thresholds)} and {len(dilate_iters)}."
    if isinstance(save_interval, list):
        assert len(thresholds) == len(save_interval), f"Save interval list should have the same length as well"

    
    if workspace is not None:
        img_path = os.path.join(workspace, 
                                img_path)
        seg_path = os.path.join(workspace, seg_path)
        output_folder = os.path.join(workspace, output_folder)
    
    base_name = os.path.basename(seg_path)
    input_mask = tifffile.imread(seg_path)
    ori_img = tifffile.imread(img_path)


    os.makedirs(output_folder , exist_ok=True)
    
    # Record the start time
    start_time = datetime.now()
    print(f"""{start_time.strftime("%Y-%m-%d %H:%M:%S")}
    Making growing for 
        Img: {img_path}
        Threshold for Img {thresholds}
        Seed {seg_path} to grow {dilate_iters} iterations
        Early stopping: min_diff = {min_diff} and tolerate_iters = {tolerate_iters}
            """)
    
    df_log = []

    result = input_mask.copy()
    result = result.astype('uint8')
    for i, (threshold,dilate_iter) in enumerate(zip(thresholds,dilate_iters)):
        # Set the count for check diff for each growing threshold
        count_below_threshold = 0
        
        if isinstance(save_interval, list):
                real_save_interval = save_interval[i]
        elif isinstance(save_interval, int):
            real_save_interval = save_interval
        
        threshold_name = "_".join(str(s) for s in thresholds[:i+1])
        dilate_name = "_".join(str(s) for s in dilate_iters[:i+1])
        
        
        print(f"threshold:{threshold} dilate_iter:{dilate_iter}.real_save_interval:{real_save_interval}")
        for i_dilate in range(1, dilate_iter+1):
            threshold_binary = ori_img > threshold
            
            # Get the input size for the log
            input_size = np.sum(result!=0)
            
            ## Making grow for one iteration
            result = sprout_core.dilation_one_iter(result, threshold_binary ,
                                            touch_rule = touch_rule,
                                            to_grow_ids=to_grow_ids)
            
            # Get the output size for the log
            output_size = np.sum(result!=0)
            
            output_path = os.path.join(output_folder, f'{base_name}_iter_{i_dilate}_dilate_{dilate_name}_thre_{threshold_name}.tif')
            
                
            if i_dilate%real_save_interval==0 or count_below_threshold >= tolerate_iters:
                df_log.append({'id': (i*dilate_iter)+i_dilate, 
                            'grow_size': output_size,
                            'full_size':np.sum(threshold_binary),
                            'cur_threshold': threshold,
                            "file_name": os.path.basename(output_path),
                            'full_path': os.path.abspath(output_path),
                            'cur_dilate_step': i_dilate,
                            })
            
            ## Check if output size and input 's diff is bigger than min_diff
            if output_size - input_size < min_diff:
                count_below_threshold += 1
            else:
                count_below_threshold = 0
            if i_dilate%real_save_interval==0 or i_dilate ==dilate_iter or count_below_threshold >= tolerate_iters:
                
                result,_ = sprout_core.reorder_segmentation(result, sort_ids=is_sort)
                tifffile.imwrite(output_path, 
                    result,
                    compression ='zlib')
                print(f"\tGrown result has been saved {output_path}")
                print(f"\tIter:{i_dilate}. Last Input size = {input_size} and Output_size = {output_size}")

                
                if count_below_threshold >= tolerate_iters:
                    print(f"\tBreaking at iteration {i_dilate} with Input size = {input_size} and Output_size = {output_size}")
                    break
            
            # if i_dilate%real_save_interval==0 or i_dilate ==dilate_iter:
            #     # output_path = os.path.join(workspace, f'result/ai/dila_{dilate_iter}_{threshold}_rule_{touch_rule}.tif')
              
                
            #     print(f"\tGrown result has been saved {output_path}")
            #     print(f"\tIter:{i_dilate}. Last Input size = {input_size} and Output_size = {output_size}")
            #     tifffile.imwrite(output_path, 
            #     result,
            #     compression ='zlib')
        print(f"\tFinish growing. Last Input size = {input_size} and Output_size = {output_size}")
    
    end_time = datetime.now()
    running_time = end_time - start_time
    total_seconds = running_time.total_seconds()
    minutes, _ = divmod(total_seconds, 60)
    print(f"Running time:{minutes}")
    
    df_log = pd.DataFrame(df_log)
    
    log_path =  os.path.join(output_folder, f'grow_log_{base_name}.csv')   
    df_log.to_csv(log_path, index = False)

    grow_dict = {
        "log_path":log_path,
        "output_folder": output_folder
    }
    
    # Make meshes  
    if is_make_meshes:  
        tif_files = glob.glob(os.path.join(output_folder, '*.tif'))

        for tif_file in tif_files:
            make_mesh.make_mesh_for_tiff(tif_file,output_folder,
                                num_threads=num_threads,no_zero = True,
                                colormap = "color10",
                                downsample_scale=downsample_scale,
                                step_size=step_size)
    
    return grow_dict

if __name__ == "__main__":
    
    # Get the file path from the first command-line argument or use the default
    file_path = sys.argv[1] if len(sys.argv) > 1 else './make_grow_result.yaml'
    
    _, extension = os.path.splitext(file_path)
    print(f"processing config the file {file_path}")
    if extension == '.yaml':
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            boundary_path = config.get('boundary_path', None)
            
            to_grow_ids = config.get('to_grow_ids', None)
            num_threads = config.get('num_threads', None)
            
            
            grow_to_end = config.get('grow_to_end', False)  
            is_sort = config.get('is_sort', True) 
            min_diff = config.get('min_diff', 50) 
            tolerate_iters = config.get('tolerate_iters', 3) 
            
            # For mesh making
            is_make_meshes = config.get('is_make_meshes', False) 
            downsample_scale = config.get('downsample_scale', 10) 
            step_size  = config.get('step_size', 1)     
            
            final_grow_output_folder =config.get("final_grow_output_folder",None)
            name_prefix = config.get("name_prefix","final_grow")
            simple_naming = config.get("simple_naming",True)
            
        
        load_config_yaml(config)
    

    grow_dict = grow_mp(
        workspace = workspace,
        img_path = img_path,
        seg_path = seg_path,
        output_folder = output_folder,
         
        dilate_iters = dilate_iters,
        thresholds = thresholds,
        num_threads = num_threads,
        
        save_interval = save_interval,  
        touch_rule = touch_rule, 
        
        
        grow_to_end = grow_to_end,
        to_grow_ids = to_grow_ids,
        
        final_grow_output_folder =final_grow_output_folder,
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
    
    vis_lib.plot_grow(pd.read_csv(grow_dict['log_path']),
              grow_dict['log_path'] +".png"
              )
    