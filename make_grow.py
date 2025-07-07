# import nibabel as nib
import os, sys
from skimage.morphology import ball
import tifffile
from datetime import datetime
import yaml
import numpy as np
import pandas as pd
import glob

import threading

import sprout_core.sprout_core as sprout_core 
import sprout_core.config_core as config_core 
import sprout_core.vis_lib as vis_lib
import make_mesh

from multiprocessing import cpu_count
# Maximum threads for multiprocessing
max_threads = cpu_count()
lock = threading.Lock()


# Function to recursively create global variables from the config dictionary
            

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


    unique_ids = np.unique(input_mask)
    # remove the background id
    unique_ids = unique_ids[unique_ids!=0]        
    if to_grow_ids is not None:
        label_id_list = np.intersect1d(unique_ids, to_grow_ids)
    else:
        label_id_list = unique_ids

    
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
        result[boundary] = 0
    
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
    dilation_steps = kwargs.get('dilation_steps', None)
    thresholds = kwargs.get('thresholds', None)  
    upper_thresholds = kwargs.get('upper_thresholds', None)  
    
    num_threads = kwargs.get('num_threads', None) 
    save_every_n_iters = kwargs.get('save_every_n_iters', None)  
    touch_rule = kwargs.get('touch_rule', "stop")  
    grow_to_end = kwargs.get('grow_to_end', False)  
    
    workspace = kwargs.get('workspace', None)
    
    img_path = kwargs.get('img_path', None)
    seg_path = kwargs.get('seg_path', None) 
    boundary_path  = kwargs.get('boundary_path', None)
    
    img = kwargs.get('img', None)
    seg = kwargs.get('seg', None)
    boundary = kwargs.get('boundary', None)
    
    output_folder = kwargs.get('output_folder', None) 
    final_grow_output_folder = kwargs.get('final_grow_output_folder', None) 

    
    base_name = kwargs.get('base_name', None)  
    use_simple_naming = kwargs.get('use_simple_naming', True)  
    
    to_grow_ids = kwargs.get('to_grow_ids', None) 
    is_sort = kwargs.get('is_sort', True) 
    min_growth_size = kwargs.get('min_growth_size', 50) 
    no_growth_max_iter = kwargs.get('no_growth_max_iter', 3) 
    
    # For mesh making
    is_make_meshes = kwargs.get('is_make_meshes', False) 
    downsample_scale = kwargs.get('downsample_scale', 10) 
    step_size  = kwargs.get('step_size', 1) 
    
    # for napari
    return_for_napari = kwargs.get('return_for_napari', False)
    
    
    
    
    default_grow_to_end_iter = 200
    
    if grow_to_end:
        dilation_steps = [default_grow_to_end_iter] * len(dilation_steps)    
    
    if num_threads is None:
        num_threads = max(1, max_threads // 2)

    if num_threads>=max_threads:
        num_threads =  max(1,max_threads-1)
    


    # Ensure thresholds and dilation_steps have the same length
    if isinstance(thresholds, int):
        thresholds = [thresholds]
    if isinstance(dilation_steps, int):
        dilation_steps = [dilation_steps]

    thresholds, upper_thresholds = config_core.check_and_assign_thresholds(thresholds, upper_thresholds, reverse= True)
    
    assert len(thresholds) == len(dilation_steps), f"thresholds and dilation_steps must have the same length, but got {len(thresholds)} and {len(dilation_steps)}."
    
    
    if isinstance(save_every_n_iters, list):
        assert len(thresholds) == len(save_every_n_iters), f"Save interval list should have the same length as well"
    if isinstance(save_every_n_iters, int):
        save_every_n_iters = [save_every_n_iters] * len(thresholds)
    if save_every_n_iters is None:
        save_every_n_iters = dilation_steps
    
    if workspace is not None:
        img_path = os.path.join(workspace, img_path)
        seg_path = os.path.join(workspace, seg_path)
        output_folder = os.path.join(workspace, output_folder)
   
    base_name = config_core.check_and_assign_base_name(base_name, img_path, "grown_result")

    
    # lodading the image and segmentation mask
    # If img and seg are provided, use them; otherwise, read from paths
    if img is None:
        img = tifffile.imread(img_path)   

    if seg is None:
        seg = tifffile.imread(seg_path)
    
    
    # Loading a boundary if it's provided
    if boundary is None and boundary_path is not None:
        if workspace is not None:
            boundary_path = os.path.join(workspace, boundary_path)
        boundary = tifffile.imread(boundary_path)
        boundary = sprout_core.check_and_cast_boundary(boundary)
    elif boundary is not None:
        boundary = sprout_core.check_and_cast_boundary(boundary)
        
    
    output_folder = os.path.join(output_folder, base_name)
    
    os.makedirs(output_folder , exist_ok=True)
    

    
    # Record the start time
    start_time = datetime.now()
    # Print
    
    # Convert paths to absolute paths if they are not already
    if output_folder is not None and not os.path.isabs(output_folder):
        output_folder = os.path.abspath(output_folder)
    
    values_to_print = {   
            "Image Path": img_path,
            "Segmentation Path": seg_path,
            "Boundary Path": boundary_path,      
            "grow_to_end" : grow_to_end,
            "Dilate Iterations": dilation_steps,
            "Grow Thresholds": thresholds,
            "Grow upper thresholds": upper_thresholds,
            "Output Folder": output_folder,
            "Save every iterations": save_every_n_iters,
            "num_threads": num_threads,
            "Early stopping": f"min_growth_size = {min_growth_size} and no_growth_max_iter = {no_growth_max_iter}"
            }
    print("Start time: "+start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Growing on: {img_path}")
    for key, value in values_to_print.items():
        print(f"  {key}: {value}")

    # initialize the log list
    # This will be used to save the growing results
    df_log = []
    
    # Initialize a dictionary to store the growing results
    grows_dict ={}

    # create the result array
    # If the input mask has more than 65535 ids, convert it to uint16
    result = seg.copy()
    if np.unique(result).size > 255:
        print("Input mask has more than 65535 ids, converting to uint16")
        result = result.astype('uint16')
    else:
        result = result.astype('uint8')
        
    # Iterate through and make growth results
    for i, (threshold ,upper_threshold,dilate_iter) in enumerate(zip(thresholds , upper_thresholds,dilation_steps)):
        # Set the count for check diff for each growing threshold
        count_below_threshold = 0
        
        
        threshold_name = "_".join(str(s) for s in thresholds[:i+1])
        dilate_name = "_".join(str(s) for s in dilation_steps[:i+1])
        
        if upper_threshold is not None:
            threshold_binary = (img>=threshold) & (img<=upper_threshold)
        else:
            threshold_binary = img >= threshold

            
        full_size = np.sum(threshold_binary)
        print(f"Size of the threshold {threshold} to {upper_threshold} mask: {full_size}")
        for i_dilate in range(1, dilate_iter+1):

            
            # Get the input size for the log
            input_size = np.sum(result!=0)
            result = dilation_one_iter_mp(result, threshold_binary ,
                                          num_threads=num_threads,
                                            touch_rule = touch_rule,
                                            to_grow_ids=to_grow_ids,
                                            boundary=boundary)
            
            ## Check if output size and input 's diff is bigger than min_growth_size

            # Get the output size for the log
            output_size = np.sum(result!=0)
            if output_size - input_size < min_growth_size:
                count_below_threshold += 1
            else:
                count_below_threshold = 0
                
            # Situations to save grow results 
            # When it ends:
            # 1. Reach the final iter, 
            # 2. Not been growing for sometime
            # 3. Grow to the size of the current threshold   
            if (i_dilate% save_every_n_iters[i]==0 or 
                i_dilate ==dilate_iter or 
                count_below_threshold >= no_growth_max_iter or
                (grow_to_end == True and abs(full_size - output_size) < 0.05) ):
                

                cur_threshold = f"{threshold}_{upper_threshold}"
                if use_simple_naming:
                    output_grow_name = f'INTER_{base_name}_{cur_threshold}_{i_dilate}'

                else:
                    output_grow_name = f'INTER_{base_name}_iter_{i_dilate}_dilate_{dilate_name}_thre_{threshold_name}_{upper_threshold}'
                   
            
                output_path = os.path.join(output_folder, output_grow_name + ".tif")
                # Write the log
                df_log.append({'id': (i*dilate_iter)+i_dilate, 
                    'grow_size': output_size,
                    'full_size': full_size,
                    'cur_threshold': cur_threshold,
                    "file_name": os.path.basename(output_path),
                    'full_path': os.path.abspath(output_path),
                    'cur_dilate_step': i_dilate,
                    })
                
                
                result,_ = sprout_core.reorder_segmentation(result, sort_ids=is_sort)
                tifffile.imwrite(output_path,  result, compression ='zlib')
                if return_for_napari:
                    grows_dict[output_grow_name] =result
                
                print(f"\tGrown result has been saved {os.path.abspath(output_path)}")
                print(f"\tIter:{i_dilate}. Last Input size = {input_size} and Output_size = {output_size}")

                
                # Early stopping conditions
                if count_below_threshold >= no_growth_max_iter:
                    print(f"\tNot growing for {no_growth_max_iter} iters\nBreaking at iteration {i_dilate} with Input size = {input_size} and Output_size = {output_size}")
                    break
                # If grow to end, check if the size is similar to the threshold size
                # If the size is similar to the threshold size, break
                if (grow_to_end == True and abs(full_size - output_size) < 0.05) :
                    print(f"\tGrow size is similar to the threshold size\nBreaking at iteration {i_dilate}: Input size = {input_size}, Output_size = {output_size} and size of threshold binary = {full_size}")
                    break
            
        print(f"\tFinish growing. Last Input size = {input_size} and Output_size = {output_size}\n")
    
    ## Save the final grow output as the final_<img_name>
    final_grow_name = f"FINAL_GROW_{base_name}"
    if final_grow_output_folder is not None:
        final_output_path = os.path.join(final_grow_output_folder,f"{final_grow_name}.tif")
    else:
        final_output_path = os.path.join(output_folder,f"{final_grow_name}.tif")
    tifffile.imwrite(final_output_path, result, compression ='zlib')
    if return_for_napari:
        grows_dict[final_grow_name] =result
    

    total_seconds = (datetime.now() - start_time).total_seconds()
    minutes, s = divmod(total_seconds, 60)
    print(f"Running time:{minutes} minutes {round(s,2)} sec\n")
    
    # Save the dataframe of the growing log
    df_log = pd.DataFrame(df_log)
    log_path =  os.path.join(output_folder, f'grow_log_{base_name}.csv')   
    df_log.to_csv(log_path, index = False)

    # Save the configuration parameters used for growing
    config_core.save_config_with_output({
        "params": kwargs},output_folder)

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
    log_dict = {
        "final_output_path": final_output_path,
        "log_path":log_path,
        "output_folder": output_folder
    }
    
    return grows_dict ,log_dict


def run_make_grow(file_path):
          
    _, extension = os.path.splitext(file_path)
    print(f"processing config the file {file_path}")
    if extension == '.yaml':
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            optional_params = config_core.validate_input_yaml(config, config_core.input_val_make_grow)
        
    _,log_dict = grow_mp(
        workspace = optional_params['workspace'],
        img_path = config['img_path'] ,
        seg_path = config['seg_path'],
        
        output_folder = config['output_folder'],
        
        boundary_path = optional_params['boundary_path'],
        
        dilation_steps = config['dilation_steps'],
        thresholds = config['thresholds'],
        upper_thresholds = optional_params["upper_thresholds"],
        num_threads = config['num_threads'],
        
        save_every_n_iters = config['save_every_n_iters'],  
        touch_rule = config['touch_rule'], 
        
        
        grow_to_end = optional_params["grow_to_end"],
        to_grow_ids = optional_params["to_grow_ids"],
        
        final_grow_output_folder =optional_params["final_grow_output_folder"],
        base_name =  optional_params["base_name"],
        use_simple_naming =  optional_params["use_simple_naming"],
        

        is_sort = optional_params['is_sort'],
        min_growth_size = optional_params['min_growth_size'],
        no_growth_max_iter = optional_params['no_growth_max_iter'],
        
        # For mesh making
        is_make_meshes = optional_params['is_make_meshes'],
        downsample_scale = optional_params['downsample_scale'],
        step_size  = optional_params['step_size']
              
        )
    
    vis_lib.plot_grow(pd.read_csv(log_dict['log_path']),
              log_dict['log_path'] +".png"
              )

if __name__ == "__main__":
    
    # Get the file path from the first command-line argument or use the default
    file_path = sys.argv[1] if len(sys.argv) > 1 else './make_grow.yaml'
    
    run_make_grow(file_path)
    
   
    