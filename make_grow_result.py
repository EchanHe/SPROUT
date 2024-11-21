# import nibabel as nib
import os, sys
from skimage.morphology import ball
import tifffile
from datetime import datetime
import json ,yaml
import numpy as np
import pandas as pd
import glob
import multiprocessing
max_threads = multiprocessing.cpu_count()

import sprout_core.sprout_core as sprout_core 
import sprout_core.vis_lib as vis_lib

import make_mesh
import threading
lock = threading.Lock()


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

def grow_function(result, threshold_binary, label_id_list,touch_rule):
    
    for label_id in label_id_list:
    
        print(f"processing class id {label_id}")
        
        
        
        dilated_binary_label_id = (result ==label_id)
        dilated_binary_label_id = sprout_core.dilation_binary_img_on_sub(dilated_binary_label_id, 
                                                                    margin = 2, kernal_size = 1)

        if touch_rule == 'stop':
            # This is the binary for non-label of the updated mask
            binary_non_label = (result !=label_id) & (result != 0)
            # See if original mask overlay with grown label_id mask
            overlay = np.logical_and(binary_non_label, dilated_binary_label_id)
                        
            # # Check if there are any True values in the resulting array
            # HAS_OVERLAY = np.any(overlay)
            
            # Quicker way to do intersection check
            inter = np.sum(binary_non_label[dilated_binary_label_id])
            HAS_OVERLAY = inter>0
            
            # print(f"""
            #     np.sum((result ==label_id)){np.sum((result ==label_id))},
            #     np.sum(dilated_binary_label_id){np.sum(dilated_binary_label_id)},
            #     np.sum(overlay){np.sum(overlay)},
            #     np.sum(binary_non_label){np.sum(binary_non_label)}
            #     """)
            
            if HAS_OVERLAY:
                dilated_binary_label_id[overlay] = False
        with lock:        
            result[dilated_binary_label_id & threshold_binary] = label_id  


def dilation_one_iter_mp(input_mask, threshold_binary, 
                            num_threads,
                             touch_rule = 'stop',
                             segments=None, ero_shape = 'ball',
                             to_grow_ids = None):
    if to_grow_ids is None:
        label_id_list = np.unique(input_mask)
        label_id_list = label_id_list[label_id_list!=0]
    else:
        label_id_list = to_grow_ids
    
    result = input_mask.copy()  
    
    
    sublists = [label_id_list[i::num_threads] for i in range(num_threads)]
    # Create a list to hold the threads
    threads = []
    for sublist in sublists:
        print(f"Processing sublist {sublist}")
        
        thread = threading.Thread(target=grow_function, args=(result, 
                                                              threshold_binary,
                                                              sublist,
                                                              touch_rule))
        threads.append(thread)
        thread.start()
        
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    
    # for label_id in label_id_list:
    #     dilated_binary_label_id = (result ==label_id)
    #     dilated_binary_label_id = sprout_core.dilation_binary_img_on_sub(dilated_binary_label_id, 
    #                                                             margin = 2, kernal_size = 1)
        
        
    #     if touch_rule == 'stop':
    #         # This is the binary for non-label of the updated mask
    #         binary_non_label = (result !=label_id) & (result != 0)
    #         # See if original mask overlay with grown label_id mask
    #         overlay = np.logical_and(binary_non_label, dilated_binary_label_id)
                        
    #         # # Check if there are any True values in the resulting array
    #         # HAS_OVERLAY = np.any(overlay)
            
    #         # Quicker way to do intersection check
    #         inter = np.sum(binary_non_label[dilated_binary_label_id])
    #         HAS_OVERLAY = inter>0
            
    #         # print(f"""
    #         #     np.sum((result ==label_id)){np.sum((result ==label_id))},
    #         #     np.sum(dilated_binary_label_id){np.sum(dilated_binary_label_id)},
    #         #     np.sum(overlay){np.sum(overlay)},
    #         #     np.sum(binary_non_label){np.sum(binary_non_label)}
    #         #     """)
            
    #         if HAS_OVERLAY:
    #             dilated_binary_label_id[overlay] = False
        
    #     result[dilated_binary_label_id & threshold_binary] = label_id  
        
        ## Save the results if there is needed         
    
    return result

        
def grow_mp(**kwargs):
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
    step_size  = kwargs.get('step_size', 2) 
    
    
    if num_threads is None:
        num_threads = max_threads-1
    print(f"Parallel into {num_threads} threads")
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
    
    # Iterate through and make growth results
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
            
            ##
            
            ## Making grow for one iteration
            # result = sprout_core.dilation_one_iter(result, threshold_binary ,
            #                                 touch_rule = touch_rule,
            #                                 to_grow_ids=to_grow_ids)
            
        
            result = dilation_one_iter_mp(result, threshold_binary ,
                                          num_threads=num_threads,
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
    step_size  = kwargs.get('step_size', 2) 
    
    
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
    

    ############ Config
    file_path = './make_grow_result.yaml'
    
    _, extension = os.path.splitext(file_path)
    print(f"processing config he file {file_path}")
    if extension == '.json':
        
        load_config_json(file_path)
    elif extension == '.yaml':
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            to_grow_ids = config.get('to_grow_ids', None)
            num_threads = config.get('num_threads', None)
        load_config_yaml(config)
    
    
    # workspace = r'C:\Users\Yichen\OneDrive\work\codes\nhm_bounti_pipeline\result\procavia'
    
    # tif_file = os.path.join(workspace, 'thre_ero_seed/thre_ero_2iter_thre4500.tif')
    # # tif_file = os.path.join(workspace, 'test/input/thre_ero_2iter_thre4500.tif')
    
    # ori_path = os.path.join(workspace, 'input/procaviaH4981C_0001.tif.resampled_400_600.tif')
    
    # # ori_path = os.path.join(workspace, 'input/procaviaH4981C_0001.tif.resampled.tif')
      
    # ## Test by decrease the label values
    # # input_mask = np.where(np.isin(input_mask, [4,5,8,9,11,17,18,20 ]) , input_mask, 0)
    # # input_mask = np.where(np.isin(input_mask, [5,6,10,11]) , input_mask, 0)
    
    # workspace = r'C:\Users\Yichen\OneDrive\work\codes\nhm_bounti\result\foram_james'
    # ori_path = os.path.join(workspace, 'input/ai/final.20180802_VERSA_1905_ASB_OLK_st016_bl4_fo1_recon.tif')
    # tif_file = os.path.join(workspace, 'seed_test/seed_ero_3_thre0_segs_20.tif')
    # thresholds = [0]
    # dilate_iters = 4
    # touch_rule = "no"
    
    # grow_dict = main(        
    #     dilate_iters = dilate_iters,
    #     thresholds = thresholds,
    #     save_interval = save_interval,  
    #     touch_rule = touch_rule, 
        
    #     workspace = workspace,
    #     img_path = img_path,
    #     seg_path = seg_path,
    #     output_folder = output_folder,
    #     to_grow_ids = to_grow_ids
    #      )
    
    
    grow_dict = grow_mp(        
        dilate_iters = dilate_iters,
        thresholds = thresholds,
        save_interval = save_interval,  
        touch_rule = touch_rule, 
        
        workspace = workspace,
        img_path = img_path,
        seg_path = seg_path,
        output_folder = output_folder,
        to_grow_ids = to_grow_ids,
        num_threads = num_threads,
         )
    
    vis_lib.plot_grow(pd.read_csv(grow_dict['log_path']),
              grow_dict['log_path'] +".png"
              )
    
    
    

    
    # assert len(thresholds) == len(dilate_iters), f"thresholds and dilate_iters must have the same length, but got {len(thresholds)} and {len(dilate_iters)}."
     
    # img_path = os.path.join(workspace, 
    #                         img_path)
    # seg_path = os.path.join(workspace, seg_path)
    # base_name = os.path.basename(seg_path)
    # input_mask = tifffile.imread(seg_path)
    # ori_img = tifffile.imread(img_path)

    # output_folder = os.path.join(workspace, output_folder)
    # os.makedirs(output_folder , exist_ok=True)
    
    # # Record the start time
    # start_time = datetime.now()
    # print(f"""{start_time.strftime("%Y-%m-%d %H:%M:%S")}
    # Making growing for 
    #     Img: {img_path}
    #     Threshold for Img {thresholds}
    #     Seed {seg_path} to grow {dilate_iters} iterations
    #         """)
    
    # result = input_mask.copy()
    # for i, (threshold,dilate_iter) in enumerate(zip(thresholds,dilate_iters)):
        
    #     for i_dilate in range(1, dilate_iter+1):
    #         threshold_binary = ori_img > threshold

    #         result = sprout_core.dilation_one_iter(result, threshold_binary ,
    #                                           touch_rule = touch_rule)
            
            
    #         threshold_name = "_".join(str(s) for s in thresholds[:i+1])
    #         dilate_name = "_".join(str(s) for s in dilate_iters[:i+1])
            
    #         if dilate_iter%save_interval==0:
    #             # output_path = os.path.join(workspace, f'result/ai/dila_{dilate_iter}_{threshold}_rule_{touch_rule}.tif')
    #             output_path = os.path.join(output_folder, f'{base_name}_iter_{i_dilate}_dilate_{dilate_name}_thre_{threshold_name}.tif')
                
    #             print(f"Grown result has been saved {output_path}")
                
    #             tifffile.imwrite(output_path, 
    #             result,
    #             compression ='zlib')
    
    
    # end_time = datetime.now()
    # running_time = end_time - start_time
    # total_seconds = running_time.total_seconds()
    # minutes, _ = divmod(total_seconds, 60)
    # print(f"Running time:{minutes}")
