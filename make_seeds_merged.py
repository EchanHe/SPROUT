import numpy as np
import pandas as pd
import tifffile
# from PIL import Image
from tifffile import imread, imwrite
import os,sys
from datetime import datetime
import glob
import threading
lock = threading.Lock()
import time
import yaml


# Add the lib directory to the system path
import sprout_core.sprout_core as sprout_core
from sprout_core.sprout_core import reorder_segmentation



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

def detect_inter(ccomp_combine_seed,ero_seed, seed_ids, inter_log , lock):
    """
    Detect intersections between seeds and combined components.

    Args:
        ccomp_combine_seed (np.ndarray): Combined seed mask.
        ero_seed (np.ndarray): Eroded seed mask.
        seed_ids (list): List of seed IDs to process.
        inter_log (dict): Dictionary to log intersection results.
        lock (threading.Lock): Threading lock for shared resources.
    """
    
    for seed_id in seed_ids:
        seed_ccomp = ero_seed == seed_id
                    
        # Check if there are intersection using the sum of intersection
        inter = np.sum(ccomp_combine_seed[seed_ccomp])

        with lock:
            if inter>0:
                inter_log["inter_count"] +=1
                
                inter_log["inter_ids"] = np.append(inter_log["inter_ids"] , seed_id)
                
                prop = round(inter / np.sum(ccomp_combine_seed),4)*100
                inter_log["inter_props"] = np.append(inter_log["inter_props"], prop)


def make_seeds_merged_path_wrapper(img_path,
                              threshold,
                              output_folder,
                              n_iters,
                              segments,
                              boundary_path=None,
                              num_threads=1,
                              background=0,
                              sort=True,
                              no_split_limit=3,
                              min_size=5,
                              min_split_prop=0.01,
                              min_split_sum_prop=0,
                              save_every_iter=False,
                              save_merged_every_iter=False,
                              name_prefix="Merged_seed",
                              init_segments=None,
                              footprint="ball",
                              upper_threshold = None
                              ):
    """
    Wrapper for make_seeds_merged_mp that performs erosion-based merged seed generation with multi-threading.

    Args:
        img_path (str): Path to the input image.
        threshold (int): Threshold for segmentation. One value
        output_folder (str): Directory to save output seeds.
        n_iters (int): Number of erosion iterations.
        segments (int): Number of segments to extract.
        boundary_path (str, optional): Path to boundary image. Defaults to None.
        num_threads (int, optional): Number of threads to use. Defaults to 1.
        background (int, optional): Background value. Defaults to 0.
        sort (bool, optional): Whether to sort output segment IDs. Defaults to True.
        no_split_limit (int, optional): Early Stop Check: Limit for consecutive no-split iterations. Defaults to 3.
        min_size (int, optional): Minimum size for segments. Defaults to 5.
        min_split_prop (float, optional): Minimum proportion to consider a split. Defaults to 0.01.
        min_split_sum_prop (float, optional): Minimum proportion of (sub-segments from next step)/(current segments)
            to consider a split. Defaults to 0.
        save_every_iter (bool, optional): Save results at every iteration. Defaults to False.
        save_merged_every_iter (bool, optional): Save merged results at every iteration. Defaults to False.
        name_prefix (str, optional): Prefix for output file names. Defaults to "Merged_seed".
        init_segments (int, optional): Initial segments. Defaults to None.
        footprint (str, optional): Footprint shape for erosion. Defaults to "ball".
        

    Returns:
        tuple: Merged seeds, original combine ID map, and output dictionary.
    """
    # Read the image
    img = tifffile.imread(img_path)
    print(f"Loaded image from: {img_path}")
    
    # Read the boundary if provided
    boundary = None
    if boundary_path is not None:
        boundary = tifffile.imread(boundary_path)
    
    # Prepare values for printing
    start_time = datetime.now()
    values_to_print = {
        "Boundary Path": boundary_path if boundary_path else "None"
    }

    # Print detailed values
    print("Start time: " + start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Processing Image: {img_path}")
    for key, value in values_to_print.items():
        print(f"  {key}: {value}")
    
    
    # Call the original function
    seed ,ori_combine_ids_map , output_dict=make_seeds_merged_mp(img=img,
                        threshold=threshold,
                        output_folder=output_folder,
                        n_iters=n_iters,
                        segments=segments,
                        boundary=boundary,
                        num_threads=num_threads,
                        no_split_limit=no_split_limit,
                        min_size=min_size,
                        sort=sort,
                        min_split_prop=min_split_prop,
                        background=background,
                        save_every_iter=save_every_iter,
                        save_merged_every_iter=save_merged_every_iter,
                        name_prefix=name_prefix,
                        init_segments=init_segments,
                        footprint=footprint,
                        min_split_sum_prop=min_split_sum_prop,
                        upper_threshold = upper_threshold)


    end_time = datetime.now()
    running_time = end_time - start_time
    total_seconds = running_time.total_seconds()
    minutes, _ = divmod(total_seconds, 60)
    print(f"Running time:{minutes}")
    
    return seed ,ori_combine_ids_map , output_dict


def make_seeds_merged_mp(img,
                        threshold,
                        output_folder,
                        n_iters, 
                        segments,
                        boundary = None,
                        num_threads = 1,
                        background = 0,
                        sort = True,
                        no_split_limit =3,
                        min_size=5,
                        min_split_prop = 0.01,
                        min_split_sum_prop = 0,
                        save_every_iter = False,
                        save_merged_every_iter = False,
                        name_prefix = "Merged_seed",
                        init_segments = None,
                        footprint = "ball",
                        upper_threshold = None
                      ):
    """
    Erosion-based merged seed generation with multi-threading.

    Args:
        img (np.ndarray): Input image.
        threshold (int): Threshold for segmentation. One value
        output_folder (str): Directory to save output seeds.
        n_iters (int): Number of erosion iterations.
        segments (int): Number of segments to extract.
        boundary (np.ndarray, optional): Boundary mask. Defaults to None.
        num_threads (int, optional): Number of threads to use. Defaults to 1.
        background (int, optional): Background value. Defaults to 0.
        sort (bool, optional): Whether to sort output segment IDs. Defaults to True.
        no_split_limit (int, optional): Early Stop Check: Limit for consecutive no-split iterations. Defaults to 3.
        min_size (int, optional): Minimum size for segments. Defaults to 5.
        min_split_prop (float, optional): Minimum proportion to consider a split. Defaults to 0.01.
        min_split_sum_prop (float, optional): Minimum proportion of (sub-segments from next step)/(current segments)
            to consider a split. Defaults to 0. Range: 0 to 1
        save_every_iter (bool, optional): Save results at every iteration. Defaults to False.
        save_merged_every_iter (bool, optional): Save merged results at every iteration. Defaults to False.
        name_prefix (str, optional): Prefix for output file names. Defaults to "Merged_seed".
        init_segments (int, optional): Number of segments for the first seed, defaults is None.
            A small number of make the initial sepration faster, as normally the first seed only has a big one component
        footprint (str, optional): Footprint shape for erosion. Defaults to "ball".

    Returns:
        tuple: Merged seeds, original combine ID map, and output dictionary.
    """


    values_to_print = {
        "Threshold": threshold,
        "upper_threshold": upper_threshold,
        "Output Folder": output_folder,
        "Erosion Iterations": n_iters,
        "Segments": segments,
        "Number of Threads": num_threads,
        "No Split Limit for iters": no_split_limit,
        "Component Minimum Size": min_size,
        "Sort": sort,
        "Minimum Split Proportion": min_split_prop,
        "Background Value": background,
        "Save Every Iteration": save_every_iter,
        "Save Merged Every Iteration": save_merged_every_iter,
        "Name Prefix": name_prefix,
        "Footprint": footprint,
        "Minimum Split Sum Proportion": min_split_sum_prop
    }

    for key, value in values_to_print.items():
        print(f"  {key}: {value}")

    output_name = f"{name_prefix}_thre_{threshold}_{upper_threshold}_ero_{n_iters}"
    output_folder = os.path.join(output_folder, output_name)
    os.makedirs(output_folder,exist_ok=True)
    
    max_splits = segments
    
    
    if upper_threshold is not None:
        assert threshold<upper_threshold, "lower_threshold must be smaller than upper_threshold"
        img = (img>=threshold) & (img<=upper_threshold)
    else:
        img = img>=threshold
    # img = img<=threshold

    if boundary is not None:
        boundary = sprout_core.check_and_cast_boundary(boundary)
        img[boundary] = False

    if init_segments is None:
        init_segments = segments

    init_seed, _ = sprout_core.get_ccomps_with_size_order(img,init_segments)
    
    output_img_name = f'thre_{threshold}_{upper_threshold}_ero_0.tif'
    if save_every_iter:
        imwrite(os.path.join(output_folder,output_img_name), init_seed, 
            compression ='zlib')
    
    init_ids = [int(value) for value in np.unique(init_seed) if value != background]
    max_seed_id = int(np.max(init_ids))


    combine_seed = init_seed.copy()
    combine_seed = combine_seed.astype('uint16')


    output_dict = {"total_id": {0: len(np.unique(combine_seed))-1},
                   "split_id" : {0: {}},
                   "split_ori_id": {0: {}},
                   "split_ori_id_filtered":  {0: {}},
                   "split_prop":  {0: {}},
                   "cur_seed_name": {0: output_img_name},
                   "output_folder":output_folder
                   }

    ori_combine_ids_map = {}
    for value in init_ids :
        ori_combine_ids_map[value] = [value]
    
    # Count for no_split_limit
    no_consec_split_count = 0
    

    for ero_iter in range(1, n_iters+1):
        
        
        print(f"working on erosion {ero_iter}")
        
        output_dict["split_id"][ero_iter] = {}
        output_dict["split_ori_id"][ero_iter] = {}
        output_dict["split_ori_id_filtered"][ero_iter] = {}
        output_dict["split_prop"][ero_iter] = {}
        
        img = sprout_core.erosion_binary_img_on_sub(img, kernal_size = 1,footprint=footprint)
        seed, _ = sprout_core.get_ccomps_with_size_order(img,segments)
        seed = seed.astype('uint16')
        
        output_img_name = f'thre_{threshold}_{upper_threshold}_ero_{ero_iter}.tif'
        output_dict["cur_seed_name"][threshold] = output_img_name
        
        if save_every_iter:
            output_path = os.path.join(output_folder, output_img_name)
            print(f"\tSaving {output_path}")
            
            imwrite(output_path, seed, 
                compression ='zlib')

        seed_ids = [int(value) for value in np.unique(seed) if value != background]
        combine_ids = [int(value) for value in np.unique(combine_seed) if value != background]
       
        has_split = False
        ## Comparing each ccomp from eroded seed
        ## to each ccomp from the original seed
        
        inter_log = {
            "inter_count":0,
            "inter_ids": np.array([]),
            "inter_props": np.array([])
        }
        
        for combine_id in combine_ids:
            ccomp_combine_seed = combine_seed == combine_id
            
            inter_log["inter_count"] = 0
            inter_log["inter_ids"] = np.array([])
            inter_log["inter_props"] = np.array([])
            
            sublists = [seed_ids[i::num_threads] for i in range(num_threads)]
             # Create a list to hold the threads
            threads = []
            for sublist in sublists:
                thread = threading.Thread(target=detect_inter, args=(ccomp_combine_seed,
                                                                     seed,
                                                                     sublist,
                                                                     inter_log,
                                                                     lock))
                threads.append(thread)
                thread.start()
                
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
        
        
            ## if there are any intersection between seed and the init seed
            if inter_log["inter_count"]>1:
                temp_inter_count = inter_log["inter_count"]
                temp_inter_ids = inter_log["inter_ids"]
                temp_inter_props = inter_log["inter_props"]
                
                print(f'\t{combine_id} has been split to {temp_inter_count} parts. Ids are {temp_inter_ids}')
                print(f"\tprops are: {temp_inter_props}")
                
                sum_inter_props = np.sum(temp_inter_props)
                print(f"\tSplit prop is {sum_inter_props}")
                if sum_inter_props<min_split_sum_prop:
                    has_split = False
                else:
                    combine_seed[combine_seed == combine_id] =0
                    filtered_inter_ids = temp_inter_ids[temp_inter_props>min_split_prop]
                    
                    if len(filtered_inter_ids)>0:
                        has_split = True
                    
                    new_ids = []
                    for inter_id in filtered_inter_ids:
                        max_seed_id +=1
                        
                        combine_seed[seed == inter_id] = max_seed_id
                        # new_ids.append(max_seed_id)     
                        
                        new_ids.append(max_seed_id)
                        for key,value in ori_combine_ids_map.items():
                            if combine_id in value:
                                ori_combine_ids_map[key].append(max_seed_id)
                                break
                                # if len(value) <= max_splits:
                                    
                    output_dict["split_id"][ero_iter][combine_id] = new_ids
                    output_dict["split_ori_id"][ero_iter][combine_id] = inter_log["inter_ids"]
                    output_dict["split_ori_id_filtered"][ero_iter][combine_id] = filtered_inter_ids
                    output_dict["split_prop"][ero_iter][combine_id] = inter_log["inter_props"]                    
        
            output_dict["total_id"][ero_iter] = len(np.unique(combine_seed))-1
    
        if has_split:
            no_consec_split_count=0
        else:
            no_consec_split_count+=1
            
        
        if save_merged_every_iter:
            combine_seed,_ = reorder_segmentation(combine_seed, min_size=min_size, sort_ids=sort)
            output_path = os.path.join(output_folder,output_name+f'ero_{ero_iter}_sorted.tif')
            
            print(f"\tSaving final output:{output_path}")
            imwrite(output_path, combine_seed, 
                compression ='zlib')    
        
        if no_consec_split_count>=no_split_limit:
            print(f"detect non split for {no_consec_split_count}rounds")
            print(f"break loop at {ero_iter} iter")
            break
        

    # output_path = os.path.join(output_folder,output_name+'.tif')
    # print(f"\tSaving final output:{output_path}")
    # imwrite(output_path, combine_seed, 
    #     compression ='zlib')
    
    combine_seed,_ = reorder_segmentation(combine_seed, min_size=min_size, sort_ids=sort)
    output_path = os.path.join(output_folder,output_name+'_sorted.tif')
    print(f"\tSaving final output:{output_path}")
    imwrite(output_path, combine_seed, 
        compression ='zlib')
    
             
    return combine_seed,ori_combine_ids_map, output_dict    


def make_seeds_merged_by_thres_path_wrapper(img_path,
                                       thresholds,
                                       output_folder,
                                       n_iters,
                                       segments,
                                       num_threads=1,
                                       boundary_path=None,
                                       background=0,
                                       sort=True,
                                       
                                       no_split_limit=3,
                                       min_size=5,
                                       min_split_prop=0.01,
                                       min_split_sum_prop=0,
                                       
                                       save_every_iter=False,
                                       save_merged_every_iter=False,
                                       name_prefix="Merged_seed",
                                       init_segments=None,
                                       footprint="ball",
                                       
                                       upper_thresholds = None
                                       ):
    """
    Wrapper for make_seeds_merged_by_thres_mp that performs thresholds-based merged seed generation.

    Args:
        img_path (str): Path to the input image.
        thresholds (list): List of thresholds for segmentation.
        output_folder (str): Directory to save the output.
        n_iters (int): Number of erosion iterations.
        segments (int): Number of segments to extract.
        num_threads (int, optional): Number of threads for parallel processing. Defaults to 1.
        boundary_path (str, optional): Path to boundary image. Defaults to None.
        background (int, optional): Value of background pixels. Defaults to 0.
        sort (bool, optional): Whether to sort segment IDs. Defaults to True.
        
        no_split_limit (int, optional): Early Stop Check: Limit for consecutive no-split iterations. Defaults to 3.
        min_size (int, optional): Minimum size for segments. Defaults to 5.
        min_split_prop (float, optional): Minimum proportion to consider a split. Defaults to 0.01.
        min_split_sum_prop (float, optional): Minimum proportion of (sub-segments from next step)/(current segments)
            to consider a split. Defaults to 0.
            
        save_every_iter (bool, optional): Save results at every iteration. Defaults to False.
        save_merged_every_iter (bool, optional): Save merged results at every iteration. Defaults to False.
        name_prefix (str, optional): Prefix for output file names. Defaults to "Merged_seed".
        init_segments (int, optional): Number of segments for the first seed, defaults is None.
            A small number of make the initial sepration faster, as normally the first seed only has a big one component
        footprint (str, optional): Footprint shape for erosion. Defaults to "ball".


    Returns:
        tuple: Merged seed, original combine ID map, and output dictionary.
    """
    # Read the image
    img = tifffile.imread(img_path)
    print(f"Loaded image from: {img_path}")
    
    # Read the boundary if provided
    boundary = None
    if boundary_path is not None:
        boundary = tifffile.imread(boundary_path)
    
    # Prepare values for printing
    start_time = datetime.now()
    values_to_print = {
        "Boundary Path": boundary_path if boundary_path else "None"
    }

    # Print detailed values
    print("Start time: " + start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Processing Image: {img_path}")
    for key, value in values_to_print.items():
        print(f"  {key}: {value}")
    
    # Call the original function
    combine_seed,ori_combine_ids_map, output_dict  = make_seeds_merged_by_thres_mp(img=img,
                                  thresholds=thresholds,
                                  output_folder=output_folder,
                                  n_iters=n_iters,
                                  segments=segments,
                                  boundary=boundary,
                                  num_threads=num_threads,
                                  no_split_limit=no_split_limit,
                                  min_size=min_size,
                                  sort=sort,
                                  min_split_prop=min_split_prop,
                                  background=background,
                                  save_every_iter=save_every_iter,
                                  save_merged_every_iter=save_merged_every_iter,
                                  name_prefix=name_prefix,
                                  init_segments=init_segments,
                                  footprint=footprint,
                                  min_split_sum_prop=min_split_sum_prop,
                                  
                                  upper_thresholds = upper_thresholds)

    end_time = datetime.now()
    running_time = end_time - start_time
    total_seconds = running_time.total_seconds()
    minutes, _ = divmod(total_seconds, 60)
    print(f"Running time:{minutes}")
    
    return combine_seed,ori_combine_ids_map, output_dict

def make_seeds_merged_by_thres_mp(img,
                        thresholds,
                        output_folder,
                        n_iters, 
                        segments,
                        
                        num_threads = 1,
                        boundary =None,
                        background = 0,
                        sort = True,
                        
                        no_split_limit =3,
                        min_size=5,
                        min_split_prop = 0.01,
                        min_split_sum_prop = 0,
                        
                        save_every_iter = False,
                        save_merged_every_iter = False,
                        name_prefix = "Merged_seed",
                        init_segments = None,
                        footprint = "ball",
                        
                        upper_thresholds = None
                    ):
    """
    Thresholds-based merged seed generation.

    Args:
        img (np.ndarray): Input image.
        thresholds (list): List of thresholds for segmentation.
        output_folder (str): Directory to save the output.
        n_iters (int): Number of erosion iterations.
        segments (int): Number of segments to extract.
        num_threads (int, optional): Number of threads for parallel processing. Defaults to 1.
        boundary (np.ndarray, optional): Boundary mask. Defaults to None.
        background (int, optional): Value of background pixels. Defaults to 0.
        sort (bool, optional): Whether to sort segment IDs. Defaults to True.
        
        no_split_limit (int, optional): Early Stop Check: Limit for consecutive no-split iterations. Defaults to 3.
        min_size (int, optional): Minimum size for segments. Defaults to 5.
        min_split_prop (float, optional): Minimum proportion to consider a split. Defaults to 0.01.
        min_split_sum_prop (float, optional): Minimum proportion of (sub-segments from next step)/(current segments)
            to consider a split. Defaults to 0.
            
        save_every_iter (bool, optional): Save results at every iteration. Defaults to False.
        save_merged_every_iter (bool, optional): Save merged results at every iteration. Defaults to False.
        name_prefix (str, optional): Prefix for output file names. Defaults to "Merged_seed".
        init_segments (int, optional): Number of segments for the first seed, defaults is None.
            A small number of make the initial sepration faster, as normally the first seed only has a big one component
        footprint (str, optional): Footprint shape for erosion. Defaults to "ball".


    Returns:
        tuple: Merged seed, original combine ID map, and output dictionary.
    """


    values_to_print = {
        "Thresholds": thresholds,
        "upper_thresholds": upper_thresholds,
        "Output Folder": output_folder,
        "Erosion Iterations": n_iters,
        "Segments": segments,
        "Number of Threads": num_threads,
        "No Split Limit for iters": no_split_limit,
        "Component Minimum Size": min_size,
        "Sort": sort,
        "Minimum Split Proportion": min_split_prop,
        "Background Value": background,
        "Save Every Iteration": save_every_iter,
        "Save Merged Every Iteration": save_merged_every_iter,
        "Name Prefix": name_prefix,
        "Footprint": footprint,
        "Minimum Split Sum Proportion": min_split_sum_prop
    }

    for key, value in values_to_print.items():
        print(f"  {key}: {value}")


    output_name = f"{name_prefix}_ero_{n_iters}"
    output_folder = os.path.join(output_folder, output_name)
    os.makedirs(output_folder,exist_ok=True)
    
    max_splits = segments
    
    if init_segments is None:
        init_segments = segments

    if upper_thresholds is not None:
        assert len(thresholds) == len(upper_thresholds), "lower_thresholds and upper_thresholds should have the same length"
        assert thresholds[0]<upper_thresholds[0], "lower_threshold must be smaller than upper_threshold"
        mask = (img>=thresholds[0]) & (img<=upper_thresholds[0])
    else:
        mask = img>=thresholds[0]
    
    
    if boundary is not None:
        boundary = sprout_core.check_and_cast_boundary(boundary)
        img[boundary] = False
    
    for ero_iter in range(1, n_iters+1):
        mask = sprout_core.erosion_binary_img_on_sub(mask, kernal_size = 1,footprint=footprint)
    init_seed, _ = sprout_core.get_ccomps_with_size_order(mask,init_segments)
    
    output_img_name = f'thre_{thresholds[0]}_ero_{n_iters}.tif'
    if save_every_iter:
        imwrite(os.path.join(output_folder,output_img_name), init_seed, 
            compression ='zlib')
            
    
    init_ids = [int(value) for value in np.unique(init_seed) if value != background]
    max_seed_id = int(np.max(init_ids))


    combine_seed = init_seed.copy()
    combine_seed = combine_seed.astype('uint16')

    output_dict = {"total_id": {0: len(np.unique(combine_seed))-1},
                   "split_id" : {0: {}},
                   "split_ori_id": {0: {}},
                   "split_ori_id_filtered":  {0: {}},
                   "split_prop":  {0: {}}, 
                   "cur_seed_name": {0: output_img_name},
                   "output_folder":output_folder
                   }

    ori_combine_ids_map = {}
    for value in init_ids :
        ori_combine_ids_map[value] = [value]
    
    no_consec_split_count = 0
    

    for idx_threshold, threshold in enumerate(thresholds[1:]):
        print(f"working on thre {threshold}")
        
        output_dict["split_id"][threshold] = {}
        output_dict["split_ori_id"][threshold] = {}
        output_dict["split_ori_id_filtered"][threshold] = {}
        output_dict["split_prop"][threshold] = {}
        
        
        if upper_thresholds is not None:
            assert threshold<upper_thresholds[1 + idx_threshold], "lower_threshold must be smaller than upper_threshold"
            mask = (img>=threshold) & (img<=upper_thresholds[1 + idx_threshold])
        else:
            mask = img>=threshold

        # mask = img>=threshold
        
        for ero_iter in range(1, n_iters+1):
            mask = sprout_core.erosion_binary_img_on_sub(mask, kernal_size = 1,footprint=footprint)
        seed, _ = sprout_core.get_ccomps_with_size_order(mask,segments)
        seed = seed.astype('uint16')
        
        output_img_name = f'thre_{threshold}_ero_{n_iters}.tif'
        output_dict["cur_seed_name"][threshold] = output_img_name
        if save_every_iter:
            output_path = os.path.join(output_folder,output_img_name)
            print(f"\tSaving {output_path}")
            imwrite(output_path, seed, compression ='zlib')
        
        seed_ids = [int(value) for value in np.unique(seed) if value != background]
        combine_ids = [int(value) for value in np.unique(combine_seed) if value != background]
        
        
        has_split = False
        ## Comparing each ccomp from eroded seed
        ## to each ccomp from the original seed
        
        inter_log = {
            "inter_count":0,
            "inter_ids": np.array([]),
            "inter_props": np.array([])
        }
        
        for combine_id in combine_ids:
            ccomp_combine_seed = combine_seed == combine_id
            
            inter_log["inter_count"] = 0
            inter_log["inter_ids"] = np.array([])
            inter_log["inter_props"] = np.array([])
            
            sublists = [seed_ids[i::num_threads] for i in range(num_threads)]
             # Create a list to hold the threads
            threads = []
            for sublist in sublists:
                thread = threading.Thread(target=detect_inter, args=(ccomp_combine_seed,
                                                                     seed,
                                                                     sublist,
                                                                     inter_log,
                                                                     lock))
                threads.append(thread)
                thread.start()
                
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
        
        
            ## if there are any intersection between seed and the init seed
            if inter_log["inter_count"]>1:
                temp_inter_count = inter_log["inter_count"]
                temp_inter_ids = inter_log["inter_ids"]
                temp_inter_props = inter_log["inter_props"]
                
                
                print(f'\t{combine_id} has been split to {temp_inter_count} parts. Ids are {temp_inter_ids}')
                print(f"\tprops are: {temp_inter_props}")
                
                sum_inter_props = np.sum(temp_inter_props)
                print(f"\tSplit prop is {sum_inter_props}")
                
                if sum_inter_props<min_split_sum_prop:
                    has_split = False
                else:
                    combine_seed[combine_seed == combine_id] =0
                    filtered_inter_ids = temp_inter_ids[temp_inter_props>min_split_prop]
                    
                
                    if len(filtered_inter_ids)>0:
                        has_split = True
                    
                    new_ids = []
                    for inter_id in filtered_inter_ids:
                        max_seed_id +=1
                        
                        combine_seed[seed == inter_id] = max_seed_id
                        # new_ids.append(max_seed_id)     
                        
                        new_ids.append(max_seed_id)
                        for key,value in ori_combine_ids_map.items():
                            if combine_id in value:
                                ori_combine_ids_map[key].append(max_seed_id)
                                break
                                # if len(value) <= max_splits:
                                    
                    output_dict["split_id"][threshold][combine_id] = new_ids
                    output_dict["split_ori_id"][threshold][combine_id] = inter_log["inter_ids"]
                    output_dict["split_ori_id_filtered"][threshold][combine_id] = filtered_inter_ids
                    output_dict["split_prop"][threshold][combine_id] = inter_log["inter_props"]
                
        
            output_dict["total_id"][threshold] = len(np.unique(combine_seed))-1
    
        if has_split:
            no_consec_split_count=0
        else:
            no_consec_split_count+=1
            
        if no_consec_split_count>=no_split_limit:
                print(f"detect non split for {no_consec_split_count}rounds")
                print(f"break loop at {threshold} threshold")
                break
        

    
    # output_path = os.path.join(output_folder,output_name+'.tif')
    # print(f"\tSaving final output:{output_path}")
    # imwrite(output_path, combine_seed, 
    #     compression ='zlib')
    
    
    if save_merged_every_iter:
        combine_seed,_ = reorder_segmentation(combine_seed, min_size=min_size, sort_ids=sort)
        output_path = os.path.join(output_folder,output_name+f'ero_{ero_iter}_sorted.tif')
        
        print(f"\tSaving final output:{output_path}")
        imwrite(output_path, combine_seed, 
            compression ='zlib')    
        
    combine_seed,_ = reorder_segmentation(combine_seed, min_size=min_size, sort_ids=sort)
    output_path = os.path.join(output_folder,output_name+'_sorted.tif')
    print(f"\tSaving final output:{output_path}")
    imwrite(output_path, combine_seed, 
        compression ='zlib')
    
             
    return combine_seed,ori_combine_ids_map, output_dict  


if __name__ == "__main__":        
   
    # Get the file path from the first command-line argument or use the default
    file_path = sys.argv[1] if len(sys.argv) > 1 else './make_seeds_merged.yaml'

    
    _, extension = os.path.splitext(file_path)
    print(f"processing config the file {file_path}")
    if extension == '.yaml':
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            
            upper_thresholds = config.get('upper_thresholds', None)
            
            boundary_path = config.get('boundary_path', None)
            background = config.get('background', 0)
            sort = config.get('sort', True)
            
            no_split_limit = config.get('no_split_limit', 3)
            min_size = config.get('min_size', 5)
            min_split_prop = config.get('min_split_prop', 0.01)
            min_split_sum_prop = config.get('min_split_sum_prop', 0)
            
            save_every_iter = config.get('save_every_iter', False)
            save_merged_every_iter = config.get('save_merged_every_iter', False)
            name_prefix = config.get('name_prefix', "Merged_seed")
            init_segments = config.get('init_segments', None)
            footprints = config.get('footprints', "ball")
        
        load_config_yaml(config)
        
        
    
    if isinstance(thresholds, int):
        seed_merging_mode = "ERO"
    elif isinstance(thresholds, list):
        if all(isinstance(t, int) for t in thresholds):
            if len(thresholds) == 1:
                seed_merging_mode = "ERO"
                thresholds = thresholds[0]
                if upper_thresholds is not None:
                    upper_thresholds = upper_thresholds[0]
                
            elif len(thresholds) > 1:
                seed_merging_mode = "THRE"
        else:
            raise ValueError("'thresholds' must be an int or a list of int(s).")
    
    
    # Track the overall start time
    overall_start_time = time.time()
    
    
    
    img = imread(img_path)
    name_prefix = os.path.basename(img_path)
    
    if boundary_path is not None:
        boundary = imread(boundary_path)
    else:
        boundary = None
    
    if seed_merging_mode == "ERO":
        seed ,ori_combine_ids_map , output_dict=  make_seeds_merged_mp(img,
                                            thresholds,
                                            output_folder,
                                            n_iters, 
                                            segments,
                                            boundary = boundary,
                                            num_threads = num_threads,
                                            background = background,
                                            sort = sort,
                                            no_split_limit =no_split_limit,
                                            min_size=min_size,
                                            min_split_prop = min_split_prop,
                                            min_split_sum_prop = min_split_sum_prop,
                                            save_every_iter = save_every_iter,
                                            save_merged_every_iter = save_merged_every_iter,
                                            name_prefix = name_prefix,
                                            init_segments = init_segments,
                                            footprint = footprints,
                                            upper_threshold = upper_thresholds)
    
    # pd.DataFrame(ori_combine_ids_map).to_csv(os.path.join(output_folder, 'ori_combine_ids_map.csv'),index=False)
    
    elif seed_merging_mode=="THRE":
   
        seed ,ori_combine_ids_map , output_dict= make_seeds_merged_by_thres_mp(img,
                                    thresholds,
                                    output_folder,
                                    n_iters, 
                                    segments,
                                    num_threads = num_threads,
                                    
                                    boundary = boundary,
                                    background = background,
                                    sort = sort,
                                    no_split_limit =no_split_limit,
                                    min_size=min_size,
                                    min_split_prop = min_split_prop,
                                    min_split_sum_prop = min_split_sum_prop,
                                    save_every_iter = save_every_iter,
                                    save_merged_every_iter = save_merged_every_iter,
                                    name_prefix = name_prefix,
                                    init_segments = init_segments,
                                    footprint = footprints,
                                    upper_thresholds = upper_thresholds)
                

    
    
    pd.DataFrame(output_dict).to_csv(os.path.join(output_folder, 'output_dict.csv'),index=False)
    # Track the overall end time
    overall_end_time = time.time()

    # Output the total running time
    print(f"Total running time: {overall_end_time - overall_start_time:.2f} seconds")
        

