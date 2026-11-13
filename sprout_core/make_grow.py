# import nibabel as nib
import os, sys
import itertools
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from scipy import ndimage as ndi
from skimage.morphology import ball, disk
import tifffile
from datetime import datetime
import yaml
import numpy as np
import pandas as pd
import glob

import threading

import sprout_core.sprout_core as sprout_core 
from sprout_core.sprout_core import binary_dilation
import sprout_core.config_core as config_core 
import sprout_core.vis_lib as vis_lib

from multiprocessing import cpu_count
# Maximum threads for multiprocessing
max_threads = cpu_count()
lock = threading.Lock()


# Function to recursively create global variables from the config dictionary
            

def grow_function(result, threshold_binary, label_id_list,touch_rule,non_bg_mask, allowed=None, cancel=None):
    """
    Perform the growth operation for each label ID in the label ID list.

    Args:
        result (np.ndarray): The mask being grown.
        threshold_binary (np.ndarray): Binary mask threshold for the guide.
        label_id_list (list): List of label IDs to grow.
        touch_rule (str): Rule for handling overlaps (e.g., 'stop').
        cancel (threading.Event, optional): checked before every label; raises Cancelled.
    """
    for label_id in label_id_list:
        sprout_core.check_cancel(cancel)
        dilated_binary_label_id = (result ==label_id)
        if allowed is not None:
            # only label voxels inside the allowed region (threshold, not boundary) grow
            dilated_binary_label_id &= allowed
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

def _write_mask_stop(input_mask, label_id, bb, footprint, allowed=None):
    """Voxels that `label_id` writes in one 'stop' iteration, inside bounding box `bb`.

    Same as grow_function on its single-threaded path: dilate the label by one voxel,
    minus the voxels that belonged to ANOTHER label at the start of the iteration.
    It depends only on `input_mask` (the start-of-iteration state), not on what other
    labels write during the iteration, so labels can be computed in parallel.

    allowed: bool mask of bb's shape. Only label voxels inside it are dilated, so a
    seed voxel outside the threshold range or on the boundary cannot grow out (a seed
    sitting on a thin boundary or a dark gap used to grow to both sides).
    """
    sub = input_mask[bb]
    own = sub == label_id
    dilated = binary_dilation(own if allowed is None else own & allowed, footprint)
    taken_by_others = (sub != 0) & ~own
    return dilated & ~taken_by_others


def _compute_parallel_write_in_order(task, label_ids, num_threads, result, cancel=None):
    """Run task(label_id) -> (label_id, bb, write_mask) on a thread pool, but apply the
    writes to `result` strictly in the order of `label_ids`.

    With touch_rule='stop' this reproduces the single-threaded result exactly, for any
    number of threads: two labels competing for the same new voxel -> the one written
    later (higher id) keeps it, as in the sequential loop. A small window of pending
    tasks bounds the memory held by finished-but-not-yet-written masks.
    cancel (threading.Event, optional) is checked before every write: Cancelled is raised
    and the tasks not started yet are dropped.
    """
    workers = max(1, int(num_threads or 1))
    ids = iter(list(label_ids))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque(pool.submit(task, lid) for lid in itertools.islice(ids, 2 * workers))
        while pending:
            if sprout_core._is_cancelled(cancel):
                for f in pending:
                    f.cancel()
                raise sprout_core.Cancelled()
            label_id, bb, write = pending.popleft().result()
            result[bb][write] = label_id
            nxt = next(ids, None)
            if nxt is not None:
                pending.append(pool.submit(task, nxt))


def dilation_one_iter_mp(input_mask, threshold_binary,
                            num_threads,
                             touch_rule = 'stop',
                             to_grow_ids = None,
                             boundary = None,
                             cancel = None):
    """
    Perform one iteration of dilation by using grow_function(...) in multi-threading.

    Args:
        result (np.ndarray): The mask being grown.
        threshold_binary (np.ndarray): Binary mask threshold for the guide.
        num_threads (int): Number of threads to use.
        touch_rule (str): Rule for handling overlaps.
        to_grow_ids (list, optional): Specific IDs to grow. Defaults to None.
        boundary (np.ndarray, optional): Boundary mask to constrain growth. Defaults to None.
        cancel (threading.Event, optional): checked between labels; raises Cancelled.

    Returns:
        np.ndarray: The updated mask after one dilation iteration.
    """


    if touch_rule not in ('stop', 'overwrite'):
        raise ValueError(f"Unknown touch_rule: {touch_rule}. Must be 'stop' or 'overwrite'.")

    # One pass gives every label's bounding box (and which labels exist), instead of
    # np.unique plus three full-volume comparisons per label.
    objects = ndi.find_objects(input_mask)
    present = [i + 1 for i, s in enumerate(objects) if s is not None]
    if to_grow_ids is not None:
        label_id_list = np.intersect1d(present, to_grow_ids)
    else:
        label_id_list = present

    result = input_mask.copy()
    non_bg_mask = result != 0

    if touch_rule == 'stop':
        # Labels are computed in parallel from the start-of-iteration state and written in
        # ascending id order: deterministic and identical to the single-threaded result.
        # (The old version let threads read `result` while others were writing to it, so
        # contested voxels went to whichever thread was faster.)
        footprint = ball(1) if input_mask.ndim == 3 else disk(1)
        shape = input_mask.shape

        def task(label_id):
            box = objects[int(label_id) - 1]
            bb = tuple(slice(max(s.start - 1, 0), min(s.stop + 1, n)) for s, n in zip(box, shape))
            allowed = np.asarray(threshold_binary[bb], bool)
            if boundary is not None:
                allowed = allowed & np.logical_not(boundary[bb])
            return label_id, bb, _write_mask_stop(input_mask, label_id, bb, footprint, allowed)

        _compute_parallel_write_in_order(task, label_id_list, num_threads, result, cancel)
    else:
        # 'overwrite': a label may overwrite another label's original voxels, so each label
        # depends on the ones processed before it. Keep the sequential algorithm.
        allowed = np.asarray(threshold_binary, bool)
        if boundary is not None:
            allowed = allowed & np.logical_not(boundary)
        grow_function(result, threshold_binary, label_id_list, touch_rule, non_bg_mask, allowed, cancel)

    
    if boundary is not None:
        result[boundary] = 0

    result[(non_bg_mask ==False) & (threshold_binary==False)] = 0

    return result

        
def grow_mp(**kwargs):
    """
    Perform multi-step, multi-threaded seed growth for image segmentation.

    This function expands initial seed masks across an image based on threshold criteria
    and morphological dilation. It supports multi-step growth using different thresholds
    and dilation iterations, parallel processing, optional early stopping, and
    exporting of intermediate and final results.

    Parameters
    ----------
    img : np.ndarray, optional
        Input grayscale image. If not provided, `img_path` must be given.
    img_path : str, optional
        Path to the image file.
    seg : np.ndarray, optional
        Initial seed mask (segmentation). If not provided, `seg_path` must be given.
    seg_path : str, optional
        Path to the seed mask file.
    boundary : np.ndarray, optional
        Optional mask to constrain the growth.
    boundary_path : str, optional
        Path to the boundary mask file.
    workspace : str, optional
        Base path prepended to image, segmentation, boundary, and output paths.
    output_folder : str
        Directory to store intermediate and final results.
    final_grow_output_folder : str, optional
        Alternative directory to store the final grown result.
    base_name : str, optional
        Name prefix for result files. Defaults to inferred from image name.
    dilation_steps : int or list[int]
        Number of dilation iterations per step.
    thresholds : int or list[int]
        Lower threshold(s) to guide growth.
    upper_thresholds : int or list[int], optional
        Optional upper threshold(s) to constrain intensity range.
    touch_rule : str, default='stop'
        Rule for growth collision. Currently only 'stop' is supported.
    num_threads : int, optional
        Number of parallel threads to use. Defaults to half of available CPUs.
    save_every_n_iters : int or list[int], optional
        Interval(s) for saving intermediate results. Defaults to `dilation_steps`.
    grow_to_end : bool, default=False
        If True, grow until no further size increase or mask saturation.
    to_grow_ids : list[int], optional
        Subset of label IDs to grow. If None, all will be grown.
    is_sort : bool, default=False
        Whether to re-sort label IDs before saving the final result.
    min_growth_size : int, default=50
        Minimum increase in size to consider a growth step meaningful.
    no_growth_max_iter : int, default=3
        Stop early if no meaningful growth occurs for this many iterations.
    return_for_napari : bool, default=False
        If True, return result as a dictionary of arrays for napari visualization.
    use_simple_naming : bool, default=True
        If True, use a simple naming scheme for intermediate files.
    is_make_meshes : bool, optional
        If True, generate meshes for all intermediate and final results.
    downsample_scale : int, default=10
        Scale factor to downsample image before mesh generation.
    step_size : int, default=1
        Step size for marching cubes algorithm in mesh generation.
    cancel : threading.Event, optional
        Checked before every growth iteration and between labels; when set,
        sprout_core.Cancelled is raised (files already written stay). Not saved in the config.
    progress : callable(done, total, text), optional
        Called after every growth iteration (total = all iterations of all thresholds;
        a threshold that stops early counts as done). Not saved in the config.

    Returns
    -------
    grows_dict : dict
        Dictionary mapping names (str) to grown segmentation arrays (np.ndarray).
        Only returned if `return_for_napari=True`.
    log_dict : dict
        Dictionary containing:
            - "final_output_path": path to the final grown .tif file
            - "log_path": CSV file recording growth steps and sizes
            - "output_folder": directory containing all output files

    Notes
    -----
    - Intermediate results are saved with prefix `INTER_`, final result with `FINAL_GROW_`.
    - Growth is constrained by thresholds and optionally by a boundary mask.
    - Growth stops early if saturation is reached or no further meaningful change occurs.
    - If `is_make_meshes` is True, mesh (.ply) files will be generated using marching cubes.
    - The function logs and saves all configurations used for reproducibility.
    """
    # Stop / progress hooks (napari); popped so that the saved config stays as it was
    cancel = kwargs.pop('cancel', None)
    progress = kwargs.pop('progress', None)
    values = dict(kwargs)

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
    is_sort = kwargs.get('is_sort', False) 
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
        if boundary_path is not None:
            boundary_path = os.path.join(workspace, boundary_path)
   
    base_name = config_core.check_and_assign_base_name(base_name, img_path, "grown_result")

    
    # loading the image and segmentation mask

    
    img = config_core.check_and_load_data(img, img_path, "img")
    seg = config_core.check_and_load_data(seg, seg_path, "seg")
    boundary = config_core.check_and_load_data(boundary, boundary_path, "boundary", must_exist=False)
    

    
    
    config_core.valid_input_data(img, seg=seg, boundary=boundary)
    boundary = config_core.check_and_cast_boundary(boundary)
    
    output_folder = os.path.join(output_folder, base_name)
    config_core.warn_earlier_results(output_folder)
    os.makedirs(output_folder , exist_ok=True)
    if final_grow_output_folder is not None:
        os.makedirs(final_grow_output_folder, exist_ok=True)
    

    
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
            "to_grow_ids"  : to_grow_ids,
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
    # Pick the dtype from the largest label id, not from the number of labels:
    # ids {7, 300} used to be cast to uint8, turning 300 into 44 (and merging it
    # with an existing label 44). uint8 is still used whenever every id fits.
    result = sprout_core.cast_labels(seg, np.uint8)
    # Non-zero voxels of `result`, kept up to date: each iteration counts only its output
    result_size = np.count_nonzero(result)
    steps = sprout_core.StepProgress(progress, sum(dilation_steps))

    # Iterate through and make growth results
    for i, (threshold ,upper_threshold,dilate_iter) in enumerate(zip(thresholds , upper_thresholds,dilation_steps)):
        # Set the count for check diff for each growing threshold
        count_below_threshold = 0
        stage_end = sum(dilation_steps[:i+1])

        
        threshold_name = "_".join(str(s) for s in thresholds[:i+1])
        dilate_name = "_".join(str(s) for s in dilation_steps[:i+1])
        
        if upper_threshold is not None:
            threshold_binary = (img>=threshold) & (img<=upper_threshold)
        else:
            threshold_binary = img >= threshold

            
        full_size = np.sum(threshold_binary)
        print(f"Size of the threshold {threshold} to {upper_threshold} mask: {full_size}")
        for i_dilate in range(1, dilate_iter+1):
            sprout_core.check_cancel(cancel)

            # Get the input size for the log (the previous iteration's output size)
            input_size = result_size
            result = dilation_one_iter_mp(result, threshold_binary ,
                                          num_threads=num_threads,
                                            touch_rule = touch_rule,
                                            to_grow_ids=to_grow_ids,
                                            boundary=boundary,
                                            cancel=cancel)
            steps.step(f"threshold {threshold}: iteration {i_dilate}/{dilate_iter}")

            ## Check if output size and input 's diff is bigger than min_growth_size

            # Get the output size for the log
            output_size = result_size = np.count_nonzero(result)
            if output_size - input_size < min_growth_size:
                count_below_threshold += 1
            else:
                count_below_threshold = 0
                
            # Situations to save grow results 
            # When it ends:
            # 1. Reach the final iter, 
            # 2. Not been growing for sometime
            # 3. Grow to the size of the current threshold  
            if no_growth_max_iter is None:
                is_early_stop = False
            else:
                is_early_stop = (count_below_threshold >= no_growth_max_iter)
             
            if (i_dilate% save_every_n_iters[i]==0 or 
                i_dilate ==dilate_iter or 
                is_early_stop or
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
                
                # Do not sort intermediate results, as it will cause the ids to change
                # So will mess up the growing id.
                # result,_ = sprout_core.reorder_segmentation(result, sort_ids=is_sort)
                tifffile.imwrite(output_path,  result, compression ='zlib')
                if return_for_napari:
                    grows_dict[output_grow_name] =result
                
                print(f"\tGrown result has been saved {os.path.abspath(output_path)}")
                print(f"\tIter:{i_dilate}. Last Input size = {input_size} and Output_size = {output_size}")

                
                # Early stopping conditions (no_growth_max_iter=None: early stop is off)
                if is_early_stop:
                    print(f"\tNot growing for {no_growth_max_iter} iters\nBreaking at iteration {i_dilate} with Input size = {input_size} and Output_size = {output_size}")
                    break
                # If grow to end, check if the size is similar to the threshold size
                # If the size is similar to the threshold size, break
                if (grow_to_end == True and abs(full_size - output_size) < 0.05) :
                    print(f"\tGrow size is similar to the threshold size\nBreaking at iteration {i_dilate}: Input size = {input_size}, Output_size = {output_size} and size of threshold binary = {full_size}")
                    break
            
        print(f"\tFinish growing. Last Input size = {input_size} and Output_size = {output_size}\n")
        steps.jump(stage_end, f"threshold {threshold}: done")    # an early stop counts as done
    
    ## Save the final grow output as the final_<img_name>
    # only sort it in the end
    sprout_core.check_cancel(cancel)
    result,_ = sprout_core.reorder_segmentation(result, sort_ids=is_sort)
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

    config_core.save_used_config(output_folder, config_core.used_config(values, config_core.input_val_make_grow),
                                 "--grow")

    # Make meshes  
    if is_make_meshes:  
        try:
            from sprout_core.make_mesh import make_mesh_for_tiff
        except ImportError as e:
            raise ImportError("make_mesh module is not available due to import error.") from e
        
        tif_files = glob.glob(os.path.join(output_folder, '*.tif'))

        for tif_file in tif_files:
            sprout_core.check_cancel(cancel)
            make_mesh_for_tiff(tif_file,output_folder,
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
        
        save_every_n_iters = optional_params['save_every_n_iters'],  
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
    
    try:
        vis_lib.plot_grow(pd.read_csv(log_dict['log_path']), os.path.splitext(log_dict['log_path'])[0] + ".png")
    except ModuleNotFoundError:                   # matplotlib is not a dependency of sprout-seg
        print("No plot of the grow log: it needs matplotlib (pip install matplotlib).")

if __name__ == "__main__":
    
    # Get the file path from the first command-line argument or use the default
    if len(sys.argv) > 1:
        print(f"Reading config file from command-line argument: {sys.argv[1]}")
        file_path = sys.argv[1]
    else:
        print("No config file specified in arguments. Using default: ./template/make_grow.yaml")
        file_path = './template/make_grow.yaml'
    
    
    run_make_grow(file_path)
    
   
    