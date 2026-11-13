import numpy as np
import pandas as pd

# from PIL import Image
from tifffile import imread, imwrite
import os,sys

import time
import yaml
from scipy.spatial import ConvexHull, QhullError

# Add the lib directory to the system path
import sprout_core.sprout_core as sprout_core
import sprout_core.config_core as config_core
from sprout_core.sprout_core import reorder_segmentation
from multiprocessing import cpu_count
max_threads = cpu_count()


def describe_boundary(boundary, boundary_path):
    """Human-readable boundary for logs. In napari the boundary is an array with no path."""
    if boundary_path:
        return boundary_path
    if boundary is not None:
        return f"array {boundary.shape}, {int(np.count_nonzero(boundary)):,} boundary voxels"
    return "None"


def split_size_check(mask, split_size_limit):
    split_size_lower = split_size_limit[0]
    split_size_upper = split_size_limit[1]
    split_size_condition = True
    if split_size_lower is not None or split_size_upper is not None:
        split_size_temp = np.sum(mask)
        if (split_size_lower is not None) and (split_size_temp < split_size_lower): 
            split_size_condition = False 
        if (split_size_upper is not None) and (split_size_temp > split_size_upper):
            split_size_condition = False 
    
    return split_size_condition

def split_convex_hull_check(mask, split_convex_hull_limit):
    lower = split_convex_hull_limit[0]
    upper = split_convex_hull_limit[1]
    split_size_condition = True
    if lower is not None or upper is not None:
        # split_size_temp = np.sum(mask)
        coords = np.column_stack(np.where(mask))
        if len(coords) >= 4:  # Convex hull needs at least 4 points in 3D
            try:
                hull = ConvexHull(coords)
                convex_hull_volume = int(hull.volume)
            except QhullError:      # all voxels in one plane or line: no volume
                convex_hull_volume = 0
        else:
            return False
        if (lower is not None) and (convex_hull_volume < lower):
            split_size_condition = False
            print(f"no split: convex hull volume {convex_hull_volume} is below the lower limit {lower}")
        if (upper is not None) and (convex_hull_volume > upper):
            split_size_condition = False
            print(f"no split: convex hull volume {convex_hull_volume} is above the upper limit {upper}")
        
    return split_size_condition

def detect_inter(ccomp_combine_seed, ero_seed, inter_log, background=0):
    """
    Detect the seeds inside a combined component, in one pass and in ascending id order.

    Args:
        ccomp_combine_seed (np.ndarray): Combined seed mask.
        ero_seed (np.ndarray): Eroded seed mask.
        inter_log (dict): Dictionary to log intersection results.
        background (int, optional): Background value. Defaults to 0.
    """
    inter_counts = np.bincount(ero_seed[ccomp_combine_seed])
    inter_ids = np.flatnonzero(inter_counts)
    inter_ids = inter_ids[inter_ids != background]
    region_size = np.sum(ccomp_combine_seed)
    inter_log["inter_count"] = inter_ids.size
    inter_log["inter_ids"] = inter_ids.astype(float)
    inter_log["inter_props"] = np.array([round(c / region_size, 6) * 100 for c in inter_counts[inter_ids]])
    inter_log["inter_sizes"] = inter_counts[inter_ids]

def save_seed(seed, output_folder, output_name, return_for_napari=False, seeds_dict=None):
    """
    Save the seed to a specified folder.

    Args:
        seed (np.ndarray): Seed data to save.
        output_folder (str): Folder to save the seed.
        output_name (str): Name of the output file.
        return_for_napari (bool, optional): Whether to save in napari format. Defaults to False.
        seeds_dict (dict, optional): Dictionary to store seeds for napari. Defaults to None.
    """
    output_path = os.path.join(output_folder, output_name)
    print(f"\tSaving {os.path.abspath(output_path)}")
    imwrite(output_path, seed, compression='zlib')
    
    if return_for_napari:
        seeds_dict[output_name] = seed



class _History:
    """What adaptive seed did, for adaptive_seed_regions.csv and adaptive_seed_steps.csv.

    Only records: the seeds are the same with or without it. `setting` is the setting of a
    step that changes from step to step: {"erosion": n}, or {"threshold": t, "upper": u}.
    """

    def __init__(self, ids, setting):
        self.seeds = {int(i): dict(first_step=0, **setting, began="first step") for i in ids}
        self.steps = [dict(step=0, **setting, regions=len(self.seeds), split=0,
                           new_seeds=len(self.seeds), pruned=0, seeds_after=len(self.seeds), note="")]
        self.setting = setting

    def start_step(self, step, setting, regions):
        self._close()
        self.setting = setting
        self.steps.append(dict(step=step, **setting, regions=int(regions), split=0,
                               new_seeds=0, pruned=0, seeds_after=None, note=""))

    def replaced(self, old_id, new_ids):
        """A region of the running seed was split into these parts (two or more) in this step."""
        row, n = self.steps[-1], len(new_ids)
        self.seeds.pop(int(old_id), None)
        row["split"] += 1
        row["new_seeds"] += n
        for k, new_id in enumerate(new_ids, start=1):
            self.seeds[int(new_id)] = dict(first_step=row["step"], **self.setting, began=f"separated ({k} of {n})")

    def renumbered(self, class_mapping):
        """The running seed was pruned and renumbered (prune_every_iter)."""
        before = len(self.seeds)
        self.seeds = {int(new): self.seeds[int(old)] for old, new in class_mapping.items() if int(old) in self.seeds}
        self.steps[-1]["pruned"] = before - len(self.seeds)

    def stopped(self, steps_without_split):
        self.steps[-1]["note"] = f"stopped: no split for {steps_without_split} steps"

    def _close(self):
        if self.steps[-1]["seeds_after"] is None:
            self.steps[-1]["seeds_after"] = len(self.seeds)

    def save(self, output_folder, final_seed, class_mapping):
        """The two CSV files, for the final seed and its renumbering from the running seed."""
        self._close()
        areas = sprout_core._bincount(final_seed, int(final_seed.max()) + 1)
        rows = [dict(id=int(new), area=int(areas[int(new)]), **self.seeds[int(old)])
                for old, new in class_mapping.items() if int(old) in self.seeds]
        columns = ["id", "area", "first_step", *self.setting, "began"]
        pd.DataFrame(sorted(rows, key=lambda row: row["id"]), columns=columns).to_csv(
            os.path.join(output_folder, "adaptive_seed_regions.csv"), index=False)
        pd.DataFrame(self.steps, columns=["step", *self.setting, "regions", "split", "new_seeds",
                                          "pruned", "seeds_after", "note"]).to_csv(
            os.path.join(output_folder, "adaptive_seed_steps.csv"), index=False)


def make_adaptive_seed_ero(
                         
                        threshold,
                        output_folder,
                        erosion_steps, 
                        segments,
                        
                        img = None,
                        img_path = None,
                        
                        boundary = None,
                        boundary_path = None,
                        
                        num_threads = None,
                        background = 0,
                        sort = True,
                        
                        no_split_max_iter =3,
                        min_size=1,
                        min_split_ratio = 0,
                        min_split_total_ratio = 0,
                        
                        save_every_iter = True,
                        prune_every_iter = None,

                        base_name = None,
                        
                        init_segments = None,
                        last_segments = None,
                        
                        footprints = None,
                        upper_threshold = None,
                        split_size_limit = (None,None),
                        split_convex_hull_limit = (None, None),
                        return_for_napari = False,
                        save_final_for_batch = False,
                        cancel = None,
                        progress = None
                      ):
    """
    Erosion-based merged seed generation with multi-threading.

    Args:
        img (np.ndarray): Input image.
        threshold (int): Threshold for segmentation. One value
        output_folder (str): Directory to save output seeds.
        erosion_steps (int): Number of erosion iterations.
        segments (int): Number of segments to extract.
        boundary (np.ndarray, optional): Boundary mask. Defaults to None.
        num_threads (int, optional): Number of threads to use. Defaults to 1.
        background (int, optional): Background value. Defaults to 0.
        sort (bool, optional): Whether to sort output segment IDs. Defaults to True.
        no_split_max_iter (int, optional): Early Stop Check: Limit for consecutive no-split iterations. Defaults to 3.
        min_size (int, optional): Minimum size for segments. Defaults to 1.
        min_split_ratio (float, optional): Minimum proportion to consider a split. Defaults to 0.
        min_split_total_ratio (float, optional): Minimum proportion of (sub-segments from next step)/(current segments)
            to consider a split. Defaults to 0. Range: 0 to 1
        save_every_iter (bool, optional): Save results at every iteration. Defaults to True.
        prune_every_iter (bool, optional): After every intermediate iteration, remove regions
            smaller than min_size, keep the top-N and renumber the running seed. This changes
            the final result. None (default) = same as save_every_iter, which reproduces the
            behaviour of earlier versions where saving implied pruning.

        name_prefix (str, optional): Prefix for output file names. Defaults to "Merged_seed".
        init_segments (int, optional): Number of segments for the first seed, defaults is None.
            A small number of make the initial sepration faster, as normally the first seed only has a big one component
        footprints (str, optional): Footprints shape for erosion. Defaults to None.
        split_size_limit (optional): create a split if the region size (np.sum(mask)) is within the limit
        split_convex_hull_limit: create a split if the the convex hull's area/volume is within the limit
        cancel (threading.Event, optional): checked before every erosion step and between regions;
            when set, sprout_core.Cancelled is raised (files already written stay).
        progress (callable(done, total, text), optional): called after the initial seed and
            after every erosion step.

    Returns:
        tuple: Merged seeds, original combine ID map, and output dictionary.
    """
    values = dict(locals())                       # the arguments, for used_config.yaml
    
    img = config_core.check_and_load_data(img, img_path, "img")

    boundary = config_core.check_and_load_data(boundary, boundary_path, "boundary", must_exist=False)
    config_core.valid_input_data(img, boundary=boundary)    
    
    
    min_split_ratio = min_split_ratio*100
    min_split_total_ratio = min_split_total_ratio*100

    if num_threads is None:
        num_threads = max(1, max_threads // 2)

    if num_threads>=max_threads:
        num_threads = max(1,max_threads-1)



    
    segments_list = config_core.check_and_assign_segment_list(segments, init_segments,  
                                                              last_segments, erosion_steps=erosion_steps)
    
    base_name = config_core.check_and_assign_base_name(base_name, img_path, "adapt_seed")

    root_output_folder = output_folder
    output_folder = os.path.join(output_folder , base_name)

    # if sub_folder is None:
    #     output_folder = os.path.join(output_folder, output_name)
    # else:
    #     output_folder = os.path.join(output_folder, sub_folder)
    output_folder = os.path.abspath(output_folder)
    config_core.warn_earlier_results(output_folder)
    os.makedirs(output_folder,exist_ok=True)


    footprint_list = config_core.check_and_assign_footprint(footprints, erosion_steps)
    threshold,upper_threshold = config_core.check_and_assign_threshold(threshold, upper_threshold)

    # None keeps the legacy behaviour: saving intermediate results also pruned the
    # running seed, so the final result depended on save_every_iter.
    if prune_every_iter is None:
        prune_every_iter = save_every_iter

    values_to_print = {
        "img_path": img_path,
        "Threshold": threshold,
        "upper_threshold": upper_threshold,
        "Output Folder": output_folder,
        "Erosion Iterations": erosion_steps,
        "Segments": segments_list,
        "Segments for the first output": init_segments,
        "Segments for the final result": last_segments,
        "boundary_path": boundary_path,
        "Boundary": describe_boundary(boundary, boundary_path),
        "Number of Threads": num_threads,
        "Sort": sort,
        "Background Value": background,
        "Save Every Iteration": save_every_iter,
        "Prune Every Iteration": prune_every_iter,
        "Footprints": footprint_list,
        "No Split Limit for iters": no_split_max_iter,
        "Component Minimum Size": min_size,
        "Minimum Split Proportion (%)": min_split_ratio,
        "Minimum Split Sum Proportion (%)": min_split_total_ratio,
        "split_size_limit": split_size_limit,
        "split_convex_hull_limit": split_convex_hull_limit
    }

    for key, value in values_to_print.items():
        print(f"  {key}: {value}")

    
    

    
    
    
    if upper_threshold is not None:
        img = (img>=threshold) & (img<=upper_threshold)
    else:
        img = img>=threshold


    if boundary is not None:
        boundary = config_core.check_and_cast_boundary(boundary)
        img[boundary] = False


    steps = sprout_core.StepProgress(progress, erosion_steps + 1)
    sprout_core.check_cancel(cancel)
    init_seed, _ = sprout_core.get_ccomps_with_size_order(img,segments_list[0])
    steps.step("initial seed")
    
    seeds_dict = {}
    
    output_img_name = f'INTER_thre_{threshold}_{upper_threshold}_ero_0.tif'
    if save_every_iter:
        save_seed(init_seed, output_folder, output_img_name, return_for_napari=return_for_napari, seeds_dict=seeds_dict)      
        
    
    init_ids = [int(value) for value in sprout_core.unique_labels(init_seed) if value != background]
    if not init_ids:
        raise RuntimeError("No components found from the initial seeds. Exiting.")
    
    max_seed_id = int(np.max(init_ids))
    history = _History(init_ids, {"erosion": 0})


    combine_seed = init_seed.copy()
    # at least uint16 as before; new ids from splits are checked with ensure_label_capacity
    combine_seed = sprout_core.cast_labels(combine_seed, np.uint16)


    output_dict = {"total_id": {0: len(sprout_core.unique_labels(combine_seed))-1},
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
    
    # Count for no_split_max_iter
    no_consec_split_count = 0
    

    for ero_iter in range(1, erosion_steps+1):
        
        
        print(f"working on erosion {ero_iter}")
        sprout_core.check_cancel(cancel)
        
        output_dict["split_id"][ero_iter] = {}
        output_dict["split_ori_id"][ero_iter] = {}
        output_dict["split_ori_id_filtered"][ero_iter] = {}
        output_dict["split_prop"][ero_iter] = {}
        
        img = sprout_core.erosion_binary_img_on_sub(img, kernal_size = 1,footprint=footprint_list[ero_iter-1])
        sprout_core.check_cancel(cancel)
        seed, _ = sprout_core.get_ccomps_with_size_order(img, segments_list[ero_iter])
        seed = sprout_core.cast_labels(seed, np.uint16)
        history.start_step(ero_iter, {"erosion": ero_iter}, seed.max())
        

        
        if save_every_iter:
            output_img_name = f'INTER_thre_{threshold}_{upper_threshold}_ero_{ero_iter}.tif'
            output_dict["cur_seed_name"][ero_iter] = output_img_name
            save_seed(seed, output_folder, output_img_name, return_for_napari=return_for_napari, seeds_dict=seeds_dict)



        combine_ids = [int(value) for value in sprout_core.unique_labels(combine_seed) if value != background]
       
        has_split = False
        ## Comparing each ccomp from eroded seed
        ## to each ccomp from the original seed
        
        inter_log = {
            "inter_count":0,
            "inter_ids": np.array([]),
            "inter_props": np.array([])
        }
        
        for combine_id in combine_ids:
            sprout_core.check_cancel(cancel)
            ccomp_combine_seed = combine_seed == combine_id
            
            detect_inter(ccomp_combine_seed, seed, inter_log, background)
        
        
            # The parts that count: at least min_size voxels and more than min_split_ratio of the region.
            # A region splits into two or more of them; else it stays as it is (specks do not split it).
            counted = (inter_log["inter_sizes"] >= (min_size or 0)) & (inter_log["inter_props"] > min_split_ratio)
            if np.count_nonzero(counted) > 1:
                temp_inter_count = inter_log["inter_count"]
                temp_inter_ids = inter_log["inter_ids"]
                temp_inter_props = inter_log["inter_props"]
                
                # print(f'\t{combine_id} has been split to {temp_inter_count} parts. Ids are {temp_inter_ids}')
                # print(f"\tprops are: {temp_inter_props}")
                
                sum_inter_props = np.sum(temp_inter_props)
                
                
                
                if not split_size_check(ccomp_combine_seed, split_size_limit):
                    print(f"no split for region {combine_id}: size {int(np.sum(ccomp_combine_seed))} is outside split_size_limit {split_size_limit}")

                split_condition =  (sum_inter_props>=min_split_total_ratio) and\
                    split_size_check(ccomp_combine_seed, split_size_limit) and \
                        split_convex_hull_check(ccomp_combine_seed, split_convex_hull_limit)
                
                # print(f"\tSplit prop is {sum_inter_props}")
                if split_condition:
                    combine_seed[combine_seed == combine_id] =0
                    filtered_inter_ids = temp_inter_ids[counted]
                    has_split = True
                    
                    new_ids = []
                    for inter_id in filtered_inter_ids:
                        max_seed_id +=1
                        combine_seed = sprout_core.ensure_label_capacity(combine_seed, max_seed_id)
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
                    history.replaced(combine_id, new_ids)
        
        if combine_ids:
            output_dict["total_id"][ero_iter] = len(sprout_core.unique_labels(combine_seed))-1
    
        if has_split:
            no_consec_split_count=0
        else:
            no_consec_split_count+=1
        steps.step(f"erosion {ero_iter}/{erosion_steps}")
            
        
        # Intermediate iterations only (the last one is finalised after the loop).
        # Pruning (min_size / top-N / renumber) changes the running seed and therefore
        # the final result; saving alone must not, so it saves a pruned copy.
        if ero_iter != erosion_steps and (prune_every_iter or save_every_iter):
            pruned_seed, pruned_ids = reorder_segmentation(combine_seed, min_size=min_size,
                              sort_ids=sort, top_n=segments_list[-1])
            if prune_every_iter:
                combine_seed = pruned_seed
                history.renumbered(pruned_ids)
            if save_every_iter:
                output_img_name = f'INTER_adaptive_thre_{threshold}_{upper_threshold}_ero_{ero_iter}.tif'
                save_seed(pruned_seed, output_folder, output_img_name, return_for_napari=return_for_napari, seeds_dict=seeds_dict)
            
        
        if no_consec_split_count>=no_split_max_iter:
            print(f"detect non split for {no_consec_split_count}rounds")
            print(f"break loop at {ero_iter} iter")
            history.stopped(no_consec_split_count)
            break
        


    
    sprout_core.check_cancel(cancel)
    steps.jump(steps.total, "saving")    # an early stop counts as done
    combine_seed, final_ids = reorder_segmentation(combine_seed, min_size=min_size, sort_ids=sort,
                                                   top_n=segments_list[-1])
    # if sort:
    #     output_path = os.path.join(output_folder,"FINAL_" + output_name+'_sorted.tif')
    # else:
    #     output_path = os.path.join(output_folder,"FINAL_" + output_name+'.tif')
    
    # output_img_name = "FINAL_adaptive_seed.tif" + output_name+'_sorted.tif'
    output_img_name = "FINAL_adaptive_seed.tif"
    
    save_seed(combine_seed, output_folder, output_img_name, return_for_napari=return_for_napari, seeds_dict=seeds_dict)

    if save_final_for_batch:
        save_seed(combine_seed, root_output_folder, base_name + "_FINAL_adaptive_seed.tif", 
                  return_for_napari=return_for_napari, seeds_dict=seeds_dict)

    

    

    history.save(output_folder, combine_seed, final_ids)
    config_core.save_used_config(output_folder, config_core.used_config(
        values, config_core.input_val_make_adaptive_seed,
        renames={"threshold": "thresholds", "upper_threshold": "upper_thresholds"}), "--adaptive_seed")

    return seeds_dict,ori_combine_ids_map, output_dict    


def make_adaptive_seed_thre(
                        thresholds,
                        output_folder,
                        erosion_steps, 
                        segments,
                        
                        img = None,
                        img_path = None,
                        
                        boundary =None,
                        boundary_path = None,
                        
                        num_threads = None,
                        
                        background = 0,
                        sort = True,
                        
                        no_split_max_iter =3,
                        min_size=1,
                        min_split_ratio = 0,
                        min_split_total_ratio = 0,
                        
                        save_every_iter = True,
                        prune_every_iter = None,
                        base_name = None,
                       
                        init_segments = None,
                        last_segments = None,
                        
                        footprints = None,
                        
                        upper_thresholds = None,
                        split_size_limit = (None, None),
                        split_convex_hull_limit = (None, None),
        
                        return_for_napari = False,
                        save_final_for_batch = False,
                        cancel = None,
                        progress = None
                    ):
    
    """Perform adaptive seed generation and iterative splitting for image segmentation using thresholding and morphological erosion.
    This function processes an input image (or loads it from a path), applies a threshold and a given erosion number to generate initial seed regions, 
    Then it iterates through a sequence of thresholds to try to create smaller regions and more separations.
    
    
    for splitting, minimum region size, and stopping criteria. Optionally, boundaries can be masked, and results can be saved at each iteration.
    
    Parameters
    ----------
    thresholds : list or array-like
        Sequence of threshold values to apply for seed generation and splitting.
    output_folder : str
        Path to the folder where output files will be saved.
    erosion_steps : int
        Number of erosion iterations to apply at each thresholding step.
    segments : int
        Number of connected components (segments) to extract at each step.
    img : np.ndarray, optional
        Input image array. If not provided, `img_path` must be specified.
    img_path : str, optional
        Path to the input image file. Used if `img` is not provided.
    boundary : np.ndarray, optional
        Binary mask to explicitly specify boundaries to exclude from processing.
    boundary_path : str, optional
        Path to the boundary mask file. Used if `boundary` is not provided.
    num_threads : int, default=1
        Number of threads to use for parallel processing during intersection checks.
    background : int, default=0
        Value representing the background in the segmentation.
    sort : bool, default=True
        Whether to sort segment IDs by size in the final output.
    no_split_max_iter : int, default=3
        Number of consecutive iterations without splits before stopping the process.
    min_size : int, default=1
        Minimum size (in pixels) for a segment to be kept.
    min_split_ratio : float, default=0
        Minimum proportion (relative to the parent region) for a split segment to be considered valid.
    min_split_total_ratio : float, default=0
        Minimum total proportion of split segments required to perform a split.
    save_every_iter : bool, default=True
        If True, save the seed mask at every iteration.
    prune_every_iter : bool or None, default=None
        After every intermediate threshold, remove regions smaller than min_size, keep
        the top-N and renumber the running seed. This changes the final result.
        None = same as save_every_iter, which reproduces the behaviour of earlier
        versions where saving implied pruning.
    name_prefix : str, default="Merged_seed"
        Prefix for output file names.
    init_segments : int, optional
        Number of initial segments to extract. Defaults to `segments` if not provided.
    footprints : str or list, default=None
        Morphological footprints shape or list of shapes for erosion. If a list, must match `erosion_steps`.
    upper_thresholds : list or array-like, optional
        Sequence of upper threshold values for range-based thresholding. Must match `thresholds` in length.
    split_size_limit : tuple, default=(None, None)
        Minimum and maximum size limits for split segments.
    split_convex_hull_limit : tuple, default=(None, None)
        Minimum and maximum convex hull size limits for split segments.
    sub_folder : str, optional
        Subfolder name within `output_folder` for saving results.
    cancel : threading.Event, optional
        Checked before every erosion / threshold step and between regions; when set,
        sprout_core.Cancelled is raised (files already written stay).
    progress : callable(done, total, text), optional
        Called after the initial seed and after every further threshold.
        
    Returns
    -------
    combine_seed : np.ndarray
        Final merged seed mask after all iterations and splits.
    ori_combine_ids_map : dict
        Mapping from original segment IDs to their descendant IDs after splitting.
    output_dict : dict
        Dictionary containing detailed information about splits, segment IDs, and output files for each threshold.
    
    Notes
    -----
    - The function saves intermediate and final segmentation results as TIFF files in the specified output folder.
    - A CSV file summarizing the split operations and parameters is also saved.

    """
    values = dict(locals())                       # the arguments, for used_config.yaml

    img = config_core.check_and_load_data(img, img_path, "img")
    boundary = config_core.check_and_load_data(boundary, boundary_path, "boundary", must_exist=False)
    config_core.valid_input_data(img, boundary=boundary)   

    if num_threads is None:
        num_threads = max(1, max_threads // 2)

    if num_threads>=max_threads:
        num_threads = max(1,max_threads-1)

    min_split_ratio = min_split_ratio*100
    min_split_total_ratio = min_split_total_ratio*100

    # output_name = f"{name_prefix}_ero_{erosion_steps}"
    
    
    # if sub_folder is None:
    #     output_folder = os.path.join(output_folder, output_name)
    # else:
    #     output_folder = os.path.join(output_folder, sub_folder)

    

    base_name = config_core.check_and_assign_base_name(base_name, img_path, "adapt_seed")
    
    root_output_folder = output_folder
    output_folder = os.path.join(output_folder , base_name)
    output_folder = os.path.abspath(output_folder)
    config_core.warn_earlier_results(output_folder)
    os.makedirs(output_folder,exist_ok=True)

    footprint_list = config_core.check_and_assign_footprint(footprints, erosion_steps)
    thresholds, upper_thresholds = config_core.check_and_assign_thresholds(thresholds, upper_thresholds)
    
    segments_list = config_core.check_and_assign_segment_list(segments, init_segments,
                                                              last_segments, n_threhsolds=len(thresholds))

    # None keeps the legacy behaviour: saving intermediate results also pruned the
    # running seed, so the final result depended on save_every_iter.
    if prune_every_iter is None:
        prune_every_iter = save_every_iter

    values_to_print = {
        "img_path": img_path,
        "Thresholds": thresholds,
        "upper_thresholds": upper_thresholds,
        "Output Folder": output_folder,
        "Erosion Iterations": erosion_steps,
        "Segments": segments_list,
        "Segments for the first output": init_segments,
        "Segments for the final result": last_segments,
        "boundary_path": boundary_path,
        "Boundary": describe_boundary(boundary, boundary_path),
        "Number of Threads": num_threads,
        "Sort": sort,
        "Background Value": background,
        "Save Every Iteration": save_every_iter,
        "Prune Every Iteration": prune_every_iter,
        "Footprints": footprint_list,
        "No Split Limit for iters": no_split_max_iter,
        "Component Minimum Size": min_size,
        "Minimum Split Proportion (%)": min_split_ratio,
        "Minimum Split Sum Proportion (%)": min_split_total_ratio,
        "split_size_limit": split_size_limit,
        "split_convex_hull_limit": split_convex_hull_limit
    }

    for key, value in values_to_print.items():
        print(f"  {key}: {value}")



    # The boundary is removed from every threshold mask, including this first one
    # (Supplementary Algorithm 2b, step 2). It is never written into img: img can be
    # the caller's array (e.g. a napari image layer), which must not be modified.
    if boundary is not None:
        boundary = config_core.check_and_cast_boundary(boundary)

    if upper_thresholds[0] is not None:
        mask = (img>=thresholds[0]) & (img<=upper_thresholds[0])
    else:
        mask = img>=thresholds[0]
    if boundary is not None:
        mask[boundary] = False

    steps = sprout_core.StepProgress(progress, len(thresholds))
    for ero_iter in range(1, erosion_steps+1):
        sprout_core.check_cancel(cancel)
        mask = sprout_core.erosion_binary_img_on_sub(mask, kernal_size = 1,
                                                     footprint=footprint_list[ero_iter-1])
    sprout_core.check_cancel(cancel)
    init_seed, _ = sprout_core.get_ccomps_with_size_order(mask,segments_list[0])
    steps.step(f"threshold {thresholds[0]}")
    
    seeds_dict = {}
    
    output_img_name = f'INTER_thre_{thresholds[0]}_{upper_thresholds[0]}_ero_{erosion_steps}.tif'
    if save_every_iter:
        save_seed(init_seed, output_folder, output_img_name, 
                  return_for_napari=return_for_napari, seeds_dict=seeds_dict)
            
    
    init_ids = [int(value) for value in sprout_core.unique_labels(init_seed) if value != background]
    if not init_ids:
        raise RuntimeError("No components found from the initial seeds. Exiting.")
    max_seed_id = int(np.max(init_ids))
    history = _History(init_ids, {"threshold": thresholds[0], "upper": upper_thresholds[0]})


    combine_seed = init_seed.copy()
    # at least uint16 as before; new ids from splits are checked with ensure_label_capacity
    combine_seed = sprout_core.cast_labels(combine_seed, np.uint16)

    output_dict = {"total_id": {0: len(sprout_core.unique_labels(combine_seed))-1},
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
    

    for idx_threshold, (threshold, upper_threshold) in enumerate(zip(thresholds[1:], upper_thresholds[1:])):
        print(f"working on thre ({threshold}, {upper_threshold})")
        sprout_core.check_cancel(cancel)
        
        output_dict["split_id"][threshold] = {}
        output_dict["split_ori_id"][threshold] = {}
        output_dict["split_ori_id_filtered"][threshold] = {}
        output_dict["split_prop"][threshold] = {}
        
        
        if upper_threshold is not None:
            mask = (img>=threshold) & (img<=upper_threshold)
        else:
            mask = img>=threshold
        if boundary is not None:
            mask[boundary] = False

        for ero_iter in range(1, erosion_steps+1):
            sprout_core.check_cancel(cancel)
            mask = sprout_core.erosion_binary_img_on_sub(mask, kernal_size = 1,
                                                         footprint=footprint_list[ero_iter-1])
        sprout_core.check_cancel(cancel)
        seed, _ = sprout_core.get_ccomps_with_size_order(mask,segments_list[idx_threshold+1])
        seed = sprout_core.cast_labels(seed, np.uint16)
        history.start_step(idx_threshold + 1, {"threshold": threshold, "upper": upper_threshold}, seed.max())
        

        if save_every_iter:
            output_img_name = f'INTER_thre_{threshold}_{upper_threshold}_ero_{erosion_steps}.tif'
            output_dict["cur_seed_name"][threshold] = output_img_name
            
            save_seed(seed, output_folder, output_img_name, 
                      return_for_napari=return_for_napari, seeds_dict=seeds_dict)

        
        combine_ids = [int(value) for value in sprout_core.unique_labels(combine_seed) if value != background]
        
        
        has_split = False
        ## Comparing each ccomp from eroded seed
        ## to each ccomp from the original seed
        
        inter_log = {
            "inter_count":0,
            "inter_ids": np.array([]),
            "inter_props": np.array([])
        }
        
        for combine_id in combine_ids:
            sprout_core.check_cancel(cancel)
            ccomp_combine_seed = combine_seed == combine_id
            
            detect_inter(ccomp_combine_seed, seed, inter_log, background)
        
        
            # The parts that count: at least min_size voxels and more than min_split_ratio of the region.
            # A region splits into two or more of them; else it stays as it is (specks do not split it).
            counted = (inter_log["inter_sizes"] >= (min_size or 0)) & (inter_log["inter_props"] > min_split_ratio)
            if np.count_nonzero(counted) > 1:
                temp_inter_count = inter_log["inter_count"]
                temp_inter_ids = inter_log["inter_ids"]
                temp_inter_props = inter_log["inter_props"]
                
                
                # print(f'\t{combine_id} has been split to {temp_inter_count} parts. Ids are {temp_inter_ids}')
                # print(f"\tprops are: {temp_inter_props}")
                
                sum_inter_props = np.sum(temp_inter_props)
                # print(f"\tSplit prop is {sum_inter_props}")
                

                if not split_size_check(ccomp_combine_seed, split_size_limit):
                    print(f"no split for region {combine_id}: size {int(np.sum(ccomp_combine_seed))} is outside split_size_limit {split_size_limit}")
                        

                split_condition =  (sum_inter_props>=min_split_total_ratio) and\
                    split_size_check(ccomp_combine_seed, split_size_limit) and \
                        split_convex_hull_check(ccomp_combine_seed, split_convex_hull_limit)

                if split_condition:
                    combine_seed[combine_seed == combine_id] =0
                    filtered_inter_ids = temp_inter_ids[counted]
                    has_split = True
                    
                    # inter_coords_map = {
                    #     inter_id: np.where(seed == inter_id)
                    #     for inter_id in filtered_inter_ids
                    # }
                    # combine_seed_optimized = combine_seed.copy()
                    new_ids = []
                    for inter_id in filtered_inter_ids:
                        max_seed_id +=1
                        combine_seed = sprout_core.ensure_label_capacity(combine_seed, max_seed_id)
                        combine_seed[seed == inter_id] = max_seed_id
                        
                        # TODO Compare use np.where or direct bool
                        # current_time = time.time()
                        # combine_seed[seed == inter_id] = max_seed_id
                        # # new_ids.append(max_seed_id)     
                        # end_time = time.time()
                        # print(f"Processing inter_id {inter_id} took {end_time - current_time:.4f} seconds")
                        
                        
                        # current_time = time.time()
                        # coords = np.where(seed == inter_id)
                        # combine_seed_optimized[coords] = max_seed_id
                        # end_time = time.time()
                        # print(f"Optimizing inter_id {inter_id} took {end_time - current_time:.4f} seconds")

                        
                        # if not np.array_equal(combine_seed, combine_seed_optimized):
                        #     print(f"❌ Inconsistent results for inter_id={inter_id}")
                        #     diff = np.logical_xor(combine_seed, combine_seed_optimized)
                        #     print(f"  -> Different voxels count: {np.sum(diff)}")
                        # else:
                        #     print(f"✅ Match for inter_id={inter_id}")
                        # End TODO
                        
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
                    history.replaced(combine_id, new_ids)
                
        
        if combine_ids:
            output_dict["total_id"][threshold] = len(sprout_core.unique_labels(combine_seed))-1
    
        if has_split:
            no_consec_split_count=0
        else:
            no_consec_split_count+=1
        steps.step(f"threshold {threshold}")
            
        if no_consec_split_count>=no_split_max_iter:
                print(f"detect non split for {no_consec_split_count}rounds")
                print(f"break loop at {threshold} threshold")
                history.stopped(no_consec_split_count)
                break
        
        # Intermediate thresholds only (the last one is finalised after the loop).
        # Pruning (min_size / top-N / renumber) changes the running seed and therefore
        # the final result; saving alone must not, so it saves a pruned copy.
        if idx_threshold != len(thresholds)-2 and (prune_every_iter or save_every_iter):
            pruned_seed, pruned_ids = reorder_segmentation(combine_seed, min_size=min_size, sort_ids=sort,
                                                  top_n=segments_list[-1])
            if prune_every_iter:
                combine_seed = pruned_seed
                history.renumbered(pruned_ids)
            if save_every_iter:
                output_img_name = f'INTER_adaptive_thre_{threshold}_{upper_threshold}_ero_{erosion_steps}.tif'
                save_seed(pruned_seed, output_folder, output_img_name, return_for_napari=return_for_napari, seeds_dict=seeds_dict)
         
    sprout_core.check_cancel(cancel)
    steps.jump(steps.total, "saving")    # an early stop counts as done
    combine_seed, final_ids = reorder_segmentation(combine_seed, min_size=min_size, sort_ids=sort,
                                                   top_n=segments_list[-1])
    # if sort:
    #     output_path = os.path.join(output_folder,"FINAL_" + output_name+'_sorted.tif')
    # else:
    #     output_path = os.path.join(output_folder,"FINAL_" + output_name+'.tif')
    output_img_name = "FINAL_adaptive_seed.tif"
    save_seed(combine_seed, output_folder, output_img_name, 
              return_for_napari=return_for_napari, seeds_dict=seeds_dict)
    if save_final_for_batch:
        save_seed(combine_seed, root_output_folder, base_name + "_FINAL_adaptive_seed.tif", 
                  return_for_napari=return_for_napari, seeds_dict=seeds_dict)
    
    history.save(output_folder, combine_seed, final_ids)
    config_core.save_used_config(output_folder, config_core.used_config(
        values, config_core.input_val_make_adaptive_seed), "--adaptive_seed")
    
             
    return seeds_dict,ori_combine_ids_map, output_dict  

def run_make_adaptive_seed(file_path):
  
    _, extension = os.path.splitext(file_path)
    print(f"processing config the file {file_path}")
    if extension == '.yaml':
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            optional_params = config_core.validate_input_yaml(config, config_core.input_val_make_adaptive_seed)
            

    # Checking if it's merged from erosion of threhsolds
    if isinstance(config['thresholds'], (int, float)):
        seed_merging_mode = "ERO"
    elif isinstance(config['thresholds'], list):
        if all(isinstance(t, (int, float)) for t in config['thresholds']):
            if len(config['thresholds']) == 1:
                seed_merging_mode = "ERO"
                config['thresholds'] = config['thresholds'][0]
                if optional_params['upper_thresholds'] is not None:
                    optional_params['upper_thresholds'] = optional_params['upper_thresholds'][0]
                
            elif len(config['thresholds']) > 1:
                seed_merging_mode = "THRE"
        else:
            raise ValueError("'thresholds' must be a number or a list of numbers.")
    
    
    # Track the overall start time
    overall_start_time = time.time()
    
    ## Use path then actual img data as the input        
    # img = imread(config['img_path'])
    # if optional_params['boundary_path'] is not None:
    #     boundary = imread(optional_params['boundary_path'])
    # else:
    #     boundary = None
    
    # sub_folder = os.path.basename(config['img_path'])
    
    if seed_merging_mode == "ERO":
        seeds_dict ,ori_combine_ids_map , output_dict=  make_adaptive_seed_ero(
                                    threshold=config['thresholds'],
                                    output_folder=config['output_folder'],
                                    erosion_steps=config['erosion_steps'], 
                                    segments= config['segments'],
                                    num_threads = config['num_threads'],
                                    
                                    img_path = config['img_path'],
                                    boundary_path = optional_params['boundary_path'],                                            
                                    
                                    background = optional_params["background"],
                                    sort = optional_params["sort"],
                                    
                                    base_name=optional_params["base_name"],
                                    
                                    
                                    no_split_max_iter =optional_params["no_split_max_iter"],
                                    min_size=optional_params["min_size"],
                                    min_split_ratio = optional_params["min_split_ratio"],
                                    min_split_total_ratio = optional_params["min_split_total_ratio"],
                                    
                                    save_every_iter = optional_params["save_every_iter"],
                                    
                                    prune_every_iter = optional_params["prune_every_iter"],
                                    
                                    init_segments = optional_params["init_segments"],
                                    last_segments = optional_params["last_segments"],
                                    footprints = optional_params["footprints"],
                                    
                                    upper_threshold = optional_params["upper_thresholds"],
                                    split_size_limit= optional_params["split_size_limit"],
                                    split_convex_hull_limit = optional_params["split_convex_hull_limit"],
                                    
                                    
                                    return_for_napari = False
                                    # sub_folder = sub_folder,
                                    # name_prefix = optional_params["name_prefix"],    
                                    )
    
    # pd.DataFrame(ori_combine_ids_map).to_csv(os.path.join(output_folder, 'ori_combine_ids_map.csv'),index=False)
    
    elif seed_merging_mode=="THRE":
   
        seeds_dict ,ori_combine_ids_map , output_dict= make_adaptive_seed_thre(

                                    thresholds=config['thresholds'],
                                    output_folder=config['output_folder'],
                                    erosion_steps=config['erosion_steps'], 
                                    segments= config['segments'],
                                    
                                    num_threads = config['num_threads'],
                                    
                                    img_path = config['img_path'],
                                    boundary_path = optional_params['boundary_path'],    
                                    
                                    background = optional_params["background"],
                                    sort = optional_params["sort"],
                                    
                                    base_name=optional_params["base_name"],
                                    
                                    no_split_max_iter =optional_params["no_split_max_iter"],
                                    min_size=optional_params["min_size"],
                                    min_split_ratio = optional_params["min_split_ratio"],
                                    min_split_total_ratio = optional_params["min_split_total_ratio"],
                                    
                                    save_every_iter = optional_params["save_every_iter"],
                                    
                                    prune_every_iter = optional_params["prune_every_iter"],
                                                                        
                                    init_segments = optional_params["init_segments"],
                                    last_segments= optional_params["last_segments"],
                                    footprints = optional_params["footprints"],
                                    
                                    upper_thresholds = optional_params["upper_thresholds"],
                                    split_size_limit= optional_params["split_size_limit"],
                                    split_convex_hull_limit = optional_params["split_convex_hull_limit"],
                                    
                                    return_for_napari=False            
                                    )
                

    

    
    # Track the overall end time
    overall_end_time = time.time()

    # Output the total running time
    print(f"Total running time: {overall_end_time - overall_start_time:.2f} seconds")
        

    

if __name__ == "__main__":        
   
    # Get the file path from the first command-line argument or use the default

    if len(sys.argv) > 1:
        print(f"Reading config file from command-line argument: {sys.argv[1]}")
        file_path = sys.argv[1]
    else:
        print("No config file specified in arguments. Using default: ./template/make_adaptive_seed.yaml")
        file_path = './template/make_adaptive_seed.yaml'

    run_make_adaptive_seed(file_path=file_path)
  