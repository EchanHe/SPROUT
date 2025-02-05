# import nibabel as nib
import numpy as np
import os, sys
from scipy import ndimage as ndi
import warnings
from skimage.morphology import ball
import tifffile
import gc
import csv
from datetime import datetime
import json
from skimage import measure

from skimage.measure import marching_cubes
import trimesh


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

def volume_import(volume_path, dtype = np.uint16):
    file = os.path.join(volume_path)
    volume = tifffile.imread(file)
    volume_array = np.array(volume, dtype=dtype)
    return volume_array

def get_largest(label, segments):
    """Get the largest ccomp

    Args:
        label (_type_): _description_
        segments (_type_): _description_

    Returns:
        _type_: _description_
    """
    labels, _ = ndi.label(label)
    assert (labels.max() != 0)
    number = 0
    try:
        bincount = np.bincount(labels.flat)[1:]
        bincount_sorted = np.sort(bincount)[::-1]
        largest = labels-labels
        m=0
        for i in range(segments):
            index = int(np.where(bincount == bincount_sorted[i])[0][m]) + 1
            ilargest = labels == index
            largest += np.where(ilargest, i + 1, 0)
        if i == segments-1:
            number = segments
    except:
        warnings.warn(f"Number of segments should be reduced to {i}")
        if number == 0:
            number = i
    return largest,number, bincount_sorted[:segments]

def find_seg_by_morpho_trans(input_mask, threshold_binary,  dilate_iter,
                             touch_rule = 'stop',
                             segments=None, ero_shape = 'ball'):
    # if segments is None:
    #     total_segs = len(np.unique(input_mask))
    # else:
    #     total_segs = segments+1
    label_id_list = np.unique(input_mask)
    label_id_list = label_id_list[label_id_list!=0]
    
    result = input_mask.copy()  
    
    ### Option 1 do all dilation iterations for every label
    
    # for label_id in range(1, total_segs):
    # for label_id in label_id_list:
    #     # First iteration. For the label
    #     # Get the result for dilaition
    #     dilated_binary_label_id = (result ==label_id)
    #     for i in range(dilate_iter):
    #         dilated_binary_label_id = sprout_core.dilation_binary_img_on_sub(dilated_binary_label_id, 
    #                                                                             margin = 2, kernal_size = 1)
    #         print(f"Size of each iter for label:{label_id} is {np.sum(dilated_binary_label_id)}")
    #     if touch_rule == 'stop':
    #         # This is the binary for non-label of the updated mask
    #         binary_non_label = (result !=label_id) & (result != 0)
    #         # See if original mask overlay with grown label_id mask
    #         overlay = np.logical_and(binary_non_label, dilated_binary_label_id)
    #         # Check if there are any True values in the resulting array
    #         HAS_OVERLAY = np.any(overlay)
            
    #         print(f"""
    #               np.sum((result ==label_id)){np.sum((result ==label_id))},
    #               np.sum(dilated_binary_label_id){np.sum(dilated_binary_label_id)},
    #               np.sum(overlay){np.sum(overlay)},
    #               np.sum(binary_non_label){np.sum(binary_non_label)}
    #               """)
            
    #         if HAS_OVERLAY:
    #             dilated_binary_label_id[overlay] = False
    #     result[dilated_binary_label_id & threshold_binary] = label_id
        # result[dilated_binary_label_id] = label_id
        
    ### Option 2 do all the label grow for every dilation iteration 
    
    for i in range(dilate_iter):
        print(f"Growing on {i} iter, with Rule:{touch_rule}")
        for label_id in label_id_list:
            dilated_binary_label_id = (result ==label_id)
            dilated_binary_label_id = dilation_binary_img_on_sub(dilated_binary_label_id, 
                                                                    margin = 2, kernal_size = 1)
            
            
            if touch_rule == 'stop':
                # This is the binary for non-label of the updated mask
                binary_non_label = (result !=label_id) & (result != 0)
                # See if original mask overlay with grown label_id mask
                overlay = np.logical_and(binary_non_label, dilated_binary_label_id)
                # Check if there are any True values in the resulting array
                HAS_OVERLAY = np.any(overlay)
                
                print(f"""
                    np.sum((result ==label_id)){np.sum((result ==label_id))},
                    np.sum(dilated_binary_label_id){np.sum(dilated_binary_label_id)},
                    np.sum(overlay){np.sum(overlay)},
                    np.sum(binary_non_label){np.sum(binary_non_label)}
                    """)
                
                if HAS_OVERLAY:
                    dilated_binary_label_id[overlay] = False
            
            result[dilated_binary_label_id & threshold_binary] = label_id  
        
        ## Save the results if there is needed         
    
    return result



def get_ccomps_with_size_order(volume, segments, min_vol = None):
    """Get the largest ccomp

    Args:
        label (_type_): _description_
        segments (_type_): _description_

    Returns:
        _type_: _description_
    """
    labeled_image = measure.label(volume, background=0, 
                                  return_num=False, connectivity=2)
    component_sizes = np.bincount(labeled_image.ravel())[1:] 
    
    component_sizes_sorted = -np.sort(-component_sizes,)
    
    component_labels = np.unique(labeled_image)[1:]
    # props = measure.regionprops(label_image)
    largest_labels = component_labels[np.argsort(component_sizes[component_labels - 1])[::-1][:segments]]
    
    output = np.zeros_like(labeled_image)
    for label_id, label in enumerate(largest_labels):
        output[labeled_image == label] = label_id+1
    
    return output, component_sizes_sorted[:segments]

    
def grow(labels, number):
    grownlabels = np.copy(labels)
    for i in range(number):
        filtered = np.where(labels==i+1,1,0)
        grown = ndi.binary_dilation(np.copy(filtered), structure=ball(2)).astype(np.uint16)
        grownlabels = np.where(np.copy(grown), i + 1, np.copy(grownlabels))
        del grown
        del filtered
    return grownlabels


def bbox2_3D(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax

def segmentation(volume_array, initial_threshold, target_threshold, segments, iterations, label = False, label_preserve = False, 
                 seed_dilation = False,
                 return_log_dict = True):
    """_summary_

    Args:
        volume_array (_type_): 3D NumPy array of the volume data (np.uint16)
        initial_threshold (_type_): _description_
        target_threshold (_type_): _description_
        segments (_type_): number of segments to be created (int)
        iterations (_type_): number of iterations (int)
        label (bool, optional): False (bool(if unused)\np.uint16). Defaults to False.
        label_preserve (bool, optional): whether the label numbering and boundaries should be 
        preserved if False the label connectivity will be recomputed default = False (bool). Defaults to False.
        
        seed_dilation (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    start_time = datetime.now()
    if label == False:
        volume_label = volume_array > initial_threshold
    else:
        volume_label = label


    if label_preserve == False:
        #get the top largest ccomps
        seed, number, ccomp_sizes= get_largest(volume_label,segments)
        
    else:
        # Use the input as volume
        seed = volume_label
        number = segments


    if seed_dilation == True:
    # Grow the seed
        formed_seed = grow(seed, number)
    else:
        formed_seed = seed
    
    # Label volume is the seed.
    labeled_volume = np.copy(formed_seed)

    for i in range(iterations+1):
        if i%10==0:
            print(f"in iter {i}")
        
        # Iterations
        
        
        # Gradually decrease the threshold
        #if label == False:
            #volume_label = volume_array > initial_threshold
        volume_label = volume_array > initial_threshold - (i * (initial_threshold - target_threshold) / iterations)
        
        # labeled_volume is the output each loop
        # 0 Should be the back ground, Set the seed !=0 to zero.
        # First iteration: Remove any positive value from the seed.
        
        volume_label = np.where(labeled_volume != 0, False, volume_label)
    
        for j in range(number):
        # Going through segments
            try:
                rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(labeled_volume == j + 1)
            except:
                rmin, rmax, cmin, cmax, zmin, zmax = -1, 1000000, -1, 1000000, -1, 1000000
            maximum = labeled_volume.shape
            rmin = max(0, rmin - int((rmax - rmin) * 0.1))
            rmax = min(int((rmax - rmin) * 0.1) + rmax, maximum[0])
            cmin = max(0, cmin - int((cmax - cmin) * 0.1))
            cmax = min(int((cmax - cmin) * 0.1) + cmax, maximum[1])
            zmin = max(0, zmin - int((zmax - zmin) * 0.1))
            zmax = min(int((zmax - zmin) * 0.1) + zmax, maximum[2])
            temp_label = np.copy(volume_label)
            
            # The sub where j+1 appear in the labeled_volume
            reduced_labeled_volume = labeled_volume[rmin:rmax, cmin:cmax, zmin:zmax]
            
            # Get the label of seg j+1
            # temp_label = Set the j+1 label to TRUE on volume_label
            temp_label[rmin:rmax, cmin:cmax, zmin:zmax] = np.copy(volume_label)[rmin:rmax, cmin:cmax,
                                                          zmin:zmax] + (
                                                                  reduced_labeled_volume == j + 1)
                                                          
            # get idx of j+1                                                          
            pos = np.where(reduced_labeled_volume == j + 1)
            
            # Set j+1 to 0
            labeled_volume[rmin:rmax, cmin:cmax, zmin:zmax] = np.where(reduced_labeled_volume == j + 1, 0,
                                                                       reduced_labeled_volume)
            
            # Only the region, get label (ccomps)
            # ccomps on the temp_label
            labeled_temp, _ = ndi.label(np.copy(temp_label[rmin:rmax, cmin:cmax, zmin:zmax]))
            
            # Get the value of the first position of reduced_labeled_volume == j + 1
            # Seems to get the index from ccomps this time for j+1
            try:
                index = int(labeled_temp[pos[0][0], pos[1][0], pos[2][0]])
            except:
                index = 1
            try:
                # Where labeled_temp == index
                relabelled = np.copy(labeled_temp) == index
                
                # where relabelled is true, set it to j+1
                # Where is not true, set it un-changed
                labeled_volume[rmin:rmax, cmin:cmax, zmin:zmax] = np.where(np.copy(relabelled), j + 1,
                                                                           labeled_volume[rmin:rmax, cmin:cmax,
                                                                           zmin:zmax])
                del temp_label
                del pos
                del labeled_temp
                del relabelled
                gc.collect()
            except:
                print(f"missing {j}")
    
    ### Caculating the sum for each label.
    max_label = labeled_volume.max()            
    # Initialize the dictionary to store the sums
    label_sums = {}

    # Iterate through each label from 1 to max_label (excluding 0)
    for label in range(1, max_label + 1):
        label_sums[label] = int(np.sum(labeled_volume == label))
    
                
    # Capture the end time
    end_time = datetime.now()
    # Calculate the duration
    duration = end_time - start_time                
    # Collect the arguments in a dictionary
    args_dict = {
        "Method": "Ori Bounti",
        "volume_array shape": list(volume_array.shape),
        "initial_threshold": int(initial_threshold),
        "target_threshold": int(target_threshold),
        "num_of_segments": segments,
        "iterations": iterations,
        "label": label,
        "label_preserve": label_preserve,
        "seed_dilation": seed_dilation,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": str(duration),
        f'size_top_{segments}_ccomp_of_init_thre': ccomp_sizes.tolist(),
        f'size_after': label_sums
    }            
             
    # Return the seed used for generation
    # And the end result
    if return_log_dict:
        return labeled_volume, formed_seed , args_dict
    else:
        return labeled_volume, formed_seed





def segmentation_flood(volume_array, initial_threshold, target_threshold,
                       segments, iterations, label=False,
                         label_preserve=False, seed_dilation=False):
    start_time = datetime.now()
    if type(label) == bool:
        volume_label = volume_array > initial_threshold
    else:
        volume_label = label

    if label_preserve == False:
        seed, number, ccomp_sizes = get_largest(volume_label, segments)
    else:
        seed = volume_label
        number = segments

    if seed_dilation == True:
        formed_seed = grow(seed, number)
    else:
        formed_seed = seed

    labeled_volume = np.copy(formed_seed)

    for i in range(iterations + 1):
        volume_label = volume_array > initial_threshold - (
                    i * (initial_threshold - target_threshold) / iterations)
        volume_label = np.where(labeled_volume != 0, 0, volume_label)
        print(np.max(volume_label))
        for j in range(5):
            shift1 = np.zeros_like(labeled_volume)
            shift1[0:-1, 0:, 0:] = np.copy(labeled_volume[1:, 0:, 0:])
            shift2 = np.zeros_like(labeled_volume)
            shift2[1:, 0:, 0:] = np.copy(labeled_volume[:-1, 0:, 0:])
            shift3 = np.zeros_like(labeled_volume)
            shift3[0:, 0:-1, 0:] = np.copy(labeled_volume[0:, 1:, 0:])
            shift4 = np.zeros_like(labeled_volume)
            shift4[0:, 1:, 0:] = np.copy(labeled_volume[0:, :-1, 0:])
            shift5 = np.zeros_like(labeled_volume)
            shift5[0:, 0:, 0:-1] = np.copy(labeled_volume[0:, 0:, 1:])
            shift6 = np.zeros_like(labeled_volume)
            shift6[0:, 0:, 1:] = np.copy(labeled_volume[0:, 0:, :-1])
            shift = np.where(shift1 !=0, shift1, np.where(shift2 !=0, shift2, np.where(shift3 != 0, shift3, np.where(shift4 != 0, shift4, np.where(shift5 !=0, shift5, np.where(shift6 !=0, shift6, 0))))))
            print(np.max(shift))
            temp_growth = np.where(volume_label != 0, shift, 0)
            print(np.max(temp_growth))
            labeled_volume = np.where(np.copy(temp_growth) != 0, np.copy(temp_growth),np.copy(labeled_volume))
            all_zeros = not np.any(np.where(temp_growth == 0,False,True))
            del shift, shift1, shift2, shift3, shift4, shift5, shift6
            del temp_growth
            gc.collect()

            if all_zeros == True:
                break
        gc.collect()
        
        
    max_label = labeled_volume.max()            
    # Initialize the dictionary to store the sums
    label_sums = {}

    # Iterate through each label from 1 to max_label (excluding 0)
    for label in range(1, max_label + 1):
        label_sums[label] = int(np.sum(labeled_volume == label))
        
        
    # Capture the end time
    end_time = datetime.now()
    # Calculate the duration
    duration = end_time - start_time                
    # Collect the arguments in a dictionary
    args_dict = {
        "Method": "Flood Bounti",
        "volume_array shape": list(volume_array.shape),
        "initial_threshold": int(initial_threshold),
        "target_threshold": int(target_threshold),
        "num_of_segments": segments,
        "iterations": iterations,
        "label": label,
        "label_preserve": label_preserve,
        "seed_dilation": seed_dilation,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": str(duration),
        f'size_top_{segments}_ccomp_of_init_thre': ccomp_sizes.tolist(),
        f'size_after': label_sums
    }            
                
        # step = hx_project.create('HxUniformLabelField3')
        # step.name = input.name + f".Step{i}"
        # step.set_array(np.array(labeled_volume, dtype=np.ushort))
        # step.bounding_box = input.bounding_box
    return labeled_volume, formed_seed

def segmentation_ero(volume_array, initial_threshold, target_threshold, segments, iterations, label = False, label_preserve = False, 
                 seed_dilation = False):
    """_summary_

    Args:
        volume_array (_type_): 3D NumPy array of the volume data (np.uint16)
        initial_threshold (_type_): _description_
        target_threshold (_type_): _description_
        segments (_type_): number of segments to be created (int)
        iterations (_type_): number of iterations (int)
        label (bool, optional): False (bool(if unused)\np.uint16). Defaults to False.
        label_preserve (bool, optional): whether the label numbering and boundaries should be 
        preserved if False the label connectivity will be recomputed default = False (bool). Defaults to False.
        
        seed_dilation (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # Capture the start time
    start_time = datetime.now()
    # if label == False:
    #     volume_label = volume_array > initial_threshold
    # else:
    #     volume_label = label


    volume_label = volume_array > initial_threshold
    
    for i in range(2):
        volume_label = sprout_core.erosion_binary_img_on_sub(volume_label, kernal_size = 1)

    if label_preserve == False:
        #get the top largest ccomps
        seed, number, ccomp_sizes = get_largest(volume_label,segments)
        
    else:
        # Use the input as volume
        seed = volume_label
        number = segments
    

    if seed_dilation == True:
    # Grow the seed
        formed_seed = grow(seed, number)
    else:
        formed_seed = seed
    
    # Label volume is the seed.
    labeled_volume = np.copy(formed_seed)

    for i in range(iterations+1):
        if i%10==0:
            print(f"in iter {i}")
        
        # Iterations
        
        
        # Gradually decrease the threshold
        #if label == False:
            #volume_label = volume_array > initial_threshold
        
        # Select the one above certain threshold.    
        volume_label = volume_array > initial_threshold - (i * (initial_threshold - target_threshold) / iterations)
        
        # Set the seed !=0 to zero.
        # labeled_volume is the output each loop
        volume_label = np.where(labeled_volume != 0, False, volume_label)
        for j in range(number):
        # Going through segments
            try:
                rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(labeled_volume == j + 1)
            except:
                rmin, rmax, cmin, cmax, zmin, zmax = -1, 1000000, -1, 1000000, -1, 1000000
            maximum = labeled_volume.shape
            rmin = max(0, rmin - int((rmax - rmin) * 0.1))
            rmax = min(int((rmax - rmin) * 0.1) + rmax, maximum[0])
            cmin = max(0, cmin - int((cmax - cmin) * 0.1))
            cmax = min(int((cmax - cmin) * 0.1) + cmax, maximum[1])
            zmin = max(0, zmin - int((zmax - zmin) * 0.1))
            zmax = min(int((zmax - zmin) * 0.1) + zmax, maximum[2])
            temp_label = np.copy(volume_label)
            
            # The sub where j+1 appear
            reduced_labeled_volume = labeled_volume[rmin:rmax, cmin:cmax, zmin:zmax]
            
            # Get the label of seg j+1
            temp_label[rmin:rmax, cmin:cmax, zmin:zmax] = np.copy(volume_label)[rmin:rmax, cmin:cmax,
                                                          zmin:zmax] + (
                                                                  reduced_labeled_volume == j + 1)
                                                          
            # get idx of j+1                                                          
            pos = np.where(reduced_labeled_volume == j + 1)
            
            # Set j+1 to 0
            labeled_volume[rmin:rmax, cmin:cmax, zmin:zmax] = np.where(reduced_labeled_volume == j + 1, 0,
                                                                       reduced_labeled_volume)
            
            # Only the region, get label (ccomps)
            # ccomps on the temp_label
            labeled_temp, _ = ndi.label(np.copy(temp_label[rmin:rmax, cmin:cmax, zmin:zmax]))
            
            # Get the value of the first position of reduced_labeled_volume == j + 1
            try:
                index = int(labeled_temp[pos[0][0], pos[1][0], pos[2][0]])
            except:
                index = 1
            try:
                # Where labeled_temp == index
                relabelled = np.copy(labeled_temp) == index
                
                # where relabelled is true, set it to j+1
                # Where is not true, set it un-changed
                labeled_volume[rmin:rmax, cmin:cmax, zmin:zmax] = np.where(np.copy(relabelled), j + 1,
                                                                           labeled_volume[rmin:rmax, cmin:cmax,
                                                                           zmin:zmax])
                del temp_label
                del pos
                del labeled_temp
                del relabelled
                gc.collect()
            except:
                print(f"missing {j}")

    # Capture the end time
    end_time = datetime.now()
    # Calculate the duration
    duration = end_time - start_time                
    # Collect the arguments in a dictionary
    args_dict = {
        "Method": "Erosion bounti",
        "volume_array shape": list(volume_array.shape),
        "initial_threshold": initial_threshold,
        "target_threshold": target_threshold,
        "segments": segments,
        "iterations": iterations,
        "label": label,
        "label_preserve": label_preserve,
        "seed_dilation": seed_dilation,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": str(duration),
        f'size_top_{segments}_ccomp': ccomp_sizes.tolist()
    }
    filename = 'Bounti_run_log.json'
    write_json(filename, args_dict)

    # file_path = 'Bounti_run_log.csv'
    # file_exists = os.path.isfile(file_path)
    # # Write the dictionary to a CSV file
    # with open(file_path, 'w', newline='') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=args_dict.keys())
    #     if not file_exists:
    #         writer.writeheader()
    #     writer.writerow(args_dict)
    
    # Return the seed used for generation
    # And the end result
    return labeled_volume, formed_seed

def binary_stack_to_mesh(input_volume , threshold, downsample_scale=20,
                         face_color= [128]*4):
    
    input_volume = input_volume > threshold
    verts, faces, normals, _ = marching_cubes(input_volume, level=0.5)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    target_faces = mesh.faces.shape[0] // downsample_scale
    simplified_mesh = mesh.simplify_quadric_decimation(target_faces)

    simplified_mesh.visual.face_colors = face_color
    return simplified_mesh
    # simplified_mesh.export(os.path.join(output_dir, f"{threshold}.ply".format(id)))
# if __name__ == "__main__":
    ## TEst some bounti function
    

if __name__ == "__main__":
    workspace = r'C:\Users\Yichen\OneDrive\work\codes\nhm_bounti_pipeline\result\procavia'
    
    tif_file = os.path.join(workspace, 'thre_ero_seed/thre_ero_2iter_thre4500.tif')
    # tif_file = os.path.join(workspace, 'test/input/thre_ero_2iter_thre4500.tif')
    
    ori_path = os.path.join(workspace, 'input/procaviaH4981C_0001.tif.resampled_400_600.tif')
    
    # ori_path = os.path.join(workspace, 'input/procaviaH4981C_0001.tif.resampled.tif')
    
    
    ## Test by decrease the label values
    # input_mask = np.where(np.isin(input_mask, [4,5,8,9,11,17,18,20 ]) , input_mask, 0)
    # input_mask = np.where(np.isin(input_mask, [5,6,10,11]) , input_mask, 0)
    
    workspace = r'C:\Users\Yichen\OneDrive\work\codes\nhm_bounti_pipeline\result\foram_james'
    ori_path = os.path.join(workspace, 'input/ai/final.20180802_VERSA_1905_ASB_OLK_st016_bl4_fo5_recon.tif')
    tif_file = os.path.join(workspace, 'thre_ero_seed/final.20180802_VERSA_1905_ASB_OLK_st016_bl4_fo5_recon.tif.tif')
    base_name = os.path.basename(tif_file)
    
    input_mask = tifffile.imread(tif_file)
    ori_mask = tifffile.imread(ori_path)
    thresholds = [0]
    dilate_iters = [4]
    touch_rule = "no"
    for threshold in thresholds:
        for dilate_iter in dilate_iters:
            threshold_binary = ori_mask > threshold

            result = find_seg_by_morpho_trans(input_mask, threshold_binary , dilate_iter,
                                              touch_rule = touch_rule)
            
            
            # output_path = os.path.join(workspace, f'result/ai/dila_{dilate_iter}_{threshold}_rule_{touch_rule}.tif')
            output_path = os.path.join(workspace, f'result/ai/{base_name}.tif')
            tifffile.imwrite(output_path, 
            result,
            compression ='zlib')
    
    
    
    
