import numpy as np
from skimage.morphology import binary_dilation, binary_erosion, cube,square, binary_closing, ball, disk
from skimage import measure
from datetime import datetime
import os
import tifffile



ball_fp_YZ = np.array(
                [[[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]],

                [[0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0]],

                [[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]]], dtype=np.uint8)

ball_fp_XZ = np.array(
                [[[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]],

                [[0, 0, 0],
                    [1, 1, 1],
                    [0, 0, 0]],

                [[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]]], dtype=np.uint8)

ball_fp_XY = np.array(
                [[[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],

                [[0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]],

                [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]]], dtype=np.uint8)

fp_Z = np.array(
                [[[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]],

                [[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]],

                [[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]]], dtype=np.uint8)

fp_X = np.array(
                [[[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],

                [[0, 0, 0],
                    [1, 1, 1],
                    [0, 0, 0]],

                [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]]], dtype=np.uint8)

fp_Y = np.array(
                [[[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],

                [[0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0]],

                [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]]], dtype=np.uint8)

def get_sub_binary_image_by_pos(image_3d, margin):


    # Find the indices where the image is 1
    indices = np.argwhere(image_3d == True)


    is_3d = (image_3d.ndim == 3)
    if is_3d:
        # Get the min and max indices along each dimension
        min_z, min_y, min_x = indices.min(axis=0)
        max_z, max_y, max_x = indices.max(axis=0)
    else:
        # Get the min and max indices along each dimension
        min_y, min_x = indices.min(axis=0)
        max_y, max_x = indices.max(axis=0)
    # print(f"Bounding box for 3D image: x_min={min_x}, x_max={max_x}, y_min={min_y}, y_max={max_y}, z_min={min_z}, z_max={max_z}")

    if is_3d:
        # Adjust the bounding box with the margin
        min_z = max(min_z - margin, 0)
        max_z = min(max_z + margin, image_3d.shape[0] - 1)
        min_y = max(min_y - margin, 0)
        max_y = min(max_y + margin, image_3d.shape[1] - 1)
        min_x = max(min_x - margin, 0)
        max_x = min(max_x + margin, image_3d.shape[2] - 1)
        subset_3d = image_3d[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
        max_min_ids = [min_z, max_z, min_y ,max_y, min_x, max_x]
    else:
        min_y = max(min_y - margin, 0)
        max_y = min(max_y + margin, image_3d.shape[0] - 1)
        min_x = max(min_x - margin, 0)
        max_x = min(max_x + margin, image_3d.shape[1] - 1)
        subset_3d = image_3d[min_y:max_y+1, min_x:max_x+1]
        max_min_ids = [ min_y ,max_y, min_x, max_x]
    

    # print(f"original image shape:{image_3d.shape}, subset shape:{subset_3d.shape}")
    
    
    return subset_3d, max_min_ids


def dilation_binary_img_on_sub(input, margin, kernal_size, is_round=True):
    
    is_3d = (input.ndim == 3)
    if is_3d:
        
        if len(np.argwhere(input == True)) == 0:
            return input
        else:
            subset_3d , max_min_ids= get_sub_binary_image_by_pos(input, margin = margin)
            min_z, max_z, min_y ,max_y, min_x, max_x = max_min_ids

            if is_round:
                subset_3d = binary_dilation(subset_3d, ball(kernal_size))
            else:
                subset_3d = binary_dilation(subset_3d, cube(kernal_size))


            input[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1] = subset_3d
    else:

        if is_round:
            input = binary_dilation(input, disk(kernal_size))
        else:
            input = binary_dilation(input, square(kernal_size))
        
    
    return input

#### CCOMPS stuffs

def view_ccomps(input):
    subset_3d , _= get_sub_binary_image_by_pos(input, margin = 1)
    labeled_image = measure.label(subset_3d, background=0, return_num=False, connectivity=2)
    # Calculate the size of each connected component
    component_sizes = np.bincount(labeled_image.ravel())[1:]  # Skip the background count

    # Print the size of each component
    for i, size in enumerate(component_sizes, start=1):
        print(f"Component {i}: Size = {size}")

def keep_largest_ccomps(input):
    
    subset_3d , max_min_ids= get_sub_binary_image_by_pos(input, margin = 1)
    min_z, max_z, min_y ,max_y, min_x, max_x = max_min_ids
    
    labeled_image = measure.label(subset_3d, background=0, return_num=False, connectivity=2)
    # Calculate the size of each connected component
    # component_sizes = np.bincount(labeled_image.ravel())[1:]  # Skip the background count

    # Calculate the size of each connected component
    component_sizes = np.bincount(labeled_image.ravel())[1:]  # Skip the background count

    # Identify the largest component
    largest_component_label = np.argmax(component_sizes) + 1  # Labels are 1-based

    # Create a new binary image with only the largest component
    largest_component = (labeled_image == largest_component_label).astype(int)

    output = np.zeros(input.shape)

    output[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1] = largest_component

    return output

def keep_ccomps(input, top_n=None , threshold=0):
    
    subset_3d , max_min_ids= get_sub_binary_image_by_pos(input, margin = 1)
    min_z, max_z, min_y ,max_y, min_x, max_x = max_min_ids
    
    labeled_image = measure.label(subset_3d, background=0, return_num=False, connectivity=2)
    # Calculate the size of each connected component
    # component_sizes = np.bincount(labeled_image.ravel())[1:]  # Skip the background count

    # Calculate the size of each connected component
    component_sizes = np.bincount(labeled_image.ravel())[1:]  # Skip the background count

    if threshold==0:
        component_labels = np.unique(labeled_image)[1:]
    else:  
        # Get component labels that are above the specified size
        component_labels = np.where(component_sizes >= threshold)[0] + 1  # Labels are 1-based

    if top_n is not None:
        largest_labels = component_labels[np.argsort(component_sizes[component_labels - 1])[-top_n:]]
    else:
        largest_labels = component_labels

    # Create an output image with only the desired components retained
    output_sub = np.zeros_like(subset_3d)
    for label in largest_labels:
        output_sub[labeled_image == label] = 1
        
    output = np.zeros_like(input)
    output[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1] = output_sub
    
    return output


def get_ccomps_with_size_order(volume, segments=None, min_vol = None):
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
    
    
    
    component_labels = np.unique(labeled_image)[1:]
    if min_vol is not None:
        valid_components = component_sizes >= min_vol
        component_labels = component_labels[valid_components]
        component_sizes = component_sizes[valid_components]
        
    component_sizes_sorted = -np.sort(-component_sizes,)
    if segments is None:
        segments = len(component_sizes_sorted)
    # props = measure.regionprops(label_image)
    
    largest_labels = component_labels[np.argsort(component_sizes[component_labels - 1])[::-1][:segments]]
    
    output = np.zeros_like(labeled_image)
    for label_id, label in enumerate(largest_labels):
        output[labeled_image == label] = label_id+1
    
    return output, component_sizes_sorted[:segments]


def erosion_binary_img_on_sub(input, kernal_size = 1, footprint='ball'):
    
    if footprint == 'ball' or footprint == 'cube':
        assert type(kernal_size) is int
    
    is_3d = (input.ndim == 3)
    if is_3d:
        if len(np.argwhere(input == True)) == 0:
            return input
        else:
            subset_3d , max_min_ids= get_sub_binary_image_by_pos(input, margin = 1)
            min_z, max_z, min_y ,max_y, min_x, max_x = max_min_ids

            if footprint == 'ball':
                subset_3d = binary_erosion(subset_3d, ball(kernal_size))
            elif footprint == 'cube':
                subset_3d = binary_erosion(subset_3d, cube(kernal_size))
            elif footprint =='ball_XY':
                subset_3d = binary_erosion(subset_3d, ball_fp_XY)
            elif footprint =='ball_XZ':
                subset_3d = binary_erosion(subset_3d, ball_fp_XZ)
            elif footprint =='ball_YZ':
                subset_3d = binary_erosion(subset_3d, ball_fp_YZ)
            elif footprint =='X':
                subset_3d = binary_erosion(subset_3d, fp_X)
            elif footprint =='Y':
                subset_3d = binary_erosion(subset_3d, fp_Y)
            elif footprint =='Z':
                subset_3d = binary_erosion(subset_3d, fp_Z)
            elif footprint =='2XZ_1Y':
                subset_3d = binary_erosion(subset_3d, ball_fp_XZ)
                subset_3d = binary_erosion(subset_3d, ball_fp_XZ)
                subset_3d = binary_erosion(subset_3d, fp_Y)
            elif footprint =='2XY_1Z':
                subset_3d = binary_erosion(subset_3d, ball_fp_XY)
                subset_3d = binary_erosion(subset_3d, ball_fp_XY)
                subset_3d = binary_erosion(subset_3d, fp_Z)
            elif footprint =='2YZ_1X':
                subset_3d = binary_erosion(subset_3d, ball_fp_YZ)
                subset_3d = binary_erosion(subset_3d, ball_fp_YZ)
                subset_3d = binary_erosion(subset_3d, fp_X)
            
            input[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1] = subset_3d
    else:
        if footprint == 'disk':
            input = binary_erosion(input, disk(kernal_size))
        elif footprint == 'square':
            input = binary_erosion(input, square(kernal_size))
        # input = binary_erosion(input, square(kernal_size))
    
    return input

def erosion_binary_img_on_sub_custom(input, footprint):
    
    is_3d = (input.ndim == 3)
    if is_3d:
        subset_3d , max_min_ids= get_sub_binary_image_by_pos(input, margin = 1)
        min_z, max_z, min_y ,max_y, min_x, max_x = max_min_ids

        subset_3d = binary_erosion(subset_3d, footprint)

        
        input[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1] = subset_3d
    else:
        subset_3d = binary_erosion(subset_3d, footprint)
        # input = binary_erosion(input, square(kernal_size))
    
    return input


def closing_binary_img_on_sub(input, margin, kernal_size, is_round=True):
    
    is_3d = (input.ndim == 3)
    if is_3d:
        if len(np.argwhere(input == True)) == 0:
            return input
        else:
        
            subset_3d , max_min_ids= get_sub_binary_image_by_pos(input, margin = margin)
            min_z, max_z, min_y ,max_y, min_x, max_x = max_min_ids

            if is_round:
                subset_3d = binary_closing(subset_3d, ball(kernal_size))
            else:
                subset_3d = binary_closing(subset_3d, cube(kernal_size))

            # subset_3d = binary_closing(subset_3d, cube(kernal_size))
            input[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1] = subset_3d
    else:
        if is_round:
            input = binary_closing(input, disk(kernal_size))
        else:
            input = binary_closing(input, square(kernal_size))
    
    return input

def closing_binary_img_on_sub_one_step_iter(input, iterations, footprint='ball'):
    if footprint =='ball' :
        kernal_size=1
        is_round = True
    elif footprint =='cube':
        kernal_size=3
        is_round = False

    for i in range(iterations):
        input = dilation_binary_img_on_sub(input, margin=2,kernal_size=kernal_size,is_round=is_round)
        
    for i in range(iterations):    
        input = erosion_binary_img_on_sub(input,kernal_size=kernal_size, footprint=footprint)
    return input
        


def find_gaps_between_two(input_1, input_2, background = None):
       
    
    ### Goal: Close holes
    ## Method: Apply close to Input 1 and 2
    # close_size = 9
    # margin = ((close_size-1)//2) +1
    
    # input_1 = closing_binary_img_on_sub(input_1,margin=margin, kernal_size=close_size)
    # input_2 = closing_binary_img_on_sub(input_2,margin=margin, kernal_size=close_size)
    
    # close_iters = 4
    
    # input_1 = closing_binary_img_on_sub_one_step_iter(input_1,close_iters)
    # input_2 = closing_binary_img_on_sub_one_step_iter(input_2,close_iters)
    
    combined = np.logical_or(input_1,input_2)   
    ### Goal: find the link between Input 1 and 2


    ## Method1: Apply dilation to Input 1 and 2, 
    ## and find the intersection between two inputs
    dilation_iter = 7
    dilation_size=1

    for i in range(dilation_iter):
        # prev_suture_inter_sum = suture_inter_sum
        input_1 = dilation_binary_img_on_sub(input_1, margin=2, kernal_size=dilation_size)
        input_2 = dilation_binary_img_on_sub(input_2, margin=2, kernal_size=dilation_size)
        
    
    inter_dilation = np.logical_and(input_1, input_2)
    if np.sum(inter_dilation)==0:
        # print("No intersection have been found")
        return None
    
    ## Method2: Apply close to the segmetnation of both input1 and 2.
    ## In order to find if it is possible to close any gap between 1 and 2
    # close_size = 11   
    # margin = ((close_size-1)//2) +1
    # input_for_closing = combined.copy()
    # combined_for_closing = closing_binary_img_on_sub(input_for_closing,margin=margin, kernal_size=close_size)

    input_for_closing = combined.copy()
    close_iters = 7
    combined_for_closing = closing_binary_img_on_sub_one_step_iter(input_for_closing,close_iters)

    inter_dilation_and_closing = np.logical_and(inter_dilation,
                                                combined_for_closing)
    
    ### Goal: Make the final result more smooth and grown
    ## Dilation on the intersection a little
    # if np.sum(inter_dilation_and_closing)==0:
    #     print("No intersection have been found")
    #     return None
    
    # dilation_iter = 1
    # dilation_size=1
    # for i in range(dilation_iter):
    #     # prev_suture_inter_sum = suture_inter_sum
    #     inter_dilation_and_closing = dilation_binary_img_on_sub(inter_dilation_and_closing, 
    #                                                             margin=2, kernal_size=dilation_size)
    

    ### Goal: Refine the intersection
    ### (1) Use Results from Find intersection 1 and 2
    ### (2) Remove result areas that are parts of input1 and input2
    
    inter_dilation_and_closing[combined==True]=False
    if background is not None:
        inter_dilation_and_closing[background==True]=False
        
    if np.sum(inter_dilation_and_closing)==0:
        print("No intersection have been found")
        return None
    
    return inter_dilation_and_closing

def binary_to_colour_stack(input):
    """Turn binary to colour 3D stack [D, H, W, 3]

    Args:
        input (_type_): Have values from 1 to 6
        background, ROI 1, ROI 2, gap, other ROI, compare
    """
    
    colors = np.array([
        [0, 0, 0], # black
        ((100)*3),    # Mid grey
        (50, 50, 50),    # dark grey
        (255, 0, 0),    # red
        (211, 211, 211), # Light Gray
        (255, 182, 193)  # light red
    ])
    shape_rgb = input.shape + (3,)
    output_rgb = np.zeros(shape_rgb, dtype=np.uint8)
    for i in range(6):
        output_rgb[input == i] = colors[i]     
        
    return output_rgb
    


def erosion_suture(suture, bone, bg):
    """Erode suture area/volume if elements touch background
    Don't erode if it touches bone

    Args:
        suture (_type_): _description_
        bone (_type_): _description_
        bg (_type_): 
    """
    
    return


######### For ero bones

def find_seed_by_ero(volume_array, threshold , segments, ero_iter, 
                    output_dir, 
                    ero_shape = 'ball',
                    SAVE_ALL_ERO= True):
     # Capture the start time
    start_time = datetime.now()
    
    volume_label = volume_array > threshold
    
    log_dict = {"Method": "find_seed_by_ero",
                  "volume_array shape": list(volume_array.shape),
                  "threshold": threshold,
                  "segments": segments,
                  "ero_shape": ero_shape,
                  "Whole Volume": str(np.sum(volume_label))
                  }
    
    for i_iter in range(0,ero_iter+1):
        
        if i_iter!=0:
            volume_label = erosion_binary_img_on_sub(volume_label, kernal_size = 1)
        
        seed_file = os.path.join(output_dir , 
                            f"seed_ero_{i_iter}_thre{threshold}_segs_{segments}.tif")
        
        seed, ccomp_sizes = get_ccomps_with_size_order(volume_label,segments)
        tifffile.imwrite(seed_file, seed,
                         compression ='zlib')
        
        args_dict = {
            # "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            # "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            # "duration": str(duration),
            f'size_top_{segments}_ccomp': ccomp_sizes.tolist(),
            f'number_of_ccopms_found': len(ccomp_sizes.tolist()),
            'output_seed_dir': output_dir
        }
        
        log_dict[f"ero_iter{i_iter}"]=args_dict
        # .append(args_dict)
    
        
    # seed, ccomp_sizes = get_ccomps_with_size_order(volume_label,segments)
    
    # Capture the end time
    end_time = datetime.now()
    # Calculate the duration
    duration = end_time - start_time   
    
    log_dict["start_time"] =   start_time.strftime("%Y-%m-%d %H:%M:%S") 
    log_dict["end_time"] =   end_time.strftime("%Y-%m-%d %H:%M:%S")
    log_dict["duration"] =   str(duration)
     
        
    return log_dict


def find_seed_by_ero_custom(volume_array, threshold , segments, ero_iter, 
                    output_dir, 
                    ero_shape = 'ball',
                    SAVE_ALL_ERO= True,
                    footprints = None):
     # Capture the start time
    start_time = datetime.now()
    
    volume_label = volume_array > threshold
    
    log_dict = {"Method": "find_seed_by_ero",
                  "volume_array shape": list(volume_array.shape),
                  "threshold": threshold,
                  "segments": segments,
                  "ero_shape": ero_shape,
                  "Whole Volume": str(np.sum(volume_label))
                  }
    

    
    for i_iter in range(0,ero_iter+1):
        
        if i_iter!=0:
            volume_label = erosion_binary_img_on_sub(volume_label, 
                                                                   footprint = footprints[i_iter-1])
                                                                #    footprint='ball_YZ')
        
        seed_file = os.path.join(output_dir , 
                            f"seed_ero_{i_iter}_thre{threshold}_segs_{segments}.tif")
        
        seed, ccomp_sizes = get_ccomps_with_size_order(volume_label,segments)
        tifffile.imwrite(seed_file, seed,
                         compression ='zlib')
        
        args_dict = {
            # "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            # "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            # "duration": str(duration),
            f'size_top_{segments}_ccomp': ccomp_sizes.tolist(),
            f'number_of_ccopms_found': len(ccomp_sizes.tolist()),
            'output_seed_dir': output_dir
        }
        
        log_dict[f"ero_iter{i_iter}"]=args_dict
        # .append(args_dict)
    
        
    # seed, ccomp_sizes = get_ccomps_with_size_order(volume_label,segments)
    
    # Capture the end time
    end_time = datetime.now()
    # Calculate the duration
    duration = end_time - start_time   
    
    log_dict["start_time"] =   start_time.strftime("%Y-%m-%d %H:%M:%S") 
    log_dict["end_time"] =   end_time.strftime("%Y-%m-%d %H:%M:%S")
    log_dict["duration"] =   str(duration)
     
        
    return log_dict


### For grow 

def dilation_one_iter(input_mask, threshold_binary, 
                             touch_rule = 'stop',
                             segments=None, ero_shape = 'ball'):
    label_id_list = np.unique(input_mask)
    label_id_list = label_id_list[label_id_list!=0]
    
    result = input_mask.copy()  
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
    #         dilated_binary_label_id = suture_morpho.dilation_binary_img_on_sub(dilated_binary_label_id, 
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

def binary_stack_to_mesh(input_volume , threshold, downsample_scale=20,
                         face_color= [128]*4):
    
    input_volume = input_volume > threshold
    verts, faces, normals, _ = marching_cubes(input_volume, level=0.5)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    target_faces = mesh.faces.shape[0] // downsample_scale
    simplified_mesh = mesh.simplify_quadric_decimation(target_faces)

    simplified_mesh.visual.face_colors = face_color
    return simplified_mesh

# import numpy as np
# import tifffile

# # Example 3D array (e.g., read from a TIFF file)
# array_3d = np.random.randint(0, 11, (6, 500, 500))  # Example 3D array with values from 0 to 10

# # Define 10 colors (R, G, B)
# colors = np.array([
#     [255, 0, 0],    # Red
#     [0, 255, 0],    # Green
#     [0, 0, 255],    # Blue
#     [255, 255, 0],  # Yellow
#     [255, 0, 255],  # Magenta
#     [0, 255, 255],  # Cyan
#     [128, 0, 0],    # Maroon
#     [0, 128, 0],    # Dark Green
#     [0, 0, 128],    # Navy
#     [128, 128, 128], # Gray
#     [255, 255, 255] # black
# ])

# # Add an extra dimension for RGB channels
# shape_4d = array_3d.shape + (3,)
# array_4d = np.zeros(shape_4d, dtype=np.uint8)
# print(array_4d.shape)
# # Assign colors based on the value in the original 3D array
# for i in range(11):
#     array_4d[array_3d == i] = colors[i]

# # Save the 4D array as a TIFF file
# # tifffile.imwrite('colored_image.tiff', array_4d)

# # Save the 4D array as a TIFF file
# tifffile.imwrite('data/bones_suture/output_test_colour.tif', array_4d)