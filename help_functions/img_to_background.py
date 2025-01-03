import numpy as np
# from PIL import Image
from tifffile import imread, imwrite
import traceback
import os
import BounTI

def replace_values_in_array(arr, values_to_replace, target_value):
    try:
        # Check if the input is a NumPy array
        if not isinstance(arr, np.ndarray):
            raise ValueError("Input is not a NumPy array")

        # Check if the array is 3D
        if arr.ndim != 3:
            raise ValueError("Input array is not 3-dimensional")

        # Check if values_to_replace is a list
        if not isinstance(values_to_replace, list):
            raise ValueError("values_to_replace should be a list")

        # Perform the replacement
        for value in values_to_replace:
            arr[arr == value] = target_value

        return arr
    except Exception as e:
        print(f"Error processing the array: {e}")
        traceback.print_exc()
        return None
if __name__ == "__main__":   
    # Example usage
    # file_path = 'example.tiff'
    workspace = r'C:\Users\Yichen\OneDrive\work\codes\nhm_bounti_pipeline\result\foram_james'
    file_1_path = os.path.join(workspace, 'thre_ero_seed/thre_ero_2iter_thre4500.tif')
    file_2_path = os.path.join(workspace, 'thre_ero_seed/thre_ero_3iter_thre4500.tif')

    img1_ids =[5,6]
    img2_ids = [1,2,3,4,5,6]


    output_dir = os.path.join(workspace, 'thre_ero_seed/merge')
    os.makedirs(output_dir,exist_ok=True)
    output_path = os.path.join(output_dir, 'thre4500_merge_iter2_4.tif')



    # Read the TIFF file
    img1 = imread(file_1_path)
    img2 = imread(file_2_path)

    binary_img1 = np.isin(img1, img1_ids)
    binary_img2 = np.isin(img2, img2_ids)

    result = np.logical_or(binary_img1, binary_img2)

    seed, ccomp_sizes = BounTI.get_ccomps_with_size_order(result,30)

    imwrite(output_path, seed,compression ='zlib')

    # img_array = read_tiff_as_array(file_path)

    # # If the image was read successfully, proceed with value replacement
    # if img_array is not None:
    #     modified_array = replace_values_in_array(img_array, values_to_replace, target_value)
    #     if modified_array is not None:
    #         print("Value replacement successful.")
            
    #         imwrite(output_path, modified_array)
    #         # Save or further process the modified_array as needed
    #     else:
    #         print("Value replacement failed.")
    # else:
    #     print("Failed to read the TIFF file.")
