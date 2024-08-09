import numpy as np
# from PIL import Image
from tifffile import imread, imwrite
import traceback
import os

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
    workspace = r'C:\Users\Yichen\OneDrive\work\codes\nhm_bounti_pipeline\result\procavia'
    file_path = os.path.join(workspace, 'thre_ero_seed/thre_ero_2iter_thre4500.tif')

    output_dir = os.path.join(workspace, 'thre_ero_seed/merge')
    os.makedirs(output_dir,exist_ok=True)
    output_path = os.path.join(output_dir, 'thre_ero_2iter_thre4500_merge.tif')

    values_to_replace = [7]

    ## Set the target value to remove the values_to_replace
    target_value = 2

    # Read the TIFF file
    img_array = imread(file_path)
    # img_array = read_tiff_as_array(file_path)

    # If the image was read successfully, proceed with value replacement
    if img_array is not None:
        modified_array = replace_values_in_array(img_array, values_to_replace, target_value)
        if modified_array is not None:
            print("Value replacement successful.")
            
            imwrite(output_path, modified_array,compression ='zlib')
            # Save or further process the modified_array as needed
        else:
            print("Value replacement failed.")
    else:
        print("Failed to read the TIFF file.")
