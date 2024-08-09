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
    workspace = r'C:\Users\Yichen\OneDrive\work\codes\nhm_bounti_pipeline\result\procavia'
    file_path = os.path.join(workspace, 'thre_ero_seed/thre_ero_2iter_thre4500.tif')
    file_2_path = os.path.join(workspace, 'thre_ero_seed/thre_ero_3iter_thre4500.tif')

    img_ids =[5,6]



    output_dir = os.path.join(workspace, 'thre_ero_seed/merge')
    os.makedirs(output_dir,exist_ok=True)
    output_path = os.path.join(output_dir, 'thre4500_merge_iter2_4.tif')



    # Read the TIFF file
    img = imread(file_path)

    binary_img = np.isin(img, img_ids)

    seed, ccomp_sizes = BounTI.get_ccomps_with_size_order(binary_img,30)

    imwrite(output_path, seed,compression ='zlib')
