import numpy as np
# from PIL import Image
from tifffile import imread, imwrite
import traceback
import os,sys

lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../sprout_core'))
sys.path.insert(0, lib_path)
import sprout_core

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

# Example usage
# file_path = 'example.tiff'
# workspace = r'C:\Users\Yichen\OneDrive\work\codes\nhm_bounti\result\manual_input_seeds\Cebus_apella'
# file_path = os.path.join(workspace, 'comp_seed_ero_2_thre180_segs_40.to-labelfield-8_bits.tif')

workspace = r'C:\Users\Yichen\OneDrive\work\codes\nhm_bounti\result\manual_input_seeds\Talpa_europaea'
file_path = os.path.join(workspace, 'comp_seed_ero_1_thre42000_segs_40.to-labelfield-8_bits.tif')

# workspace = r'C:\Users\Yichen\OneDrive\work\codes\nhm_bounti\result\manual_input_seeds\Phacochoerus_aethiopicus'
# file_path = os.path.join(workspace, 'comp_seed_ero_3_thre115_segs_40.to-labelfield-8_bits.tif')

base_name = os.path.basename(file_path)

output_dir = os.path.join(workspace, 'output')
os.makedirs(output_dir,exist_ok=True)
output_path = os.path.join(output_dir, f'{base_name}_split.tif')



# Read the TIFF file
img = imread(file_path)

to_check_ids = [1]
n_ccomps = [6]

# to_check_ids = [1,15]
# n_ccomps = [6,2]

JUST_SPLIT_ID = False

if JUST_SPLIT_ID:
    img_output = np.zeros_like(img)
else:
    img_output = img.copy()

for to_check_id,n_ccomp in zip(to_check_ids, n_ccomps):
    
    binary_img = img == to_check_id
    
    max_id = np.max(img_output)
    # print(f"max_id:{max_id}")
    seed, ccomp_sizes = sprout_core.get_ccomps_with_size_order(binary_img, n_ccomp)
    print(f"size of split comps:{ccomp_sizes}")
   
    seed_offset = np.where(seed != 0, seed + max_id, 0)
    print(np.sum(img_output == to_check_id))
    if JUST_SPLIT_ID:
        img_output = img_output + seed_offset
    else:
        img_output = np.where(img_output != to_check_id, img_output, seed_offset)


imwrite(output_path, img_output,compression ='zlib')
