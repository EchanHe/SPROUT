import numpy as np
# from PIL import Image
from tifffile import imread, imwrite
import traceback
import os
from sprout_core import sprout_core


if __name__ == "__main__":   
    ### Input parameter
    workspace = r'C:\Users\Yichen\OneDrive\work\codes\nhm_bounti_pipeline\result\procavia'
    file_path = os.path.join(workspace, 'thre_ero_seed/thre_ero_2iter_thre4500.tif')

    img_ids =[5,6]
    segments = 30
    
    output_dir = os.path.join(workspace, 'thre_ero_seed/merge')
    os.makedirs(output_dir,exist_ok=True)
    output_path = os.path.join(output_dir, 'thre4500_merge_iter2_4.tif')
    
    #####


    # Read the TIFF file
    img = imread(file_path)

    binary_img = np.isin(img, img_ids)

    seed, ccomp_sizes = sprout_core.get_ccomps_with_size_order(binary_img,segments)

    imwrite(output_path, seed,compression ='zlib')
