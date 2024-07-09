# import nibabel as nib
import os, sys
from skimage.morphology import ball
import tifffile
from datetime import datetime
from skimage.measure import marching_cubes
import json ,yaml


import suture_morph.suture_morpho as suture_morpho

# Function to recursively create global variables from the config dictionary
def load_config_yaml(config, parent_key=''):
    for key, value in config.items():
        if isinstance(value, dict):
            load_config_yaml(value, parent_key='')
        else:
            globals()[parent_key + key] = value
            

# Define a function to read the configuration and set variables dynamically
def load_config_json(file_path):
    with open(file_path, 'r') as config_file:
        config = json.load(config_file)

    # Dynamically set variables in the global namespace
 
    for key, value in config.items():
        globals()[key] = value


if __name__ == "__main__":
    
    
    ############ Config
    file_path = './make_grow_result.yaml'
    
    _, extension = os.path.splitext(file_path)
    print(f"processing config he file {file_path}")
    if extension == '.json':
        
        load_config_json(file_path)
    elif extension == '.yaml':
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        load_config_yaml(config)
    
    # workspace = r'C:\Users\Yichen\OneDrive\work\codes\nhm_bounti_pipeline\result\procavia'
    
    # tif_file = os.path.join(workspace, 'thre_ero_seed/thre_ero_2iter_thre4500.tif')
    # # tif_file = os.path.join(workspace, 'test/input/thre_ero_2iter_thre4500.tif')
    
    # ori_path = os.path.join(workspace, 'input/procaviaH4981C_0001.tif.resampled_400_600.tif')
    
    # # ori_path = os.path.join(workspace, 'input/procaviaH4981C_0001.tif.resampled.tif')
      
    # ## Test by decrease the label values
    # # input_mask = np.where(np.isin(input_mask, [4,5,8,9,11,17,18,20 ]) , input_mask, 0)
    # # input_mask = np.where(np.isin(input_mask, [5,6,10,11]) , input_mask, 0)
    
    # workspace = r'C:\Users\Yichen\OneDrive\work\codes\nhm_bounti\result\foram_james'
    # ori_path = os.path.join(workspace, 'input/ai/final.20180802_VERSA_1905_ASB_OLK_st016_bl4_fo1_recon.tif')
    # tif_file = os.path.join(workspace, 'seed_test/seed_ero_3_thre0_segs_20.tif')
    # thresholds = [0]
    # dilate_iters = 4
    # touch_rule = "no"
    ori_path = os.path.join(workspace, 
                            ori_path)
    tif_file = os.path.join(workspace, tif_file)
    base_name = os.path.basename(tif_file)
    input_mask = tifffile.imread(tif_file)
    ori_mask = tifffile.imread(ori_path)

    output_folder = os.path.join(workspace, output_folder)
    os.makedirs(output_folder , exist_ok=True)
    
    for threshold in thresholds:
        for dilate_iter in range(1, dilate_iters+1, save_interval):
            threshold_binary = ori_mask > threshold

            result = suture_morpho.dilation_one_iter(input_mask, threshold_binary ,
                                              touch_rule = touch_rule)
            
            
            # output_path = os.path.join(workspace, f'result/ai/dila_{dilate_iter}_{threshold}_rule_{touch_rule}.tif')
            output_path = os.path.join(output_folder, f'{base_name}_dilate_{dilate_iter}_thre_{threshold}.tif')
            tifffile.imwrite(output_path, 
            result,
            compression ='zlib')
    
    
    
