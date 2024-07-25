import numpy as np
# from PIL import Image
from tifffile import imread, imwrite
import traceback
import os,sys

import glob
import threading
import time
import json, yaml

# Add the lib directory to the system path
import suture_morph.suture_morpho as suture_morpho


def load_config_yaml(config, parent_key=''):
    for key, value in config.items():
        if isinstance(value, dict):
            load_config_yaml(value, parent_key='')
        else:
            globals()[parent_key + key] = value

def separate_chambers(file_path,
                      base_name,
                      output_folder,
                      n_iters, 
                      segments,
                      no_split_limit =3,
                      min_vol=10,
                      prop_thre = 0.1,
                      save_every_iter = False
                      ):
    segments = 25
    max_splits = segments
    
    img = imread(file_path)
    img = img==255

    seed, _ = suture_morpho.get_ccomps_with_size_order(img,segments, min_vol = min_vol)

    bone_ids = np.unique(seed)
    bone_ids = [int(value) for value in bone_ids if value != 0]
    ori_bone_ids = bone_ids.copy()
    max_seed_id = int(np.max(bone_ids))
    
    bone_ids_dict = {}
    for value in bone_ids :
        bone_ids_dict[value] = [value]
        
    split_log = {}
    
    ero_img = img.copy()
    
    no_consec_split_count = 0

    for n_iter in range(n_iters):
        
        split_log[n_iter] = {}
        
        print(f"working on erosion {n_iter}")
        ero_img = suture_morpho.erosion_binary_img_on_sub(ero_img, kernal_size = 1)
        
        ero_seed, _ = suture_morpho.get_ccomps_with_size_order(ero_img,segments)

        ero_bone_ids = np.unique(ero_seed)
        ero_bone_ids = [int(value) for value in ero_bone_ids if value != 0]

        bone_ids = np.unique(seed)
        bone_ids = [int(value) for value in bone_ids if value != 0]

        
        has_split = False
        ## Comparing each ccomp from eroded seed
        ## to each ccomp from the original seed
        for bone_id in bone_ids:
            comp = seed == bone_id

            inter_count = 0
            inter_ids = np.array([])
            inter_props = np.array([])
            
            
            for ero_bone_id in ero_bone_ids:
                ero_comp = ero_seed == ero_bone_id
                
                # start_time = time.time()
                # inter = np.sum(np.logical_and(comp,ero_comp))
                # end_time = time.time()
                # elapsed_time = end_time - start_time
                # print(f"Elapsed time: {elapsed_time:.6f} seconds:value:{inter}")
                # start_time = time.time()
                inter = np.sum(comp[ero_comp])
                # end_time = time.time()
                # elapsed_time = end_time - start_time
                # print(f"Elapsed time: {elapsed_time:.6f} seconds:value:{inter}")
                # prop = round(np.sum(np.logical_and(comp,ero_comp)) / np.sum(comp),4)
                # print(f"{bone_id} for ero {ero_bone_id} has intersect {inter}\nprop{prop}")
                
                
                if inter>0:
                    inter_count+=1
                    inter_ids = np.append(inter_ids , ero_bone_id)
                    prop = round(inter / np.sum(comp),4)*100
                    inter_props = np.append(inter_props, prop)

                    # prop = round(np.sum(np.logical_and(comp,ero_comp)) / np.sum(comp),4)
            ## When a ccomps has been split into multiple ccomps in the next ero step.   
            if inter_count>1:
                # prop_thre = 0.1
               
                print(f'{bone_id} has been split to {inter_count} parts. Ids are {inter_ids}')
                print(f"props are: {inter_props}")
                print(f"Remove parts that have proportion smaller than prop_thre")
                seed[seed == bone_id] =0
                inter_ids = inter_ids[inter_props>prop_thre]
                
                new_ids = []
                for inter_id in inter_ids:
                    seed[ero_seed == inter_id] = max_seed_id+1
                    
                    # split_list = bone_ids_dict[bone_id]
                    # if len(split_list) <= max_splits:
                    #     bone_ids_dict[bone_id].append(max_seed_id+1)
                    for key,value in bone_ids_dict.items():
                        if bone_id in value:
                            if len(value) <= max_splits:
                                bone_ids_dict[key].append(max_seed_id+1)
                                new_ids.append(max_seed_id+1)
                                break
                                
                    max_seed_id +=1
                
                split_log[n_iter][bone_id] = new_ids
                    
                has_split = True
                
            # elif inter_count==0:
            #     print(f'{bone_id} has been completely erode')

        if has_split:
            no_consec_split_count=0
        else:
            no_consec_split_count+=1
            
        
        if no_consec_split_count>=no_split_limit:
            print(f"detect non split for {no_consec_split_count}rounds")
            print(f"break loop at {n_iter} iter")
            break
    

    output_log = {"ori_comps_to_final":bone_ids_dict,
                  "split_each_iter":split_log}
    with open(os.path.join(output_folder,base_name+'.json'), 'w') as file:
        json.dump(output_log, file, indent=4)
    imwrite(os.path.join(output_folder,base_name+'.tif'), seed, 
        compression ='zlib')
             
    return seed,bone_ids_dict, split_log

def gen_mesh(volume, threshold, output_path):

    # output_path = os.path.join(output_dir, f"{threshold}.ply")
    if os.path.isfile(output_path):
        return
    else:
        output = suture_morpho.binary_stack_to_mesh(volume, threshold)
        output.export(output_path)
def foram_seed_mp(file_paths, output_folder):
    for file_path in file_paths:
        print(file_path)
        
        # volume = imread(file_path)
        base_name = os.path.basename(file_path)
        output_dir = os.path.join(output_folder,base_name)
        os.makedirs(output_dir, exist_ok=True)
        
        
        separate_chambers(file_path, 
                        output_folder = output_dir,
                        base_name = base_name,
                        n_iters=n_iters,segments=segments)
        # gen_mesh(volume, 0, os.path.join('result/foram_james/whole_mesh/',
        #                                  base_name+'.ply'))

if __name__ == "__main__":        
    # file_path = '/result/foram_james/input/ai/final.20180802_VERSA_1905_ASB_OLK_st016_bl4_fo1_recon.tif'    

    # file_paths = glob.glob('result/foram_james/input/ai/*.tif')
    # output_folder = 'result/foram_james/thre_ero_seed/'

    config_path = './make_seeds_foram.yaml'
    _, extension = os.path.splitext(config_path)
    print(f"processing config he file {config_path}")
    if extension == '.yaml':
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        load_config_yaml(config)

    num_threads = 5
    file_paths = glob.glob(os.path.join(input_folder,"*.tif"))
    
    
    threads = []
    sub_file_paths = [file_paths[i::num_threads] for i in range(num_threads)]
    # Start a new thread for each sublist
    for sublist in sub_file_paths:
        thread = threading.Thread(target=foram_seed_mp, args=(sublist,output_folder))
        threads.append(thread)
        thread.start()
        
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    sys.exit()
    #### For single file instance
    
    file_path = './result/foram_james/input/ai_v2/final.20180719_VERSA_1905_ASB_OLK_st014_bl4_fo1_recon.tif'
    output_folder = './result/foram_james/seed_ai_v2/'
    base_name = os.path.basename(file_path)
    seed ,bone_ids_dict , split_log=  separate_chambers(file_path, 
                                                        output_folder = output_folder,
                                                        base_name = base_name,
                                                        n_iters=2,segments=25)
    
    # imwrite(os.path.join('./result/foram_james/seed_ai_v2/',base_name+'.tif'), seed, 
    #     compression ='zlib')
    # bone_ids_dict = {int(key): str(value) for key, value in bone_ids_dict.items()}
    
# separate_chambers(file_path)

