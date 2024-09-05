import numpy as np
import pandas as pd
# from PIL import Image
from tifffile import imread, imwrite
import os,sys

import glob
import threading
lock = threading.Lock()
import time
import json, yaml

# Add the lib directory to the system path
import suture_morph.suture_morpho as suture_morpho
from suture_morph.suture_morpho import reorder_segmentation



def load_config_yaml(config, parent_key=''):
    for key, value in config.items():
        if isinstance(value, dict):
            load_config_yaml(value, parent_key='')
        else:
            globals()[parent_key + key] = value

def detect_inter(ccomp_combine_seed,ero_seed, seed_ids, inter_log , lock):
    for seed_id in seed_ids:
        seed_ccomp = ero_seed == seed_id
                    
        # Check if there are intersection using the sum of intersection
        inter = np.sum(ccomp_combine_seed[seed_ccomp])

        with lock:
            if inter>0:
                inter_log["inter_count"] +=1
                
                inter_log["inter_ids"] = np.append(inter_log["inter_ids"] , seed_id)
                
                prop = round(inter / np.sum(ccomp_combine_seed),4)*100
                inter_log["inter_props"] = np.append(inter_log["inter_props"], prop)

def merged_seeds_log_results(output_dict=None, **kwargs):
    if output_dict is None:
        # log_dict = {}  # Initialize a new dictionary if none is provided

        output_dict = {}

    for key, value in kwargs.items():
        output_dict[key] = value  # Assign the key-value pairs to the dictionary

    return output_dict

def make_seeds_merged_mp(img,
                      threshold,
                      output_folder,
                      n_iters, 
                      segments,
                      num_threads = 1,
                      no_split_limit =3,
                      min_size=5,
                      sort = True,
                      prop_thre = 0.01,
                      background = 0,
                      save_every_iter = False,
                      name_prefix = "comp_seed",
                      init_segments = None,
                      footprint = "ball",
                      ):

    output_name = f"{name_prefix}_thre_{threshold}_ero_{n_iters}"
    output_folder = os.path.join(output_folder, output_name)
    os.makedirs(output_folder,exist_ok=True)
    
    max_splits = segments
    
    img = img>=threshold

    if init_segments is None:
        init_segments = segments

    init_seed, _ = suture_morpho.get_ccomps_with_size_order(img,init_segments)
    
    output_img_name = f'thre_{threshold}_ero_0.tif'
    if save_every_iter:
        imwrite(os.path.join(output_folder,output_img_name), init_seed, 
            compression ='zlib')
    
    init_ids = [int(value) for value in np.unique(init_seed) if value != background]
    max_seed_id = int(np.max(init_ids))


    combine_seed = init_seed.copy()
    combine_seed = combine_seed.astype('uint16')


    output_dict = {"total_id": {0: len(np.unique(combine_seed))-1},
                   "split_id" : {0: {}},
                   "split_ori_id": {0: {}},
                   "split_ori_id_filtered":  {0: {}},
                   "split_prop":  {0: {}},
                   "cur_seed_name": {0: output_img_name}
                   }

    ori_combine_ids_map = {}
    for value in init_ids :
        ori_combine_ids_map[value] = [value]
    
    no_consec_split_count = 0
    

    for ero_iter in range(1, n_iters+1):
        
        
        print(f"working on erosion {ero_iter}")
        
        output_dict["split_id"][ero_iter] = {}
        output_dict["split_ori_id"][ero_iter] = {}
        output_dict["split_ori_id_filtered"][ero_iter] = {}
        output_dict["split_prop"][ero_iter] = {}
        
        img = suture_morpho.erosion_binary_img_on_sub(img, kernal_size = 1,footprint=footprint)
        seed, _ = suture_morpho.get_ccomps_with_size_order(img,segments)
        
        output_img_name = f'thre_{threshold}_ero_{ero_iter}.tif'
        output_dict["cur_seed_name"][threshold] = output_img_name
        
        if save_every_iter:
            output_path = os.path.join(output_folder, output_img_name)
            print(f"\tSaving {output_path}")
            imwrite(output_path, seed, 
                compression ='zlib')

        seed_ids = [int(value) for value in np.unique(seed) if value != background]
        combine_ids = [int(value) for value in np.unique(combine_seed) if value != background]
       
        has_split = False
        ## Comparing each ccomp from eroded seed
        ## to each ccomp from the original seed
        
        inter_log = {
            "inter_count":0,
            "inter_ids": np.array([]),
            "inter_props": np.array([])
        }
        
        for combine_id in combine_ids:
            ccomp_combine_seed = combine_seed == combine_id
            
            inter_log["inter_count"] = 0
            inter_log["inter_ids"] = np.array([])
            inter_log["inter_props"] = np.array([])
            
            sublists = [seed_ids[i::num_threads] for i in range(num_threads)]
             # Create a list to hold the threads
            threads = []
            for sublist in sublists:
                thread = threading.Thread(target=detect_inter, args=(ccomp_combine_seed,
                                                                     seed,
                                                                     sublist,
                                                                     inter_log,
                                                                     lock))
                threads.append(thread)
                thread.start()
                
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
        
        
            ## if there are any intersection between seed and the init seed
            if inter_log["inter_count"]>1:
                temp_inter_count = inter_log["inter_count"]
                temp_inter_ids = inter_log["inter_ids"]
                temp_inter_props = inter_log["inter_props"]
                
                print(f'\t{combine_id} has been split to {temp_inter_count} parts. Ids are {temp_inter_ids}')
                print(f"\tprops are: {temp_inter_props}")
                
                combine_seed[combine_seed == combine_id] =0
                filtered_inter_ids = temp_inter_ids[temp_inter_props>prop_thre]
                
                if len(filtered_inter_ids)>0:
                    has_split = True
                
                new_ids = []
                for inter_id in filtered_inter_ids:
                    max_seed_id +=1
                    
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
                
        
            output_dict["total_id"][ero_iter] = len(np.unique(combine_seed))-1
    
        if has_split:
            no_consec_split_count=0
        else:
            no_consec_split_count+=1
            
        
        if no_consec_split_count>=no_split_limit:
            print(f"detect non split for {no_consec_split_count}rounds")
            print(f"break loop at {ero_iter} iter")
            break
        

    output_path = os.path.join(output_folder,output_name+'.tif')
    print(f"\tSaving final output:{output_path}")
    imwrite(output_path, combine_seed, 
        compression ='zlib')
    
    combine_seed,_ = reorder_segmentation(combine_seed, min_size=min_size, sort_ids=sort)
    output_path = os.path.join(output_folder,output_name+'_sorted.tif')
    print(f"\tSaving final output:{output_path}")
    imwrite(output_path, combine_seed, 
        compression ='zlib')
    
             
    return combine_seed,ori_combine_ids_map, output_dict    


def make_seeds_merged_by_thres_mp(img,
                      thresholds,
                      output_folder,
                      n_iters, 
                      segments,
                      num_threads = 1,
                      no_split_limit =3,
                      min_size=5,
                      sort = True,
                      prop_thre = 0.01,
                      background = 0,
                      save_every_iter = False,
                      name_prefix = "comp_seed",
                      init_segments = None,
                      footprint = "ball",
                      min_split_prop = 0
                      ):

    output_name = f"{name_prefix}_ero_{n_iters}"
    output_folder = os.path.join(output_folder, output_name)
    os.makedirs(output_folder,exist_ok=True)
    
    max_splits = segments
    
    if init_segments is None:
        init_segments = segments


    mask = img>=thresholds[0]
    for ero_iter in range(1, n_iters+1):
        mask = suture_morpho.erosion_binary_img_on_sub(mask, kernal_size = 1,footprint=footprint)
    init_seed, _ = suture_morpho.get_ccomps_with_size_order(mask,init_segments)
    
    output_img_name = f'thre_{thresholds[0]}_ero_{n_iters}.tif'
    if save_every_iter:
        imwrite(os.path.join(output_folder,output_img_name), init_seed, 
            compression ='zlib')
            
    
    init_ids = [int(value) for value in np.unique(init_seed) if value != background]
    max_seed_id = int(np.max(init_ids))


    combine_seed = init_seed.copy()
    combine_seed = combine_seed.astype('uint16')

    output_dict = {"total_id": {0: len(np.unique(combine_seed))-1},
                   "split_id" : {0: {}},
                   "split_ori_id": {0: {}},
                   "split_ori_id_filtered":  {0: {}},
                   "split_prop":  {0: {}}, 
                   "cur_seed_name": {0: output_img_name}
                   }

    ori_combine_ids_map = {}
    for value in init_ids :
        ori_combine_ids_map[value] = [value]
    
    no_consec_split_count = 0
    

    for threshold in thresholds[1:]:
        print(f"working on thre {threshold}")
        
        output_dict["split_id"][threshold] = {}
        output_dict["split_ori_id"][threshold] = {}
        output_dict["split_ori_id_filtered"][threshold] = {}
        output_dict["split_prop"][threshold] = {}
        
        mask = img>=threshold
        for ero_iter in range(1, n_iters+1):
            mask = suture_morpho.erosion_binary_img_on_sub(mask, kernal_size = 1,footprint=footprint)
        seed, _ = suture_morpho.get_ccomps_with_size_order(mask,segments)
        
        output_img_name = f'thre_{threshold}_ero_{n_iters}.tif'
        output_dict["cur_seed_name"][threshold] = output_img_name
        if save_every_iter:
            output_path = os.path.join(output_folder,output_img_name)
            print(f"\tSaving {output_path}")
            imwrite(output_path, seed, compression ='zlib')
        
        seed_ids = [int(value) for value in np.unique(seed) if value != background]
        combine_ids = [int(value) for value in np.unique(combine_seed) if value != background]
        
        
        has_split = False
        ## Comparing each ccomp from eroded seed
        ## to each ccomp from the original seed
        
        inter_log = {
            "inter_count":0,
            "inter_ids": np.array([]),
            "inter_props": np.array([])
        }
        
        for combine_id in combine_ids:
            ccomp_combine_seed = combine_seed == combine_id
            
            inter_log["inter_count"] = 0
            inter_log["inter_ids"] = np.array([])
            inter_log["inter_props"] = np.array([])
            
            sublists = [seed_ids[i::num_threads] for i in range(num_threads)]
             # Create a list to hold the threads
            threads = []
            for sublist in sublists:
                thread = threading.Thread(target=detect_inter, args=(ccomp_combine_seed,
                                                                     seed,
                                                                     sublist,
                                                                     inter_log,
                                                                     lock))
                threads.append(thread)
                thread.start()
                
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
        
        
            ## if there are any intersection between seed and the init seed
            if inter_log["inter_count"]>1:
                temp_inter_count = inter_log["inter_count"]
                temp_inter_ids = inter_log["inter_ids"]
                temp_inter_props = inter_log["inter_props"]
                
                sum_inter_props = np.sum(temp_inter_props)
                print(f'\t{combine_id} has been split to {temp_inter_count} parts. Ids are {temp_inter_ids}')
                print(f"\tprops are: {temp_inter_props}")
                
                print(f"\tSplit prop is {sum_inter_props}")
                if sum_inter_props<min_split_prop:
                    has_split = False
                else:
                    combine_seed[combine_seed == combine_id] =0
                    filtered_inter_ids = temp_inter_ids[temp_inter_props>prop_thre]
                    
                    



                    if len(filtered_inter_ids)>0:
                        has_split = True
                    
                    new_ids = []
                    for inter_id in filtered_inter_ids:
                        max_seed_id +=1
                        
                        combine_seed[seed == inter_id] = max_seed_id
                        # new_ids.append(max_seed_id)     
                        
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
                
        
            output_dict["total_id"][threshold] = len(np.unique(combine_seed))-1
    
        if has_split:
            no_consec_split_count=0
        else:
            no_consec_split_count+=1
            
        if no_consec_split_count>=no_split_limit:
                print(f"detect non split for {no_consec_split_count}rounds")
                print(f"break loop at {threshold} threshold")
                break
        

    
    output_path = os.path.join(output_folder,output_name+'.tif')
    print(f"\tSaving final output:{output_path}")
    imwrite(output_path, combine_seed, 
        compression ='zlib')
    
    combine_seed,_ = reorder_segmentation(combine_seed, min_size=min_size, sort_ids=sort)
    output_path = os.path.join(output_folder,output_name+'_sorted.tif')
    print(f"\tSaving final output:{output_path}")
    imwrite(output_path, combine_seed, 
        compression ='zlib')
    
             
    return combine_seed,ori_combine_ids_map, output_dict  


if __name__ == "__main__":        
   
    
    # img_path = './result/dog_lucy/input/Beagle_220783_8bits.tif'
    # output_folder = './result/dog_lucy/seeds'
    # threshold = 121
    
    img_path = './result/foram_james/input/ai/final.20180719_VERSA_1905_ASB_OLK_st014_bl4_fo1_recon.tif'  
    output_folder = './result/foram_james/seeds_test_gen_time_mp' 
    threshold = 266
    
    # Track the overall start time
    overall_start_time = time.time()
    
    
    # seed ,ori_combine_ids_map , output_dict=  separate_chambers(img_path, 
    #                                                     threshold = threshold,
    #                                                     output_folder = output_folder,
    #                                                     n_iters=3,segments=20,
    #                                                     save_every_iter = True)
    img = imread(img_path)
    name_prefix = os.path.basename(img_path)
    seed ,ori_combine_ids_map , output_dict=  make_seeds_merged_mp(img, 
                                                threshold = threshold,
                                                output_folder = output_folder,
                                                n_iters=3,segments=20,
                                                num_threads = 10,
                                                min_size= 100,
                                                save_every_iter = True,
                                                name_prefix=name_prefix)
    # pd.DataFrame(ori_combine_ids_map).to_csv(os.path.join(output_folder, 'ori_combine_ids_map.csv'),index=False)
    pd.DataFrame(output_dict).to_csv(os.path.join(output_folder, 'output_dict.csv'),index=False)
    
    
    

    

    # Track the overall end time
    overall_end_time = time.time()

    # Output the total running time
    print(f"Total running time: {overall_end_time - overall_start_time:.2f} seconds")
        
        
        
        
 # file_path = '/result/foram_james/input/ai/final.20180802_VERSA_1905_ASB_OLK_st016_bl4_fo1_recon.tif'    

    # file_paths = glob.glob('result/foram_james/input/ai/*.tif')
    # output_folder = 'result/foram_james/thre_ero_seed/'

    # config_path = './make_seeds_foram.yaml'
    # _, extension = os.path.splitext(config_path)
    # print(f"processing config he file {config_path}")
    # if extension == '.yaml':
    #     with open(config_path, 'r') as file:
    #         config = yaml.safe_load(file)
    #     load_config_yaml(config)

    # num_threads = 5
    # file_paths = glob.glob(os.path.join(input_folder,"*.tif"))
    
    
    # threads = []
    # sub_file_paths = [file_paths[i::num_threads] for i in range(num_threads)]
    # # Start a new thread for each sublist
    # for sublist in sub_file_paths:
    #     thread = threading.Thread(target=foram_seed_mp, args=(sublist,output_folder))
    #     threads.append(thread)
    #     thread.start()
        
    # # Wait for all threads to complete
    # for thread in threads:
    #     thread.join()
    
    # sys.exit()
    #### For single file instance
    
    # img_path = './result/foram_james/input/ai/final.20180719_VERSA_1905_ASB_OLK_st014_bl4_fo1_recon.tif'
    # output_folder = './result/foram_james/seed_ai_v2/'

    # imwrite(os.path.join('./result/foram_james/seed_ai_v2/',base_name+'.tif'), seed, 
    #     compression ='zlib')
    # bone_ids_dict = {int(key): str(value) for key, value in bone_ids_dict.items()}
    
# separate_chambers(file_path)

