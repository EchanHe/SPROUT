import numpy as np
import pandas as pd
# from PIL import Image
from tifffile import imread, imwrite
import traceback
import os,sys

import glob
import threading
lock = threading.Lock()
import time
import json, yaml

# Add the lib directory to the system path
import suture_morph.suture_morpho as suture_morpho
from help_functions.sort_and_filer_seg_by_size import reorder_segmentation


def load_config_yaml(config, parent_key=''):
    for key, value in config.items():
        if isinstance(value, dict):
            load_config_yaml(value, parent_key='')
        else:
            globals()[parent_key + key] = value

def separate_chambers(img_path,
                      threshold,
                      output_folder,
                      n_iters, 
                      segments,
                      no_split_limit =3,
                      min_size=5,
                      sort = True,
                      prop_thre = 0.01,
                      background = 0,
                      save_every_iter = False
                      ):

    base_name = os.path.basename(img_path)
    output_folder = os.path.join(output_folder, base_name)
    os.makedirs(output_folder,exist_ok=True)
    
    max_splits = segments
    
    img = imread(img_path)
    img = img>=threshold

    init_seed, _ = suture_morpho.get_ccomps_with_size_order(img,segments)
    if save_every_iter:
        imwrite(os.path.join(output_folder,f'thre_{threshold}_ero_0.tif'), init_seed, 
            compression ='zlib')
    
    init_ids = [int(value) for value in np.unique(init_seed) if value != background]
    max_seed_id = int(np.max(init_ids))


    combine_seed = init_seed.copy()
    combine_seed = combine_seed.astype('uint8')

    output_dict = {"total_id": {0: len(np.unique(combine_seed))-1},
                   "split_id" : {0: {}},
                   "split_ori_id": {0: {}},
                   "split_ori_id_filtered":  {0: {}},
                   "split_prop":  {0: {}}
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
        
        img = suture_morpho.erosion_binary_img_on_sub(img, kernal_size = 1)
        seed, _ = suture_morpho.get_ccomps_with_size_order(img,segments)
        if save_every_iter:
            output_path = os.path.join(output_folder,f'thre_{threshold}_ero_{ero_iter}.tif')
            print(f"\tSaving {output_path}")
            imwrite(output_path, seed, 
                compression ='zlib')

        seed_ids = [int(value) for value in np.unique(seed) if value != 0]
        combine_ids = [int(value) for value in np.unique(combine_seed) if value != background]
       
        has_split = False
        ## Comparing each ccomp from eroded seed
        ## to each ccomp from the original seed
               
        for combine_id in combine_ids:
            comp = combine_seed == combine_id
            inter_count = 0
            inter_ids = np.array([])
            inter_props = np.array([])
            
            # Iter through each id in the current seed
            for seed_id in seed_ids:
                seed_comp = seed == seed_id
                
                # Check if there are intersection using the sum of intersection
                inter = np.sum(comp[seed_comp])
            
                if inter>0:
                    inter_count+=1
                    inter_ids = np.append(inter_ids , seed_id)
                    
                    prop = round(inter / np.sum(comp),4)*100
                    inter_props = np.append(inter_props, prop)
             ## if there are any intersection between seed and the init seed
            if inter_count>1:
                
                
                print(f'\t{combine_id} has been split to {inter_count} parts. Ids are {inter_ids}')
                print(f"\tprops are: {inter_props}")
                
                combine_seed[combine_seed == combine_id] =0
                filtered_inter_ids = inter_ids[inter_props>prop_thre]
                
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
                output_dict["split_ori_id"][ero_iter][combine_id] = inter_ids
                output_dict["split_ori_id_filtered"][ero_iter][combine_id] = filtered_inter_ids
                output_dict["split_prop"][ero_iter][combine_id] = inter_props
                
        
            output_dict["total_id"][ero_iter] = len(np.unique(combine_seed))-1
    
        if has_split:
            no_consec_split_count=0
        else:
            no_consec_split_count+=1
            
        
        if no_consec_split_count>=no_split_limit:
            print(f"detect non split for {no_consec_split_count}rounds")
            print(f"break loop at {ero_iter} iter")
            break
        

    
    output_path = os.path.join(output_folder,base_name+'.tif')
    print(f"\tSaving final output:{output_path}")
    imwrite(output_path, combine_seed, 
        compression ='zlib')
    
    combine_seed,_ = reorder_segmentation(combine_seed, min_size=min_size, sort_ids=sort)
    output_path = os.path.join(output_folder,base_name+'_sorted.tif')
    print(f"\tSaving final output:{output_path}")
    imwrite(output_path, combine_seed, 
        compression ='zlib')
    
             
    return combine_seed,ori_combine_ids_map, output_dict    
    #     for bone_id in bone_ids:
    #         comp = seed == bone_id

    #         inter_count = 0
    #         inter_ids = np.array([])
    #         inter_props = np.array([])
            
            
    #         for ero_bone_id in ero_bone_ids:
    #             ero_comp = ero_seed == ero_bone_id
                
    #             # start_time = time.time()
    #             # inter = np.sum(np.logical_and(comp,ero_comp))
    #             # end_time = time.time()
    #             # elapsed_time = end_time - start_time
    #             # print(f"Elapsed time: {elapsed_time:.6f} seconds:value:{inter}")
    #             # start_time = time.time()
    #             inter = np.sum(comp[ero_comp])
    #             # end_time = time.time()
    #             # elapsed_time = end_time - start_time
    #             # print(f"Elapsed time: {elapsed_time:.6f} seconds:value:{inter}")
    #             # prop = round(np.sum(np.logical_and(comp,ero_comp)) / np.sum(comp),4)
    #             # print(f"{bone_id} for ero {ero_bone_id} has intersect {inter}\nprop{prop}")
                
                
    #             if inter>0:
    #                 inter_count+=1
    #                 inter_ids = np.append(inter_ids , ero_bone_id)
    #                 prop = round(inter / np.sum(comp),4)*100
    #                 inter_props = np.append(inter_props, prop)

    #                 # prop = round(np.sum(np.logical_and(comp,ero_comp)) / np.sum(comp),4)
    #         ## When a ccomps has been split into multiple ccomps in the next ero step.   
    #         if inter_count>1:
    #             # prop_thre = 0.1
               
    #             print(f'{bone_id} has been split to {inter_count} parts. Ids are {inter_ids}')
    #             print(f"props are: {inter_props}")
    #             print(f"Remove parts that have proportion smaller than prop_thre")
    #             seed[seed == bone_id] =0
    #             inter_ids = inter_ids[inter_props>prop_thre]
                
    #             new_ids = []
    #             for inter_id in inter_ids:
    #                 seed[ero_seed == inter_id] = max_seed_id+1
                    
    #                 # split_list = bone_ids_dict[bone_id]
    #                 # if len(split_list) <= max_splits:
    #                 #     bone_ids_dict[bone_id].append(max_seed_id+1)
    #                 for key,value in bone_ids_dict.items():
    #                     if bone_id in value:
    #                         if len(value) <= max_splits:
    #                             bone_ids_dict[key].append(max_seed_id+1)
    #                             new_ids.append(max_seed_id+1)
    #                             break
                                
    #                 max_seed_id +=1
                
    #             split_log[n_iter][bone_id] = new_ids
                    
    #             has_split = True
                
    #         # elif inter_count==0:
    #         #     print(f'{bone_id} has been completely erode')

    #     if has_split:
    #         no_consec_split_count=0
    #     else:
    #         no_consec_split_count+=1
            
        
    #     if no_consec_split_count>=no_split_limit:
    #         print(f"detect non split for {no_consec_split_count}rounds")
    #         print(f"break loop at {n_iter} iter")
    #         break
    

    # output_log = {"ori_comps_to_final":bone_ids_dict,
    #               "split_each_iter":split_log}
    # with open(os.path.join(output_folder,base_name+'.json'), 'w') as file:
    #     json.dump(output_log, file, indent=4)
    # imwrite(os.path.join(output_folder,base_name+'.tif'), seed, 
    #     compression ='zlib')
             
    # return seed,bone_ids_dict, split_log


def detect_inter(comp,ero_seed, seed_ids, inter_log , lock):
    for seed_id in seed_ids:
        seed_comp = ero_seed == seed_id
                    
        # Check if there are intersection using the sum of intersection
        inter = np.sum(comp[seed_comp])

        with lock:
            if inter>0:
                inter_log["inter_count"] +=1
                
                inter_log["inter_ids"] = np.append(inter_log["inter_ids"] , seed_id)
                
                prop = round(inter / np.sum(comp),4)*100
                inter_log["inter_props"] = np.append(inter_log["inter_props"], prop)

def separate_chambers_mp(img_path,
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
                      save_every_iter = False
                      ):

    base_name = os.path.basename(img_path)
    output_folder = os.path.join(output_folder, base_name)
    os.makedirs(output_folder,exist_ok=True)
    
    max_splits = segments
    
    img = imread(img_path)
    img = img>=threshold

    init_seed, _ = suture_morpho.get_ccomps_with_size_order(img,segments)
    if save_every_iter:
        imwrite(os.path.join(output_folder,f'thre_{threshold}_ero_0.tif'), init_seed, 
            compression ='zlib')
    
    init_ids = [int(value) for value in np.unique(init_seed) if value != background]
    max_seed_id = int(np.max(init_ids))


    combine_seed = init_seed.copy()
    combine_seed = combine_seed.astype('uint8')

    output_dict = {"total_id": {0: len(np.unique(combine_seed))-1},
                   "split_id" : {0: {}},
                   "split_ori_id": {0: {}},
                   "split_ori_id_filtered":  {0: {}},
                   "split_prop":  {0: {}}
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
        
        img = suture_morpho.erosion_binary_img_on_sub(img, kernal_size = 1)
        seed, _ = suture_morpho.get_ccomps_with_size_order(img,segments)
        if save_every_iter:
            output_path = os.path.join(output_folder,f'thre_{threshold}_ero_{ero_iter}.tif')
            print(f"\tSaving {output_path}")
            imwrite(output_path, seed, 
                compression ='zlib')

        seed_ids = [int(value) for value in np.unique(seed) if value != 0]
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
            comp = combine_seed == combine_id
            
            # === to delete
            # inter_count = 0
            # inter_ids = np.array([])
            # inter_props = np.array([])
            
            inter_log["inter_count"] = 0
            inter_log["inter_ids"] = np.array([])
            inter_log["inter_props"] = np.array([])
            
            sublists = [seed_ids[i::num_threads] for i in range(num_threads)]
             # Create a list to hold the threads
            threads = []
            for sublist in sublists:
                thread = threading.Thread(target=detect_inter, args=(comp,
                                                                     seed,
                                                                     sublist,
                                                                     inter_log,
                                                                     lock))
                threads.append(thread)
                thread.start()
                
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # === to delete   
            # Iter through each id in the current seed
            # for seed_id in seed_ids:
            #     seed_comp = seed == seed_id
                
            #     # Check if there are intersection using the sum of intersection
            #     inter = np.sum(comp[seed_comp])
            
            #     if inter>0:
            #         inter_count+=1
            #         inter_ids = np.append(inter_ids , seed_id)
                    
            #         prop = round(inter / np.sum(comp),4)*100
            #         inter_props = np.append(inter_props, prop)
            # === to delete    
        
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
        

    
    output_path = os.path.join(output_folder,base_name+'.tif')
    print(f"\tSaving final output:{output_path}")
    imwrite(output_path, combine_seed, 
        compression ='zlib')
    
    combine_seed,_ = reorder_segmentation(combine_seed, min_size=min_size, sort_ids=sort)
    output_path = os.path.join(output_folder,base_name+'_sorted.tif')
    print(f"\tSaving final output:{output_path}")
    imwrite(output_path, combine_seed, 
        compression ='zlib')
    
             
    return combine_seed,ori_combine_ids_map, output_dict    





def foram_seed_mp(file_paths, output_folder):
    for img_path in file_paths:
        print(img_path)
        
        # volume = imread(file_path)
        base_name = os.path.basename(img_path)
        
        
        
        output_dir = os.path.join(output_folder,base_name)
        os.makedirs(output_dir, exist_ok=True)
        
        
        separate_chambers(img_path, 
                        output_folder = output_dir,
                        base_name = base_name,
                        n_iters=n_iters,segments=segments)

if __name__ == "__main__":        
   
    
    # img_path = './result/dog_lucy/input/Beagle_220783_8bits.tif'
    # output_folder = './result/dog_lucy/seeds'
    # threshold = 121
    
    img_path = './result/foram_james/input/ai/final.20180719_VERSA_1905_ASB_OLK_st014_bl4_fo1_recon.tif'  
    output_folder = './result/foram_james/seeds_test_gen_time_mp' 
    threshold = 255
    
    # Track the overall start time
    overall_start_time = time.time()
    
    
    # seed ,ori_combine_ids_map , output_dict=  separate_chambers(img_path, 
    #                                                     threshold = threshold,
    #                                                     output_folder = output_folder,
    #                                                     n_iters=3,segments=20,
    #                                                     save_every_iter = True)
    
    seed ,ori_combine_ids_map , output_dict=  separate_chambers_mp(img_path, 
                                                threshold = threshold,
                                                output_folder = output_folder,
                                                n_iters=3,segments=20,
                                                num_threads = 10,
                                                save_every_iter = True)
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

