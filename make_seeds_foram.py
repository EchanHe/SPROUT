import numpy as np
# from PIL import Image
from tifffile import imread, imwrite
import traceback
import os,sys
import BounTI
import glob

# Add the lib directory to the system path
lib_path = os.path.abspath('../util')
sys.path.insert(0, lib_path)
import suture_morpho
import threading

def separate_chambers(file_path):
    img = imread(file_path)
    img = img==255

    seed, ccomp_sizes = BounTI.get_ccomps_with_size_order(img,25)

    bone_ids = np.unique(seed)
    bone_ids = [value for value in bone_ids if value != 0]

    ero_img = img.copy()
    n_iters = 30
    no_consec_split_count = 0

    for n_iter in range(n_iters):
        print(f"working on erosion {n_iter}")
        ero_img = suture_morpho.erosion_binary_img_on_sub(ero_img, kernal_size = 1)
        ero_seed, ccomp_sizes = BounTI.get_ccomps_with_size_order(ero_img,25)

        ero_bone_ids = np.unique(ero_seed)
        ero_bone_ids = [value for value in ero_bone_ids if value != 0]

        bone_ids = np.unique(seed)
        bone_ids = [value for value in bone_ids if value != 0]

        
        has_split = False
        for bone_id in bone_ids:
            comp = seed == bone_id

            inter_count = 0
            inter_ids = []
            inter_props = []
            
            
            for ero_bone_id in ero_bone_ids:
                ero_comp = ero_seed == ero_bone_id
                
                inter = np.sum(np.logical_and(comp,ero_comp))
                prop = round(np.sum(np.logical_and(comp,ero_comp)) / np.sum(comp),4)
                # print(f"{bone_id} for ero {ero_bone_id} has intersect {inter}\nprop{prop}")
                
                
                if inter>0:
                    inter_count+=1
                    inter_ids.append(ero_bone_id)
                    inter_props.append(round(np.sum(np.logical_and(comp,ero_comp)) / np.sum(comp),4))
                    # prop = round(np.sum(np.logical_and(comp,ero_comp)) / np.sum(comp),4)
                    
            if inter_count>1:
                print(f'{bone_id} has been split to {inter_count} parts. Ids are {inter_ids}')
                print(f"props are: {inter_props}")
                seed[seed == bone_id] =0
                for inter_id in inter_ids:
                    seed[ero_seed == inter_id] = np.max(np.unique(seed))+1
                has_split = True
                
            # elif inter_count==0:
            #     print(f'{bone_id} has been completely erode')

        if has_split:
            no_consec_split_count=0
        else:
            no_consec_split_count+=1
            
        
        if no_consec_split_count>=3:
            print(f"detect non split for {no_consec_split_count}rounds")
            print(f"break loop at {n_iter} iter")
            break
                
    return seed

def gen_mesh(volume, threshold, output_path):

    # output_path = os.path.join(output_dir, f"{threshold}.ply")
    if os.path.isfile(output_path):
        return
    else:
        output = BounTI.binary_stack_to_mesh(volume, threshold)
        output.export(output_path)
def foram_seed_mp(file_paths, output_folder):
    for file_path in file_paths:
        print(file_path)
        seed = separate_chambers(file_path)
        # volume = imread(file_path)
        base_name = os.path.basename(file_path)
        # gen_mesh(volume, 0, os.path.join('result/foram_james/whole_mesh/',
        #                                  base_name+'.ply'))
        
        imwrite(os.path.join(output_folder,base_name+'.tif'), seed, 
                compression ='zlib')

if __name__ == "__main__":        
    # file_path = '/result/foram_james/input/ai/final.20180802_VERSA_1905_ASB_OLK_st016_bl4_fo1_recon.tif'    

    file_paths = glob.glob('result/foram_james/input/ai/*.tif')
    output_folder = 'result/foram_james/thre_ero_seed/'
    num_threads = 5
    
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
    

# separate_chambers(file_path)

