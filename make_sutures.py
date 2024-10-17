# import util.img_process as img_process

import os,sys

import numpy as np

from skimage.morphology import binary_dilation,binary_closing, cube,square

import suture_morph.suture_morpho as suture_morpho
import suture_morph.img_process as img_process
# importlib.reload(suture_morpho)
import threading

import tifffile
import itertools
import time
import logging
import pandas as pd
import csv
import math
import make_mesh


def is_island_id(value):
    if np.isnan(value) or (value is None) or (math.isnan(value)):
        return False
    elif value <=0:
        return False
    else:
        return True

def gen_suture_dict(df):
    row_dict = df.iloc[0].to_dict()
    
    result = {}
    
    frontal_L = row_dict.get('frontal_L', -1)
    frontal_R = row_dict.get('frontal_R', -1)
    parietal_L = row_dict.get('parietal_L', -1)
    parietal_R = row_dict.get('parietal_R', -1)
    occipital = row_dict.get('occipital', -1)
    jugal_L = row_dict.get('jugal_L', -1)
    jugal_R = row_dict.get('jugal_R', -1)
    squamosal_L = row_dict.get('squamosal_L', -1)
    squamosal_R = row_dict.get('squamosal_R', -1)
    maxillary_L = row_dict.get('maxillary_L', -1)
    maxillary_R = row_dict.get('maxillary_R', -1)
    premaxillary_L = row_dict.get('premaxillary_L', -1)
    premaxillary_R = row_dict.get('premaxillary_R', -1)
    nasal_L = row_dict.get('nasal_L', -1)
    nasal_R = row_dict.get('nasal_R', -1)
    interparietal = row_dict.get('interparietal', -1)
    palatine_L = row_dict.get('palatine_L', -1)
    palatine_R = row_dict.get('palatine_R', -1)


    # Directly assign all possible key-value pairs to the result dictionary
    result[f'interfrontal_{frontal_L}_{frontal_R}'] = (frontal_L, frontal_R)
    
    
    result[f'sagittal_{parietal_L}_{parietal_R}'] = (parietal_L, parietal_R)
    result[f'coronal_L_{parietal_L}_{frontal_L}'] = (parietal_L, frontal_L)
    result[f'coronal_R_{parietal_R}_{frontal_R}'] = (parietal_R, frontal_R)
    
    
    result[f'occipitoparietal_L_{parietal_L}_{occipital}'] = (parietal_L, occipital)
    result[f'occipitoparietal_R_{parietal_R}_{occipital}'] = (parietal_R, occipital)
    
    
    result[f'frontojugal_L_{frontal_L}_{jugal_L}'] = (frontal_L, jugal_L)
    result[f'frontojugal_R_{frontal_R}_{jugal_R}'] = (frontal_R, jugal_R)
    
    
    result[f'sphenofrontal_L_{frontal_L}_{squamosal_L}'] = (frontal_L, squamosal_L)
    result[f'sphenofrontal_R_{frontal_R}_{squamosal_R}'] = (frontal_R, squamosal_R)
    
    result[f'squamosalparietal_L_{parietal_L}_{squamosal_L}'] = (parietal_L, squamosal_L)
    result[f'squamosalparietal_R_{parietal_R}_{squamosal_R}'] = (parietal_R, squamosal_R)
    
    result[f'squamosooccipital_L_{occipital}_{squamosal_L}'] = (occipital, squamosal_L)
    result[f'squamosooccipital_R_{occipital}_{squamosal_R}'] = (occipital, squamosal_R)
    
    result[f'temporozygomatic_L_{jugal_L}_{squamosal_L}'] = (jugal_L, squamosal_L)
    result[f'temporozygomatic_R_{jugal_R}_{squamosal_R}'] = (jugal_R, squamosal_R)
    
    
    result[f'frontomaxillary_L_{frontal_L}_{maxillary_L}'] = (frontal_L, maxillary_L)
    result[f'frontomaxillary_R_{frontal_R}_{maxillary_R}'] = (frontal_R, maxillary_R)
    
    result[f'maxillaryjugal_L_{jugal_L}_{maxillary_L}'] = (jugal_L, maxillary_L)
    result[f'maxillaryjugal_R_{jugal_R}_{maxillary_R}'] = (jugal_R, maxillary_R)
    
    result[f'intermaxilllary_{maxillary_L}_{maxillary_R}'] = (maxillary_L, maxillary_R)
    
    
    result[f'frontopremaxillary_L_{frontal_L}_{premaxillary_L}'] = (frontal_L, premaxillary_L)
    result[f'frontopremaxillary_R_{frontal_R}_{premaxillary_R}'] = (frontal_R, premaxillary_R)
    result[f'maxillarypremaxillary_L_{maxillary_L}_{premaxillary_L}'] = (maxillary_L, premaxillary_L)
    result[f'maxillarypremaxillary_R_{maxillary_R}_{premaxillary_R}'] = (maxillary_R, premaxillary_R)
    
    
    result[f'frontonasal_L_{frontal_L}_{nasal_L}'] = (frontal_L, nasal_L)
    result[f'frontonasal_R_{frontal_R}_{nasal_R}'] = (frontal_R, nasal_R)
    
    result[f'nasomaxillary_L_{maxillary_L}_{nasal_L}'] = (maxillary_L, nasal_L)
    result[f'nasomaxillary_R_{maxillary_R}_{nasal_R}'] = (maxillary_R, nasal_R)
    
    result[f'nasopremaxillary_L_{premaxillary_L}_{nasal_L}'] = (premaxillary_L, nasal_L)
    result[f'nasopremaxillary_R_{premaxillary_R}_{nasal_R}'] = (premaxillary_R, nasal_R)
    
    
    result[f'lambdoid_L_{parietal_L}_{interparietal}'] = (parietal_L, interparietal)
    result[f'lambdoid_R_{parietal_R}_{interparietal}'] = (parietal_R, interparietal)
    result[f'occipital_{occipital}_{interparietal}'] = (occipital, interparietal)
    
    
    result[f'maxillarypalatine_L_{maxillary_L}_{palatine_L}'] = (maxillary_L, palatine_L)
    result[f'maxillarypalatine_R_{maxillary_R}_{palatine_R}'] = (maxillary_R, palatine_R)
    result[f'interpalatine_{palatine_R}_{palatine_L}'] = (palatine_R, palatine_L)

    # if frontal_L!=-1 and frontal_R!=-1:
    #     result[f'interfrontal_{frontal_L}_{frontal_R}'] = (frontal_L, frontal_R)
    # if parietal_L!=-1 and parietal_R!=-1:
    #     result[f'sagittal_{parietal_L}_{parietal_R}'] = (parietal_L, parietal_R)
        
    # if parietal_L!=-1 and frontal_L!=-1:
    #     result[f'coronal_L_{parietal_L}_{frontal_L}'] = (parietal_L, frontal_L)
    # if parietal_R!=-1 and frontal_R!=-1:
    #     result[f'coronal_R_{parietal_R}_{frontal_R}'] = (parietal_R, frontal_R)
    
    # if parietal_L!=-1 and occipital!=-1:
    #     result[f'occipitoparietal_L_{parietal_L}_{occipital}'] = (parietal_L, occipital)    
    # if parietal_R!=-1 and occipital!=-1:
    #     result[f'occipitoparietal_R_{parietal_R}_{occipital}'] = (parietal_R, occipital)    
        
    # if frontal_L!=-1 and jugal_L!=-1:
    #     result[f'frontojugal_L_{frontal_L}_{jugal_L}'] = (frontal_L, jugal_L)
    # if frontal_R!=-1 and jugal_R!=-1:
    #     result[f'frontojugal_R_{frontal_R}_{jugal_R}'] = (frontal_R, jugal_R)    

    # if frontal_L!=-1 and squamosal_L!=-1:
    #     result[f'sphenofrontal_L_{frontal_L}_{squamosal_L}'] = (frontal_L, squamosal_L)
    # if frontal_R!=-1 and squamosal_R!=-1:
    #     result[f'sphenofrontal_R_{frontal_R}_{squamosal_R}'] = (frontal_R, squamosal_R)    

    # if parietal_L!=-1 and squamosal_L!=-1:
    #     result[f'squamosalparietal_L_{parietal_L}_{squamosal_L}'] = (parietal_L, squamosal_L)
    # if parietal_R!=-1 and squamosal_R!=-1:
    #     result[f'squamosalparietal_R_{parietal_R}_{squamosal_R}'] = (parietal_R, squamosal_R)   

    # if occipital!=-1 and squamosal_L!=-1:
    #     result[f'squamosooccipital_L_{occipital}_{squamosal_L}'] = (occipital, squamosal_L)
    # if occipital!=-1 and squamosal_R!=-1:
    #     result[f'squamosooccipital_R_{occipital}_{squamosal_R}'] = (occipital, squamosal_R)    

    # if jugal_L!=-1 and squamosal_L!=-1:
    #     result[f'temporozygomatic_L_{jugal_L}_{squamosal_L}'] = (jugal_L, squamosal_L)
    # if jugal_R!=-1 and squamosal_R!=-1:
    #     result[f'temporozygomatic_R_{jugal_R}_{squamosal_R}'] = (jugal_R, squamosal_R)    
        
    # if frontal_L!=-1 and maxillary_L!=-1:
    #     result[f'frontomaxillary_L_{frontal_L}_{maxillary_L}'] = (frontal_L, maxillary_L)
    # if frontal_R!=-1 and maxillary_R!=-1:
    #     result[f'frontomaxillary_R_{frontal_R}_{maxillary_R}'] = (frontal_R, maxillary_R)    
        
    # if jugal_L!=-1 and maxillary_L!=-1:
    #     result[f'maxillaryjugal_L_{jugal_L}_{maxillary_L}'] = (jugal_L, maxillary_L)
    # if jugal_R!=-1 and maxillary_R!=-1:
    #     result[f'maxillaryjugal_R_{jugal_R}_{maxillary_R}'] = (jugal_R, maxillary_R)    
        
    # if maxillary_L!=-1 and maxillary_R!=-1:
    #     result[f'intermaxilllary_{maxillary_L}_{maxillary_R}'] = (maxillary_L, maxillary_R)     
       
    
    
    # if frontal_L!=-1 and premaxillary_L!=-1:
    #     result[f'frontopremaxillary_L_{frontal_L}_{premaxillary_L}'] = (frontal_L, premaxillary_L)
    # if frontal_R!=-1 and premaxillary_R!=-1:
    #     result[f'frontopremaxillary_R_{frontal_R}_{premaxillary_R}'] = (frontal_R, premaxillary_R)      
      
    # if maxillary_L!=-1 and premaxillary_L!=-1:
    #     result[f'maxillarypremaxillary_L_{maxillary_L}_{premaxillary_L}'] = (maxillary_L, premaxillary_L)
    # if maxillary_R!=-1 and premaxillary_R!=-1:
    #     result[f'maxillarypremaxillary_R_{maxillary_R}_{premaxillary_R}'] = (maxillary_R, premaxillary_R)          

          
        
    # if frontal_L!=-1 and nasal_L!=-1:
    #     result[f'frontonasal_L_{frontal_L}_{nasal_L}'] = (frontal_L, nasal_L)
    # if frontal_R!=-1 and nasal_R!=-1:
    #     result[f'frontonasal_R_{frontal_R}_{nasal_R}'] = (frontal_R, nasal_R)   
        
    # if maxillary_L!=-1 and nasal_L!=-1:
    #     result[f'nasomaxillary_L_{maxillary_L}_{nasal_L}'] = (maxillary_L, nasal_L)
    # if maxillary_R!=-1 and nasal_R!=-1:
    #     result[f'nasomaxillary_R_{maxillary_R}_{nasal_R}'] = (maxillary_R, nasal_R) 
    
    # if premaxillary_L!=-1 and nasal_L!=-1:
    #     result[f'nasopremaxillary_L_{premaxillary_L}_{nasal_L}'] = (premaxillary_L, nasal_L)
    # if premaxillary_R!=-1 and nasal_R!=-1:
    #     result[f'nasopremaxillary_R_{premaxillary_R}_{nasal_R}'] = (premaxillary_R, nasal_R) 
        
    # if parietal_L!=-1 and interparietal!=-1:
    #     result[f'lambdoid_L_{parietal_L}_{interparietal}'] = (parietal_L, interparietal)
    # if parietal_R!=-1 and interparietal!=-1:
    #     result[f'lambdoid_R_{parietal_R}_{interparietal}'] = (parietal_R, interparietal)   
        
    # if occipital!=-1 and interparietal!=-1:
    #     result[f'occipital_{occipital}_{interparietal}'] = (occipital, interparietal)

    # if maxillary_L!=-1 and palatine_L!=-1:
    #     result[f'maxillarypalatine_L_{maxillary_L}_{palatine_L}'] = (maxillary_L, palatine_L)
    # if maxillary_R!=-1 and palatine_R!=-1:
    #     result[f'maxillarypalatine_R_{maxillary_R}_{palatine_R}'] = (maxillary_R, palatine_R) 
        
    # if palatine_R!=-1 and palatine_L!=-1:
    #     result[f'interpalatine_{palatine_R}_{palatine_L}'] = (palatine_R, palatine_L)
    


    return result


def gen_suture_dict_v2(part_id_dict, df_mapping):
    # Convert the row to a dictionary
    # part_id_dict = df.iloc[0].to_dict()

    result = {}
    for index, row in df_mapping.iterrows():
        result[row['result']] = (part_id_dict[row['part_1']],
                                part_id_dict[row['part_2']])
    return result

def find_sutures_mp(img, key_value_list, output_folder):
    """
    
    combinations: tuple of two id ()
    """
    id_list = np.unique(img)
    id_list =  np.array([x for x in id_list if x !=0])
    mask = np.isin(img, id_list)
    background = np.where(mask, True, np.zeros((img.shape)))
    
    
    for key, value in key_value_list:
        print(f"Processing {key}")

        # output_suture = np.zeros_like(img)
        bone_1_id =  df_output.loc[key,'part_1_value'] = value[0]
        bone_2_id =  df_output.loc[key,'part_2_value'] = value[1]
        
        # print(f"Finding gaps for {bone_1_id} and {bone_2_id}")
        if bone_2_id == bone_1_id:
            print(f"No gaps between {bone_1_id} and {bone_2_id}")
            continue 
        
        bone_1 = img==bone_1_id
        bone_2 = img==bone_2_id
        
        result = suture_morpho.find_gaps_between_two(bone_1,bone_2,background)
            
        if result is None:
            print(f"No gaps between {bone_1_id} and {bone_2_id}")
            
        else:
            result = result.astype('uint8')
            if SAVE_ISLANDS:
                result[bone_1 & (result!=1)] = 2
                result[bone_2 & (result!=1)] = 3
            
            # output_suture[result==True]=250
            output_path = os.path.join(output_folder, f'{key}.tif')
            
            tifffile.imwrite(output_path, result, compression ='zlib')
            print(f"Result {key} has been saved to {output_path}")
            df_output.loc[key,'output_path'] = os.path.abspath(output_path)

def find_sutures(img, key_value_list, output_folder):
    """
    
    combinations: tuple of two id ()
    """
    id_list = np.unique(img)
    id_list =  np.array([x for x in id_list if x !=0])
    mask = np.isin(img, id_list)
    background = np.where(mask, True, np.zeros((img.shape)))
    
   
   
    csv_data = [
        ['suture_name', 'id', 'file_name', 'has_suture']
    ]
    # Add data to CSV
    id_counter = 1
    for key, value in key_value_list:
        print(f"Processing {key}")
        # output_suture = np.zeros_like(img)
        bone_1_id = value[0]
        bone_2_id = value[1]
        if is_island_id(bone_1_id) and is_island_id(bone_2_id):
        # print(f"Finding gaps for {bone_1_id} and {bone_2_id}")
        
            if bone_1_id == bone_2_id:
                result =None
            else:
                bone_1 = img==bone_1_id
                bone_2 = img==bone_2_id
                
                result = suture_morpho.find_gaps_between_two(bone_1,bone_2,background)
                
            if result is None:
                print(f"No gaps between {bone_1_id} and {bone_2_id}")
                has_suture = False
                output_path = None
            else:
                has_suture = True
                result = result.astype('uint8')
                if SAVE_ISLANDS:
                    result[bone_1 & (result!=1)] = 2
                    result[bone_2 & (result!=1)] = 3
                
                # output_suture[result==True]=250
                output_path = os.path.join(output_folder, f'{key}.tif')
                
                tifffile.imwrite(output_path, result, compression ='zlib')
        else:
            has_suture = False
            output_path = None
    
        csv_data.append([
            key,  # Extract the suture name prefix (interfrontal, sagittal)
            id_counter,
            output_path,
            has_suture
        ])
        id_counter += 1

    # Write to CSV file
    csv_path = os.path.join(output_folder, 'suture_log.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)

def merge_sutures(df):
    output = None
    for _, row in df.iterrows():
        path = row['output_path']
        suture_id = row['result_id']
        print(f"Merging {path}")

        if isinstance(path, str):
            seg = tifffile.imread(path)

            if output is None:
                output =  np.zeros_like(seg)
            
            output = np.where(seg == 1, suture_id, output)

    return output


if __name__ == "__main__":

    df_id = pd.read_csv("./template/scan_bone_ids_multi.csv")
    df_mapping = pd.read_csv("./template/suture_bone_mapping.csv")
    
    # for idx, file_path in enumerate(df['seg_file']):
    suture_morpho.check_tiff_files(df['seg_file'])
    # sys.exit(

    SAVE_ISLANDS = False
    n_threads = 20
    
    # suture_dict = gen_suture_dict(df_id)
    
    workspace = r'result/procavia/seg/'
    output_folder = "sutures_v2"
    output_folder = os.path.join(workspace, output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    
    df_output_list = []
    try:
        for index, row in df_id.iterrows():
            specimen = row['specimen']
            print(f"working on {specimen}")

            cur_output_folder = os.path.join(output_folder, specimen)
            os.makedirs(cur_output_folder, exist_ok=True)
            
            df_output = (df_mapping.copy())
            df_output.set_index("result", inplace=True)
            df_output.loc[:,"seg_file"] =os.path.abspath(row['seg_file'])
            df_output.loc[:,"specimen"] = specimen
            
            for idx_map, row_map in df_mapping.iterrows():
    
                df_output.loc[row_map['result'], 'part_1_value'] = int(row[row_map['part_1']])
                df_output.loc[row_map['result'], 'part_2_value'] = int(row[row_map['part_2']])
            
            # suture_dict = gen_suture_dict_v2(row.to_dict(), df_mapping)
            
            
            main_list = list(zip(df_output['part_1_value'], df_output['part_2_value']))
            
            # main_list = list(suture_dict.items())
            
            img = img_process.imread(row['seg_file'])
            print(f"Detecting {len(main_list)} sutures")

            sublists = [main_list[i::n_threads] for i in range(n_threads)]

            # Create a list to hold the threads
            threads = []

            # Start a new thread for each sublist
            for sublist in sublists:
                thread = threading.Thread(target=find_sutures_mp, args=(img, sublist, cur_output_folder,))
                threads.append(thread)
                thread.start()
                
            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            print("All threads have completed.")
            
            output_csv_path = os.path.join(cur_output_folder,f"output_{specimen}.csv")
            # df_output['merge_suture_id'] = range(1, len(df_output) + 1)

            merged_suture = merge_sutures(df_output)
            tifffile.imwrite(os.path.join(output_folder,f"merged_suture_{specimen}.tif"),
                    merged_suture,
                    compression='zlib')

            df_output.to_csv(output_csv_path)
            df_output_list.append(df_output)
        
    except Exception as e:
        print(f" Error: {e}")

        
    try:
        pd.concat(df_output_list, axis=0).to_csv(os.path.join(output_folder,"output_csv.csv"))
    except Exception as e:
        print(f"Error: {e}")

    
    pd.concat(df_output_list, axis=0).to_csv(os.path.join(output_folder,"output_csv.csv"))
    
    

    
    
    # suture_dict = gen_suture_dict_v2(df_id.iloc[0].to_dict(), df_mapping)
    # sys.exit()

    
    # img = img[60:430,...]



    # img = img[334,...]



    
    
    ### Combinations using dict


    # id_list = np.unique(img)
    # id_list =  np.array([x for x in id_list if x !=0])
    # # bone_id_list = [5,  6,  9, 11, 19, 23, 24]

    # # bone_id_list =  [x for x in all_id if x not in (0, 40)]

    # # background = np.logical_and((img!=0), (img!=40))
    # mask = np.isin(img, id_list)
    # background = np.where(mask, True, np.zeros((img.shape)))

    # for key, value in suture_dict.items():
    #     print(f"Processing {key}")
    #     output_suture = np.zeros_like(img)
    #     bone_1_id = value[0]
    #     bone_2_id = value[1]
        
    #     # print(f"Finding gaps for {bone_1_id} and {bone_2_id}")
        
    #     bone_1 = img==bone_1_id
    #     bone_2 = img==bone_2_id
        
    #     result = suture_morpho.find_gaps_between_two(bone_1,bone_2,background)
            
    #     if result is None:
    #         print(f"No gaps between {bone_1_id} and {bone_2_id}")
    #     else:
            
    #         if SAVE_ISLANDS:
    #             result[bone_1] = 2
    #             result[bone_2] = 3
            
    #         # output_suture[result==True]=250
    #         output_path = os.path.join(output_folder, f'{key}.tif')
    #         result = result.astype('uint8')
    #         tifffile.imwrite(output_path, result, compression ='zlib')










#### Combinations using list
# combinations = list(itertools.combinations(bone_id_list, 2))



# combinations = [(5,6), (5,10), (6,11)]

# print(f"Number of combinations: {len(combinations)}")

# workspace = r'result/procavia/seg/'
# output_folder = "sutures"
# output_folder = os.path.join(workspace, output_folder)
# os.makedirs(output_folder, exist_ok=True)

# for comb in combinations:
#     output_img = img.copy()
#     bone_1_id = comb[0]
#     bone_2_id = comb[1]
    
#     # print(f"Finding gaps for {bone_1_id} and {bone_2_id}")
    
#     bone_1 = img==bone_1_id
#     bone_2 = img==bone_2_id
    
#     result = suture_morpho.find_gaps_between_two(bone_1,bone_2,background)
        
#     if result is None:
#         print(f"No gaps between {bone_1_id} and {bone_2_id}")
#     else:
#         output_img[result==True]=250
#         output_path = os.path.join(output_folder, f'{bone_1_id}_{bone_2_id}.tif')

#         tifffile.imwrite(output_path, output_img)
  
