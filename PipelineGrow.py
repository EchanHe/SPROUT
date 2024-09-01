import make_grow_result
import make_mesh
import yaml
import os
import glob
import pandas as pd
import suture_morph.vis_lib as vis_lib
import suture_morph.suture_morpho as suture_morpho

def load_config_yaml(config, parent_key=''):
    for key, value in config.items():
        if isinstance(value, dict):
            load_config_yaml(value, parent_key='')
        else:
            globals()[parent_key + key] = value
            
            
if __name__ == "__main__":
    
    file_path = 'PipelineGrow.yaml'
    _, extension = os.path.splitext(file_path)
    print(f"processing config he file {file_path}")
    if extension == '.yaml':
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            workspace = config.get("workspace","")
            to_grow_ids = config.get("to_grow_ids" , None)
        load_config_yaml(config)
    
    
    
    df = pd.read_csv(csv_path)

    #TODO check if This df fits the requirements
    
    #a check to see if all files exist
    suture_morpho.check_tiff_files(df['img_path'])
    suture_morpho.check_tiff_files(df['seg_path'])
    
    
    grow_dict_list = []
    
    # Or the multi thread version:
    for idx, row in df.iterrows():
      
        img_path = row["img_path"]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        seg_path = row["seg_path"]  
        seg_name = os.path.splitext(os.path.basename(seg_path))[0]
        
        grow_name = img_name + "_" + seg_name
        
        
        if 'output_folder' in df.columns:
            output_folder = row['output_folder']
        else:
            output_folder = os.path.join(output_root_dir, grow_name)
        
        if 'save_interval' in df.columns:
            save_interval = eval(row['save_interval'])

        if 'thresholds' in df.columns:
            thresholds = eval(row['thresholds'])
        
        if 'dilate_iters' in df.columns:
            dilate_iters = eval(row['dilate_iters'])
            
        if 'to_grow_ids' in df.columns:
            thresholds = eval(row['to_grow_ids'])
            
        if 'touch_rule' in df.columns:
            touch_rule = row['touch_rule']

        
        grow_dict = make_grow_result.main(
            dilate_iters = dilate_iters,
            thresholds = thresholds,
            save_interval = save_interval,  
            touch_rule = touch_rule, 
            
            workspace = workspace,
            img_path = img_path,
            seg_path = seg_path,
            output_folder = output_folder,
            to_grow_ids = to_grow_ids
        )
        
        # TODO , check is it possible to making plot multi-thread using plt
        grow_dict_list.append(grow_dict)

        # if is_make_mesh:
        #     mesh_folder = grow_dict['output_folder']
        #     tif_files = glob.glob(os.path.join(mesh_folder, '*.tif'))
        #     for tif_file in tif_files:
        #         make_mesh.make_mesh_for_tiff(tif_file,mesh_folder,
        #                             num_threads,no_zero = True,
        #                             colormap = "color10")

    for grow_dict in grow_dict_list:
        vis_lib.plot_grow(pd.read_csv(grow_dict['log_path']),
            grow_dict['log_path'] +".png")