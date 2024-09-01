import make_seeds
import make_seeds_all
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
    
    file_path = 'PipelineSeed.yaml'
    _, extension = os.path.splitext(file_path)
    print(f"processing config he file {file_path}")
    if extension == '.yaml':
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            workspace = config.get("workspace","")
        load_config_yaml(config)
    

    df = pd.read_csv(csv_path)
    
    #TODO check if This df fits the requirements
    
    #a check to see if all files exist
    suture_morpho.check_tiff_files(df['file_name'])

    
    output_dict_list = []
    

    for idx, row in df.iterrows():
        file_name = row["file_name"]
        file_name_no_ext = os.path.splitext(os.path.basename(file_name))[0]
        
        target_thresholds = eval(row['target_thresholds'])
        
        if 'output_seed_folder' in df.columns:
            output_seed_folder = row['output_seed_folder']
        else:
            output_seed_folder = os.path.join(output_seed_root_dir, file_name_no_ext)
        
        print(file_name, target_thresholds)

        output_dict = make_seeds_all.main(workspace=workspace,
                                      file_name= file_name,
                                      output_log_file = output_log_file,
                                      output_seed_folder = output_seed_folder,
                                      ero_iters = ero_iters,
                                      target_thresholds = target_thresholds,
                                      segments = segments
                                      )
        
        output_dict_list.append(output_dict)
        # make_seeds_all.plot(output_dict , os.path.join(output_seed_folder,"log.png"))
        
    for idx, output_dict in enumerate(output_dict_list):
        make_seeds_all.plot(output_dict , os.path.join(output_seed_root_dir,f"{idx}.png"))

# output = make_seeds.main(workspace= workspace,
#                 file_name= file_name,
#                 output_log_file = output_log_file,
#                 output_seed_folder = output_seed_folder,
#                 num_threads = num_threads,
#                 ero_iters = ero_iters,
#                 target_thresholds = target_thresholds,
#                 segments = segments,
#                 footprints = footprints)

# output_seed_folder = output[0]

# tif_files = glob.glob(os.path.join(output_seed_folder, '*.tif'))

# for tif_file in tif_files:
#     make_mesh.make_mesh_for_tiff(tif_file,output_seed_folder,
#                         num_threads,no_zero = True,
#                         colormap = "color10")