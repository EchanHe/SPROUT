import make_seeds
import make_mesh
import yaml
import os
import glob

def load_config_yaml(config, parent_key=''):
    for key, value in config.items():
        if isinstance(value, dict):
            load_config_yaml(value, parent_key='')
        else:
            globals()[parent_key + key] = value
            
            
file_path = 'PipelineSeedMesh.yaml'
_, extension = os.path.splitext(file_path)
print(f"processing config he file {file_path}")
if extension == '.yaml':
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    load_config_yaml(config)
    


output = make_seeds.main(workspace= workspace,
                file_name= file_name,
                output_log_file = output_log_file,
                output_seed_folder = output_seed_folder,
                num_threads = num_threads,
                ero_iters = ero_iters,
                target_thresholds = target_thresholds,
                segments = segments,
                footprints = footprints)

output_seed_folder = output[0]

tif_files = glob.glob(os.path.join(output_seed_folder, '*.tif'))

for tif_file in tif_files:
    make_mesh.make_mesh_for_tiff(tif_file,output_seed_folder,
                        num_threads,no_zero = True,
                        colormap = "color10")