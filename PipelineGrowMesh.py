import make_grow_result
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
            
            
file_path = 'PipelineGrowMesh.yaml'
_, extension = os.path.splitext(file_path)

with open(file_path, 'r') as file:
    data = yaml.safe_load(file)

# Safely get values from the YAML data
to_grow_ids = data.get('to_grow_ids', None)

print(f"processing config he file {file_path}")
if extension == '.yaml':
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    load_config_yaml(config)
    


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

mesh_folder = grow_dict['output_folder']

tif_files = glob.glob(os.path.join(mesh_folder, '*.tif'))

for tif_file in tif_files:
    make_mesh.make_mesh_for_tiff(tif_file,mesh_folder,
                        num_threads,no_zero = True,
                        colormap = "color10")