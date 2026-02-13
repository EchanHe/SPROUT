import tifffile
import yaml
import os,sys
import pandas as pd

import sprout_core.config_core as config_core
import sprout_core.sprout_core as sprout_core
from pathlib import Path

from sprout_core.nninteractive_predict import nninter_main, parse_point_config, parse_scribble_config
from torch.cuda import is_available as torch_cuda_is_available

def run_batch_nninteractive(file_path):
    _, extension = os.path.splitext(file_path)
    print(f"processing config the file {file_path}")
    if extension == '.yaml':
        with open(file_path, 'r') as file:
            yaml_config = yaml.safe_load(file)

    print("Config for pipeline")
    for key, value in yaml_config.items():
        print(f"\t{key}: {value}")

    csv_path = yaml_config['csv_path']
    df = pd.read_csv(csv_path)

    sprout_core.check_tiff_files(df['img_path'])
    sprout_core.check_tiff_files(df['seg_path'])

    for index, row in df.iterrows():
        ## Initial the config and optional parameters for each row
        yaml_config.pop("csv_path", None)
        config = config_core.merge_row_and_yaml_no_conflict(dict(row), yaml_config)
        optional_params = config_core.validate_input_yaml(config, config_core.input_val_nninteractive_run)
        
        # Derive some parameters for nninter_main()
        
        # use filename as the outfolder's subfolder
        per_file_output_folder = os.path.join(config['output_folder'], Path(config['img_path']).stem)
        os.makedirs(per_file_output_folder, exist_ok=True)

        point_config = parse_point_config(optional_params)
        
        scribble_config = parse_scribble_config(optional_params)



        device = optional_params['device']
        if device.startswith('cuda') and not torch_cuda_is_available():
            print("[WARNING] CUDA not available, switching to CPU")
            device = 'cpu'

        try:
            nn_output = nninter_main(
                model_path=config['model_path'],
                img_path=config['img_path'],
                seg_path=config['seg_path'],
                device=device,
                prompt_type=config['prompt_type'],
                point_config=point_config,
                scribble_config=scribble_config,
                output_folder=per_file_output_folder,
                return_per_class_masks=optional_params['return_per_class_masks']
            )
            if len(nn_output) == 3:
                total_mask, _, log_dict = nn_output
                
            elif len(nn_output) == 2:
                total_mask, log_dict = nn_output
            output_path = Path(config['output_folder']) / f"{Path(config['img_path']).stem}_segmentation.tif"
            tifffile.imwrite(output_path, total_mask , compression='zlib')
            df.loc[index,'output_folder'] = log_dict['output_folder']
            
        except Exception as e:
            print(f"Error processing row {index}: {e}")
        
            df.loc[index,'error'] = str(e)
    
    df.to_csv(csv_path + "_running_results.csv", index = False)

if __name__ == "__main__":
    # Get the file path from the first command-line argument or use the default
    if len(sys.argv) > 1:
        print(f"Reading config file from command-line argument: {sys.argv[1]}")
        file_path = sys.argv[1]
    else:
        print("No config file specified in arguments. Using default: ./template/batch_sam.yaml")
        file_path = './template/batch_nninteractive.yaml'
        
    run_batch_nninteractive(file_path)