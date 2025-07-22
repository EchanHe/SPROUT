import yaml
import os,sys
import pandas as pd

import sprout_core.config_core as config_core
import sprout_core.sprout_core as sprout_core

from sam_predict import sam_predict

def run_batch_sam(file_path):
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
        optional_params = config_core.validate_input_yaml(config, config_core.input_val_sam_run)

        try:
            _, log_dict =sam_predict(
                img_path=config['img_path'],
                seg_path=config['seg_path'],
                output_folder=config['output_folder'],
                output_filename=optional_params['output_filename'],
                n_points_per_class=optional_params['n_points_per_class'],
                prompt_type=optional_params['prompt_type'],
                sample_neg_each_class=optional_params['sample_neg_each_class'],
                negative_points=optional_params['negative_points'],
                per_cls_mode=optional_params['per_cls_mode'],
                which_sam=optional_params['which_sam'],
                sam_checkpoint=optional_params['sam_checkpoint'],
                sam_model_type=optional_params['sam_model_type'],
                sam2_checkpoint=optional_params['sam2_checkpoint'],
                sam2_model_cfg=optional_params['sam2_model_cfg'])
            df.loc[index,'output_folder'] = log_dict['output_folder']
            
        except Exception as e:
            print(f"Error processing row {index}: {e}")
        
            df.loc[index,'error'] = str(e)
    
    df.to_csv(csv_path + "_running_results.csv", index = False)

if __name__ == "__main__":
    # Get the file path from the first command-line argument or use the default
    file_path = sys.argv[1] if len(sys.argv) > 1 else './batch_sam.yaml'

    run_batch_sam(file_path)