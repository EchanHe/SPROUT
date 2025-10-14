import make_grow

import yaml
import os,sys
import glob
import pandas as pd

import sprout_core.config_core as config_core
import sprout_core.sprout_core as sprout_core

            


def run_batch_grow(file_path):
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
        optional_params = config_core.validate_input_yaml(config, config_core.input_val_make_grow)    

        try:
            _, log_dict = make_grow.grow_mp(
                img_path = config['img_path'] ,
                seg_path = config['seg_path'],
                boundary_path = optional_params['boundary_path'],
                output_folder = config['output_folder'],

                workspace = None,               

                dilation_steps = config['dilation_steps'],
                thresholds = config['thresholds'],
                upper_thresholds = optional_params["upper_thresholds"],
                num_threads = config['num_threads'],
                
                save_every_n_iters = config['save_every_n_iters'],  
                touch_rule = config['touch_rule'],             
    
                grow_to_end = optional_params["grow_to_end"],
                to_grow_ids = optional_params["to_grow_ids"],
                
                final_grow_output_folder = config['output_folder'],
                
                # base_name =  optional_params["base_name"],
                use_simple_naming =  optional_params["use_simple_naming"],    

                is_sort = optional_params['is_sort'],
                min_growth_size = optional_params['min_growth_size'],
                no_growth_max_iter = optional_params['no_growth_max_iter'],

   
                # For mesh making
                is_make_meshes = optional_params['is_make_meshes'],
                downsample_scale = optional_params['downsample_scale'],
                step_size  = optional_params['step_size']
                
            )

            df.loc[index,'final_output_path'] = log_dict['final_output_path']
            df.loc[index,'output_folder'] = log_dict['output_folder']
        
        except Exception as e:
            print(f"Error occurs when growing on {config['img_path']}")
            df.loc[index,'error'] = str(e)
    
    df.to_csv(csv_path + "_running_results.csv", index = False)



if __name__ == "__main__":
    
    # Get the file path from the first command-line argument or use the default
   
    if len(sys.argv) > 1:
        print(f"Reading config file from command-line argument: {sys.argv[1]}")
        file_path = sys.argv[1]
    else:
        print("No config file specified in arguments. Using default: ./template/batch_grow.yaml")
        file_path = './template/batch_grow.yaml'
    
    run_batch_grow(file_path)
