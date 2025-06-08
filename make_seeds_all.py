import tifffile

import numpy as np
import threading
import os, sys
from datetime import datetime
import itertools

import make_mesh
import glob

lock = threading.Lock()


import multiprocessing
max_threads = multiprocessing.cpu_count()

# import configparser
import sprout_core.sprout_core as sprout_core
import sprout_core.config_core as config_core
import sprout_core.vis_lib as vis_lib

import make_seeds


import json, yaml
            
pre_set_footprint_list = [
    ["ball"],
    ["ball_XY"],
    ["ball_YZ"] ,
    ["ball_XZ"]

]

pre_set_sub_folders = [
    "ball",
    "ball_XY",
    "ball_YZ",
    "ball_XZ"
]


# pre_set_sub_folders = [
#     "seeds_ball",
#     "seeds_XY",
#     "seeds_YZ",
#     "seeds_XZ"
# ]


pre_set_footprint_list_2d = [
    ["disk"],
    ["X"],
    ["Y"] 
]

pre_set_sub_folders_2d = [
    "disk",
    "X",
    "Y"]

# pre_set_sub_folders_2d = [
#     "seeds_disk",
#     "seeds_X",
#     "seeds_Y"]


support_footprints =['ball','cube',
                     'ball_XY','ball_XZ','ball_YZ',
                     'X','Y','Z',
                     '2XZ_1Y','2XY_1Z','2YZ_1X']




def for_pipeline_wrapper(img_path, boundary_path=None, **kwargs):
    """
    Wrapper for the `for_pipeline` function that reads `img_path` and `boundary_path`
    and passes them along with additional keyword arguments to the original function.
    
    Parameters:
        img_path (str): Path to the image file to be processed.
        boundary_path (str, optional): Path to the boundary file. Defaults to None.
        **kwargs: Additional parameters to pass to `make_seeds_all`.
    """

    
    # Prepare values for printing
    values_to_print = {
        "Boundary Path": boundary_path if boundary_path else "None"
    }
    # Print detailed values
    print(f"Processing Image: {img_path}")
    for key, value in values_to_print.items():
        print(f"  {key}: {value}")
    
    
    # Read the image from img_path
    img = tifffile.imread(img_path)
    # print(f"Loaded image from: {img_path}")
    
    # Read the boundary from boundary_path, if provided
    boundary = None
    if boundary_path is not None:
        boundary = tifffile.imread(boundary_path)
    
    # Call the original function with all the necessary arguments
    output_dict = make_seeds_all(img=img, boundary=boundary, **kwargs)
    
    return output_dict



def make_seeds_all(**kwargs):
    img = kwargs.get('img', None) 
    boundary  = kwargs.get('boundary', None) 
    
    # Input and Output
    workspace = kwargs.get('workspace', None)
    file_name = kwargs.get('file_name', None)  
    

    output_folder = kwargs.get('output_folder', None) 
    name_prefix = kwargs.get('name_prefix', None) 
    
    # If input does not contain image,
    # But has image file path, then read it in
    if img is None:
        file_path = os.path.join(workspace, file_name)
        img = tifffile.imread(file_path)
        img_name = file_path
        name_prefix = os.path.basename(file_name)
    else:
        img_name = name_prefix
    
    if workspace is not None:
        output_folder =os.path.join(workspace, output_folder)
    
    # Seed generation related 
    ero_iters = kwargs.get('ero_iters', None)
    thresholds = kwargs.get('thresholds', None) 
    
    upper_thresholds = kwargs.get('upper_thresholds', None) 

    if isinstance(thresholds, int):
        thresholds = [thresholds]
    
    if isinstance(upper_thresholds, int):
        upper_thresholds = [upper_thresholds]
 
    segments = kwargs.get('segments', None)  
    num_threads = kwargs.get('num_threads', None) 
    
    is_make_meshes = kwargs.get('is_make_meshes', False) 
    downsample_scale = kwargs.get('downsample_scale', 10) 
    step_size  = kwargs.get('step_size', 1) 

    input_footprints = kwargs.get('input_footprints', None)

    if num_threads is None:
        num_threads = 1

    if num_threads>=max_threads:
        num_threads = max_threads-1


    if upper_thresholds is not None:
        assert len(thresholds) == len(upper_thresholds), "Thresholds and upper thresholds do not have the same length."   
        for a, b in zip(thresholds, upper_thresholds):
            assert a < b, "lower_threshold must be smaller than upper_threshold"

    is_3d = (img.ndim == 3)
    


    output_dict = {
        "output_seed_sub_folders":[],
        "output_log_files":[]
    }
    
    # for ero_iter in ero_iters:
        
    ## Dealing with footprints
    if input_footprints is None:
        # if input footprints is not provided, use pre-set footprints
        if is_3d:
            footprint_list = [footprint*ero_iters for footprint in pre_set_footprint_list]
            output_seed_sub_folders = pre_set_sub_folders
        else:
            footprint_list = [footprint*ero_iters for footprint in pre_set_footprint_list_2d]
            output_seed_sub_folders = pre_set_sub_folders_2d
    else:
        if isinstance(input_footprints, str):
        # asser len(input_footprints)
        # if it is provided, check if it is valid
            assert input_footprints in support_footprints, f"footprint {input_footprints} is invalid, use supported footprints"
            footprint_list = [[input_footprints]*ero_iters]
            # check_support_footprint = [footprint in support_footprints for footprint in input_footprints]
            # if not np.all(check_support_footprint):
            #     raise ValueError(f"footprint {input_footprints} is invalid, use supported footprints")
            # footprint_list = [[footprint]*ero_iters for footprint in input_footprints]
            output_seed_sub_folders = [input_footprints]   
        elif isinstance(input_footprints, list):
            assert len(input_footprints) ==ero_iters, "If input_footprints is a list, it must have the same length as ero_iters"
            
            check_support_footprint = [footprint in support_footprints for footprint in input_footprints]
            if not np.all(check_support_footprint):
                raise ValueError(f"footprint {input_footprints} is invalid, use supported footprints")
            
            footprint_list = input_footprints
            output_seed_sub_folders = ["custom_footprints"]
    # Create threshold pairs
    if upper_thresholds is not None:
        thresholds = list(zip(thresholds, upper_thresholds))
    threshold_ero_iter_pairs = list(itertools.product(thresholds, [ero_iters]))  
    sublists = [threshold_ero_iter_pairs[i::num_threads] for i in range(num_threads)]   

    for footprints, output_seed_sub_folder in zip(footprint_list,output_seed_sub_folders):

        # Init the folders and path for output files
        output_seed_sub_folder = os.path.join(output_folder,name_prefix, output_seed_sub_folder)
        os.makedirs(output_seed_sub_folder , exist_ok=True)
        

        output_json_path = os.path.join(output_seed_sub_folder,
                                        f"seed_log_{datetime.now().strftime('%Y%m%d_%H%M')}.json")


        start_time = datetime.now()
        print(f"""{start_time.strftime("%Y-%m-%d %H:%M:%S")}
        Making erosion seeds for 
            Img: {img_name}
            Is 3D image: {is_3d}
            Threshold for Img {thresholds}
            Erode {ero_iters} iterations
            Keeping {segments} components
            Erosion footprints {footprints}
            Running in {num_threads} threads
            Output Folder {output_seed_sub_folder}
                """)      

        # Create a list to hold the threads
        threads = []


        # Start a new thread for each sublist
        for sublist in sublists:
        
            thread = threading.Thread(target=make_seeds.find_seed_by_ero_mp, args=(img,sublist, segments,
                                                                        output_seed_sub_folder,output_json_path, footprints,
                                                                        boundary))
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # print(f"All threads have completed. Log is saved at {output_json_path},seeds are saved at {output_seed_folder}")

        
        end_time = datetime.now()
        running_time = end_time - start_time
        total_seconds = running_time.total_seconds()
        minutes, _ = divmod(total_seconds, 60)
        print(f"Running time:{minutes}")
    
        output_dict["output_seed_sub_folders"].append(output_seed_sub_folder)
        output_dict["output_log_files"].append(output_json_path)
        
        # Make meshes  
        if is_make_meshes:  
            tif_files = glob.glob(os.path.join(output_seed_sub_folder, '*.tif'))

            for tif_file in tif_files:
                make_mesh.make_mesh_for_tiff(tif_file,output_seed_sub_folder,
                                    num_threads=num_threads,no_zero = True,
                                    colormap = "color10",
                                    downsample_scale=downsample_scale,
                                    step_size=step_size)
        
        config_core.save_config_with_output({
            "output_dict": output_dict,
            "params": kwargs},output_seed_sub_folder)
        
    return output_dict
 
def plot(output_dict, full_log_plot_path):
    output_log_files = output_dict["output_log_files"]
    output_seed_sub_folders = output_dict["output_seed_sub_folders"]
    
    plot_list = []
    for output_log_file, output_seed_sub_folder in zip(output_log_files,output_seed_sub_folders):
        
        with open(output_log_file, 'r') as config_file:
            json_data = json.load(config_file)
        
        vis_lib.plot_seeds_log_json(json_data, os.path.join(output_seed_sub_folder, "seeds.png"))
        
        # plot_data = vis_lib.seeds_json_to_plot_ready(output_log_file)
        # vis_lib.plot_seeds_log(plot_data, os.path.join(output_seed_sub_folder, "seeds.png"))
        
        plot_list.append(os.path.join(output_seed_sub_folder, "seeds.png"))

    
    vis_lib.merge_plots(plot_list, full_log_plot_path)

if __name__ == "__main__":
    # Get the file path from the first command-line argument or use the default
    file_path = sys.argv[1] if len(sys.argv) > 1 else './make_seeds_all.yaml'
    
    _, extension = os.path.splitext(file_path)
    print(f"processing config the file {file_path}")

    if extension == '.yaml':
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            optional_params = config_core.validate_input_yaml(config, config_core.input_val_make_seeds_all)
            
        # optional_params_2 = sprout_core.assign_optional_params(config, sprout_core.optional_params_default_seeds)
        
    
    if optional_params['boundary_path'] is not None:
        boundary = tifffile.imread(optional_params['boundary_path'])
    else:
        boundary = None
    
    output_dict = make_seeds_all(
        boundary = boundary,
        workspace= optional_params['workspace'],
            file_name= config['file_name'],
            output_folder = config['output_folder'],
            num_threads = config['num_threads'],
            ero_iters = config['ero_iters'],
            thresholds = config['thresholds'],
            segments = config['segments'],
            upper_thresholds = optional_params['upper_thresholds'],
            
            is_make_meshes = optional_params['is_make_meshes'],
            downsample_scale = optional_params['downsample_scale'],
            step_size  = optional_params['step_size'],
            input_footprints = optional_params['footprints']
            )
    
    
    # Make plot based on the seeds log json
    # Doing this after parallel/multi processing
    # plot(output_dict, os.path.join(os.path.join(optional_params['workspace'], config['output_folder'], "full_log.png")))
