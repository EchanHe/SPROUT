import numpy as np
from skimage.measure import marching_cubes
import trimesh
import os, glob
import tifffile
import threading
import json


import json, yaml


# Function to recursively create global variables from the config dictionary
def load_config_yaml(config, parent_key=''):
    for key, value in config.items():
        if isinstance(value, dict):
            load_config_yaml(value, parent_key='')
        else:
            globals()[parent_key + key] = value
            

# Define a function to read the configuration and set variables dynamically
def load_config_json(file_path):
    with open(file_path, 'r') as config_file:
        config = json.load(config_file)

    # Dynamically set variables in the global namespace
 
    for key, value in config.items():
        globals()[key] = value

# def load_config(file_path):
#     try:
#         with open(file_path, 'r') as config_file:
#             config = json.load(config_file)
#     except FileNotFoundError:
#         print(f"Error: The file {file_path} was not found.")
#         return None
#     except json.JSONDecodeError:
#         print(f"Error: The file {file_path} is not a valid JSON file.")
#         return None

#     # Fetch configuration values with default fallback
#     MULTI_FILES = config.get("MULTI_FILES")
#     workspace = config.get("workspace")
#     num_threads = config.get("num_threads")
#     input_folder = config.get("input_folder")
#     output_folder = config.get("output_folder")
#     tif_path = config.get("tif_path")

#     # Perform necessary checks and validations
#     if MULTI_FILES is None:
#         print("Error: 'MULTI_FILES' is required in the configuration.")
#         return None

#     if MULTI_FILES:
#         if not input_folder:
#             print("Error: 'input_folder' must be present when 'MULTI_FILES' is True.")
#             return None
#     else:
#         if not tif_path:
#             print("Error: 'tif_path' must be present when 'MULTI_FILES' is False.")
#             return None

#     # Optional: Further validation can be added here
#     required_fields = {
#         "workspace": workspace,
#         "num_threads": num_threads,
#         "output_folder": output_folder,
#     }

#     for field, value in required_fields.items():
#         if value is None:
#             print(f"Error: '{field}' is required in the configuration.")
#             return None

#     # Set global variables (if required)
#     globals().update(required_fields)
#     globals().update({
#         "MULTI_FILES": MULTI_FILES,
#         "input_folder": input_folder,
#         "tif_path": tif_path
#     })

#     # Configuration loaded successfully
#     return config



def stack_to_mesh(bone_id_list , output_dir, downsample_scale=10):
    for id in bone_id_list:
        temp = (volume_array ==id).astype('uint8')
            # # Use marching cubes to obtain the surface mesh
        verts, faces, normals, values = marching_cubes(temp, level=0.5)
        # Step 3: Create a Trimesh object
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        
        # Simplify the mesh
        # Set target number of faces (e.g., reduce to 50% of the original number of faces)
        target_faces = mesh.faces.shape[0] // downsample_scale
        simplified_mesh = mesh.simplify_quadric_decimation(target_faces)
        
        color = colors[id%len(colors)]
        simplified_mesh.visual.face_colors = color
        # Step 4: Save the mesh to a file
        
        simplified_mesh.export(os.path.join(output_dir, "{}.ply".format(id)))

def binary_stack_to_mesh(input_volume , threshold, output_dir, downsample_scale=10):
    
    input_volume = input_volume > threshold
    verts, faces, normals, _ = marching_cubes(input_volume, level=0.5)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    target_faces = mesh.faces.shape[0] // downsample_scale
    simplified_mesh = mesh.simplify_quadric_decimation(target_faces)

    simplified_mesh.visual.face_colors = [128,128,128,128]
    simplified_mesh.export(os.path.join(output_dir, f"{threshold}.ply".format(id)))


colors = [
    [255, 0, 0, 255],      # Red
    [0, 255, 0, 255],      # Green
    [0, 0, 255, 255],      # Blue
    [255, 255, 0, 255],    # Yellow
    [0, 255, 255, 255],    # Cyan
    [255, 0, 255, 255],    # Magenta
    [255, 165, 0, 255],    # Orange
    [128, 0, 128, 255],    # Purple
    [165, 42, 42, 255],    # Brown
    [0, 128, 128, 255],    # Teal
    [255, 182, 193, 255],  # Light Pink
    [0, 100, 0, 255],      # Dark Green
    [135, 206, 235, 255],  # Sky Blue
    [255, 215, 0, 255],    # Gold
    [0, 139, 139, 255],    # Dark Cyan
    [255, 20, 147, 255],   # Deep Pink
    [255, 140, 0, 255],    # Dark Orange
    [75, 0, 130, 255],     # Indigo
    [128, 128, 0, 255],    # Olive
    [139, 0, 139, 255],    # Dark Magenta
    [255, 105, 180, 255],  # Hot Pink
    [0, 0, 139, 255],      # Dark Blue
    [0, 255, 127, 255],    # Spring Green
    [127, 255, 0, 255],    # Chartreuse
    [0, 191, 255, 255],    # Deep Sky Blue
    [218, 165, 32, 255],   # Goldenrod
    [147, 112, 219, 255],  # Medium Purple
    [60, 179, 113, 255],   # Medium Sea Green
    [210, 105, 30, 255],   # Chocolate
    [70, 130, 180, 255],   # Steel Blue
    [222, 184, 135, 255],  # Burlywood
    [255, 69, 0, 255],     # Orange Red
    [139, 69, 19, 255],    # Saddle Brown
    [255, 228, 181, 255],  # Moccasin
    [72, 61, 139, 255],    # Dark Slate Blue
    [85, 107, 47, 255],    # Dark Olive Green
    [240, 230, 140, 255],  # Khaki
    [173, 216, 230, 255],  # Light Blue
    [244, 164, 96, 255],   # Sandy Brown
    [32, 178, 170, 255],   # Light Sea Green
    [199, 21, 133, 255],   # Medium Violet Red
    [0, 250, 154, 255],    # Medium Spring Green
    [220, 20, 60, 255],    # Crimson
    [123, 104, 238, 255],  # Medium Slate Blue
    [144, 238, 144, 255],  # Light Green
    [255, 250, 205, 255]   # Lemon Chiffon
]

if __name__ == "__main__":
    ############ Config
    file_path = './make_mesh.yaml'
    
    _, extension = os.path.splitext(file_path)
    print(f"processing config he file {file_path}")
    if extension == '.json':
        
        load_config_json(file_path)
    elif extension == '.yaml':
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        load_config_yaml(config)


    ################

    if WHOLE_MESH:
        os.makedirs(output_whole_folder , exist_ok=True)
        volume = tifffile.imread(img_file)
        binary_stack_to_mesh(volume , threshold,
                             output_whole_folder,
                             downsample_scale=10)
    
    ###
    # MULTI_FILES = True                                       
    # num_threads = 13
    if MULTI_FILES:
        # workspace = r'C:\Users\Yichen\OneDrive\work\codes\nhm_bounti_pipeline\result\foram_james'    
        # input_folder = 'result/ai/'
        # output_folder = 'result/ai/'
        
        
        input_folder = os.path.join(workspace, input_folder)
        output_folder = os.path.join(workspace, output_folder)
        
        tif_files = glob.glob(os.path.join(input_folder, '*.tif'))
        
        for tif_file in tif_files:
            print(f"Creating meshes for {tif_file}")
            # Extract the base name without extension
            base_name = os.path.basename(tif_file)
            folder_name = os.path.splitext(base_name)[0]
            
            # Create a new directory with the name of the .tif file (without extension)
            output_sub_dir = os.path.join(output_folder, folder_name)
            os.makedirs(output_sub_dir, exist_ok=True)
            
            
            volume = tifffile.imread(tif_file)
            volume_array = np.array(volume)

            #Create ID for non background (e.g. 0)
            all_id = np.unique(volume_array)
            bone_id_list =  [x for x in all_id if x !=0]
            
            
            sublists = [bone_id_list[i::num_threads] for i in range(num_threads)]

            # Create a list to hold the threads
            threads = []

            # Start a new thread for each sublist
            for sublist in sublists:
                thread = threading.Thread(target=stack_to_mesh, args=(sublist,output_sub_dir,))
                threads.append(thread)
                thread.start()
                
            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            print(f"{tif_file} has completed.")


    else:
            
        ### To change your input params ####
        # tif_path = "output_procavia/seg_6000_3000.tif"
        # output_dir = "output_procavia/mesh_seg_6000_3000"
        tif_path = os.path.join(workspace,tif_path)
        output_folder = os.path.join(workspace, output_folder)
        
        os.makedirs(output_folder,exist_ok=True)


       
        directory, filename = os.path.split(tif_path)
        volume = tifffile.imread(tif_path)
        volume_array = np.array(volume)

        all_id = np.unique(volume_array)
        bone_id_list =  [x for x in all_id if x !=0]
         
        sublists = [bone_id_list[i::num_threads] for i in range(num_threads)]

        # Create a list to hold the threads
        threads = []

        # Start a new thread for each sublist
        for sublist in sublists:
            thread = threading.Thread(target=stack_to_mesh, args=(sublist,))
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        print("All threads have completed.")