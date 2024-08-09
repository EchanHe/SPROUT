import numpy as np
from skimage.measure import marching_cubes
import trimesh
import os, glob
import tifffile
import threading
import json
from datetime import datetime
import matplotlib.pyplot as plt
import json, yaml
from skimage.draw import polygon
from PIL import Image

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
        
        if id==0:
            color = [255, 255, 255, 255]
        # if id>=1 and id <=10:
        #     texture = draw_star(background_color= colors[(id-1)%len(colors)][:3], star_color = [255,255,255])
        #     color = get_vertex_colors_texture(simplified_mesh,texture, scale = 1)
        # elif id>=11 and id<=20:
        #     texture = draw_stride(colors[(id-1)%len(colors)][:3],[255,255,255])
        #     color = get_vertex_colors_texture(simplified_mesh,texture, scale = 1)
        else:
            color = colors[(id-1)%len(colors)]
        
        simplified_mesh.visual.vertex_colors = color
        # Step 4: Save the mesh to a file
        
        simplified_mesh.export(os.path.join(output_dir, "{}.ply".format(id)))

def binary_stack_to_mesh(input_volume , threshold, output_dir, downsample_scale=10):
    
    input_volume = input_volume > threshold
    verts, faces, normals, _ = marching_cubes(input_volume, level=0.5)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    target_faces = mesh.faces.shape[0] // downsample_scale
    simplified_mesh = mesh.simplify_quadric_decimation(target_faces)

    simplified_mesh.visual.vertex_colors = [128,128,128,128]
    simplified_mesh.export(os.path.join(output_dir, f"{threshold}.ply".format(id)))

def split_ints_by_deli(numbers, deli):
    # Sort the list
    sorted_numbers = sorted(numbers)
    
    # Initialize variables
    sublists = []
    current_sublist = []
    
    # Iterate through sorted numbers
    for number in sorted_numbers:
        # Append the number to the current sublist
        current_sublist.append(number)
        
        # Check if the number is greater than a multiple of 5
        if number >= deli and number % deli == 0:
            # Add the current sublist to the list of sublists
            sublists.append(current_sublist)
            # Start a new sublist
            current_sublist = []
    
    # Add any remaining numbers to the sublists
    if current_sublist:
        sublists.append(current_sublist)
    
    return sublists


def apply_planar_mapping(mesh,image, scale = 10):
    # Calculate UVs based on the method
    uv_coords = np.zeros((mesh.vertices.shape[0], 2))
    min_bounds, max_bounds = mesh.bounds

    # Planar mapping based on x and z (ignores y)
    uv_coords[:, 0] = (mesh.vertices[:, 0] - min_bounds[0]) / (max_bounds[0] - min_bounds[0]) * scale
    uv_coords[:, 1] = (mesh.vertices[:, 2] - min_bounds[2]) / (max_bounds[2] - min_bounds[2]) *scale
    
    mesh.visual.uv = uv_coords
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv_coords, image=image)
    
    return mesh

def get_vertex_colors_texture(mesh,image, scale = 10):
    texture_colors = np.array(image)
    texture_height, texture_width = texture_colors.shape[:2]
    
    # Calculate UVs based on the method
    uv_coords = np.zeros((mesh.vertices.shape[0], 2))
    min_bounds, max_bounds = mesh.bounds

    # Planar mapping based on x and z (ignores y)
    uv_coords[:, 0] = (mesh.vertices[:, 0] - min_bounds[0]) / (max_bounds[0] - min_bounds[0]) * scale
    uv_coords[:, 1] = (mesh.vertices[:, 2] - min_bounds[2]) / (max_bounds[2] - min_bounds[2]) *scale
    
    # Find the minimum and maximum UV coordinates
    uv_min = uv_coords.min(axis=0)
    uv_max = uv_coords.max(axis=0)

    # Normalize UV coordinates to the range [0, 1]
    uv_normalized = (uv_coords - uv_min) / (uv_max - uv_min)
    
    # Map UV coordinates to texture indices
    u_indices = (uv_normalized[:, 0] * (texture_width - 1)).astype(int)
    v_indices = ((1-uv_normalized[:, 1]) * (texture_height - 1)).astype(int)
    

    
    vertex_colors = texture_colors[v_indices, u_indices]
   
    return vertex_colors
    

def draw_stride(left_color , right_color, height = 100, width = 100):
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Assign colors to the left and right halves
    image[:, :width//2] = left_color
    image[:, width//2:] = right_color
    return image


def draw_star(background_color, star_color, star_radius=20, num_points=10, image_size=(100,100)):
    """
    Draws a star in the middle of the image.

    :param image_size: Tuple of (height, width) for the image.
    :param background_color: RGB color for the background.
    :param star_color: RGB color for the star.
    :param star_radius: Radius of the star.
    :param num_points: Number of points on the star.
    :return: Numpy array representing the image with the star.
    """
    # Initialize the image with the background color
    star_image = np.full((image_size[0], image_size[1], 3), background_color, dtype=np.uint8)
    # Calculate the center of the image
    center_y, center_x = image_size[0] // 2, image_size[1] // 2
    # Calculate angles for star points
    angles = np.linspace(0, 2 * np.pi, 2 * num_points, endpoint=False)
    
    # Alternate between inner and outer radius for points
    radii = np.empty(2 * num_points)
    radii[0::2] = star_radius
    radii[1::2] = star_radius * 0.5
    
    # Calculate x and y coordinates for the star points
    x_points = center_x + (radii * np.cos(angles)).astype(int)
    y_points = center_y + (radii * np.sin(angles)).astype(int)
    
    # Draw the star
    rr, cc = polygon(y_points, x_points)
    star_image[rr, cc] = star_color

    return Image.fromarray(star_image)

colors_46 = [
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


colors_10 = {'Red': (255, 0, 0, 255),
 'Green': (0, 255, 0, 255),
 'Blue': (0, 0, 255, 255),
 'Yellow': (255, 255, 0, 255),
 'Purple': (127, 0, 127, 255),
 'Cyan': (0, 255, 255, 255),
 'Brown': (153, 76, 0, 255),
 'Pink': (255, 191, 204, 255),
 'Orange': (255, 178, 0, 255),
 'Black': (0, 0, 0, 255)}


default_color = [245,245,245,255]

def save_colormap(save_path = None):
    # Create a colormap
    fig, ax = plt.subplots(figsize=(6, 2))

    # Display colors as a colormap
    for i, color in enumerate(colors_10):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))

    ax.set_xlim(0, len(colors_10))
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(len(colors_10)) + 0.5)
    ax.set_xticklabels(list(colors_10.keys()), rotation=90)
    ax.set_title("Distinct Colors")
    
    # Save the figure to a given location
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')

def merge_plys(ply_files,output_path, keep_color=True,keep_color_files=None):
    
    mesh_list = []
    try:
        for ply_file in ply_files:
            mesh = trimesh.load_mesh(ply_file)
                
            if (keep_color_files is not None) and  (ply_file not in keep_color_files):
                mesh.visual.vertex_colors = [255,255,255,255]
            mesh_list.append(mesh)
        combined_mesh = trimesh.util.concatenate(mesh_list)
        combined_mesh.export(output_path)
    # # Initialize lists to store vertices and faces
    # all_vertices = []
    # all_faces = []
    # colors = []
    # vertex_offset = 0
    # try:
    #     # Iterate over each PLY file
    #     for ply_file in ply_files:
    #         # Load the PLY file using trimesh
    #         mesh = trimesh.load_mesh(ply_file)
    #         # Check if vertex colors are available and access them
    #         # if 'vertex_colors' in mesh.metadata:
    #         #     colors = mesh.visual.vertex_colors
    #         #     print("Vertex colors are available.")
    #         # else:
    #         #     print("No vertex colors found in the PLY file.")
    #         if (keep_color_files is None) or (ply_file in keep_color_files):
    #             colors.append(mesh.visual.vertex_colors)
    #         else:
    #             # if do not preserve colour, set as white
    #             colors.append(len(mesh.visual.vertex_colors) * [default_color])
    #                 # Append vertices and adjust faces
    #         all_vertices.append(mesh.vertices)
    #         all_faces.append(mesh.faces + vertex_offset)
            
    #         # Update the vertex offset for the next mesh
    #         vertex_offset += len(mesh.vertices)

    #     # Combine all vertices and faces
    #     combined_vertices = np.vstack(all_vertices)
    #     combined_faces = np.vstack(all_faces)
    #     combined_vertex_colors = np.vstack(colors)

    #     # Create a new mesh with the combined vertices and faces
    #     combined_mesh = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces)
    #     combined_mesh.visual.vertex_colors = combined_vertex_colors

    #     # Export the combined mesh to a PLY file
    #     combined_mesh.export(output_path)

        print(f"Combined mesh saved to {output_path}")
    except Exception as e:
        print(f"Error during merge_plys(ply_files, keep_color=True): {e}")

def make_mesh_for_tiff(tif_file,output_folder,
                       num_threads,no_zero = True,
                       colormap = "color10"):
    print(f"Creating meshes for {tif_file}")
    # Extract the base name without extension
    base_name = os.path.basename(tif_file)
    folder_name = os.path.splitext(base_name)[0]
    global colors
    if colormap=="color10":
        colors = list(colors_10.values())
    
    
    # Create a new directory with the name of the .tif file (without extension)
    output_sub_dir = os.path.join(output_folder, folder_name)
    os.makedirs(output_sub_dir, exist_ok=True)
    
    
    volume = tifffile.imread(tif_file)
    global volume_array 
    volume_array = np.array(volume)

    #Create ID for non background (e.g. 0)
    id_list = np.unique(volume_array)
    if no_zero:
        id_list =  np.array([x for x in id_list if x !=0])
    
    
    sublists = [id_list[i::num_threads] for i in range(num_threads)]

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


    ### For merging:
    ply_files = np.array([os.path.join(output_sub_dir, f"{number}.ply") for number in id_list])
    merge_ply_output_path = os.path.join(output_sub_dir,"merged.ply")
    
    merge_plys(ply_files,merge_ply_output_path)
    
    deli=len(colors)
    for end in range(deli, max(id_list) + deli + 1, deli):
        start = end - deli
        # print(start, end)
        array_ids = np.logical_and(id_list>=start, id_list<end)
        # print(ply_files[array_ids])
        keep_color_files = ply_files[array_ids]
        if keep_color_files.size!=0:
            merge_ply_output_path = os.path.join(output_sub_dir,f"merged{start+1}_to_{end}.ply")
            merge_plys(ply_files,merge_ply_output_path,keep_color_files=keep_color_files)

    print(f"{tif_file} has completed.\n")
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

    
    start_time = datetime.now()
    print(f"""{start_time.strftime("%Y-%m-%d %H:%M:%S")}
    Mode WHOLE_MESH:{WHOLE_MESH}
    Mode MULTI_FILES:{MULTI_FILES}
            """)
    ################

    
    ###
    # MULTI_FILES = True                                       
    # num_threads = 13
    if MULTI_FILES:
        output_folder = os.path.join(workspace, output_folder)
        # workspace = r'C:\Users\Yichen\OneDrive\work\codes\nhm_bounti_pipeline\result\foram_james'    
        # input_folder = 'result/ai/'
        # output_folder = 'result/ai/'

        input_folder = os.path.join(workspace, input_folder)
        print(f"""
        Input folder {input_folder}      
        output_folder:{output_folder}
                """)
            
        
        tif_files = glob.glob(os.path.join(input_folder, '*.tif'))
        
        for tif_file in tif_files:
            make_mesh_for_tiff(tif_file,output_folder, num_threads,no_zero = True)


    # Only for 
    if SINGLE:
        ### To change your input params ####
        # tif_path = "output_procavia/seg_6000_3000.tif"
        # output_dir = "output_procavia/mesh_seg_6000_3000"
        
        os.makedirs(output_folder,exist_ok=True)

        make_mesh_for_tiff(tif_file,output_folder, num_threads,no_zero = True)   
        

    if WHOLE_MESH:
        os.makedirs(output_folder , exist_ok=True)
        volume = tifffile.imread(img_file)
        binary_stack_to_mesh(volume , threshold,
                             output_folder,
                             downsample_scale=10)
        
        
#### Old codes for multi files
            # print(f"Creating meshes for {tif_file}")
            # # Extract the base name without extension
            # base_name = os.path.basename(tif_file)
            # folder_name = os.path.splitext(base_name)[0]
            
            # # Create a new directory with the name of the .tif file (without extension)
            # output_sub_dir = os.path.join(output_folder, folder_name)
            # os.makedirs(output_sub_dir, exist_ok=True)
            
            
            # volume = tifffile.imread(tif_file)
            # volume_array = np.array(volume)

            # #Create ID for non background (e.g. 0)
            # all_id = np.unique(volume_array)
            # bone_id_list =  [x for x in all_id if x !=0]
            
            
            # sublists = [bone_id_list[i::num_threads] for i in range(num_threads)]

            # # Create a list to hold the threads
            # threads = []

            # # Start a new thread for each sublist
            # for sublist in sublists:
            #     thread = threading.Thread(target=stack_to_mesh, args=(sublist,output_sub_dir,))
            #     threads.append(thread)
            #     thread.start()
                
            # # Wait for all threads to complete
            # for thread in threads:
            #     thread.join()

            # print(f"{tif_file} has completed.\n")