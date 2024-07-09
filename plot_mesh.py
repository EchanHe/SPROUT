import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import threading
from datetime import datetime
import threading

import glob 
def load_meshes(file_paths):
    """Load a list of meshes from file paths."""
    meshes = []
    for file_path in file_paths:
        try:
            mesh = trimesh.load(file_path)
            meshes.append(mesh)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return meshes

def convert_rgba_to_01(rgba):
    """Convert RGBA values from 0-255 range to 0-1 range."""
    return [x / 255.0 for x in rgba]

def plot_edges(meshes, output_file):
    """Plot a list of meshes in one plot and save the plot."""
    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot(111, projection='3d')

    min_bound_x,min_bound_y,min_bound_z =10000,10000,10000
    max_bound_x,max_bound_y,max_bound_z =0,0,0

    for mesh in meshes:
        # Plot each mesh
        # vertices = mesh.vertices
        # faces = mesh.faces
        
        min_bound_x,min_bound_y,min_bound_z = min(min_bound_x,mesh.bounds[0][0]),min(min_bound_y,mesh.bounds[0][1]),min(min_bound_z,mesh.bounds[0][2])
        max_bound_x,max_bound_y,max_bound_z = max(max_bound_x,mesh.bounds[1][0]),max(max_bound_y,mesh.bounds[1][1]),max(max_bound_z,mesh.bounds[1][2])
        # min_bound = mesh.bounds[0]  # (min_x, min_y, min_z)
        # max_bound = mesh.bounds[1]  # (max_x, max_y, max_z)
        colour = mesh.visual.face_colors[0]
        
        colour = convert_rgba_to_01(colour)
        colour[3]=1
        # colour = (1,1,1,1)
        edges = mesh.edges_unique
        edge_points = mesh.vertices[edges]
        lines = Line3DCollection(edge_points, colors=colour, linewidths=1)  # Adjust color, linewidth, and alpha
        ax.add_collection3d(lines)
        
        # for face in faces:
        #     triangle = vertices[face]
        #     ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], color='b', alpha=0.3, edgecolor='none')

    # # Get bounds of the mesh
    # min_bound = mesh.bounds[0]  # (min_x, min_y, min_z)
    # max_bound = mesh.bounds[1]  # (max_x, max_y, max_z)

    # Setting the limits of the plot to the bounds of the mesh
    ax.set_xlim(min_bound_x, max_bound_x)
    ax.set_ylim(min_bound_y, max_bound_y)
    ax.set_zlim(min_bound_z, max_bound_z)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")
    # plt.show()


def plot_3d_mesh(mesh , output_file):
    """
    Plot a 3D mesh model using matplotlib.
    
    Parameters:
    mesh (trimesh.Trimesh): The 3D mesh model to be plotted.
    """
    # Create a figure and a 3D axis
    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot(111, projection='3d')


    colour = convert_rgba_to_01(mesh.visual.face_colors[0])
    
    # Extract vertices and faces
    # vertices = mesh.vertices
    # faces = mesh.faces

    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:,1], 
                    triangles=mesh.faces, Z=mesh.vertices[:,2],
                    color=colour) 

    # # Plot each triangle
    # for face in faces:
    #     triangle = vertices[face]
    #     tri = plt.Polygon(triangle, alpha=0.5, edgecolor='k')
    #     ax.add_patch(tri)
    #     art3d.pathpatch_2d_to_3d(tri, z=0)

    # Auto scale to the mesh size
    scale = mesh.vertices.flatten()
    ax.auto_scale_xyz(scale, scale, scale)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")
    
def plot_each_mesh(meshes,filename_prefix,output_folder):
    for i in range(len(meshes)):
        plot_meshes_subplots(meshes , 
                             os.path.join(output_folder, f'{filename_prefix}_{i}.png'), 
                             focus_id = i)


def plot_meshes_subplots(meshes , output_file, focus_id = None):
    """
    Plot mesh models using matplotlib.
    
    Parameters:
    mesh (trimesh.Trimesh): The 3D mesh model to be plotted.
    """
    # Create a figure and a 3D axis
    fig = plt.figure(figsize=(30,30))
    # ax = fig.add_subplot(111, projection='3d')

    vertices_total=[]
    
    # Define the elevation and azimuth angles
    elevation_angles = [0, 45, 90]  # Vertical angles
    azimuth_angles = range(0, 360, 90)  # Horizontal angles
    azimuth_angles = [0, 90, 180]  # Horizontal angles
    
    min_bound_x,min_bound_y,min_bound_z =10000,10000,10000
    max_bound_x,max_bound_y,max_bound_z =0,0,0
    
    plot_index = 1
    for elev in elevation_angles:
        for azim in azimuth_angles:
            ax = fig.add_subplot(len(elevation_angles), len(azimuth_angles), 
                                 plot_index, projection='3d')
            for mesh_id, mesh in enumerate(meshes):
                vertices_total+=list(mesh.vertices.flatten())
                min_bound_x,min_bound_y,min_bound_z = min(min_bound_x,mesh.bounds[0][0]),\
                    min(min_bound_y,mesh.bounds[0][1]),\
                        min(min_bound_z,mesh.bounds[0][2])
                max_bound_x,max_bound_y,max_bound_z = max(max_bound_x,mesh.bounds[1][0]),\
                    max(max_bound_y,mesh.bounds[1][1]),\
                        max(max_bound_z,mesh.bounds[1][2])
                        
                if focus_id is not None:
                    if mesh_id == focus_id:
                        colour = [1,0,0,1]
                    else:
                        colour = [0.3]*4
                else:    
                    colour = convert_rgba_to_01(mesh.visual.face_colors[0])
                
                # Extract vertices and faces
                # vertices = mesh.vertices
                # faces = mesh.faces

                ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:,1], 
                                triangles=mesh.faces, Z=mesh.vertices[:,2],
                                color=colour) 

            
            ax.auto_scale_xyz(vertices_total, vertices_total, vertices_total)
            ax.set_xlim(min_bound_x, max_bound_x)
            ax.set_ylim(min_bound_y, max_bound_y)
            ax.set_zlim(min_bound_z, max_bound_z)

        # Set labels
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # Set the view angles
            ax.view_init(elev=elev, azim=azim)
            
            # Set title
            ax.set_title(f"Elev: {elev}, Azim: {azim}")
            plot_index += 1
    
    plt.tight_layout()
    plt.savefig(output_file)
    # plt.savefig(output_file)
    print(f"Plot finish")


def plot_meshes_subplots_mp(fig, meshes , focus_id = None):
    """
    Plot a 3D mesh model using matplotlib.
    
    Parameters:
    mesh (trimesh.Trimesh): The 3D mesh model to be plotted.
    """
    # Create a figure and a 3D axis
    # fig = plt.figure(figsize=(30,30))
    # ax = fig.add_subplot(111, projection='3d')

    vertices_total=[]
    
    # Define the elevation and azimuth angles
    elevation_angles = [0, 45]  # Vertical angles
    azimuth_angles = range(0, 360, 90)  # Horizontal angles
    
    min_bound_x,min_bound_y,min_bound_z =10000,10000,10000
    max_bound_x,max_bound_y,max_bound_z =0,0,0
    
    plot_index = 1
    for elev in elevation_angles:
        for azim in azimuth_angles:
            ax = fig.add_subplot(len(elevation_angles), len(azimuth_angles), 
                                 plot_index, projection='3d')
            for mesh_id, mesh in enumerate(meshes):
                vertices_total+=list(mesh.vertices.flatten())
                min_bound_x,min_bound_y,min_bound_z = min(min_bound_x,mesh.bounds[0][0]),\
                    min(min_bound_y,mesh.bounds[0][1]),\
                        min(min_bound_z,mesh.bounds[0][2])
                max_bound_x,max_bound_y,max_bound_z = max(max_bound_x,mesh.bounds[1][0]),\
                    max(max_bound_y,mesh.bounds[1][1]),\
                        max(max_bound_z,mesh.bounds[1][2])
                        
                if focus_id is not None:
                    if mesh_id == focus_id:
                        colour = [1,0,0,1]
                    else:
                        colour = [0.3]*4
                else:    
                    colour = convert_rgba_to_01(mesh.visual.face_colors[0])
                
                # Extract vertices and faces
                # vertices = mesh.vertices
                # faces = mesh.faces

                ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:,1], 
                                triangles=mesh.faces, Z=mesh.vertices[:,2],
                                color=colour) 

            
            ax.auto_scale_xyz(vertices_total, vertices_total, vertices_total)
            ax.set_xlim(min_bound_x, max_bound_x)
            ax.set_ylim(min_bound_y, max_bound_y)
            ax.set_zlim(min_bound_z, max_bound_z)

        # Set labels
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # Set the view angles
            ax.view_init(elev=elev, azim=azim)
            
            # Set title
            ax.set_title(f"Elev: {elev}, Azim: {azim}")
            plot_index += 1
    
    
    # plt.savefig(output_file)
    print(f"Plot finish")



def plot_meshes(meshes , output_file, focus_id = False):
    """
    Plot a 3D mesh model using matplotlib.
    
    Parameters:
    mesh (trimesh.Trimesh): The 3D mesh model to be plotted.
    """
    # Create a figure and a 3D axis
    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot(111, projection='3d')

    vertices_total=[]
    
    min_bound_x,min_bound_y,min_bound_z =10000,10000,10000
    max_bound_x,max_bound_y,max_bound_z =0,0,0
    for mesh_id, mesh in enumerate(meshes):
        vertices_total+=list(mesh.vertices.flatten())
        min_bound_x,min_bound_y,min_bound_z = min(min_bound_x,mesh.bounds[0][0]),\
            min(min_bound_y,mesh.bounds[0][1]),\
                min(min_bound_z,mesh.bounds[0][2])
        max_bound_x,max_bound_y,max_bound_z = max(max_bound_x,mesh.bounds[1][0]),\
            max(max_bound_y,mesh.bounds[1][1]),\
                max(max_bound_z,mesh.bounds[1][2])
                
        if focus_id!=False:
            if mesh_id == focus_id:
                colour = [1,0,0,1]
            else:
                colour = [0.3]*4
        else:    
            colour = convert_rgba_to_01(mesh.visual.face_colors[0])
        
        # Extract vertices and faces
        # vertices = mesh.vertices
        # faces = mesh.faces

        ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:,1], 
                        triangles=mesh.faces, Z=mesh.vertices[:,2],
                        color=colour) 

    
    ax.auto_scale_xyz(vertices_total, vertices_total, vertices_total)
    ax.set_xlim(min_bound_x, max_bound_x)
    ax.set_ylim(min_bound_y, max_bound_y)
    ax.set_zlim(min_bound_z, max_bound_z)

   # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # for angle in range(0, 360, 45):
    #     ax.view_init(elev=30, azim=angle)
    #     plt.savefig(f"output_procavia/3d_mesh_view_angle_{angle}.png")

    # Loop over different azimuth and elevation angles
    elevation_angles = [0, 45]  # Vertical angles
    azimuth_angles = range(0, 360, 90)  # Horizontal angles

    for elev in elevation_angles:
        for azim in azimuth_angles:
            ax.view_init(elev=elev, azim=azim)
            plt.savefig(f"output_procavia/screen_shot/{focus_id}_mesh_{elev}_azim_{azim}.png")

    # plt.savefig(output_file)
    print(f"Plot saved as {output_file}")

def plot_part_v_whole(part_mesh, whole_mesh , output_file, title = ""):
    """
    Plot a 3D mesh model using matplotlib.
    
    Parameters:
    mesh (trimesh.Trimesh): The 3D mesh model to be plotted.
    """
    # Create a figure and a 3D axis
    fig = plt.figure(figsize=(30,30))
    
        # Define the elevation and azimuth angles
    elevation_angles = [0, 45, 90]  # Vertical angles
    azimuth_angles = range(0, 360, 90)  # Horizontal angles
    azimuth_angles = [0,90,180] # Horizontal angles
    
    # azimuth_angles = [None]
    # elevation_angles = [None]
    
    plot_index = 1
    for elev in elevation_angles:
        for azim in azimuth_angles:
            ax = fig.add_subplot(len(elevation_angles), len(azimuth_angles), 
                                 plot_index, projection='3d')
    # ax = fig.add_subplot(111, projection='3d')


            
            ax.plot_trisurf(whole_mesh.vertices[:, 0], whole_mesh.vertices[:,1], 
                    triangles=whole_mesh.faces, Z=whole_mesh.vertices[:,2],
                    color=[0.1]*4 ) 
            
            ax.plot_trisurf(part_mesh.vertices[:, 0], part_mesh.vertices[:,1], 
                    triangles=part_mesh.faces, Z=part_mesh.vertices[:,2],
                    color=[1,0,0,1]  ) 
        
            vertices = whole_mesh.vertices.flatten()
            ax.auto_scale_xyz(vertices, vertices, vertices)
            ax.set_xlim(whole_mesh.bounds[0][0], whole_mesh.bounds[1][0])
            ax.set_ylim(whole_mesh.bounds[0][1], whole_mesh.bounds[1][1])
            ax.set_zlim(whole_mesh.bounds[0][2], whole_mesh.bounds[1][2])

            # Set labels
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Set the view angles
            ax.view_init(elev=elev, azim=azim)
            
            # Set title
            ax.set_title(f"{title}: Elev: {elev}, Azim: {azim}", fontsize=20)
            plot_index += 1
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    plt.savefig(output_file)

def plots_mp(mesh_folders, workspace , whole_mesh_path):
    for mesh_folder in mesh_folders:
        subemesh_files = glob.glob(os.path.join(mesh_folder, '*.ply'))
        subemeshes = load_meshes(subemesh_files)
        output_folder = os.path.join(workspace, 'plot')
        base_name = os.path.basename(mesh_folder)
        output_plot_file = os.path.join(output_folder, f'{base_name}.png')
        print(f"Saving plot to :{output_plot_file}")
        os.makedirs(output_folder,exist_ok=True)
        os.makedirs(output_folder,exist_ok=True)
        # output_plot_file = os.path.join(workspace, 'plot/seg_ero_1iter_thre3500.png')
        
        #########  Plot all separated meshes (colorised) into one space. （meshlab vibe) ###########
        
        plot_meshes_subplots(subemeshes, output_plot_file)
        
        
        #########  Plot one highlighted mesh among all generated meshes meshes.  ###########
        plot_each_mesh(subemeshes,"highlight_mesh",output_folder)
        

        ########## This is used to plot sub vs whole ########
               
        sub_folder_name = os.path.basename(mesh_folder)
        
        output_dir = os.path.join("plot",sub_folder_name) 
        output_dir = os.path.join(workspace, output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        sub_mesh_files = glob.glob(os.path.join(mesh_folder, '*.ply'))
        print(f"Plotting meshes for:{sub_mesh_files}")
        
        
        # List of PLY file paths
        whole_mesh = trimesh.load(whole_mesh_path)
        for sub_mesh_file in sub_mesh_files:
            sub_mesh = trimesh.load(sub_mesh_file)
            
            sub_name = os.path.basename(sub_mesh_file)
            whole_name = os.path.basename(whole_mesh_path)
            
            output_path = os.path.join(output_dir, f'{sub_name}_v_{whole_name}.png' )
            
            plot_part_v_whole(sub_mesh, whole_mesh, output_path , f'{sub_name}_v_{whole_name}')

def main():
    
    num_threads = 4
    workspace = r'C:\Users\Yichen\OneDrive\work\codes\nhm_bounti_pipeline\result\foram_james'
    folder = 'thre_ero_seed_mesh'
    whole_mesh_path = "whole_mesh/0.ply"
    
    folder = os.path.join(workspace, folder)
    mesh_folders = [os.path.join(folder, name) for name in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, name))]

    print(mesh_folders)
    
    sublists = [mesh_folders[i::num_threads] for i in range(num_threads)]

    # Create a list to hold the threads
    threads = []
    # Start a new thread for each sublist
    for sublist in sublists:
        thread = threading.Thread(target=plots_mp, args=(mesh_folders, workspace,whole_mesh_path,))
        threads.append(thread)
        thread.start()
        
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    sys.exit(0)
    # Capture the start time
    start_time = datetime.now()
    
    num_threads = 4
    workspace = r'C:\Users\Yichen\OneDrive\work\codes\nhm_bounti_pipeline\result\foram_james'
    
    sub_mesh_folder = "result/for_plot"
    sub_mesh_folder = os.path.join(workspace, sub_mesh_folder)
    

    
    subemesh_files = glob.glob(os.path.join(sub_mesh_folder, '*.ply'))
    subemeshes = load_meshes(subemesh_files)
    output_folder = os.path.join(workspace, 'plot')
    
    
    base_name = os.path.basename(sub_mesh_folder)
    
    output_plot_file = os.path.join(output_folder, f'{base_name}.png')
    print(f"Saving plot to :{output_plot_file}")
    os.makedirs(output_folder,exist_ok=True)
    # output_plot_file = os.path.join(workspace, 'plot/seg_ero_1iter_thre3500.png')
    
    #########  Plot all separated meshes (colorised) into one space. （meshlab vibe) ###########
    
    plot_meshes_subplots(subemeshes, output_plot_file)
    
    
    #########  Plot one highlighted mesh among all generated meshes meshes.  ###########
    plot_each_mesh(subemeshes,"highlight_mesh",output_folder)
    
    
    ########## This is used to plot sub vs whole ########
    
    whole_mesh_path = "whole_mesh/0.ply"
    whole_mesh_path = os.path.join(workspace, whole_mesh_path)
    
   
    sub_mesh_folder = "result/for_plot/"
    sub_mesh_folder = os.path.join(workspace, sub_mesh_folder)
    
    sub_folder_name = os.path.basename(sub_mesh_folder)
    
    output_dir = os.path.join("plot",sub_folder_name) 
    output_dir = os.path.join(workspace, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    sub_mesh_files = glob.glob(os.path.join(sub_mesh_folder, '*.ply'))
    print(f"Plotting meshes for:{sub_mesh_files}")
    
    
    # List of PLY file paths
    whole_mesh = trimesh.load(whole_mesh_path)
    for sub_mesh_file in sub_mesh_files:
        sub_mesh = trimesh.load(sub_mesh_file)
        
        sub_name = os.path.basename(sub_mesh_file)
        whole_name = os.path.basename(whole_mesh_path)
        
        output_path = os.path.join(output_dir, f'{sub_name}_v_{whole_name}.png' )
        
        plot_part_v_whole(sub_mesh, whole_mesh, output_path , f'{sub_name}_v_{whole_name}')
    
    
    
        # Capture the end time
    end_time = datetime.now()
    # Calculate the duration
    duration = end_time - start_time  
    
    print(f"time for generating plots: {duration}")
    
    
    #James foram 0:02:40.852117
    return
    
    

    ply_files = glob.glob('output_procavia/mesh_seg_6000_3000/*.ply')
    meshes = load_meshes(ply_files)
    
    # ply_files = [
    #     'output/mesh/6.ply',
    #     'output/mesh/7.ply',
    #     'output/mesh/45.ply',
    #     # 'output/mesh/4.ply',
    #     # 'output/mesh/5.ply'
    #     # Add more paths as needed
    # ]

    # Load the meshes
    meshes = load_meshes(ply_files)
    
    # Plot the meshes and save the plot
    output_plot_file = 'output_procavia/mesh_seg_6000_3000.png'
    
    # plot_3d_mesh(meshes[0], output_plot_file)
    # plot_meshes(meshes, output_plot_file)
    
    

if __name__ == "__main__":
    main()
