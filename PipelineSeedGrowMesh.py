import make_grow_result
import make_mesh
from make_seeds_foram import separate_chambers
import yaml
import os
import glob


# img_paths = './result/foram_james/input/ai/*.tif'
img_paths = glob.glob('./result/foram_james/input/ai/*.tif')
print(img_paths)
output_folder = './result/foram_james/seeds/'

grow_folder_name = "foram_grow"
for img_path in img_paths:
    seed ,bone_ids_dict , split_log=  separate_chambers(img_path, 
                                                        threshold = 255,
                                                        output_folder = output_folder,
                                                        n_iters=5,segments=25,
                                                        save_every_iter = True)

    base_name = os.path.basename(img_path)
    workspace = os.path.join(output_folder , base_name)
    
    output_grow_folder = os.path.join(workspace,grow_folder_name)
    seeds = glob.glob(os.path.join(workspace,f"{base_name}*.tif"))

    print(f"Growing on seed files:{seeds}")
    #Make grow on both sorted and not sorted grown results
    for seed_path in seeds:

        output = make_grow_result.main(
            dilate_iters = [4],
            thresholds = [0],
            save_interval = 2,  
            touch_rule = 'no', 
            
            workspace = None,
            img_path = img_path,
            seg_path = seed_path,
            output_folder = output_grow_folder
        )

    mesh_folder = output_grow_folder
    num_threads = 10
    tif_files = glob.glob(os.path.join(mesh_folder, '*.tif'))
    print(f"\nMaking meshes for {tif_files}")
    for tif_file in tif_files:
        make_mesh.make_mesh_for_tiff(tif_file,mesh_folder,
                            num_threads,no_zero = True,
                            colormap = "color10")