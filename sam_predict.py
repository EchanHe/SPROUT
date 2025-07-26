import numpy as np
import os
import json


import tempfile
from datetime import datetime
from tqdm import tqdm

import sprout_core.sam_core as sam_core
import sprout_core.config_core as config_core
import tifffile
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="SAM segmentation script")
    parser.add_argument("--img_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--seg_path", type=str, required=True, help="Path to input segmentation")
    parser.add_argument("--n_points_per_class", type=int, default=3, help="Number of points per class")
    parser.add_argument("--prompt_type", type=str, choices=["point", "bbox"], default="point", help="Prompt type")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder")
    parser.add_argument("--output_filename", type=str, default="output.tif", help="Output filename")
    parser.add_argument("--device", type=str, default=None, help="Device to run SAM on (e.g., 'cuda:0' or 'cpu')")
    parser.add_argument("--sample_neg_each_class", action="store_true", help="Sample negative points for each class")
    parser.add_argument("--negative_points", type=int, default=1, help="Number of negative points per class")
    parser.add_argument("--per_cls_mode", action="store_true", help="Use per-class mode")
    parser.add_argument("--which_sam", type=str, choices=["sam1", "sam2"], default="sam1", help="Which SAM model to use")
    parser.add_argument("--sam_checkpoint", type=str, default="../segment-anything-main/checkpoints/sam_vit_h_4b8939.pth", help="SAM1 checkpoint path")
    parser.add_argument("--sam_model_type", type=str, default="vit_h", help="SAM1 model type")
    parser.add_argument("--sam2_checkpoint", type=str, default="../sam2/checkpoints/sam2.1_hiera_large.pt", help="SAM2 checkpoint path")
    parser.add_argument("--sam2_model_cfg", type=str, default="../sam2/configs/sam2.1/sam2.1", help="SAM2 config path")
    parser.add_argument("--custom_checkpoint", type=str, default=None, help="Optional custom checkpoint path")
    return parser.parse_args()

def run_sam_yaml(file_path):
    _, extension = os.path.splitext(file_path)
    print(f"processing config the file {file_path}")

    if extension == '.yaml':
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            optional_params = config_core.validate_input_yaml(config, config_core.input_val_sam_run)

    print(config)
    print(optional_params)
    
    sam_predict(
        img_path=config['img_path'],
        seg_path=config['seg_path'],
        output_folder=config['output_folder'],
        output_filename=optional_params['output_filename'],
        n_points_per_class=optional_params['n_points_per_class'],
        prompt_type=optional_params['prompt_type'],
        sample_neg_each_class=optional_params['sample_neg_each_class'],
        negative_points=optional_params['negative_points'],
        sample_method=optional_params['sample_method'],
        per_cls_mode=optional_params['per_cls_mode'],
        which_sam=optional_params['which_sam'],
        sam_checkpoint=optional_params['sam_checkpoint'],
        sam_model_type=optional_params['sam_model_type'],
        sam2_checkpoint=optional_params['sam2_checkpoint'],
        sam2_model_cfg=optional_params['sam2_model_cfg'])

def sam_predict(img_path,
                seg_path,
                n_points_per_class=3,
                prompt_type='point',  # 'point' or 'bbox'
                output_folder="./sam_fused_output",
                output_filename= None,
                sample_neg_each_class=False,
                negative_points=1,
                sample_method='random',
                per_cls_mode=True,
                which_sam='sam1',
                sam_checkpoint="../segment-anything-main/checkpoints/sam_vit_h_4b8939.pth",
                sam_model_type="vit_h",
                sam2_checkpoint="../sam2/checkpoints/sam2.1_hiera_large.pt",
                sam2_model_cfg="../sam2/configs/sam2.1/sam2.1"
):
    """
    Main function to run SAM segmentation and save results.
    """

    log_dict = {
        "img_path": img_path,
        "seg_path": seg_path,
        "output_folder": output_folder}
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    all_sam_dirs = {}
    # if the image is 3D, we will extract slices along X, Y, Z axes
    img = tifffile.imread(img_path)
    seg = tifffile.imread(seg_path)
    
    if output_filename is None:
        output_filename = "sam_prediction"+os.path.splitext(os.path.basename(img_path))[0] + ".tif"
    
    if img.ndim == 3:
    
        for axis in ['X', 'Y', 'Z']:
            
            # base_temp_dir = os.path.join(tempfile.gettempdir() ,"sprout", f"{timestamp}" ,axis)
            
            base_temp_dir = os.path.join(output_folder ,"tempfile" ,axis)
            output_prompt_dir = os.path.join(base_temp_dir, "prompts")
            output_img_dir = os.path.join(base_temp_dir, "imgs")
            print(f"Using temporary output folders:\n  Prompts: {output_prompt_dir}\n  Images:  {output_img_dir}")

            

            outputs = sam_core.extract_slices_and_prompts(
                img = img,
                seg = seg,
                output_prompt_dir=output_prompt_dir,
                output_img_dir=output_img_dir,
                axis=axis,
                n_points_per_class=n_points_per_class,
                slice_range=None,
                prompt_type=prompt_type,  # 'point' or 'bbox'
                sample_neg_each_class=sample_neg_each_class,
                negative_points=negative_points,
                sample_method=sample_method
            )
            
            
            sam_dir = os.path.join(base_temp_dir, "sam")
            sam_core.init_and_run(outputs["images_dir"], 
                        which_sam = which_sam,
                        device=None,
                        prompt_dir=outputs["prompts_dir"], 
                        multimask_output=False,
                        save_dir=sam_dir, 
                        overlay_dir=None,
                        sam_checkpoint = sam_checkpoint,
                        sam_model_type = sam_model_type, 
                        sam2_checkpoint = sam2_checkpoint,
                        sam2_model_cfg = sam2_model_cfg)
            
            all_sam_dirs[axis] = sam_dir

        seg_fused = sam_core.combine_3d_segmentations(
            img = img,
            seg = seg,
            sam_dirs=all_sam_dirs,
            output_folder=output_folder,
            output_filename=output_filename,
            lower_threshold=None,
            upper_threshold=None,
            per_cls_mode=per_cls_mode
        )
    else:
        # base_temp_dir = os.path.join(tempfile.gettempdir() ,"sprout", f"{timestamp}" )
        
        base_temp_dir = os.path.join(output_folder ,"tempfile" )
        output_prompt_dir = os.path.join(base_temp_dir, "prompts")
        output_img_dir = os.path.join(base_temp_dir, "imgs")
        print(f"Using temporary output folders:\n  Prompts: {output_prompt_dir}\n  Images:  {output_img_dir}")

        
        outputs = sam_core.extract_slices_and_prompts(
            img = img,
            seg = seg,
            output_prompt_dir=output_prompt_dir,
            output_img_dir=output_img_dir,
            n_points_per_class=n_points_per_class,
            prompt_type=prompt_type,  # 'point' or 'bbox'
            per_slice_2d_input=True,
            sample_neg_each_class = sample_neg_each_class,
            negative_points=negative_points
        )
        
        
        sam_dir = os.path.join(base_temp_dir, "sam")
        sam_core.init_and_run(outputs["images_dir"], 
                    which_sam = which_sam,
                    device=None,
                    prompt_dir=outputs["prompts_dir"], 
                    multimask_output=False,
                    save_dir=sam_dir, 
                    overlay_dir=None,
                    sam_checkpoint = sam_checkpoint,
                    sam_model_type = sam_model_type, 
                    sam2_checkpoint = sam2_checkpoint,
                    sam2_model_cfg = sam2_model_cfg)
    
        seg_fused = sam_core.combine_2d_segmentations(
            sam_dir = sam_dir,
            output_folder=output_folder,
            output_filename=output_filename,
            label_mapping=None,
            priority_list=None ,
            ignore_classes= None,
            lower_threshold=None,
            upper_threshold=None
            
        )
    
    return seg_fused, log_dict

# main execution
if __name__ == "__main__":
    
    run_sam_yaml("./template/sam_predict.yaml")

    # To prevent running the code below, simply comment it out or remove it.
    # Alternatively, you can use sys.exit() to stop execution after run_sam_yaml if needed.

    import sys
    sys.exit()
    
    # image_3d_path = r"C:\Users\Yichen\OneDrive\work\codes\napari\output\skull_sam\Gorilla_gorilla_clean.tif"
    # seg_3d_path = r"C:\Users\Yichen\OneDrive\work\codes\napari\output\skull_sam\Gorilla_gorilla_seg_selected.tif"
    # n_points_per_class = 3
    # prompt_type = 'point'  # 'point' or 'bbox'
    # output_folder = r"C:\Users\Yichen\OneDrive\work\codes\napari\output\skull_sam\sam_fused_output"
    # output_filename = "gorilla_seg_selected_final.tif"


    img_path = r"C:\Users\Yichen\OneDrive\work\codes\napari\data\for_testing\2d_img_16bits.tif"
    seg_path = r"C:\Users\Yichen\OneDrive\work\codes\napari\data\for_testing\2d_seed.tif"
    # img_path = r"C:\Users\Yichen\OneDrive\work\codes\napari\data\for_testing\foram_img.tif"
    # seg_path = r"C:\Users\Yichen\OneDrive\work\codes\napari\data\for_testing\foram_seg.tif"
    
    img_path = r"C:\Users\Yichen\OneDrive\work\codes\napari\output\skull_sam\Gorilla_gorilla_clean.tif"
    seg_path = r"C:\Users\Yichen\OneDrive\work\codes\napari\output\skull_sam\Gorilla_gorilla_seg_selected.tif"
    
    n_points_per_class = 3
    prompt_type = 'point'  # 'point' or 'bbox'
    # output_folder = r"C:\Users\Yichen\OneDrive\work\codes\napari\output\skull_sam\sam_fused_output"
    # output_filename = "gorilla_seg_selected_final.tif"
    
    output_folder = r"C:\Users\Yichen\OneDrive\work\codes\napari\output\skull_sam_output\gorilla"
    output_filename = "2d.tif"
    
    sample_neg_each_class = True  # whether to sample negative points for each class
    negative_points = 1  # number of negative points per class, default is
    
    per_cls_mode = True  # whether to use per-class mode, default is True
    
    which_sam = 'sam1'  # 'sam1' or 'sam2'
    sam_checkpoint = "../segment-anything-main/checkpoints/sam_vit_h_4b8939.pth"
    sam_model_type = "vit_h"
    sam2_checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt"
    sam2_model_cfg = "../sam2/configs/sam2.1/sam2.1"
    
    


    args = parse_args()

    img_path = args.img_path
    seg_path = args.seg_path
    n_points_per_class = args.n_points_per_class
    prompt_type = args.prompt_type
    output_folder = args.output_folder
    output_filename = args.output_filename
    sample_neg_each_class = args.sample_neg_each_class
    negative_points = args.negative_points
    per_cls_mode = args.per_cls_mode
    which_sam = args.which_sam
    sam_checkpoint = args.sam_checkpoint
    sam_model_type = args.sam_model_type
    sam2_checkpoint = args.sam2_checkpoint
    sam2_model_cfg = args.sam2_model_cfg
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    all_sam_dirs = {}
    
    # if the image is 3D, we will extract slices along X, Y, Z axes
    img = tifffile.imread(img_path)
    seg = tifffile.imread(seg_path)
    if img.ndim == 3:
    
        for axis in ['X', 'Y', 'Z']:
            
            base_temp_dir = os.path.join(tempfile.gettempdir() ,"sprout", f"{timestamp}" ,axis)
            output_prompt_dir = os.path.join(base_temp_dir, "prompts")
            output_img_dir = os.path.join(base_temp_dir, "imgs")
            print(f"Using temporary output folders:\n  Prompts: {output_prompt_dir}\n  Images:  {output_img_dir}")

            

            outputs = sam_core.extract_slices_and_prompts(
                img = img,
                seg = seg,
                output_prompt_dir=output_prompt_dir,
                output_img_dir=output_img_dir,
                axis=axis,
                n_points_per_class=n_points_per_class,
                slice_range=None,
                prompt_type=prompt_type,  # 'point' or 'bbox'
                sample_neg_each_class=sample_neg_each_class,
                negative_points=negative_points
            )
            
            
            sam_dir = os.path.join(base_temp_dir, "sam")
            sam_core.init_and_run(outputs["images_dir"], 
                        which_sam = which_sam,
                        device=None,
                        prompt_dir=outputs["prompts_dir"], 
                        multimask_output=False,
                        save_dir=sam_dir, 
                        overlay_dir=None,
                        sam_checkpoint = sam_checkpoint,
                        sam_model_type = sam_model_type, 
                        sam2_checkpoint = sam2_checkpoint,
                        sam2_model_cfg = sam2_model_cfg)
            
            all_sam_dirs[axis] = sam_dir

        seg_fused = sam_core.combine_3d_segmentations(
            img = img,
            seg = seg,
            sam_dirs=all_sam_dirs,
            output_folder=output_folder,
            output_filename=output_filename,
            lower_threshold=None,
            upper_threshold=None,
            per_cls_mode=per_cls_mode
        )
    else:
        base_temp_dir = os.path.join(tempfile.gettempdir() ,"sprout", f"{timestamp}" )
        output_prompt_dir = os.path.join(base_temp_dir, "prompts")
        output_img_dir = os.path.join(base_temp_dir, "imgs")
        print(f"Using temporary output folders:\n  Prompts: {output_prompt_dir}\n  Images:  {output_img_dir}")

        
        outputs = sam_core.extract_slices_and_prompts(
            img = img,
            seg = seg,
            output_prompt_dir=output_prompt_dir,
            output_img_dir=output_img_dir,
            n_points_per_class=n_points_per_class,
            prompt_type=prompt_type,  # 'point' or 'bbox'
            per_slice_2d_input=True,
            sample_neg_each_class = sample_neg_each_class,
            negative_points=negative_points
        )
        
        
        sam_dir = os.path.join(base_temp_dir, "sam")
        sam_core.init_and_run(outputs["images_dir"], 
                    which_sam = which_sam,
                    device=None,
                    prompt_dir=outputs["prompts_dir"], 
                    multimask_output=False,
                    save_dir=sam_dir, 
                    overlay_dir=None,
                    sam_checkpoint = sam_checkpoint,
                    sam_model_type = sam_model_type, 
                    sam2_checkpoint = sam2_checkpoint,
                    sam2_model_cfg = sam2_model_cfg)
    
        seg_fused = sam_core.combine_2d_segmentations(
            sam_dir = sam_dir,
            output_folder=output_folder,
            output_filename=output_filename,
            label_mapping=None,
            priority_list=None ,
            ignore_classes= None,
            lower_threshold=None,
            upper_threshold=None
            
        )