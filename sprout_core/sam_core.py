
import os
import json
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import re
import tifffile
from datetime import datetime
from skimage.measure import regionprops,label
import tempfile
from sklearn.cluster import KMeans
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

import sprout_core.config_core as config_core



try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("[ERROR] `sam2` is not installed. Please install it before running SAM2 inference.")
    build_sam2 = None
    SAM2ImagePredictor = None

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("[ERROR] `segment_anything` (SAM1) is not installed. Please install it before running SAM1 inference.")
    sam_model_registry = None
    SamPredictor = None


from collections import defaultdict


def group_points_by_name(points, labels, names):
    grouped = defaultdict(lambda: {'coords': [], 'labels': []})
    for pt, lb, name in zip(points, labels, names):
        if pt is None or name is None:
            continue
        grouped[name]['coords'].append(pt)
        grouped[name]['labels'].append(lb)
    return grouped

def group_boxes_by_name(boxes, names):
    grouped = defaultdict(list)
    for box, name in zip(boxes, names):
        if box is not None and name is not None:
            grouped[name].append(box)
    return grouped


def group_prompts_by_name(points, labels, boxes, names):
    grouped = defaultdict(lambda: {"points": [], "labels": [], "boxes": []})
    for pt, lb, box, name in zip(points, labels, boxes, names):
        group = grouped[name]
        if pt is not None:
            group["points"].append(pt)
            group["labels"].append(lb)
        if box is not None:
            group["boxes"].append(box)
    return grouped


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    return image


def load_prompt(prompt_path):
    with open(prompt_path, 'r') as f:
        data = json.load(f)

    points, labels, names, boxes = [], [], [], []

    for item in data:
        if isinstance(item, dict):
            if 'point' in item:
                points.append(item['point'])
                labels.append(item.get('label', 1))
                names.append(item.get('name', None))
                boxes.append(None)
            elif 'box' in item:
                boxes.append(item['box'])  # [x0, y0, x1, y1]
                points.append(None)
                labels.append(None)
                names.append(item.get('name', None))
        else:
            raise ValueError("Unsupported prompt format.")

    return points, labels, names, boxes


def save_mask(mask, save_dir, image_name, prompt_name=None):
    
    if save_dir is None:
        return  # No save directory specified, skip saving mask
    
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.splitext(image_name)[0]
    suffix = f"_{prompt_name}" if prompt_name else ""
    out_path = os.path.join(save_dir, f"{base}{suffix}.png")
    cv2.imwrite(out_path, mask.astype(np.uint8) * 255)


def save_overlay(image, mask, save_dir, image_name, prompt_name=None,
                 points=None, labels=None, boxes=None):
    
    if save_dir is None:
        return  # No overlay directory specified, skip saving overlay
    
    os.makedirs(save_dir, exist_ok=True)
    # overlay = image.copy()
    
    overlay = image.copy()
    overlay[mask > 0] = (overlay[mask > 0] * 0.7 + np.array([0, 0, 255]) * 0.3).astype(np.uint8)



    # Adaptive thickness/size based on image resolution
    h, w = image.shape[:2]
    scale = max(h, w) / 512  # 512 is a reference size, adjust as needed
    thickness = max(1, int(2 * scale))
    radius = max(3, int(2 * scale))

    # Draw points
    if points is not None and labels is not None:
        for pt, lb in zip(points, labels):
            color = (0, 255, 0) if lb == 1 else (255, 0, 0)  # green or red
            cv2.circle(overlay, tuple(map(int, pt)), radius, color, thickness=-1)

    # Draw boxes
    if boxes is not None:
        for box in boxes:
            x0, y0, x1, y1 = map(int, box)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color=(255, 255, 0), thickness=thickness)

    base = os.path.splitext(image_name)[0]
    suffix = f"_{prompt_name}" if prompt_name else ""
    out_path = os.path.join(save_dir, f"{base}{suffix}_overlay.png")
    # OpenCV uses BGR, but overlay is RGB (from PIL/np). Convert before saving.
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, overlay_bgr)


# def validate_prompts(pr)

def batch_prompts_predictor(predictor , all_points,all_labels ,boxes, multimask_output,  image, device):
    if all_points is not None:
        coords_torch = torch.as_tensor(np.array(all_points), dtype=torch.float, device=device)
        transform_coords = predictor.transform.apply_coords_torch(
            coords_torch, image.shape[:2])
        labels_torch = torch.as_tensor(np.array(all_labels), dtype=torch.float, device=device)
    else:
        transform_coords = None
        labels_torch = None
    
    if boxes is not None:
        input_boxes = torch.as_tensor(np.array(boxes), dtype=torch.float, device=device)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    else:
        transformed_boxes = None
    masks, _, _ = predictor.predict_torch(
        point_coords=transform_coords,
        point_labels=labels_torch,
        boxes=transformed_boxes,
        multimask_output=multimask_output,
    )    

    return masks.detach().cpu().numpy()


def run_sam_infer(
    image_input,
    predictor,
    which_sam,
    device,
    multimask_output=True,
    prompt_dir=None,
    mask_input=None,
    save_dir="outputs",
    overlay_dir="overlays"
    
):
    if os.path.isdir(image_input):
        image_list = [f for f in os.listdir(image_input) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        image_list.sort()
        image_paths = [os.path.join(image_input, f) for f in image_list]
    else:
        image_paths = [image_input]
        image_list = [os.path.basename(image_input)]

    for img_path, img_name in tqdm(zip(image_paths, image_list), total=len(image_paths)):
        image = load_image(img_path)
        

        # --- prepare optional mask input ---
        mask_path = None
        mask_lowres = None
        if mask_input:
            if os.path.isdir(mask_input):
                base_name = os.path.splitext(img_name)[0]
                mask_path = os.path.join(mask_input, f"{base_name}.npy")
            else:
                mask_path = mask_input

            if os.path.exists(mask_path):
                mask_lowres = np.load(mask_path)
                assert isinstance(mask_lowres, np.ndarray), f"Mask at {mask_path} is not a valid array"
                assert mask_lowres.shape == (1, 256, 256), \
                    f"Mask at {mask_path} must be of shape (1, 256, 256), got {mask_lowres.shape}"


        # --- default fallback if no prompt ---
        prompt_file = os.path.join(prompt_dir, os.path.splitext(img_name)[0] + ".json") if prompt_dir else None
        if not prompt_file or not os.path.exists(prompt_file):
            continue  # Skip if no prompt file found

        # --- load prompts ---
        points, labels, names, boxes = load_prompt(prompt_file)

        # --- group by name, merge all prompts ---
        grouped_prompts = group_prompts_by_name(points, labels, boxes, names)
        
        # Validate: for each name in grouped_prompts, there should be at most one bbox or none
        for name, content in grouped_prompts.items():
            box_list = content["boxes"] if content["boxes"] else []
            if len(box_list) > 1:
                print(f"[ERROR] {img_name} | prompt-{name}: More than one bounding box found for this name. Only one box per name is allowed.")
                continue  # Skip this group if invalid
            
        # --- validate prompt batch mode ---
        prompt_batch_mode = True
        if len(set([len(content["points"]) for content in grouped_prompts.values()])) != 1:
            prompt_batch_mode = False
        
        if not grouped_prompts:
            continue
        else:
            predictor.set_image(image)
        
        
        if prompt_batch_mode:
            # --- process grouped prompts ---
            all_points = []
            all_labels = []
            all_boxes = []
            for name, content in grouped_prompts.items():
                
                pts = np.array(content["points"]) if content["points"] else None
                lbs = np.array(content["labels"]) if content["labels"] else None
                box_list = content["boxes"] if content["boxes"] else None

                if pts is None:
                    all_points = None
                    all_labels = None
                else:         
                    all_points.append(pts.tolist() if pts is not None else [])
                    all_labels.append(lbs.tolist() if lbs is not None else [])
                
                if box_list is None:
                    all_boxes = None
                else:
                    all_boxes.append(box_list if box_list is not None else [])
            
            
                
            try:
                if which_sam == "sam1":
                    masks = batch_prompts_predictor(predictor, all_points , all_labels,
                                                    all_boxes,
                                                    multimask_output, image, device)
                else:
                    masks, scores, _ = predictor.predict(
                        point_coords= np.array(all_points) if all_points else None,
                        point_labels= np.array(all_labels) if all_labels else None,
                        box =np.array(all_boxes) if all_boxes else None,
                        multimask_output=multimask_output,
                        mask_input=mask_lowres
                    )
                    
                    
                part_names = list(grouped_prompts.keys())
                # masks is a list of masks, each corresponding to a prompt            
                for i, mask_with_multi in enumerate(masks):
                    for j, mask in enumerate(mask_with_multi):
                        if multimask_output:
                            mask_name = f"{part_names[i]}_m{j}"
                        else:
                            mask_name = part_names[i]
                    
                        save_mask(mask, save_dir, img_name, mask_name)
                        save_overlay(image, mask, overlay_dir, img_name, mask_name,
                                    points=all_points[i] if all_points else None, 
                                    labels=all_labels[i] if all_labels else None, 
                                    boxes= all_boxes[i] if all_boxes else None)
            except Exception as e:
                print(f"[ERROR] {img_name} | prompt-{name}: {e}")
        else:
            # --- process grouped prompts one by one ---
            for name, content in grouped_prompts.items():
                
                
                pts = np.array(content["points"]) if content["points"] else None
                lbs = np.array(content["labels"]) if content["labels"] else None
                box_list = content["boxes"] if content["boxes"] else None

                try:
                    masks, scores, _ = predictor.predict(
                        point_coords=pts if pts is not None else None,
                        point_labels=lbs if lbs is not None else None,
                        box=np.array(box_list) if box_list else None,  
                        multimask_output=multimask_output,
                        mask_input=mask_lowres
                    )
                    for j, mask in enumerate(masks):
                        if multimask_output:
                            mask_name = f"{name}_m{j}"
                        else:
                            mask_name = name
                        
                        save_mask(mask, save_dir, img_name, mask_name)
                        save_overlay(image, mask, overlay_dir, img_name, mask_name,
                                    points=pts, labels=lbs, boxes=box_list)
                except Exception as e:
                    print(f"[ERROR] {img_name} | prompt-{name}: {e}")



def init_and_run(image_input, 
                  which_sam,
                  device=None,
                  prompt_dir=None, 
                  multimask_output=True,
                  save_dir=None, 
                  overlay_dir=None,
                  sam_checkpoint = "../segment-anything-main/checkpoints/sam_vit_h_4b8939.pth",
                  sam_model_type = "vit_h", 
                  sam2_checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt",
                  sam2_model_cfg = "../sam2/configs/sam2.1/sam2.1",
                  custom_checkpoint = None
                  ):

    # Assign device if not provided, with error handling
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("[WARNING] CUDA is not available. Using CPU.")
            device = torch.device("cpu")
    else:
        if isinstance(device, str):
            device = torch.device(device)
        elif not isinstance(device, torch.device):
            raise ValueError("Device must be a string or torch.device object.")
    # which_sam should be passed as argument, do not override here
    # ---- model paths ----

    if which_sam == "sam1" and SamPredictor is None:
        raise ImportError("SAM1 is not available. Please install it.")
    if which_sam == "sam2" and SAM2ImagePredictor is None:
        raise ImportError("SAM2 is not available. Please install it.")
    
    if which_sam == "sam1":
        try:
            # using sam1
            # sam_checkpoint = "../segment-anything-main/checkpoints/sam_vit_h_4b8939.pth"
            # model_type = "vit_h"
            sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            predictor = SamPredictor(sam)
        except Exception as e:
            print(f"[ERROR] Failed to load SAM1 model: {e}")
            exit(1)
    elif which_sam == "sam2":
        try:
            # using sam2
            # sam2_checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt"
            # model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
            sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=device)
            predictor = SAM2ImagePredictor(sam2_model)
        except Exception as e:
            print(f"[ERROR] Failed to load SAM2 model: {e}")
            exit(1)
    if custom_checkpoint is not None:
        try:
            print(f"Loading custom checkpoint from {custom_checkpoint}")
            predictor.model.load_state_dict(torch.load(custom_checkpoint, map_location=device))
        except Exception as e:
            print(f"[ERROR] Failed to load custom checkpoint: {e}")
            exit(1)


    run_sam_infer(image_input, predictor, 
                  which_sam=which_sam,
                  device=device,
                    prompt_dir=prompt_dir, 
                   multimask_output=multimask_output,
                   save_dir=save_dir, 
                   overlay_dir=overlay_dir)



def combine_3d_segmentations(
    img=None,
    seg=None,
    img_path=None,
    seg_path=None,
    sam_dirs=None,
    output_folder="./sam_fused_output",
    output_filename="seg_final.tif",
    lower_threshold=None,
    upper_threshold=None,
    label_mapping=None,
    save_final_path=True,
    per_cls_mode=False,
    merge_class_order=None
):
    """
    Combine SAM segmentation results from X/Y/Z axis by majority voting.

    Args:
        img: 3D grayscale numpy array (optional if img_path is provided)
        seg: 3D segmentation ground truth (optional, just used for label mapping)
        img_path: path to 3D grayscale image (used if img is None)
        seg_path: path to seg (used if seg is None)
        sam_dirs: dict with keys {'X', 'Y', 'Z'} and path to SAM output dirs
        output_folder: where to save final combined segmentation
        lower_threshold: int or None
        upper_threshold: int or None
        label_mapping: dict mapping class names (e.g., 'class_1') to int labels
        save_final_path: whether to write final .tif to output_folder

    Returns:
        seg_final: 3D numpy array with fused segmentation
    """
    

    img = config_core.check_and_load_data(img, img_path, "img")
    seg = config_core.check_and_load_data(seg, seg_path, "seg")



    os.makedirs(output_folder, exist_ok=True)
    
    outputs = {}
    pattern = re.compile(r"^(.+?)_(\d+)_(\w+)\.png$")

    if label_mapping is None:
        assert seg is not None, "Either `label_mapping` or `seg` must be provided."
        unique_classes = np.unique(seg)
        unique_classes = unique_classes[unique_classes != 0]
        label_mapping = {f"class_{i}": i for i in unique_classes}
    if merge_class_order is not None:
        merge_order = merge_class_order
    else:
        merge_order = list(label_mapping.keys())

    #make sure sam_dirs key only contains 'X', 'Y', 'Z'
    if sam_dirs is None or not isinstance(sam_dirs, dict) or not all(k in ['X', 'Y', 'Z'] for k in sam_dirs.keys()):
        raise ValueError("sam_dirs must be a dict with keys 'X', 'Y', 'Z' and their corresponding paths.")

    for slice_axis, sam_dir in sam_dirs.items():
        if slice_axis == 'Z':
            output_shape = img.shape
        elif slice_axis == 'Y':
            output_shape = (img.shape[1], img.shape[0], img.shape[2])
        elif slice_axis == 'X':
            output_shape = (img.shape[2], img.shape[0], img.shape[1])
        else:
            raise ValueError(f"Unknown slice_axis: {slice_axis}")
        print(f"Processing SAM outputs for axis: {slice_axis} with shape {output_shape}")
        
        file_list = os.listdir(sam_dir)
        
        if per_cls_mode:
            outputs[slice_axis] = {}
            ## Per class output Mode ##
            for cls_name in label_mapping:
                outputs[slice_axis][cls_name] = np.zeros(output_shape, dtype=np.uint8)
            for fname in tqdm(file_list):
                match = pattern.match(fname)
                if not match:
                    continue
                _, slice_idx, cls = match.group(1), int(match.group(2)), match.group(3)
                if cls not in label_mapping:
                    print(f"[WARNING] Class '{cls}' not in label_mapping. Skipping.")
                    continue
                path = os.path.join(sam_dir, fname)
                mask = np.array(Image.open(path)) > 127
                outputs[slice_axis][cls][slice_idx][mask] = 1
            # Rearrange to match (Z, H, W)
            if slice_axis == 'X':
                for cls in outputs[slice_axis]:
                    outputs[slice_axis][cls] = np.moveaxis(outputs[slice_axis][cls], 0, 2)
            elif slice_axis == 'Y':
                for cls in outputs[slice_axis]:
                    outputs[slice_axis][cls] = np.moveaxis(outputs[slice_axis][cls], 0, 1)
          
        else:
            class_to_files = defaultdict(list)
            
            for fname in tqdm(file_list):
                match = pattern.match(fname)
                if not match:
                    continue
                _, slice_idx, cls = match.group(1), int(match.group(2)), match.group(3)
                if cls in label_mapping:
                    class_to_files[cls].append(fname)
                
            output_axis = np.zeros(output_shape, dtype=np.uint8)
            
            for cls in merge_order:
                if cls not in class_to_files:
                    continue
                label_id = label_mapping[cls]
                for fname in class_to_files[cls]:
                    match = pattern.match(fname)
                    if not match:
                        continue
                    _, slice_idx, _ = match.group(1), int(match.group(2)), match.group(3)
                    path = os.path.join(sam_dir, fname)
                    mask = np.array(Image.open(path)) > 127
                    output_axis[slice_idx][(mask) & (output_axis[slice_idx] == 0)] = label_id

            #     if cls not in label_mapping:
            #     print(f"[WARNING] Class '{cls}' not in label_mapping. Skipping.")
            #     continue
            # path = os.path.join(sam_dir, fname)
            # mask = np.array(Image.open(path)) > 127
            # output_axis[slice_idx][mask] = label_mapping[cls]

            # Rearrange to match (Z, H, W)
            if slice_axis == 'X':
                output_axis = np.moveaxis(output_axis, 0, 2)
            elif slice_axis == 'Y':
                output_axis = np.moveaxis(output_axis, 0, 1)
            outputs[slice_axis] = output_axis



    
    if per_cls_mode:
        final_masks = {}
            
        for cls_name in label_mapping:
            mask_x = outputs['X'][cls_name]
            mask_y = outputs['Y'][cls_name]
            mask_z = outputs['Z'][cls_name]
            vote_stack = np.stack([mask_x, mask_y, mask_z], axis=0)  # shape: (3, Z, H, W)
            vote_sum = np.sum(vote_stack, axis=0)  # shape: (Z, H, W)
            final_mask = (vote_sum >= 2).astype(np.uint8)
            final_masks[cls_name] = final_mask

            # Save per-class final masks
            output_path = os.path.join(output_folder, f"final_{cls_name}.tif")
            tifffile.imwrite(output_path, final_mask, compression='zlib')

            # Clean outputs['X'][cls_name], outputs['Y'][cls_name], outputs['Z'][cls_name]
            outputs['X'][cls_name] = None
            outputs['Y'][cls_name] = None
            outputs['Z'][cls_name] = None

  

        final_mask = np.zeros_like(next(iter(final_masks.values())), dtype=np.uint8)

        for cls_name in merge_order:
            if cls_name not in final_masks:
                print(f"[WARNING] Class {cls_name} not found in final_masks, skipping.")
                continue
            mask = final_masks[cls_name]
            label_id = label_mapping[cls_name]
            final_mask[(mask > 0) & (final_mask == 0)] = label_id
            
        if save_final_path:
            out_path = os.path.join(output_folder, output_filename)
            tifffile.imwrite(out_path, final_mask, compression='zlib')
            print(f"✅ Saved merged final_mask to {out_path}")

    else:
        seg_x, seg_y, seg_z = outputs['X'], outputs['Y'], outputs['Z']
        votes = np.stack([seg_x, seg_y, seg_z], axis=0)  # (3, Z, H, W)

        # Create grayscale threshold mask
        mask_valid = np.ones_like(img, dtype=bool)
        if lower_threshold is not None:
            mask_valid &= (img >= lower_threshold)
        if upper_threshold is not None:
            mask_valid &= (img <= upper_threshold)

        # Majority voting
        final_mask = np.zeros_like(seg_x, dtype=np.uint8)
        for i in range(seg_x.shape[0]):
            for j in range(seg_x.shape[1]):
                for k in range(seg_x.shape[2]):
                    if not mask_valid[i, j, k]:
                        continue
                    voxel = votes[:, i, j, k]
                    counts = np.bincount(voxel)
                    if len(counts) > 1:
                        final_mask[i, j, k] = np.argmax(counts)

        if save_final_path:
            if not (output_filename.endswith('.tif') or output_filename.endswith('.tiff')):
                raise ValueError("output_filename must end with '.tif' or '.tiff'")
            os.makedirs(output_folder, exist_ok=True)
            final_path = os.path.join(output_folder, output_filename)
            tifffile.imwrite(final_path, final_mask, compression='zlib')
            print(f"✅ Saved majority-voted segmentation to: {final_path}")


    return final_mask



def save_as_8bit_png(img_slice, output_path):
    """
    Save an image slice (any bit depth) as an 8-bit grayscale PNG for visualization.
    
    Parameters:
    - img_slice: 2D numpy array, can be uint8, uint16, float32, etc.
    - output_path: str, path to save the PNG file.
    """
    if not isinstance(img_slice, np.ndarray):
        raise TypeError("Input must be a NumPy array")

    # Normalize if not already 8-bit
    if img_slice.dtype != np.uint8:
        # Rescale to 0–255 linearly
        img_min = img_slice.min()
        img_max = img_slice.max()
        if img_max == img_min:
            img_8bit = np.zeros_like(img_slice, dtype=np.uint8)
        else:
            img_8bit = ((img_slice - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        img_8bit = img_slice

    # Save as 8-bit PNG
    Image.fromarray(img_8bit).save(output_path)


def sample_points(mask, n=3, method='kmeans'):
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) <= n:
        return coords

    if method == 'kmeans':
        kmeans = KMeans(n_clusters=n, random_state=0).fit(coords)
        return np.round(kmeans.cluster_centers_).astype(int)

    elif method == 'center_edge':
        dist = distance_transform_edt(mask)
        center = np.unravel_index(np.argmax(dist), dist.shape)
        edge_coords = np.column_stack(np.where((dist > 0) & (dist < 3)))
        if len(edge_coords) >= (n - 1):
            edge_samples = edge_coords[np.random.choice(len(edge_coords), n - 1, replace=False)]
        else:
            edge_samples = edge_coords
        return np.vstack(([center], edge_samples))

    elif method == 'skeleton':
        from skimage.morphology import skeletonize
        skel = skeletonize(mask)
        coords = np.column_stack(np.where(skel))
        return coords[np.random.choice(len(coords), min(n, len(coords)), replace=False)]

    elif method == 'random':
        return coords[np.random.choice(len(coords), n, replace=False)]
    else:
        raise ValueError(f"Unknown sampling method: {method}. Supported methods: 'kmeans', 'center_edge', 'skeleton', 'random'.")



def extract_slices_and_prompts(
    
    img_path = None,
    seg_path = None,
    img = None,
    seg = None,
    output_prompt_dir=None,
    output_img_dir=None,
    axis='X',
    n_points_per_class=3,
    slice_range=None,
    prompt_type='point', # 'point' or 'bbox'
    per_slice_2d_input = False,
    sample_neg_each_class=False,
    negative_points=None,
    sample_method='random'  # 'kmeans', 'center_edge', 'skeleton', 'random'
):
    assert prompt_type in ['point', 'bbox'], "prompt_type must be 'point' or 'bbox'"

    img = config_core.check_and_load_data(img, img_path, "img")
    seg = config_core.check_and_load_data(seg, seg_path, "seg")
    config_core.valid_input_data(img, seg = seg)

    if output_prompt_dir is None or output_img_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        base_temp_dir = os.path.join(tempfile.gettempdir(), axis, f"{timestamp}")
        output_prompt_dir = os.path.join(base_temp_dir, "prompts")
        output_img_dir = os.path.join(base_temp_dir, "imgs")
        print(f"Using temporary output folders:\n  Prompts: {output_prompt_dir}\n  Images:  {output_img_dir}")

    os.makedirs(output_prompt_dir, exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)


    print(f"Loaded image shape: {img.shape}")
    print(f"Loaded seg shape: {seg.shape}")

    if negative_points is None:
        negative_points = n_points_per_class

    # Determine slicing behavior
    if per_slice_2d_input:
        if img.ndim == 2:
            img = img[np.newaxis, ...]
            seg = seg[np.newaxis, ...]
        elif img.ndim != 3:
            raise ValueError("When using per_slice_2d_input=True, input must be 2D or stack of 2D slices (3D).")
        num_slices = img.shape[0]
        print(f"Input treated as 2D slices directly, total slices: {num_slices}")

    else:
        axes = {'Z': 0, 'Y': 1, 'X': 2}
        ax = axes[axis.upper()]
        img = np.moveaxis(img, ax, 0)
        seg = np.moveaxis(seg, ax, 0)
        num_slices = img.shape[0]
        print(f"Slicing along axis: {axis} (axis index {ax}), total slices: {num_slices}")


    for idx in tqdm(range(num_slices)):
        if slice_range and not (slice_range[0] <= idx < slice_range[1]):
            continue

        # print(f"\nProcessing slice {idx+1}/{num_slices}")
        img_slice = img[idx]
        seg_slice = seg[idx]
        prompts = []

        class_ids = np.unique(seg_slice)
        class_ids = class_ids[class_ids != 0]
        # print(f"  Found class IDs (excluding background): {class_ids}")

        for cls in class_ids:
            # print(f"    Processing class: {cls}")
            mask = (seg_slice == cls).astype(np.uint8)

            if prompt_type == 'point':
                coords = np.column_stack(np.where(mask > 0))
                # print(f"      Number of positive pixels: {len(coords)}")

                # if len(coords) == 0:
                #     print(f"      No pixels found for class {cls}, skipping.")
                #     continue

                # Sample n positive points
                sampled = sample_points(mask, n=n_points_per_class, method=sample_method)
                for pt in sampled:
                    prompts.append({"point": pt[::-1].tolist(), "label": 1, "name": f"class_{cls}"})
                    

                if sample_neg_each_class:
                    # Sample negative points separately for each *other* class
                    for neg_cls in class_ids:
                        if neg_cls == cls:
                            continue
                        neg_mask = (seg_slice == neg_cls)
                        if np.any(neg_mask):
                            sampled_neg = sample_points(neg_mask, n=negative_points, method=sample_method)
                            for pt in sampled_neg:
                                prompts.append({"point": pt[::-1].tolist(), "label": 0, "name": f"class_{cls}"})
                        else:
                            print(f"      No negative points found for class {cls} vs {neg_cls}.")
                        
                        ### deprecated code for sampling negative points,
                        # neg_coords = np.column_stack(np.where(neg_mask))
                        # if len(neg_coords) > 0:
                        #     sampled_neg = neg_coords[np.random.choice(len(neg_coords), min(negative_points, len(neg_coords)), replace=False)]
                        #     for pt in sampled_neg:
                        #         prompts.append({"point": pt[::-1].tolist(), "label": 0, "name": f"class_{cls}"})
                        # else:
                        #     print(f"      No negative points found for class {cls} vs {neg_cls}.")
                else:
                    # Sample from all other non-zero, non-cls regions
                    # sample n negative points
                    other_mask = (seg_slice != cls) & (seg_slice > 0)
                    other_coords = np.column_stack(np.where(other_mask > 0))
                    if np.any(other_mask):
                        sampled_neg = sample_points(other_mask, n=negative_points, method=sample_method)
                        for pt in sampled_neg:
                            prompts.append({"point": pt[::-1].tolist(), "label": 0, "name": f"class_{cls}"})
                    else:
                        print(f"      No negative points found for class {cls}.")
                        
                    # if len(other_coords) > 0:
                    #     sampled_neg = other_coords[np.random.choice(len(other_coords), min(negative_points, len(other_coords)), replace=False)]
                    #     for pt in sampled_neg:
                    #         prompts.append({"point": pt[::-1].tolist(), "label": 0, "name": f"class_{cls}"})
                    # else:
                    #     print(f"      No negative points found for class {cls}.")

            elif prompt_type == 'bbox':
                labeled = label(mask)
                props = regionprops(labeled)
                if props:
                    for i, region in enumerate(props):
                        minr, minc, maxr, maxc = region.bbox
                        # print(f"      Bounding box {i+1}: [minc={minc}, minr={minr}, maxc={maxc}, maxr={maxr}]")
                        prompts.append({
                            "box": [minc, minr, maxc, maxr],
                            "name": f"class_{cls}"
                        })
                else:
                    print(f"      No regionprops found for class {cls}.")

        with open(os.path.join(output_prompt_dir, f"name_{idx}.json"), "w") as f:
            json.dump(prompts, f, separators=(",", ":"))
        save_as_8bit_png(img_slice, os.path.join(output_img_dir, f"name_{idx}.png"))

    print("\n✅ All slices processed and saved.")
    return {
        "prompts_dir": output_prompt_dir,
        "images_dir": output_img_dir}


def combine_2d_segmentations(

    sam_dir,
    output_folder,
    output_filename,
    label_mapping=None,
    priority_list=None ,
    ignore_classes= None,
    lower_threshold=None,
    upper_threshold=None
):
    """
    Combine SAM segmentation results from X/Y/Z axis by majority voting.

    Args:
        img: 3D grayscale numpy array (optional if img_path is provided)
        seg: 3D segmentation ground truth (optional, just used for label mapping)
        img_path: path to 3D grayscale image (used if img is None)
        seg_path: path to seg (used if seg is None)
        sam_dirs: dict with keys {'X', 'Y', 'Z'} and path to SAM output dirs
        output_folder: where to save final combined segmentation
        lower_threshold: int or None
        upper_threshold: int or None
        label_mapping: dict mapping class names (e.g., 'class_1') to int labels
        save_final_path: whether to write final .tif to output_folder

    Returns:
        seg_final: 3D numpy array with fused segmentation
    """
    

    if label_mapping is None:
        to_make_label_mapping = True
        label_mapping = {}
    else:
        to_make_label_mapping = False

    pattern = re.compile(r"^(.+)_([^_]+)\.png$")

  
        
    file_list = os.listdir(sam_dir)

    results = {}
    next_label = 1
    for fname in tqdm(file_list):
        match = pattern.match(fname)
        if not match:
            continue
        file, cls = match.group(1), match.group(2)
        # if cls not in label_mapping:
        #     print(f"[WARNING] Class '{cls}' not in label_mapping. Skipping.")
        #     continue
        
        if ignore_classes and cls in ignore_classes:
            continue
        
        if to_make_label_mapping and cls not in label_mapping:
                label_mapping[cls] = next_label
                next_label += 1
        if results.get(file) is None:
            results[file] = [(fname, cls)]
        else:
            results[file].append((fname, cls))

    # make sure every element in priority_list is in label_mapping
    if priority_list is not None:
        for cls in priority_list:
            if cls not in label_mapping:
                print(f"[WARNING] Class '{cls}' in priority_list not found in label_mapping. Adding it.")
    

        
    for file, infer_list in results.items():
        if priority_list:
            # prio_set = set(priority_list)
            # Sort infer_list based on priority_list
            # If cls is not in priority_list, it will be sorted to the end
            infer_list.sort(
                key=lambda x: (
                    len(priority_list) - priority_list.index(x[1]) if x[1] in priority_list else 0
                    
                )
            )


        path = os.path.join(sam_dir, infer_list[0][0])
        mask = np.array(Image.open(path))
        output_mask = np.zeros(mask.shape, dtype=np.uint8)
        for fname, cls in infer_list:
            if ignore_classes and cls in ignore_classes:
                continue
            
            path = os.path.join(sam_dir, fname)
            mask = np.array(Image.open(path)) > 127
            
            output_mask[mask] = label_mapping[cls]    

        output_path = os.path.join(output_folder,output_filename)
        if not os.path.exists(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray(output_mask).save(output_path, format='png')
   

## Deprecated function, use extract_slices_and_prompts instead
def extract_slices_and_prompts_2d(
    
    img_path = None,
    seg_path = None,
    img = None,
    seg = None,
    output_prompt_dir=None,
    output_img_dir=None,
    n_points_per_class=3,
    prompt_type='point', # 'point' or 'bbox'
):
    assert prompt_type in ['point', 'bbox'], "prompt_type must be 'point' or 'bbox'"

    img = config_core.check_and_load_data(img, img_path, "img")
    seg = config_core.check_and_load_data(seg, seg_path, "seg")
    config_core.valid_input_data(img, seg = seg)
    
    if output_prompt_dir is None or output_img_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        base_temp_dir = os.path.join(tempfile.gettempdir(), f"{timestamp}")
        output_prompt_dir = os.path.join(base_temp_dir, "prompts")
        output_img_dir = os.path.join(base_temp_dir, "imgs")
        print(f"Using temporary output folders:\n  Prompts: {output_prompt_dir}\n  Images:  {output_img_dir}")

    os.makedirs(output_prompt_dir, exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)


    print(f"Loaded img shape: {img.shape}")
    print(f"Loaded seg shape: {seg.shape}")


    prompts = []

    class_ids = np.unique(seg)
    class_ids = class_ids[class_ids != 0]
    # print(f"  Found class IDs (excluding background): {class_ids}")

    for cls in class_ids:
        # print(f"    Processing class: {cls}")
        mask = (seg == cls).astype(np.uint8)

        if prompt_type == 'point':
            coords = np.column_stack(np.where(mask > 0))
            # print(f"      Number of positive pixels: {len(coords)}")

            # if len(coords) == 0:
            #     print(f"      No pixels found for class {cls}, skipping.")
            #     continue
            
            # Sample n positive points   
            sampled = coords[np.random.choice(len(coords), min(n_points_per_class, len(coords)), replace=False)]
            for pt in sampled:
                prompts.append({"point": pt[::-1].tolist(), "label": 1, "name": f"class_{cls}"})

            # sample n negative points
            other_mask = (seg != cls) & (seg > 0)
            other_coords = np.column_stack(np.where(other_mask > 0))
            if len(other_coords) > 0:
                sampled_neg = other_coords[np.random.choice(len(other_coords), min(n_points_per_class, len(other_coords)), replace=False)]
                for pt in sampled_neg:
                    prompts.append({"point": pt[::-1].tolist(), "label": 0, "name": f"class_{cls}"})
            else:
                print(f"      No negative points found for class {cls}.")

        elif prompt_type == 'bbox':
            labeled = label(mask)
            props = regionprops(labeled)
            if props:
                for i, region in enumerate(props):
                    minr, minc, maxr, maxc = region.bbox
                    # print(f"      Bounding box {i+1}: [minc={minc}, minr={minr}, maxc={maxc}, maxr={maxr}]")
                    prompts.append({
                        "box": [minc, minr, maxc, maxr],
                        "name": f"class_{cls}"
                    })
            else:
                print(f"      No regionprops found for class {cls}.")

    with open(os.path.join(output_prompt_dir, f"name.json"), "w") as f:
        json.dump(prompts, f, separators=(",", ":"))
    save_as_8bit_png(img, os.path.join(output_img_dir, f"name.png"))

    print("\n✅ All slices processed and saved.")
    return {
        "prompts_dir": output_prompt_dir,
        "images_dir": output_img_dir}
    

if __name__ == "__main__":
    
    sam_dirs = {
        'X': r'C:\Users\Yichen\OneDrive\work\codes\napari\output\skull_sam_output\Lemur_outputs_sam1_x',
        'Y': r'C:\Users\Yichen\OneDrive\work\codes\napari\output\skull_sam_output\Lemur_outputs_sam1_y',
        'Z': r'C:\Users\Yichen\OneDrive\work\codes\napari\output\skull_sam_output\Lemur_outputs_sam1_z'
    }
    
    
    # sam_dirs = {
    #     'X': r'C:\Users\Yichen\OneDrive\work\codes\napari\output\foram_sam\outputs_sam1_x',
    #     'Y': r'C:\Users\Yichen\OneDrive\work\codes\napari\output\foram_sam\outputs_sam1_y',
    #     'Z': r'C:\Users\Yichen\OneDrive\work\codes\napari\output\foram_sam\outputs_sam1_z'
    # }

    # img_path =r'C:\Users\Yichen\OneDrive\work\codes\napari\data\for_testing\foram_img.tif'
    # seg_path = r'C:\Users\Yichen\OneDrive\work\codes\napari\data\for_testing\foram_seg.tif'
    # output_folder=r"C:\Users\Yichen\OneDrive\work\codes\napari\output\foram_sam\per_cls_mode"

    img_path =r'C:\Users\Yichen\OneDrive\work\codes\napari\output\skull_sam\Lemur_catta_resam_clean_input.tif'
    seg_path = r'C:\Users\Yichen\OneDrive\work\codes\napari\output\skull_sam\thre_7350_None_ero_0.tif'
    output_folder=r"C:\Users\Yichen\OneDrive\work\codes\napari\output\skull_sam_output\per_cls_mode"
    
    combine_3d_segmentations(
    img=None,
    seg=None,
    img_path=img_path,
    seg_path=seg_path,
    sam_dirs=sam_dirs,
    output_folder=output_folder,
    output_filename="seg_final.tif",
    lower_threshold=None,
    upper_threshold=None,
    label_mapping=None,
    save_final_path=True,
    per_cls_mode=False
    )