from tifffile import imread
from PIL import Image
import numpy as np
import os, sys
import logging
from pathlib import Path
from monai.visualize.utils import blend_images
import torch
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_img_tiff(img_path):
    # import the tiff stack
    img = imread(img_path)

    logging.info(f"Image:{img_path}. Image shape: {img.shape} , image max and min:{np.max(img)} , {np.min(img)}")
    return img

# def read_imgs_as_stack(img_dir):
# # read through images
# # img_dir = Path('data/test_tiff_reader/seg')

#     img_files = os.listdir(img_dir)

#     img_list =[]
#     for img_file in img_files:
#         img_path = os.path.join(img_dir, img_file)
#         img = Image.open(img_path)
#         img_list.append(img)
#     img_stack = np.stack(img_list)
#     logging.info(f"Image in folder:{img_dir}. Image shape: {img_stack.shape} , image max and min:{np.max(img_stack)} , {np.min(img_stack)}")
    
#     return img_stack


def read_imgs_as_stack(img_dir):
    """
    Reads all image files from a specified directory into a single numpy array stack.
    
    Parameters:
        img_dir (str or Path): The directory containing image files.
    
    Returns:
        numpy.ndarray: A stack of images as a numpy array.
    """
    # List all files in the directory
    img_files = os.listdir(img_dir)
    
    # Filter out files to include only images (assumed formats: .jpg, .jpeg, .png)
    img_files = [file for file in img_files if file.lower().endswith(('.jpg', '.jpeg', '.png', 'tif', 'tiff'))]
    
    
    # Sort files alphabetically to maintain consistent order
    img_files.sort()
    logging.info(f"Reading {len(img_files)}, and first five are: {img_files[:5]}")
    # Prepare list to hold image data
    img_list = []
    
    # Read each image file
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        img = Image.open(img_path)
        img_array = np.array(img)
        img_list.append(img_array)
        logging.debug(f"Read image {img_file} with shape {img_array.shape}")

    # Stack images into a single numpy array
    img_stack = np.stack(img_list)
    
    # Log summary information about the stack
    logging.info(f"Images read from folder: {img_dir}")
    logging.info(f"Stack shape: {img_stack.shape}")
    logging.info(f"Image stack max and min values: {np.max(img_stack)}, {np.min(img_stack)}")
    
    return img_stack


def save_stack_as_pngs(image_stack, image_dir, prefix):
    """
    Reads an image stack from the specified directory and saves each slice as a PNG file.
    
    Args:
    image_dir (str): Path to the directory containing image stack files.
    prefix (str): Prefix for the output PNG filenames.
    """
    logging.info(f"Images to save from folder: {image_dir}")
    logging.info(f"Stack shape: {image_stack.shape}")
    logging.info(f"Image stack max and min values: {np.max(image_stack)}, {np.min(image_stack)}")
    
    # Ensure the target directory exists
    os.makedirs(image_dir, exist_ok=True)
    # Save each slice to a PNG file
    for i in range(image_stack.shape[0]):
        img = Image.fromarray(image_stack[i])
        img.save(os.path.join(image_dir, f"{prefix}_{i+1:04d}.png"))
        
def create_emtpy_stack_in_between(image_stack, x):
    # The number of slices
    D = image_stack.shape[0]
    
    logging.info(f"number of pixels equal to one before: {np.sum(image_stack==1)}")

    # Indices of slices that will be selected regularly
    selected_indices = np.arange(0, D, x)
    # print(selected_indices)

    # Handle the slices in between
    for i in range(len(selected_indices) - 1):
        start = selected_indices[i] + 1
        end = selected_indices[i + 1]
        # Reverse the slices between selected indices
        image_stack[start:end] =  0
        # print(image_stack[start:end].shape)
        # reversed_slices
        # new_order_stack.extend(reversed_slices)

    # Handle any remaining slices after the last selected index if not evenly divisible
    if selected_indices[-1] + 1 < D:
        remaining_start = selected_indices[-1] + 1
        remaining_end = D
        logging.info(f"Remaining slices from {remaining_start} to {remaining_end - 1}: {image_stack[remaining_start:remaining_end].shape}")
        image_stack[remaining_start:remaining_end] =  0
        
    logging.info(f"number of pixels equal to one after: {np.sum(image_stack==1)} by only using slices every {x}")    
    return image_stack


def save_new_img_stride(image_stack, step):
    """
    Subsamples an image array by selecting every 'step'-th slice along the first dimension.

    Args:
    image_stack (numpy.ndarray): Input image array with shape DHW or DHWC.
    step (int): Step size for subsampling, defaults is 10.

    Returns:
    numpy.ndarray: Subsampled image array.
    """
    
    logging.info(f"shape: {image_stack.shape}")
    if image_stack.ndim != 3:
        raise ValueError("Image must be a 3D array with shape (channels, height, width)")
    if step <= 0:
        raise ValueError("Step must be a positive integer")
    subsampled_image = image_stack[::step, ...]
    logging.info(f"shape after stride:{subsampled_image.shape}")    
    return subsampled_image


    


def blend_images_masks(image_stack, mask_stack):
    """
    Subsamples an image array by selecting every 'step'-th slice along the first dimension.

    Args:
    image_stack (numpy.ndarray): Input image array with shape DHW or DHWC.
    mask_stack (numpy.ndarray): Input image array with shape DHW or DHWC.

    Returns:
    numpy.ndarray: blend image array.
    """
    
    logging.info(f"shape: {image_stack.shape}")
    if image_stack.ndim < 3:
        raise ValueError("Image must be a 3D array with shape (channels, height, width)")
    if mask_stack.ndim < 3:
        raise ValueError("Image must be a 3D array with shape (channels, height, width)")
    # if step <= 0:
    #     raise ValueError("Step must be a positive integer")
    # subsampled_image = image_stack[::step, ...]
    # logging.info(f"shape after stride:{subsampled_image.shape}")    
    blend_img_list = []
    for image, mask in zip(image_stack, mask_stack):
        blend_img = blend_images(np.expand_dims(image,0), 
                        np.expand_dims(mask,0),
                            alpha=0.5, 
                            cmap='hsv', 
                            rescale_arrays=True, 
                            transparent_background=True)
        
        blend_img = (np.transpose(blend_img, (1,2, 0))*255).astype('uint8')
        blend_img_list.append(blend_img)
    
    blend_img_stack = np.stack(blend_img_list)
    
    logging.info(f"The image has shape of :{image_stack.shape}")
    logging.info(f"The mask has shape of :{mask_stack.shape}")
    logging.info(f"The blend image has shape of :{blend_img_stack.shape}")
    return blend_img_stack


def stack_to_monai_format(img_stack):
    # input is DHW
    #adding one as the channel, DCHW
    img_stack_t = np.expand_dims(img_stack,axis=1)
    # Covert to CHWD
    img_stack_t = np.transpose(img_stack_t,axes=(1,2,3,0))
    # add batch size: BCHWD
    img_stack_t = np.expand_dims(img_stack_t,axis=0)
    
    # Convert to tensor
    y_tensor = torch.tensor(img_stack_t)
    
    return y_tensor


# def save_stack_as_tiff(image_dir, output_filename):
#     """
#     Reads an image stack from the specified directory and saves the stack as a single TIFF file.
    
#     Args:
#     image_dir (str): Path to the directory containing image stack files.
#     output_filename (str): Filename for the output TIFF file.
#     """
#     # List all files in the directory
#     files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
#     files.sort()  # Sort files to maintain the order



#     # Read and stack images
#     img_list = [np.array(Image.open(os.path.join(image_dir, file))) for file in files]
#     image_stack = np.stack(img_list, axis=0)
    
#     # Save all slices in one TIFF file
#     output_path = os.path.join(image_dir, output_filename)
#     with Image.open(os.path.join(image_dir, files[0])) as img:
#         img.save(output_path, save_all=True, append_images=[Image.fromarray(img_list[i]) for i in range(1, len(img_list))])
