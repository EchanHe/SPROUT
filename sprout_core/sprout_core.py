import numpy as np
from skimage.morphology import ball, disk
from scipy import ndimage as ndi
from skimage import measure
from datetime import datetime
import os
import threading
import tifffile

from scipy.ndimage import binary_fill_holes

from sprout_core import config_core
# One Stop exception for all of SPROUT (also raised by boundary, prompt_points, prompt_seg)
from sprout_core.boundary import Cancelled, _is_cancelled


def check_cancel(cancel):
    """Raise Cancelled if `cancel` (a threading.Event, a callable returning bool, or None) is set."""
    if _is_cancelled(cancel):
        raise Cancelled()


class StepProgress:
    """Counts done steps for progress(done, total, text); step() may be called from any thread.

    progress=None: step() does nothing (CLI / scripts / YAML runs).
    """

    def __init__(self, progress, total):
        self.progress = progress
        self.total = max(int(total), 1)
        self.done = 0
        self._lock = threading.Lock()

    def step(self, text="", n=1):
        if self.progress is None:
            return
        with self._lock:                 # reported in order (progress should be quick)
            self.done = min(self.done + n, self.total)
            self.progress(self.done, self.total, text)

    def jump(self, done, text=""):
        """Set the done count (e.g. a stage that stopped early counts as done)."""
        if self.progress is None:
            return
        with self._lock:
            self.done = min(max(int(done), self.done), self.total)
            self.progress(self.done, self.total, text)


def assign_optional_params(config, keys_with_defaults):
    """
    Assign configuration values from YAML or DataFrame, falling back to default values if missing.

    Args:
        config (dict): Parsed YAML configuration as a dictionary.
        keys_with_defaults (dict): Dictionary of keys and their default values.

    Returns:
        dict: A dictionary of configuration values with assigned defaults if missing.
    """
    config_values = {}
    
    for key, default in keys_with_defaults.items():
        # Check if the key exists in YAML
        if config and key in config:
            config_values[key] = config[key]
        else:
            config_values[key] = default
    
    return config_values


def run_threads(target, args_list):
    """Run target(*args) in one thread per entry of `args_list` and wait for all of them.

    An exception in a thread is not lost (threading only prints it): once every thread
    has finished, the first exception raised is re-raised here, with its traceback.
    """
    errors = []

    def run(*args):
        try:
            target(*args)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=run, args=tuple(args)) for args in args_list]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]






ball_fp_YZ = np.array(
                [[[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]],

                [[0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0]],

                [[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]]], dtype=np.uint8)

ball_fp_XZ = np.array(
                [[[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]],

                [[0, 0, 0],
                    [1, 1, 1],
                    [0, 0, 0]],

                [[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]]], dtype=np.uint8)

ball_fp_XY = np.array(
                [[[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],

                [[0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]],

                [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]]], dtype=np.uint8)

fp_Z = np.array(
                [[[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]],

                [[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]],

                [[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]]], dtype=np.uint8)

fp_X = np.array(
                [[[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],

                [[0, 0, 0],
                    [1, 1, 1],
                    [0, 0, 0]],

                [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]]], dtype=np.uint8)

fp_Y = np.array(
                [[[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],

                [[0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0]],

                [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]]], dtype=np.uint8)

fp_X_2d = np.array([[0, 0, 0],
                    [1, 1, 1],
                    [0, 0, 0]],  dtype=np.uint8)

fp_Y_2d = np.array([[0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0]],  dtype=np.uint8)

def _foreground(image):
    """The voxels that `image == True` selects: a bool image as it is, else the voxels equal to 1."""
    return image if image.dtype == bool else (image == True)


def _foreground_min_max(fg):
    """Per axis (min, max) index of the True voxels of bool `fg`, None if there are none.

    Same values (np.intp) as np.argwhere(fg).min / max(axis=0), from any() projections:
    no array of all coordinates (3 x 8 bytes per voxel) is made.
    """
    lo, hi = [], []
    for _ in range(fg.ndim):
        proj = fg.any(axis=tuple(range(1, fg.ndim))) if fg.ndim > 1 else fg
        idx = np.flatnonzero(proj)
        if idx.size == 0:
            return None
        lo.append(idx[0])
        hi.append(idx[-1])
        if fg.ndim > 1:
            fg = fg[idx[0]:idx[-1] + 1].any(axis=0)
    return lo, hi


def get_sub_binary_image_by_pos(image_3d, margin):


    # Find the bounding box of the voxels where the image is 1
    min_max = _foreground_min_max(_foreground(image_3d))
    if min_max is None:
        # what np.argwhere(...).min() raised for an empty image
        raise ValueError("zero-size array to reduction operation minimum which has no identity")


    is_3d = (image_3d.ndim == 3)
    if is_3d:
        # Get the min and max indices along each dimension
        (min_z, min_y, min_x), (max_z, max_y, max_x) = min_max
    else:
        # Get the min and max indices along each dimension
        (min_y, min_x), (max_y, max_x) = min_max
    # print(f"Bounding box for 3D image: x_min={min_x}, x_max={max_x}, y_min={min_y}, y_max={max_y}, z_min={min_z}, z_max={max_z}")

    if is_3d:
        # Adjust the bounding box with the margin
        min_z = max(min_z - margin, 0)
        max_z = min(max_z + margin, image_3d.shape[0] - 1)
        min_y = max(min_y - margin, 0)
        max_y = min(max_y + margin, image_3d.shape[1] - 1)
        min_x = max(min_x - margin, 0)
        max_x = min(max_x + margin, image_3d.shape[2] - 1)
        subset_3d = image_3d[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
        max_min_ids = [min_z, max_z, min_y ,max_y, min_x, max_x]
    else:
        min_y = max(min_y - margin, 0)
        max_y = min(max_y + margin, image_3d.shape[0] - 1)
        min_x = max(min_x - margin, 0)
        max_x = min(max_x + margin, image_3d.shape[1] - 1)
        subset_3d = image_3d[min_y:max_y+1, min_x:max_x+1]
        max_min_ids = [ min_y ,max_y, min_x, max_x]
    

    # print(f"original image shape:{image_3d.shape}, subset shape:{subset_3d.shape}")
    
    
    return subset_3d, max_min_ids


# ---------------------------------------------------------------------------
# Footprint size convention used throughout SPROUT
#
#   kernal_size / size is a RADIUS for every footprint shape:
#     ball(r) / disk(r)              -> diameter 2r + 1
#     cube_fp(r) / square_fp(r)      -> edge     2r + 1   (r = 1 -> 3x3x3 / 3x3)
#
# skimage's cube(n) / square(n) take the EDGE length instead, so cube(1) is a
# single voxel and eroding with it does nothing. Always build cubes/squares
# through cube_fp / square_fp, never with cube(kernal_size) directly.
# The Boundary tab (sprout_core/boundary.py) uses the same radius convention,
# which also matches Avizo's "Cube, neighbourhood 26, size s".
# ---------------------------------------------------------------------------
def cube_fp(radius):
    """3D cube footprint of the given radius: edge 2*radius + 1."""
    return np.ones((2 * int(radius) + 1,) * 3, np.uint8)


def square_fp(radius):
    """2D square footprint of the given radius: edge 2*radius + 1."""
    return np.ones((2 * int(radius) + 1,) * 2, np.uint8)


# skimage's binary_erosion / binary_dilation / binary_closing are deprecated (removed in 0.28).
# They call these scipy functions, so the results are the same.
def binary_erosion(image, footprint):
    """Binary erosion; voxels outside the image count as foreground (as in skimage)."""
    return ndi.binary_erosion(image, structure=footprint, border_value=True)


def binary_dilation(image, footprint):
    """Binary dilation (as in skimage)."""
    return ndi.binary_dilation(image, structure=footprint)


def binary_closing(image, footprint):
    """Binary closing: dilation, then erosion (as in skimage)."""
    return binary_erosion(binary_dilation(image, footprint), footprint)


def dilation_binary_img_on_sub(input, margin, kernal_size, is_round=True):
    """Binary dilation on the bounding box of the foreground.

    kernal_size is a radius: ball/disk(kernal_size) if is_round, else a cube/square
    with edge 2*kernal_size + 1.
    """
    if margin <=kernal_size:
        margin = kernal_size + 1
    
    is_3d = (input.ndim == 3)
    if is_3d:
        
        if not _foreground(input).any():
            return input
        else:
            subset_3d , max_min_ids= get_sub_binary_image_by_pos(input, margin = margin)
            min_z, max_z, min_y ,max_y, min_x, max_x = max_min_ids

            if is_round:
                subset_3d = binary_dilation(subset_3d, ball(kernal_size))
            else:
                subset_3d = binary_dilation(subset_3d, cube_fp(kernal_size))


            input[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1] = subset_3d
    else:
        if not _foreground(input).any():
            return input
        subset_2d , max_min_ids= get_sub_binary_image_by_pos(input, margin = margin)
        min_y ,max_y, min_x, max_x = max_min_ids

        if is_round:
            subset_2d = binary_dilation(subset_2d, disk(kernal_size))
        else:
            subset_2d = binary_dilation(subset_2d, square_fp(kernal_size))
        input[min_y:max_y+1, min_x:max_x+1] = subset_2d
    
    return input

FOOTPRINTS_3D = ('ball', 'cube', 'ball_XY', 'ball_XZ', 'ball_YZ', 'X', 'Y', 'Z', '2XZ_1Y', '2XY_1Z', '2YZ_1X')
FOOTPRINTS_2D = ('ball', 'cube', 'disk', 'square', 'X', 'Y')


def erosion_binary_img_on_sub(input, kernal_size = 1, footprint='ball'):
    """Binary erosion on the bounding box of the foreground.

    kernal_size is a radius for 'ball'/'disk' and 'cube'/'square' alike: with the
    default kernal_size=1, one call erodes by one voxel, using a 3x3x3 cross
    ('ball', 6-neighbourhood) or a full 3x3x3 cube ('cube', 26-neighbourhood).
    The other footprints ('ball_XY', 'X', ...) are fixed one-voxel footprints.
    """
    if footprint == 'ball' or footprint == 'cube':
        assert type(kernal_size) is int
    
    is_3d = (input.ndim == 3)
    footprints = FOOTPRINTS_3D if is_3d else FOOTPRINTS_2D
    if footprint not in footprints:
        raise ValueError(f"Footprint '{footprint}' does not work on a {input.ndim}D image, use one of {footprints}.")
    if is_3d:
        if not _foreground(input).any():
            return input
        else:
            subset_3d , max_min_ids= get_sub_binary_image_by_pos(input, margin = 1)
            min_z, max_z, min_y ,max_y, min_x, max_x = max_min_ids

            if footprint == 'ball':
                subset_3d = binary_erosion(subset_3d, ball(kernal_size))
            elif footprint == 'cube':
                subset_3d = binary_erosion(subset_3d, cube_fp(kernal_size))
            elif footprint =='ball_XY':
                subset_3d = binary_erosion(subset_3d, ball_fp_XY)
            elif footprint =='ball_XZ':
                subset_3d = binary_erosion(subset_3d, ball_fp_XZ)
            elif footprint =='ball_YZ':
                subset_3d = binary_erosion(subset_3d, ball_fp_YZ)
            elif footprint =='X':
                subset_3d = binary_erosion(subset_3d, fp_X)
            elif footprint =='Y':
                subset_3d = binary_erosion(subset_3d, fp_Y)
            elif footprint =='Z':
                subset_3d = binary_erosion(subset_3d, fp_Z)
            elif footprint =='2XZ_1Y':
                subset_3d = binary_erosion(subset_3d, ball_fp_XZ)
                subset_3d = binary_erosion(subset_3d, ball_fp_XZ)
                subset_3d = binary_erosion(subset_3d, fp_Y)
            elif footprint =='2XY_1Z':
                subset_3d = binary_erosion(subset_3d, ball_fp_XY)
                subset_3d = binary_erosion(subset_3d, ball_fp_XY)
                subset_3d = binary_erosion(subset_3d, fp_Z)
            elif footprint =='2YZ_1X':
                subset_3d = binary_erosion(subset_3d, ball_fp_YZ)
                subset_3d = binary_erosion(subset_3d, ball_fp_YZ)
                subset_3d = binary_erosion(subset_3d, fp_X)

            
            input[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1] = subset_3d
    else:
        if not _foreground(input).any():
            return input
        subset_2d , max_min_ids= get_sub_binary_image_by_pos(input, margin = 1)
        min_y ,max_y, min_x, max_x = max_min_ids
        
        if footprint == 'disk' or footprint == 'ball':
            subset_2d = binary_erosion(subset_2d, disk(kernal_size))
        elif footprint == 'square' or footprint == 'cube':
            subset_2d = binary_erosion(subset_2d, square_fp(kernal_size))
        elif footprint =='X':
            subset_2d = binary_erosion(subset_2d, fp_X_2d)
        elif footprint =='Y':
            subset_2d = binary_erosion(subset_2d, fp_Y_2d)
            
        input[min_y:max_y+1, min_x:max_x+1] = subset_2d
        # input = binary_erosion(input, square(kernal_size))
    
    return input


def closing_binary_img_on_sub(input, margin, kernal_size, is_round=True):
    """Binary closing on the bounding box; kernal_size is a radius (see cube_fp)."""
    if margin <=kernal_size:
        margin = kernal_size + 1
        
    is_3d = (input.ndim == 3)
    if is_3d:
        if not _foreground(input).any():
            return input
        else:
        
            subset_3d , max_min_ids= get_sub_binary_image_by_pos(input, margin = margin)
            min_z, max_z, min_y ,max_y, min_x, max_x = max_min_ids

            if is_round:
                subset_3d = binary_closing(subset_3d, ball(kernal_size))
            else:
                subset_3d = binary_closing(subset_3d, cube_fp(kernal_size))

            # subset_3d = binary_closing(subset_3d, cube(kernal_size))
            input[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1] = subset_3d
    else:
        if not _foreground(input).any():
            return input
        subset_2d , max_min_ids= get_sub_binary_image_by_pos(input, margin = margin)
        min_y ,max_y, min_x, max_x = max_min_ids
        if is_round:
            subset_2d = binary_closing(subset_2d, disk(kernal_size))
        else:
            subset_2d = binary_closing(subset_2d, square_fp(kernal_size))
        input[min_y:max_y+1, min_x:max_x+1] = subset_2d
    
    return input

def closing_binary_img_on_sub_one_step_iter(input, iterations, footprint='ball'):
    
    is_3d = (input.ndim == 3)
    
    if footprint =='ball':
        kernal_size=1
        is_round = True
    elif footprint =='cube':
        kernal_size=1          # radius 1 -> 3x3x3 cube (was edge 3 before the radius convention)
        is_round = False

    if is_3d:
        for _ in range(iterations):
            input = dilation_binary_img_on_sub(input, margin=2,kernal_size=kernal_size,is_round=is_round)
            
        for _ in range(iterations):    
            input = erosion_binary_img_on_sub(input,kernal_size=kernal_size, footprint=footprint)
    else:
        kernal_size=1
        for _ in range(iterations):
            input = dilation_binary_img_on_sub(input, margin=2,kernal_size=kernal_size,is_round=is_round)
            
        for _ in range(iterations):    
            input = erosion_binary_img_on_sub(input,kernal_size=kernal_size, footprint='disk')        
        
    return input
        

#### CCOMPS stuffs


def view_ccomps(input):
    subset_3d , _= get_sub_binary_image_by_pos(input, margin = 1)
    labeled_image = measure.label(subset_3d, background=0, return_num=False)
    # Calculate the size of each connected component
    component_sizes = np.bincount(labeled_image.ravel())[1:]  # Skip the background count

    # Print the size of each component
    for i, size in enumerate(component_sizes, start=1):
        print(f"Component {i}: Size = {size}")

def keep_largest_ccomps(input):
    
    subset_3d , max_min_ids= get_sub_binary_image_by_pos(input, margin = 1)
    min_z, max_z, min_y ,max_y, min_x, max_x = max_min_ids
    
    labeled_image = measure.label(subset_3d, background=0, return_num=False)
    # Calculate the size of each connected component
    # component_sizes = np.bincount(labeled_image.ravel())[1:]  # Skip the background count

    # Calculate the size of each connected component
    component_sizes = np.bincount(labeled_image.ravel())[1:]  # Skip the background count

    # Identify the largest component
    largest_component_label = np.argmax(component_sizes) + 1  # Labels are 1-based

    # Create a new binary image with only the largest component
    largest_component = (labeled_image == largest_component_label).astype(int)

    output = np.zeros(input.shape)

    output[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1] = largest_component

    return output

def keep_ccomps(input, top_n=None , threshold=0):
    
    subset_3d , max_min_ids= get_sub_binary_image_by_pos(input, margin = 1)
    min_z, max_z, min_y ,max_y, min_x, max_x = max_min_ids
    
    labeled_image = measure.label(subset_3d, background=0, return_num=False)
    # Calculate the size of each connected component
    # component_sizes = np.bincount(labeled_image.ravel())[1:]  # Skip the background count

    # Calculate the size of each connected component
    component_sizes = np.bincount(labeled_image.ravel())[1:]  # Skip the background count

    if threshold==0:
        component_labels = np.unique(labeled_image)[1:]
    else:  
        # Get component labels that are above the specified size
        component_labels = np.where(component_sizes >= threshold)[0] + 1  # Labels are 1-based

    if top_n is not None:
        largest_labels = component_labels[np.argsort(component_sizes[component_labels - 1])[-top_n:]]
    else:
        largest_labels = component_labels

    # Create an output image with only the desired components retained
    output_sub = np.zeros_like(subset_3d)
    for label in largest_labels:
        output_sub[labeled_image == label] = 1
        
    output = np.zeros_like(input)
    output[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1] = output_sub
    
    return output


_CHUNK = 1 << 22    # voxels per chunk in _bincount / _apply_lut (bounds their intp temporaries)


def _bincount(labels, length):
    """np.bincount(labels.ravel(), minlength=length) in chunks, without a full-size intp copy.

    labels: non-negative integers below `length`, of a dtype that casts safely to intp.
    Mostly-zero chunks count only their non-zero voxels (seeds are often a small part of a scan).
    """
    flat = labels.reshape(-1)
    step = max(_CHUNK, 4 * length)
    counts = np.zeros(length, np.intp)
    for s in range(0, flat.size, step):
        chunk = flat[s:s + step]
        nonzero = chunk != 0
        n = int(np.count_nonzero(nonzero))
        if 2 * n > chunk.size:
            counts += np.bincount(chunk, minlength=length)
        else:
            counts += np.bincount(chunk[nonzero], minlength=length)
            counts[0] += chunk.size - n
    return counts


def _apply_lut(lut, labels, out=None):
    """lut[labels] in chunks (fancy indexing copies all of `labels` to intp first).

    out: a C-contiguous array of labels' shape and lut's dtype; may be `labels` itself.
    When lut[0] is 0, mostly-zero chunks only look up their non-zero voxels.
    """
    if out is None:
        out = np.empty(labels.shape, lut.dtype)
    src, dst = labels.reshape(-1), out.reshape(-1)
    sparse = lut.size > 0 and lut[0] == 0
    in_place = out is labels
    for s in range(0, src.size, _CHUNK):
        chunk = src[s:s + _CHUNK]
        if sparse:
            index = np.flatnonzero(chunk)
            if 2 * index.size <= chunk.size:
                values = lut[chunk[index]]
                if not in_place:
                    dst[s:s + _CHUNK] = 0
                dst[s + index] = values
                continue
        dst[s:s + _CHUNK] = lut[chunk]
    return out


def unique_labels(labels):
    """np.unique(labels) of an integer array, without sorting all of it when most voxels are 0."""
    flat = labels.reshape(-1)
    nonzero = flat != 0
    values = np.unique(flat[nonzero])
    if not nonzero.all():
        values = np.insert(values, np.searchsorted(values, 0), 0)
    return values


def get_ccomps_with_size_order(volume, segments=None, min_vol = None):
    """Get the largest ccomp

    Args:
        label (_type_): _description_
        segments (_type_): _description_

    Returns:
        _type_: _description_
    """
    labeled_image, n_labels = measure.label(volume, background=0,
                                  return_num=True)
    component_sizes = _bincount(labeled_image, n_labels + 1)[1:]


    # measure.label numbers the components 1..n_labels, so no np.unique is needed
    component_labels = np.arange(1, n_labels + 1)
    if min_vol is not None:
        valid_components = component_sizes >= min_vol
        component_labels = component_labels[valid_components]
        component_sizes = component_sizes[valid_components]
        
    component_sizes_sorted = -np.sort(-component_sizes,)
    if segments is None:
        segments = len(component_sizes_sorted)
    elif segments> len(component_sizes_sorted):
        segments = len(component_sizes_sorted)
    # props = measure.regionprops(label_image)
    largest_labels = component_labels[np.argsort(component_sizes)[::-1][:segments]]

    # Relabel in place with one lookup table (was one full scan per component)
    lut = np.zeros(n_labels + 1, labeled_image.dtype)
    lut[largest_labels] = np.arange(1, len(largest_labels) + 1)
    output = _apply_lut(lut, labeled_image, out=labeled_image)

    return output, component_sizes_sorted[:segments]


## For junctions finding

def find_gaps_between_two(input_1, input_2, background = None,
                          max_width = 7,
                          close_size = 7):
    assert input_1.ndim == input_2.ndim, "input 1 and 2 should have the same dimension"
    is_3d = (input_1.ndim == 3)
    ### Goal: Close holes
    ## Method: Apply close to Input 1 and 2
    # close_size = 9
    # margin = ((close_size-1)//2) +1
    
    # input_1 = closing_binary_img_on_sub(input_1,margin=margin, kernal_size=close_size)
    # input_2 = closing_binary_img_on_sub(input_2,margin=margin, kernal_size=close_size)
    
    # close_iters = 4
    
    # input_1 = closing_binary_img_on_sub_one_step_iter(input_1,close_iters)
    # input_2 = closing_binary_img_on_sub_one_step_iter(input_2,close_iters)
    
    combined = np.logical_or(input_1,input_2)   
    ### Goal: find the link between Input 1 and 2


    ## Method1: Apply dilation to Input 1 and 2, 
    
    if is_3d:
        ## and find the intersection between two inputs
        dilation_iter = max_width
        dilation_size = 1

        for i in range(dilation_iter):
            # prev_suture_inter_sum = suture_inter_sum
            input_1 = dilation_binary_img_on_sub(input_1, margin=2, kernal_size=dilation_size)
            input_2 = dilation_binary_img_on_sub(input_2, margin=2, kernal_size=dilation_size)
            
            if i == 0:
                inter_1_dilation = np.logical_and(input_1, input_2)
    else:
        input_1 = binary_dilation(input_1, disk(max_width))
        input_2 = binary_dilation(input_2, disk(max_width))
    
    inter_dilation = np.logical_and(input_1, input_2)
    if np.sum(inter_dilation)==0:
        # print("No intersection have been found")
        return None
    
    ## Method2: Apply close to the segmentation of both input1 and 2.
    ## In order to find if it is possible to close any gap between 1 and 2
    # close_size = 11   
    # margin = ((close_size-1)//2) +1
    # input_for_closing = combined.copy()
    # combined_for_closing = closing_binary_img_on_sub(input_for_closing,margin=margin, kernal_size=close_size)

    input_for_closing = combined.copy()
    if is_3d:
        close_iters = close_size
        combined_for_closing = closing_binary_img_on_sub_one_step_iter(input_for_closing,close_iters)
    else:
        combined_for_closing = binary_closing(input_for_closing,disk(close_size))

    inter_dilation_and_closing = np.logical_and(inter_dilation,
                                                combined_for_closing)
    
    ### Goal: Make the final result more smooth and grown
    ## Dilation on the intersection a little
    # if np.sum(inter_dilation_and_closing)==0:
    #     print("No intersection have been found")
    #     return None
    
    # dilation_iter = 1
    # dilation_size=1
    # for i in range(dilation_iter):
    #     # prev_suture_inter_sum = suture_inter_sum
    #     inter_dilation_and_closing = dilation_binary_img_on_sub(inter_dilation_and_closing, 
    #                                                             margin=2, kernal_size=dilation_size)
    

    ### Goal: Refine the intersection
    ### (1) Use Results from Find intersection 1 and 2
    ### (2) Remove result areas that are parts of input1 and input2
    inter_dilation_and_closing[combined==True]=False
    
    # Set anything belong to the background (not the classes that form the sutures) to False
    if background is not None:
        inter_dilation_and_closing[background==True]=False
        
    if np.sum(inter_dilation_and_closing)==0:
        print("No intersection have been found")
        return None

    ### Make sutures better
    # If the intersection is very thin (bones are too close), use dilation to increase the suture sizes a little.
    
    ### Clean potential noises
    # Using ccomps, either by numbers or the size of comps
    # Also another way is to detect the neighbour pixel, if they are all other 
    
    
    return inter_dilation_and_closing



def find_gaps_between_two_avizo_version(input_1, input_2, background = None,
                          max_width = 7,
                          close_size = 7):
       
    
    ### Goal: Close holes
    ## Method: Apply close to Input 1 and 2
    # close_size = 9
    # margin = ((close_size-1)//2) +1
    
    # input_1 = closing_binary_img_on_sub(input_1,margin=margin, kernal_size=close_size)
    # input_2 = closing_binary_img_on_sub(input_2,margin=margin, kernal_size=close_size)
    
    # close_iters = 4
    
    # input_1 = closing_binary_img_on_sub_one_step_iter(input_1,close_iters)
    # input_2 = closing_binary_img_on_sub_one_step_iter(input_2,close_iters)
    
    combined = np.logical_or(input_1,input_2)   
    ### Goal: find the link between Input 1 and 2


    ## Method1: Apply dilation to Input 1 and 2, 
    ## and find the intersection between two inputs
    dilation_iter = max_width
    dilation_size = 1

    for i in range(dilation_iter):
        # prev_suture_inter_sum = suture_inter_sum
        input_1 = dilation_binary_img_on_sub(input_1, margin=2, kernal_size=dilation_size)
        input_2 = dilation_binary_img_on_sub(input_2, margin=2, kernal_size=dilation_size)
        
        if i == 0:
            inter_1_dilation = np.logical_and(input_1, input_2)
    
    inter_dilation = np.logical_and(input_1, input_2)
    if np.sum(inter_dilation)==0:
        # print("No intersection have been found")
        return None
    
    ## Method2: Apply close to the segmentation of both input1 and 2.
    ## In order to find if it is possible to close any gap between 1 and 2
    # close_size = 11   
    # margin = ((close_size-1)//2) +1
    # input_for_closing = combined.copy()
    # combined_for_closing = closing_binary_img_on_sub(input_for_closing,margin=margin, kernal_size=close_size)

    input_for_closing = combined.copy()
    close_iters = close_size
    combined_for_closing = closing_binary_img_on_sub_one_step_iter(input_for_closing,close_iters)

    inter_dilation_and_closing = np.logical_and(inter_dilation,
                                                combined_for_closing)
    
    ### Goal: Make the final result more smooth and grown
    ## Dilation on the intersection a little
    # if np.sum(inter_dilation_and_closing)==0:
    #     print("No intersection have been found")
    #     return None
    
    # dilation_iter = 1
    # dilation_size=1
    # for i in range(dilation_iter):
    #     # prev_suture_inter_sum = suture_inter_sum
    #     inter_dilation_and_closing = dilation_binary_img_on_sub(inter_dilation_and_closing, 
    #                                                             margin=2, kernal_size=dilation_size)
    
    inter_dilation_and_closing = np.logical_or(inter_dilation_and_closing, inter_1_dilation)
    ### Goal: Refine the intersection
    ### (1) Use Results from Find intersection 1 and 2
    ### (2) Remove result areas that are parts of input1 and input2
    inter_dilation_and_closing[combined==True]=False
    
    # Set anything belong to the background (not the classes that form the sutures) to False
    if background is not None:
        inter_dilation_and_closing[background==True]=False
        
    if np.sum(inter_dilation_and_closing)==0:
        print("No intersection have been found")
        return None

    ### Make sutures better
    # If the intersection is very thin (bones are too close), use dilation to increase the suture sizes a little.
    
    ### Clean potential noises
    # Using ccomps, either by numbers or the size of comps
    # Also another way is to detect the neighbour pixel, if they are all other 
    
    
    return inter_dilation_and_closing

def binary_to_colour_stack(input):
    """Turn binary to colour 3D stack [D, H, W, 3]

    Args:
        input (_type_): Have values from 1 to 6
        background, ROI 1, ROI 2, gap, other ROI, compare
    """
    
    colors = np.array([
        [0, 0, 0], # black
        ((100)*3),    # Mid grey
        (50, 50, 50),    # dark grey
        (255, 0, 0),    # red
        (211, 211, 211), # Light Gray
        (255, 182, 193)  # light red
    ])
    shape_rgb = input.shape + (3,)
    output_rgb = np.zeros(shape_rgb, dtype=np.uint8)
    for i in range(6):
        output_rgb[input == i] = colors[i]     
        
    return output_rgb
    


def erosion_suture(suture, bone, bg):
    """Erode suture area/volume if elements touch background
    Don't erode if it touches bone

    Args:
        suture (_type_): _description_
        bone (_type_): _description_
        bg (_type_): 
    """
    
    return


######### Seeds in sprout #####

def find_seed_by_ero(volume_array, threshold , segments, ero_iter, 
                    output_dir, 
                    ero_shape = 'ball',
                    SAVE_ALL_ERO= True):
     # Capture the start time
    start_time = datetime.now()
    
    volume_label = volume_array > threshold
    
    log_dict = {"Method": "find_seed_by_ero",
                  "volume_array shape": list(volume_array.shape),
                  "threshold": threshold,
                  "segments": segments,
                  "ero_shape": ero_shape,
                  "Whole Volume": str(np.sum(volume_label))
                  }
    
    for i_iter in range(0,ero_iter+1):
        
        if i_iter!=0:
            volume_label = erosion_binary_img_on_sub(volume_label, kernal_size = 1)
        
        seed_file = os.path.join(output_dir , 
                            f"seed_ero_{i_iter}_thre{threshold}_segs_{segments}.tif")
        
        seed, ccomp_sizes = get_ccomps_with_size_order(volume_label,segments)
        tifffile.imwrite(seed_file, seed,
                         compression ='zlib')
        
        args_dict = {
            # "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            # "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            # "duration": str(duration),
            f'size_top_{segments}_ccomp': ccomp_sizes.tolist(),
            f'number_of_ccopms_found': len(ccomp_sizes.tolist()),
            'output_seed_dir': output_dir
        }
        
        log_dict[f"ero_iter{i_iter}"]=args_dict
        # .append(args_dict)
    
        
    # seed, ccomp_sizes = get_ccomps_with_size_order(volume_label,segments)
    
    # Capture the end time
    end_time = datetime.now()
    # Calculate the duration
    duration = end_time - start_time   
    
    log_dict["start_time"] =   start_time.strftime("%Y-%m-%d %H:%M:%S") 
    log_dict["end_time"] =   end_time.strftime("%Y-%m-%d %H:%M:%S")
    log_dict["duration"] =   str(duration)
     
        
    return log_dict


def find_seed_by_ero_custom(volume_array, threshold , segments, ero_iter, 
                    output_dir, 
                    footprints = None,
                    upper_threshold = None,
                    boundary = None ,
                    is_return_seeds = False,
                    cancel = None,
                    step = None):
    """cancel: threading.Event (or None), checked before every erosion step; raises Cancelled.
    step: callable(text) called after every seed written (one progress step), or None."""
     # Capture the start time
    start_time = datetime.now()

    seeds_dict = {}
    
    # Thresholding 
    if upper_threshold is None:
        volume_label = volume_array >= threshold
    else:
        volume_label = (volume_array>=threshold) & (volume_array<=upper_threshold)
    
    # Use boundary if provided.
    if boundary is not None:
        boundary = config_core.check_and_cast_boundary(boundary)
        volume_label[boundary] = False
    
    # Logging
    log_dict = {"Method": "find_seed_by_ero_custom",
                  "volume_array shape": list(volume_array.shape),
                  "threshold": threshold,
                  "upper threshold": upper_threshold,
                  "segments": segments,
                  "footprints": footprints,
                  "Whole Volume": int(np.sum(volume_label)),
                  "seeds": []
                  }
    

    for i_iter in range(0,ero_iter+1):
        check_cancel(cancel)
        if i_iter!=0:
            volume_label = erosion_binary_img_on_sub(volume_label,
                                                                   footprint = footprints[i_iter-1])
                                                                #    footprint='ball_YZ')
            check_cancel(cancel)

        # Connected components
        seed, ccomp_sizes = get_ccomps_with_size_order(volume_label,segments)
        
        # Save seed from each threshold, ero combination
        seed_name = f"seed_ero_{i_iter}_thre{threshold}_{upper_threshold}_segs_{segments}.tif"
        seed_file = os.path.join(output_dir, 
                    seed_name)

        # at least uint16 as before; wider only if there are more than 65535 seeds
        tifffile.imwrite(seed_file, cast_labels(seed, np.uint16),
                         compression ='zlib')
        
        if is_return_seeds:
            # the same dtype as the file (at least uint16), not measure.label's int32
            seeds_dict[os.path.basename(output_dir) + "_" +seed_name] = cast_labels(seed, np.uint16)
        
        args_dict = {
            # "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            # "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            # "duration": str(duration),
            'ero_iter' : i_iter,
            'size_ccomp': ccomp_sizes.tolist(),
            'number_of_ccomps': len(ccomp_sizes.tolist()),
            'output_seed_dir': output_dir,
            'output_path': os.path.abspath(seed_file)
        }
        
        log_dict["seeds"].append(args_dict)
        if step is not None:
            step(f"threshold {threshold}: erosion {i_iter}/{ero_iter}")

        # log_dict[f"ero_iter{i_iter}"]=args_dict
        # .append(args_dict)
    
        
    # seed, ccomp_sizes = get_ccomps_with_size_order(volume_label,segments)
    
    # Capture the end time
    end_time = datetime.now()
    # Calculate the duration
    duration = end_time - start_time   
    
    log_dict["start_time"] =   start_time.strftime("%Y-%m-%d %H:%M:%S") 
    log_dict["end_time"] =   end_time.strftime("%Y-%m-%d %H:%M:%S")
    log_dict["duration"] =   str(duration)
     
        
    return seeds_dict , log_dict


def ero_one_iter(input_mask, 
                      segments, footprint = None):
    volume_label = erosion_binary_img_on_sub(input_mask, 
                                            footprint = footprint)
    seed, ccomp_sizes = get_ccomps_with_size_order(volume_label,segments)
    
    return seed, ccomp_sizes
### For grow 

def dilation_one_iter(input_mask, threshold_binary, 
                             touch_rule = 'stop',
                             segments=None, ero_shape = 'ball',
                             to_grow_ids = None):
    if to_grow_ids is None:
        label_id_list = np.unique(input_mask)
        label_id_list = label_id_list[label_id_list!=0]
    else:
        label_id_list = to_grow_ids
    
    result = input_mask.copy()  
    for label_id in label_id_list:
        dilated_binary_label_id = (result ==label_id)
        dilated_binary_label_id = dilation_binary_img_on_sub(dilated_binary_label_id, 
                                                                margin = 2, kernal_size = 1)
        
        
        if touch_rule == 'stop':
            # This is the binary for non-label of the updated mask
            binary_non_label = (result !=label_id) & (result != 0)
            # See if original mask overlay with grown label_id mask
            overlay = np.logical_and(binary_non_label, dilated_binary_label_id)
                        
            # # Check if there are any True values in the resulting array
            # HAS_OVERLAY = np.any(overlay)
            
            # Quicker way to do intersection check
            inter = np.sum(binary_non_label[dilated_binary_label_id])
            HAS_OVERLAY = inter>0
            
            # print(f"""
            #     np.sum((result ==label_id)){np.sum((result ==label_id))},
            #     np.sum(dilated_binary_label_id){np.sum(dilated_binary_label_id)},
            #     np.sum(overlay){np.sum(overlay)},
            #     np.sum(binary_non_label){np.sum(binary_non_label)}
            #     """)
            
            if HAS_OVERLAY:
                dilated_binary_label_id[overlay] = False
        
        result[dilated_binary_label_id & threshold_binary] = label_id  
        
        ## Save the results if there is needed         
    
    return result


def reorder_segmentation(segmentation, min_size=None, sort_ids=True, top_n=None):
    """
    Reorder segmentation labels based on their size and optionally remove small segments.

    Args:
        segmentation (np.ndarray): Array with class labels (0 is background).
        min_size (int, optional): Minimum size for a segment to be kept.
        sort_ids (bool, optional): If True, sort class IDs by segment size (default: True).
        top_n (int, optional): If set, keep only the top N largest segments.

    Returns:
        tuple:
            - reordered_segmentation (np.ndarray): Array with reordered class labels.
            - class_mapping (dict): Mapping from old to new class labels.
    """
    # Non-negative integer ids up to 2**24: sizes from one bincount, relabelled with a
    # lookup table (the loop below scans the whole array twice per class).
    max_id = None
    if segmentation.dtype.kind in 'iu' and np.can_cast(segmentation.dtype, np.intp) and segmentation.size:
        max_id = int(segmentation.max())
        if max_id >= 2 ** 24 or (segmentation.dtype.kind == 'i' and int(segmentation.min()) < 0):
            max_id = None
    if max_id is not None:
        counts = _bincount(segmentation, max_id + 1)
        unique_classes = np.flatnonzero(counts[1:]) + 1
        sizes = dict(zip(unique_classes.astype(segmentation.dtype), counts[unique_classes]))
    else:
        # Find unique classes excluding background
        unique_classes = np.unique(segmentation)
        unique_classes = unique_classes[unique_classes != 0]

        # Calculate the size of each segment
        sizes = {cls: np.sum(segmentation == cls) for cls in unique_classes}

    # Print the size of each class
    # print("Original Class Sizes:")
    # for cls, size in sizes.items():
    #     print(f"Class {cls}: Size {size}")
    
    # Sort classes by size in descending order
    sorted_classes = sorted(sizes, key=sizes.get, reverse=True)

    # Keep only the top N largest segments if specified
    if top_n is not None:
        sorted_classes = sorted_classes[:top_n]

    # Filter out small segments if min_size is specified
    if min_size is not None:
        sorted_classes = [cls for cls in sorted_classes if sizes[cls] >= min_size]

    # Determine new class IDs based on sorting preference
    if sort_ids:
        class_mapping = {old: new for new, old in enumerate(sorted_classes, start=1)}
    else:
        class_mapping = {old: old for old in sorted_classes}

    # Create a new segmentation array with reordered class labels
    if max_id is not None and not sort_ids and min_size is None and top_n is None:
        # every class keeps its id
        reordered_segmentation = segmentation.copy()
    elif max_id is not None:
        lut = np.zeros(max_id + 1, segmentation.dtype)
        if class_mapping:
            lut[np.fromiter(class_mapping.keys(), np.intp)] = np.fromiter(class_mapping.values(), np.int64)
        reordered_segmentation = _apply_lut(lut, segmentation)
    else:
        reordered_segmentation = np.zeros_like(segmentation)
        for old, new in class_mapping.items():
            reordered_segmentation[segmentation == old] = new

    # Print the mapping and reordered sizes
    # print("\nReordered Class Mapping and Sizes:")
    # for old, new in class_mapping.items():
    #     print(f"Old Class {old} -> New Class {new}, Size: {sizes[old]}")

    return reordered_segmentation, class_mapping

def fill_holes_seg(seg):
    # img = img[0,...]

    # Find unique labels in the mask (excluding 0, the background)
    unique_labels = np.unique(seg)
    unique_labels = unique_labels[unique_labels != 0]

    # Create a new mask to store the filled regions
    filled_mask = np.zeros_like(seg)


    # Process each label individually
    for label in unique_labels:
        print(f"working on {label}")
        # Create a binary mask for the current label
        binary_region = (seg == label)
        print("before filling ", np.sum(binary_region))
        # Fill holes in the binary mask
        filled_region = binary_fill_holes(binary_region)
        print("after filling ", np.sum(filled_region))
        
        # Add the filled region back to the filled_mask
        filled_mask[filled_region] = label
    
    return filled_mask



def merge_overlayed_classes(mask1, mask2, merged=True):
    """
    Processes segmentation masks, finding all overlayed classes of mask2 for each class in mask1.
    If merged=True, multiple overlapping classes in mask2 will be merged into the smallest label.

    Args:
        mask1 (numpy.ndarray): First segmentation mask (classes from 0 to N, where 0 is background).
        mask2 (numpy.ndarray): Second segmentation mask (classes from 0 to N, where 0 is background).
        merged (bool): If True, multiple overlayed classes in mask2 will be merged to the smallest one.

    Returns:
        numpy.ndarray: Updated mask2 with merged classes (if merged=True).
        dict: Dictionary showing overlay relationships {class_in_mask1: {overlapping_classes_in_mask2}}.
    """
    assert mask1.shape == mask2.shape, "Masks must have the same shape"

    unique_classes_mask1 = np.unique(mask1)
    unique_classes_mask1 = unique_classes_mask1[unique_classes_mask1 != 0]  # Remove background

    overlay_map = {}  # Stores {class in mask1: set of classes in mask2}

    updated_mask2 = mask2.copy()

    for class_label in unique_classes_mask1:
        # Find where mask1 is equal to the current class
        mask1_region = (mask1 == class_label)

        # Get all overlapping classes from mask2
        overlapping_classes = np.unique(mask2[mask1_region])
        overlapping_classes = overlapping_classes[overlapping_classes != 0]  # Remove background

        if len(overlapping_classes) == 0:
            continue  # No overlap

        overlay_map[class_label] = set(overlapping_classes)

        # Merge logic: Assign multiple overlapping labels to the smallest label
        if merged and len(overlapping_classes) > 1:
            print(class_label)
            smallest_label = min(overlapping_classes)
            for label_to_merge in overlapping_classes:
                updated_mask2[mask2 == label_to_merge] = smallest_label

    return updated_mask2, overlay_map


### Checkings:
def check_tiff_files(tifffile_list):
    """Do a quick check on whether all tiff files exist 

    Args:
        tifffile_list (_type_): list of tiff files
    """
    
    # Iterate over the 'seg_file' column in the DataFrame
    
    for file_path in tifffile_list:
        # Check if the file exists
        # print(file_path)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")
        try:
            # Attempt to open the file using tifffile.imread
            tifffile.imread(file_path)
        except Exception as e:
            print(f"File at index {file_path} cannot be opened. Error: {e}")
            raise
    
    # Release variables (optional in Python, done automatically by garbage collection)
    # del df


def check_files(file_list):
    """Do a quick check on whether all files exist 

    Args:
        tifffile_list (_type_): list of tiff files
    """
    
    # Iterate over the 'seg_file' column in the DataFrame
    
    for file_path in file_list:
        # Check if the file exists
        print(file_path)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")
    
    # Release variables (optional in Python, done automatically by garbage collection)
    # del df


# import numpy as np
# import tifffile

# # Example 3D array (e.g., read from a TIFF file)
# array_3d = np.random.randint(0, 11, (6, 500, 500))  # Example 3D array with values from 0 to 10

# # Define 10 colors (R, G, B)
# colors = np.array([
#     [255, 0, 0],    # Red
#     [0, 255, 0],    # Green
#     [0, 0, 255],    # Blue
#     [255, 255, 0],  # Yellow
#     [255, 0, 255],  # Magenta
#     [0, 255, 255],  # Cyan
#     [128, 0, 0],    # Maroon
#     [0, 128, 0],    # Dark Green
#     [0, 0, 128],    # Navy
#     [128, 128, 128], # Gray
#     [255, 255, 255] # black
# ])

# # Add an extra dimension for RGB channels
# shape_4d = array_3d.shape + (3,)
# array_4d = np.zeros(shape_4d, dtype=np.uint8)
# print(array_4d.shape)
# # Assign colors based on the value in the original 3D array
# for i in range(11):
#     array_4d[array_3d == i] = colors[i]

# # Save the 4D array as a TIFF file
# # tifffile.imwrite('colored_image.tiff', array_4d)

# # Save the 4D array as a TIFF file
# tifffile.imwrite('data/bones_suture/output_test_colour.tif', array_4d)


_LABEL_DTYPES = (np.uint8, np.uint16, np.uint32, np.uint64)


def label_dtype(max_id, min_dtype=np.uint8):
    """Smallest unsigned dtype, not smaller than `min_dtype`, that holds label ids up to `max_id`.

    Choose by the largest id, not by the number of labels: ids {7, 300} are only two
    labels but 300 does not fit in uint8 (it would wrap to 44 and merge with label 44).
    """
    min_size = np.dtype(min_dtype).itemsize
    for dt in _LABEL_DTYPES:
        if np.dtype(dt).itemsize >= min_size and int(max_id) <= np.iinfo(dt).max:
            return np.dtype(dt)
    raise ValueError(f"label id {max_id} does not fit in any unsigned integer type")


def cast_labels(labels, min_dtype=np.uint8, copy=True):
    """Cast a label image to label_dtype(labels.max(), min_dtype) without losing any id."""
    if labels.size and labels.min() < 0:
        raise ValueError(f"Label ids cannot be negative, found {labels.min()}.")
    max_id = int(labels.max()) if labels.size else 0
    return labels.astype(label_dtype(max_id, min_dtype), copy=copy)


def ensure_label_capacity(labels, max_id):
    """Return `labels`, promoted to a wider dtype only if `max_id` would not fit."""
    if int(max_id) <= np.iinfo(labels.dtype).max:
        return labels
    return labels.astype(label_dtype(max_id, labels.dtype))


def convert_seg_to_8bit(seg):
    # If a seg (16 or 32 bits) has less than 256 classes,
    # Convert it to 8bits to save some spaces
    
    # Get the number of unique classes in the array
    unique_classes = np.unique(seg)

    # Check if the number of unique classes is less than or equal to 255
    if len(unique_classes) <= 255:
        # Convert to 8-bit (uint8)
        print(f"Array has {len(unique_classes)} unique classes. Converting to 8-bit.")
        return seg.astype(np.uint8)
    else:
        # Keep it as is (16-bit)
        print(f"Array has {len(unique_classes)} unique classes. Keeping it as 16-bit.")
        return seg