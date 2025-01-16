import numpy as np
import tifffile

def reorder_segmentation(segmentation, min_size=None, sort_ids=True):
    """
    Reorder segmentation labels based on their size and optionally remove small segments.
    
    Parameters:
    - segmentation: numpy array with class labels from 0 to n (0 is background).
    - min_size: minimum size for a segment to be kept (optional).
    - sort_ids: boolean indicating whether to sort class IDs by size (default: True).
    
    Returns:
    - reordered_segmentation: numpy array with reordered class labels.
    """
    # Find unique classes excluding background
    unique_classes = np.unique(segmentation)
    unique_classes = unique_classes[unique_classes != 0]
    
    # Calculate the size of each segment
    sizes = {cls: np.sum(segmentation == cls) for cls in unique_classes}

    # Print the size of each class
    print("Original Class Sizes:")
    for cls, size in sizes.items():
        print(f"Class {cls}: Size {size}")
    
    # Sort classes by size in descending order
    sorted_classes = sorted(sizes, key=sizes.get, reverse=True)

    # Filter out small segments if min_size is specified
    if min_size is not None:
        sorted_classes = [cls for cls in sorted_classes if sizes[cls] >= min_size]

    # Determine new class IDs based on sorting preference
    if sort_ids:
        class_mapping = {old: new for new, old in enumerate(sorted_classes, start=1)}
    else:
        class_mapping = {old: old for old in sorted_classes}

    # Create a new segmentation array with reordered class labels
    reordered_segmentation = np.zeros_like(segmentation)
    for old, new in class_mapping.items():
        reordered_segmentation[segmentation == old] = new

    # Print the mapping and reordered sizes
    print("\nReordered Class Mapping and Sizes:")
    for old, new in class_mapping.items():
        print(f"Old Class {old} -> New Class {new}, Size: {sizes[old]}")

    return reordered_segmentation, class_mapping
if __name__ == "__main__":   
    # Example usage
    # segmentation = np.array([
    #     [0, 1, 1, 0, 2, 2],
    #     [0, 1, 1, 0, 2, 2],
    #     [0, 0, 0, 0, 0, 0],
    #     [3, 3, 0, 4, 4, 4],
    #     [3, 3, 0, 4, 4, 4]
    # ])
    
    ### Input parameter
    segmentation_path = "./result/foram_james/seeds/combine/combined.tif"
    min_size = 5
    output_path = "./result/foram_james/seeds/combine/combined_sorted.tif"

    #####

    segmentation = tifffile.imread(segmentation_path)

    reordered_segmentation, class_mapping = reorder_segmentation(segmentation, min_size=min_size)

    # print("Original Segmentation:")
    # print(segmentation)

    # print("\nReordered Segmentation:")
    # print(reordered_segmentation)

    print("\nClass Mapping:")
    print(class_mapping)

    tifffile.imwrite(output_path, 
                     reordered_segmentation,
                     compression='zlib')
