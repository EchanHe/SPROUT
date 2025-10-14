import os
import pandas as pd



def remove_prefix(filename, prefix):
    """
    Safely remove a prefix from the start of a filename.

    Args:
        filename (str): The filename to process.
        prefix (str): The prefix to remove.

    Returns:
        str: The filename with the prefix removed if it exists, otherwise unchanged.
    """
    if filename.startswith(prefix):
        return filename[len(prefix):]
    return filename

def remove_suffix(filename, suffix):
    """
    Safely remove a suffix from the end of a filename.

    Args:
        filename (str): The filename to process.
        suffix (str): The suffix to remove.

    Returns:
        str: The filename with the suffix removed if it exists, otherwise unchanged.
    """
    if filename.endswith(suffix):
        return filename[: -len(suffix)]
    return filename


def sort_dict_and_extract_values(input_dict):
    """
    Sort a dictionary by keys alphabetically and extract values in that order.

    Args:
        input_dict (dict): The dictionary to process.

    Returns:
        list: A list of values sorted by the keys.
    """
    # Sort the dictionary by keys alphabetically
    sorted_keys = sorted(input_dict.keys())
    
    # Extract values in the sorted order of keys
    sorted_values = [input_dict[key] for key in sorted_keys]
    
    return sorted_values

def align_files_to_df(img_folder=None, seg_folder=None, boundary_folder=None,
                        seg_prefix="", seg_suffix="", 
                        boundary_prefix="", boundary_suffix="", 
                        match_type="exact"):
    """
    Align files from img, seg, and boundary folders and return a DataFrame.

    Args:
        img_folder (str): Path to the folder containing image files.
        seg_folder (str): Path to the folder containing segmentation files.
        boundary_folder (str): Path to the folder containing boundary files.
        seg_prefix (str): Prefix for segmentation filenames.
        seg_suffix (str): Suffix for segmentation filenames.
        boundary_prefix (str): Prefix for boundary filenames.
        boundary_suffix (str): Suffix for boundary filenames.
        match_type (str): Type of matching: "exact" for identical filenames, or "base" for using img as the base.

    Returns:
        pd.DataFrame: A DataFrame with aligned file paths.
    """
    # Helper function to list and sort files (excluding extensions)
    def list_files(folder):
        return sorted(
            [f for f in os.listdir(folder) if f.lower().endswith(('.tif', '.tiff'))]
        )
    

    def list_files(folder, prefix="", suffix="", must_include=True):
        """
        List and process files in a folder based on prefix and suffix.

        Args:
            folder (str): Path to the folder.
            prefix (str): Prefix to match filenames.
            suffix (str): Suffix to match filenames.

        Returns:
            dict: Dictionary with base filenames as keys and full paths as values.
        """
        # return {
        #     remove_suffix(remove_prefix(os.path.splitext(f)[0], prefix), suffix): sorted(
        #     [f for f in os.listdir(folder) if f.lower().endswith(('.tif', '.tiff'))]
        # )
        # }

        if must_include:
            return {
                remove_prefix(os.path.splitext(f)[0], prefix).replace(suffix, ""): os.path.join(folder, f)
                for f in os.listdir(folder) if f.lower().endswith(('.tif', '.tiff')) and os.path.splitext(f)[0].startswith(prefix)
                and os.path.splitext(f)[0].endswith(suffix)
            }
        else:
            return {
                remove_prefix(os.path.splitext(f)[0], prefix).replace(suffix, ""): os.path.join(folder, f)
                for f in os.listdir(folder) if f.lower().endswith(('.tif', '.tiff'))
            }

    
    # def get_abs_path(folder, file_list):
    #     # return [os.path.abspath(os.path.join(folder, f)) for f in file_list]
    #     return [os.path.abspath(f) for f in file_list]
        
    
    def get_abs_path(file_list):
        return [os.path.abspath(f) for f in file_list]
        

    # Initialize dictionaries for file lists
    file_lists = {}


    if img_folder:
        file_lists['img'] = list_files(img_folder)
    if seg_folder:
        file_lists['seg'] = list_files(seg_folder, prefix = seg_prefix,
                                       suffix=seg_suffix)
    if boundary_folder:
        file_lists['boundary'] = list_files(boundary_folder, prefix = boundary_prefix,
                                       suffix=boundary_suffix)


    data = {}

    if match_type == "exact":
        # Find common filenames exactly across all folders
        common_files = set(file_lists['img'].keys())
        if seg_folder:
            common_files &= set(file_lists['seg'].keys())
        if boundary_folder:
            common_files &= set(file_lists['boundary'].keys())
    elif match_type == "base":
        # Use img filenames as the base and match with seg and boundary files
        common_files = set(file_lists['img'].keys())

        if seg_folder:
            common_files &= {f.replace(seg_prefix, "").replace(seg_suffix, "") for f in file_lists['seg'].keys()}
        if boundary_folder:
            common_files &= {f.replace(boundary_prefix, "").replace(boundary_suffix, "") for 
                             f in file_lists['boundary'].keys()}    
    
    elif match_type == "sorted":
        if img_folder:
            data["img_path"] = get_abs_path(sort_dict_and_extract_values(file_lists['img']))
            # get_abs_path(img_folder, file_lists['img'])

        if seg_folder:
            
            data["seg_path"] = get_abs_path(sort_dict_and_extract_values(file_lists['seg']))
            # get_abs_path(seg_folder, file_lists['seg'])

        if boundary_folder:
            data["boundary_path"] = get_abs_path(sort_dict_and_extract_values(file_lists['boundary']))
            # get_abs_path(boundary_folder, file_lists['boundary'])
        
        df = pd.DataFrame(data)
        return df
    else:
        raise ValueError("Invalid match_type. Use 'exact' or 'base'.")
    
    sorted_keys = sorted(list(common_files))
    
    if img_folder:
        data["img_path"] = get_abs_path([file_lists['img'][f] for f in sorted_keys])
    if seg_folder:
        data["seg_path"] = get_abs_path([file_lists['seg'][f] for f in sorted_keys])
    if boundary_folder:
        data["boundary_path"] = get_abs_path([file_lists['boundary'][f] for f in sorted_keys])

    df = pd.DataFrame(data)

    return df

if __name__ == "__main__":
    # Example usage
    # df = align_files_to_df(
    #     img_folder = "data/test_batch/test_get_tiff/img_1",
    #     seg_folder= "data/test_batch/test_get_tiff/seg_fix",
    #     match_type = "base",
    #     seg_prefix="seg_"
    #     )  # Prompt user for folder paths or pass them as arguments
    # print("Aligned file paths:")
    # print(df)

    df = align_files_to_df(
        img_folder = "./data/haymar/20250203 Cell segmentation/20240313 Static/2d/",
        # seg_folder= "./data/haymar/green/2d/seeds",
        match_type = "sorted"
        )  # Prompt user for folder paths or pass them as arguments
    print("Aligned file paths:")
    print(df)

    df.to_csv("./data/haymar/20250203 Cell segmentation/20240313 Static/seed.csv", index=False)

    # Optionally save to a CSV file
    # save_csv = input("Do you want to save the result to a CSV file? (y/n): ").strip().lower()
    # if save_csv == 'y':
    #     output_csv = input("Enter path for the output CSV file: ").strip()
    #     df.to_csv(output_csv, index=False)
    #     print(f"Aligned file paths saved to: {output_csv}")
