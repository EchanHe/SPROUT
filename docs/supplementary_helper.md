# Helper Functions

## Editing scripts
Editing scripts for segmentations (seeds and results) are provided in the `./help_functions/` folder:

- **`filter_class.py`**: Retains only the specified IDs in a segmentation.
- **`merge_class.py`**: Merges a list of IDs into a single ID in a segmentation.
- **`merge_imgs.py`**: Merges two segmentations (IDs for each segmentation can be specified).
- **`sort_and_filter_seg_by_size.py`**: Orders segments by size and removes those smaller than a given size.
- **`split_class.py`**: Splits a segment into multiple segments if separations (disconnected components) exist.

## Processing files

- **`read_tiff_to_csv.py`**: Reads `.tif` or `.tiff` files from specified folders (e.g., `img`, `seg`, and `boundary`), aligns them based on filenames, and outputs a CSV file. This CSV file can then be used as input for pipeline scripts. Options are provided to handle prefixes and suffixes for alignment.


## Miscellaneous: Unzipping Morphosource Zip Files


The following scripts provide utilities for handling [Morphosource datasets](https://www.morphosource.org/) in the `./help_functions/morphosource/` folder:

- **`morphosource_unzip.ipynb`**: Jupyter Notebook version for interactive use.
- **`morphosource_unzip.py`**: Python script for command-line usage. Input configurations (the input folder and the output folder) are stored in `morpho.yaml`.

### Functionality:
- Unzips all `*.zip` files from a specified input folder to a target output folder.
- Logs important details for each file, including:
  - Path of the original `.zip` file.
  - Path of the extracted contents.
  - Status of the extraction (success/failure).
- Supports batch unzipping of Morphosource datasets in a folder.

### Morphosource File Structure:
These scripts handle the expected Morphosource file structure, ensuring extracted files are well-organized for subsequent processing.
