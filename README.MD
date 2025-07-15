# SPROUT
_A semi-automated, threshold-based segmentation toolkit for 2D/3D grayscale images_

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Input Data Format](#input-data-format)
  - [Seed Generation](#seed-generation)
  - [Merged Seed Generation](#merged-seed-generation)
  - [Seed Growth](#seed-growth)
  - [Batch Processing Pipelines](#batch-processing-pipelines)
- [View & Edit (napari)](#view--edit-napari)
- [Demo Results](#demo-results)
- [Extras](#extra-features)

## Introduction

**SPROUT: Semi-automated Parcellation of Region Outputs Using Thresholding and Transformation**  
SPROUT is a tool for parcellating segments from **3D scans** (e.g., micro-CT, MRI, and other imaging data) using semi-automated seed generation and growth. It can be applied to a diverse range of datasets, from **skulls and entire skeletons** (e.g., isolating individual bones) to **organs and man-made structures** (e.g., concrete blocks). 

New in February 2025: SPROUT now supports 2D greyscale images.

Example applications and method details are provided in:  
**Y. He, M. Camaiti, L.E. Roberts, J.M. Mulqueeney, M. Didziokas, A. Goswami. 2024.**  
[*Introducing SPROUT (Semi-automated Parcellation of Region Outputs Using Thresholding and Transformation): an adaptable computer vision tool to generate 3D and 2D segmentations.*](https://www.biorxiv.org/content/10.1101/2024.11.22.624847v1.abstract)  


## Prerequisites

1. **Python and Environment Setup**
   - **Install Python**: If Python is not already installed, we recommend downloading the [Anaconda distribution](https://www.anaconda.com/products/distribution). After providing your email, follow the download link and select the version suitable for your operating system.
   - **Set up Python Environment**:
     - For Windows users with Anaconda, open the **Anaconda Prompt** from the Start menu.
     - For other installations, open a terminal with Python access.

2. **Create and Configure Python Environment**
   - SPROUT supports **Python 3.10**. Required libraries and versions are:
     ```
     NumPy: 1.26.4
     Pandas: 2.2.1
     Scikit-image: 0.22.0
     Tifffile: 2024.2.12
     Pyyaml: 6.0.1
     Trimesh: 4.3.1
     Matplotlib: 3.8.3
     open3d: 0.18.0
     ```
   - Steps to create the environment:
     1. **Create the environment**:
        ```bash
        conda create -n sprout python=3.10
        ```
     2. **Activate the environment**:
        ```bash
        conda activate sprout
        ```
     3. **Install dependencies**:
        ```bash
        pip install numpy==1.26.4 pandas==2.2.1 scikit-image==0.22.0 tifffile==2024.2.12 pyyaml==6.0.1 trimesh==4.3.1 matplotlib==3.8.3 open3d==0.18.0
        ```

### 🧪 Recommended Visualization Tools

**💡 Strongly recommended: [napari](https://napari.org/)**  
SPROUT comes with a napari plugin for interactive 2D/3D visualization and editing of seeds, segmentation results, and more.

To install napari and the SPROUT plugin:

```bash
pip install napari
cd ./napari_sprout
pip install -e .
```
You can launch napari by running napari in the command line.
The SPROUT plugin will appear under the Plugins menu in the napari UI.

----------

### Other Optional Tools

You can use other tools, for detail please see: [Extras](#extra-features)



## Usage

SPROUT follows a **three-step workflow** for multi-class segmentation:  

1. **Seed Generation**  
   Identify and segment individual components within a **2D or 3D image** by mainly tuning **thresholding and erosion**. This step generates candidate seeds for further processing. Alternatively, a function called merged seeds generation can be used to generate better seeds.

2. **Candidate Seed Selection**  
   Review and choose the candidate seed for growth. This ensures that the segmentation process focuses on the desired regions.  

3. **Growth**  
   Expand the candidate seed to match the target regions, ensuring accurate segmentation of structures within the image.  



![workflow](./material/flowchart.png)

A detailed tutorial with step-by-step instructions and screenshots is available in the [`./material/`](./material/) directory.


### **Input Data Format**
SPROUT processes 2D and 3D grayscale images stored in **TIFF (`.tif`, `.tiff`) format**. 

- **Supported file types:** `.tif`, `.tiff`
- **Image Type:** 2D or 3D grayscale images
- **Bit Depth:** 8-bit, 16-bit, or 32-bit (floating point)
- **File Structure:**  
   - 2D images: `(Height, Width)`
   - 3D images: `(Depth, Height, Width)`

**Unsupported inputs:** Color images.

Two low-resolution scans are available in `./data`, including a **dog skeleton** `demo_dog.tiff` and a **foraminifera chamber segmentation** `demo_foram_segmentation.tif`.  
Additional demo data for the tutorial can be found [here](https://zenodo.org/records/14358829).

### 🌰 Run seed generation

**on a single image**
```bash
python sprout.py --seeds --config path/to/seed_config.yaml
```
**in batch mode**
```bash
python sprout.py --seeds --batch --config path/to/batch_seed.yaml

```

For full details on YAML parameters and other configs see:
- [Seed configuration options](./docs/config_seed.md)

For example, see the YAML configuration files in `./template/`.


### 🌱 Run Growth

**on a single image**
```bash
python sprout.py --grow --config path/to/grow_config.yaml
```
**in batch mode**
```bash
python sprout.py --grow --batch --config path/to/batch_grow.yaml

```

For full details on YAML parameters and other configs see:
- [Growth configuration options](./docs/config_grow.md)

For example, see the YAML configuration files in `./template/`.




## View & Edit (napari)

TODO


You can also visualize, inspect, and edit seeds or results using 3D image processing tools such as **3D Slicer**, **DragonFly**, and **AVIZO**. Alternatively, you can generate meshes and view or edit them in mesh-based tools like **MeshLab**. See Extra section


## Demo Results

SPROUT has been successfully applied to a variety of datasets, including:
- **Micro-CT full skeleton**
- **Micro-CT skull**
- **MRI heart**
- **Micro-CT concrete**
- **Binary segmentation of Forams**

The image below illustrates the outcomes from left to right:
1. **Volumetric Representations**: Visualizations of the original image stacks.
2. **Merged Seeds**: Results after applying SPROUT's segmentation and merging processes.
3. **Grow Results**: Results after growing on the candidate seeds

![Result1](./material/result_1.png)



----------
## Extra Features

   - [Visualize by Meshes](#visualize-by-meshes)
   - [Avizo Addons](#avizo-addons)
   - [ImageJ Macros](#imagej-mdacros)
   - [Helper Functions](#helper-functions)
   - [Parallelization Support](#parallelization-support)
   - [BounTI Multithreaded Implementation](#bounti-multithreaded-implementation)
   - [Unzipping Morphosource Zip Files](#miscellaneous-unzipping-morphosource-zip-files)


### Optional Tools
1. **MeshLab**  
   - MeshLab is a tool for viewing 3D meshes generated by SPROUT.

2. **Optional Visualization Tools**  
   - 3D: Tools like **DragonFly** or **Avizo** can be used to visualize and manipulate SPROUT's segmentation results.
   - 2D: Tools like **ImageJ** can be used to visualize and manipulate SPROUT's 2D segmentation results.

### Visualize by Meshes
SPROUT provide scripts to generate meshes for visualisation.

#### **`make_mesh.py`**
Generates meshes for segmentations.
   - **Usage**:
      - Run with the default configuration:  
         ```bash
         python make_mesh.py  
         ```
         *(Reads the `./make_mesh.yaml` file)*  
      - Run with a custom configuration:  
         ```bash
         python make_mesh.py <config>.yaml  
         ```
      - Run with a custom configuration:  
         ```bash
         python make_mesh.py ./template/make_mesh.yaml
         ```
      An example YAML configuration file is available at `./template/make_mesh.yaml`, which generates meshes for the grown results from the example in the `PipelineGrow.py`

   - **Key Parameters**:  
      - `img_path`: Path to the input segmentation. (Specify **either** `img_path` or `input_folder`)  
      - `input_folder`: A folder containing multiple segmentations.  
      - `output_folder`: Directory where the generated meshes will be saved. 
      - `num_threads`: Number of threads for parallel processing.  
      - `downsample_scale`: Mesh simplification factor (*higher values reduce mesh complexity*).
      - `step_size`: Controls the **marching cubes** mesh reconstruction (*1 = original size; higher values generate simpler meshes*).  
  
  - **Outputs**, For each **input segmentation**, a corresponding folder (matching the segmentation filename) will be created, containing:  
      - **Individual Meshes** (`.ply` format)  
         - Example: `1.ply`, `2.ply`, `3.ply` (each representing a separate segmentation class).  
      - **Merged Meshes**  
         - `merged.ply`: Combined mesh of all classes.  
         - `merged_n_to_m.ply`: Merged meshes for **class groups (e.g., classes 1–10, 11–20, etc.)**.  

Meshes use a 10-color cyclic colormap, where class IDs 1–10 map to distinct colors and repeat for IDs beyond 10.  
![Color Map](./material/colourmap.png)


### Avizo Addons
Addon scripts are saved in `./avizo` folder. 
Change the extension from `py` to `pyscro`, and drag into AVIZO's `Project View` to use it.

#### **`PreProcess.py`**  
It is a script designed for pre-processing images based on the skull-suture workflow. It simplifies the pre-processing pipeline by combining multiple functions into a single add-on, eliminating the need to open multiple functions in AVIZO. 

When setting up the input in the add-on UI, the following operations can be performed:
- **Cropping**: Define a region of interest to focus on specific parts of the image.
- **Swapping Axes**: Reorient the volume by swapping axes as needed.
- **Resampling**: Scale voxel sizes to adjust the resolution of the dataset.
- **Converting to 8-bit**: Reduce the bit depth of the image for standardization or compatibility.

Additionally, it automatically creates all essential objects useful for preprocessing, such as:
- Volume rendering
- Ortho-slices
- Axes
- Bounding boxes

**Transformations (e.g., rotations)** are not currently supported and must be performed separately.

#### **`LoadVisFiles.py`**  

It is a script designed to batch process and visualize segmentation labels in AVIZO.

- **Batch Loading and Conversion**: Load and convert 8-bit segmentation labels into AVIZO in a single batch, streamlining the workflow. Supports reordering material/class IDs based on size using AVIZO's `reorder` function. Some AVIZO versions may not support the reorder function. In such cases, simply untick the reorder option during processing.
- **Visualization Options**: Provides options for visualizing segmentation labels in either voxel or mesh format, enabling flexible data exploration.
- **Material Name Mapping**: Outputs the relationship between material names and their corresponding segmentation classes, ensuring clarity and traceability during visualization.

#### **`Edit.py`**  

Performs the same tasks described in the Helper Functions for Editing section, combining multiple editing operations into a single script for convenience.

---

### **ImageJ Macros**  

SPROUT provides ImageJ macros (`.ijm` files) in the `./plugins` folder to facilitate segmentation visualization and editing within ImageJ. These macros are **recommended for use with 2D images**.  

You can either **run the scripts directly in the Macro Editor** or **install the `.ijm` files** for easier access. Refer to the official ImageJ [documentation](https://imagej.net/ij/developer/macro/macros.html) for detailed instructions on macro installation and usage.  

#### **`load_vis_results.ijm`**  
This macro loads and colorizes all image files in a selected folder using the `glasbey` color lookup table (`LUT`). It is useful for quickly visualizing multiple segmentation results (e.g., seeds and grown results) in ImageJ.  
 
  - Asks for an input folder containing segmentation results  
  - Loads all images in the folder into ImageJ  
  - Applies `Lookup Tables` → `glasbey` for colorized segmentation display  

#### **`SPROUT_edit.ijm`**  
This macro allows users to manually edit segmentation masks in ImageJ using the `Multi-point tool` providing an intuitive way to refine **seeds** and **grown results**. 
1. Users can select regions of interest with the `Multi-point tool` and apply the following operations:  
   1. **Merge segmentation classes**: Combine selected segments into a single class.  
   2. **Convert selected segments to background**: Remove unwanted regions by setting them as background.  
   3. **Keep only selected regions**: Retain only the chosen areas and set all other segments as background.  
2. Users can use ROI tools like rectangle, polygon and free-hand to select regions and get the selected regions into user-input value. Then a split connected components can be performed, if users separate a region into multiple.   


### Helper Functions

#### Editing scripts
Editing scripts for segmentations (seeds and results) are provided in the `./help_functions/` folder:

- **`filter_class.py`**: Retains only the specified IDs in a segmentation.
- **`merge_class.py`**: Merges a list of IDs into a single ID in a segmentation.
- **`merge_imgs.py`**: Merges two segmentations (IDs for each segmentation can be specified).
- **`sort_and_filter_seg_by_size.py`**: Orders segments by size and removes those smaller than a given size.
- **`split_class.py`**: Splits a segment into multiple segments if separations (disconnected components) exist.

#### Processing files

- **`read_tiff_to_csv.py`**: Reads `.tif` or `.tiff` files from specified folders (e.g., `img`, `seg`, and `boundary`), aligns them based on filenames, and outputs a CSV file. This CSV file can then be used as input for pipeline scripts. Options are provided to handle prefixes and suffixes for alignment.

---
### Parallelization Support

All SPROUT scripts (`make_seeds.py`, `make_seeds_all.py`, `make_seeds_merged.py`, `make_grow_result.py`, `make_mesh.py`, `PipelineSeed.py`, `PipelineGrow.py`) support **parallelization** using Python's `threading` library. This allows for faster execution by distributing tasks across multiple threads.

- **How It Works**:  
  You can specify the number of threads in the configuration file (`num_threads`). By utilizing multiple threads, the scripts can process data simultaneously, significantly improving efficiency for large datasets or computationally intensive tasks.

- **Recommendation**:  
  For optimal performance, set `num_threads` to match the number of CPU cores on your system.

---
### BounTI Multithreaded Implementation

SPROUT was inspired by **BounTI**, a boundary-preserving threshold iteration approach:  
**M. Didziokas, E. Pauws, L. Kölby, R. H. Khonsari, M. Moazen. 2024.**  
[*BounTI: A user-friendly tool for automatic hard tissue segmentation.*](https://pubmed.ncbi.nlm.nih.gov/38760955/) *Journal of Anatomy 245, 829–841*.

The script `./BounTI/run_bounti_mp.py` is designed to run **BounTI** and **BounTI Flood** using a multithreaded approach. It efficiently processes a list of initial thresholds in parallel, leveraging multiple threads for improved performance.

---
### Miscellaneous: Unzipping Morphosource Zip Files


The following scripts provide utilities for handling [Morphosource datasets](https://www.morphosource.org/) in the `./help_functions/morphosource/` folder:

- **`morphosource_unzip.ipynb`**: Jupyter Notebook version for interactive use.
- **`morphosource_unzip.py`**: Python script for command-line usage. Input configurations (the input folder and the output folder) are stored in `morpho.yaml`.

#### Functionality:
- Unzips all `*.zip` files from a specified input folder to a target output folder.
- Logs important details for each file, including:
  - Path of the original `.zip` file.
  - Path of the extracted contents.
  - Status of the extraction (success/failure).
- Supports batch unzipping of Morphosource datasets in a folder.

#### Morphosource File Structure:
These scripts handle the expected Morphosource file structure, ensuring extracted files are well-organized for subsequent processing.

