# SPROUT
_A semi-automated, threshold-based segmentation toolkit for 2D/3D grayscale images_

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Usage](#️-usage)
   - [Input Data Format](#input-data-format)
   - [Seed Generation](#-run-seed-generation)
   - [Adaptive Seed Generation](#-run-adaptive-seed-generation)
   - [Growth](#-run-growth)
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




## ⚙️ Usage

SPROUT follows a **two-step workflow** for multi-class segmentation:

1. **Seed Generation**  
   The goal of this step is to reduce the regions of the image by breaking it into multiple distinct components (splits). There are two strategies:
   
   - [**Vanilla seed generation**](#-run-seed-generation): generate multiple seeds by varying configurations such as thresholds and erosion steps.
   - [**Adaptive seed generation**](#-run-adaptive-seed-generation): automatically combine information across seeds from different configurations to produce a often better and final seed.
2. **Growth**  
   [This step](#-run-growth) grows a seed by expanding the reduced components back toward their full target shapes. Growth is guided by image intensity and connectivity.

**🧪 Interactive Inspection & Editing**

In practice, both seed and grow results often require **visualisation, quality checking, or manual editing**, depending on your segmentation goals.

To support this, SPROUT is fully integrated into the **Napari** ecosystem as a plugin, please see [napari-sprout](#view--edit-napari). 



![workflow](./material/flowchart.png)



### **Input Data Format**
SPROUT processes 2D and 3D grayscale images stored in **TIFF (`.tif`, `.tiff`) format**. 

- **Supported file types:** `.tif`, `.tiff`
- **Image Type:** 2D or 3D grayscale images
- **Bit Depth:** 8-bit, 16-bit, or 32-bit (floating point)
- **File Structure:**  
   - 2D images: `(Height, Width)`
   - 3D images: `(Depth, Height, Width)`

**Unsupported inputs:** Color images.

This repo has a few small scans for testing in `./demo_data`, including a **dog skeleton** `dog_img.tif`, a **foraminifera chamber segmentation** `foram_img.tif` and a 2D cell microscopy image `2d_cell.tif`. Their corresponding seed images are also provided in the same directory.  
Additional demo data can be found [here](https://zenodo.org/records/14358829).

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

### 🧠 Run Adaptive Seed Generation

**on a single image**
```bash
python sprout.py --adaptive_seed --config path/to/adaptive_seed_config.yaml
```

**in batch mode**

```bash
python sprout.py --adaptive_seed --batch --config path/to/batch_adaptive_seed.yaml
```

For full details on YAML parameters and other configs see:

-   [Adaptive seed configuration options](./docs/config_adaptive_seed.md)
    

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

### 🧠 Run Foundation Model Prediction

**on a single image or a folder of images**
```bash
python sprout.py --sam --config path/to/config.yaml
```

**in batch mode**
```bash
python sprout.py --sam --batch --config path/to/batch_config.yaml
```
For full details on YAML parameters and other configs see:
- [Foundation Model configuration options](./docs/config_sam.md)

For example, see the YAML configuration files in `./template/`.

### ⚡ Parallelization Support

`sprout.py` can run tasks in parallel using Python’s `threading`. Just set `num_threads` in your config to speed up processing, especially for large datasets.





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

### Other Optional Tools
You can use other tools to visualize and edit SPROUT's segmentation results, see [Optional Tools](./docs/supplementary_extra_plugins.md) for details.

### Helper Functions
In napari-sprout and plugins, we provided GUI version of editing results. You can also find the script-version in [Helper Functions](./docs/supplementary_helper.md).

### BounTI Multithreaded Implementation

SPROUT was inspired by **BounTI**, a boundary-preserving threshold iteration approach:  
**M. Didziokas, E. Pauws, L. Kölby, R. H. Khonsari, M. Moazen. 2024.**  
[*BounTI: A user-friendly tool for automatic hard tissue segmentation.*](https://pubmed.ncbi.nlm.nih.gov/38760955/) *Journal of Anatomy 245, 829–841*.

The script `./BounTI/run_bounti_mp.py` is designed to run **BounTI** and **BounTI Flood** using a multithreaded approach. It efficiently processes a list of initial thresholds in parallel, leveraging multiple threads for improved performance.









