# SPROUT

[![PyPI](https://img.shields.io/pypi/v/napari-sprout.svg)](https://pypi.org/project/napari-sprout/)
[![tests](https://github.com/EchanHe/SPROUT/actions/workflows/tests.yml/badge.svg)](https://github.com/EchanHe/SPROUT/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## TL;DR 
SPROUT is a lightweight segmentation toolkit for fine-grained and multi-part segmentation of grayscale 2D/3D images.

```bash
pip install "napari-sprout[all]"     # then start napari: Plugins → SPROUT
```

More ways to install: [Install](#install).

Quickstart: [Quickstart with Demo Data](./docs/demo_data_tutorial.md)

Video demo — Getting started with GUI-supported SPROUT using napari:
<a href="https://www.youtube.com/watch?v=6rA7wW6f0SU" target="_blank">
  <img src="https://img.youtube.com/vi/6rA7wW6f0SU/0.jpg" width="360">
</a>

Click [here](https://www.youtube.com/watch?v=4DHI1gs1lzc&list=PL8tsvVKJ9n6FjYYplo-IAwKBLxDh578aj) for the playlist of all SPROUT videos (mainly the napari plugin). See [Usage](#usage) section for command-line usage.

---

- [Introduction](#introduction)
- [Install](#install)
- [Usage](#usage)
   - [Terms](#terms)
   - [Input data](#input-data)
   - [Quickstart with demo data](#quickstart-with-demo-data)
   - [Command line](#command-line)
- [View & Edit (napari)](#view--edit-napari)
- [Demo Results](#demo-results)
- [Related tools](#related-tools)
- [Extras](#extra-features)

## Introduction

**SPROUT (Semi-automated Parcellation of Region Outputs Using Thresholding and Transformation)**  
SPROUT is an interpretable workflow for semi-automated, multi-class segmentation of volumetric images (e.g., CT, MRI). It requires no training data and is designed for researchers facing challenging segmentation tasks, such as structures with fine details or weak boundaries, where data for deep learning is scarce. 
  
**Y. He, M. Camaiti, L.E. Roberts, J.M. Mulqueeney, E. Ioannou, M. Didziokas, A. Goswami. 2024.**  
[*SPROUT: A User-friendly, Scalable Toolkit for Multi-class Segmentation of Volumetric Images*](https://www.biorxiv.org/content/10.1101/2024.11.22.624847v2.abstract)  


## Install

SPROUT needs Python 3.10 or newer. We recommend a conda environment ([Miniforge](https://conda-forge.org/download/)):

```bash
conda create -n sprout python=3.12
conda activate sprout
pip install "napari-sprout[all]"     # SPROUT, the napari plugin, napari and Qt
```

Then start napari (`napari`) and open **Plugins → SPROUT**. The `sprout` command is installed too.

Other ways:

| You want | Run |
|---|---|
| Only the command line (e.g. on a cluster, no napari) | `pip install sprout-seg` |
| The plugin, napari already installed | `pip install napari-sprout` |
| nnInteractive (Prompt (AI) tab, `sprout --prompt --nninteractive`) | `pip install "sprout-seg[ai]"`. With an NVIDIA GPU, first install PyTorch for your CUDA version ([pytorch.org](https://pytorch.org/get-started/locally/)). |
| Meshes (`make_mesh`) | `pip install "sprout-seg[mesh]"` |
| SAM | see [External Models](docs/external_model_installation.md#sam-segment-anything-model) |
| To change the code | `git clone https://github.com/EchanHe/SPROUT.git`, `cd SPROUT`, `pip install -e . -e ./napari-sprout` |

The paper's results were made with Python 3.10, numpy 1.26.4, pandas 2.2.1, scikit-image 0.22.0, scipy 1.15.3, tifffile 2024.2.12, napari 0.6.1. The tests run on these versions and on the newest ones.


## Usage

SPROUT has two steps:

<img src="./docs/material/flowchart.png" alt="SPROUT workflow" width="600">

1. **Seeds**: thresholds and erosion split the image into separate regions.
   - **Seed generation**: one seed per setting of thresholds and erosion.
   - **Adaptive seed**: combines the seeds of several settings into one final seed, often better than any single setting.
   - **Sweep adaptive seed**: one seed per threshold, then all of them are combined. No early stop, and fast with many seeds. Adaptive or sweep? See [the comparison](./docs/config_sweep_adaptive_seed.md#which-adaptive-seed-should-i-use).
2. **Growth**: seeds grow back to the full shape of each part.
   - **Threshold-based growth**, guided by image intensity.
   - **Seeds as prompts** for nnInteractive (3D) or SAM (slice by slice, fused into 3D).

Seeds and grown results often need checking or editing: see the [napari plugin](#view--edit-napari).

### Terms

The same words are used in the code, the YAML configs, the napari plugin and the paper.

| Term | Meaning |
|---|---|
| **segmentation** | A label map: each voxel holds an integer id, 0 is background. In napari it is a *Labels layer*. Seeds and the results of growth and of interactive models are all segmentations. |
| **seed** / **seeds** | A segmentation made by seed generation, where structures are shrunk apart by thresholds and erosion. Seeds are the input of growth. |
| **region** | The voxels that share one id. A region has an id but no meaning yet: the id is just a number, and regions can be split, merged or removed. |
| **class** | A semantic category, e.g. *head* or *femur*. Class mapping assigns regions to classes; the id of a class is its *class id*. Interactive models (nnInteractive) segment one class at a time, so map the seeds to classes first. |
| **step** | One threshold or erosion setting of a sweep adaptive seed run. Each step gives one mask, in `steps/`. |
| **segments** | The parameter "keep the N largest regions" (`segments`; adaptive seed also has `init_segments` and `last_segments` for its first and last step). It never means a region itself. |
| **grown segmentation** | The result of growth ("Grow result" in napari); likewise the *nnInteractive segmentation* ("Prompt result"). |


### Input data

2D or 3D greyscale TIFF (`.tif`, `.tiff`), 8-bit, 16-bit or float, with shape `(Height, Width)` or `(Depth, Height, Width)`. Colour images are not supported.

### Quickstart with demo data

- [Demo data and basic usage](docs/demo_data_tutorial.md): a dog skeleton, foraminifera chambers and a 2D cell image, in `./demo_data` with their seeds. More demo data on [Zenodo](https://zenodo.org/records/16857648).
- [Scenarios](docs/scenarios.md): advanced usage, as questions and answers.

### Command line

Each step is one command with a YAML config. Templates are in [`template/`](./template/).

| Step | Command | Parameters |
|---|---|---|
| Seed generation | `sprout --seeds --config make_seeds.yaml` | [config_seed.md](./docs/config_seed.md) |
| Adaptive seed | `sprout --adaptive_seed --config make_adaptive_seed.yaml` | [config_adaptive_seed.md](./docs/config_adaptive_seed.md) |
| Sweep adaptive seed | `sprout --sweep_adaptive --config sweep_adaptive_seed.yaml` | [config_sweep_adaptive_seed.md](./docs/config_sweep_adaptive_seed.md) |
| Growth | `sprout --grow --config make_grow.yaml` | [config_grow.md](./docs/config_grow.md) |
| nnInteractive | `sprout --prompt --nninteractive --config nninteractive_predict.yaml` | [config_nninteractive.md](./docs/config_nninteractive.md) |
| SAM | `sprout --prompt --sam --config sam_predict.yaml` | [config_sam.md](./docs/config_sam.md) |

- Many images: add `--batch` and use a batch config with a CSV of images (`template/batch_*.yaml`; [video](https://www.youtube.com/watch?v=5o-PJ34Yirc)).
- Threads: `num_threads` in the config.
- In a clone of this repository, without installing, `python sprout.py` does the same as `sprout`.
- `sprout --help` lists all options.

## View & Edit (napari)

Video: editing segmentation results in napari-sprout  
<a href="https://www.youtube.com/watch?v=H9pLkmy6WwA" target="_blank">
  <img src="https://img.youtube.com/vi/H9pLkmy6WwA/0.jpg" width="360">
</a>

To launch **napari-sprout**:

1. Start napari by running `napari` in a terminal or command prompt.
2. Open the `napari-sprout` plugin from the napari menu: Plugins → SPROUT.

From here, you can use a set of tools to load, view, and edit your data.  

-   **SPROUT**: tabs for the whole pipeline. Parameters match the YAML configs (Import / Export YAML).
    -   **1. Seeds**: seed and adaptive seed generation.
    -   **2. Grow**: grow seeds into a segmentation.
    -   **3. Prompt (AI)**: seeds as prompts for nnInteractive ([install](#install)).
    -   **ROI & Boundary**: run a tab inside a region of interest; make a boundary mask (top-hat).
-   **Edit**: keep or remove regions (Ctrl+click), replace ids, merge layers, split, filter, fill holes, morphology.
-   **Label Mapping**: map regions to classes, e.g. `1 → bone_1`, `2 → bone_2`.
-   **Batch Builder**: batch CSVs, optionally with per-image thresholds (e.g. Otsu); reference shapes.
-   **Data Info**: shape, value range, label count and sizes of the selected layer.


In addition to `napari-sprout`, **napari** has its own editing tools (e.g. **Brush** to add regions, **Erase** to remove).


You can also visualize, inspect, and edit seeds or results using 3D image processing tools such as **3D Slicer**, **Dragonfly**, and **AVIZO**. Alternatively, you can generate meshes and view or edit them in mesh-based tools like **MeshLab**. See [Extra section](#extra-features)




## Demo Results

Batch results: dog bones in CT scans with different anatomy and postures.

<img src="./docs/material/dog.png" alt="Dog bones" width="700">


Batch results: foraminifera chambers from binary masks (masks work as input too), with different numbers of chambers.

<img src="./docs/material/foram_v3_no_3.png" alt="Foraminifera chambers" width="700">

Gallery of other results

<img src="./docs/material/result_1.png" alt="Other results" width="560">

_Note: All results underwent manual post-processing primarily to verify and correct segmentation errors._


## Related tools

- [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive): 3D interactive segmentation from points, scribbles, boxes and lassos. SPROUT seeds can be its prompts (Prompt (AI) tab).
- [SAM](https://github.com/facebookresearch/segment-anything) / [SAM 2](https://github.com/facebookresearch/sam2): promptable segmentation of 2D images. SPROUT seeds can be its prompts.
- [Biomedisa](https://biomedisa.info): fills in a 3D segmentation from a few labelled slices.
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet): accurate supervised segmentation when training labels exist; corrected SPROUT results can be its training labels.
- [BounTI](https://pubmed.ncbi.nlm.nih.gov/38760955/): boundary-preserving threshold iteration, which inspired SPROUT (M. Didziokas, E. Pauws, L. Kölby, R. H. Khonsari, M. Moazen. 2024. *Journal of Anatomy* 245, 829–841).

----------
## Extra Features

### Other Optional Tools
You can use other tools to visualize and edit SPROUT's segmentation results, see [Optional Tools](./docs/supplementary_extra_plugins.md) for details.

### Helper Functions
In napari-sprout and plugins, we provided GUI version of editing results. You can also find the script-version in [Helper Functions](./docs/supplementary_helper.md).

### BounTI Multithreaded Implementation

The script `./BounTI/run_bounti_mp.py` runs **BounTI** ([Related tools](#related-tools)) and **BounTI Flood** for a list of initial thresholds in parallel.









