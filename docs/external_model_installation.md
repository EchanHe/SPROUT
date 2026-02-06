# External Model Installation Guide

This guide covers the installation of optional external segmentation models for SPROUT.

---

## Table of Contents

- [SAM (Segment Anything Model)](#sam-segment-anything-model)
- [nnInteractive](#nninteractive)

---

## SAM (Segment Anything Model)

SPROUT supports integration with Segment Anything Model for enhanced segmentation capabilities.

If you want to use prompts generated from SPROUT in SAM, you can install SAM models like [SAM](https://github.com/facebookresearch/segment-anything) and [SAM2](https://github.com/facebookresearch/sam2). Currently only supports SAM1 and SAM2-based models

### Core Dependencies

```bash
pip install torch torchvision torchaudio
pip install opencv-python-headless
```

⚠️ For optimal performance, we recommend using GPU-enabled PyTorch. You can find the correct installation command based on your system here:  
[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### 📦 Install Segment Anything Models

You also need to install the SAM and/or SAM2 libraries:

-   **[SAM (Segment Anything Model)](https://github.com/facebookresearch/segment-anything)**
    
    ```bash
    git clone https://github.com/facebookresearch/segment-anything.git
    cd segment-anything
    pip install -e .
    ```

-   **[SAM2 (Segment Anything v2)](https://github.com/facebookresearch/sam2)**

    ```bash
    git clone https://github.com/facebookresearch/sam2.git
    cd sam2
    pip install -e .
    ```

## nnInteractive
See [Official repo](https://github.com/MIC-DKFZ/nnInteractive) for detail information

 
You need a Linux or Windows computer with a Nvidia GPU. 10GB of VRAM is recommended. Small objects should work with <6GB.

#### Install PyTorch
Go to the [PyTorch homepage](https://pytorch.org/get-started/locally/) and pick the right configuration.
Note that since recently PyTorch needs to be installed via pip. This is fine to do within your conda environment.

For Ubuntu with a Nvidia GPU, pick 'stable', 'Linux', 'Pip', 'Python', 'CUDA12.6' (if all drivers are up to date, otherwise use and older version):

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

#### Install nnInteractive
Either install via pip:
`pip install nninteractive`

Or clone and install this repository:
```bash
git clone https://github.com/MIC-DKFZ/nnInteractive
cd nnInteractive
pip install -e .
```