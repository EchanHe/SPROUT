# Demo Data and Basic Usage
This document introduces the demo datasets and the basic usage of SPROUT.  
We demonstrate how to use **seed generation**, **adaptive seed generation**, **growth**, and **SproutSAM**.  

For more advanced scenarios and parameter tuning, see [scenarios.md](scenarios.md).

For the napari quickstart, see [napari Quickstart Guide](napari_quickstart.md).

## Specs
Windows 11, equipped with an 11th Gen Intel(R) Core(TM) i7-11800H CPU @ 2.30 GHz, 48 GB RAM.


## Demo: Dog Veterinary CT, Target = Bones

**Dataset**  
- File: `dog_img.tif` (resampled CT volume)  
- Shape (z, y, x): **(691, 161, 122)**  
- Intensity range: **0–255 (8-bit)**  

We can understand that using a threshold range of **(120, 255)** (using `preview` function in `napari-sprout`) yields a relatively complete reconstruction of the skeleton. 

### Example: simple seed generation

To segment **four specific bones** mentioned in the manuscript (scapula, humerus, radius, and ulna), we tested several thresholds — (210,255), (220,255), (230,255) — with erosion values 0 or 1, `segments = 8`, and different erosion footprints (ball, XY disk, XZ disk, YZ disk).  

A configuration file is provided in `./template/make_seeds.yaml`. Run with:

```bash
python sprout.py --seeds --config ./template/make_seeds.yaml
```

The process finishes in under one minute (3 threads).

From the results, the best setting was:

-   **threshold = (220, 255)**
-   **erosion = 0**
-   **segments = 8**
    
This produces eight connected components, including the four target bones and extra parts. The final cleanup can be done interactively with `napari-sprout`->`sprout_edit`.

### Example: threshold-based adaptive seed generation
Instead of using a single threshold, we can define a list threshold ranges, for example lower thresholds from 120 to 220 with a step of 10.

The output will segment the four target bones, but allows a more flexible parameter selection (i.e., instead of a fixed single threshold).

Also if the task requires segmenting more bones, adaptive seed generation can retain small parts.

### Example: Threshold-based Growth

 Growth can be applied by with multiple threshold ranges (from narrow to wide). This allows seed regions to expand incrementally and helps control overgrowth. 

The example seed can be found in: `./demo_data/dog_seed.tif`

A configuration file is provided in `./template/make_seeds.yaml`. Run with:

```bash
python sprout.py --grow --config ./template/make_grow.yaml
```

In this example, we use **6 threshold ranges**, lower threshold decreasing from `220` to `120`:

-   First 4 ranges: **20 dilation steps** each
    
-   Fifth range: **5 dilation steps**
    
-   Final range: **1 dilation step**

The process finishes in around 20 seconds (4 threads).    

As a rule of thumb, we recommend using **larger dilation steps in the early stages** ( early stopping can prevent unnecessary computation), ensuring components grow to similar sizes at each level. In later stages, **fewer dilation steps** are sufficient to refine fine details.

---
## Demo Data: Foraminifera Binary Segmentation

**Dataset**  
- File: `foram_img.tif`
- Shape (z, y, x): **(215, 1015, 992)**
- Intensity range: Binary (0, 255)

### **Erosion-based Adaptive Seeding**

The goal is to segment **individual chambers**. This demo image contains **18 chambers** in total.

-   Because the input is binary, a direct threshold only produces a **single region**.
    
-   A preview without erosion yields **6 connected regions**, with many chambers fused together.
    
-   Increasing erosion helps separation, but excessive erosion causes chamber loss. For example, at `erosion = 5`, most chambers are separated, but only **13 regions** remain because smaller chambers are eroded away.

A configuration file is provided in `./template/make_seeds.yaml`. Run with:

```bash
python sprout.py --adaptive_seed --config ./template/make_adaptive_seed.yaml
```

The process finishes in around 8 minutes (4 threads).

The final cleanup can be done interactively with `napari-sprout`->`sprout_edit`.

### **Single-threshold Grow**

Since this dataset is **binary**, a simple grow operation can be applied with single threshold. 
Using **grow iterations = 10**, ensuring that all chambers are fully expanded back after erosion.

The process finishes in around 2 minutes (4 threads).


## Demo Data: 2D Cell Microscopy Image
Data: `2d_cell.tif`
- Shape (y, x): **(2796, 2796)**
### Example: Use SproutSAM
In this example, we use **`2d_cell_seed.tif`** as the input seed mask to generate prompts.  
The configuration file is provided at `./template/sam_predict.yaml`.

Run the following command:

```bash
python sprout.py --sam --config ./template/sam_predict.yaml
```