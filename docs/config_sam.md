## 🧠 Foundation Model Segmentation Config (`--sam`)

```bash
python sprout.py --sam --config path/to/config.yaml
```

A template of the configuration file for SAM-based segmentation can be found at:  [`../template/sam_predict.yaml`](../template/sam_predict.yaml).

This mode uses **foundation models** (e.g., Segment Anything Model, SAM, SAM2) to generate segmentation masks using point or box prompts derived from an existing segmentation mask.
### Required Parameters

| Parameter            | Required | Type  | Description                                                                           |
| -------------------- | -------- | ----- | ------------------------------------------------------------------------------------- |
| `img_path`           | ✅        | `str` | Path to the input grayscale image (2D or 3D `.tif`).                                  |
| `seg_path`           | ✅        | `str` | Path to the seed/segmentation mask used to generate prompts.                          |
| `output_folder`      | ✅        | `str` | Root directory to save all outputs, including prompts, masks, and final segmentation. |
| `n_points_per_class` | ✅        | `int` | Number of **positive points** to sample from each class.                              |
| `negative_points`    | ✅        | `int` | Number of **negative points** to sample for each class.                               |
| `prompt_type`        | ✅        | `str` | Type of prompt to use. Supported: `"point"` or `"bbox"`.                              |
| `which_sam`          | ✅        | `str` | Model family: `"sam1"` (original Meta SAM) or `"sam2"` (e.g., Hiera-SAM).             |


### Optional Parameters
| Parameter               | Required | Type   | Description                                                                                |
| ----------------------- | -------- | ------ | ------------------------------------------------------------------------------------------ |
| `output_filename`       | ❌        | `str`  | Name of final segmentation output file. Defaults to base name of image.                    |
| `sample_neg_each_class` | ❌        | `bool` | Whether to sample negative points **from each other class separately** (default: `False`). |
| `per_cls_mode`          | ❌        | `bool` | If `True`, generate individual mask per class and then fuse by majority voting.            |
| `workspace`             | ❌        | `str`  | Optional root path prefix. If set, all paths are interpreted as relative to this.          |

### SAM Config Parameters
| Parameter        | Required | Type  | Description                                         |
| ---------------- | -------- | ----- | --------------------------------------------------- |
| `sam_checkpoint` | `which_sam`  is `"sam1"`        | `str` | Path to the `.pth` checkpoint file of SAM1.         |
| `sam_model_type` | `which_sam`  is `"sam1"`        | `str` | Model type: one of `'vit_b'`, `'vit_l'`, `'vit_h'`. |
| `sam2_checkpoint` | `which_sam`  is `"sam2"`        | `str` | Path to the `.pt` checkpoint for SAM2.                   |
| `sam2_model_cfg`  | `which_sam`  is `"sam2"`        | `str` | Path to the config file (`.yaml`) for SAM2 architecture. |
| `custom_checkpoint` | ❌        | `str` | Optional custom checkpoint path for SAM1/SAM2. If provided, overrides. |


## SAM Config Parameters in batch (`--sam --batch`)

Batch mode allows you to predict segmentations for multiple images in a single run using a CSV file. Each row in the CSV corresponds to one image and one segmentation, and may optionally override additional parameters.

```bash
python sprout.py --sam --batch --config path/to/batch_config.yaml
```

Templates of the configuration file and the csv file for sam can be found at: [`../template/batch_sam.yaml`](../template/batch_sam.yaml) and [../template/sam_input.csv](../template/sam_input.csv).


### New required YAML parameter for batch mode:

| Parameter  | Required | Type  | Description                                                                      |
| ---------- | -------- | ----- | -------------------------------------------------------------------------------- |
| `csv_path` | ✅        | `str` | Path to a CSV file. Must contain at least the `img_path` and `seg_path` columns. |


In addition, all **required parameters** (`output_folder` etc.) must be either: : Globally defined in the YAML file, **or** Provided per image in the CSV file.