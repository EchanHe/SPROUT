## nnInteractive Config Parameters (`--prompt --nninteractive`)

```bash
sprout --prompt --nninteractive --config path/to/config.yaml
```

A template of the configuration file can be found at: [`../template/nninteractive_predict.yaml`](../template/nninteractive_predict.yaml)

SPROUT turns the seeds into prompts for [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive), one class at a time: points sampled from each seed region, or the seed regions themselves as scribbles. nnInteractive then segments each class in 3D. Map the seeds to classes first (napari: Label Mapping): each id of `seg_path` is one class.

Install: `pip install "sprout-seg[ai]"`; with an NVIDIA GPU, install PyTorch for your CUDA version first ([pytorch.org](https://pytorch.org/get-started/locally/)). The model weights (e.g. the folder `nnInteractive_v1.0`) are downloaded as described in the [nnInteractive README](https://github.com/MIC-DKFZ/nnInteractive). See also [External Models](./external_model_installation.md#nninteractive).

### In napari

The Prompt (AI) tab runs the same prediction, with the same parameters. The weights folder there is `model_path`.

## Required Parameters

| Parameter | Required | Type | Description |
|---|---|---|---|
| `model_path` | ✅ | `str` | Folder of the nnInteractive weights. |
| `img_path` | ✅ | `str` | Path to the input image (`.tif`, `.tiff` or `.png`). |
| `seg_path` | ✅ | `str` | Seeds (or any segmentation) the prompts are made from; each id is one class. |
| `output_folder` | ✅ | `str` | Folder for the results. |
| `prompt_type` | ✅ | `str` | `point`: points sampled from each seed region. `scribble`: the seed regions are the scribbles. |

## Optional Parameters

| Parameter | Default | Description |
|---|---|---|
| `device` | `cuda:0` | `cuda:0`, `cpu`, ... Falls back to CPU, with a warning, when CUDA is not available. |
| `init_seg_path` | none | A segmentation to start from, e.g. an earlier result to refine. |
| `return_per_class_masks` | `false` | Also save one binary mask per class, in `segs/`. |
| `save_prob_maps` | `false` | Also save the foreground probability of each class (float32, 0 to 1), in `prob/`. |
| `do_autozoom` | `true` | nnInteractive AutoZoom: when the prediction reaches the edge of the model patch, zoom out and refine. |
| `seed` | `0` | Random seed of the point sampling (not a SPROUT seed): the same value gives the same points. |
| `verbose` | `true` | Print progress. |

### Points (`prompt_type: point`)

| Parameter | Default | Description |
|---|---|---|
| `default_n_pos` | `1` | Positive points per class. |
| `default_n_neg` | `0` | Negative points per class. |
| `default_method` | `random` | Where the points go: `random`, `grid`, `kmeans` or `center_edge`. |
| `negative_from_bg` | `true` | Negative points may come from the background. |
| `negative_from_other_classes` | `true` | Negative points may come from the other classes. |
| `negative_per_other_class` | `false` | Sample negative points from each other class separately. |
| `class_config` | none | Per class: `n_pos`, `n_neg`, `method`, e.g. `{1: {n_pos: 3, n_neg: 1, method: grid}}`. |

### Scribbles (`prompt_type: scribble`)

| Parameter | Default | Description |
|---|---|---|
| `use_negative_scribble` | `false` | Also give the other classes as negative scribbles. |

### Seed ROI

Each class can be limited to the bounding box of its seed, plus a margin.

| Parameter | Default | Description |
|---|---|---|
| `seed_roi` | `off` | `off`. `postmask`: predict on the whole image, then clear each class outside its box (removes leaks into other parts). `crop`: nnInteractive sees only the box (faster, less memory). A warning is printed when a prediction reaches the edge of its box. |
| `seed_roi_margin` | `32` | Margin around the seed box: one value or `[z, y, x]`. |
| `seed_roi_margin_unit` | `voxel` | `voxel`, or `percent` of the seed box size on each axis. |

## Output

In `output_folder`:

- `<image>_segmentation.tif`: all classes, one id per class.
- `<image>_point_prompts.csv`: the points used (point prompts).
- `segs/<image>_class_<id>.tif` with `return_per_class_masks`; `prob/<image>_class_<id>_prob.tif` with `save_prob_maps`.
- `used_config.yaml`: the parameters of the run (not in batch mode).

## Batch (`--prompt --nninteractive --batch`)

```bash
sprout --prompt --nninteractive --batch --config path/to/batch_config.yaml
```

The batch config is the config above with `csv_path` in place of `img_path` and `seg_path`. The CSV has the columns `img_path` and `seg_path`, one row per image, and may set other parameters per image (a parameter goes in the CSV or in the YAML, not both). Each image gets a subfolder of `output_folder`, named after the image.

A template: [`../template/batch_nninteractive.yaml`](../template/batch_nninteractive.yaml) with [`../template/nninteractive_input.csv`](../template/nninteractive_input.csv) (or `sprout --template nninteractive --batch`).
