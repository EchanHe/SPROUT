## Grow Config Parameters (`--grow`)

```bash
sprout --grow --config path/to/config.yaml
```

A template of the configuration file for grow can be found at: [`../template/make_grow.yaml`](../template/make_grow.yaml).

This configuration file defines the arguments used for growing labeled seeds in SPROUT. It grows labels based on intensity thresholds and morphological dilation operations. Below is a reference table describing each parameter’s usage, type, and default behavior.

### Required Parameters

| Parameter        | Required | Type            | Description                                                                          |
| ---------------- | -------- | --------------- | ------------------------------------------------------------------------------------ |
| `img_path`       | ✅        | `str`           | Path to the input image file (must end with `.tif` or `.tiff`).                      |
| `seg_path`       | ✅        | `str`           | Path to the seed/segmentation image to grow (must end with `.tif` or `.tiff`), such as `FINAL_adaptive_seed.tif` or `FINAL_sweep_seed.tif`. |
| `thresholds`     | ✅        | `int` or `list` | List of intensity lower thresholds to grow along. Should be ordered from high to low, to gradually allow growing more area/volume     |
| `dilation_steps` | ✅        | `int` or `list` | Number of dilation steps at each threshold. Should match the length of `thresholds`. Or set as an int, it will propagate to match length of `thresholds`|
| `touch_rule`     | ✅        | `str`           | Rule when a label touches another. Currently supports `"stop"` or `"overwrite"`.                 |
| `output_folder`  | ✅        | `str`           |  Root directory for saving growth results and metadata. The final output will be saved in a subfolder named after `base_name`. If `base_name` is not set, the base name will default to the input image filename (without extension).  |
| `num_threads`    | ✅        | `int`           | Number of threads used during growth.                                                |


### Optional Parameters
| Parameter                  | Required | Type            | Description                                                                                        |
| -------------------------- | -------- | --------------- | -------------------------------------------------------------------------------------------------- |
| `upper_thresholds`         | ❌        | `int` or `list` | Optional upper bounds for each threshold. Must match length if provided.                           |
| `boundary_path`            | ❌        | `str`           | Optional binary mask image that limits growth region.                                              |
| `grow_to_end`              | ❌        | `bool`          | If `True`, ignore `dilation_steps` and grow as close to thresholded mask as possible.              |
| `save_every_n_iters`       | ❌        | `int`           | Save intermediate result every N iterations. Default is `None` (disabled).                         |
| `final_grow_output_folder` | ❌        | `str`           | Separate folder for saving final grown result. Defaults to `output_folder`.                        |
| `workspace`         | ❌        | `str`              | Root directory. If set, all relative paths will be joined with it. Default is empty string `""`. |
| `base_name`                | ❌        | `str`           | Prefix name for sub-output folder. Defaults to base name of `img_path`.                                 |
| `use_simple_naming`        | ❌        | `bool`          | If `True`, use simplified output file names.                                                       |
| `to_grow_ids`              | ❌        | `list[int]`     | List of label/class IDs to grow. Use `~` or omit to grow all.                                            |
| `is_sort`                  | ❌        | `bool`          | Whether to sort grown segments by size.                                   |
| `no_growth_max_iter`       | ❌        | `int`           |Early stopping: Maximum number of consecutive dilation steps with no growth. Default is `3`. |
| `min_growth_size`          | ❌        | `int`           |Early stopping: Minimum number of new pixels for a step to count as “growth”. Default is `50`.                     |

### Optional Mesh Parameters

We are currently using napari to render our results, so if you only want to visualize the results, you can skip these parameters.

| Parameter          | Required | Type   | Description                                                                                     |
| ------------------ | -------- | ------ | ----------------------------------------------------------------------------------------------- |
| `is_make_meshes`    | ❌        | `bool`             | Whether to save surface meshes using the Marching Cubes algorithm. Default is `False`. |
| `downsample_scale`  | ❌        | `int`              | Downsampling factor for mesh generation. Default is `10`. |
| `step_size`         | ❌        | `int`              | Step size for Marching Cubes. Higher values produce simpler meshes. Default is `1`. |


## Output

In `<output_folder>/<base_name>/`:

| File | Content |
|---|---|
| `FINAL_GROW_<base_name>.tif` | The result, numbered by size with `is_sort`. In `final_grow_output_folder` if it is set. |
| `INTER_<base_name>_<threshold>_<upper>_<n>.tif` | The result after `n` dilations: every `save_every_n_iters`, at the end of each threshold, and at an early stop. |
| `grow_log_<base_name>.csv` | One row per `INTER_*.tif`: its size, the size the threshold allows, and its full path. |
| `grow_log_<base_name>.png` | A plot of that table (needs matplotlib). |
| `.ply` files | With `is_make_meshes`: meshes of the results. |
| `used_config.yaml` | Every parameter of the run, the defaults filled in, and the SPROUT version. It is also a config: `sprout --grow --config used_config.yaml` runs it again (relative paths are from the folder you run in). |

A run into a folder with earlier results overwrites files of the same name (a warning says so). To keep a result, use another `output_folder`.

## Grow Config Parameters in batch (`--grow --batch`)

Batch mode allows you to grow segmentations for multiple images in a single run using a CSV file. Each row in the CSV corresponds to one image and one segmentation, and may optionally override additional parameters.

```bash
sprout --grow --batch --config path/to/batch_config.yaml
```

Templates of the configuration file and the csv file for grow can be found at: [`../template/batch_grow.yaml`](../template/batch_grow.yaml) and [../template/grow_input.csv](../template/grow_input.csv).


### New required YAML parameter for batch mode:

| Parameter  | Required | Type  | Description                                                                      |
| ---------- | -------- | ----- | -------------------------------------------------------------------------------- |
| `csv_path` | ✅        | `str` | Path to a CSV file. Must contain at least the `img_path` and `seg_path` columns. |


In addition, all **required grow parameters** (`thresholds`, `dilation_steps`, `touch_rule`, etc.) must be either: : Globally defined in the YAML file, **or** Provided per image in the CSV file.

The table `<output_folder>/<csv name>_results.csv` has one row per image, with an `error` column for a row that failed. Its `final_output_path` column has the result of each row.

### Example

If you want to grow seeds for multiple image/segmentation pairs using the same global dilation parameters, but different thresholds per image, you can define:

#### `batch_grow_config.yaml`

```yaml
csv_path: my_batch_grow.csv
output_folder: ./results/batch_grow/
dilation_steps: [20,20,5,5]
touch_rule: stop
num_threads: 4

```

#### `my_batch_grow.csv`

```csv
img_path,seg_path,thresholds
./data/img1.tif,./seeds/img1_seed.tif,[180,160,140,120]
./data/img2.tif,./seeds/img2_seed.tif,[2000,1800,1600,1400]

```

This setup will grow the provided seed masks using their own `thresholds` while keeping `dilation_steps`, `touch_rule`, and `output_folder` from the YAML file.