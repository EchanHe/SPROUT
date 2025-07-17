## 🧠 Adaptive Seed Generation Config Parameters (`--adaptive_seed`)

```bash
python sprout.py --adaptive_seed --config path/to/config.yaml
```

A template of the configuration file for adaptive seed generation can be found at: [`../template/make_adaptive_seed.yaml`](../template/make_adaptive_seed.yaml).

This mode generates seeds by either varying thresholds or erosion steps and tracking how segments split across steps. It generates a stable seed based on merging seeds generated through the steps with multiple structural criteria.

Similar to the `--seeds` mode, it requires a configuration file with parameters for seed generation.

## Required Parameters

| Parameter         | Required | Type               | Description |
|-------------------|----------|--------------------|-------------|
| `img_path`        | ✅        | `str`              | Path to the input image file (must end with `.tif` or `.tiff`). |
| `thresholds`    | ✅        | `int` or `list` | A single threshold or a list. If a list, seeds are merged by increasing thresholds.           |
| `erosion_steps`   | ✅        | `int`              | Number of erosion iterations. Seeds will be saved after each step up to this number. |
| `segments`        | ✅        | `int`              | The number of largest connected components to retain during segmentation. |
| `output_folder`   | ✅        | `str`              |  Root directory for saving seeds and metadata. The final output will be saved in a subfolder named after `base_name`. If `base_name` is not set, the base name will default to the input image filename (without extension). |
| `num_threads`     | ✅        | `int`              | Number of threads to use during seed generation. Recommended to match the number of thresholds. |


## Optional Parameters
| Parameter                 | Required | Type            | Description                                                                               |
| ------------------------- | -------- | --------------- | ----------------------------------------------------------------------------------------- |
| `upper_thresholds`        | ❌        | `int` or `list` | Optional upper thresholds. If set, thresholding becomes `img >= threshold & img <= upper_threshold`.                   |
| `boundary_path`           | ❌        | `str`           | Path to a binary image used to restrict seed generation region.                           |
| `background`              | ❌        | `int`           | Background value (default: `0`).                                                          |
| `sort`                    | ❌        | `bool`          | Whether to sort segment labels by size (default: `True`).                                 |
| `no_split_max_iter`       | ❌        | `int`           | Early stopping: number of iterations without any new split before stopping. Default: `3`. |
| `min_size`                | ❌        | `int`           | Minimum size to keep a segment. Default: `5`.                                             |
| `min_split_ratio`         | ❌        | `float`         | Minimum area/volume ratio of a sub-region to its region to consider as split. Default: `0.01`.                 |
| `min_split_total_ratio`   | ❌        | `float`         | Minimum ratio of the sum of new sub-regions to  original region to count as valid split.               |
| `split_size_limit`        | ❌        | `list[2]`       | Only split segments whose size is within the specified min/max range.                     |
| `split_convex_hull_limit` | ❌        | `list[2]`       | Only split segments whose convex hull area/volume is within this range.                   |
| `init_segments`           | ❌        | `int`           | Number of components used to initialize, can be used for skip small fragments in the first step.                     |
| `last_segments`           | ❌        | `int`           | Optional constraint on number of segments in the final step.                         |
| `footprints`              | ❌        | `str` or `list` | Erosion kernel(s): `"ball"`, `"cube"`, `"ball_XY"`, `"X"`, etc.       
| `save_every_iter`         | ❌        | `bool`          | Whether to save segmentation results at every iteration. Default: `True`.                 |
| `base_name`               | ❌        | `str`           | Name of the subfolder in `output_folder/` to save results. Defaults to image file name.   |
                    |


## Seed Generation Config Parameters in batch (`--adaptive_seed`)

Adaptive seed batch mode supports multi-image processing using a CSV file, following the same structure as seed generation batch mode. Each row must contain an `img_path`, and can override YAML defaults for parameters such as `thresholds`, `segments`, etc. For CSV format and parameter priority rules, refer to the [seed batch mode section](../docs/config_seed.md#seed-generation-config-parameters-in-batch---seeds).


```bash
python sprout.py --adaptive_seed --batch --config path/to/batch_config.yaml
```