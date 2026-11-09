## 🧠 Adaptive Seed Generation Config Parameters (`--adaptive_seed`)

```bash
python sprout.py --adaptive_seed --config path/to/config.yaml
```

A template of the configuration file for adaptive seed generation can be found at: [`../template/make_adaptive_seed.yaml`](../template/make_adaptive_seed.yaml).

This mode generates seeds by either varying thresholds or erosion steps and tracking how segments split across steps. It generates a stable seed based on merging seeds generated through the steps with multiple structural criteria.

Similar to the `--seeds` mode, it requires a configuration file with parameters for seed generation.

This is the method of the paper. For many seeds fast, see [sweep adaptive seed](./config_sweep_adaptive_seed.md). With close thresholds, set `no_split_max_iter` above the number of thresholds, or the run may stop early.

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
| `min_size`                | ❌        | `int`           | Minimum size to keep a segment. Default: `1`.                                             |
| `min_split_ratio`         | ❌        | `float`         | Minimum area/volume ratio of a sub-region to its region to consider as split. Default: `0`.                 |
| `min_split_total_ratio`   | ❌        | `float`         | Minimum ratio of the sum of new sub-regions to  original region to count as valid split.               |
| `split_size_limit`        | ❌        | `list[2]`       | Only split segments whose size is within the specified min/max range.                     |
| `split_convex_hull_limit` | ❌        | `list[2]`       | Only split segments whose convex hull area/volume is within this range.                   |
| `init_segments`           | ❌        | `int`           | Number of components used to initialize, can be used for skip small fragments in the first step.                     |
| `last_segments`           | ❌        | `int`           | Optional constraint on number of segments in the final step.                         |
| `footprints`              | ❌        | `str` or `list` | Erosion kernel(s): `"ball"`, `"cube"`, `"ball_XY"`, `"X"`, etc. Each erosion step removes one voxel layer: `"ball"` = 3×3×3 cross (6-neighbourhood), `"cube"` = full 3×3×3 cube (26-neighbourhood). SPROUT sizes are radii (size 1 = 3×3×3), unlike skimage's `cube(n)` (edge length). Before Oct 2026, `"cube"` did not erode at all (a bug).       
| `save_every_iter`         | ❌        | `bool`          | Whether to save segmentation results at every iteration. Default: `True`.                 |
| `prune_every_iter`        | ❌        | `bool`          | After every intermediate step, remove regions smaller than `min_size`, keep the top-N and renumber the running seed. This changes the final result. If unset, it follows `save_every_iter`, which reproduces earlier versions (where saving implied pruning). Set it explicitly to make saving independent of the result. |
| `base_name`               | ❌        | `str`           | Name of the subfolder in `output_folder/` to save results. Defaults to image file name.   |
                    |


## Output

In `<output_folder>/<base_name>/`:

| File | Content |
|---|---|
| `FINAL_adaptive_seed.tif` | The seeds. Use it as `seg_path` of [growth](./config_grow.md). |
| `adaptive_seed_regions.csv` | One row per seed: `id`, `area`, the step where it began (`first_step` and its threshold or erosion), and how it `began` (below). |
| `adaptive_seed_steps.csv` | One row per step: its threshold or erosion, `regions` in it, how many seeds `split`, `shrunk` or were `dropped`, `new_seeds`, `pruned` (with `prune_every_iter`), `seeds_after` it, and a `note` when the run stopped early. |
| `INTER_*.tif` | With `save_every_iter: true`: the seed of every step, and the running result after it. |
| `config_<time>.yaml`, `output_dict_<time>.csv` | The config of this run, and its record of every split. |

In `adaptive_seed_regions.csv`, `began` is `first step` (a region of the first step), `separated (k of N)` (part k of a region that split into N parts), or `shrank (rest disappeared)` (a region with one part large enough to count, see `min_split_ratio`).

## Seed Generation Config Parameters in batch (`--adaptive_seed`)

Adaptive seed batch mode supports multi-image processing using a CSV file, following the same structure as seed generation batch mode. Each row must contain an `img_path`, and can override YAML defaults for parameters such as `thresholds`, `segments`, etc. For CSV format and parameter priority rules, refer to the [seed batch mode section](../docs/config_seed.md#seed-generation-config-parameters-in-batch---seeds).


```bash
python sprout.py --adaptive_seed --batch --config path/to/batch_config.yaml
```