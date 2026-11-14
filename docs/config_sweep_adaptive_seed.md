## Sweep Adaptive Seed Config Parameters (`--sweep_adaptive`)

```bash
sprout --sweep_adaptive --config path/to/config.yaml
```

A template of the configuration file can be found at: [`../template/sweep_adaptive_seed.yaml`](../template/sweep_adaptive_seed.yaml)

Sweep adaptive seed makes one seed per threshold (or per erosion step), then combines them into one final seed.

With `direction: strict_to_loose` (the default), the steps are combined from the strictest to the loosest:

- a region that touches no seed emerges as a new seed;
- a region that touches one seed makes that seed grow;
- a region that touches several seeds locks them, so they stay apart.

With `direction: loose_to_strict`, the steps are combined the other way: seeds start large and split or shrink.

### Which adaptive seed should I use?

| | Sweep adaptive seed | Adaptive seed (`--adaptive_seed`) |
|---|---|---|
| Method | Every step is used | The method of the paper: follows splits, stops after `no_split_max_iter` steps without a split |
| Splits | Every split is kept, small pieces too (raise `min_size`) | Each split is checked (`min_split_ratio` and the split limits) |
| Many seeds | Run time hardly changes | Run time grows with the number of seeds |
| A long list of close thresholds | Every threshold is used | Set `no_split_max_iter` above the number of thresholds, or it may stop early |
| Memory | All steps at once, see [Large images](#large-images) | One step at a time |
| napari | Seeds tab, Method `Sweep (Erosion)` or `Sweep (Thresholds)` | Seeds tab, Method `Adaptive (Erosion)` or `Adaptive (Thresholds)` |

Use sweep adaptive seed when you want many seeds fast. More thresholds give more seeds, also small ones from noise: `min_size` drops them. Use adaptive seed for fewer, cleaner seeds, or to reproduce results of the paper. The two give different seeds, so compare them on your own data.

### In napari

In the Seeds tab, set Method to `Sweep (Thresholds)` (two or more threshold rows, in descending or ascending order; one step per row) or `Sweep (Erosion)` (one row; one step per erosion step). Segments and the footprints work as here, Threads too for `Sweep (Thresholds)`, and `Min segment size` under the advanced parameters is `min_size`.

napari runs `strict_to_loose` and adds one layer, `<image>_FINAL_sweep_seed`, which the Grow tab picks up. The files below are written to the output folder. The other parameters of this page are for YAML runs.

### Large images

By default all steps stay in memory: about 1 byte per voxel per step (2 if a step keeps more than 255 regions), plus up to 7 bytes per voxel per thread. If a run warns or runs out of memory, set `memory_limit_gb`. Steps that do not fit are then read back from disk, and fewer threads run at once: slower, same result. The least that works is about 7 bytes per voxel (one thread, and the combine): 7 GB for a scan of 1,000 million voxels.

## Required Parameters

| Parameter | Required | Type | Description |
|---|---|---|---|
| `img_path` | ✅ | `str` | Path to the input image (`.tif` or `.tiff`). |
| `output_folder` | ✅ | `str` | Folder for the results. |
| `thresholds` | ✅ | `int`, `float` or `list` | Several values: one step per threshold, in descending or ascending order. One value: an erosion sweep, one step per erosion. Decimals work on float images, such as probability maps. |
| `erosion_steps` | ✅ | `int` | Several thresholds: erosions applied at each step. One threshold: the number of erosion steps. |
| `segments` | ✅ | `int` | Keep the N largest regions of each step. |
| `num_threads` | ✅ | `int` | Threads for the steps of a threshold sweep. The result does not depend on it. |

## Optional Parameters

| Parameter | Default | Description |
|---|---|---|
| `direction` | `strict_to_loose` | `strict_to_loose`, `loose_to_strict`, or `both` to write both results. |
| `min_size` | `1` | Regions smaller than this are ignored. This has a large effect on the number of seeds. |
| `top_n` | none | Keep only the N largest seeds of the result. |
| `input_mode` | `auto` | `label`: each id of a step is one region. `binary`: find the connected components of a step (touching at a corner counts, as everywhere in SPROUT). |
| `upper_thresholds` | none | One value per threshold, for a range: `threshold <= image <= upper`. An erosion sweep takes one value. |
| `boundary_path` | none | A boundary mask: its voxels are excluded. |
| `footprints` | `ball` | Erosion footprint, see [seed generation](./config_seed.md). |
| `base_name` | image name | Name of the result folder. |
| `workspace` | none | Prepended to all relative paths. |
| `save_every_iter` | `false` | Save the combined seed after every step. |
| `memory_limit_gb` | none: all in memory | Memory for the steps, in GB; see [Large images](#large-images). The run prints what it chose. |
| `verbose` | `true` | Print progress. |
| `seeds_folder` | none | Skip the sweep and use the `step_*.tif` files of this folder (the `steps` folder of an earlier run). A folder that does not exist is an error. |
| `seeds_sweep_direction` | from the file names | Order of the files in `seeds_folder`: `s2l` or `l2s`. Needed only for files with other names. |

Only for `strict_to_loose`:

| Parameter | Default | Description |
|---|---|---|
| `s2l_min_current_coverage` | none | How much of a current seed a new region must cover to grow it. |
| `s2l_max_emerge_coverage` | none | How much of a new region may overlap seeds if it is to emerge as a new seed. |
| `s2l_mask_by_last_seed` | `false` | Keep the result inside the loosest step. |

Only for `loose_to_strict`:

| Parameter | Default | Description |
|---|---|---|
| `l2s_min_new_coverage` | `0.5` | How much of a region must overlap a seed to be assigned to it. |
| `l2s_max_new_coverage` | `0.05` | How little of a region may overlap seeds if it is to count as new. |
| `l2s_keep_disappeared` | `true` | Keep the last shape of seeds that disappear at a stricter step. |

## Output

In `<output_folder>/<base_name>/`:

| File | Content |
|---|---|
| `FINAL_sweep_seed.tif` | The seeds, numbered by size (1 is the largest); uint16 (uint32 above 65,535 seeds). Use it as `seg_path` of [growth](./config_grow.md). |
| `sweep_regions.csv` | One row per seed: `id`, `area`, `first_step` and `last_step` (the numbers of the step files), the settings that change between steps at those steps (for example `first_threshold`, `last_threshold`), and `began`, `ended`, `at_step` (below). |
| `sweep_steps.csv` | One row per step, in the order of the combine: `step`, its settings, `regions` in it, `new_seeds` that started there, `seeds_after` it (before `top_n`), and how many of its new seeds are `in_final`. |
| `sweep_metadata.json` | The direction, the steps, and all that is known about each seed. Ids inside a seed's entry, such as `parent_seed_id`, are `original_seed_id` values. |
| `used_config.yaml` | Every parameter of the run, the defaults filled in, and the SPROUT version. It is also a config: `sprout --sweep_adaptive --config used_config.yaml` runs it again (relative paths are from the folder you run in). |
| `steps/` | The mask of each step, for example `step_000_thre_230_None_ero_0.tif`. |
| `intermediate/<direction>/after_step_NNN.tif` | With `save_every_iter: true`: the seeds after each step. |

With `direction: both`, the seeds, the two CSV files and the metadata are written once per direction, for example `FINAL_sweep_seed_strict_to_loose.tif` and `FINAL_sweep_seed_loose_to_strict.tif`.

In `sweep_regions.csv`:

| Column | Value | Meaning |
|---|---|---|
| `began` | `first step` | The seed is a region of the first step of the combine. |
| | `emerged` | A region that touched no seed. |
| | `split from N` | Split off seed N (loose to strict); `split from a seed not kept` if N was dropped by `top_n`. |
| `ended` | `locked` | At step `at_step` the seed was in one region with another seed, so both keep their shape of the step before. |
| | `never locked` | Grew until the last step (strict to loose). |
| | `disappeared` | At step `at_step` no region was left of it; its last shape is kept (`l2s_keep_disappeared`). |
| | `reached the end` | Still there at the last step (loose to strict). |

## Batch (`--sweep_adaptive --batch`)

```bash
sprout --sweep_adaptive --batch --config path/to/batch_config.yaml
```

A template can be found at: [`../template/batch_sweep_adaptive_seed.yaml`](../template/batch_sweep_adaptive_seed.yaml)

The YAML has the parameters above and `csv_path`, a CSV file with one row per image and an `img_path` column.

Any other column of the CSV overrides the YAML for that row. A blank cell keeps the value of the YAML. A row that fails does not stop the batch.

The table `<output_folder>/<csv name>_results.csv` has one row per image, with an `error` column for a row that failed. Its `final_output_path` column has the seeds of each row.
