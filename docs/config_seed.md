## 🌰 Seed Generation Config Parameters (`--seeds`)

```bash
python sprout.py --seeds --config path/to/config.yaml
```

A template of the configuration file for seed generation can be found at: [`../template/make_seeds.yaml`](../template/make_seeds.yaml).


This configuration file defines the arguments used for seed generation in SPROUT. Below is a reference table describing each parameter's usage, type, and default behavior.

### Required Parameters

| Parameter         | Required | Type               | Description |
|-------------------|----------|--------------------|-------------|
| `img_path`        | ✅        | `str`              | Path to the input image file (must end with `.tif` or `.tiff`). |
| `thresholds`      | ✅        | `int` or `list`    | A list (or single value) of lower thresholds for the initial segmentation. Seeds will be generated for each threshold. |
| `erosion_steps`   | ✅        | `int`              | Number of erosion iterations. Seeds will be saved after each step up to this number. |
| `segments`        | ✅        | `int`              | The number of largest connected components to retain during segmentation. |
| `output_folder`   | ✅        | `str`              |  Root directory for saving seeds and metadata. The final output will be saved in a subfolder named after `base_name`. If `base_name` is not set, the base name will default to the input image filename (without extension). |
| `num_threads`     | ✅        | `int`              | Number of threads to use during seed generation. Recommended to match the number of thresholds. |

### Optional Parameters

| Parameter           | Required | Type               | Description |
|---------------------|----------|--------------------|-------------|
| `upper_thresholds`  | ❌        | `int` or `list`    | Optional upper thresholds. If set, thresholding becomes `img >= threshold & img <= upper_threshold`. |
| `boundary_path`     | ❌        | `str`              | Optional path to a binary boundary image. Used create extra backgrounds in binary to explicitly create separations |
| `workspace`         | ❌        | `str`              | Root directory. If set, all relative paths will be joined with it. Default is empty string `""`. |
| `base_name`                | ❌        | `str`           | Prefix name for sub-output folder. Defaults to base name of `img_path`.                                 |
| `footprints`        | ❌        | `str` or `list`    | Shape of structuring element for erosion. Can be `"ball"`, `"cube"`, `"X"`, `"Y"`, `"Z"`, or plane-specific versions like `"ball_XY"`. |



### Optional Mesh Parameters

We are currently using napari to render our results, so if you only want to visualize the results, you can skip these parameters.

| Parameter          | Required | Type   | Description                                                                                     |
| ------------------ | -------- | ------ | ----------------------------------------------------------------------------------------------- |
| `is_make_meshes`    | ❌        | `bool`             | Whether to save surface meshes using the Marching Cubes algorithm. Default is `False`. |
| `downsample_scale`  | ❌        | `int`              | Downsampling factor for mesh generation. Default is `10`. |
| `step_size`         | ❌        | `int`              | Step size for Marching Cubes. Higher values produce simpler meshes. Default is `1`. |

---




## Seed Generation Config Parameters in batch (`--seeds`)

Batch mode allows you to generate seeds for multiple images in a single run using a CSV file. Each row in the CSV corresponds to an individual image and can optionally override parameters on a per-image basis.

```bash
python sprout.py --seeds --batch --config path/to/batch_config.yaml
```

Templates of the configuration file and the csv file for grow can be found at: [`../template/make_seeds.yaml`](../template/batch_seeds.yaml) and [../template/seeds_input.csv](../template/seeds_input.csv]).


### New required YAML parameter for batch mode:

| Parameter       | Required | Type  | Description                                                      |
| --------------- | -------- | ----- | ---------------------------------------------------------------- |
| `csv_path`      | ✅        | `str` | Path to a CSV file. Must contain at least the `img_path` column. |

In addition, all **required seed parameters** (`thresholds`, `erosion_steps`, `segments`, etc.) must be either: Globally defined in the YAML file, **or** Provided per image in the CSV file


### ✅ Example

If you want to generate seeds for multiple images with different thresholds but with other parameters same across images.

You can define a batch configuration file and a CSV file as follows:

####  `batch_config.yaml`

```yaml
csv_path: my_batch.csv
output_folder: ./results/batch_seeds/
erosion_steps: 3
segments: 30
...
```
#### `my_batch.csv`

```csv
img_path,thresholds
./data/img1.tif,[130,140]
./data/img3.tif,[150,160]

```