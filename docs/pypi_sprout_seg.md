# sprout-seg

The core and command line of [SPROUT](https://github.com/EchanHe/SPROUT): multi-part segmentation of 2D and 3D greyscale images (e.g. CT, MRI) without training data. For the napari plugin, see [napari-sprout](https://pypi.org/project/napari-sprout/).

- **Seeds**: thresholds and erosion split an image into parts; adaptive and sweep modes combine several settings.
- **Grow**: seeds grow back to the full shape of each part.
- **Prompts**: seeds as prompts for nnInteractive or SAM.
- **Batch**: many images from one CSV. Runs on CPU.

## Install

```bash
pip install sprout-seg               # command line only (e.g. on a cluster)
pip install "napari-sprout[all]"     # with the napari plugin, napari and Qt
```

Extras: `sprout-seg[ai]` (nnInteractive; with an NVIDIA GPU, install PyTorch for your CUDA version first), `sprout-seg[mesh]` (meshes).

## Use

Every step is a YAML config ([templates](https://github.com/EchanHe/SPROUT/tree/main/template)):

```bash
sprout --template seeds          # writes make_seeds.yaml here
sprout --seeds --config make_seeds.yaml
sprout --sweep_adaptive --config sweep_adaptive_seed.yaml
sprout --adaptive_seed --config make_adaptive_seed.yaml
sprout --grow --config make_grow.yaml
sprout --seeds --batch --config batch_seeds.yaml
sprout --help
```

Docs, demo data and videos: [github.com/EchanHe/SPROUT](https://github.com/EchanHe/SPROUT).

## Cite

Y. He, M. Camaiti, L.E. Roberts, J.M. Mulqueeney, E. Ioannou, M. Didziokas, A. Goswami. *SPROUT: A User-friendly, Scalable Toolkit for Multi-class Segmentation of Volumetric Images.* [bioRxiv 2024](https://www.biorxiv.org/content/10.1101/2024.11.22.624847v2.abstract).

MIT License.
