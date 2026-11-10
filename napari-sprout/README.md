# napari-sprout

The napari plugin of [SPROUT](https://github.com/EchanHe/SPROUT): multi-part segmentation of 2D and 3D greyscale images (e.g. CT, MRI) without training data.

- **Seeds**: thresholds and erosion split an image into parts; adaptive and sweep modes combine several settings.
- **Grow**: seeds grow back to the full shape of each part.
- **Edit**: split, merge, relabel and clean regions.
- **Prompt (AI)**: seeds as prompts for nnInteractive.
- **ROI & Boundary**: work inside a region; make boundary masks.

## Install

```bash
pip install "napari-sprout[all]"     # with napari and Qt
pip install napari-sprout            # napari already installed
```

For the Prompt (AI) tab also run `pip install "sprout-seg[ai]"` (with an NVIDIA GPU, install PyTorch for your CUDA version first).

## Use

Start napari and open **Plugins → SPROUT**. Videos: [SPROUT playlist](https://www.youtube.com/watch?v=4DHI1gs1lzc&list=PL8tsvVKJ9n6FjYYplo-IAwKBLxDh578aj). Docs and the command line (`sprout`): [github.com/EchanHe/SPROUT](https://github.com/EchanHe/SPROUT).

## Cite

Y. He, M. Camaiti, L.E. Roberts, J.M. Mulqueeney, E. Ioannou, M. Didziokas, A. Goswami. *SPROUT: A User-friendly, Scalable Toolkit for Multi-class Segmentation of Volumetric Images.* [bioRxiv 2024](https://www.biorxiv.org/content/10.1101/2024.11.22.624847v2.abstract).

MIT License.
