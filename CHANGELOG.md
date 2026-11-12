# Changelog

## Unreleased

### New
- `sprout --template <step>` (add `--batch`) writes the YAML template of a step into the current folder: no clone needed.
- Template for batch nnInteractive (`template/batch_nninteractive.yaml`).

### Results that change
- napari Sweep with a boundary (no ROI) gave wrong seeds in 1.1.0; it now gives the YAML result.
- napari Edit > Merge: ids above 255 no longer wrap in a uint8 layer.
- Sweep loose to strict: a seed whose region is left out of a step (as by `segments`) and comes back later splits again; its later splits were dropped.

### Changed
- napari: layers of another shape are greyed out, "(shape mismatch)", in Seeds, Grow, Prompt (AI) and Edit > Merge, and Import YAML does not pick them.
- napari Edit > Merge: rows Into / Add / Added IDs (Keep, Shift up).
- `sprout`: a missing config or option gives a short message and where to find templates and docs, not a traceback.
- Templates: batch adaptive seed has the `min_size` and `min_split_ratio` of the single one (1 and 0, were 5 and 0.01); nnInteractive `default_method` is `random`, as in the docs and napari (was `grid`).

### Fixed
- The batch sweep template's CSV (`sweep_adaptive_seeds_input.csv`) was missing.

## 1.1.0 (2026-11-10)

### New
- **pip install**: `pip install "napari-sprout[all]"` (plugin, napari and Qt) or `pip install sprout-seg` (command line only). Extras: `sprout-seg[ai]` (nnInteractive), `sprout-seg[mesh]`.
- The `sprout` command works in any folder (`python sprout.py` still works in a clone).
- napari: **Prompt (AI)** tab: seeds → nnInteractive (2.6, or 1.1.3), as scribbles or points.
- napari: **ROI & Boundary** tab: run any tab inside a region of interest; make boundary masks (top-hat).
- napari: Threshold Preview; progress and Stop for Seeds, Grow and Boundary.
- YAML: `seed` (nnInteractive pipeline, default 0); `prune_every_iter` (adaptive seed).
- **Sweep adaptive seed** (`--sweep_adaptive`, napari Seeds tab `Sweep (Erosion)` / `Sweep (Thresholds)`): one seed per threshold or erosion step, combined into `FINAL_sweep_seed.tif`, with `sweep_regions.csv` and `sweep_steps.csv`. See `docs/config_sweep_adaptive_seed.md`.
- Adaptive seed: `adaptive_seed_regions.csv` and `adaptive_seed_steps.csv` say how each seed began and what each step did (same seeds).
- Float images: decimal thresholds in seeds, adaptive seed, sweep and grow, in the YAML and in napari, for example on probability maps.
- Sweep: `memory_limit_gb` caps the memory of large scans (steps on disk, fewer threads; same result).

### Results that change
- Grow: multi-threaded runs give the single-thread result; seed voxels outside the allowed range do not grow.
- Label dtype follows the largest id (ids above 255 no longer wrap and merge).
- `cube` / `square` footprints: size is the radius (cube erosion had no effect).
- Adaptive seed: deterministic id order; the boundary also applies in threshold mode.
- nnInteractive pipeline: fixed random seed (earlier runs were not reproducible).
- napari Edit: "Split all" + "Sort by size" now splits every region.
- Grow: `touch_rule: overwrite` stays inside the threshold (it left a one-voxel ring outside).
- A non-integer threshold is no longer cut: 150.7 was 150.

### Changed
- napari: Edit "all regions" operations are much faster; Undo restores the layer it changed.
- napari: result layers are at least uint16; files keep the smallest dtype.
- napari: the old Prompts and Load docks were removed.
- napari: Edit and Label Mapping results keep the scale and translate of their layer.
- napari: Seeds and Grow Export write the file of a layer, and Import finds the layer by it.
- Adaptive seed is much faster with many seeds, with the same results.
- Adaptive seed is about 2x faster on large scans with small seeds, with the same results.
- A footprint for the other dimension is an error (it did nothing). A negative label id is an error (it wrapped around).
- `num_threads` above the CPU cores falls back to the cores, with a warning (it was an error).
- `make_*.py` and `batch_*.py` moved into `sprout_core/` (`from sprout_core import make_seeds`); `python make_seeds.py` no longer runs, use `sprout` or `python sprout.py`.
- The plugin folder is `napari-sprout/` (was `napari_sprout/`); the plugin no longer edits `sys.path`.
- open3d is no longer listed (no code used it).
- Works with numpy 2, pandas 3, scikit-image 0.26 and napari 0.10 (Qt6). Erosion and dilation use scipy (scikit-image removes `binary_erosion`, `cube` and others), with the same results.

## 1.0.0 (2026-10-15)
First release.
