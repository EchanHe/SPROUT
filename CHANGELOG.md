# Changelog

## 1.1.0 (unreleased)

### New
- napari: **Prompt (AI)** tab: seeds → nnInteractive (2.6, or 1.1.3), as scribbles or points.
- napari: **ROI & Boundary** tab: run any tab inside a region of interest; make boundary masks (top-hat).
- napari: Threshold Preview; progress and Stop for Seeds, Grow and Boundary.
- YAML: `seed` (nnInteractive pipeline, default 0); `prune_every_iter` (adaptive seed).

### Results that change
- Grow: multi-threaded runs give the single-thread result; seed voxels outside the allowed range do not grow.
- Label dtype follows the largest id (ids above 255 no longer wrap and merge).
- `cube` / `square` footprints: size is the radius (cube erosion had no effect).
- Adaptive seed: deterministic id order; the boundary also applies in threshold mode.
- nnInteractive pipeline: fixed random seed (earlier runs were not reproducible).
- napari Edit: "Split all" + "Sort by size" now splits every region.

### Changed
- napari: Edit "all regions" operations are much faster; Undo restores the layer it changed.
- napari: result layers are at least uint16; files keep the smallest dtype.
- napari: the old Prompts and Load docks were removed.

## 1.0.0 (2026-10-15)
First release.
