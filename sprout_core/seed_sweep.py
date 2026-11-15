"""Generate an ordered sweep of segmentation masks from a raw image.

Two modes, auto-detected from the YAML / function arguments:
  - threshold mode  (thresholds is a list with >1 element):
      For each threshold, apply threshold + fixed erosion_steps, save one TIFF.
      Steps are independent so can be parallelised with ThreadPoolExecutor.
  - erosion mode  (thresholds is a single int or list of one element):
      Apply threshold once, then erode progressively; each step depends on
      the previous eroded image, so it runs sequentially.

Output files are named:
  step_{index:03d}_thre_{lower}_{upper}_ero_{ero}.tif
sorted by step index so the combine stage can load them in order.
"""

import os
import re
import sys
import time
import yaml
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tifffile import TiffFile, imread, imwrite

import sprout_core.sprout_core as sprout_core
import sprout_core.config_core as config_core


STEP_NAME = re.compile(r"step_\d+_thre_(.+)_(.+)_ero_(\d+)\.tif$")


def _apply_threshold(img, lower, upper):
    if upper is not None:
        return (img >= lower) & (img <= upper)
    return img >= lower


def _apply_erosions(mask, footprint_list, n_steps):
    for i in range(n_steps):
        mask = sprout_core.erosion_binary_img_on_sub(
            mask, kernal_size=1, footprint=footprint_list[i]
        )
    return mask


def _in_memory(seed):
    """The step as the combine keeps it in memory: uint8 when its ids fit (as with segments
    up to 255), else as it is. The file of the step stays uint16; the ids are the same."""
    return sprout_core.cast_labels(seed, np.uint8, copy=False)


def _save_step(seed, output_folder, step_index, lower, upper, ero_steps):
    fname = f"step_{step_index:03d}_thre_{lower}_{upper}_ero_{ero_steps}.tif"
    path = Path(output_folder) / fname
    imwrite(str(path), seed, compression="zlib")
    return path


def _resolve_segments_list(segments, n_steps):
    """Expand segments to a per-step list and validate length."""
    if isinstance(segments, int):
        return [segments] * n_steps
    if isinstance(segments, list):
        if len(segments) != n_steps:
            raise ValueError(
                f"segments list length ({len(segments)}) must equal number of steps ({n_steps})."
            )
        return [int(s) for s in segments]
    raise ValueError(f"segments must be an int or list of ints, got {type(segments)}.")


def _generate_one_threshold_step(img, lower, upper, footprint_list, erosion_steps,
                                  segments, step_index, output_folder, boundary=None, uint16_labels=True):
    """Generate and save one seed for a single threshold (threshold mode)."""
    mask = _apply_threshold(img, lower, upper)
    if boundary is not None:
        mask[boundary] = False
    mask = _apply_erosions(mask, footprint_list, erosion_steps)
    seed, _ = sprout_core.get_ccomps_with_size_order(mask, segments, uint16=uint16_labels)
    seed = sprout_core.cast_labels(seed, np.uint16, copy=False)
    path = _save_step(seed, output_folder, step_index, lower, upper, erosion_steps)
    return step_index, path, _in_memory(seed)


class StepFiles(Sequence):
    """Steps saved by a sweep, read from disk only when one is used.

    The combine uses one step at a time, so a scan whose steps do not fit in memory together can
    still be combined. A slice is another StepFiles: seeds[::-1] reads nothing.
    """

    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return StepFiles(self.paths[index])
        return imread(str(self.paths[index]))

    def shape_of(self, index):
        """Shape of a step from its file header, without reading its voxels."""
        with TiffFile(str(self.paths[index])) as tif:
            return tuple(tif.series[0].shape)


def _single_upper(upper_thresholds):
    """The upper threshold of an erosion sweep, given as one value or a list of one."""
    if upper_thresholds is None or isinstance(upper_thresholds, (int, float)):
        return upper_thresholds
    if len(upper_thresholds) != 1:
        raise ValueError(f"An erosion sweep has one threshold, so it takes one upper threshold. Got {upper_thresholds}.")
    return upper_thresholds[0]


def _detect_threshold_direction(thresholds, upper_thresholds):
    """Detect sweep direction from threshold ordering.

    s2l (strict-to-loose): lower is non-increasing (↓) so range widens over steps.
    l2s (loose-to-strict): lower is non-decreasing (↑) so range narrows over steps.
    When lower is constant, decide by the upper thresholds; default "s2l".
    Raises ValueError on a non-monotonic sequence (neither consistently
    increasing nor decreasing), which has no well-defined sweep direction.
    """
    n = len(thresholds)
    if n <= 1:
        return "s2l"

    lower_decreases = any(thresholds[i] > thresholds[i + 1] for i in range(n - 1))
    lower_increases = any(thresholds[i] < thresholds[i + 1] for i in range(n - 1))

    if lower_decreases and lower_increases:
        raise ValueError(
            "thresholds must be monotonic: all non-increasing (s2l) or all "
            f"non-decreasing (l2s). Got a non-monotonic sequence: {thresholds}."
        )
    if lower_decreases:
        return "s2l"
    if lower_increases:
        return "l2s"

    # lower is constant — decide by upper thresholds
    if upper_thresholds is not None:
        valid_ups = [u for u in upper_thresholds if u is not None]
        if len(valid_ups) > 1:
            upper_increases = any(valid_ups[i] < valid_ups[i + 1] for i in range(len(valid_ups) - 1))
            upper_decreases = any(valid_ups[i] > valid_ups[i + 1] for i in range(len(valid_ups) - 1))
            if upper_increases and upper_decreases:
                raise ValueError(
                    "upper_thresholds must be monotonic when lower thresholds are "
                    f"constant. Got a non-monotonic sequence: {upper_thresholds}."
                )
            if upper_increases:
                return "s2l"
            if upper_decreases:
                return "l2s"

    return "s2l"


def detect_direction_from_files(paths):
    """Sweep direction of saved step files, read from their names.

    Raises ValueError on a name that is not of a step file, and on files that
    cannot come from one sweep.
    """
    steps = []
    for path in paths:
        match = STEP_NAME.search(Path(path).name)
        if match is None:
            raise ValueError(f"Cannot read the sweep direction from {Path(path).name}: set seeds_sweep_direction.")
        lower, upper, erosion = match.groups()
        steps.append((float(lower), None if upper == "None" else float(upper), int(erosion)))
    lowers, uppers, erosions = (list(values) for values in zip(*steps))
    if len(set(erosions)) == 1:
        return _detect_threshold_direction(lowers, uppers)
    if len(set(lowers)) > 1 or erosions != sorted(erosions):
        raise ValueError(f"Step files must be monotonic, they look like files of two runs: {[Path(p).name for p in paths]}")
    return "l2s"


def _boundary_mask(img, boundary):
    """The boundary as a bool mask of the image's shape (0/1 labels would index, not mask)."""
    if boundary is None:
        return None
    boundary = np.asarray(boundary)
    if boundary.shape != np.shape(img):
        raise ValueError(f"Image and boundary must have the same shape, got {np.shape(img)} and {boundary.shape}.")
    return config_core.check_and_cast_boundary(boundary)


def seed_sweep_threshold(
    img,
    output_folder,
    thresholds,
    upper_thresholds=None,
    erosion_steps=0,
    segments=500,
    boundary=None,
    footprints=None,
    num_threads=1,
    verbose=True,
    seeds_out=None,
    cancel=None,
    step=None,
):
    """Threshold sweep: each threshold → one TIFF, steps are parallel.

    Returns (ordered_paths, sweep_direction) where sweep_direction is "s2l" or "l2s".
    s2l: lower ↓ / upper ↑ — step_000 is strictest (range widens over steps).
    l2s: lower ↑ / upper ↓ — step_000 is loosest (range narrows over steps).
    Direction is auto-detected from threshold ordering.
    cancel: checked before every step, raises Cancelled. step: called after every step.
    """
    boundary = _boundary_mask(img, boundary)
    sweep_direction = _detect_threshold_direction(thresholds, upper_thresholds)
    reverse = (sweep_direction == "s2l")
    thresholds, upper_thresholds = config_core.check_and_assign_thresholds(
        thresholds, upper_thresholds, reverse=reverse
    )
    footprint_list = config_core.check_and_assign_footprint(footprints, erosion_steps) if erosion_steps > 0 else []
    segments_list = _resolve_segments_list(segments, len(thresholds))

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    if verbose:
        print(
            f"[seed_sweep] threshold mode: {len(thresholds)} steps, "
            f"erosion_steps={erosion_steps}, segments={segments}, threads={num_threads}, "
            f"sweep_direction={sweep_direction}"
        )

    ordered_paths = [None] * len(thresholds)
    if seeds_out is not None:
        seeds_out[:] = [None] * len(thresholds)
    uint16_labels = sprout_core.Uint16Labels()       # one for all the steps

    def _worker(args):
        idx, lower, upper = args
        sprout_core.check_cancel(cancel)
        done = _generate_one_threshold_step(
            img, lower, upper, footprint_list, erosion_steps,
            segments_list[idx], idx, output_folder, boundary, uint16_labels
        )
        if seeds_out is None:
            done = done[:2] + (None,)       # a future keeps its result: do not keep the step alive in it
        if step is not None:
            step(f"threshold {lower}")
        return done

    pairs = [(i, lower, upper) for i, (lower, upper) in enumerate(zip(thresholds, upper_thresholds))]

    if num_threads > 1:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(_worker, p): p[0] for p in pairs}
            for future in as_completed(futures):
                idx, path, seed = future.result()
                ordered_paths[idx] = path
                if seeds_out is not None:
                    seeds_out[idx] = seed
                if verbose:
                    print(f"[seed_sweep] step {idx:03d} done: {path.name}")
    else:
        for p in pairs:
            idx, path, seed = _worker(p)
            ordered_paths[idx] = path
            if seeds_out is not None:
                seeds_out[idx] = seed
            if verbose:
                print(f"[seed_sweep] step {idx:03d} done: {path.name}")

    return ordered_paths, sweep_direction


def seed_sweep_erosion(
    img,
    output_folder,
    threshold,
    upper_threshold=None,
    erosion_steps=5,
    segments=500,
    boundary=None,
    footprints=None,
    verbose=True,
    seeds_out=None,
    cancel=None,
    step=None,
):
    """Erosion sweep: one threshold, progressively eroded, steps are sequential.

    Returns (ordered_paths, "l2s") — step_000 has no erosion (loosest) so the
    natural file order is loose-to-strict.
    cancel: checked before every step, raises Cancelled. step: called after every step.
    """
    boundary = _boundary_mask(img, boundary)
    threshold, upper_threshold = config_core.check_and_assign_threshold(threshold, upper_threshold)
    footprint_list = config_core.check_and_assign_footprint(footprints, erosion_steps) if erosion_steps > 0 else []
    n_steps = erosion_steps + 1  # step 0 (no erosion) + steps 1..erosion_steps
    segments_list = _resolve_segments_list(segments, n_steps)

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    if verbose:
        print(
            f"[seed_sweep] erosion mode: erosion_steps={erosion_steps}, "
            f"threshold={threshold}, segments={segments}"
        )

    mask = _apply_threshold(img, threshold, upper_threshold)
    if boundary is not None:
        mask[boundary] = False

    ordered_paths = []
    uint16_labels = sprout_core.Uint16Labels()       # one for all the steps

    # Step 0: no erosion
    seed, _ = sprout_core.get_ccomps_with_size_order(mask, segments_list[0], uint16=uint16_labels)
    seed = sprout_core.cast_labels(seed, np.uint16, copy=False)
    path = _save_step(seed, output_folder, 0, threshold, upper_threshold, 0)
    ordered_paths.append(path)
    if seeds_out is not None:
        seeds_out.append(_in_memory(seed))
    if verbose:
        print(f"[seed_sweep] step 000 done: {path.name}")
    if step is not None:
        step("erosion 0")

    # Steps 1..erosion_steps: progressive erosion
    for ero_iter in range(1, erosion_steps + 1):
        sprout_core.check_cancel(cancel)
        mask = sprout_core.erosion_binary_img_on_sub(
            mask, kernal_size=1, footprint=footprint_list[ero_iter - 1]
        )
        seed, _ = sprout_core.get_ccomps_with_size_order(mask, segments_list[ero_iter], uint16=uint16_labels)
        seed = sprout_core.cast_labels(seed, np.uint16, copy=False)
        path = _save_step(seed, output_folder, ero_iter, threshold, upper_threshold, ero_iter)
        ordered_paths.append(path)
        if seeds_out is not None:
            seeds_out.append(_in_memory(seed))
        if verbose:
            print(f"[seed_sweep] step {ero_iter:03d} done: {path.name}")
        if step is not None:
            step(f"erosion {ero_iter}")

    return ordered_paths, "l2s"


def run_seed_sweep(config_path):
    """Entry point for YAML-driven seed sweep."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    optional = config_core.validate_input_yaml(config, config_core.input_val_seed_sweep)

    workspace = optional.get("workspace", "")
    img = config_core.check_and_load_data(None, os.path.join(workspace, config["img_path"]), "img")
    boundary_path = optional["boundary_path"]
    if boundary_path:
        boundary_path = os.path.join(workspace, boundary_path)
    boundary = config_core.check_and_load_data(
        None, boundary_path, "boundary", must_exist=False
    )
    if boundary is not None:
        boundary = config_core.check_and_cast_boundary(boundary)

    base_name = config_core.check_and_assign_base_name(
        optional["base_name"], config["img_path"], "seed_sweep"
    )

    output_folder = config["output_folder"]
    if workspace:
        output_folder = os.path.join(workspace, output_folder)
    output_folder = os.path.join(output_folder, base_name)
    output_folder = os.path.abspath(output_folder)

    thresholds = config["thresholds"]
    upper_thresholds = optional["upper_thresholds"]
    erosion_steps = config["erosion_steps"]
    segments = config["segments"]
    num_threads = config.get("num_threads", 1)
    footprints = optional["footprints"]
    verbose = optional.get("verbose", True)

    start = time.time()

    # Auto-detect mode: single int or [x] → erosion, list >1 → threshold
    if isinstance(thresholds, (int, float)) or (isinstance(thresholds, list) and len(thresholds) == 1):
        single_threshold = thresholds[0] if isinstance(thresholds, list) else thresholds
        single_upper = _single_upper(upper_thresholds)
        ordered_paths, _ = seed_sweep_erosion(
            img=img,
            output_folder=output_folder,
            threshold=single_threshold,
            upper_threshold=single_upper,
            erosion_steps=erosion_steps,
            segments=segments,
            boundary=boundary,
            footprints=footprints,
            verbose=verbose,
        )
    else:
        ordered_paths, _ = seed_sweep_threshold(
            img=img,
            output_folder=output_folder,
            thresholds=thresholds,
            upper_thresholds=upper_thresholds,
            erosion_steps=erosion_steps,
            segments=segments,
            boundary=boundary,
            footprints=footprints,
            num_threads=num_threads,
            verbose=verbose,
        )

    config_core.save_used_config(output_folder, config_core.used_config(
        {**config, **optional}, config_core.input_val_seed_sweep))
    print(f"[seed_sweep] {len(ordered_paths)} steps saved to {output_folder}")
    print(f"[seed_sweep] total time: {time.time() - start:.2f}s")
    return ordered_paths


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_seed_sweep(sys.argv[1])
    else:
        run_seed_sweep("./template/seed_sweep.yaml")
