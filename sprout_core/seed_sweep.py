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
import sys
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tifffile import imwrite

import sprout_core.sprout_core as sprout_core
import sprout_core.config_core as config_core


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
                                  segments, step_index, output_folder, boundary=None):
    """Generate and save one seed for a single threshold (threshold mode)."""
    mask = _apply_threshold(img, lower, upper)
    if boundary is not None:
        mask[boundary] = False
    mask = _apply_erosions(mask, footprint_list, erosion_steps)
    seed, _ = sprout_core.get_ccomps_with_size_order(mask, segments)
    seed = seed.astype("uint16")
    path = _save_step(seed, output_folder, step_index, lower, upper, erosion_steps)
    return step_index, path


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
):
    """Threshold sweep: each threshold → one TIFF, steps are parallel.

    Returns (ordered_paths, sweep_direction) where sweep_direction is "s2l" or "l2s".
    s2l: lower ↓ / upper ↑ — step_000 is strictest (range widens over steps).
    l2s: lower ↑ / upper ↓ — step_000 is loosest (range narrows over steps).
    Direction is auto-detected from threshold ordering.
    """
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

    def _worker(args):
        idx, lower, upper = args
        return _generate_one_threshold_step(
            img, lower, upper, footprint_list, erosion_steps,
            segments_list[idx], idx, output_folder, boundary
        )

    pairs = [(i, lower, upper) for i, (lower, upper) in enumerate(zip(thresholds, upper_thresholds))]

    if num_threads > 1:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(_worker, p): p[0] for p in pairs}
            for future in as_completed(futures):
                idx, path = future.result()
                ordered_paths[idx] = path
                if verbose:
                    print(f"[seed_sweep] step {idx:03d} done → {path.name}")
    else:
        for p in pairs:
            idx, path = _worker(p)
            ordered_paths[idx] = path
            if verbose:
                print(f"[seed_sweep] step {idx:03d} done → {path.name}")

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
):
    """Erosion sweep: one threshold, progressively eroded, steps are sequential.

    Returns (ordered_paths, "l2s") — step_000 has no erosion (loosest) so the
    natural file order is loose-to-strict.
    """
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

    # Step 0: no erosion
    seed, _ = sprout_core.get_ccomps_with_size_order(mask.copy(), segments_list[0])
    seed = seed.astype("uint16")
    path = _save_step(seed, output_folder, 0, threshold, upper_threshold, 0)
    ordered_paths.append(path)
    if verbose:
        print(f"[seed_sweep] step 000 done → {path.name}")

    # Steps 1..erosion_steps: progressive erosion
    for ero_iter in range(1, erosion_steps + 1):
        mask = sprout_core.erosion_binary_img_on_sub(
            mask, kernal_size=1, footprint=footprint_list[ero_iter - 1]
        )
        seed, _ = sprout_core.get_ccomps_with_size_order(mask, segments_list[ero_iter])
        seed = seed.astype("uint16")
        path = _save_step(seed, output_folder, ero_iter, threshold, upper_threshold, ero_iter)
        ordered_paths.append(path)
        if verbose:
            print(f"[seed_sweep] step {ero_iter:03d} done → {path.name}")

    return ordered_paths, "l2s"


def run_seed_sweep(config_path):
    """Entry point for YAML-driven seed sweep."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    optional = config_core.validate_input_yaml(config, config_core.input_val_seed_sweep)

    img = config_core.check_and_load_data(None, config["img_path"], "img")
    boundary = config_core.check_and_load_data(
        None, optional["boundary_path"], "boundary", must_exist=False
    )
    if boundary is not None:
        boundary = config_core.check_and_cast_boundary(boundary)

    base_name = config_core.check_and_assign_base_name(
        optional["base_name"], config["img_path"], "seed_sweep"
    )

    workspace = optional.get("workspace", "")
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
    if isinstance(thresholds, int) or (isinstance(thresholds, list) and len(thresholds) == 1):
        single_threshold = thresholds if isinstance(thresholds, int) else thresholds[0]
        single_upper = (
            upper_thresholds
            if (upper_thresholds is None or isinstance(upper_thresholds, int))
            else (upper_thresholds[0] if len(upper_thresholds) == 1 else None)
        )
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

    config_core.save_config_with_output({"params": config}, output_folder)
    print(f"[seed_sweep] {len(ordered_paths)} steps saved to {output_folder}")
    print(f"[seed_sweep] total time: {time.time() - start:.2f}s")
    return ordered_paths


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_seed_sweep(sys.argv[1])
    else:
        run_seed_sweep("./template/seed_sweep.yaml")
