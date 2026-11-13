"""Pipeline: seed sweep + adaptive combine in one YAML-driven run.

Phase 1 (sweep): generates an ordered series of labeled TIFF masks from a raw
image using threshold or erosion sweeping.  Skipped if `seeds_folder` is
provided in the YAML and that folder already exists.

Phase 2 (combine): runs strict-to-loose, loose-to-strict, or both on the sweep
output using the fast combine logic from adaptive_seed_combine.py.

YAML parameter naming:
  - Sweep params: img_path, thresholds, erosion_steps, segments, num_threads,
    upper_thresholds, boundary_path, footprints, base_name, workspace
  - Shared combine params: direction, min_size, input_mode,
    top_n, save_every_iter, verbose
  - s2l-specific: s2l_min_current_coverage, s2l_max_emerge_coverage
  - l2s-specific: l2s_min_new_coverage, l2s_max_new_coverage,
    l2s_keep_disappeared
  - Skip sweep: seeds_folder (path to existing folder of step_*.tif files)

Output, in <output_folder>/<base_name>/:
  - FINAL_sweep_seed.tif: the seeds, the input of Grow, numbered by size (1 = largest)
  - sweep_regions.csv: one row per seed: area, the steps where it began and ended, and how
  - sweep_steps.csv: one row per step, in the order of the combine
  - sweep_metadata.json: the direction, the steps, and everything the combine knows about each seed
  - steps/: the mask of each step
  - intermediate/<direction>/after_step_NNN.tif: the seeds after each step, with save_every_iter
With direction `both` the first four are written twice, with _strict_to_loose and _loose_to_strict
in the name.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import tifffile
import yaml

import sprout_core.config_core as config_core
import sprout_core.sprout_core as sprout_core
from sprout_core.adaptive_seed_combine import (
    COMBINE_L2S_KEYS,
    COMBINE_S2L_KEYS,
    COMBINE_SHARED_KEYS,
    adaptive_seed_l2s_from_seeds,
    adaptive_seed_s2l_from_seeds,
)
from sprout_core.seed_sweep import (
    STEP_NAME,
    StepFiles,
    _single_upper,
    detect_direction_from_files,
    seed_sweep_erosion,
    seed_sweep_threshold,
)


# ---------------------------------------------------------------------------
# YAML → internal parameter mapping
# ---------------------------------------------------------------------------

def _map_yaml_to_combine_kwargs(optional, direction):
    """Translate prefixed YAML params to internal function parameter names."""
    shared = {
        "min_area": optional.get("min_size", 1),
        "input_mode": optional.get("input_mode", "auto"),
        "top_n": optional.get("top_n", None),
        "save_intermediate": optional.get("save_every_iter", False),
        "verbose": optional.get("verbose", True),
    }

    s2l_extra = {
        # YAML: s2l_min_current_coverage → internal: min_current_coverage
        "min_current_coverage": optional.get("s2l_min_current_coverage", None),
        # YAML: s2l_max_emerge_coverage → internal: max_emerge_coverage
        "max_emerge_coverage": optional.get("s2l_max_emerge_coverage", None),
        # YAML: s2l_mask_by_last_seed → internal: mask_by_last_seed
        "mask_by_last_seed": optional.get("s2l_mask_by_last_seed", False),
    }

    l2s_extra = {
        # YAML: l2s_min_new_coverage → internal: min_new_coverage
        "min_new_coverage": optional.get("l2s_min_new_coverage", 0.5),
        # YAML: l2s_max_new_coverage → internal: max_new_coverage
        "max_new_coverage": optional.get("l2s_max_new_coverage", 0.05),
        "keep_disappeared": optional.get("l2s_keep_disappeared", True),
    }

    if direction == "strict_to_loose":
        return {**shared, **s2l_extra}
    if direction == "loose_to_strict":
        return {**shared, **l2s_extra}
    # both: split by key sets in run logic
    return {**shared, **s2l_extra, **l2s_extra}


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _step_settings(path):
    """Lower, upper and erosion of a step, from its file name. Empty for files of other names, and
    for an upper threshold of None."""
    match = STEP_NAME.search(Path(path).name)
    if not match:
        return "", "", ""
    lower, upper, erosion = match.groups()
    return lower, "" if upper == "None" else upper, erosion


STEP_NUMBER = re.compile(r"^step_(\d+)_")
SETTINGS = ("threshold", "upper", "erosion")      # the columns of _step_settings, as sweep_regions.csv names them
SUFFIXES = {"strict_to_loose": "_strict_to_loose", "loose_to_strict": "_loose_to_strict"}


def _step_numbers(step_paths):
    """The number in the file name of each step (step_003_... -> 3), or its place in the combine."""
    numbers = [STEP_NUMBER.match(Path(path).name) for path in step_paths]
    return [int(m.group(1)) if m else index for index, m in enumerate(numbers)]


def _began(item, final_ids):
    if item["start_index"] == 0:
        return "first step"
    parent = item.get("parent_seed_id")
    if parent is None:
        return "emerged"
    return f"split from {final_ids[parent]}" if parent in final_ids else "split from a seed not kept"


def _ended(item, numbers):
    """How the seed ended and at which step: locked / never locked (strict to loose),
    disappeared / reached the end (loose to strict)."""
    if item["status"] == "locked_pre_merge":
        return "locked", numbers[item["merge_index"]]
    if item["status"] == "disappeared":
        return "disappeared", numbers[item["disappear_index"]]
    return ("never locked" if item["finalization_reason"] == "never_merged" else "reached the end"), ""


def _regions_table(seed_metadata, step_paths):
    """One row per seed. Of the step settings, only those that change from step to step get columns."""
    numbers = _step_numbers(step_paths)
    settings = [_step_settings(path) for path in step_paths]
    varying = [i for i in range(3) if len({s[i] for s in settings}) > 1] or [0]
    final_ids = {item["original_seed_id"]: item["seed_id"] for item in seed_metadata}
    columns = ["id", "area", "first_step", "last_step"] + \
        [f"{end}_{SETTINGS[i]}" for i in varying for end in ("first", "last")] + ["began", "ended", "at_step"]
    rows = []
    for item in seed_metadata:
        first, last = item["start_index"], item["last_index"]
        row = {"id": item["seed_id"], "area": item["area"], "first_step": numbers[first], "last_step": numbers[last]}
        for i in varying:
            row[f"first_{SETTINGS[i]}"], row[f"last_{SETTINGS[i]}"] = settings[first][i], settings[last][i]
        row["began"] = _began(item, final_ids)
        row["ended"], row["at_step"] = _ended(item, numbers)
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _steps_table(seed_metadata, step_paths, step_log):
    """One row per step, in the order of the combine."""
    numbers = _step_numbers(step_paths)
    started = [item["start_index"] for item in seed_metadata]
    rows = []
    for index, path in enumerate(step_paths):
        lower, upper, erosion = _step_settings(path)
        log = step_log[index] if step_log else {}
        rows.append({"step": numbers[index], "lower": lower, "upper": upper, "erosion": erosion,
                     "regions": log.get("regions", ""), "new_seeds": log.get("new_seeds", ""),
                     "seeds_after": log.get("seeds_after", ""), "in_final": started.count(index),
                     "file": Path(path).name})
    return pd.DataFrame(rows, columns=["step", "lower", "upper", "erosion", "regions", "new_seeds", "seeds_after",
                                       "in_final", "file"])


def _save_result(combine_seed, seed_metadata, step_paths, output_folder, direction, step_log=None, suffix=""):
    """Write the result of one direction; `step_paths` and `step_log` are in the order of its combine."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    label_path = output_folder / f"FINAL_sweep_seed{suffix}.tif"
    tifffile.imwrite(str(label_path), combine_seed, compression="zlib")
    metadata = {"direction": direction, "steps": [Path(path).name for path in step_paths], "seeds": seed_metadata}
    (output_folder / f"sweep_metadata{suffix}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _regions_table(seed_metadata, step_paths).to_csv(output_folder / f"sweep_regions{suffix}.csv", index=False)
    _steps_table(seed_metadata, step_paths, step_log).to_csv(output_folder / f"sweep_steps{suffix}.csv", index=False)

    print(f"  seeds: {len(seed_metadata)}")
    print(f"  final: {label_path}")
    return label_path


# ---------------------------------------------------------------------------
# Core pipeline (accepts config dict — used by both single and batch runs)
# ---------------------------------------------------------------------------

STEP_BYTES = 2      # per voxel of a step in memory: uint8 when its ids fit, else uint16 (the plan takes the most)
WORK_BYTES = 7      # per voxel of a step being computed: its int32 labels, the mask, the uint16 copy


OUT_OF_MEMORY_HINT = ("[sweep_adaptive] Out of memory: set memory_limit_gb in the config to keep the steps "
                      "on disk and run fewer threads at once (slower, same result).")


def _memory_plan(voxels, n_steps, num_threads, memory_limit_gb, verbose):
    """(steps in memory, threads) that stay under memory_limit_gb.

    None: all steps in memory and num_threads, and only a warning when that may not fit in the free
    memory. Otherwise the steps stay in memory if they fit next to the steps being computed, or the
    combine reads them back from disk one at a time; fewer threads compute fewer steps at once. Both
    only change the speed, not the result.
    """
    steps, work = n_steps * voxels * STEP_BYTES, voxels * WORK_BYTES
    if memory_limit_gb is None:
        free = psutil.virtual_memory().available
        if verbose and steps + num_threads * work > free:
            print(f"[sweep_adaptive] [WARNING] the steps and {num_threads} thread(s) may need about "
                  f"{(steps + num_threads * work) / 2 ** 30:.1f} GB, {free / 2 ** 30:.1f} GB is free. "
                  f"If the run runs out of memory, set memory_limit_gb.")
        return True, num_threads
    limit = memory_limit_gb * 2 ** 30
    in_memory = steps + num_threads * work <= limit
    threads = num_threads if in_memory else max(1, min(num_threads, int(limit // work)))
    if verbose:
        where = "kept in memory" if in_memory else "read from disk one at a time"
        print(f"[sweep_adaptive] memory limit {limit / 2 ** 30:.1f} GB (memory_limit_gb): steps {where} "
              f"(all steps take about {steps / 2 ** 30:.1f} GB), {threads} thread(s) "
              f"(about {work / 2 ** 30:.1f} GB each)")
        if threads < num_threads:
            print(f"[sweep_adaptive] num_threads {num_threads} -> {threads} to stay under the memory limit")
        if work > limit:
            print("[sweep_adaptive] [WARNING] one step alone needs more than the memory limit")
    return in_memory, threads


def _n_steps(thresholds, erosion_steps):
    return erosion_steps + 1 if _is_erosion_sweep(thresholds) else len(thresholds)


def _is_erosion_sweep(thresholds):
    return isinstance(thresholds, (int, float)) or (isinstance(thresholds, list) and len(thresholds) == 1)


def _run_sweep(img, folder, thresholds, upper_thresholds, erosion_steps, segments, boundary, footprints,
               num_threads, verbose, seeds, cancel=None, step=None):
    """Phase 1: one threshold is an erosion sweep, several are a threshold sweep. Fills `seeds`."""
    if _is_erosion_sweep(thresholds):
        return seed_sweep_erosion(
            img=img,
            output_folder=folder,
            threshold=thresholds[0] if isinstance(thresholds, list) else thresholds,
            upper_threshold=_single_upper(upper_thresholds),
            erosion_steps=erosion_steps,
            segments=segments,
            boundary=boundary,
            footprints=footprints,
            verbose=verbose,
            seeds_out=seeds,
            cancel=cancel,
            step=step,
        )
    return seed_sweep_threshold(
        img=img,
        output_folder=folder,
        thresholds=thresholds,
        upper_thresholds=upper_thresholds,
        erosion_steps=erosion_steps,
        segments=segments,
        boundary=boundary,
        footprints=footprints,
        num_threads=num_threads,
        verbose=verbose,
        seeds_out=seeds,
        cancel=cancel,
        step=step,
    )


def make_sweep_adaptive_seed(img, thresholds, erosion_steps, segments, output_folder, base_name="sweep",
                             upper_thresholds=None, boundary=None, footprints=None, num_threads=1, min_size=1,
                             cancel=None, progress=None, memory_limit_gb=None):
    """Sweep and strict-to-loose combine of an array, for napari. Returns {layer name: seeds}.

    Writes the same files as a YAML run. cancel: raises Cancelled between steps.
    progress: callable(done, total, text). memory_limit_gb: as in the YAML.
    """
    values = dict(locals())                       # the arguments, for used_config.yaml
    n_steps = _n_steps(thresholds, erosion_steps)
    steps = sprout_core.StepProgress(progress, n_steps + 1)
    folder = os.path.abspath(os.path.join(output_folder, base_name))
    config_core.warn_earlier_results(folder)
    in_memory, threads = _memory_plan(img.size, n_steps, num_threads, memory_limit_gb, False)
    seeds = [] if in_memory else None
    try:
        paths, sweep_direction = _run_sweep(
            img, os.path.join(folder, "steps"), thresholds, upper_thresholds, erosion_steps, segments, boundary,
            footprints, threads, False, seeds, cancel, steps.step,
        )
        if not in_memory:
            seeds = StepFiles(paths)
        if sweep_direction == "l2s":
            seeds, paths = seeds[::-1], paths[::-1]
        step_log = []
        result, metadata = adaptive_seed_s2l_from_seeds(seeds, min_area=min_size, input_mode="label",
                                                        verbose=False, cancel=cancel, step_log=step_log)
    except MemoryError as e:
        raise MemoryError(f"{e}\n{OUT_OF_MEMORY_HINT}") from e
    sprout_core.check_cancel(cancel)
    _save_result(result, metadata, paths, folder, "strict_to_loose", step_log)
    # this route always combines strict to loose, each id of a step one region
    config_core.save_used_config(folder, config_core.used_config(
        {**values, "direction": "strict_to_loose", "input_mode": "label", "save_every_iter": False, "verbose": False},
        config_core.input_val_sweep_adaptive_seed), "--sweep_adaptive")
    steps.step("combine")
    return {f"{base_name}_FINAL_sweep_seed": sprout_core.cast_labels(result, np.uint16)}


def run_sweep_adaptive_seed_from_dict(config):
    """Run the full sweep + combine pipeline from a config dict. Returns the paths of the final seeds."""
    try:
        return _run_from_dict(config)
    except MemoryError as e:
        raise MemoryError(f"{e}\n{OUT_OF_MEMORY_HINT}") from e


def _run_from_dict(config):
    optional = config_core.validate_input_yaml(config, config_core.input_val_sweep_adaptive_seed)

    workspace = optional.get("workspace", "") or ""
    img_path = config["img_path"]
    if workspace:
        img_path = os.path.join(workspace, img_path)

    output_folder_root = config["output_folder"]
    if workspace:
        output_folder_root = os.path.join(workspace, output_folder_root)

    base_name = config_core.check_and_assign_base_name(
        optional["base_name"], img_path, "sweep_adaptive"
    )
    output_folder_root = os.path.abspath(os.path.join(output_folder_root, base_name))
    config_core.warn_earlier_results(output_folder_root)

    direction = optional.get("direction", "strict_to_loose")
    seeds_folder = optional.get("seeds_folder", None)
    verbose = optional.get("verbose", True)

    start = time.time()

    # ------------------------------------------------------------------ #
    # Phase 1: sweep (or load existing)                                   #
    # ------------------------------------------------------------------ #
    if seeds_folder:
        if workspace:
            seeds_folder = os.path.join(workspace, seeds_folder)
        if not Path(seeds_folder).is_dir():
            raise FileNotFoundError(f"seeds_folder does not exist: {seeds_folder}")
        if verbose:
            print(f"[sweep_adaptive] skipping sweep, loading from {seeds_folder}")
        ordered_paths = tif_paths = sorted(Path(seeds_folder).glob("step_*.tif"))
        if not tif_paths:
            raise ValueError(f"No step_*.tif files found in seeds_folder: {seeds_folder}")
        seeds = StepFiles(tif_paths)
        first_shape = seeds.shape_of(0)
        if _memory_plan(int(np.prod(first_shape)), len(seeds), 1, optional.get("memory_limit_gb"), verbose)[0]:
            seeds = list(seeds)
        # The direction comes from the file names unless the YAML sets it
        sweep_direction = config.get("seeds_sweep_direction") or detect_direction_from_files(tif_paths)
    else:
        sweep_folder = os.path.join(output_folder_root, "steps")
        seeds = []

        img = config_core.check_and_load_data(None, img_path, "img")
        boundary_path = optional["boundary_path"]
        if boundary_path and workspace:
            boundary_path = os.path.join(workspace, boundary_path)
        boundary = config_core.check_and_load_data(
            None, boundary_path, "boundary", must_exist=False
        )
        if boundary is not None:
            boundary = config_core.check_and_cast_boundary(boundary)

        thresholds = config["thresholds"]
        upper_thresholds = optional["upper_thresholds"]
        erosion_steps = config["erosion_steps"]
        segments_count = config["segments"]
        num_threads = config.get("num_threads", optional.get("num_threads", 1))
        footprints = optional["footprints"]

        if verbose:
            print(f"[sweep_adaptive] Phase 1: sweep into {sweep_folder}")

        in_memory, num_threads = _memory_plan(img.size, _n_steps(thresholds, erosion_steps), num_threads,
                                              optional.get("memory_limit_gb"), verbose)
        seeds = seeds if in_memory else None
        ordered_paths, sweep_direction = _run_sweep(
            img, sweep_folder, thresholds, upper_thresholds, erosion_steps, segments_count, boundary,
            footprints, num_threads, verbose, seeds,
        )
        if not in_memory:
            seeds = StepFiles(ordered_paths)

    # ------------------------------------------------------------------ #
    # Phase 2: combine                                                     #
    # ------------------------------------------------------------------ #
    if verbose:
        print(f"[sweep_adaptive] Phase 2: combine (direction={direction})")

    combine_kwargs = _map_yaml_to_combine_kwargs(optional, direction)

    combine_kwargs.pop("intermediate_folder", None)

    # Align seeds to each combine direction.
    # s2l expects seeds[0] = strictest mask (smallest range).
    # l2s expects seeds[0] = loosest mask (largest range).
    # sweep_direction tells us what order the sweep files were saved in.
    if sweep_direction == "s2l":
        # step_000 = strictest → already correct for s2l; reverse for l2s
        seeds_s2l, paths_s2l = seeds, ordered_paths
        seeds_l2s, paths_l2s = seeds[::-1], ordered_paths[::-1]
    else:
        # step_000 = loosest → already correct for l2s; reverse for s2l
        seeds_l2s, paths_l2s = seeds, ordered_paths
        seeds_s2l, paths_s2l = seeds[::-1], ordered_paths[::-1]

    def combine(name, kwargs):
        """One direction: its seeds and files in the order of its combine, and the result written."""
        combine_fn, seeds_in, paths_in = {
            "strict_to_loose": (adaptive_seed_s2l_from_seeds, seeds_s2l, paths_s2l),
            "loose_to_strict": (adaptive_seed_l2s_from_seeds, seeds_l2s, paths_l2s),
        }[name]
        step_log = []
        if kwargs.get("save_intermediate"):
            kwargs = {**kwargs, "intermediate_folder": os.path.join(output_folder_root, "intermediate", name),
                      "intermediate_prefix": "after_step", "intermediate_numbers": _step_numbers(paths_in)}
        combine_seed, seed_metadata = combine_fn(seeds_in, step_log=step_log, **kwargs)
        print(f"[sweep_adaptive] {name} result:")
        return _save_result(combine_seed, seed_metadata, paths_in, output_folder_root, name, step_log,
                            SUFFIXES[name] if direction == "both" else "")

    if direction == "both":
        shared = {k: v for k, v in combine_kwargs.items() if k in COMBINE_SHARED_KEYS}
        finals = [combine("strict_to_loose", {**shared, **{k: v for k, v in combine_kwargs.items()
                                                            if k in COMBINE_S2L_KEYS}}),
                  combine("loose_to_strict", {**shared, **{k: v for k, v in combine_kwargs.items()
                                                            if k in COMBINE_L2S_KEYS}})]
    else:
        finals = [combine(direction, combine_kwargs)]

    config_core.save_used_config(output_folder_root, config_core.used_config(
        {**config, **optional}, config_core.input_val_sweep_adaptive_seed), "--sweep_adaptive")
    print(f"[sweep_adaptive] done in {time.time() - start:.2f}s")
    return finals


def run_sweep_adaptive_seed(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    run_sweep_adaptive_seed_from_dict(config)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_sweep_adaptive_seed(sys.argv[1])
    else:
        run_sweep_adaptive_seed("./template/sweep_adaptive_seed.yaml")
