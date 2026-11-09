"""Pipeline: seed sweep + adaptive combine in one YAML-driven run.

Phase 1 (sweep): generates an ordered series of labeled TIFF masks from a raw
image using threshold or erosion sweeping.  Skipped if `seeds_folder` is
provided in the YAML and that folder already exists.

Phase 2 (combine): runs strict-to-loose, loose-to-strict, or both on the sweep
output using the fast combine logic from adaptive_seed_combine.py.

YAML parameter naming:
  - Sweep params: img_path, thresholds, erosion_steps, segments, num_threads,
    upper_thresholds, boundary_path, footprints, base_name, workspace
  - Shared combine params: direction, min_area, connectivity, input_mode,
    top_n, save_intermediate, verbose
  - s2l-specific: s2l_min_current_coverage, s2l_max_emerge_coverage
  - l2s-specific: l2s_min_new_coverage, l2s_max_new_coverage,
    l2s_keep_disappeared
  - Skip sweep: seeds_folder (path to existing folder of step_*.tif files)
"""

import json
import os
import sys
import time
from pathlib import Path

import tifffile
import yaml

import sprout_core.config_core as config_core
from sprout_core.adaptive_seed_combine import (
    COMBINE_L2S_KEYS,
    COMBINE_S2L_KEYS,
    COMBINE_SHARED_KEYS,
    adaptive_seed_l2s_from_seeds,
    adaptive_seed_s2l_from_seeds,
)
from sprout_core.seed_sweep import seed_sweep_erosion, seed_sweep_threshold


# ---------------------------------------------------------------------------
# YAML → internal parameter mapping
# ---------------------------------------------------------------------------

def _map_yaml_to_combine_kwargs(optional, direction):
    """Translate prefixed YAML params to internal function parameter names."""
    shared = {
        "min_area": optional.get("min_area", 1),
        "connectivity": optional.get("connectivity", 1),
        "input_mode": optional.get("input_mode", "auto"),
        "top_n": optional.get("top_n", None),
        "save_intermediate": optional.get("save_intermediate", False),
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

def _save_result(combine_seed, seed_metadata, output_folder, output_name):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    label_path = output_folder / f"{output_name}_label.tif"
    metadata_path = output_folder / f"{output_name}_metadata.json"
    tifffile.imwrite(str(label_path), combine_seed, compression="zlib")
    metadata_path.write_text(json.dumps(seed_metadata, indent=2), encoding="utf-8")
    print(f"  label  → {label_path}")
    print(f"  meta   → {metadata_path}")
    print(f"  seeds  : {len(seed_metadata)}")


# ---------------------------------------------------------------------------
# Core pipeline (accepts config dict — used by both single and batch runs)
# ---------------------------------------------------------------------------

def run_sweep_adaptive_seed_from_dict(config):
    """Run the full sweep + combine pipeline from a config dict."""
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

    direction = optional.get("direction", "strict_to_loose")
    seeds_folder = optional.get("seeds_folder", None)
    verbose = optional.get("verbose", True)

    start = time.time()

    # ------------------------------------------------------------------ #
    # Phase 1: sweep (or load existing)                                   #
    # ------------------------------------------------------------------ #
    if seeds_folder and Path(seeds_folder).exists():
        if verbose:
            print(f"[sweep_adaptive] skipping sweep — loading from {seeds_folder}")
        tif_paths = sorted(Path(seeds_folder).glob("step_*.tif"))
        if not tif_paths:
            raise ValueError(f"No step_*.tif files found in seeds_folder: {seeds_folder}")
        seeds = [tifffile.imread(str(p)) for p in tif_paths]
        # When loading pre-existing files the sweep direction cannot be inferred
        # from the data; specify `seeds_sweep_direction` in YAML if the files
        # were produced by an l2s sweep (default: "s2l" for backward compat).
        sweep_direction = optional.get("seeds_sweep_direction", "s2l")
    else:
        sweep_folder = os.path.join(output_folder_root, "sweep")

        img = config_core.check_and_load_data(None, img_path, "img")
        boundary = config_core.check_and_load_data(
            None, optional["boundary_path"], "boundary", must_exist=False
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
            print(f"[sweep_adaptive] Phase 1: sweep → {sweep_folder}")

        if isinstance(thresholds, int) or (isinstance(thresholds, list) and len(thresholds) == 1):
            single_threshold = thresholds if isinstance(thresholds, int) else thresholds[0]
            single_upper = (
                upper_thresholds
                if (upper_thresholds is None or isinstance(upper_thresholds, int))
                else (upper_thresholds[0] if len(upper_thresholds) == 1 else None)
            )
            ordered_paths, sweep_direction = seed_sweep_erosion(
                img=img,
                output_folder=sweep_folder,
                threshold=single_threshold,
                upper_threshold=single_upper,
                erosion_steps=erosion_steps,
                segments=segments_count,
                boundary=boundary,
                footprints=footprints,
                verbose=verbose,
            )
        else:
            ordered_paths, sweep_direction = seed_sweep_threshold(
                img=img,
                output_folder=sweep_folder,
                thresholds=thresholds,
                upper_thresholds=upper_thresholds,
                erosion_steps=erosion_steps,
                segments=segments_count,
                boundary=boundary,
                footprints=footprints,
                num_threads=num_threads,
                verbose=verbose,
            )

        seeds = [tifffile.imread(str(p)) for p in ordered_paths]

    if verbose:
        print(f"[sweep_adaptive] loaded {len(seeds)} sweep steps (sweep_direction={sweep_direction})")

    # ------------------------------------------------------------------ #
    # Phase 2: combine                                                     #
    # ------------------------------------------------------------------ #
    if verbose:
        print(f"[sweep_adaptive] Phase 2: combine (direction={direction})")

    combine_kwargs = _map_yaml_to_combine_kwargs(optional, direction)

    if combine_kwargs.get("save_intermediate"):
        combine_kwargs["intermediate_folder"] = os.path.join(
            output_folder_root, "combine_intermediate"
        )

    combine_output_folder = os.path.join(output_folder_root, "combine")

    # Align seeds to each combine direction.
    # s2l expects seeds[0] = strictest mask (smallest range).
    # l2s expects seeds[0] = loosest mask (largest range).
    # sweep_direction tells us what order the sweep files were saved in.
    if sweep_direction == "s2l":
        # step_000 = strictest → already correct for s2l; reverse for l2s
        seeds_s2l = seeds
        seeds_l2s = list(reversed(seeds))
    else:
        # step_000 = loosest → already correct for l2s; reverse for s2l
        seeds_l2s = seeds
        seeds_s2l = list(reversed(seeds))

    if direction == "both":
        shared = {k: v for k, v in combine_kwargs.items() if k in COMBINE_SHARED_KEYS}
        s2l_kw = {**shared, **{k: v for k, v in combine_kwargs.items() if k in COMBINE_S2L_KEYS},
                  "intermediate_prefix": "s2l"}
        l2s_kw = {**shared, **{k: v for k, v in combine_kwargs.items() if k in COMBINE_L2S_KEYS},
                  "intermediate_prefix": "l2s"}
        s2l_seed, s2l_meta = adaptive_seed_s2l_from_seeds(seeds_s2l, **s2l_kw)
        l2s_seed, l2s_meta = adaptive_seed_l2s_from_seeds(seeds_l2s, **l2s_kw)
        print("[sweep_adaptive] strict_to_loose result:")
        _save_result(s2l_seed, s2l_meta, combine_output_folder, "s2l")
        print("[sweep_adaptive] loose_to_strict result:")
        _save_result(l2s_seed, l2s_meta, combine_output_folder, "l2s")
    elif direction == "loose_to_strict":
        combine_seed, seed_metadata = adaptive_seed_l2s_from_seeds(seeds_l2s, **combine_kwargs)
        print(f"[sweep_adaptive] {direction} result:")
        _save_result(combine_seed, seed_metadata, combine_output_folder, "l2s")
    else:
        combine_seed, seed_metadata = adaptive_seed_s2l_from_seeds(seeds_s2l, **combine_kwargs)
        print(f"[sweep_adaptive] {direction} result:")
        _save_result(combine_seed, seed_metadata, combine_output_folder, "s2l")

    config_core.save_config_with_output({"params": config}, output_folder_root)
    print(f"[sweep_adaptive] done in {time.time() - start:.2f}s")


def run_sweep_adaptive_seed(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    run_sweep_adaptive_seed_from_dict(config)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_sweep_adaptive_seed(sys.argv[1])
    else:
        run_sweep_adaptive_seed("./template/sweep_adaptive_seed.yaml")
