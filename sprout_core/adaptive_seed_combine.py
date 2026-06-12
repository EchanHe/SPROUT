"""Unified combine logic for adaptive seed generation.

Provides both strict-to-loose (grow/lock/emerge) and loose-to-strict
(shrink/split/disappear) directions via a single module.

A *seed* is one segmentation label map; the `seeds` argument is the ordered
list of per-step seeds produced by a sweep. Each step compares the *new
regions* (connected blocks of the next seed) against the *current regions*
(blocks already in combine_seed; one per seed_id).

Overlap thresholds (all fractions in 0..1). NOTE: they only matter for
*imperfect* nesting — if every new region maps cleanly to exactly one current
region (full containment, no grazing of neighbours, no brand-new structure),
their value makes no difference. They exist to arbitrate the messy cases:
edge-grazing, two old regions bridged into one, or new structure appearing.

  s2l (seeds grow as threshold loosens), per new region:
    min_current_coverage : a current region is "covered" when
        overlap / current-region-area >= this. 1 covered -> grow;
        >=2 covered -> merge-lock. Higher = guards against a new region that
        merely grazes a neighbour being wrongly merged. (default None = any
        overlap counts; typical 0.5)
    max_emerge_coverage : when no current region is covered, the new region may
        emerge as a brand-new seed only if its max overlap with any current
        region / new-region-area <= this; else it is dropped. (typical 0.05)

  l2s (seeds shrink as threshold tightens), per new region:
    min_new_coverage : assign the new region to a current region when
        overlap / new-region-area >= this (>=2 candidates -> split).
        (default 0.5)
    max_new_coverage : an unassigned new region becomes brand-new only if its
        max overlap / new-region-area <= this; else ignored. (default 0.05)
"""

import json
import re
import time
from pathlib import Path

import numpy as np
import tifffile
from skimage import measure


# ---------------------------------------------------------------------------
# Parameter key sets for direction splitting (used by both/direction logic)
# ---------------------------------------------------------------------------

COMBINE_SHARED_KEYS = frozenset({
    "min_area", "connectivity", "input_mode", "top_n",
    "save_intermediate", "intermediate_folder", "verbose", "log_every",
})
COMBINE_S2L_KEYS = frozenset({
    "min_current_coverage", "max_emerge_coverage",
    "intermediate_prefix", "vectorize_overlap",
})
COMBINE_L2S_KEYS = frozenset({
    "min_new_coverage", "max_new_coverage",
    "keep_disappeared", "intermediate_prefix",
})


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _threshold_from_name(path):
    match = re.search(r"thre_(\d+)", Path(path).name)
    if match is None:
        raise ValueError(f"Could not find a threshold value in file name: {path}")
    return int(match.group(1))


def load_threshold_sweep_seeds(file_paths, sort_by_threshold=True, strict_to_loose=True):
    """Load threshold-sweep TIFF masks, ordered strict-to-loose by default."""
    ordered_paths = [Path(path) for path in file_paths]
    if sort_by_threshold:
        ordered_paths = sorted(
            ordered_paths,
            key=_threshold_from_name,
            reverse=strict_to_loose,
        )
    seeds = [tifffile.imread(path) for path in ordered_paths]
    return seeds, ordered_paths


def _validate_seeds(seeds):
    if not seeds:
        raise ValueError("seeds must contain at least one mask.")
    array_seeds = [np.asarray(seed) for seed in seeds]
    shape = array_seeds[0].shape
    if array_seeds[0].ndim not in (2, 3):
        raise ValueError("seeds must contain 2D or 3D masks.")
    for index, seed in enumerate(array_seeds):
        if seed.shape != shape:
            raise ValueError(
                f"All seed masks must have the same shape. "
                f"seeds[0] has {shape}, seeds[{index}] has {seed.shape}."
            )
    return array_seeds


def _detect_input_mode(seeds, input_mode):
    if input_mode not in ("auto", "binary", "label"):
        raise ValueError("input_mode must be 'auto', 'binary', or 'label'.")
    if input_mode != "auto":
        return input_mode
    for seed in seeds:
        nonzero_values = np.unique(seed)
        nonzero_values = nonzero_values[nonzero_values != 0]
        if len(nonzero_values) > 1:
            return "label"
        if len(nonzero_values) == 1 and nonzero_values[0] != 1:
            return "label"
    return "binary"


def extract_regions(mask, min_area=1, connectivity=1, input_mode="binary", return_labeled=False):
    """Extract regions from a binary mask or an already-labelled segmentation.

    In binary mode, nonzero voxels are split into connected components. In label
    mode, each nonzero value is treated as one already-separated region.

    If return_labeled=True, region dicts omit the 'mask' key and the labeled
    array is returned as a second value. Used by the vectorized overlap path.
    """
    regions = []

    if input_mode == "binary":
        labeled = measure.label(mask.astype(bool), background=0, connectivity=connectivity)
        sizes = np.bincount(labeled.ravel())
        for label_id in range(1, len(sizes)):
            area = int(sizes[label_id])
            if area < min_area:
                continue
            r = {"area": area, "source_label": int(label_id)}
            if not return_labeled:
                r["mask"] = labeled == label_id
            regions.append(r)
        if return_labeled:
            return regions, labeled
        return regions

    for source_label in sorted(np.unique(mask)):
        if source_label == 0:
            continue
        region_mask = mask == source_label
        area = int(region_mask.sum())
        if area < min_area:
            continue
        if return_labeled:
            regions.append({"area": area, "source_label": int(source_label)})
        else:
            regions.append({"mask": region_mask, "area": area, "source_label": int(source_label)})

    if return_labeled:
        return regions, mask
    return regions


def _metadata_list(metadata_by_seed_id):
    metadata = []
    for seed_id in sorted(metadata_by_seed_id):
        item = dict(metadata_by_seed_id[seed_id])
        item["seed_id"] = int(seed_id)
        metadata.append(item)
    return metadata


def _apply_top_n(combine_seed, metadata_by_seed_id, top_n):
    if top_n is None:
        return combine_seed, metadata_by_seed_id
    seed_areas = []
    for seed_id in sorted(metadata_by_seed_id):
        area = int((combine_seed == seed_id).sum())
        seed_areas.append((area, seed_id))
    keep_ids = {seed_id for _, seed_id in sorted(seed_areas, reverse=True)[:top_n]}
    output = np.zeros_like(combine_seed)
    new_metadata = {}
    next_id = 1
    for _, old_seed_id in sorted(seed_areas, reverse=True):
        if old_seed_id not in keep_ids:
            continue
        output[combine_seed == old_seed_id] = next_id
        item = dict(metadata_by_seed_id[old_seed_id])
        item["original_seed_id"] = int(old_seed_id)
        item["area"] = int((output == next_id).sum())
        new_metadata[next_id] = item
        next_id += 1
    return output, new_metadata


def _save_intermediate_seed(combine_seed, intermediate_folder, prefix, step_index):
    intermediate_folder = Path(intermediate_folder)
    intermediate_folder.mkdir(parents=True, exist_ok=True)
    output_path = intermediate_folder / f"{prefix}_step_{step_index:03d}.tif"
    tifffile.imwrite(output_path, combine_seed, compression="zlib")
    return output_path


def _save_outputs(combine_seed, seed_metadata, ordered_paths, output_folder, output_name):
    label_path = output_folder / f"{output_name}_label.tif"
    metadata_path = output_folder / f"{output_name}_metadata.json"
    ordered_paths_path = output_folder / f"{output_name}_ordered_paths.txt"
    tifffile.imwrite(label_path, combine_seed, compression="zlib")
    metadata_path.write_text(json.dumps(seed_metadata, indent=2), encoding="utf-8")
    ordered_paths_path.write_text(
        "\n".join(str(path) for path in ordered_paths), encoding="utf-8"
    )
    print(f"Saved seed label: {label_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved ordered paths: {ordered_paths_path}")
    print(f"Number of seeds: {len(seed_metadata)}")


# ---------------------------------------------------------------------------
# Strict-to-loose internals
# ---------------------------------------------------------------------------

def _step_vectorized(
    new_labeled,
    regions,
    combine_seed,
    metadata_by_seed_id,
    locked_seed_ids,
    next_seed_id,
    seed_index,
    min_current_coverage,
    max_emerge_coverage,
):
    """One seed step using a sparse overlap matrix (faster for many regions)."""
    from scipy.sparse import csr_matrix

    nonzero = new_labeled > 0
    if not np.any(nonzero):
        return 0, 0, 0, 0, next_seed_id

    nl = new_labeled[nonzero].astype(np.int32)
    cs = combine_seed[nonzero].astype(np.int32)
    n_new = int(nl.max()) + 1
    n_old = max(int(cs.max()) + 1, 1)
    overlap_mtx = csr_matrix(
        (np.ones(nl.size, dtype=np.int32), (nl, cs)),
        shape=(n_new, n_old),
    )

    emerged_count = grown_count = merge_count = skipped_locked_count = 0

    for region in regions:
        src_label = region["source_label"]
        if src_label >= n_new:
            continue

        row = overlap_mtx.getrow(src_label)
        bg_mask = row.indices != 0
        old_ids = row.indices[bg_mask]
        counts = row.data[bg_mask]
        region_area = region["area"]

        if min_current_coverage is None and max_emerge_coverage is None:
            significant_seed_ids = [int(sid) for sid in old_ids]
            is_emerge = len(significant_seed_ids) == 0
        else:
            if old_ids.size == 0:
                significant_seed_ids = []
                max_curr_overlap_ratio = 0.0
            else:
                max_overlap_count = int(counts.max())
                max_curr_overlap_ratio = max_overlap_count / region_area
                significant_seed_ids = []
                for seed_id, overlap_count in zip(old_ids, counts):
                    seed_area = metadata_by_seed_id[int(seed_id)]["area"]
                    seed_overlap_ratio = int(overlap_count) / seed_area if seed_area else 0.0
                    if min_current_coverage is None or seed_overlap_ratio >= min_current_coverage:
                        significant_seed_ids.append(int(seed_id))

            is_emerge = len(significant_seed_ids) == 0
            if is_emerge and max_emerge_coverage is not None:
                is_emerge = max_curr_overlap_ratio <= max_emerge_coverage

        if is_emerge:
            combine_seed[new_labeled == src_label] = next_seed_id
            metadata_by_seed_id[next_seed_id] = {
                "area": region_area,
                "source_label": int(src_label),
                "start_index": int(seed_index),
                "last_index": int(seed_index),
                "finalization_reason": None,
                "status": "active",
            }
            next_seed_id += 1
            emerged_count += 1
            continue

        if len(significant_seed_ids) == 0:
            continue

        if len(significant_seed_ids) == 1:
            seed_id = significant_seed_ids[0]
            if seed_id in locked_seed_ids:
                skipped_locked_count += 1
                continue
            combine_seed[combine_seed == seed_id] = 0
            combine_seed[new_labeled == src_label] = seed_id
            metadata_by_seed_id[seed_id]["area"] = region_area
            metadata_by_seed_id[seed_id]["source_label"] = int(src_label)
            metadata_by_seed_id[seed_id]["last_index"] = int(seed_index)
            grown_count += 1
            continue

        merge_count += 1
        for seed_id in significant_seed_ids:
            locked_seed_ids.add(seed_id)
            metadata_by_seed_id[seed_id]["status"] = "locked_pre_merge"
            if metadata_by_seed_id[seed_id]["finalization_reason"] is None:
                metadata_by_seed_id[seed_id]["finalization_reason"] = "pre_merge"
                metadata_by_seed_id[seed_id]["merge_index"] = int(seed_index)
                metadata_by_seed_id[seed_id]["merged_with"] = [
                    int(item) for item in significant_seed_ids
                ]

    return emerged_count, grown_count, merge_count, skipped_locked_count, next_seed_id


# ---------------------------------------------------------------------------
# Loose-to-strict internals
# ---------------------------------------------------------------------------

def _current_seed_ids(combine_seed):
    return [int(sid) for sid in np.unique(combine_seed) if sid != 0]


def _choose_old_seed_for_region(
    combine_seed,
    region,
    min_new_coverage,
    max_new_coverage,
):
    """Match one region of the next (stricter) seed to the current seeds.

    Compares a single `region` (a connected block of the next sweep seed)
    against the seeds already in `combine_seed`, scored by how much of the
    region's own area each current seed covers:
      - returns a current seed id if coverage >= `min_new_coverage`
        (the region is that seed's shrunk / continued form; ambiguous if >1);
      - returns None if the region barely overlaps any current seed
        (max coverage <= `max_new_coverage`) → treat it as a brand-new region;
      - returns "ignore" otherwise (overlaps a seed but not enough to claim).
    """
    overlap_values = combine_seed[region["mask"]]
    overlap_values = overlap_values[overlap_values != 0]
    if overlap_values.size == 0:
        return None, False

    overlap_counts = np.bincount(overlap_values)
    seed_ids = np.flatnonzero(overlap_counts)
    if seed_ids.size == 0:
        return None, False

    current_ratios = {
        int(sid): int(overlap_counts[sid]) / region["area"] for sid in seed_ids
    }
    significant_ids = [
        sid
        for sid, ratio in current_ratios.items()
        if min_new_coverage is None or ratio >= min_new_coverage
    ]

    if significant_ids:
        chosen = max(significant_ids, key=lambda sid: current_ratios[sid])
        return chosen, len(significant_ids) > 1

    if max_new_coverage is None:
        return None, False

    if max(current_ratios.values(), default=0.0) <= max_new_coverage:
        return None, False

    return "ignore", False


# ---------------------------------------------------------------------------
# Public API: strict-to-loose
# ---------------------------------------------------------------------------

def adaptive_seed_s2l_from_seeds(
    seeds,
    min_area=1,
    connectivity=1,
    input_mode="auto",
    top_n=None,
    min_current_coverage=None,
    max_emerge_coverage=None,
    save_intermediate=False,
    intermediate_folder=None,
    intermediate_prefix="adaptive_seed_s2l",
    verbose=False,
    log_every=1,
    vectorize_overlap=False,
):
    """Strict-to-loose adaptive seed: emerge / grow / lock events."""
    if min_area < 1:
        raise ValueError("min_area must be >= 1.")
    if top_n is not None and top_n < 1:
        raise ValueError("top_n must be None or >= 1.")
    if log_every < 1:
        raise ValueError("log_every must be >= 1.")
    if min_current_coverage is not None and not 0 <= min_current_coverage <= 1:
        raise ValueError("min_current_coverage must be None or between 0 and 1.")
    if max_emerge_coverage is not None and not 0 <= max_emerge_coverage <= 1:
        raise ValueError("max_emerge_coverage must be None or between 0 and 1.")
    if save_intermediate and intermediate_folder is None:
        raise ValueError("intermediate_folder is required when save_intermediate=True.")

    start_time = time.perf_counter()
    array_seeds = _validate_seeds(seeds)
    input_mode = _detect_input_mode(array_seeds, input_mode)

    combine_seed = np.zeros(array_seeds[0].shape, dtype=np.uint32)
    metadata_by_seed_id = {}
    locked_seed_ids = set()
    next_seed_id = 1

    initial_regions = extract_regions(
        array_seeds[0], min_area=min_area, connectivity=connectivity, input_mode=input_mode
    )
    for region in initial_regions:
        combine_seed[region["mask"]] = next_seed_id
        metadata_by_seed_id[next_seed_id] = {
            "area": int(region["area"]),
            "source_label": int(region["source_label"]),
            "start_index": 0,
            "last_index": 0,
            "finalization_reason": None,
            "status": "active",
        }
        next_seed_id += 1

    if verbose:
        print(
            f"[s2l] start: seeds={len(array_seeds)}, "
            f"shape={array_seeds[0].shape}, input_mode={input_mode}, "
            f"initial_regions={len(initial_regions)}"
        )

    if save_intermediate:
        _save_intermediate_seed(combine_seed, intermediate_folder, intermediate_prefix, 0)

    for seed_index, seed in enumerate(array_seeds[1:], start=1):
        step_start = time.perf_counter()

        if vectorize_overlap:
            regions, new_labeled = extract_regions(
                seed, min_area=min_area, connectivity=connectivity,
                input_mode=input_mode, return_labeled=True,
            )
            emerged_count, grown_count, merge_count, skipped_locked_count, next_seed_id = (
                _step_vectorized(
                    new_labeled, regions, combine_seed, metadata_by_seed_id,
                    locked_seed_ids, next_seed_id, seed_index,
                    min_current_coverage, max_emerge_coverage,
                )
            )
        else:
            regions = extract_regions(
                seed, min_area=min_area, connectivity=connectivity, input_mode=input_mode
            )
            emerged_count = grown_count = merge_count = skipped_locked_count = 0

            for region in regions:
                overlap_values = combine_seed[region["mask"]]

                if min_current_coverage is None and max_emerge_coverage is None:
                    significant_seed_ids = [
                        int(sid) for sid in np.unique(overlap_values) if sid != 0
                    ]
                    is_emerge = len(significant_seed_ids) == 0
                else:
                    overlap_values = overlap_values[overlap_values != 0]
                    if overlap_values.size == 0:
                        significant_seed_ids = []
                        max_curr_overlap_ratio = 0.0
                    else:
                        overlap_counts = np.bincount(overlap_values)
                        max_overlap_count = int(overlap_counts.max())
                        max_curr_overlap_ratio = max_overlap_count / region["area"]
                        significant_seed_ids = []
                        for sid in np.flatnonzero(overlap_counts):
                            cnt = int(overlap_counts[sid])
                            seed_area = metadata_by_seed_id[int(sid)]["area"]
                            ratio = cnt / seed_area if seed_area else 0.0
                            if min_current_coverage is None or ratio >= min_current_coverage:
                                significant_seed_ids.append(int(sid))

                    is_emerge = len(significant_seed_ids) == 0
                    if is_emerge and max_emerge_coverage is not None:
                        is_emerge = max_curr_overlap_ratio <= max_emerge_coverage

                if is_emerge:
                    combine_seed[region["mask"]] = next_seed_id
                    metadata_by_seed_id[next_seed_id] = {
                        "area": int(region["area"]),
                        "source_label": int(region["source_label"]),
                        "start_index": int(seed_index),
                        "last_index": int(seed_index),
                        "finalization_reason": None,
                        "status": "active",
                    }
                    next_seed_id += 1
                    emerged_count += 1
                    continue

                if len(significant_seed_ids) == 0:
                    continue

                if len(significant_seed_ids) == 1:
                    seed_id = significant_seed_ids[0]
                    if seed_id in locked_seed_ids:
                        skipped_locked_count += 1
                        continue
                    combine_seed[combine_seed == seed_id] = 0
                    combine_seed[region["mask"]] = seed_id
                    metadata_by_seed_id[seed_id]["area"] = int(region["area"])
                    metadata_by_seed_id[seed_id]["source_label"] = int(region["source_label"])
                    metadata_by_seed_id[seed_id]["last_index"] = int(seed_index)
                    grown_count += 1
                    continue

                merge_count += 1
                for seed_id in significant_seed_ids:
                    locked_seed_ids.add(seed_id)
                    metadata_by_seed_id[seed_id]["status"] = "locked_pre_merge"
                    if metadata_by_seed_id[seed_id]["finalization_reason"] is None:
                        metadata_by_seed_id[seed_id]["finalization_reason"] = "pre_merge"
                        metadata_by_seed_id[seed_id]["merge_index"] = int(seed_index)
                        metadata_by_seed_id[seed_id]["merged_with"] = [
                            int(item) for item in significant_seed_ids
                        ]

        if save_intermediate:
            _save_intermediate_seed(combine_seed, intermediate_folder, intermediate_prefix, seed_index)

        if verbose and (seed_index % log_every == 0 or seed_index == len(array_seeds) - 1):
            print(
                f"[s2l] step {seed_index}/{len(array_seeds) - 1}: "
                f"regions={len(regions)}, grown={grown_count}, emerged={emerged_count}, "
                f"merge_regions={merge_count}, locked={len(locked_seed_ids)}, "
                f"skipped_locked={skipped_locked_count}, seeds={next_seed_id - 1}, "
                f"step={time.perf_counter() - step_start:.2f}s, "
                f"total={time.perf_counter() - start_time:.2f}s"
            )

    for seed_id, item in metadata_by_seed_id.items():
        if item["finalization_reason"] is None:
            item["finalization_reason"] = "never_merged"
        item["area"] = int((combine_seed == seed_id).sum())

    # Drop ghost seeds with no voxels left (e.g. an emerged seed later fully
    # overwritten by a neighbouring grow); they hold no pixels in combine_seed.
    metadata_by_seed_id = {
        sid: item for sid, item in metadata_by_seed_id.items() if item["area"] > 0
    }

    combine_seed, metadata_by_seed_id = _apply_top_n(combine_seed, metadata_by_seed_id, top_n)
    seed_metadata = _metadata_list(metadata_by_seed_id)

    if verbose:
        print(f"[s2l] done: seeds={len(seed_metadata)}, elapsed={time.perf_counter() - start_time:.2f}s")

    return combine_seed, seed_metadata


def adaptive_seed_s2l_from_tiff_files(file_paths, **kwargs):
    load_kwargs = {}
    for key in ("sort_by_threshold", "strict_to_loose"):
        if key in kwargs:
            load_kwargs[key] = kwargs.pop(key)
    seeds, ordered_paths = load_threshold_sweep_seeds(file_paths, **load_kwargs)
    combine_seed, seed_metadata = adaptive_seed_s2l_from_seeds(seeds, **kwargs)
    return combine_seed, seed_metadata, ordered_paths


# ---------------------------------------------------------------------------
# Public API: loose-to-strict
# ---------------------------------------------------------------------------

def adaptive_seed_l2s_from_seeds(
    seeds,
    min_area=1,
    connectivity=1,
    input_mode="auto",
    min_new_coverage=0.5,
    max_new_coverage=0.05,
    keep_disappeared=True,
    top_n=None,
    save_intermediate=False,
    intermediate_folder=None,
    intermediate_prefix="adaptive_seed_l2s",
    verbose=False,
    log_every=1,
):
    """Loose-to-strict adaptive seed: shrink / split / disappear events."""
    if min_area < 1:
        raise ValueError("min_area must be >= 1.")
    if log_every < 1:
        raise ValueError("log_every must be >= 1.")
    if top_n is not None and top_n < 1:
        raise ValueError("top_n must be None or >= 1.")
    if min_new_coverage is not None and not 0 <= min_new_coverage <= 1:
        raise ValueError("min_new_coverage must be None or between 0 and 1.")
    if max_new_coverage is not None and not 0 <= max_new_coverage <= 1:
        raise ValueError("max_new_coverage must be None or between 0 and 1.")
    if save_intermediate and intermediate_folder is None:
        raise ValueError("intermediate_folder is required when save_intermediate=True.")

    start_time = time.perf_counter()
    array_seeds = _validate_seeds(seeds)
    input_mode = _detect_input_mode(array_seeds, input_mode)

    combine_seed = np.zeros(array_seeds[0].shape, dtype=np.uint32)
    metadata_by_seed_id = {}
    locked_seed_ids = set()
    next_seed_id = 1

    initial_regions = extract_regions(
        array_seeds[0], min_area=min_area, connectivity=connectivity, input_mode=input_mode
    )
    for region in initial_regions:
        combine_seed[region["mask"]] = next_seed_id
        metadata_by_seed_id[next_seed_id] = {
            "area": int(region["area"]),
            "source_label": int(region["source_label"]),
            "start_index": 0,
            "last_index": 0,
            "status": "active",
            "finalization_reason": None,
        }
        next_seed_id += 1

    if verbose:
        print(
            f"[l2s] start: seeds={len(array_seeds)}, "
            f"shape={array_seeds[0].shape}, input_mode={input_mode}, "
            f"initial_regions={len(initial_regions)}"
        )

    if save_intermediate:
        _save_intermediate_seed(combine_seed, intermediate_folder, intermediate_prefix, 0)

    for seed_index, seed in enumerate(array_seeds[1:], start=1):
        step_start = time.perf_counter()
        old_seed_ids = _current_seed_ids(combine_seed)
        regions = extract_regions(
            seed, min_area=min_area, connectivity=connectivity, input_mode=input_mode
        )

        old_to_regions = {sid: [] for sid in old_seed_ids}
        new_region_ids = []
        ignored_region_count = ambiguous_region_count = 0

        for region_id, region in enumerate(regions):
            chosen, is_ambiguous = _choose_old_seed_for_region(
                combine_seed, region, min_new_coverage, max_new_coverage
            )
            if is_ambiguous:
                ambiguous_region_count += 1
            if chosen == "ignore":
                ignored_region_count += 1
            elif chosen is None:
                new_region_ids.append(region_id)
            elif chosen in old_to_regions:
                old_to_regions[chosen].append(region_id)

        shrink_count = split_count = disappeared_count = new_count = 0

        for old_seed_id in old_seed_ids:
            if old_seed_id in locked_seed_ids:
                continue
            assigned = old_to_regions[old_seed_id]
            if len(assigned) == 0:
                disappeared_count += 1
                metadata_by_seed_id[old_seed_id]["status"] = "disappeared"
                metadata_by_seed_id[old_seed_id]["finalization_reason"] = "disappeared"
                metadata_by_seed_id[old_seed_id]["disappear_index"] = int(seed_index)
                # Finalize once: lock the seed so it is not re-matched or
                # re-flagged on later (stricter) steps. keep_disappeared only
                # decides whether its last-shape voxels are kept.
                locked_seed_ids.add(old_seed_id)
                if not keep_disappeared:
                    combine_seed[combine_seed == old_seed_id] = 0
                continue

            assigned = sorted(assigned, key=lambda rid: regions[rid]["area"], reverse=True)
            combine_seed[combine_seed == old_seed_id] = 0

            if len(assigned) == 1:
                region = regions[assigned[0]]
                combine_seed[region["mask"]] = old_seed_id
                metadata_by_seed_id[old_seed_id]["area"] = int(region["area"])
                metadata_by_seed_id[old_seed_id]["source_label"] = int(region["source_label"])
                metadata_by_seed_id[old_seed_id]["last_index"] = int(seed_index)
                metadata_by_seed_id[old_seed_id]["status"] = "active"
                shrink_count += 1
                continue

            split_count += 1
            child_seed_ids = [old_seed_id]
            first_region = regions[assigned[0]]
            combine_seed[first_region["mask"]] = old_seed_id
            metadata_by_seed_id[old_seed_id]["area"] = int(first_region["area"])
            metadata_by_seed_id[old_seed_id]["source_label"] = int(first_region["source_label"])
            metadata_by_seed_id[old_seed_id]["last_index"] = int(seed_index)
            metadata_by_seed_id[old_seed_id]["status"] = "active"
            metadata_by_seed_id[old_seed_id].setdefault("split_events", []).append(
                {"split_index": int(seed_index), "parent_seed_id": int(old_seed_id)}
            )
            for rid in assigned[1:]:
                region = regions[rid]
                combine_seed[region["mask"]] = next_seed_id
                child_seed_ids.append(next_seed_id)
                metadata_by_seed_id[next_seed_id] = {
                    "area": int(region["area"]),
                    "source_label": int(region["source_label"]),
                    "start_index": int(seed_index),
                    "last_index": int(seed_index),
                    "status": "active",
                    "finalization_reason": None,
                    "parent_seed_id": int(old_seed_id),
                    "split_from": int(old_seed_id),
                    "split_index": int(seed_index),
                }
                next_seed_id += 1
            metadata_by_seed_id[old_seed_id]["split_events"][-1]["child_seed_ids"] = [
                int(sid) for sid in child_seed_ids
            ]

        for region_id in new_region_ids:
            region = regions[region_id]
            combine_seed[region["mask"]] = next_seed_id
            metadata_by_seed_id[next_seed_id] = {
                "area": int(region["area"]),
                "source_label": int(region["source_label"]),
                "start_index": int(seed_index),
                "last_index": int(seed_index),
                "status": "active",
                "finalization_reason": None,
                "created_as": "new_region",
            }
            next_seed_id += 1
            new_count += 1

        if save_intermediate:
            _save_intermediate_seed(combine_seed, intermediate_folder, intermediate_prefix, seed_index)

        if verbose and (seed_index % log_every == 0 or seed_index == len(array_seeds) - 1):
            print(
                f"[l2s] step {seed_index}/{len(array_seeds) - 1}: "
                f"regions={len(regions)}, shrink={shrink_count}, split={split_count}, "
                f"disappeared={disappeared_count}, new={new_count}, "
                f"ambiguous={ambiguous_region_count}, ignored={ignored_region_count}, "
                f"seeds={next_seed_id - 1}, step={time.perf_counter() - step_start:.2f}s, "
                f"total={time.perf_counter() - start_time:.2f}s"
            )

    for seed_id, item in metadata_by_seed_id.items():
        if item["finalization_reason"] is None:
            item["finalization_reason"] = "active_at_end"
        item["area"] = int((combine_seed == seed_id).sum())

    combine_seed, metadata_by_seed_id = _apply_top_n(combine_seed, metadata_by_seed_id, top_n)
    seed_metadata = _metadata_list(metadata_by_seed_id)

    if verbose:
        print(f"[l2s] done: seeds={len(seed_metadata)}, elapsed={time.perf_counter() - start_time:.2f}s")

    return combine_seed, seed_metadata


def adaptive_seed_l2s_from_tiff_files(file_paths, **kwargs):
    load_kwargs = {}
    for key in ("sort_by_threshold", "strict_to_loose"):
        if key in kwargs:
            load_kwargs[key] = kwargs.pop(key)
    if "strict_to_loose" not in load_kwargs:
        load_kwargs["strict_to_loose"] = False
    seeds, ordered_paths = load_threshold_sweep_seeds(file_paths, **load_kwargs)
    combine_seed, seed_metadata = adaptive_seed_l2s_from_seeds(seeds, **kwargs)
    return combine_seed, seed_metadata, ordered_paths


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def adaptive_seed_combine_from_seeds(seeds, direction="strict_to_loose", **kwargs):
    """Run combine in one or both directions.

    direction: "strict_to_loose" | "loose_to_strict" | "both"

    For "both", kwargs are split by prefix: s2l_* go to strict-to-loose,
    l2s_* go to loose-to-strict, unprefixed shared params go to both.
    Returns (combine_seed, metadata) for single direction, or
    ((s2l_seed, s2l_meta), (l2s_seed, l2s_meta)) for "both".
    """
    if direction == "strict_to_loose":
        return adaptive_seed_s2l_from_seeds(seeds, **kwargs)
    if direction == "loose_to_strict":
        return adaptive_seed_l2s_from_seeds(seeds, **kwargs)
    if direction == "both":
        shared = {k: v for k, v in kwargs.items() if k in COMBINE_SHARED_KEYS}
        s2l_kwargs = {**shared, **{k: v for k, v in kwargs.items() if k in COMBINE_S2L_KEYS}}
        l2s_kwargs = {**shared, **{k: v for k, v in kwargs.items() if k in COMBINE_L2S_KEYS}}

        if "intermediate_prefix" not in s2l_kwargs:
            s2l_kwargs["intermediate_prefix"] = "s2l"
        if "intermediate_prefix" not in l2s_kwargs:
            l2s_kwargs["intermediate_prefix"] = "l2s"

        r_s2l = adaptive_seed_s2l_from_seeds(seeds, **s2l_kwargs)
        r_l2s = adaptive_seed_l2s_from_seeds(seeds, **l2s_kwargs)
        return r_s2l, r_l2s

    raise ValueError(f"direction must be 'strict_to_loose', 'loose_to_strict', or 'both'. Got: {direction!r}")


def adaptive_seed_combine_from_tiff_files(file_paths, direction="strict_to_loose", **kwargs):
    """Like adaptive_seed_combine_from_seeds but loads from TIFF paths."""
    load_kwargs = {}
    for key in ("sort_by_threshold", "strict_to_loose"):
        if key in kwargs:
            load_kwargs[key] = kwargs.pop(key)

    if direction == "loose_to_strict" and "strict_to_loose" not in load_kwargs:
        load_kwargs["strict_to_loose"] = False

    seeds, ordered_paths = load_threshold_sweep_seeds(file_paths, **load_kwargs)
    result = adaptive_seed_combine_from_seeds(seeds, direction=direction, **kwargs)

    if direction == "both":
        return result, ordered_paths
    combine_seed, seed_metadata = result
    return combine_seed, seed_metadata, ordered_paths
