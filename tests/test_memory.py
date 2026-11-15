"""
Memory of seeds, classic adaptive seed and sweep, traced by tracemalloc in bytes per voxel of
the image: labels in uint16, label images of a step freed before the next step labels again,
sweep steps in uint8 in memory. Grow keeps its bounding boxes instead of scanning every iteration.
Every result is the same as before.
"""
import contextlib
import io
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import tifffile
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sprout_core import (adaptive_seed_combine, make_adaptive_seed, make_grow, make_seeds,  # noqa: E402
                         make_sweep_adaptive_seed, seed_sweep)
import sprout_core.sprout_core as sprout_core  # noqa: E402

pytestmark = pytest.mark.core


def blobs():
    """Smooth noise: about 16 % of the voxels are above the second threshold."""
    g = ndi.gaussian_filter(np.random.default_rng(0).random((96, 96, 96)), 2)
    img = ((g - g.min()) / (g.max() - g.min()) * 255).astype(np.uint8)
    return img, [int(np.percentile(img, q)) for q in (80, 85, 88, 91)]


def peak_per_voxel(run, img):
    with contextlib.redirect_stdout(io.StringIO()):
        tracemalloc.start()
        try:
            run()
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    return peak / img.size


# Before 1.1.1: seeds 10.9 (4 footprints on 4 threads 38.7), adaptive erosion 15.3, adaptive
# thresholds 16.6, sweep on 4 threads 27.7 bytes per voxel.
def test_seeds_keep_one_label_image_at_a_time(tmp_path):
    img, t = blobs()
    peak = peak_per_voxel(lambda: sprout_core.find_seed_by_ero_custom(img, t[1], 50, 3, str(tmp_path), ["ball"] * 3),
                          img)
    assert peak < 7.0, f"{peak:.1f} bytes per voxel"


def test_seeds_on_four_threads(tmp_path):
    img, t = blobs()
    peak = peak_per_voxel(lambda: make_seeds.make_seeds(img=img, thresholds=[t[1]], erosion_steps=3, segments=50,
                                                        num_threads=4, output_folder=str(tmp_path), base_name="f"), img)
    assert peak < 28, f"{peak:.1f} bytes per voxel"


def test_sweep_on_four_threads(tmp_path):
    img, t = blobs()
    peak = peak_per_voxel(lambda: make_sweep_adaptive_seed.make_sweep_adaptive_seed(
        img, sorted(t, reverse=True), 0, 50, str(tmp_path), num_threads=4), img)
    assert peak < 23, f"{peak:.1f} bytes per voxel"


@pytest.mark.parametrize("mode", ["erosion", "thresholds"])
def test_adaptive_seed_frees_the_labels_of_earlier_steps(tmp_path, mode):
    img, t = blobs()
    common = dict(img=img, segments=50, num_threads=1, output_folder=str(tmp_path), base_name="m",
                  save_every_iter=False)
    if mode == "erosion":
        run = lambda: make_adaptive_seed.make_adaptive_seed_ero(threshold=t[1], erosion_steps=3, **common)  # noqa: E731
    else:
        run = lambda: make_adaptive_seed.make_adaptive_seed_thre(thresholds=t, erosion_steps=0, **common)  # noqa: E731
    peak = peak_per_voxel(run, img)
    assert peak < 12.5, f"{peak:.1f} bytes per voxel"


# uint16 labels: the same as measure.label, also with more than 65535 components (then int32).
def masks():
    rng = np.random.default_rng(1)
    yield "3d blobs", blobs()[0] > 140
    yield "2d", ndi.gaussian_filter(rng.random((200, 300)), 2) > 0.5
    ties = np.zeros((20, 20, 20), bool)
    ties[2:5, 2:5, 2:5] = ties[2:5, 10:13, 2:5] = ties[12:15, 2:5, 10:13] = True         # three of 27 voxels
    yield "equal sizes", ties
    yield "empty", np.zeros((10, 10, 10), bool)


@pytest.mark.parametrize("name,mask", list(masks()), ids=[n for n, _ in masks()])
@pytest.mark.parametrize("segments,min_vol", [(None, None), (5, None), (20, 30)])
def test_uint16_labels_are_those_of_measure_label(name, mask, segments, min_vol):
    old, old_sizes = sprout_core.get_ccomps_with_size_order(mask, segments, min_vol)
    new, new_sizes = sprout_core.get_ccomps_with_size_order(mask, segments, min_vol, uint16=True)
    assert new.dtype == np.uint16 and np.array_equal(new, old) and np.array_equal(new_sizes, old_sizes)


def test_more_than_65535_components_are_labelled_in_int32():
    mask = np.zeros((100, 100, 100), bool)
    mask[::2, ::2, ::2] = True                                     # 125,000 single voxels
    old, _ = sprout_core.get_ccomps_with_size_order(mask, 70000)
    new, _ = sprout_core.get_ccomps_with_size_order(mask, 70000, uint16=True)
    assert new.dtype == np.int32 and np.array_equal(new, old) and new.max() == 70000


def noisy(shape=(32, 128, 256), every=(2, 2, 3)):
    """Single voxels of 200 on 0, more than 65535 of them (88,064), as a noisy scan has at a low
    threshold."""
    img = np.zeros(shape, np.uint8)
    img[::every[0], ::every[1], ::every[2]] = 200
    return img


# Before 1.1.1, more than 65535 components cost 7 bytes per voxel (scipy labelled again into int32,
# 11 bytes in int64 above 2**31 voxels) and two more labellings in every step.
def test_more_than_65535_components_cost_no_more_than_int32():
    mask = noisy((96, 256, 512), (4, 4, 8)) > 100                 # 98,304 in 12.6 million voxels
    peak = peak_per_voxel(lambda: sprout_core.get_ccomps_with_size_order(mask, 40, uint16=True), mask)
    assert peak < 5.5, f"{peak:.1f} bytes per voxel"


@pytest.fixture
def uint16_tries(monkeypatch):
    """How many labellings tried uint16 (scipy's own labelling, which ndi.label also calls)."""
    from scipy.ndimage import _ni_label
    tries, label = [], _ni_label._label

    def counted(mask, structure, out):
        tries.append(out.dtype == np.uint16)
        return label(mask, structure, out)
    monkeypatch.setattr(_ni_label, "_label", counted)
    return tries


def test_after_one_mask_does_not_fit_in_uint16_the_run_labels_in_int32_at_once(uint16_tries):
    mask = noisy() > 100
    old, old_sizes = sprout_core.get_ccomps_with_size_order(mask, 40)
    uint16_tries.clear()
    run = sprout_core.Uint16Labels()
    for _ in range(3):
        new, new_sizes = sprout_core.get_ccomps_with_size_order(mask, 40, uint16=run)
        assert new.dtype == np.int32 and np.array_equal(new, old) and np.array_equal(new_sizes, old_sizes)
    assert uint16_tries == [True, False, False, False]           # one try, then one labelling each
    small = blobs()[0] > 140
    assert sprout_core.get_ccomps_with_size_order(small, 40, uint16=sprout_core.Uint16Labels())[0].dtype == np.uint16
    assert sprout_core.get_ccomps_with_size_order(small, 40, uint16=True)[0].dtype == np.uint16


@pytest.mark.parametrize("method", ["seeds", "sweep thresholds", "sweep erosion", "adaptive erosion",
                                    "adaptive thresholds"])
def test_a_run_tries_uint16_once_on_each_thread_at_most(tmp_path, uint16_tries, method):
    img = noisy()
    common = dict(img=img, segments=40, output_folder=str(tmp_path), base_name="n")
    run = {
        "seeds": lambda: make_seeds.make_seeds(thresholds=[100], erosion_steps=2, num_threads=2, **common),
        "sweep thresholds": lambda: seed_sweep.seed_sweep_threshold(img, str(tmp_path), [100, 150, 190], segments=40,
                                                                    num_threads=2),
        "sweep erosion": lambda: seed_sweep.seed_sweep_erosion(img, str(tmp_path), 100, erosion_steps=2, segments=40),
        "adaptive erosion": lambda: make_adaptive_seed.make_adaptive_seed_ero(
            threshold=100, erosion_steps=2, num_threads=1, save_every_iter=False, **common),
        "adaptive thresholds": lambda: make_adaptive_seed.make_adaptive_seed_thre(
            thresholds=[100, 150, 190], erosion_steps=0, num_threads=1, save_every_iter=False, **common),
    }[method]
    threads = 2 if method in ("seeds", "sweep thresholds") else 1      # their first steps start together
    with contextlib.redirect_stdout(io.StringIO()):
        run()
        first = sum(uint16_tries)
        assert 1 <= first <= threads, f"{first} tries of uint16"
        run()                                                    # a new run tries again
    assert first < sum(uint16_tries) <= 2 * threads


def test_without_scipys_own_labelling_ndi_label_does_it(monkeypatch):
    monkeypatch.setattr(sprout_core, "_ni_label", None)
    for mask, dtype in ((noisy() > 100, np.int32), (blobs()[0] > 140, np.uint16)):
        old, old_sizes = sprout_core.get_ccomps_with_size_order(mask, 40)
        run = sprout_core.Uint16Labels()
        for _ in range(2):
            new, new_sizes = sprout_core.get_ccomps_with_size_order(mask, 40, uint16=run)
            assert new.dtype == dtype and np.array_equal(new, old) and np.array_equal(new_sizes, old_sizes)
        assert run.too_many == (dtype == np.int32)


def test_sweep_keeps_its_steps_in_uint8_and_writes_them_in_uint16(tmp_path):
    img, t = blobs()
    steps = []
    paths, _ = seed_sweep.seed_sweep_threshold(img, str(tmp_path), sorted(t, reverse=True), segments=50,
                                               verbose=False, seeds_out=steps)
    for step, path in zip(steps, paths):
        on_disk = tifffile.imread(path)
        assert step.dtype == np.uint8 and on_disk.dtype == np.uint16 and np.array_equal(step, on_disk)


# Grow: the bounding boxes are found once per run and then kept up to date by the writes.
def grow_case():
    rng = np.random.default_rng(2)
    img = (ndi.gaussian_filter(rng.random((40, 50, 60)), 2) * 255 / 0.65).clip(0, 255).astype(np.uint8)
    seg = np.zeros(img.shape, np.uint8)
    for i, (z, y, x) in enumerate(rng.integers(5, 35, (12, 3)), start=1):
        seg[z:z + 3, y:y + 3, x:x + 3] = i
    boundary = np.zeros(img.shape, bool)
    boundary[:, 25, :] = True                                      # a wall the labels grow up to
    return img, seg, boundary


def test_grow_finds_the_boxes_once_and_gives_the_same_result(tmp_path, monkeypatch):
    img, seg, boundary = grow_case()
    kw = dict(img=img, seg=seg, boundary=boundary, thresholds=[150, 120, 100], dilation_steps=[4, 4, 3],
              touch_rule="stop", num_threads=3, return_for_napari=True, save_every_n_iters=None)
    with contextlib.redirect_stdout(io.StringIO()):
        new = make_grow.grow_mp(output_folder=str(tmp_path / "new"), **kw)[0]
    calls = []
    find_objects = ndi.find_objects

    def scanned(*args, **kwargs):
        calls.append(1)
        return find_objects(*args, **kwargs)

    def every_iteration(*args, boxes=None, **kwargs):         # the boxes found again in every iteration
        return dilation(*args, **kwargs)

    monkeypatch.setattr(make_grow.ndi, "find_objects", scanned)
    with contextlib.redirect_stdout(io.StringIO()):
        make_grow.grow_mp(output_folder=str(tmp_path / "count"), **kw)
    assert len(calls) == 1
    dilation = make_grow.dilation_one_iter_mp
    monkeypatch.setattr(make_grow, "dilation_one_iter_mp", every_iteration)
    with contextlib.redirect_stdout(io.StringIO()):
        old = make_grow.grow_mp(output_folder=str(tmp_path / "old"), **kw)[0]
    assert len(calls) == 1 + sum(kw["dilation_steps"]) and new.keys() == old.keys()
    assert all(np.array_equal(new[k], old[k]) for k in new)
    final = next(v for k, v in new.items() if k.startswith("FINAL"))
    assert np.count_nonzero(final) > 3 * np.count_nonzero(seg) and not final[boundary].any()   # it grew, not past the wall


# The combine of steps on disk (memory_limit_gb): one step in memory at a time and the ids renumbered in
# place. Before: 10.6 (loose to strict) and 11.0 (strict to loose) bytes per voxel; now its uint32 seeds and
# the step it reads, about 7.
def sparse_steps(folder):
    """Steps of a few % of the voxels, as bone in a scan."""
    g = ndi.gaussian_filter(np.random.default_rng(0).random((96, 96, 96)), 2)
    img = ((g - g.min()) / (g.max() - g.min()) * 255).astype(np.uint8)
    thresholds = [int(np.percentile(img, q)) for q in (99, 98, 97, 96)]
    with contextlib.redirect_stdout(io.StringIO()):
        paths, _ = seed_sweep.seed_sweep_threshold(img, str(folder), thresholds, segments=50, verbose=False)
    return img, paths


@pytest.mark.parametrize("direction", ["strict_to_loose", "loose_to_strict"])
def test_the_combine_of_steps_on_disk_keeps_one_step(tmp_path, monkeypatch, direction):
    img, paths = sparse_steps(tmp_path)
    monkeypatch.setattr(sprout_core, "_CHUNK", 1 << 14)            # chunks much smaller than the image, as in a scan
    combine, steps = {"strict_to_loose": (adaptive_seed_combine.adaptive_seed_s2l_from_seeds, paths),
                      "loose_to_strict": (adaptive_seed_combine.adaptive_seed_l2s_from_seeds, paths[::-1])}[direction]
    peak = peak_per_voxel(lambda: combine(seed_sweep.StepFiles(steps), min_area=1, input_mode="label", verbose=False),
                          img)
    assert peak < 8.5, f"{peak:.1f} bytes per voxel"


def test_the_ids_are_renumbered_in_place():
    rng = np.random.default_rng(3)
    combine_seed = rng.integers(0, 30, (20, 20, 20)).astype(np.uint32)
    metadata = {sid: {"status": "active"} for sid in range(1, 30)}
    expected, expected_metadata = adaptive_seed_combine._apply_top_n(combine_seed.copy(), dict(metadata), 10)
    renumbered, new_metadata = adaptive_seed_combine._apply_top_n(combine_seed, dict(metadata), 10)
    assert renumbered is combine_seed and np.array_equal(renumbered, expected) and new_metadata == expected_metadata
    assert renumbered.max() == 10
