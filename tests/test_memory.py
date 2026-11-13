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

from sprout_core import make_adaptive_seed, make_grow, make_seeds, make_sweep_adaptive_seed, seed_sweep  # noqa: E402
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
