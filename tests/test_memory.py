"""
Peak memory of seeds and classic adaptive seed: label images of a step are freed before the
next step labels again (traced by tracemalloc, in bytes per voxel of the image).
"""
import contextlib
import io
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sprout_core import make_adaptive_seed  # noqa: E402
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


# Before 1.1.1: seeds 10.9, adaptive erosion 15.3, adaptive thresholds 16.6 bytes per voxel.
def test_seeds_keep_one_label_image_at_a_time(tmp_path):
    img, t = blobs()
    peak = peak_per_voxel(lambda: sprout_core.find_seed_by_ero_custom(img, t[1], 50, 3, str(tmp_path), ["ball"] * 3),
                          img)
    assert peak < 9.5, f"{peak:.1f} bytes per voxel"


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
