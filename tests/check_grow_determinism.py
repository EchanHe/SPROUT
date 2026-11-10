"""
Checks that Grow gives the same result for any number of threads, equal to the
original single-threaded algorithm.

The old multi-threaded grow let threads read the label map while others were
writing to it, so contested voxels went to whichever thread was faster (on the dog
demo: 300 voxels differ between 1 and 8 threads, and between two 8-thread runs).

Run in the `sprout` env:
    python tests/check_grow_determinism.py
"""
import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile
from skimage.morphology import ball, binary_dilation, disk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from sprout_core import make_grow  # noqa: E402

FAILED = []


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILED.append(msg)


def reference_one_iter(input_mask, threshold_binary, to_grow_ids=None, boundary=None):
    """Single-threaded reference, touch_rule='stop' (ground truth).

    Only label voxels inside the allowed region (threshold range and not boundary)
    are dilated; seed voxels outside it are kept but never grow (rule "B2").
    """
    ids = np.unique(input_mask)
    ids = ids[ids != 0]
    if to_grow_ids is not None:
        ids = np.intersect1d(ids, to_grow_ids)
    result = input_mask.copy()
    non_bg = result != 0
    allowed = threshold_binary.astype(bool)
    if boundary is not None:
        allowed &= ~boundary.astype(bool)
    fp = ball(1) if input_mask.ndim == 3 else disk(1)
    for lid in ids:                                   # ascending id order
        d = binary_dilation((result == lid) & allowed, fp)
        d[(result != lid) & non_bg & d] = False
        result[d] = lid
    if boundary is not None:
        result[boundary] = 0
    result[(~non_bg) & (~threshold_binary)] = 0
    return result


rng = np.random.default_rng(1)
n = bad = 0
for trial in range(30):
    shape = (12, 30, 30) if trial % 3 else (60, 60)
    seg = np.zeros(shape, np.uint16)
    for lid in range(1, int(rng.integers(5, 40))):
        pos = tuple(int(rng.integers(0, s)) for s in shape)
        sl = tuple(slice(max(p - 2, 0), p + int(rng.integers(1, 4))) for p in pos)
        seg[sl] = lid
    thr = rng.random(shape) > 0.2
    bnd = rng.random(shape) > 0.95 if trial % 2 else None
    ids = [1, 2, 3, 5, 8] if trial % 5 == 0 else None
    ref = reference_one_iter(seg, thr, ids, bnd)
    for t in (1, 3, 8):
        n += 1
        bad += not np.array_equal(make_grow.dilation_one_iter_mp(seg, thr, t, to_grow_ids=ids, boundary=bnd), ref)
ok(bad == 0, f"one iteration, 30 random crowded label maps (3D/2D, boundary, to_grow_ids) x 1/3/8 threads: "
             f"{n - bad}/{n} equal to the single-threaded reference")

dog = tifffile.imread(ROOT / "demo_data" / "dog_img.tif")
ds = tifffile.imread(ROOT / "demo_data" / "dog_seed.tif")


def grow(threads):
    with contextlib.redirect_stdout(io.StringIO()):
        out, _ = make_grow.grow_mp(img=dog, seg=ds, thresholds=[180, 150, 120], upper_thresholds=[255, 255, 255],
                                   dilation_steps=[4, 4, 4], output_folder=tempfile.mkdtemp(), num_threads=threads,
                                   return_for_napari=True)
    return out


ref = grow(1)
for t in (2, 8, 8, 8):
    r = grow(t)
    ok(sorted(r) == sorted(ref) and all(np.array_equal(r[k], ref[k]) for k in ref),
       f"dog grow_mp: {t} threads == 1 thread (all {len(ref)} outputs)")

# ---- seeds outside the allowed region must not grow out (they used to)
def grow_small(img, seg, thresholds, steps, boundary=None):
    with contextlib.redirect_stdout(io.StringIO()):
        out, _ = make_grow.grow_mp(img=img, seg=seg, boundary=boundary, thresholds=thresholds,
                                   upper_thresholds=[255] * len(thresholds), dilation_steps=steps,
                                   output_folder=tempfile.mkdtemp(), num_threads=4, return_for_napari=True,
                                   is_sort=False)
    return [v for k, v in out.items() if "FINAL" in k][0]


flat = np.full((20, 30, 60), 200, np.uint8)
seed = np.zeros(flat.shape, np.uint8)
seed[10, 15, 30], seed[10, 15, 5] = 1, 2
plane = np.zeros(flat.shape, bool)
plane[:, :, 30] = True
r = grow_small(flat, seed, [100], [10], boundary=plane)
ok((r == 1).sum() == 0 and (r == 2).sum() > 0, "seed ON a 1-voxel boundary does not grow to both sides (used to)")
slab = np.zeros(flat.shape, bool)
slab[:, :, 25:35] = True
s2 = np.zeros(flat.shape, np.uint8)
s2[10, 15, 34] = 1
ok((grow_small(flat, s2, [100], [10], boundary=slab) == 1).sum() == 0,
   "seed inside a thick boundary, at its edge, does not grow out (used to)")
gap = flat.copy()
gap[:, :, 30] = 50                                    # a dark 1-voxel suture below the threshold
s3 = np.zeros(flat.shape, np.uint8)
s3[10, 15, 30] = 1
r = grow_small(gap, s3, [100], [10])
ok((r == 1).sum() == 1 and r[10, 15, 30] == 1, "seed ON a dark gap is kept but does not grow into the bones (used to bridge them)")
r = grow_small(gap, s3, [100, 40], [10, 10])
ok((r == 1).sum() > 1, "...and grows from there once a looser stage includes the gap")

# ---- early stop switched off (napari: "Early stop if no growth" unticked) used to raise TypeError
def grow_early_stop(no_growth_max_iter):
    with contextlib.redirect_stdout(io.StringIO()):
        out, _ = make_grow.grow_mp(img=dog, seg=ds, thresholds=[150], upper_thresholds=[255], dilation_steps=[6],
                                   output_folder=tempfile.mkdtemp(), num_threads=4, return_for_napari=True,
                                   min_growth_size=10**9, no_growth_max_iter=no_growth_max_iter)
    return out


try:
    off = [v for k, v in grow_early_stop(None).items() if "FINAL" in k][0]
    on = [v for k, v in grow_early_stop(1).items() if "FINAL" in k][0]
    ok(int((off > 0).sum()) > int((on > 0).sum()),
       f"no_growth_max_iter=None grows all 6 steps ({int((off > 0).sum())} voxels; "
       f"early stop after 1 step: {int((on > 0).sum())})")
except TypeError as e:
    ok(False, f"no_growth_max_iter=None raised TypeError: {e}")

print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
sys.exit(1 if FAILED else 0)
