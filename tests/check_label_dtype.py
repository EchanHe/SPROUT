"""
Checks that label ids survive Grow / Seeds / adaptive seeds, whatever their values.

Grow used to pick uint8 vs uint16 from the NUMBER of labels, so ids {7, 300}
came back as {7, 44} (300 wraps to 44 in uint8) and could merge with label 44.

Run in the `sprout` env:
    python tests/check_label_dtype.py
"""
import contextlib
import glob
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from sprout_core.make_grow import grow_mp  # noqa: E402
from sprout_core.make_seeds import make_seeds  # noqa: E402
from sprout_core import sprout_core as sc  # noqa: E402

FAILED = []


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILED.append(msg)


def quiet(fn, **kw):
    out = tempfile.mkdtemp()
    with contextlib.redirect_stdout(io.StringIO()):
        res = fn(output_folder=out, return_for_napari=True, **kw)
    return res, out


def grow_final(img, seg, **kw):
    kw = dict(thresholds=[100], upper_thresholds=[255], dilation_steps=[3], num_threads=1, is_sort=False) | kw
    (out, _), _ = quiet(grow_mp, img=img, seg=seg, **kw)
    return [v for k, v in out.items() if "FINAL" in k][0]


# ---- the helper
cases = [(0, np.uint8, "uint8"), (255, np.uint8, "uint8"), (256, np.uint8, "uint16"),
         (65535, np.uint8, "uint16"), (65536, np.uint8, "uint32"), (5, np.uint16, "uint16"),
         (2 ** 32, np.uint8, "uint64")]
for max_id, mn, expect in cases:
    got = str(sc.label_dtype(max_id, mn))
    ok(got == expect, f"label_dtype({max_id}, min={np.dtype(mn)}) = {got}")
a = np.zeros(3, np.uint16)
ok(sc.ensure_label_capacity(a, 70000).dtype == np.uint32 and sc.ensure_label_capacity(a, 5) is a,
   "ensure_label_capacity widens only when needed")

# ---- Grow keeps ids
img = np.full((10, 20, 40), 200, np.uint8)
seg = np.zeros(img.shape, np.uint16)
seg[5, 10, 5], seg[5, 10, 35] = 7, 300
f = grow_final(img, seg)
ok(sorted(set(np.unique(f).tolist()) - {0}) == [7, 300], f"grow ids {{7, 300}} kept (old code: {{7, 44}}), dtype {f.dtype}")

seg[5, 10, 5] = 44                                     # 300 % 256 == 44
f = grow_final(img, seg)
ok(ndi.label(f == 44)[1] == 1 and (f == 300).any(), "grow ids {44, 300} stay two regions (old code merged them)")

seg32 = np.zeros(img.shape, np.uint32)
seg32[5, 10, 5], seg32[5, 10, 35] = 5, 70000
f = grow_final(img, seg32)
ok(sorted(set(np.unique(f).tolist()) - {0}) == [5, 70000] and f.dtype == np.uint32, f"grow id 70000 kept, dtype {f.dtype}")

small = np.zeros(img.shape, np.uint16)
small[5, 10, 5], small[5, 10, 35] = 1, 2
ok(grow_final(img, small).dtype == np.uint8, "grow with ids <= 255 still returns uint8 (memory unchanged)")

# relabelling the demo seeds with large, order-preserving ids must not change the result
dog = tifffile.imread(ROOT / "demo_data" / "dog_img.tif")
dseed = tifffile.imread(ROOT / "demo_data" / "dog_seed.tif")
ids = np.unique(dseed)[1:]
lut = np.zeros(int(dseed.max()) + 1, np.uint32)
lut[ids] = np.arange(1, ids.size + 1) * 1000           # 1 -> 1000, 2 -> 2000, ... (order kept)
kw = dict(thresholds=[150, 120], upper_thresholds=[255, 255], dilation_steps=[3, 3])
a = grow_final(dog, dseed, **kw)
b = grow_final(dog, lut[dseed], **kw)
back = np.zeros(int(b.max()) + 1, np.int64)
back[lut[ids]] = ids
ok(np.array_equal(a, back[b]), f"dog seeds ({ids.size} labels) relabelled to ids up to {int(lut.max())}: same grown result")

# ---- Seeds written to disk keep ids above 65535
pts = np.zeros((520, 520), np.uint8)
pts[::2, ::2] = 255                                     # 67,600 isolated pixels -> 67,600 seeds
(seeds, _), out = quiet(make_seeds, img=pts, thresholds=[100], upper_thresholds=[255], erosion_steps=0,
                        segments=70000, num_threads=1)
files = glob.glob(os.path.join(out, "**", "*.tif"), recursive=True)
saved = tifffile.imread(files[0])
ok(saved.dtype == np.uint32 and int(saved.max()) == 67600 and np.unique(saved).size - 1 == 67600,
   f"seed file with 67,600 seeds saved as {saved.dtype}, max id {int(saved.max())} (old code: uint16, wrapped)")

print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
sys.exit(1 if FAILED else 0)
