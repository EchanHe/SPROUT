"""
Checks for the (legacy) adaptive seed functions in make_adaptive_seed.py.

    1. saving intermediate results never changes the final result
       (once prune_every_iter is set explicitly)
    2. the threshold mode does not modify the caller's image
    3. the threshold mode applies the boundary from the first threshold on
    4. logs: the boundary is reported for in-memory arrays, prune_every_iter is
       printed, and the erosion-mode CSV records one seed file per iteration
    5. prune_every_iter is accepted from a YAML config

Run in the `sprout` env:
    python tests/check_adaptive_seed.py
"""
import contextlib
import glob
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import yaml  # noqa: E402

from make_adaptive_seed import make_adaptive_seed_ero, make_adaptive_seed_thre  # noqa: E402
from sprout_core import config_core  # noqa: E402

FAILED = []


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILED.append(msg)


def run(fn, img, **kw):
    """Returns (seeds_dict, captured stdout, output folder)."""
    out = tempfile.mkdtemp()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        seeds, _, _ = fn(img=img, output_folder=out, num_threads=4, return_for_napari=True, **kw)
    return seeds, buf.getvalue(), out


def final(seeds):
    return [v for k, v in seeds.items() if "FINAL" in k][0]


dog = tifffile.imread(ROOT / "demo_data" / "dog_img.tif")
ERO = dict(threshold=150, erosion_steps=5, segments=20, min_size=5)
THRE = dict(thresholds=[100, 120, 140, 160, 180, 200], erosion_steps=1, segments=8, min_size=50)

# 1. saving does not change the result once pruning is decided explicitly
for name, fn, kw in [("erosion", make_adaptive_seed_ero, ERO), ("threshold", make_adaptive_seed_thre, THRE)]:
    for prune in (False, True):
        a = final(run(fn, dog.copy(), save_every_iter=False, prune_every_iter=prune, **kw)[0])
        b = final(run(fn, dog.copy(), save_every_iter=True, prune_every_iter=prune, **kw)[0])
        ok(np.array_equal(a, b), f"{name}, prune_every_iter={prune}: save_every_iter on/off -> identical result")

# a block cut in half by a boundary plane; the two halves differ in brightness so the
# threshold steps matter
block = np.zeros((20, 30, 60), np.uint8)
block[4:16, 5:25, 5:30] = 200
block[4:16, 5:25, 30:55] = 160
cut = np.zeros(block.shape, np.uint8)
cut[:, :, 29:31] = 1

# 2. the caller's image is not modified
img = block.copy()
seeds, log, out = run(make_adaptive_seed_thre, img, boundary=cut, thresholds=[100, 150], erosion_steps=0,
                      segments=10, save_every_iter=True)
ok(np.array_equal(img, block), "threshold mode: caller's image unchanged (it used to be zeroed on the boundary)")

# 3. the boundary applies from the first threshold
first = [v for k, v in seeds.items() if k.startswith("INTER_thre_100")]
ok(bool(first) and not ((first[0] > 0) & (cut > 0)).any() and len(np.unique(first[0])) - 1 == 2,
   "threshold mode: the first-threshold seed already excludes the boundary and is split in two")
f = final(seeds)
ok(not ((f > 0) & (cut > 0)).any() and len(np.unique(f)) - 1 == 2, "threshold mode: final seed has no boundary voxels, 2 regions")

# 4. logs
ok("Boundary: array (20, 30, 60)" in log, "log reports an in-memory boundary (used to print boundary_path: None)")
ok("Prune Every Iteration: True" in log, "log prints the effective prune_every_iter (None -> follows save_every_iter)")
seeds, log, out = run(make_adaptive_seed_ero, dog.copy(), save_every_iter=True, **ERO)
csv = sorted(glob.glob(os.path.join(out, "**", "output_dict_*.csv"), recursive=True))[-1]   # written into out/<base_name>/
names = pd.read_csv(csv)["cur_seed_name"].dropna().tolist()
ok(len(names) == len(set(names)) and len(names) > 2,
   f"erosion mode: CSV keeps one seed file per iteration ({len(names)} distinct names; used to keep only 2)")

# 5. YAML config accepts the new key
cfg = yaml.safe_load(open(ROOT / "template" / "make_adaptive_seed.yaml", encoding="utf-8"))
cfg["num_threads"] = 1          # the template asks for 4, more than a small machine allows
cfg["prune_every_iter"] = False
try:
    opt = config_core.validate_input_yaml(cfg, config_core.input_val_make_adaptive_seed)
    ok(opt["prune_every_iter"] is False, "YAML: prune_every_iter accepted")
except Exception as e:
    ok(False, f"YAML: prune_every_iter rejected: {e}")
cfg.pop("prune_every_iter")
opt = config_core.validate_input_yaml(cfg, config_core.input_val_make_adaptive_seed)
ok(opt["prune_every_iter"] is None, "YAML: prune_every_iter defaults to None (legacy behaviour)")

print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
sys.exit(1 if FAILED else 0)
