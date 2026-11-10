"""
Checks the core speed-ups of the napari-sprout audit (B1, B3, B4, B5, B8): every output
is identical to the functions of the base commit (loaded with `git show <base>:<file>`),
and each is faster (or, for B8, counts fewer voxels per iteration).

    B1  reorder_segmentation: one bincount + a lookup table instead of two full scans per id
    B3  get_ccomps_with_size_order: no np.unique, one lookup table instead of a scan per id
    B4  bounding box / emptiness from any() projections instead of np.argwhere
    B5  check_and_cast_boundary: max / min / count_nonzero instead of two np.unique
    B8  Grow: one non-zero count per iteration (the previous output size is reused)

Data: the dog demo, synthetic crowded label maps (hundreds of labels, ids with gaps,
equal sizes to test the tie order), and an ant crop if scratchpad ant_bench is present
(set SPROUT_ANT_SCAN / SPROUT_ANT_LABEL to other files).

Run in the `sprout` env:
    python tests/check_core_speed.py [--base 4cec0bc] [--no-ant]
Prints PASS/FAIL per check; exits non-zero if anything failed.
"""
import argparse
import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sprout_core import make_grow  # noqa: E402
import sprout_core.config_core as cc  # noqa: E402
import sprout_core.sprout_core as sc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="4cec0bc")
ap.add_argument("--repo", default=str(ROOT), help="git repository to read the base commit from")
ap.add_argument("--no-ant", action="store_true")
ARGS = ap.parse_args()

FAILED = []
TMP = Path(tempfile.mkdtemp(prefix="sprout_core_speed_"))


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILED.append(msg)


def base_module(rel, name):
    src = subprocess.run(["git", "-C", ARGS.repo, "show", f"{ARGS.base}:{rel}"], capture_output=True,
                         check=True).stdout
    path = TMP / f"{name}.py"
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base_sc = base_module("sprout_core/sprout_core.py", "base_sprout_core")
base_cc = base_module("sprout_core/config_core.py", "base_config_core")
try:
    base_grow = base_module("sprout_core/make_grow.py", "base_make_grow")
except subprocess.CalledProcessError:   # bases before 1.1.0 keep it at the root
    base_grow = base_module("make_grow.py", "base_make_grow")


def best(fn, n=3):
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return min(t)


def outcome(fn, *a, **k):
    """('ok', result) or ('err', type name, message)."""
    try:
        return ("ok", fn(*a, **k))
    except Exception as e:  # noqa: BLE001
        return ("err", type(e).__name__, str(e))


def same(a, b):
    """Deep equality: arrays by dtype + shape + values, dicts by key / value types too."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype
                and a.shape == b.shape and np.array_equal(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        ka, kb = list(a), list(b)
        return (len(ka) == len(kb) and all(type(x) is type(y) and x == y for x, y in zip(ka, kb))
                and all(same(a[x], b[y]) for x, y in zip(ka, kb)))
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


# ---------------------------------------------------------------------- data
rng = np.random.default_rng(0)
dog = tifffile.imread(ROOT / "demo_data" / "dog_img.tif")
dog_seed = tifffile.imread(ROOT / "demo_data" / "dog_seed.tif")


def crowded(shape, n, seed=1, dtype=np.uint16, max_id=5000):
    """n Voronoi cells (random ids with gaps) on blobs, plus equal-size cubes (ties)."""
    r = np.random.default_rng(seed)
    pts = np.zeros(shape, bool)
    pts[tuple(r.integers(0, s, n) for s in shape)] = True
    _, idx = ndi.distance_transform_edt(~pts, return_indices=True)
    cell = np.zeros(shape, np.int64)
    cell[pts] = np.arange(1, pts.sum() + 1)
    cell = cell[tuple(idx)]
    blobs = ndi.gaussian_filter(r.random(shape, np.float32), 3) > 0.5
    ids = r.permutation(np.arange(1, max_id))[:cell.max() + 1]
    ids[0] = 0
    lab = np.where(blobs, ids[cell], 0).astype(dtype)
    for k in range(6):                                       # six 4x4x4 cubes: equal sizes
        z = 2 + 7 * k
        lab[z:z + 4, 2:6, 2:6] = max_id + 10 - k
        lab[z + 4:z + 7, :8, :8] = 0
    return lab


def walls(lab):
    """Binary map of the labels with a one-voxel gap between touching labels (many components)."""
    b = lab != 0
    for ax in range(lab.ndim):
        d = np.diff(lab.astype(np.int64), axis=ax) != 0
        sl = [slice(None)] * lab.ndim
        sl[ax] = slice(1, None)
        b[tuple(sl)] &= ~d
    return b


def cubes(shape, spacing=20, seed=2):
    """One cube (edge 2-6, many equal sizes) per grid cell: hundreds of separate components."""
    r = np.random.default_rng(seed)
    b = np.zeros(shape, bool)
    for z in range(0, shape[0] - spacing + 1, spacing):
        for y in range(0, shape[1] - spacing + 1, spacing):
            for x in range(0, shape[2] - spacing + 1, spacing):
                e = int(r.integers(2, 7))
                o = r.integers(0, spacing - e - 1, 3)
                b[z + o[0]:z + o[0] + e, y + o[1]:y + o[1] + e, x + o[2]:x + o[2] + e] = True
    return b


syn = crowded((160, 200, 200), 400)
syn_bin = walls(syn)
syn_cubes = cubes((160, 200, 200))
datasets_lab = {"synthetic 6.4e6, %d labels" % (np.unique(syn).size - 1): syn,
                "dog_seed": dog_seed,
                "synthetic uint32 ids > 2**24": (syn.astype(np.uint32) * 5000),
                "synthetic int32": syn.astype(np.int32)}
datasets_bin = {"synthetic walls": syn_bin, "synthetic cubes": syn_cubes, "dog > 140": dog > 140, "dog > 100": dog > 100}

ANT = None
if not ARGS.no_ant:
    cand_scan = os.environ.get("SPROUT_ANT_SCAN")
    cand_lab = os.environ.get("SPROUT_ANT_LABEL")
    if not cand_scan:
        hits = sorted(Path(tempfile.gettempdir(), "claude").glob("*/*/scratchpad/ant_bench/data/scan/06-45.tif"))
        if hits:
            cand_scan = str(hits[0])
            cand_lab = str(hits[0].parents[1] / "label" / "06-45.tif")
    if cand_scan and os.path.isfile(cand_scan):
        sl = (slice(150, 350), slice(200, 450), slice(150, 400))
        a_scan = tifffile.imread(cand_scan)[sl]
        a_lab = tifffile.imread(cand_lab)[sl] if cand_lab and os.path.isfile(cand_lab) else None
        ANT = (a_scan, a_lab)
        thr = int(np.percentile(a_scan, 70))
        datasets_bin[f"ant crop {a_scan.shape} > {thr}"] = a_scan > thr
        if a_lab is not None and a_lab.any():
            datasets_lab[f"ant crop label {a_lab.shape}"] = a_lab
        print(f"ant crop: {cand_scan} {sl}", flush=True)
    else:
        print("ant crop: not found, skipped")

# ======================================================================
print("\n== B1 reorder_segmentation")
edge = {"all zero": np.zeros((5, 6, 7), np.uint16), "one id uint8": (np.arange(60).reshape(3, 4, 5) % 7 == 0)
        .astype(np.uint8) * 9, "2d uint16": syn[80], "bool": syn[80] > 0, "float": syn[80].astype(np.float32),
        "int16 negative": (syn[80].astype(np.int16) % 50) - 3, "empty": np.zeros((0, 4), np.uint8)}
bad = []
for name, seg in {**datasets_lab, **edge}.items():
    for kw in ({}, {"sort_ids": False}, {"min_size": 50}, {"top_n": 100}, {"sort_ids": False, "min_size": 50},
               {"min_size": 50, "top_n": 30}):
        a, b = outcome(sc.reorder_segmentation, seg, **kw), outcome(base_sc.reorder_segmentation, seg, **kw)
        if not same(a, b):
            bad.append(f"{name} {kw}")
ok(not bad, f"identical array / dtype / mapping (key and value types) on {len(datasets_lab) + len(edge)} maps "
            f"x 6 options {bad}")
t_new = best(lambda: sc.reorder_segmentation(syn))
t_old = best(lambda: base_sc.reorder_segmentation(syn), 1)
ok(t_old / t_new > 5, f"reorder_segmentation, {syn.size:.1e} voxels, 400 labels: {t_old:.2f} s -> {t_new:.3f} s "
                      f"({t_old / t_new:.0f}x)")

# ======================================================================
print("\n== B3 get_ccomps_with_size_order")
edge = {"empty": np.zeros((6, 6, 6), bool), "all foreground (base raises)": np.ones((4, 4, 4), bool),
        "2d": syn_bin[80], "single voxel": np.pad(np.ones((1, 1, 1), bool), 2)}
bad, n_cases = [], 0
for name, vol in {**datasets_bin, **edge}.items():
    n_comp = sc.measure.label(vol, background=0, return_num=True)[1] if vol.size else 0
    if n_comp <= 3000:
        opts = ({}, {"segments": 10}, {"segments": 100000}, {"min_vol": 20}, {"segments": 5, "min_vol": 20})
    else:       # the base scans the whole volume once per kept component: keep at most 200
        opts = ({"segments": 10}, {"segments": 200}, {"segments": 50, "min_vol": 20})
    print(f"     {name}: {n_comp} components, {len(opts)} options")
    for kw in opts:
        n_cases += 1
        a, b = outcome(sc.get_ccomps_with_size_order, vol, **kw), outcome(base_sc.get_ccomps_with_size_order,
                                                                           vol, **kw)
        if not same(a, b) and not ("base raises" in name and b[0] == "err"):
            bad.append(f"{name} {kw}")
ok(not bad, f"identical labels / dtype / sizes on {len(datasets_bin) + len(edge)} volumes, {n_cases} cases {bad}")
a = outcome(sc.get_ccomps_with_size_order, np.ones((4, 4, 4), bool))
print(f"     all-foreground volume: base raises IndexError, now {a[0]} "
      f"{a[1][1].tolist() if a[0] == 'ok' else a[1:]}")
t_new = best(lambda: sc.get_ccomps_with_size_order(syn_bin, 50))
t_old = best(lambda: base_sc.get_ccomps_with_size_order(syn_bin, 50), 1)
ok(t_old / t_new > 1.5, f"get_ccomps_with_size_order, {syn_bin.size:.1e} voxels, "
                        f"{sc.measure.label(syn_bin, return_num=True)[1]} components, 50 kept: {t_old:.2f} s -> {t_new:.2f} s "
                        f"({t_old / t_new:.1f}x)")

# ======================================================================
print("\n== B4 bounding box / emptiness")
m3 = dog > 140
edge3 = {"dog>140": m3, "dog>140 uint8": m3.astype(np.uint8), "values {0,2} (== True is empty)": m3 * np.uint8(2),
         "empty": np.zeros((5, 6, 7), bool), "corner voxel": np.pad(np.ones((1, 1, 1), bool), ((0, 4), (4, 0), (0, 0))),
         "full": np.ones((4, 5, 6), bool), "synthetic": syn_bin}
edge2 = {"2d": m3[100], "2d empty": np.zeros((5, 6), bool), "2d uint8": m3[100].astype(np.uint8)}
bad = []
for name, m in {**edge3, **edge2}.items():
    for margin in (0, 1, 3):
        a = outcome(sc.get_sub_binary_image_by_pos, m, margin)
        b = outcome(base_sc.get_sub_binary_image_by_pos, m, margin)
        if a[0] == b[0] == "ok":
            good = same(a[1][0], b[1][0]) and [type(v) for v in a[1][1]] == [type(v) for v in b[1][1]] \
                and a[1][1] == b[1][1]
        else:
            good = a[0] == b[0] == "err" and a[1] == b[1]
        if not good:
            bad.append(f"get_sub {name} margin {margin}: {a[0]} vs {b[0]}")
    if m.dtype != bool and m.max() > 1:
        continue
    for fn, kws in ((lambda M, f, **k: f.erosion_binary_img_on_sub(M, **k),
                     [{}, {"footprint": "cube"}, {"footprint": "ball_XY"}, {"kernal_size": 2}]),
                    (lambda M, f, **k: f.dilation_binary_img_on_sub(M, **k),
                     [{"margin": 2, "kernal_size": 1}, {"margin": 1, "kernal_size": 2, "is_round": False}]),
                    (lambda M, f, **k: f.closing_binary_img_on_sub(M, **k),
                     [{"margin": 2, "kernal_size": 1}, {"margin": 3, "kernal_size": 1, "is_round": False}])):
        for kw in kws:
            if m.ndim == 2 and kw.get("footprint") == "ball_XY":
                continue
            a = outcome(fn, m.copy(), sc, **kw)
            b = outcome(fn, m.copy(), base_sc, **kw)
            if not (same(a, b) or (a[0] == b[0] == "err" and a[1:] == b[1:])):
                bad.append(f"{name} {kw}: {a[0]} vs {b[0]}")
ok(not bad, f"get_sub_binary_image_by_pos (box, ids, types, errors) and erosion / dilation / closing on sub: "
            f"identical on {len(edge3) + len(edge2)} masks {bad}")
big = np.zeros((300, 300, 300), bool)
big[20:280, 30:270, 10:290] = ndi.gaussian_filter(rng.random((260, 240, 280), np.float32), 2) > 0.45
t_new = best(lambda: sc.get_sub_binary_image_by_pos(big, 1))
t_old = best(lambda: base_sc.get_sub_binary_image_by_pos(big, 1), 1)
ok(t_old / t_new > 3, f"bounding box of a {big.size:.1e}-voxel mask ({big.mean():.0%} set): {t_old:.2f} s -> "
                      f"{t_new:.3f} s ({t_old / t_new:.0f}x)")
t_new = best(lambda: sc.erosion_binary_img_on_sub(big.copy()))
t_old = best(lambda: base_sc.erosion_binary_img_on_sub(big.copy()), 1)
ok(t_old / t_new > 1.3, f"one erosion step on it: {t_old:.2f} s -> {t_new:.2f} s ({t_old / t_new:.1f}x)")

# ======================================================================
print("\n== B5 check_and_cast_boundary")
b01 = (dog > 120).astype(np.uint8)
cases = {"bool": dog > 120, "uint8 0/1": b01, "uint8 0/255": b01 * np.uint8(255), "uint16 0/3": b01.astype(np.uint16) * 3,
         "int16 -1/0": -b01.astype(np.int16), "int16 0/-5/7": b01.astype(np.int16) * -5 + (dog > 200) * 12,
         "three values": b01 + (dog > 200), "1/2 (no zero)": b01 + 1, "all zero": np.zeros((4, 4), np.uint8),
         "all one": np.ones((4, 4), np.uint8), "float 0/1": b01.astype(np.float32), "float 0/nan":
         np.where(b01, np.nan, 0).astype(np.float32), "float 0/1/nan": np.r_[0., 1., np.nan], "empty": np.zeros(0, np.uint8),
         "uint64 0/max": b01.astype(np.uint64) * np.uint64(2 ** 64 - 1), "list": [0, 1, 1], "None": None,
         "int8 -128/0": b01.astype(np.int8) * -128}
bad = []
for name, b in cases.items():
    x, y = outcome(cc.check_and_cast_boundary, b), outcome(base_cc.check_and_cast_boundary, b)
    if not (same(x, y) or (x[0] == y[0] == "err" and x[1:] == y[1:]) or (x == y == ("ok", None))):
        bad.append(f"{name}: {x[0]} {x[1:] if x[0] == 'err' else ''} vs {y[0]} {y[1:] if y[0] == 'err' else ''}")
ok(not bad, f"same result / same error text on {len(cases)} boundaries {bad}")
bb = np.zeros((300, 300, 300), np.uint8)
bb[::3] = 255
t_new = best(lambda: cc.check_and_cast_boundary(bb))
t_old = best(lambda: base_cc.check_and_cast_boundary(bb), 1)
ok(t_old / t_new > 3, f"check_and_cast_boundary, {bb.size:.1e} uint8 voxels: {t_old:.2f} s -> {t_new:.3f} s "
                      f"({t_old / t_new:.0f}x)")

# ======================================================================
print("\n== B8 Grow: non-zero counts per iteration, identical outputs")


class CountingNumpy:
    """numpy, but counts sum / count_nonzero calls over arrays of the image size."""

    def __init__(self, size):
        self.size, self.n = size, 0

    def __getattr__(self, k):
        return getattr(np, k)

    def sum(self, a, *args, **kw):
        self.n += np.size(a) == self.size
        return np.sum(a, *args, **kw)

    def count_nonzero(self, a, *args, **kw):
        self.n += np.size(a) == self.size
        return np.count_nonzero(a, *args, **kw)


def grow(mod, img, seg, out, **kw):
    counter = CountingNumpy(img.size)
    saved = mod.np
    mod.np = counter
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            grows, log = mod.grow_mp(img=img, seg=seg, output_folder=str(out), base_name="g", num_threads=4,
                                     return_for_napari=True, **kw)
    finally:
        mod.np = saved
    files = {p.name: tifffile.imread(p) for p in sorted(Path(out).rglob("*.tif"))}
    csv = Path(log["log_path"]).read_text().replace(os.path.abspath(out), "<OUT>")   # full paths in the log
    return grows, files, csv, counter.n


d_img, d_seed = dog[200:420], dog_seed[200:420]
runs = [("dog", d_img, d_seed, dict(thresholds=[150, 120], upper_thresholds=[255, 255], dilation_steps=[3, 3],
                                    save_every_n_iters=1)),
        ("dog early stop", d_img, d_seed, dict(thresholds=[130], dilation_steps=[12], min_growth_size=500,
                                               no_growth_max_iter=2, grow_to_end=True)),
        ("synthetic", (syn > 0).astype(np.uint8) * 200, (syn * (ndi.binary_erosion(syn > 0, iterations=3))),
         dict(thresholds=[100], dilation_steps=[4], is_sort=False))]
if ANT is not None and ANT[1] is not None and ANT[1].any():
    runs.append(("ant crop", ANT[0], ANT[1], dict(thresholds=[int(np.percentile(ANT[0], 60))], dilation_steps=[3])))
for name, img, seg, kw in runs:
    gn, fn_, cn, nn = grow(make_grow, img, seg, TMP / "grow_new" / name, **kw)
    gb, fb, cb, nb = grow(base_grow, img, seg, TMP / "grow_base" / name, **kw)
    iters = sum(kw["dilation_steps"])
    ok(same(gn, gb) and same(fn_, fb) and cn == cb,
       f"{name}: {len(gn)} returned layers, {len(fn_)} tif files and the CSV log identical to base")
    ok(nn < nb, f"{name}: full-size non-zero counts {nb} -> {nn} (up to {iters} iterations)")

print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
sys.exit(1 if FAILED else 0)
