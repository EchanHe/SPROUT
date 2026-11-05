"""
Checks of the pure Prompt (AI) core: sprout_core/prompt_points.py, prompt_engine.py,
nni_engine.py, prompt_seg.py and the fixes in nninteractive_predict.py.
No nnInteractive needed (a fake session stands in for the model).

Run in the `sprout` env:
    python tests/check_prompt_core.py
Prints PASS/FAIL per check; exits non-zero if anything failed.

ROI: uses sprout_core.roi.Roi when it exists, else the stub in _prompt_check_utils.py.
"""
import ast
import contextlib
import io
import os
import subprocess
import sys
import textwrap
import threading
import types
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "napari_sprout"), str(Path(__file__).resolve().parent)]
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tifffile  # noqa: E402

FAILED = []


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILED.append(msg)


def NOLOG(msg):
    pass


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# ---------------------------------------------------------------- 1. lazy imports
print("== 1. lazy imports")
probe = textwrap.dedent(r"""
    import os, sys, json
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    sys.path[:0] = [%r, %r, %r]
    import sprout_core.prompt_points, sprout_core.prompt_seg, sprout_core.prompt_engine, sprout_core.nni_engine
    after_core = [m for m in ("torch", "nnInteractive", "segment_anything", "sklearn") if m in sys.modules]
    import napari
    import _headless_gl  # napari >= 0.7 offscreen: no GL context
    from napari_sprout._widget import SPROUTWidget
    v = napari.Viewer(show=False)
    w = SPROUTWidget(v)
    from sprout_core.nni_engine import nninteractive_status, INSTALL_HINT
    st = nninteractive_status()
    bad = [m for m in ("torch", "nnInteractive", "segment_anything", "sklearn") if m in sys.modules]
    print("RESULT" + json.dumps({"after_core": after_core, "bad": bad, "status": list(st),
                                 "hint_ok": st == (False, INSTALL_HINT)}))
    v.close()
""" % (str(ROOT), str(ROOT / "napari_sprout"), str(Path(__file__).resolve().parent)))
res = subprocess.run([sys.executable, "-B", "-c", probe], capture_output=True, text=True, timeout=600)
line = [ln for ln in res.stdout.splitlines() if ln.startswith("RESULT")]
if not line:
    print(res.stdout[-2000:], res.stderr[-3000:])
    ok(False, "lazy-import probe ran")
else:
    import json
    r = json.loads(line[0][6:])
    ok(r["after_core"] == [], f"importing the 4 core modules loads none of torch/nnInteractive/SAM/sklearn ({r['after_core']})")
    ok(r["bad"] == [], f"import napari_sprout._widget + SPROUTWidget(Viewer(show=False)) + status: none loaded ({r['bad']})")
    ok(r["hint_ok"], f"nninteractive_status() == (False, INSTALL_HINT) without nnInteractive: {r['status'][0]}")

from sprout_core import prompt_points as PP  # noqa: E402
from sprout_core import prompt_seg as PS  # noqa: E402
from sprout_core import prompt_engine as PE  # noqa: E402
from sprout_core import nni_engine as NE  # noqa: E402
from sprout_core.boundary import Cancelled  # noqa: E402
from _prompt_check_utils import (Roi, ROI_IS_STUB, WALL_MARGIN, FakeSession11, FakeSession26,  # noqa: E402
                                 normalize_calls, BALL_R, _ball_into)

print(f"     ROI: {'STUB (sprout_core.roi not importable yet)' if ROI_IS_STUB else 'real sprout_core.roi.Roi'}")

# ---------------------------------------------------------------- 2. the move is byte-identical
print("== 2. moved sampling functions")
# the version before the move: HEAD if not committed yet, else the parent of the commit that moved them
_rev = "HEAD"
if "def sample_points_3d" not in subprocess.run(["git", "show", "HEAD:sprout_core/sprout_prompt_core.py"],
                                                capture_output=True, text=True, encoding="utf-8", cwd=ROOT).stdout:
    _mv = subprocess.run(["git", "log", "-1", "--format=%H", "-S", "def sample_points_3d", "--",
                          "sprout_core/sprout_prompt_core.py"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    _rev = f"{_mv}^"
print(f"     reference: {_rev}:sprout_core/sprout_prompt_core.py")
old_src = subprocess.run(["git", "show", f"{_rev}:sprout_core/sprout_prompt_core.py"], capture_output=True,
                         text=True, encoding="utf-8", cwd=ROOT).stdout
old_tree = ast.parse(old_src)
wanted = [n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_grid_sampling", "sample_points_3d")]
ok(len(wanted) == 2, f"found _grid_sampling and sample_points_3d in {_rev}:sprout_core/sprout_prompt_core.py")
from scipy.ndimage import distance_transform_edt  # noqa: E402
old_ns = {"np": np, "KMeans": None, "distance_transform_edt": distance_transform_edt}
exec(compile(ast.Module(body=wanted, type_ignores=[]), "old_sampling", "exec"), old_ns)

rs = np.random.RandomState(123)
n_cases = n_bad = 0
for case in range(12):
    shape = tuple(int(v) for v in rs.randint(6, 25, size=3))
    mask = (rs.rand(*shape) < rs.choice([0.002, 0.05, 0.3, 0.9])).astype(np.uint8)
    for method in ("random", "grid", "center_edge"):
        for n in (1, 3, 5, 8, 27):
            for s in range(10):
                np.random.seed(s)
                a = old_ns["sample_points_3d"](mask, n=n, method=method)
                after_a = np.random.randint(1 << 30)
                np.random.seed(s)
                b = PP.sample_points_3d(mask, n=n, method=method)
                after_b = np.random.randint(1 << 30)
                c = PP.sample_points_3d(mask, n=n, method=method, rng=np.random.RandomState(s))
                n_cases += 1
                same = (np.array_equal(a, b) and a.dtype == b.dtype and a.shape == b.shape and after_a == after_b
                        and np.array_equal(a, c) and a.dtype == c.dtype)
                n_bad += not same
ok(n_bad == 0, f"sample_points_3d new == HEAD version (values, dtype, global stream after) and "
               f"rng=RandomState(s) == np.random.seed(s): {n_cases - n_bad}/{n_cases}")
coords = np.argwhere(np.ones((5, 6, 7)))
bad = 0
for s in range(10):
    np.random.seed(s)
    a = old_ns["_grid_sampling"](coords, 9)
    np.random.seed(s)
    bad += not np.array_equal(a, PP._grid_sampling(coords, 9))
    bad += not np.array_equal(a, PP._grid_sampling(coords, 9, rng=np.random.RandomState(s)))
ok(bad == 0, "_grid_sampling new == HEAD version (global and RandomState)")
m = np.ones((5, 5, 5), np.uint8)
msgs = []
for fn in (old_ns["sample_points_3d"], PP.sample_points_3d):
    try:
        fn(m, n=3, method="kmeans")
        msgs.append("no error")
    except Exception as e:
        msgs.append(f"{type(e).__name__}: {e}")
try:
    import sklearn.cluster  # noqa: F401
    have_sklearn = True
except Exception:
    have_sklearn = False
if not have_sklearn:
    ok(msgs[0] == msgs[1] == "ImportError: sklearn required. Install: pip install scikit-learn",
       f"kmeans without a working sklearn: same ImportError ({msgs[1]})")
else:
    print("     (sklearn works here: kmeans error message not compared)")

# rest of sprout_prompt_core.py unchanged + re-export line
new_src = (ROOT / "sprout_core" / "sprout_prompt_core.py").read_text(encoding="utf-8")
new_tree = ast.parse(new_src)
old_rest = [ast.dump(n) for n in old_tree.body if not (isinstance(n, ast.FunctionDef) and n.name in ("_grid_sampling", "sample_points_3d"))]
reexp = [n for n in new_tree.body if isinstance(n, ast.ImportFrom) and n.module == "sprout_core.prompt_points"]
new_rest = [ast.dump(n) for n in new_tree.body if n not in reexp]
ok(old_rest == new_rest and len(reexp) == 1 and sorted(a.name for a in reexp[0].names) == ["_grid_sampling", "sample_points_3d"],
   "sprout_prompt_core.py: only the two functions replaced by `from sprout_core.prompt_points import ...` (AST)")

# ---------------------------------------------------------------- 3. seeds_to_point_prompts == file version
print("== 3. seeds_to_point_prompts vs seed_to_point_prompts_nninteractive (dog)")
quiet(lambda: __import__("sprout_core.sprout_prompt_core"))
from sprout_core.sprout_prompt_core import seed_to_point_prompts_nninteractive  # noqa: E402
import sprout_core.sprout_prompt_core as SPC  # noqa: E402
ok(SPC.sample_points_3d is PP.sample_points_3d and SPC._grid_sampling is PP._grid_sampling,
   "sprout_prompt_core re-exports the moved functions")
SEED, IMG = "demo_data/dog_seed.tif", "demo_data/dog_img.tif"
seg = tifffile.imread(SEED)
img = tifffile.imread(IMG)
configs = [
    dict(default_n_pos=5, default_n_neg=0, default_method="random"),
    dict(default_n_pos=3, default_n_neg=3, default_method="random", negative_from_bg=True,
         negative_from_other_classes=True, negative_per_other_class=False),
    dict(default_n_pos=4, default_n_neg=2, default_method="grid", negative_from_bg=True,
         negative_from_other_classes=True, negative_per_other_class=True),
    dict(default_n_pos=3, default_n_neg=4, default_method="center_edge", negative_from_bg=False,
         negative_from_other_classes=True, negative_per_other_class=False,
         class_config={4: {"n_pos": 7, "n_neg": 1, "method": "random"}, 6: {"n_neg": 0}}),
]
for i, cfg in enumerate(configs):
    np.random.seed(7 + i)
    ref = quiet(seed_to_point_prompts_nninteractive, SEED, img_path=IMG, **cfg)
    got = PP.seeds_to_point_prompts(seg, img, image_name="dog_seed", rng=np.random.RandomState(7 + i), **cfg)
    np.random.seed(7 + i)
    got2 = PP.seeds_to_point_prompts(seg, img, image_name="dog_seed", **cfg)
    ok(ref.equals(got) and ref.equals(got2) and list(got.columns) == PP.POINT_COLUMNS,
       f"config {i} ({cfg.get('default_method')}, n_neg {cfg.get('default_n_neg')}): {len(got)} rows identical "
       f"(DataFrame.equals; RandomState and global stream)")
got = PP.seeds_to_point_prompts(seg, img, classes=[4, 6], rng=np.random.RandomState(0), default_n_pos=2,
                                default_n_neg=2, negative_per_other_class=True)
ok(sorted(got.class_id.unique()) == [4, 6] and
   set(got.point_type) >= {"positive", "negative_class3", "negative_class5"},
   "classes=[4, 6]: only those get points, negatives still from all other classes")
st = PP.class_stats(seg)
ref_ids, ref_n = np.unique(seg[seg > 0], return_counts=True)
ok(st.class_id.tolist() == ref_ids.tolist() and st.voxels.tolist() == ref_n.tolist(), "class_stats == np.unique")
big = np.zeros((4, 5, 6), np.uint32)
big[0, 0, :3] = 3_000_000
big[1, 1, 1] = 5
ok(PP.class_stats(big).values.tolist() == [[5, 1], [3_000_000, 3]], "class_stats with ids >= 2**20 (np.unique path)")
ev = threading.Event()
ev.set()
try:
    PP.seeds_to_point_prompts(seg, img, cancel=ev)
    ok(False, "cancel raises Cancelled")
except Cancelled:
    ok(True, "cancel raises Cancelled")
calls = []
PP.seeds_to_point_prompts(seg, img, progress=lambda i, n: calls.append((i, n)), rng=np.random.RandomState(0))
ok(calls == [(1, 4), (2, 4), (3, 4), (4, 4)], "progress(i, n) after each class")
e = PP.seeds_to_point_prompts(np.zeros((3, 4, 5), np.uint8), np.ones((3, 4, 5)))
ok(len(e) == 0 and list(e.columns) == PP.POINT_COLUMNS, "no classes -> empty table with the standard columns")
s2 = np.zeros((20, 30), np.uint8)
s2[5:10, 5:10] = 2
d2 = PP.seeds_to_point_prompts(s2, np.ones((20, 30)), rng=np.random.RandomState(0), default_n_pos=3)
ok(len(d2) == 3 and (d2.z == 0).all() and (s2[d2.y, d2.x] == 2).all(), "2D seeds: z = 0, points on the class")

# ---------------------------------------------------------------- 4. ROI sampling
print("== 4. ROI sampling")
roi = Roi.box(seg.shape, (300, 40, 20), (480, 140, 110))
full_allowed = roi.mask_full()
cfg = dict(default_n_pos=5, default_n_neg=3, negative_per_other_class=True)
dfr = PP.seeds_to_point_prompts(seg, img, allowed=full_allowed, rng=np.random.RandomState(1), log=NOLOG, **cfg)
inside = roi.contains(dfr[["z", "y", "x"]].to_numpy())
ok(len(dfr) > 0 and inside.all(), f"allowed=ROI: all {len(dfr)} points inside the ROI")
in_roi = PP.class_stats(seg, allowed=full_allowed)
ok(set(dfr.class_id) == set(in_roi.class_id) and set(in_roi.class_id) != set(np.unique(seg[seg > 0])),
   f"only classes with voxels inside get points ({sorted(set(dfr.class_id))} of {sorted(np.unique(seg[seg > 0]))})")
a = PP.seeds_to_point_prompts(seg, img, rng=np.random.RandomState(2), **cfg)
b = PP.seeds_to_point_prompts(seg, img, allowed=np.ones(seg.shape, bool), rng=np.random.RandomState(2), **cfg)
ok(a.equals(b), "allowed = all True == no ROI")
n_same = n_all = 0
segs, imgs = seg[200:520], img[200:520]          # a part of the dog keeps this loop short
mroi = Roi.from_mask((segs == 4) | (segs == 5) | ((imgs > 120) & (np.arange(segs.shape[0])[:, None, None] < 100)), name="m")
for r_ in (Roi.box(segs.shape, (100, 40, 20), (280, 140, 110)), Roi.box(segs.shape, (0, 0, 0), (150, 161, 122)), mroi):
    for method in ("random", "grid", "center_edge"):
        for neg in (dict(default_n_neg=0), dict(default_n_neg=3), dict(default_n_neg=2, negative_per_other_class=True)):
            kw = dict(default_n_pos=4, default_method=method, **neg)
            full = PP.seeds_to_point_prompts(segs, imgs, allowed=r_.mask_full(), rng=np.random.RandomState(3), log=NOLOG, **kw)
            crop = PP.seeds_to_point_prompts_in_roi(segs, imgs, r_, rng=np.random.RandomState(3), log=NOLOG, **kw)
            n_all += 1
            n_same += full.equals(crop)
ok(n_same == n_all, f"crop sampling (margin {WALL_MARGIN}) == whole-image sampling with allowed=mask_full(): "
                    f"{n_same}/{n_all} (box, border box, mask ROI x random/grid/center_edge x 3 negative configs)")

# ---------------------------------------------------------------- 5. points <-> df
print("== 5. points <-> prompt table")
df0 = PP.seeds_to_point_prompts(seg, img, rng=np.random.RandomState(4), default_n_pos=3, default_n_neg=2,
                                image_name="dog")
data, feats = PP.df_to_points(df0)
back = PP.points_to_df(data, feats, image_name="dog")
ok(back.equals(df0), "df -> (data, features) -> df is identical (order, values, dtypes)")
ok(data.dtype == float and data.shape == (len(df0), 3) and np.array_equal(data[:, 0], df0.z),
   "data float (N, 3) in z, y, x")
ok(feats.dtypes.astype(str).tolist() == ["int64", "bool", "object", "object"], f"feature dtypes {feats.dtypes.astype(str).tolist()}")
noisy = data + np.random.RandomState(0).uniform(-0.49, 0.49, data.shape)
ok(PP.points_to_df(noisy, feats, "dog").equals(df0), "coordinates rounded with np.rint")
rev = PP.points_to_df(data[::-1], feats.iloc[::-1], "dog")
ok(rev.reset_index(drop=True).equals(df0.iloc[::-1].reset_index(drop=True)), "row order preserved")
d2, f2 = PP.df_to_points(df0, ndim=2)
ok(d2.shape == (len(df0), 2) and np.array_equal(d2[:, 0], df0.y), "ndim=2 -> (y, x)")
b2 = PP.points_to_df(d2, f2)
ok((b2.z == 0).all() and np.array_equal(b2.y, df0.y) and np.array_equal(b2.x, df0.x), "2D data -> z = 0")
dd = PP.points_to_df(np.array([[1.2, 2.6, 3.5], [4, 5, 6]]), pd.DataFrame({"class_id": [7, 7], "label": [True, False]}))
ok(dd.point_type.tolist() == ["positive", "negative"] and dd.sample_method.tolist() == ["manual", "manual"] and
   dd[["z", "y", "x"]].values.tolist() == [[1, 3, 4], [4, 5, 6]], "feature defaults (point_type from label, 'manual'); rint half-to-even")
dd = PP.points_to_df(np.array([[1.0, 2, 3]]), None)
ok(dd.class_id.tolist() == [1] and dd.label.tolist() == [True], "no features -> class 1, positive")
e = PP.points_to_df(np.zeros((0, 3)), pd.DataFrame(columns=["class_id", "label"]))
ed, ef = PP.df_to_points(e)
ok(len(e) == 0 and ed.shape == (0, 3) and len(ef) == 0, "empty round trip")
vox = PP.points_data_to_voxels(np.array([[2.0, 4, 6]]), (1, 1, 1), (0, 0, 0), (2, 0.5, 0.5), (10, 0, 0))
ok(np.allclose(vox, [[-4, 8, 12]]), "points_data_to_voxels: scale/translate of points and image")
vox = PP.points_data_to_voxels(np.array([[3.0, 2, 2]]), (2, 0.5, 0.5), (10, 0, 0), (2, 0.5, 0.5), (10, 0, 0))
ok(np.allclose(vox, [[3, 2, 2]]), "points layer with the image's scale/translate: data == voxels")
import warnings  # noqa: E402
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    PP.points_data_to_voxels(np.zeros((1, 3)), (1,) * 3, (0,) * 3, (1,) * 3, (0,) * 3, rotate=np.eye(3))
    n0 = len(w)
    PP.points_data_to_voxels(np.zeros((1, 3)), (1,) * 3, (0,) * 3, (1,) * 3, (0,) * 3,
                             rotate=[[0, 1, 0], [1, 0, 0], [0, 0, 1]])
ok(n0 == 0 and len(w) == 1, "rotation ignored with a warning (none for identity)")
sp, sn = PP.seed_scribble(seg, 4, negative=True)
ok(sp.dtype == np.uint8 and np.array_equal(sp, (seg == 4).astype(np.uint8)) and
   np.array_equal(sn, ((seg != 4) & (seg != 0)).astype(np.uint8)), "seed_scribble == nninter_predict's scribble masks")
ok(PP.seed_scribble(seg, 4)[1] is None, "seed_scribble without negative -> None")
lab = np.zeros((3, 4, 5), np.uint16)
lab[1, 1, 1:3] = 9
ok(np.array_equal(PP.scribbles_from_layer(lab, 9), lab == 9), "scribbles_from_layer")

# ---------------------------------------------------------------- 6. filter_kwargs / make_session
print("== 6. session kwargs for nnInteractive 1.1.3 / 2.6")
want = dict(device="cpu", use_torch_compile=False, verbose=False, do_autozoom=True, use_pinned_memory=False,
            interactions_storage="auto", enable_undo=False)
logs = []
k11 = NE.filter_kwargs(FakeSession11.__init__, want, log=logs.append)
k26 = NE.filter_kwargs(FakeSession26.__init__, want, log=logs.append)
NE.filter_kwargs(FakeSession26.__init__, want, log=logs.append)
ok("use_pinned_memory" in k11 and "interactions_storage" not in k11 and "enable_undo" not in k11, "1.1.3: keeps use_pinned_memory, drops 2.6 args")
ok("use_pinned_memory" not in k26 and k26["enable_undo"] is False and k26["interactions_storage"] == "auto", "2.6: drops use_pinned_memory")
ok(len(logs) == 2, f"dropped names logged once per (function, names): {logs}")


def kwonly(**kw):
    pass


ok(NE.filter_kwargs(kwonly, want, log=None) == want, "**kwargs function gets everything")
SRC = {"1.1.3": Path(r"C:/Users/yichen/work/nnInteractive/nnInteractive/inference/inference_session.py"),
       "2.6.0": Path(r"C:/Users/yichen/miniforge3/envs/sprout-nni/Lib/site-packages/nnInteractive/inference/inference_session.py")}
for ver, fake in (("1.1.3", FakeSession11), ("2.6.0", FakeSession26)):
    p = SRC[ver]
    if not p.exists():
        print(f"     ({ver} source not found: signature not compared)")
        continue
    t = ast.parse(p.read_text(encoding="utf-8"))
    cls = [n for n in t.body if isinstance(n, ast.ClassDef) and n.name == "nnInteractiveInferenceSession"][0]
    init = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"][0]
    real = [a.arg for a in init.args.args]
    import inspect  # noqa: E402
    ok(real == list(inspect.signature(fake.__init__).parameters), f"fake {ver} __init__ signature == installed source: {real[1:]}")
try:
    import torch  # noqa: E402
except ImportError:                     # e.g. the napari 0.7 test env: the model parts are skipped
    print("SKIP 6 (cuDNN flags) and 7-10 (engine, nninter_predict): torch is not installed")
    print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'} (torch parts skipped)")
    sys.exit(1 if FAILED else 0)
old_flags = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = True, False
s = NE.make_session("cpu", session_cls=FakeSession26)
ok(s.kwargs["enable_undo"] is False and s.kwargs["interactions_storage"] == "auto" and str(s.kwargs["device"]) == "cpu"
   and torch.backends.cudnn.benchmark is False and torch.backends.cudnn.deterministic is True,
   "make_session (2.6 fake): enable_undo False, cpu, deterministic cuDNN flags set after construction")
s = NE.make_session("cpu", session_cls=FakeSession11, use_pinned_memory=True)
ok(s.kwargs["use_pinned_memory"] is True and s.kwargs["use_torch_compile"] is False, "make_session (1.1.3 fake): use_pinned_memory passed")
torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = True, False
NE.make_session("cpu", session_cls=FakeSession26, deterministic=False)
ok(torch.backends.cudnn.benchmark is True and torch.backends.cudnn.deterministic is False, "deterministic=False leaves the flags alone")
torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = old_flags

# ---------------------------------------------------------------- 7. engine with a fake session
print("== 7. engine + run_classes with a fake session")
# nninteractive_predict imports nnInteractive at module level: give it the fake.
for name in ("nnInteractive", "nnInteractive.inference"):
    sys.modules.setdefault(name, types.ModuleType(name))
fake_mod = types.ModuleType("nnInteractive.inference.inference_session")
fake_mod.nnInteractiveInferenceSession = FakeSession26
sys.modules["nnInteractive.inference.inference_session"] = fake_mod
NP = quiet(__import__, "sprout_core.nninteractive_predict", fromlist=["x"])


def new_engine(cls=FakeSession26, buf=np.zeros):
    eng = NE.NniEngine(session_factory=lambda dev: cls(device=dev), buffer_factory=lambda shp: buf(shp, np.uint8))
    eng.load("fake_model", "cpu")
    return eng


# small synthetic volume: 3 classes
shp = (40, 50, 60)
rs = np.random.RandomState(5)
vol = rs.randint(1, 255, shp).astype(np.uint8)
seeds = np.zeros(shp, np.uint16)
seeds[5:15, 5:20, 5:20] = 2
seeds[20:30, 10:40, 30:50] = 7
seeds[8:12, 30:45, 35:55] = 3
dfp = PP.seeds_to_point_prompts(seeds, vol, rng=np.random.RandomState(0), default_n_pos=3, default_n_neg=2,
                                negative_per_other_class=True)
classes = [7, 3, 2]                     # nninter_predict order: descending

n_inst = len(FakeSession26.instances)
eng = new_engine()
ok(eng.loaded and eng.info["major"] == 2 and eng.info["device"] == "cpu" and eng.capabilities == {"points", "scribble", "init_mask"},
   f"NniEngine.load with a fake 2.6 session: info {eng.info}")
ok(eng.load("fake_model", "cpu") and len(FakeSession26.instances) == n_inst + 1, "load() cached by (model_dir, device)")
eng.load("fake_model2", "cpu")
ok(len(FakeSession26.instances) == n_inst + 2 and eng.session is FakeSession26.instances[-1], "different model_dir -> release + rebuild")
names = []
eng._run(lambda: names.append(threading.current_thread().name))
ok(names[0].startswith("sprout-nni"), f"session calls run on the engine thread ({names[0]})")


def run_all(engine, prompt_type="points", **kw):
    crop = kw.pop("crop", PS.inference_crop(shp, kw.get("roi")))
    out = {}
    for r in PS.run_classes(engine, vol, seeds, kw.pop("df", dfp), kw.pop("classes", classes), crop=crop,
                            image_key=kw.pop("image_key", "vol"), prompt_type=prompt_type, **kw):
        out[r[1] if isinstance(r, tuple) else r.class_id] = r
    return out


def script_run(**kw):
    sess = FakeSession26(device="cpu")
    per = {}
    tot = quiet(NP.nninter_predict, sess, vol[None], on_class_predicted=lambda c, m: per.__setitem__(int(c), m.copy()), **kw)
    return sess, tot, per


from scipy.ndimage import grey_dilation  # noqa: E402

init_lab = grey_dilation(seeds, size=(3, 5, 5))          # an "initial segmentation" (init_seg_path)
nodf = PP.empty_points_df()
for ptype, skw, ekw in (("points", dict(df_pt=dfp), {}),
                        ("points+init", dict(df_pt=dfp, init_mask=seeds), {}),
                        ("scribble", dict(df_pt=dfp, scribble_mask=seeds), {}),
                        ("scribble", dict(df_pt=dfp, scribble_mask=seeds, scribble_config={"use_negative_scribble": True}),
                         dict(neg_scribble=True)),
                        # the YAML paths of nninter_main: scribble mode has no points (df_pt=None)
                        ("scribble", dict(scribble_mask=seeds), dict(df=nodf)),
                        ("scribble", dict(scribble_mask=seeds, scribble_config={"use_negative_scribble": True}),
                         dict(df=nodf, neg_scribble=True)),
                        ("scribble", dict(scribble_mask=seeds, init_mask=init_lab), dict(df=nodf, init_masks=init_lab)),
                        ("scribble", dict(scribble_mask=seeds, init_mask=init_lab,
                                          scribble_config={"use_negative_scribble": True}),
                         dict(df=nodf, init_masks=init_lab, neg_scribble=True)),
                        ("points", dict(df_pt=dfp, init_mask=init_lab), dict(init_masks=init_lab))):
    eng = new_engine()
    res = run_all(eng, ptype, **ekw)
    sess, tot, per = script_run(**skw)
    # script: add_*(run_prediction=False) + _predict(); engine: add_*(run_prediction=True), which runs
    # _predict() inside the add call -> the same recorded order
    e_calls = normalize_calls(eng.session.calls)
    s_calls = normalize_calls(sess.calls)
    store = PS.ResultStore()
    for r in res.values():
        store.set(r)
    comp = store.compose(shp, rule="min_id")
    same_masks = all(np.array_equal(res[c].full_mask_in(PS.inference_crop(shp)), per[c]) for c in classes)
    ok(e_calls == s_calls and same_masks and np.array_equal(comp, tot),
       f"{ptype}{' + negative scribble' if ekw.get('neg_scribble') else ''}"
       f"{' + init mask (init_seg_path)' if 'init_masks' in ekw else ''}{' (no points, as the YAML)' if 'df' in ekw else ''}"
       f": per-class call sequence, class masks and "
       f"min_id composite == nninter_predict ({sum(len(c) for c in e_calls)} calls)")

# the pipeline's fixed seed: np.random.seed(seed + class id) after each class's reset, as the engine does
_seed_calls = []
_real_seed = np.random.seed


def _rec_seed(v=None):
    _seed_calls.append(v)
    return _real_seed(v)


np.random.seed = _rec_seed
try:
    _seed_calls.clear()
    script_run(scribble_mask=seeds, seed=5)
    s_seeds = list(_seed_calls)
    _seed_calls.clear()
    run_all(new_engine(), "scribble", df=nodf, classes=[7, 3, 2], base_seed=5)
    e_seeds = list(_seed_calls)
    _seed_calls.clear()
    script_run(scribble_mask=seeds, seed=None)
    n_seeds = list(_seed_calls)
finally:
    np.random.seed = _real_seed
ok(s_seeds == [12, 8, 7] and e_seeds == s_seeds and n_seeds == [],
   f"nninter_predict(seed=5) seeds np.random with seed + class id per class like run_classes ({s_seeds}); "
   f"seed=None does not seed")
import inspect as _inspect  # noqa: E402
from sprout_core import config_core as _cc  # noqa: E402

ok(_inspect.signature(NP.nninter_predict).parameters["seed"].default == 0
   and _inspect.signature(NP.nninter_main).parameters["seed"].default == 0
   and _inspect.signature(NP.nninter_main_from_prompts).parameters["seed"].default == 0
   and _cc.input_val_nninteractive_run["seed"]["default"] == 0 and _cc.input_val_nninteractive_run["seed"]["type"] is int,
   "seed (default 0) in nninter_predict / nninter_main / nninter_main_from_prompts and the YAML schema")
_src = (ROOT / "sprout_core" / "nninteractive_predict.py").read_text(encoding="utf-8")
_bsrc = (ROOT / "sprout_core" / "batch_nninteractive.py").read_text(encoding="utf-8")
ok("seed=optional_params['seed']" in _src and "seed=optional_params['seed']" in _bsrc,
   "the YAML seed reaches nninter_main (sprout.py --prompt --nninteractive, single and batch)")

eng = new_engine()
res = run_all(eng)
ok(all(isinstance(r, PS.ClassResult) and r.mode == "scratch" and r.n_points == (dfp.class_id == r.class_id).sum()
       for r in res.values()), "ClassResult per class, mode scratch, all points used")
ok(eng.active == (("vol", tuple((0, n) for n in shp)), 2), f"engine.active = (image key, last class): {eng.active}")
n_set = sum(c[0] == "set_image" for c in eng.session.calls)
run_all(eng)
ok(sum(c[0] == "set_image" for c in eng.session.calls) == n_set == 1, "same image key + crop: set_image not repeated")

# cancel between points
store = PS.ResultStore()
for r in res.values():
    store.set(r)
snap = {c: (store.get(c).slices, store.get(c).mask.copy()) for c in store.classes()}
cnt = {"n": 0}


def cancel_after(k):
    def f():
        cnt["n"] += 1
        return cnt["n"] > k
    return f


try:
    for r in PS.run_classes(eng, vol, seeds, dfp, [7], crop=PS.inference_crop(shp), image_key="vol",
                            cancel=cancel_after(3)):
        store.set(r)
    ok(False, "Stop between points raises Cancelled")
except Cancelled:
    ok(store.classes() == sorted(snap) and all(store.get(c).slices == snap[c][0] and
                                               np.array_equal(store.get(c).mask, snap[c][1]) for c in snap),
       "Stop between points raises Cancelled; the store is unchanged")
ok(eng.active is None, "after a Stop the engine state is not reused (active None)")

# ROI: dropped points, crop == full run inside ROI
roi = Roi.box(shp, (3, 3, 28), (33, 45, 58))
inside_df = dfp[roi.contains(dfp[["z", "y", "x"]].to_numpy())].reset_index(drop=True)
eng = new_engine()
full = run_all(eng, df=inside_df)
eng2 = new_engine()
rr = run_all(eng2, df=dfp, roi=roi, crop=PS.inference_crop(shp, roi, margin=6))
rm = roi.mask_full()
n_drop = {c: int(((dfp.class_id == c) & ~roi.contains(dfp[["z", "y", "x"]].to_numpy())).sum()) for c in classes}
ok(all(not isinstance(rr[c], tuple) and rr[c].n_dropped == n_drop[c] for c in (7, 3)) and sum(n_drop.values()) > 0,
   f"points outside the ROI dropped and counted: {n_drop}")
all_full = PS.inference_crop(shp)
eq = all(np.array_equal(rr[c].full_mask_in(all_full), full[c].full_mask_in(all_full) & rm) for c in (7, 3))
ok(eq, "ROI crop run pasted back == whole-image run inside the ROI, 0 outside (classes 7, 3)")
ok(isinstance(rr[2], PS.ClassResult) or rr[2][0] == "skipped", f"class 2 (mostly outside the ROI): {rr[2] if isinstance(rr[2], tuple) else 'result'}")
crop_small = PS.inference_crop(shp, roi, margin=6)
eng3 = new_engine()
d3 = pd.concat([dfp, pd.DataFrame([dict(image_name="", class_id=7, x=1, y=1, z=1, label=True, point_type="positive",
                                         sample_method="manual")])], ignore_index=True)
r3 = run_all(eng3, df=d3, crop=crop_small, classes=[7])
ok(r3[7].n_dropped == int(((d3.class_id == 7) & ~(d3.z.between(crop_small[0].start, crop_small[0].stop - 1) &
                                                    d3.y.between(crop_small[1].start, crop_small[1].stop - 1) &
                                                    d3.x.between(crop_small[2].start, crop_small[2].stop - 1))).sum()),
   f"points outside the crop (no ROI) dropped and counted ({r3[7].n_dropped})")

# 1.1.3: points outside the nonzero box are not valid
vz = vol.copy()
vz[:, :, :8] = 0
eng11 = NE.NniEngine(session_factory=lambda dev: FakeSession11(device=dev), buffer_factory=lambda s: np.zeros(s, np.uint8))
eng11.load("m", "cpu")
eng11.set_image(vz, "vz")
ok(eng11.info["major"] == 1 and not eng11.valid_point(5, 5, 7) and eng11.valid_point(5, 5, 8),
   "1.1.3 session: points outside the image's nonzero box are invalid")
ok(eng11.session.calls[-2][2] == "float32", "1.1.3 session gets a float32 image; 2.6 the raw dtype")
pts, nd = PS.class_points(pd.DataFrame(dict(class_id=[1, 1], z=[5, 5], y=[5, 5], x=[3, 20], label=[True, True])), 1,
                          PS.inference_crop(vz.shape), valid=eng11.valid_point)
ok(pts == [(5, 5, 20, True)] and nd == 1, "class_points drops the invalid point")

# update: warm / cold / nothing / scratch
eng = new_engine()
res = run_all(eng)
store = PS.ResultStore()
for r in res.values():
    store.set(r)
crop_all = PS.inference_crop(shp)
upd = pd.concat([dfp, pd.DataFrame([dict(image_name="", class_id=2, x=12, y=12, z=12, label=False,
                                          point_type="negative", sample_method="manual"),
                                     dict(image_name="", class_id=7, x=40, y=25, z=25, label=False,
                                          point_type="negative", sample_method="manual")])], ignore_index=True)
n0 = len(eng.session.calls)
ur = run_all(eng, df=upd, mode="update", store=store, classes=[2, 7, 3])   # engine is active on class 2
new_calls = eng.session.calls[n0:]
ok(ur[2].mode == "refine_warm" and ur[7].mode == "refine_cold" and ur[3] == ("skipped", 3, "unchanged"),
   f"update: class 2 (engine active) warm, class 7 cold, class 3 unchanged: {[ur[c].mode if not isinstance(ur[c], tuple) else ur[c][2] for c in classes]}")
ok([c[0] for c in new_calls] == ["point", "predict", "reset", "init_seg", "point", "predict"],
   f"cold: reset + previous result as initial mask + the new point; warm: no reset ({[c[0] for c in new_calls]})")
fresh = run_all(new_engine(), df=upd)
ok(all(np.array_equal(ur[c].full_mask_in(crop_all), fresh[c].full_mask_in(crop_all)) for c in (2, 7)),
   "warm and cold refinements == from-scratch runs with all points (fake model)")
mv = upd.copy()
mv.loc[mv.index[0], "x"] += 1
cmv = int(mv.class_id.iloc[0])
store2 = PS.ResultStore()
for r in fresh.values():
    store2.set(r)
ur2 = run_all(new_engine(), df=mv, mode="update", store=store2, classes=[cmv])
ok(ur2[cmv].mode == "scratch", "a moved point -> scratch")
ok(PS.plan_update(store2.get(3), PS.point_key(upd, 3), True, prompt_type="points+init")[0] == "scratch",
   "prompt type changed -> scratch")
ok(PS.plan_update(None, PS.point_key(upd, 3), True)[0] == "scratch", "no previous result -> scratch")

# capabilities
class SamLikeEngine(NE.NniEngine):
    name = "SAM-like"
    capabilities = frozenset({"points", "bbox"})


sam = SamLikeEngine(session_factory=lambda dev: FakeSession26(device=dev), buffer_factory=lambda s: np.zeros(s, np.uint8))
sam.load("m", "cpu")
try:
    list(PS.run_classes(sam, vol, seeds, dfp, [2], crop=crop_all, image_key="v", prompt_type="scribble"))
    ok(False, "an engine without 'scribble' rejects prompt_type='scribble'")
except ValueError as e:
    ok("scribble" in str(e), f"an engine without 'scribble' rejects prompt_type='scribble' ({e})")
ok([k for k, _ in sam.available_prompt_types()] == ["points"] and
   [k for k, _ in eng.available_prompt_types()] == ["points", "points+init", "scribble"],
   "available_prompt_types follows capabilities")
try:
    list(PS.run_classes(sam, vol, seeds, dfp, [2], crop=crop_all, image_key="v", init_masks=init_lab))
    ok(False, "an engine without 'init_mask' rejects init_masks")
except ValueError as e:
    ok("initial mask" in str(e), f"an engine without 'init_mask' rejects init_masks ({e})")
try:
    list(PS.run_classes(eng, vol, seeds, dfp, [2], crop=crop_all, image_key="v", prompt_type="lasso"))
    ok(False, "unknown prompt type rejected")
except ValueError:
    ok(True, "unknown prompt type rejected")

# drawn scribbles (positive + negative layers) + points
scr = np.zeros(shp, np.uint16)
scr[22:26, 20:30, 35:45] = 7
scr[9:11, 8:15, 8:15] = 2
negl = np.zeros(shp, np.uint16)
negl[26:28, 20:30, 35:45] = 7
eng = new_engine()
rd = run_all(eng, "scribble", scribbles=scr, neg_scribble=negl, classes=[7, 2])
c7 = normalize_calls(eng.session.calls)[0]
ok([c[0] for c in c7[:4]] == ["scribble", "predict", "scribble", "predict"] and c7[0][2] is True and c7[2][2] is False
   and c7[4][0] == "point", "drawn scribbles: positive, negative ('not this class' layer), then the points")
sess = FakeSession26(device="cpu")
sess.set_image(vol[None])
buf = np.zeros(shp, np.uint8)
sess.set_target_buffer(buf)
sess.reset_interactions()
sess.add_scribble_interaction((scr == 7).astype(np.uint8), True)
sess.add_scribble_interaction((negl == 7).astype(np.uint8), False)
for _, row in dfp[dfp.class_id == 7].iterrows():
    sess.add_point_interaction((row.z, row.y, row.x), bool(row.label))
ok(np.array_equal(rd[7].full_mask_in(crop_all), buf > 0), "drawn scribble result == direct fake session")
ok(run_all(new_engine(), "scribble", scribbles=scr, df=PP.empty_points_df(), classes=[3])[3][0] == "skipped",
   "class without scribble or points -> skipped")
ri = run_all(new_engine(), "points+init", df=PP.empty_points_df(), classes=[2])
ok(isinstance(ri[2], PS.ClassResult) and np.array_equal(ri[2].full_mask_in(crop_all), seeds == 2),
   "seed as initial mask, no points: one prediction from the mask")

# undo / release
eng = new_engine()
ok(eng.undo() is True and eng.session.calls[-1] == ("undo",), "undo() on a 2.6 session with undo support")
eng11b = NE.NniEngine(session_factory=lambda dev: FakeSession11(device=dev), buffer_factory=lambda s: np.zeros(s, np.uint8))
eng11b.load("m", "cpu")
ok(eng11b.undo() is False, "undo() is False on 1.1.3")
eng.release()
ok(not eng.loaded and eng.info == {} and eng.active is None, "release() drops the session")
eng.close()


class OOMError(RuntimeError):
    pass


OOMError.__name__ = "OutOfMemoryError"


class OOMSession(FakeSession26):
    def _predict(self):
        raise OOMError("CUDA out of memory")


eng = NE.NniEngine(session_factory=lambda dev: OOMSession(device=dev), buffer_factory=lambda s: np.zeros(s, np.uint8))
eng.load("m", "cpu")
try:
    run_all(eng, classes=[2])
    ok(False, "OOM recognised")
except PE.EngineOutOfMemory as e:
    ok("ROI" in str(e), f"OutOfMemoryError (by name) -> EngineOutOfMemory: {e}")

# ---------------------------------------------------------------- 8. overlap rules / compose
print("== 8. overlap rules, ResultStore")
rs = np.random.RandomState(11)
shp8 = (30, 40, 50)
masks = {}
for c in (1, 4, 9, 300):
    m = np.zeros(shp8, bool)
    for _ in range(3):
        _ball_into(m, tuple(int(rs.randint(5, n - 5)) for n in shp8), int(rs.randint(4, 9)), True)
    masks[c] = m
seeds8 = np.zeros(shp8, np.uint16)
for c, m in masks.items():
    seeds8[m & (rs.rand(*shp8) < 0.3)] = c
full_crop = PS.inference_crop(shp8)
store = PS.ResultStore()
for c, m in masks.items():
    store.set(PS.result_from_buffer(c, m, full_crop))
ref = np.zeros(shp8, np.uint16)
for c in sorted(masks, reverse=True):
    ref[masks[c]] = c
comp = store.compose(shp8, rule="min_id")
ok(np.array_equal(comp, ref) and comp.dtype == np.uint16, "'min_id' == nninter_predict's descending overwrite; id 300 -> uint16")
seedc = store.compose(shp8, rule="seed", seeds=seeds8)
count = sum(m.astype(int) for m in masks.values())
exp = ref.copy()
for c, m in masks.items():
    exp[(count > 1) & m & (seeds8 == c)] = c
ok(np.array_equal(seedc, exp) and (seedc != ref).any(), f"'seed' rule: contested voxels in a claiming class's seed keep it "
                                                         f"({int((seedc != ref).sum())} voxels differ from min_id)")
ok(store.overlap_voxels() == int((count > 1).sum()), f"overlap_voxels = {store.overlap_voxels()}")
reg = (slice(3, 20), slice(10, 33), slice(0, 50))
ok(np.array_equal(store.compose(shp8, rule="seed", seeds=seeds8, region=reg), seedc[reg]) and
   np.array_equal(store.compose(shp8, rule="min_id", region=reg), comp[reg]), "region compose == full compose restricted")
small = PS.ResultStore()
small.set(PS.result_from_buffer(5, masks[1], full_crop))
ok(small.compose(shp8).dtype == np.uint8, "ids <= 255 -> uint8")
newm = np.zeros(shp8, bool)
_ball_into(newm, (15, 20, 25), 7, True)
before = store.compose(shp8, rule="seed", seeds=seeds8)
region = store.set(PS.result_from_buffer(4, newm, full_crop))
after = store.compose(shp8, rule="seed", seeds=seeds8)
changed = before != after
xor = masks[4] ^ newm
inreg = np.zeros(shp8, bool)
inreg[region] = True
ok(changed.any() and not (changed & ~xor).any() and not (changed & ~inreg).any(),
   "after set(new): changed voxels within old XOR new of that class and within the returned region")
store.revert(4)
ok(np.array_equal(store.compose(shp8, rule="seed", seeds=seeds8), before) and not store.can_revert(4),
   "revert restores the previous result (single level)")
store.remove(9)
ok(9 not in store and not (store.compose(shp8, rule="min_id") == 9).any(), "remove: class left out of the output")
r_empty = PS.result_from_buffer(8, np.zeros((4, 5, 6), bool), (slice(2, 6), slice(0, 5), slice(1, 7)))
ok(r_empty.n_voxels == 0 and r_empty.mask.size == 0, "empty result")
rr_ = PS.result_from_buffer(8, np.ones((4, 5, 6), bool), (slice(2, 6), slice(0, 5), slice(1, 7)),
                            roi_mask=np.pad(np.ones((2, 5, 6), bool), ((0, 2), (0, 0), (0, 0))))
ok(rr_.slices == (slice(2, 4), slice(0, 5), slice(1, 7)) and rr_.n_voxels == 60, "result clipped to the ROI mask, tight bbox in full coords")


def keep_largest_id(claims, seeds_region):
    out = np.zeros(next(iter(claims.values())).shape, np.int64)
    for c in sorted(claims):
        out[claims[c]] = c
    return out


PS.register_overlap_rule("max_id", keep_largest_id)
store = PS.ResultStore()
for c, m in masks.items():
    store.set(PS.result_from_buffer(c, m, full_crop))
mx = np.zeros(shp8, np.uint16)
for c in sorted(masks):
    mx[masks[c]] = c
ok(np.array_equal(store.compose(shp8, rule="max_id"), mx), "register_overlap_rule: a custom rule is used")

# ---------------------------------------------------------------- 9. nninter_predict dtype
print("== 9. nninter_predict output dtype")
s9 = np.zeros((20, 20, 20), np.uint16)
s9[2:6, 2:6, 2:6] = 300
s9[10:14, 10:14, 10:14] = 7
d9 = PP.seeds_to_point_prompts(s9, np.ones(s9.shape), rng=np.random.RandomState(0), default_n_pos=2)
tot = quiet(NP.nninter_predict, FakeSession26(device="cpu"), np.ones((1, 20, 20, 20), np.uint8), df_pt=d9)
ok(tot.dtype == np.uint16 and set(np.unique(tot)) == {0, 7, 300}, f"class 300 kept (dtype {tot.dtype}; was uint8 -> 44)")
tot = quiet(NP.nninter_predict, FakeSession26(device="cpu"), np.ones((1, 20, 20, 20), np.uint8),
            df_pt=d9[d9.class_id == 7])
ok(tot.dtype == np.uint8, "ids <= 255 still uint8")
src = (ROOT / "sprout_core" / "nninteractive_predict.py").read_text(encoding="utf-8")
ok("use_pinned_memory=True)" in src and "return_per_class_masks=True" not in src.split("def nninter_main(")[0].split("def nninter_main_from_prompts")[1]
   and "scribble_mask=scribble_mask" in src, "nninter_main_from_prompts: make_session, no return_per_class_masks to nninter_predict, scribble by keyword")

# ---------------------------------------------------------------- 10. review fixes
print("== 10. review fixes (memory, warm refine after Revert)")
import tracemalloc  # noqa: E402


def peak_of(fn):
    tracemalloc.start()
    tracemalloc.reset_peak()
    try:
        res = fn()
        return res, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


big = (128, 512, 512)
st10 = PS.ResultStore()
crop10 = PS.inference_crop(big)
for c, sl in ((1, (slice(2, 30), slice(5, 60), slice(5, 60))), (3, (slice(20, 40), slice(40, 90), slice(40, 90))),
              (2, (slice(90, 120), slice(400, 500), slice(400, 500)))):
    m = np.zeros(tuple(x.stop - x.start for x in sl), bool)
    m[::2] = True
    st10.set(PS.result_from_buffer(c, m, sl))
for rule in ("min_id", "seed"):
    out10, pk = peak_of(lambda: st10.compose(big, rule=rule, seeds=None))
    ok(pk < 1.25 * out10.nbytes, f"compose({rule!r}) on {np.prod(big) / 1e6:.0f} M voxels: peak {pk / out10.nbytes:.2f}x "
                                 f"the output (no full-size count / contested arrays)")
ref10 = np.zeros(big, np.uint8)
cnt10 = np.zeros(big, np.uint8)
for c in sorted(st10.classes(), reverse=True):
    ref10[st10.get(c).slices][st10.get(c).mask] = c
    cnt10[st10.get(c).slices] += st10.get(c).mask
ok(np.array_equal(st10.compose(big, rule="min_id"), ref10) and st10.overlap_voxels() == int((cnt10 > 1).sum()) > 0,
   f"compose / overlap_voxels values unchanged ({st10.overlap_voxels()} contested voxels)")
del ref10, cnt10
s16 = np.zeros((256, 256, 256), np.uint16)
s16[10:200, 20:30, 30:40] = 7
s16[100:110, 100:200, 5:250] = 60000
st16, pk = peak_of(lambda: PP.class_stats(s16))
ok(pk < 1.0 * s16.nbytes, f"class_stats on {s16.size / 1e6:.0f} M uint16 voxels: peak {pk / s16.nbytes:.2f}x the seeds (chunked)")
u, cnt = np.unique(s16, return_counts=True)
alw = np.zeros(s16.shape, bool)
alw[50:150] = True
u2, c2 = np.unique(s16[alw], return_counts=True)
ok(st16.values.tolist() == [[int(a), int(b)] for a, b in zip(u, cnt) if a] and
   PP.class_stats(s16, allowed=alw).values.tolist() == [[int(a), int(b)] for a, b in zip(u2, c2) if a],
   "class_stats chunked == np.unique (with and without allowed)")

# warm refine after Revert must not continue from the reverted state
eng10 = new_engine()
p_base = dfp[dfp.class_id == 2].reset_index(drop=True)
store10 = PS.ResultStore()
for r in PS.run_classes(eng10, vol, seeds, p_base, [2], crop=crop_all, image_key="v"):
    store10.set(r)
A = store10.get(2)
p2 = pd.DataFrame([dict(image_name="", class_id=2, x=12, y=12, z=12, label=False, point_type="negative",
                        sample_method="manual")])
d_p2 = pd.concat([p_base, p2], ignore_index=True)
for r in PS.run_classes(eng10, vol, seeds, d_p2, [2], crop=crop_all, image_key="v", mode="update", store=store10):
    store10.set(r)
ok(store10.get(2).mode == "refine_warm", "add p2 -> warm refine (engine holds class 2's state)")
store10.revert(2)
p3 = pd.DataFrame([dict(image_name="", class_id=2, x=15, y=8, z=7, label=True, point_type="positive",
                        sample_method="manual")])
d_p3 = pd.concat([d_p2, p3], ignore_index=True)
n0 = len(eng10.session.calls)
res10 = list(PS.run_classes(eng10, vol, seeds, d_p3, [2], crop=crop_all, image_key="v", mode="update", store=store10))[0]
calls10 = eng10.session.calls[n0:]
from _prompt_check_utils import _h  # noqa: E402
ok(res10.mode == "refine_cold" and [c[0] for c in calls10[:2]] == ["reset", "init_seg"] and
   calls10[1][1] == _h(A.full_mask_in(crop_all)) and [c[1] for c in calls10 if c[0] == "point"] == [(12, 12, 12), (7, 8, 15)],
   f"after Revert, adding p3 -> cold refine from the reverted result A with p2 + p3 ({res10.mode})")

# set_image: 2.x needs no nonzero box; 1.x computes the mask once
img10 = np.full((128, 512, 512), 7, np.uint8)
buf10 = np.zeros(img10.shape, np.uint8)
eng26 = NE.NniEngine(session_factory=lambda dev: FakeSession26(device=dev), buffer_factory=lambda s: buf10)
eng26.load("m", "cpu")
_, pk = peak_of(lambda: eng26.set_image(img10, "big"))
ok(pk < 0.25 * img10.nbytes, f"set_image on 2.x: peak {pk / img10.nbytes:.2f}x the image (no != 0 temporaries)")
eng11c = NE.NniEngine(session_factory=lambda dev: FakeSession11(device=dev), buffer_factory=lambda s: buf10)
eng11c.load("m", "cpu")
img11 = img10.copy()
img11[:, :, :9] = 0
_, pk = peak_of(lambda: eng11c.set_image(img11, "big"))
# 1.x: the float32 copy for the session (4x) + one bool mask (1x)
ok(pk < 5.5 * img10.nbytes and not eng11c.valid_point(0, 0, 8) and eng11c.valid_point(0, 0, 9),
   f"set_image on 1.x: nonzero box from one mask (peak {pk / img10.nbytes:.2f}x, incl. the 4x float32 copy)")
try:
    eng26.set_image(np.zeros((4, 5, 6), np.uint8), "zero")
    ok(False, "all-zero image rejected")
except PE.PromptEngineError:
    ok(True, "all-zero image rejected (2.x)")
for e_ in (eng10, eng26, eng11c):
    e_.close()

print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
sys.exit(1 if FAILED else 0)
