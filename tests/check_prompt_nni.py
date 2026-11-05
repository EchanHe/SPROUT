"""
Checks of the Prompt (AI) core with the REAL nnInteractive model on the dog demo.

    env sprout-nni (nnInteractive 2.6, CUDA):   python tests/check_prompt_nni.py
    env nnInteractive (1.1.3, no napari):        python tests/check_prompt_nni.py --core-only

Weights: C:/Users/yichen/work/weights/nnInteractive_v1.0 (or --weights).
Prompts: the scratchpad prompts.csv of the nnInteractive comparison (or --prompts);
falls back to sampling 5 positive + 3 + 3 negative points per class from the seeds.
Deterministic cuDNN flags are set (make_session(deterministic=True)) before the
weights load; np.random is reseeded per class (base_seed + class id).
--core-only: no napari / Qt is imported (nothing here needs it) and any nnInteractive
version is accepted (1.1.3 adds the nonzero-crop point filter check).
Prints PASS/FAIL (and INFO numbers); exits non-zero if anything failed.
"""
import argparse
import contextlib
import io
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
os.chdir(ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--core-only", action="store_true")
ap.add_argument("--weights", default=r"C:/Users/yichen/work/weights/nnInteractive_v1.0")
ap.add_argument("--prompts", default=r"C:/Users/yichen/AppData/Local/Temp/claude/c--Users-yichen-work-SPROUT/"
                                     r"96cacaa1-c67f-461d-8b7d-9cf87a400006/scratchpad/nni_test/prompts.csv")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tifffile  # noqa: E402
import torch  # noqa: E402

from sprout_core import nni_engine as NE  # noqa: E402
from sprout_core import prompt_points as PP  # noqa: E402
from sprout_core import prompt_seg as PS  # noqa: E402
from sprout_core.boundary import Cancelled  # noqa: E402
from _prompt_check_utils import Roi, ROI_IS_STUB, dice  # noqa: E402

FAILED = []
TMP = Path(tempfile.mkdtemp(prefix="sprout_prompt_nni_"))


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILED.append(msg)


def info(msg):
    print("INFO " + msg, flush=True)


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


IMG_P, SEED_P = "demo_data/dog_img.tif", "demo_data/dog_seed.tif"
img = tifffile.imread(IMG_P)
seeds = tifffile.imread(SEED_P)
print(f"     image {img.shape} {img.dtype}, seeds ids {sorted(np.unique(seeds[seeds > 0]).tolist())}; "
      f"ROI {'STUB' if ROI_IS_STUB else 'real sprout_core.roi'}; torch {torch.__version__}, "
      f"cuda {torch.cuda.is_available()}")

# ---------------------------------------------------------------- status / version
st = NE.nninteractive_status()
if args.core_only:
    ok(st[0] and st[1][0].isdigit(), f"nninteractive_status() == (True, version): {st}")
else:
    ok(st == (True, "2.6.0"), f"nninteractive_status() == (True, '2.6.0'): {st}")
MAJOR = int(st[1].split(".")[0]) if st[0] else 0
if args.core_only:
    ok("napari" not in sys.modules and "qtpy" not in sys.modules, "--core-only: napari / Qt not imported")

if Path(args.prompts).exists():
    df = pd.read_csv(args.prompts)
    info(f"prompts {args.prompts}: {len(df)} points, classes {sorted(df.class_id.unique().tolist())}")
else:
    df = PP.seeds_to_point_prompts(seeds, img, rng=np.random.RandomState(0), default_n_pos=5, default_n_neg=3,
                                   image_name="dog_seed")
    df.to_csv(TMP / "prompts.csv", index=False)
    args.prompts = str(TMP / "prompts.csv")
    info(f"prompts sampled from the seeds: {len(df)} points")
CLASSES = sorted(df.class_id.unique().tolist(), reverse=True)        # nninter_predict's order

# ---------------------------------------------------------------- the fixed scripts
from sprout_core import nninteractive_predict as NP  # noqa: E402

t0 = time.perf_counter()
with quiet():
    out = NP.nninter_main_from_prompts(args.weights, IMG_P, str(TMP / "from_prompts"), df_pt_path=args.prompts,
                                       return_per_class_masks=True, device=args.device)
ok(isinstance(out, tuple) and len(out) == 2 and sorted(out[1]) == sorted(CLASSES) and out[0].shape == img.shape
   and all(out[1][c].any() for c in CLASSES),
   f"nninter_main_from_prompts runs (prompts.csv) and returns (total, per_class) "
   f"[{time.perf_counter() - t0:.0f} s]")
if isinstance(out, tuple):
    ref = np.zeros(img.shape, np.uint8)
    for c in CLASSES:
        ref[out[1][c]] = c
    ok(np.array_equal(ref, out[0]), "its total == per-class masks painted in descending order")
    info("from_prompts voxels per class " + str({c: int(out[1][c].sum()) for c in sorted(CLASSES)}))
del out
t0 = time.perf_counter()
with quiet():
    tot, log = NP.nninter_main(args.weights, IMG_P, SEED_P, str(TMP / "main"), args.device,
                               point_config=dict(default_n_pos=1, default_n_neg=0, default_method="random"),
                               return_per_class_masks=False)
ok(tot.shape == img.shape and set(np.unique(tot)) - {0} <= set(np.unique(seeds)) and (tot > 0).any(),
   f"nninter_main builds its session on nnInteractive {st[1]} and predicts [{time.perf_counter() - t0:.0f} s]")
del tot
torch.cuda.empty_cache()

# ---------------------------------------------------------------- one session for the rest
t0 = time.perf_counter()
session = NE.make_session(args.device)
ok(torch.backends.cudnn.benchmark is False and torch.backends.cudnn.deterministic is True,
   "make_session: cudnn.benchmark False, deterministic True right after construction")
with quiet():
    session.initialize_from_trained_model_folder(args.weights)
info(f"model loaded in {time.perf_counter() - t0:.1f} s ({type(session).__module__})")


class Seeded:
    """nninter_predict with np.random.seed(base + class) after each class's reset (as the engine)."""

    def __init__(self, s, order, base=0):
        self._s, self._q, self._base = s, list(order), base

    def reset_interactions(self):
        self._s.reset_interactions()
        if self._q:
            np.random.seed(self._base + int(self._q.pop(0)))

    def __getattr__(self, k):
        return getattr(self._s, k)


def script_run():
    per = {}
    t = time.perf_counter()
    with quiet():
        total = NP.nninter_predict(Seeded(session, CLASSES), img[None], df_pt=df,
                                   on_class_predicted=lambda c, m: per.__setitem__(int(c), m.copy()))
    return total, per, time.perf_counter() - t


torch.cuda.reset_peak_memory_stats()
tot_a, per_a, ta = script_run()
peak_full = torch.cuda.max_memory_allocated() / 1e9
tot_b, per_b, tb = script_run()
same = np.array_equal(tot_a, tot_b) and all(np.array_equal(per_a[c], per_b[c]) for c in CLASSES)
dices = {c: round(dice(per_a[c], per_b[c]), 5) for c in CLASSES}
ok(same, f"determinism: two nninter_predict runs identical with the flags + per-class seeding "
         f"(Dice {dices}; {ta:.0f} s / {tb:.0f} s)")
info(f"script voxels per class {({c: int(per_a[c].sum()) for c in sorted(CLASSES)})}; peak GPU {peak_full:.2f} GB")

# ---------------------------------------------------------------- engine == script path
eng = NE.NniEngine()
eng.adopt(session)
ok(eng.info["major"] == MAJOR and eng.capabilities == {"points", "scribble", "init_mask"},
   f"engine.adopt(session): {eng.info}")
full_crop = PS.inference_crop(img.shape)
store = PS.ResultStore()
t = time.perf_counter()
for r in PS.run_classes(eng, img, seeds, df, CLASSES, crop=full_crop, image_key="dog"):
    store.set(r)
te = time.perf_counter() - t
comp = store.compose(img.shape, rule="min_id")
eng_masks = {c: store.get(c).full_mask_in(full_crop) for c in CLASSES}
exact = np.array_equal(comp, tot_a) and all(np.array_equal(eng_masks[c], per_a[c]) for c in CLASSES)
d = {c: round(dice(eng_masks[c], per_a[c]), 5) for c in CLASSES}
ok(exact or min(d.values()) >= 0.995,
   f"engine (adopt + run_classes, rule min_id) == nninter_predict: {'EXACT' if exact else 'not exact'}; "
   f"Dice {d} [{te:.0f} s]")
info(f"engine thread names used: sprout-nni; dropped points {({c: store.get(c).n_dropped for c in CLASSES})}")

# ---------------------------------------------------------------- update one class
C = 4 if 4 in CLASSES else CLASSES[0]
from scipy.ndimage import distance_transform_edt  # noqa: E402
pred = eng_masks[C]
far = distance_transform_edt(seeds != C) * pred
z, y, x = (int(v) for v in np.unravel_index(np.argmax(far), far.shape))
neg = pd.DataFrame([dict(image_name="dog_seed", class_id=C, x=x, y=y, z=z, label=False, point_type="negative",
                         sample_method="manual")])
df2 = pd.concat([df, neg], ignore_index=True)
before = {c: store.get(c) for c in CLASSES}
comp_before = store.compose(img.shape, rule="min_id")
order = [c for c in CLASSES if c != C] + [C]
ups = {}
region = None
for r in PS.run_classes(eng, img, seeds, df2, order, crop=full_crop, image_key="dog", mode="update", store=store):
    if isinstance(r, tuple):
        ups[r[1]] = r
    else:
        ups[r.class_id] = r
        region = store.set(r)
ok(all(ups[c] == ("skipped", c, "unchanged") for c in CLASSES if c != C) and
   not isinstance(ups[C], tuple) and ups[C].mode == "refine_cold",
   f"Update after a negative point at {(z, y, x)} in class {C}: only class {C} re-run "
   f"({ups[C].mode if not isinstance(ups[C], tuple) else ups[C]})")
ok(all(store.get(c) is before[c] for c in CLASSES if c != C), "other classes' stored results unchanged (same objects)")
comp_after = store.compose(img.shape, rule="min_id")
new_c = store.get(C).full_mask_in(full_crop)
changed = comp_after != comp_before
inreg = np.zeros(img.shape, bool)
if region is not None:
    inreg[region] = True
ok(not (changed & ~(pred ^ new_c)).any() and not (changed & ~inreg).any(),
   f"composite changes ({int(changed.sum())} voxels) within class {C} old XOR new and the returned region")
info(f"cold refine class {C}: {int(pred.sum())} -> {int(new_c.sum())} voxels, IoU vs before "
     f"{(pred & new_c).sum() / max((pred | new_c).sum(), 1):.4f}, voxel at the negative point now "
     f"{bool(new_c[z, y, x])}")

# warm refine: engine is active on class C now; add one more negative point
far2 = distance_transform_edt(seeds != C) * new_c
z2, y2, x2 = (int(v) for v in np.unravel_index(np.argmax(far2), far2.shape))
df3 = pd.concat([df2, pd.DataFrame([dict(image_name="dog_seed", class_id=C, x=x2, y=y2, z=z2, label=False,
                                         point_type="negative", sample_method="manual")])], ignore_index=True)
rw = list(PS.run_classes(eng, img, seeds, df3, [C], crop=full_crop, image_key="dog", mode="update", store=store))
rw = rw[0]
w_mask = rw.full_mask_in(full_crop) if not isinstance(rw, tuple) else None
ok(w_mask is not None and rw.mode == "refine_warm" and w_mask.any(),
   f"warm refine (engine still on class {C}): {rw.mode if w_mask is not None else rw}, "
   f"{int(w_mask.sum()) if w_mask is not None else 0} voxels, IoU vs previous "
   f"{(w_mask & new_c).sum() / max((w_mask | new_c).sum(), 1):.4f}" if w_mask is not None else "warm refine")

# ---------------------------------------------------------------- other prompt types (smoke)
def one(ptype, cls, **kw):
    t = time.perf_counter()
    res = list(PS.run_classes(eng, img, seeds, kw.pop("df", df), [cls], crop=kw.pop("crop", full_crop),
                              image_key="dog", prompt_type=ptype, **kw))[0]
    return res, time.perf_counter() - t


S = 5 if 5 in CLASSES else CLASSES[0]
ref_s = eng_masks[S]
for label, ptype, kw in (("seed as initial mask + points", "points+init", {}),
                         ("seed as initial mask, no points", "points+init", dict(df=PP.empty_points_df())),
                         ("seed scribble (+ negative from other classes) + points", "scribble", dict(neg_scribble=True)),
                         ("seed scribble only", "scribble", dict(df=PP.empty_points_df()))):
    r, t = one(ptype, S, **kw)
    m = r.full_mask_in(full_crop) if not isinstance(r, tuple) else np.zeros(img.shape, bool)
    ok(m.any(), f"{label}: class {S} non-empty ({int(m.sum())} voxels, IoU vs points "
                f"{(m & ref_s).sum() / max((m | ref_s).sum(), 1):.3f}, {t:.0f} s)")
drawn = np.zeros(img.shape, np.uint16)
zs = np.flatnonzero(np.any(seeds == S, axis=(1, 2)))
zmid = int(zs[len(zs) // 2])
drawn[zmid][seeds[zmid] == S] = S                       # one painted slice of the seed
negl = np.zeros(img.shape, np.uint16)
r, t = one("scribble", S, scribbles=drawn, neg_scribble=negl, df=PP.empty_points_df())
m = r.full_mask_in(full_crop) if not isinstance(r, tuple) else np.zeros(img.shape, bool)
ok(m.any(), f"drawn scribble (one painted slice z={zmid}) of class {S}: non-empty ({int(m.sum())} voxels, IoU vs "
            f"points {(m & ref_s).sum() / max((m | ref_s).sum(), 1):.3f}, {t:.0f} s)")

# ---------------------------------------------------------------- ROI around class S
zz, yy, xx = (np.flatnonzero(np.any(seeds == S, axis=tuple(a for a in range(3) if a != ax))) for ax in range(3))
lo = (max(int(zz[0]) - 15, 0), max(int(yy[0]) - 15, 0), max(int(xx[0]) - 15, 0))
hi = (int(zz[-1]) + 16, int(yy[-1]) + 16, int(xx[-1]) + 16)
roi = Roi.box(img.shape, lo, hi)
crop = PS.inference_crop(img.shape, roi, margin=32)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t = time.perf_counter()
rr = list(PS.run_classes(eng, img, seeds, df, [S], crop=crop, image_key="dog", roi=roi))[0]
t_roi = time.perf_counter() - t
peak_roi = torch.cuda.max_memory_allocated() / 1e9
mr = rr.full_mask_in(full_crop)
rm = roi.mask_full()
ok(mr.any() and not (mr & ~rm).any(), f"ROI box {roi.describe()}: class {S} result only inside the ROI")
info(f"ROI run: crop {tuple(s.stop - s.start for s in crop)} of {img.shape}, {t_roi:.1f} s, peak GPU "
     f"{peak_roi:.2f} GB (whole image script run peak {peak_full:.2f} GB); Dice vs whole-image result "
     f"inside the ROI {dice(mr, ref_s & rm):.4f}; points dropped {rr.n_dropped}")

# ---------------------------------------------------------------- 1.x nonzero-crop point filter
vz = img.copy()
vz[:, :, :20] = 0                                        # zero slab: 1.x crops it away in set_image
pt = df[df.class_id == S].iloc[0]
outside = pd.DataFrame([dict(image_name="", class_id=S, x=5, y=int(pt.y), z=int(pt.z), label=False,
                             point_type="negative", sample_method="manual")])
dfz = pd.concat([df[df.class_id == S], outside], ignore_index=True)
dfz = dfz[(dfz.x >= 20) | (dfz.x == 5)]
rz = list(PS.run_classes(eng, vz, seeds, dfz, [S], crop=full_crop, image_key="dog_zero_slab"))[0]
exp_drop = 1 if MAJOR < 2 else 0
ok(not isinstance(rz, tuple) and rz.n_dropped == exp_drop and rz.n_voxels > 0,
   f"nnInteractive {MAJOR}.x: point in the image's zero slab {'dropped (outside the nonzero crop)' if MAJOR < 2 else 'kept (2.x does not crop)'}"
   f" -> n_dropped {getattr(rz, 'n_dropped', rz)}")

# ---------------------------------------------------------------- Stop after the first class
stop = threading.Event()
got = []
try:
    for r in PS.run_classes(eng, img, seeds, df, CLASSES, crop=full_crop, image_key="dog", cancel=stop):
        got.append(r)
        stop.set()
    ok(False, "Stop after the first class raises Cancelled")
except Cancelled:
    ok(len(got) == 1, f"Stop after the first class: Cancelled, {len(got)} result")
stop2 = threading.Event()
try:
    list(PS.run_classes(eng, img, seeds, df, [S], crop=full_crop, image_key="dog", cancel=stop2,
                        on_point=lambda i, n: stop2.set() if i == 2 else None))
    ok(False, "Stop between points raises Cancelled")
except Cancelled:
    ok(eng.active is None, "Stop between points: Cancelled, engine state not reused")

# ---------------------------------------------------------------- the Qt tab with the real model
if not args.core_only:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, str(ROOT / "napari_sprout"))
    import napari  # noqa: E402
    import _headless_gl  # noqa: E402,F401  (napari >= 0.7 offscreen: no GL context)
    from qtpy.QtWidgets import QApplication  # noqa: E402
    from napari_sprout.widgets.prompt_tab import PromptTab  # noqa: E402
    import napari_sprout.widgets.prompt_tab as _prompt_tab  # noqa: E402
    _prompt_tab.SETTINGS = ("SPROUT-checks", "napari-sprout-checks")   # not the user's settings

    def wait(cond, timeout=900):
        t_ = time.time()
        while not cond():
            QApplication.processEvents()
            time.sleep(0.005)
            if time.time() - t_ > timeout:
                raise TimeoutError

    viewer = napari.Viewer(show=False)
    viewer.add_image(img, name="dog_img")
    viewer.add_labels(seeds, name="dog_seed")
    tab = PromptTab(viewer, engine_factory=lambda: eng)          # the engine adopted above (real session)
    tab._finalizer.detach()                                       # the engine is shared with later steps
    wait(lambda: tab.table.rowCount() == len(CLASSES))
    ok(tab.available and tab.banner.isHidden() and tab.run_btn.isEnabled(),
       f"tab with the real model: no banner, Run enabled ({tab._status})")
    tab.set_prompt_mode("point")
    tab.set_prompts(df)
    tab.set_checked(CLASSES)
    t = time.perf_counter()
    tab.run_checked()
    wait(lambda: not any(tab._busy.values()))
    t_tab = time.perf_counter() - t
    out = viewer.layers["dog_seed_nni"].data
    ref_store = PS.ResultStore()
    for r in PS.run_classes(eng, img, seeds, tab.prompts_df(), sorted(CLASSES), crop=full_crop, image_key="dog-ref"):
        ref_store.set(r)
    ref_comp = ref_store.compose(img.shape, rule="min_id")
    dd = {c: round(dice(out == c, ref_comp == c), 5) for c in CLASSES}
    ok(np.array_equal(out, ref_comp), f"tab Run checked (points csv) == engine run_classes + compose('min_id') in the "
                                      f"same process: {'EXACT' if np.array_equal(out, ref_comp) else 'not exact'}, "
                                      f"Dice {dd} [{t_tab:.0f} s]")
    viewer.close()

    # ------------------------------------------------ the tab == nninter_main (the CLI pipeline)
    print("== tab == nninter_main (`sprout.py --prompt --nninteractive`) on dog_img + dog_seed", flush=True)
    from scipy.ndimage import grey_dilation  # noqa: E402

    init_p = TMP / "dog_grown.tif"                       # an initial segmentation (init_seg_path)
    tifffile.imwrite(init_p, grey_dilation(seeds, size=(3, 7, 7)))
    v2 = napari.Viewer(show=False)
    v2.open(str(ROOT / IMG_P))
    v2.open(str(ROOT / SEED_P), layer_type="labels")
    v2.open(str(init_p), layer_type="labels")
    tab2 = PromptTab(v2, engine_factory=lambda: eng)
    tab2._finalizer.detach()
    wait(lambda: tab2.table.rowCount() == len(CLASSES))
    tab2.model_edit.setText(args.weights)
    tab2.per_class_cb.setChecked(True)
    YDEF = {k: v.get("default") for k, v in __import__("sprout_core.config_core", fromlist=["x"])
            .input_val_nninteractive_run.items() if "default" in v}

    def run_main(case, *, seed=0, **kw):
        """nninter_main as `run_nninteractive_yaml` calls it (schema defaults, seed 0), per-class masks from
        its files. No wrapper: the pipeline seeds np.random itself (seed, and seed + class id per class)."""
        outd = TMP / f"main_{case}_{seed}"
        point_config = {k: YDEF[k] for k in ("default_n_pos", "default_n_neg", "default_method", "negative_from_bg",
                                            "negative_from_other_classes", "negative_per_other_class")}
        point_config["class_config"] = None
        with quiet():
            total, _ = NP.nninter_main(args.weights, str(ROOT / IMG_P), str(ROOT / SEED_P), str(outd), args.device,
                                       init_seg_path=kw.get("init"), prompt_type=kw["prompt_type"],
                                       point_config=point_config, return_per_class_masks=True,
                                       scribble_config={"use_negative_scribble": kw.get("neg", False)}, seed=seed)
        per = {c: tifffile.imread(outd / f"dog_img_class_{c}.tif") > 0 for c in CLASSES}
        return total, per

    def run_tab(*, prompt_type, neg=False, init=False):
        tab2.set_prompt_mode(prompt_type)
        tab2.neg_scribble_cb.setChecked(neg)
        tab2.init_combo.setCurrentIndex(tab2.init_combo.findData("dog_grown") if init else 0)
        tab2.set_prompts(PP.empty_points_df())
        tab2.set_checked(CLASSES)
        if prompt_type == "point":                 # Generate points with the YAML defaults
            tab2.generate()
            wait(lambda: not any(tab2._busy.values()))
        tab2.run_checked()
        wait(lambda: not any(tab2._busy.values()))
        total = np.array(v2.layers["dog_seed_nni"].data)
        per = {c: np.array(v2.layers[f"dog_seed_nni_class_{c}"].data) > 0 for c in CLASSES}
        return total, per

    for case, kw in (("a scribble", dict(prompt_type="scribble")),
                     ("b scribble + use_negative_scribble", dict(prompt_type="scribble", neg=True)),
                     ("c point (YAML defaults)", dict(prompt_type="point")),
                     ("d scribble + init_seg_path", dict(prompt_type="scribble", init=True))):
        t0 = time.perf_counter()
        tt, tp = run_tab(prompt_type=kw["prompt_type"], neg=kw.get("neg", False), init=kw.get("init", False))
        t1 = time.perf_counter()
        mt, mp = run_main(case[0], prompt_type=kw["prompt_type"], neg=kw.get("neg", False),
                          init=str(init_p) if kw.get("init") else None)
        exact_t = np.array_equal(tt, mt) and tt.dtype == np.promote_types(mt.dtype, np.uint16)   # layer >= uint16
        exact_p = all(np.array_equal(tp[c], mp[c]) for c in CLASSES)
        dd = {c: round(dice(tp[c], mp[c]), 5) for c in CLASSES}
        ok(exact_t and exact_p, f"({case}) tab Run == nninter_main: total {'EXACT' if exact_t else 'differs'} "
                                f"({tt.dtype}), per-class {'EXACT' if exact_p else 'differ'}; Dice {dd} "
                                f"[tab {t1 - t0:.0f} s, script {time.perf_counter() - t1:.0f} s]")
        if case[0] in "ac":
            ut, up = run_main(case[0] + "u", seed=None, prompt_type=kw["prompt_type"])
            du = {c: round(dice(tp[c], up[c]), 5) for c in CLASSES}
            info(f"({case}) the script with seed=None (the behaviour before the fixed seed): total "
                 f"{'EXACT' if np.array_equal(tt, ut) else 'differs'}, per-class Dice {du}")
    v2.close()

    # ------------------------------------------------ order independence of the script path
    def script_classes(scribble, seed):
        per = {}
        with quiet():
            NP.nninter_predict(session, img[None], scribble_mask=scribble, seed=seed,
                               on_class_predicted=lambda c, m: per.__setitem__(int(c), m.copy()))
        return per

    only3 = np.where(seeds == 3, seeds, 0)
    all_ = script_classes(seeds, 0)
    one = script_classes(only3, 0)
    d3 = dice(all_[3], one[3])
    ok(np.array_equal(all_[3], one[3]), f"nninter_predict (seed 0): class 3 alone == class 3 among {{3, 4, 5, 6}} "
                                        f"(order independent; Dice {d3:.5f})")
    all_n, one_n = script_classes(seeds, None), script_classes(only3, None)
    info(f"with seed=None (no reseeding): class 3 alone vs among all {'EXACT' if np.array_equal(all_n[3], one_n[3]) else 'differs'} "
         f"(Dice {dice(all_n[3], one_n[3]):.5f})")

# ---------------------------------------------------------------- engine.load / release
t = time.perf_counter()
eng2 = NE.NniEngine()
with quiet():
    inf = eng2.load(args.weights, args.device)
    inf2 = eng2.load(args.weights, args.device)
ok(inf == inf2 and inf["major"] == MAJOR and inf["device"].startswith(args.device),
   f"NniEngine.load caches by (model_dir, device): {inf} [{time.perf_counter() - t:.0f} s]")
eng2.set_image(img, "dog")
pts = [(int(r.z), int(r.y), int(r.x), bool(r.label)) for r in df[df.class_id == S].itertuples()]
m2 = eng2.predict_class(pts, seed=S, class_id=S)
ok(np.array_equal(m2, per_a[S]) or dice(m2, per_a[S]) >= 0.995,
   f"engine loaded by itself == script for class {S} (Dice {dice(m2, per_a[S]):.5f}"
   f"{', EXACT' if np.array_equal(m2, per_a[S]) else ''})")
mem0 = torch.cuda.memory_allocated() / 1e9
eng2.release()
eng2.close()
eng.release()
del session
import gc  # noqa: E402
gc.collect()
torch.cuda.empty_cache()
info(f"GPU allocated before / after release: {mem0:.2f} / {torch.cuda.memory_allocated() / 1e9:.2f} GB")

shutil.rmtree(TMP, ignore_errors=True)
print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
sys.exit(1 if FAILED else 0)
