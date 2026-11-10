"""
Headless check of the run framework (round 2: C1, A5, C3, C7).

    core   make_seeds / make_adaptive_seed_ero / _thre / grow_mp take cancel= and progress=:
           progress counts every step up to its total; cancel raises sprout_core.Cancelled at
           the next step (files written so far stay); without them nothing changes (same
           arrays; the saved config has no cancel / progress key). Stop latency (INFO).
    tabs   Seeds, Grow, Boundary Compute through utils/jobs.RunControl:
             start   grey "…" line, run button off / Stop on, progress bar busy then real
             Stop    orange "Stopped after …" (+ files written), no layer added, buttons back,
                     the job thread ends by itself (no terminate), nothing written afterwards
             done    green "✓ … (t s)", one notification
             failed  MemoryError injected: red "<X> failed: not enough memory …", one
                     notification, no dialog
           Boundary Create: green line, no notification (it would repeat the line)
    texts  no "..." / "Please select" in the status texts of the run tabs

    python tests/check_jobs.py [--root <tree>] [--shots <dir>]

--root runs it against another source tree (the base); --shots saves screenshots (dark
stylesheet, 500 px wide) of Seeds / Grow / Boundary while running, stopped, failed and done.
Prints PASS/FAIL per check; exits non-zero if anything failed.
"""
import contextlib
import glob
import io
import os
import re
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _docks_common import pick_root  # noqa: E402

ROOT = pick_root(sys.argv)
SHOTS = Path(sys.argv[sys.argv.index("--shots") + 1]) if "--shots" in sys.argv else None
if SHOTS:
    SHOTS.mkdir(parents=True, exist_ok=True)

import numpy as np  # noqa: E402
import tifffile  # noqa: E402

from sprout_core import make_adaptive_seed  # noqa: E402
from sprout_core import make_grow  # noqa: E402
from sprout_core import make_seeds  # noqa: E402
import sprout_core.sprout_core as sc  # noqa: E402
from sprout_core.boundary import Cancelled  # noqa: E402

assert Path(sc.__file__).resolve().is_relative_to(ROOT.resolve()), sc.__file__
print(f"tree: {ROOT}")
FAILED = []
TMP = Path(tempfile.mkdtemp(prefix="sprout_jobs_"))


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILED.append(msg)


def info(msg):
    print("INFO " + msg, flush=True)


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def tifs(folder):
    return sorted(glob.glob(os.path.join(str(folder), "**", "*.tif"), recursive=True))


def yaml_texts(folder):
    return "".join(Path(p).read_text() for p in glob.glob(os.path.join(str(folder), "**", "config_*.yaml"),
                                                          recursive=True))


def try_run(fn, **kw):
    """(result, exception) of fn(**kw), quietly."""
    try:
        return quiet(fn, **kw), None
    except Exception as e:                  # noqa: BLE001
        return None, e


class Recorder:
    """progress(done, total, text); sets `cancel` once `stop_at` steps are done."""

    def __init__(self, cancel=None, stop_at=None):
        self.calls, self.cancel, self.stop_at = [], cancel, stop_at
        self.lock = threading.Lock()
        self.times = []

    def __call__(self, done, total, text=""):
        with self.lock:
            self.calls.append((done, total, text))
            self.times.append(time.perf_counter())
        if self.stop_at is not None and done >= self.stop_at:
            self.cancel.set()

    @property
    def total(self):
        return self.calls[-1][1] if self.calls else None

    def max_done(self):
        return max((c[0] for c in self.calls), default=0)


dog_full = tifffile.imread(ROOT / "demo_data" / "dog_img.tif")
dseed_full = tifffile.imread(ROOT / "demo_data" / "dog_seed.tif")
dog, dseed = dog_full[200:420], dseed_full[200:420]

# ======================================================================
print("== core: progress, cancel, defaults unchanged")
# ---- make_seeds (Original): 3 thresholds x 4 preset footprints x (2 + 1) seeds = 36 steps
kw = dict(img=dog, thresholds=[130, 150, 170], upper_thresholds=[255, 255, 255], erosion_steps=2,
          segments=8, num_threads=4, base_name="s", return_for_napari=True)
ref, e0 = try_run(make_seeds.make_seeds, output_folder=str(TMP / "s_ref"), **kw)
rec = Recorder()
got, e1 = try_run(make_seeds.make_seeds, output_folder=str(TMP / "s_prog"), progress=rec, **kw)
ok(e0 is None and e1 is None and rec.total == 36 and len(rec.calls) == 36 and rec.max_done() == 36,
   f"make_seeds: progress called after every seed, up to its total ({len(rec.calls)} calls, total {rec.total}, "
   f"{e1!r})")
same = (ref is not None and got is not None and ref[0].keys() == got[0].keys()
        and all(np.array_equal(ref[0][k], got[0][k]) and ref[0][k].dtype == got[0][k].dtype for k in ref[0]))
ok(same and [Path(p).name for p in tifs(TMP / "s_ref")] == [Path(p).name for p in tifs(TMP / "s_prog")],
   "make_seeds: same seeds and files with a progress hook")
ok(e1 is None and "progress" not in yaml_texts(TMP / "s_prog") and "cancel" not in yaml_texts(TMP / "s_prog"),
   "make_seeds: the saved config has no progress / cancel key")
ev = threading.Event()
rec = Recorder(ev, stop_at=5)
_, e = try_run(make_seeds.make_seeds, output_folder=str(TMP / "s_stop"), progress=rec, cancel=ev, **kw)
n_files = len(tifs(TMP / "s_stop"))
ok(isinstance(e, Cancelled) and 5 <= n_files < 36 and n_files == rec.max_done(),
   f"make_seeds: cancel -> Cancelled at the next step ({type(e).__name__}); the {n_files} seeds written stay "
   f"(= {rec.max_done()} steps done)")

# ---- adaptive (erosion): 1 initial + 4 erosion steps
kw = dict(img=dog, threshold=140, upper_threshold=255, erosion_steps=4, segments=10, num_threads=2,
          footprints="ball", base_name="ae", return_for_napari=True)
ref, e0 = try_run(make_adaptive_seed.make_adaptive_seed_ero, output_folder=str(TMP / "ae_ref"), **kw)
rec = Recorder()
got, e1 = try_run(make_adaptive_seed.make_adaptive_seed_ero, output_folder=str(TMP / "ae_prog"), progress=rec, **kw)
ok(e1 is None and rec.total == 5 and rec.max_done() == 5,
   f"make_adaptive_seed_ero: progress up to its total ({rec.calls[-1:] if rec.calls else e1!r})")
ok(ref is not None and got is not None and all(np.array_equal(ref[0][k], got[0][k]) for k in ref[0])
   and ref[1] == got[1], "make_adaptive_seed_ero: same result with a progress hook")
ev = threading.Event()
_, e = try_run(make_adaptive_seed.make_adaptive_seed_ero, output_folder=str(TMP / "ae_stop"),
               progress=Recorder(ev, stop_at=2), cancel=ev, **kw)
ok(isinstance(e, Cancelled) and not os.path.exists(TMP / "ae_stop" / "ae" / "FINAL_adaptive_seed.tif"),
   f"make_adaptive_seed_ero: cancel -> Cancelled, no final seed ({type(e).__name__})")

# ---- adaptive (thresholds): 4 thresholds
kw = dict(img=dog, thresholds=[130, 150, 170, 190], upper_thresholds=[255] * 4, erosion_steps=1, segments=10,
          num_threads=2, footprints="ball", base_name="at", return_for_napari=True, no_split_max_iter=10)
ref, e0 = try_run(make_adaptive_seed.make_adaptive_seed_thre, output_folder=str(TMP / "at_ref"), **kw)
rec = Recorder()
got, e1 = try_run(make_adaptive_seed.make_adaptive_seed_thre, output_folder=str(TMP / "at_prog"), progress=rec, **kw)
ok(e1 is None and rec.total == 4 and rec.max_done() == 4,
   f"make_adaptive_seed_thre: progress up to its total ({rec.calls[-1:] if rec.calls else e1!r})")
ok(ref is not None and got is not None and all(np.array_equal(ref[0][k], got[0][k]) for k in ref[0])
   and ref[1] == got[1], "make_adaptive_seed_thre: same result with a progress hook")
ev = threading.Event()
_, e = try_run(make_adaptive_seed.make_adaptive_seed_thre, output_folder=str(TMP / "at_stop"),
               progress=Recorder(ev, stop_at=1), cancel=ev, **kw)
ok(isinstance(e, Cancelled), f"make_adaptive_seed_thre: cancel -> Cancelled ({type(e).__name__})")

# ---- grow_mp: 2 thresholds x 5 iterations
kw = dict(img=dog, seg=dseed, thresholds=[150, 120], upper_thresholds=[255, 255], dilation_steps=[5, 5],
          num_threads=4, base_name="g", return_for_napari=True)
ref, e0 = try_run(make_grow.grow_mp, output_folder=str(TMP / "g_ref"), **kw)
rec = Recorder()
got, e1 = try_run(make_grow.grow_mp, output_folder=str(TMP / "g_prog"), progress=rec, **kw)
ok(e1 is None and rec.total == 10 and rec.max_done() == 10,
   f"grow_mp: progress after every iteration, up to its total ({rec.calls[-1:] if rec.calls else e1!r})")
ok(ref is not None and got is not None and ref[0].keys() == got[0].keys()
   and all(np.array_equal(ref[0][k], got[0][k]) for k in ref[0]), "grow_mp: same result with a progress hook")
ok(e1 is None and "progress" not in yaml_texts(TMP / "g_prog") and "cancel" not in yaml_texts(TMP / "g_prog"),
   "grow_mp: the saved config has no progress / cancel key")
ev = threading.Event()
rec = Recorder(ev, stop_at=7)
_, e = try_run(make_grow.grow_mp, output_folder=str(TMP / "g_stop"), progress=rec, cancel=ev, **kw)
ok(isinstance(e, Cancelled) and rec.max_done() == 7 and len(tifs(TMP / "g_stop")) == 1
   and not os.path.exists(TMP / "g_stop" / "g" / "FINAL_GROW_g.tif"),
   f"grow_mp: cancel -> Cancelled at the next iteration ({type(e).__name__}); the first stage's file stays, "
   f"no final file")
rec = Recorder()
_, e = try_run(make_grow.grow_mp, output_folder=str(TMP / "g_early"), progress=rec,
               **dict(kw, dilation_steps=[40, 40], min_growth_size=10 ** 9, no_growth_max_iter=2))
ok(e is None and rec.max_done() == 80, f"grow_mp: an early stop counts its stage as done ({rec.calls[-2:]})")


# ---- Stop latency (from setting the event to Cancelled), on the full dog
def longest_step(fn, kw):
    """Longest time between two progress calls (from the start) of a run without Stop."""
    rec = Recorder()
    t0 = time.perf_counter()
    quiet(fn, progress=rec, **kw)
    return float(np.max(np.diff([t0] + rec.times)))


def latency(fn, kw, wait_steps=1):
    """Stop right after the first step: the next one has just started, so ~ one step."""
    ev = threading.Event()
    rec = Recorder()
    out = {}

    def run():
        try:
            quiet(fn, progress=rec, cancel=ev, **kw)
        except Exception as e:              # noqa: BLE001
            out["e"] = e
        out["t"] = time.perf_counter()

    th = threading.Thread(target=run)
    th.start()
    while len(rec.calls) < wait_steps and th.is_alive():
        time.sleep(0.001)
    t0 = time.perf_counter()
    ev.set()
    th.join()
    return out.get("e"), out["t"] - t0


if "Cancelled" in vars(sc):
    for name, fn, kw in (
            ("make_seeds (Original, 4 thresholds, 3 erosions)", make_seeds.make_seeds,
             dict(img=dog_full, thresholds=[120, 140, 160, 180], upper_thresholds=[255] * 4, erosion_steps=3,
                  segments=10, num_threads=4, output_folder=str(TMP / "lat_s"), base_name="l")),
            ("make_adaptive_seed_thre (4 thresholds)", make_adaptive_seed.make_adaptive_seed_thre,
             dict(img=dog_full, thresholds=[120, 140, 160, 180], upper_thresholds=[255] * 4, erosion_steps=2,
                  segments=20, num_threads=4, footprints="ball", output_folder=str(TMP / "lat_a"), base_name="l",
                  no_split_max_iter=10)),
            ("grow_mp (2 x 20 iterations)", make_grow.grow_mp,
             dict(img=dog_full, seg=dseed_full, thresholds=[150, 120], dilation_steps=[20, 20], num_threads=4,
                  output_folder=str(TMP / "lat_g"), base_name="l", no_growth_max_iter=None))):
        step = longest_step(fn, kw)
        e, lat = latency(fn, kw)
        ok(isinstance(e, Cancelled), f"{name}: Stop -> Cancelled ({type(e).__name__})")
        info(f"Stop latency on dog {dog_full.shape}: {name}: {lat:.3f} s (longest step {step:.3f} s)")
else:
    ok(False, "core: no sprout_core.Cancelled / cancel= (Stop latency not measurable)")

# ======================================================================
print("\n== tabs: start / progress / Stop / done / failed")
import napari  # noqa: E402
from napari.qt import get_stylesheet  # noqa: E402
from qtpy.QtWidgets import QApplication, QMessageBox, QScrollArea  # noqa: E402

import napari_sprout.widgets.prompt_tab as PTM  # noqa: E402

PTM.SETTINGS = ("SPROUT-checks", "napari-sprout-checks")      # not the user's settings
import napari_sprout.widgets.boundary_widget as BWM  # noqa: E402
import napari_sprout.widgets.grow_widget as GWM  # noqa: E402
import napari_sprout.widgets.seed_widget as SWM  # noqa: E402
from napari_sprout._widget import SPROUTWidget  # noqa: E402
from napari_sprout.utils import status as S  # noqa: E402

UNCAUGHT = []
sys.excepthook = lambda *a: (UNCAUGHT.append("".join(traceback.format_exception(*a))), print(UNCAUGHT[-1]))
DIALOGS = []
for _k in ("warning", "critical", "information", "question"):
    setattr(QMessageBox, _k, staticmethod(lambda *a, _k=_k, **kw: DIALOGS.append(_k) or QMessageBox.Ok))
NOTES = []
for _m in (SWM, GWM, BWM):
    _m.show_info = lambda m, *a, **k: NOTES.append(("info", str(m)))
    _m.show_error = lambda m, *a, **k: NOTES.append(("error", str(m)))


def wait(cond, timeout=300, what=""):
    t = time.time()
    while not cond():
        QApplication.processEvents()
        time.sleep(0.003)
        if time.time() - t > timeout:
            raise TimeoutError(what)


def pump(n=30):
    for _ in range(n):
        QApplication.processEvents()
        time.sleep(0.003)


def lev(w):
    return S.level_of(w)


viewer = napari.Viewer(show=False)
viewer.add_image(dog_full, name="dog_img")
viewer.add_labels(dseed_full, name="dog_seed")
main = SPROUTWidget(viewer)
main.setStyleSheet(get_stylesheet("dark"))
main.resize(500, 1000)
main.show()
sw, gw, bw = main.seed_widget, main.grow_widget, main.boundary_widget
wait(lambda: sw.threshold_widget.hist._ready() and gw.threshold_widget.hist._ready(), what="histograms")
pump()
sw.output_folder_line.setText(str(TMP / "tab_seeds"))
gw.output_folder_line.setText(str(TMP / "tab_grow"))
has_jobs = hasattr(sw, "run_control") and hasattr(gw, "run_control") and hasattr(bw, "run_control")
ok(has_jobs, "Seeds, Grow and Boundary have a RunControl (utils/jobs.py)")


def shot(tab, name, bottom=True):
    if SHOTS is None:
        return
    if tab is bw:
        main.tabs.setCurrentWidget(main.roi_boundary_tabs)
        main.roi_boundary_tabs.setCurrentWidget(bw)
    else:
        main.tabs.setCurrentWidget(tab)
    pump(10)
    sa = tab.findChild(QScrollArea)
    if sa is not None:
        bar = sa.verticalScrollBar()
        if tab is bw:                       # the Method group (Compute row) in view
            sa.ensureWidgetVisible(bw.compute_status if bw.compute_status.isVisible() else bw.compute_btn, 0, 200)
        else:
            bar.setValue(bar.maximum() if bottom else 0)
    pump(10)
    main.grab().save(str(SHOTS / f"{name}.png"))


def progress_bar(tab):
    return getattr(tab, "progress_bar", None) or getattr(tab, "progress", None)


def run_and_stop(tab, start, run_btn, status, label, after_steps=1, shot_name=None):
    """Start, check the running state, Stop after the first real progress, check the end."""
    n_layers, n_notes = len(viewer.layers), len(NOTES)
    bar = progress_bar(tab)
    start()
    pump(2)
    ok("…" in status.text() and lev(status) in ("info", "warn") and not run_btn.isEnabled()
       and tab.stop_btn.isEnabled(), f"{label}: start -> grey {status.text()!r}, run button off, Stop on")
    ok(bar is not None and bar.isVisible() and (bar.maximum() == 0 or bar.value() >= 1),
       f"{label}: progress bar shown, busy until the first progress (max {bar.maximum() if bar else None})")
    wait(lambda: bar.maximum() > 0 and bar.value() >= after_steps, what=f"{label} progress")
    ok(bar.maximum() > 1 and 1 <= bar.value() < bar.maximum(),
       f"{label}: real progress {bar.value()} / {bar.maximum()} ({status.text()!r})")
    if shot_name:
        shot(tab, f"{shot_name}_running")
    job = tab.run_control.job
    t0 = time.perf_counter()
    tab.stop_btn.click()
    pump(1)
    ok(not tab.stop_btn.isEnabled() and not run_btn.isEnabled(),
       f"{label}: Stop -> {status.text()!r}; both buttons off until it has stopped")
    wait(lambda: run_btn.isEnabled(), what=f"{label} stop")
    lat = time.perf_counter() - t0
    wait(lambda: not job.isRunning(), timeout=30, what=f"{label} thread end")
    pump()
    text = status.text()
    ok(text.startswith("Stopped after ") and lev(status) == "warn" and not tab.stop_btn.isEnabled(),
       f"{label}: orange {text!r}; buttons back")
    ok(len(viewer.layers) == n_layers and len(NOTES) == n_notes,
       f"{label}: no layer added, no notification ({NOTES[n_notes:]})")
    info(f"{label}: Stop click -> stopped in {lat:.3f} s")
    return text


def counted_files(text):
    m = re.search(r"; (\d+) files? already written", text)
    return 0 if "no file written" in text else (int(m.group(1)) if m else None)


if has_jobs:
    # ---- Seeds: Stop
    main.tabs.setCurrentWidget(sw)
    mp = sw.main_param_widget
    mp.seed_method_combo.setCurrentText("Original")
    mp.populate_thresholds_from_list([120, 130, 140, 150, 160, 170], [255] * 6)
    mp.erosion_spin.setValue(3)
    mp.segments_spin.setValue(10)
    mp.thread_spin.setValue(2)
    text = run_and_stop(sw, sw.generate_btn.click, sw.generate_btn, sw.results_label, "Seeds",
                        shot_name="seeds")
    shot(sw, "seeds_stopped")
    n_disk = len(tifs(TMP / "tab_seeds"))
    ok(counted_files(text) == n_disk and re.match(r"Stopped after \d+ of \d+ seeds", text) is not None,
       f"Seeds: the line counts the {n_disk} seed files on disk")
    time.sleep(1.0)
    ok(len(tifs(TMP / "tab_seeds")) == n_disk, "Seeds: nothing is written after Stop (the job really stopped)")
    ok(not progress_bar(sw).isVisible(), "Seeds: progress bar hidden when idle")

    # ---- Seeds: done
    mp.populate_thresholds_from_list([140], [255])
    mp.erosion_spin.setValue(1)
    n_layers, n_notes = len(viewer.layers), len(NOTES)
    sw.generate_btn.click()
    wait(lambda: sw.generate_btn.isEnabled(), what="seeds done")
    pump()
    text = sw.results_label.text()
    ok(re.fullmatch(r"✓ Seeds generated: \d+ layers? \(\d+\.\d s\)", text) is not None and lev(sw.results_label) == "ok"
       and len(viewer.layers) > n_layers, f"Seeds: done -> green {text!r}, layers added")
    ok(NOTES[n_notes:] == [("info", text[2:])], f"Seeds: one notification {NOTES[n_notes:]}")
    shot(sw, "seeds_done")

    # ---- Seeds: failed (MemoryError)
    saved = SWM.make_seeds

    def oom(**kw):
        raise MemoryError("Unable to allocate 9.31 GiB for an array with shape (2000, 2000, 2500)")

    SWM.make_seeds = oom
    n_layers, n_notes, n_dlg = len(viewer.layers), len(NOTES), len(DIALOGS)
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            sw.generate_btn.click()
            wait(lambda: sw.generate_btn.isEnabled(), what="seeds failed")
            pump()
    finally:
        SWM.make_seeds = saved
    text = sw.results_label.text()
    ok(text.startswith("Seed generation failed: not enough memory (Unable to allocate 9.31 GiB")
       and lev(sw.results_label) == "error", f"Seeds: MemoryError -> red {text!r}")
    ok(NOTES[n_notes:] == [("error", text)] and len(DIALOGS) == n_dlg and len(viewer.layers) == n_layers,
       f"Seeds: one error notification, no dialog, no layer ({NOTES[n_notes:]})")
    shot(sw, "seeds_failed")

    # ---- Grow: Stop
    main.tabs.setCurrentWidget(gw)
    gw.image_combo.setCurrentText("dog_img")
    gw.seeds_combo.setCurrentText("dog_seed")
    gw.boundary_combo.setCurrentText("None")
    gw.main_param_widget.populate_thresholds_from_list([150, 120], [255, 255], [30, 30])
    gw.main_param_widget.thread_spin.setValue(2)
    text = run_and_stop(gw, gw.grow_btn.click, gw.grow_btn, gw.status_label, "Grow", after_steps=2,
                        shot_name="grow")
    shot(gw, "grow_stopped")
    n_disk = len(tifs(TMP / "tab_grow"))
    ok(re.match(r"Stopped after \d+ of 60 iterations", text) is not None and counted_files(text) == n_disk,
       f"Grow: the line counts iterations and the {n_disk} files on disk")
    ok(not progress_bar(gw).isVisible(), "Grow: progress bar hidden when idle")

    # ---- Grow: done
    gw.main_param_widget.populate_thresholds_from_list([150], [255], [2])
    n_layers, n_notes = len(viewer.layers), len(NOTES)
    gw.grow_btn.click()
    wait(lambda: gw.grow_btn.isEnabled(), what="grow done")
    pump()
    text = gw.status_label.text()
    ok(re.fullmatch(r"✓ Growth completed: \d+ layers? \(\d+\.\d s\)", text) is not None and lev(gw.status_label) == "ok"
       and len(viewer.layers) > n_layers, f"Grow: done -> green {text!r}")
    ok(NOTES[n_notes:] == [("info", text[2:])], f"Grow: one notification {NOTES[n_notes:]}")
    shot(gw, "grow_done")

    # ---- Grow: failed
    saved = GWM.grow_mp
    GWM.grow_mp = lambda **kw: (_ for _ in ()).throw(MemoryError())
    n_notes, n_dlg = len(NOTES), len(DIALOGS)
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            gw.grow_btn.click()
            wait(lambda: gw.grow_btn.isEnabled(), what="grow failed")
            pump()
    finally:
        GWM.grow_mp = saved
    text = gw.status_label.text()
    ok(text == "Growth failed: not enough memory; try an ROI or a smaller image" and lev(gw.status_label) == "error"
       and NOTES[n_notes:] == [("error", text)] and len(DIALOGS) == n_dlg,
       f"Grow: MemoryError -> red {text!r}, one notification")
    shot(gw, "grow_failed")

    # ---- Boundary
    main.tabs.setCurrentWidget(main.roi_boundary_tabs)
    main.roi_boundary_tabs.setCurrentWidget(bw)
    bw.image_combo.setCurrentText("dog_img")
    wait(lambda: bw.input_hist._ready(), what="boundary hist")
    bw.device_combo.setCurrentIndex(bw.device_combo.findData("cpu"))
    bw.panel.set_params({"kind": "black", "shape": "ball", "size": 12})
    text = run_and_stop(bw, bw.compute_btn.click, bw.compute_btn, bw.compute_status, "Boundary",
                        shot_name="boundary")
    shot(bw, "boundary_stopped")
    ok(re.match(r"Stopped after (\d+ of \d+|all \d+) chunks.*; nothing kept", text) is not None and bw._resp is None
       and not bw._computing and bw.compute_btn.isEnabled(), f"Boundary: {text!r}, no response kept")

    bw.panel.set_params({"kind": "black", "shape": "ball", "size": 2})
    n_notes = len(NOTES)
    bw.compute_btn.click()
    wait(lambda: not bw._computing, what="boundary compute")
    pump()
    text = bw.compute_status.text()
    ok(re.fullmatch(r"✓ Boundary response: black top-hat, ball, size 2 on CPU \(\d+\.\d s\)", text) is not None
       and lev(bw.compute_status) == "ok" and bw._resp is not None, f"Boundary: done -> green {text!r}")
    ok(NOTES[n_notes:] == [("info", text[2:])], f"Boundary: one notification {NOTES[n_notes:]}")
    ok(bw.progress.value() == bw.progress.maximum(), "Boundary: progress bar full after Compute")
    bw.resp_hist.set_range(25, bw.resp_hist._xmax)
    n_notes = len(NOTES)
    bw._create()
    wait(lambda: "dog_img_boundary" in viewer.layers, what="boundary create")
    pump()
    ok(bw.output_status.text().startswith("✓ Created 'dog_img_boundary':") and NOTES[n_notes:] == [],
       f"Boundary Create: green line, no notification ({NOTES[n_notes:]})")
    shot(bw, "boundary_done")

    saved = BWM.TophatPanel.compute
    BWM.TophatPanel.compute = staticmethod(lambda *a: (_ for _ in ()).throw(MemoryError("Unable to allocate 1 GiB")))
    bw.panel.set_params({"size": 3})
    n_notes, n_dlg = len(NOTES), len(DIALOGS)
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            bw.compute_btn.click()
            wait(lambda: not bw._computing, what="boundary failed")
            pump()
    finally:
        BWM.TophatPanel.compute = saved
    text = bw.compute_status.text()
    ok(text.startswith("Boundary compute failed: not enough memory (Unable to allocate 1 GiB)")
       and lev(bw.compute_status) == "error" and NOTES[n_notes:] == [("error", text)] and len(DIALOGS) == n_dlg,
       f"Boundary: MemoryError -> red {text!r}, one notification ({NOTES[n_notes:]})")
    shot(bw, "boundary_failed")
else:
    ok(False, "tabs: no cooperative Stop / real progress (the run tabs have no RunControl; Stop is "
              "QThread.terminate(), not run here)")

ok(not DIALOGS, f"no dialog ({DIALOGS})")
ok(not UNCAUGHT, f"no uncaught errors ({len(UNCAUGHT)})" + (f": {UNCAUGHT[0][-400:]}" if UNCAUGHT else ""))

# ======================================================================
print("\n== texts (C7)")
files = ["napari-sprout/napari_sprout/widgets/seed_widget.py", "napari-sprout/napari_sprout/widgets/grow_widget.py",
         "napari-sprout/napari_sprout/widgets/boundary_widget.py", "napari-sprout/napari_sprout/widgets/prompt_tab.py",
         "napari-sprout/napari_sprout/utils/histogram_threshold_widget.py",
         "napari-sprout/napari_sprout/utils/util_widget.py"]
bad = []
for f in files:
    for i, line in enumerate((ROOT / f).read_text(encoding="utf-8").splitlines(), 1):
        code = line.split("#")[0]
        if re.search(r"""(\.set|show_message|setText|_set_run_status|show_info|show_error)\(.*\.\.\.["')]""", code) \
                or "Please select" in code:
            bad.append(f"{Path(f).name}:{i}: {line.strip()[:90]}")
ok(not bad, "status texts use '…' and 'Select …'" + "".join(f"\n       {b}" for b in bad))

wait(lambda: not (bw._workers or sw.threshold_widget._workers or gw.threshold_widget._workers), timeout=60)
viewer.close()
print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
sys.exit(1 if FAILED else 0)
