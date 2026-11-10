"""
Checks for the core fixes of the napari-sprout audit (A1, B11, D1, resolve_device).

    A1  an exception inside a Seeds / adaptive-seed worker thread reaches the caller:
        make_seeds, make_adaptive_seed_ero and make_adaptive_seed_thre raise it (with the
        traceback of the failing step), and the Seeds tab shows the red status line, no
        "✓" and no "successfully" notification
    B11 the seeds that make_seeds hands to napari have the label dtype (cast_labels:
        uint8 for up to 255 ids), not the int32 of measure.label; values equal the file
    D1  importing napari_sprout / the SPROUT panel does not import matplotlib; the plot
        functions still write their png from a worker thread, without pyplot
    resolve_device(None) without torch installed -> "cpu"

Run in the `sprout` env:
    python tests/check_core_fixes.py
Prints PASS/FAIL per check; exits non-zero if anything failed.
"""
import contextlib
import glob
import io
import json
import os
import subprocess
import sys
sys.stdout.reconfigure(encoding="utf-8")   # status texts contain ✓
import tempfile
import time
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "napari-sprout"), str(Path(__file__).resolve().parent)]

import numpy as np  # noqa: E402
import tifffile  # noqa: E402

FAILED = []


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILED.append(msg)


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


dog = tifffile.imread(ROOT / "demo_data" / "dog_img.tif")[200:420]

# ======================================================================
print("== D1 matplotlib is not imported with the panel (fresh interpreter)")
probe = ("import sys; sys.path[:0] = [r'%s', r'%s']\n"
         "import napari_sprout\n"
         "from napari_sprout._widget import SPROUTWidget\n"
         "from sprout_core import make_seeds, make_grow, make_adaptive_seed\n"
         "print(sorted(m for m in sys.modules if m == 'matplotlib' or m.startswith('matplotlib.')))\n"
         % (ROOT, ROOT / "napari-sprout"))
r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=str(Path.home()),
                   env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
mods = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else f"(no output) {r.stderr[-500:]}"
ok(r.returncode == 0 and mods == "[]", f"import napari_sprout + SPROUTWidget + make_*: matplotlib modules = {mods}")

# the plot functions from a worker thread while a Qt app runs (as in napari), in a fresh
# interpreter: png files written, pyplot never imported (pyplot's Qt backend in a worker
# thread prints QBasicTimer warnings and can crash the process later)
tmp = Path(tempfile.mkdtemp(prefix="sprout_core_fixes_"))
plot_probe = r"""
import json, sys, threading, traceback
sys.path[:0] = [%r]
from qtpy.QtWidgets import QApplication
app = QApplication([])
import pandas as pd
from sprout_core import vis_lib
tmp = %r
seeds_log = [{"threshold": t, "Whole Volume": 1000 + t, "footprints": ["ball"],
              "seeds": [{"ero_iter": e, "size_ccomp": [500 - 50 * e, 200, 100 - e], "number_of_ccomps": 3}
                        for e in range(3)]} for t in (140, 160)]
grow_log = pd.DataFrame({"id": [1, 2, 3, 4], "grow_size": [10, 20, 30, 35], "full_size": [40, 40, 50, 50],
                         "cur_threshold": ["150_255", "150_255", "120_255", "120_255"],
                         "cur_dilate_step": [1, 2, 1, 2]})
errs = []
def plots():
    try:
        vis_lib.plot_seeds_log_json(seeds_log, tmp + "/seeds.png")
        vis_lib.plot_grow(grow_log, tmp + "/grow.png")
        vis_lib.merge_plots([tmp + "/seeds.png", tmp + "/seeds.png"], tmp + "/merged.png")
    except Exception:
        errs.append(traceback.format_exc())
th = threading.Thread(target=plots)
th.start()
th.join()
for _ in range(20):
    app.processEvents()
print(json.dumps({"errs": errs, "pyplot": "matplotlib.pyplot" in sys.modules}))
""" % (str(ROOT), str(tmp))
r = subprocess.run([sys.executable, "-c", plot_probe], capture_output=True, text=True, cwd=str(Path.home()),
                   env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
try:
    res = json.loads(r.stdout.strip().splitlines()[-1])
except Exception:  # noqa: BLE001
    res = {"errs": [f"exit code {r.returncode}: {r.stdout[-300:]} {r.stderr[-300:]}"], "pyplot": None}
warn = sorted({ln.strip() for ln in r.stderr.splitlines() if "QBasicTimer" in ln or "startTimer" in ln})
ok(r.returncode == 0 and not res["errs"]
   and all((tmp / f).is_file() for f in ("seeds.png", "grow.png", "merged.png")),
   f"plots written from a worker thread (exit {r.returncode}) {res['errs'][-1][-300:] if res['errs'] else ''}")
ok(res["pyplot"] is False and not warn, f"pyplot imported: {res['pyplot']}, Qt timer warnings: {warn}")

from sprout_core import make_adaptive_seed  # noqa: E402
from sprout_core import make_seeds as ms  # noqa: E402
import sprout_core.sprout_core as sc  # noqa: E402

# ======================================================================
print("\n== resolve_device(None) without torch")
from sprout_core import nni_engine  # noqa: E402
saved_torch = sys.modules.get("torch", "absent")
sys.modules["torch"] = None                     # `import torch` -> ImportError
try:
    try:
        dev = nni_engine.resolve_device(None)
    except ImportError as e:
        dev = f"ImportError: {e}"
finally:
    if saved_torch == "absent":
        del sys.modules["torch"]
    else:
        sys.modules["torch"] = saved_torch
ok(dev == "cpu", f"resolve_device(None) with no torch -> {dev!r}")
ok(nni_engine.resolve_device("cuda:1") == "cuda:1", "resolve_device('cuda:1') unchanged")

# ======================================================================
print("\n== B11 seeds handed to napari: label dtype, values as on disk")
out = tmp / "b11"
seeds, _ = quiet(ms.make_seeds, img=dog, thresholds=[140, 160], upper_thresholds=[255, 255], erosion_steps=2,
                 segments=6, num_threads=2, footprints=["ball", "ball"], output_folder=str(out),
                 base_name="dog", return_for_napari=True)
files = {os.path.basename(p): p for p in glob.glob(str(out / "**" / "*.tif"), recursive=True)}
dts = sorted({str(a.dtype) for a in seeds.values()})
same = all(np.array_equal(a, tifffile.imread(files[k.split("_", 2)[-1]])) for k, a in seeds.items())
ok(len(seeds) == 6 and all(a.dtype == sc.label_dtype(int(a.max()), np.uint16) for a in seeds.values()) and same,
   f"{len(seeds)} seeds, dtypes {dts} (expected {sc.label_dtype(6, np.uint16)}), values == files: {same}")

# ======================================================================
print("\n== A1 an exception in a worker thread reaches the caller")


class Injected(RuntimeError):
    pass


def boom(*a, **k):
    raise Injected("injected failure")


def raises(fn, **kw):
    try:
        quiet(fn, **kw)
    except Injected as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        return "boom" in tb, tb
    except Exception as e:  # noqa: BLE001
        return False, f"other exception {type(e).__name__}: {e}"
    return False, "returned normally"


saved = sc.find_seed_by_ero_custom
sc.find_seed_by_ero_custom = boom
try:
    r, why = raises(ms.make_seeds, img=dog, thresholds=[140, 160], upper_thresholds=[255, 255], erosion_steps=1,
                    segments=6, num_threads=2, output_folder=str(tmp / "a1_seeds"), return_for_napari=True)
finally:
    sc.find_seed_by_ero_custom = saved
ok(r, f"make_seeds raises the thread's exception with its traceback ({why.splitlines()[-1]})")

saved = make_adaptive_seed.detect_inter
make_adaptive_seed.detect_inter = boom
try:
    r, why = raises(make_adaptive_seed.make_adaptive_seed_ero, img=dog, threshold=140, upper_threshold=255,
                    erosion_steps=3, segments=10, num_threads=2, output_folder=str(tmp / "a1_ero"),
                    footprints="ball", return_for_napari=True)
    ok(r, f"make_adaptive_seed_ero raises the exception ({why.splitlines()[-1]})")
    r, why = raises(make_adaptive_seed.make_adaptive_seed_thre, img=dog, thresholds=[130, 150, 170],
                    upper_thresholds=[255, 255, 255], erosion_steps=1, segments=10, num_threads=2,
                    output_folder=str(tmp / "a1_thre"), footprints="ball", return_for_napari=True)
    ok(r, f"make_adaptive_seed_thre raises the exception ({why.splitlines()[-1]})")
finally:
    make_adaptive_seed.detect_inter = saved

# ---- the Seeds tab
print("\n== A1 Seeds tab: a failed run is red, no ✓, no success notification")
with contextlib.suppress(ImportError):
    import _headless_gl  # noqa: F401,E402  (offscreen GL shim for napari 0.7, if present)
import napari  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

import napari_sprout.widgets.seed_widget as swm  # noqa: E402
from napari_sprout._widget import SPROUTWidget  # noqa: E402


def wait(cond, timeout=300, what=""):
    t = time.time()
    while not cond():
        QApplication.processEvents()
        time.sleep(0.005)
        if time.time() - t > timeout:
            raise TimeoutError(what)


def pump(n=30):
    for _ in range(n):
        QApplication.processEvents()
        time.sleep(0.005)


viewer = napari.Viewer(show=False)
viewer.add_image(dog, name="dog_img")
main = SPROUTWidget(viewer)
sw = main.seed_widget
wait(lambda: sw.threshold_widget.hist._ready(), what="histogram")
pump()
notes = []
saved_info, saved_err = swm.show_info, swm.show_error
swm.show_info = lambda m, *a, **k: notes.append(("info", str(m)))
swm.show_error = lambda m, *a, **k: notes.append(("error", str(m)))
sw.output_folder_line.setText(str(tmp / "a1_tab"))
mp = sw.main_param_widget
mp.seed_method_combo.setCurrentText("Original")
mp.populate_thresholds_from_list([140, 160], [255, 255])
mp.erosion_spin.setValue(1)
mp.segments_spin.setValue(6)
n_layers = len(viewer.layers)
sc.find_seed_by_ero_custom = boom
try:
    with contextlib.redirect_stdout(io.StringIO()):
        sw.generate_seeds()
        wait(lambda: sw.generate_btn.isEnabled() and sw.worker is not None and not sw.worker.isRunning(),
             what="seeds")
        pump()
finally:
    sc.find_seed_by_ero_custom = saved
    swm.show_info, swm.show_error = saved_info, saved_err
text, level = sw.results_label.text(), getattr(sw.results_label, "level", None)
ok(level == "error" and "✓" not in text and "injected failure" in text,
   f"Seeds status after a failed run: {text!r} ({level})")
ok(not any("success" in m.lower() for _, m in notes) and any(k == "error" for k, _ in notes),
   f"notifications: {notes}")
ok(len(viewer.layers) == n_layers, f"no layer added ({len(viewer.layers) - n_layers} new)")
viewer.close()

print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
sys.exit(1 if FAILED else 0)
