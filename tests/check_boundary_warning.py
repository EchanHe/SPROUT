"""
Headless check: Generate Seeds / Start Growth refuse a non-binary boundary with a red
line under the Boundary combo (no dialog: any QMessageBox call is recorded as a failure).

Run in the `sprout` env:
    python tests/check_boundary_warning.py
"""
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "napari_sprout")]

import napari  # noqa: E402
import _headless_gl  # noqa: E402,F401  (napari >= 0.7 offscreen: no GL context)
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from qtpy.QtWidgets import QApplication, QMessageBox  # noqa: E402

from napari_sprout._widget import SPROUTWidget  # noqa: E402
from napari_sprout.utils.boundary_check import boundary_problem  # noqa: E402
from local_paths import local_path, not_set  # noqa: E402

FAILED = []


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILED.append(msg)


def wait(cond, timeout=120, what=""):
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


# ---- the rule itself (mirrors config_core.check_and_cast_boundary)
z = np.zeros((4, 4, 4), np.uint8)
idx = np.arange(64).reshape(4, 4, 4)
cases = {
    "bool": (idx % 2 == 0, None),
    "0/1": (np.where(idx % 2, 1, 0).astype(np.uint8), None),
    "0/255": (np.where(idx % 2, 255, 0).astype(np.uint8), None),
    "0/1/2": ((idx % 3).astype(np.uint8), "more than two values (0, 1, 2)"),
    "all 0": (z, "empty"),
    "all 1": (z + 1, "no 0 voxels"),
}
for name, (arr, expect) in cases.items():
    got = boundary_problem(arr)
    ok((got is None) if expect is None else (got is not None and expect in got), f"rule: {name} -> {got}")

# ---- GUI: record dialogs instead of blocking on them (none may open)
dialogs = []
QMessageBox.warning = staticmethod(lambda parent, title, text, *a, **k: dialogs.append(text) or QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda parent, title, text, *a, **k: dialogs.append(text) or QMessageBox.Ok)


def red(w):
    """The red line under the widget's Boundary combo ('' while hidden)."""
    e = getattr(w, "boundary_error", None)
    return e.text() if e is not None and not e.isHidden() and e.level == "error" else ""

viewer = napari.Viewer(show=False)
main = SPROUTWidget(viewer)
viewer.window.add_dock_widget(main)
sw, gw = main.seed_widget, main.grow_widget

block = np.zeros((20, 30, 60), np.uint8)
block[4:16, 5:25, 5:55] = 200
three = np.zeros(block.shape, np.uint8)
three[:, :, 29] = 1
three[:, :, 30] = 2                                   # 0/1/2 like the skull boundaries
viewer.add_image(block, name="block")
viewer.add_labels(three, name="three_values")
viewer.add_labels((three > 0).astype(np.uint8), name="good")
viewer.add_labels(np.zeros(block.shape, np.uint8), name="empty")
pump()

sw.image_combo.setCurrentText("block")
st = sw.threshold_widget
PV = st._preview.name      # the Seeds live layer ("Threshold Preview (Seeds)"; "Threshold Preview" before)
wait(lambda: st.hist._ready() and st._layer is not None and st._layer.name == "block")
sw.output_folder_line.setText(tempfile.mkdtemp())
sw.main_param_widget.erosion_spin.setValue(0)
sw.main_param_widget.segments_spin.setValue(10)
st.hist.set_range(100, st.hist._xmax)
sw.main_param_widget.clean_table()
st.add_btn.click()
pump()

# selecting alone pops nothing
sw.boundary_combo.setCurrentText("three_values")
pump()
ok(not dialogs and not red(sw), "selecting a non-binary boundary shows nothing yet")


def generate():
    before = {l.name for l in viewer.layers}
    sw.generate_btn.click()
    pump(60)
    return [l for l in viewer.layers if l.name not in before and l.name != PV]


new = generate()
ok(not dialogs and "more than two values (0, 1, 2)" in red(sw), f"Generate Seeds -> red line: {red(sw)!r}")
ok(not new and sw.generate_btn.isEnabled(), "seeds not generated, Generate button still usable")

sw.boundary_combo.setCurrentText("empty")
generate()
ok(not dialogs and "empty" in red(sw), "empty boundary -> red line")

sw.boundary_combo.setCurrentText("good")
before = {l.name for l in viewer.layers}
sw.generate_btn.click()
wait(lambda: {l.name for l in viewer.layers} - before - {PV}, what="generate")
pump()
new = [l for l in viewer.layers if l.name not in before and l.name != PV]
ok(not dialogs and not red(sw) and len(np.unique(new[0].data)) - 1 == 2,
   "binary boundary: no red line, seeds generated (2 regions)")
seed_name = new[0].name

# Grow
gw.image_combo.setCurrentText("block")
gw.seeds_combo.setCurrentText(seed_name)
gw.boundary_combo.setCurrentText("three_values")
pump()
before = {l.name for l in viewer.layers}
gw.grow_btn.click()
pump(60)
ok(not dialogs and "three_values" in red(gw), "Start Growth -> red line")
ok({l.name for l in viewer.layers} == before and gw.grow_btn.isEnabled(), "growth not started, button still usable")

# ---- speed of the click-time check on a real 0/1/2 skull boundary (~935M voxels)
skull = local_path("skull_dir")
path = skull / "boundary" / "Orycteropus_afer_boundary.tif" if skull else None
if path is None:
    print(f"SKIP real skull boundary: {not_set('skull_dir')}")
elif path.exists():
    b = tifffile.imread(path)
    t = time.perf_counter()
    problem = boundary_problem(b)
    ok(problem is not None, f"real skull boundary (0/1/2, 935M voxels): check takes {time.perf_counter() - t:.1f}s -> {problem}")

viewer.close()
print("viewer closed cleanly")
print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
sys.exit(1 if FAILED else 0)
