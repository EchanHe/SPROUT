"""
Headless end-to-end check of the Threshold Preview in the Seeds and Grow tabs.

Drives a real napari viewer (offscreen) and the real SPROUT main widget.
Run from anywhere, in the `sprout` env:
    python tests/check_threshold_preview.py
Prints PASS/FAIL per check; exits non-zero if anything failed.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "napari_sprout")]

import napari  # noqa: E402
import _headless_gl  # noqa: E402,F401  (napari >= 0.7 offscreen: no GL context)
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from napari_sprout._widget import SPROUTWidget  # noqa: E402
from napari_sprout.utils import histogram_threshold_widget as HTW  # noqa: E402
from sprout_core.sprout_core import erosion_binary_img_on_sub, get_ccomps_with_size_order  # noqa: E402

FAILED = []


def wait(cond, timeout=120, what=""):
    t = time.time()
    while not cond():
        QApplication.processEvents()
        time.sleep(0.005)
        if time.time() - t > timeout:
            raise TimeoutError(what)


def pump(n=20):
    for _ in range(n):
        QApplication.processEvents()
        time.sleep(0.005)


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILED.append(msg)


HIST_CALLS = []
_orig_histogram = HTW.B.histogram


def _counting_histogram(*a, **k):
    import inspect
    if inspect.stack()[1].function == "_cached_histogram":   # Threshold Preview only, not Boundary
        HIST_CALLS.append(1)
    return _orig_histogram(*a, **k)


HTW.B.histogram = _counting_histogram

viewer = napari.Viewer(show=False)
dog = tifffile.imread(ROOT / "demo_data" / "dog_img.tif")

# image loaded BEFORE the panel opens: the panels must pick it up without "Refresh Layers"
viewer.add_image(dog, name="dog")
pump()
main = SPROUTWidget(viewer)
viewer.window.add_dock_widget(main)
sw, gw = main.seed_widget, main.grow_widget
st, gt = sw.threshold_widget, gw.threshold_widget
PV = st._preview.name      # the Seeds live layer ("Threshold Preview (Seeds)"; "Threshold Preview" before)
ok([sw.image_combo.itemText(i) for i in range(sw.image_combo.count())] == ["dog"]
   and [gw.image_combo.itemText(i) for i in range(gw.image_combo.count())] == ["dog"],
   "image loaded before opening the panel shows up in Seeds and Grow without Refresh")

ok(st.create_btn.text() == gt.create_btn.text() == "Create Preview Layer",
   "Seeds and Grow have the same button: 'Create Preview Layer'")
ok(st.live_cb.isChecked() and gt.live_cb.isChecked(), "'Live preview on current slice' on by default in both")
ok(st.live_cb.text() == "Live preview on current slice (threshold only)",
   f"checkbox says the live preview is threshold only: '{st.live_cb.text()}'")

wait(lambda: st.hist._ready() and gt.hist._ready(), what="hists")
ok(len(HIST_CALLS) == 1, f"Seeds and Grow share one histogram of the same image ({len(HIST_CALLS)} computed)")
lo, hi = st.get_thresholds()
ok(st.hist.upper_note.text() == "(no limit)" and hi == 255, f"default: lower = Otsu {lo:.0f}, upper no limit -> 255")

# live slice preview
st.hist.set_range(100, st.hist._xmax)
pump()
lay = viewer.layers[PV]
z = int(round(viewer.dims.point[0]))
ok(np.array_equal(lay.data[z].astype(bool), dog[z] >= 100) and lay.data.sum() == (dog[z] >= 100).sum(),
   f"live: only slice z={z} previewed, correct")
viewer.dims.set_point(0, z + 40)
pump()
ok(lay.data[z].sum() == 0 and np.array_equal(lay.data[z + 40].astype(bool), dog[z + 40] >= 100),
   "live: follows the slice")

# checkbox off / on
st.live_cb.setChecked(False)
pump()
ok(PV not in viewer.layers and st.status.text() == "Live preview off",
   "unticking removes the live preview layer")
st.hist.set_range(110, st.hist._xmax)
viewer.dims.set_point(0, z + 10)
pump()
ok(PV not in viewer.layers, "with live off, dragging / moving slices adds nothing")
st.live_cb.setChecked(True)
pump()
zz = int(round(viewer.dims.point[0]))
ok(PV in viewer.layers
   and viewer.layers[PV].data.sum() == (dog[zz] >= 110).sum(),
   "re-ticking shows the current range again")
st.hist.set_range(100, st.hist._xmax)
pump()

# Seeds: create preview layer with erosion + CC
st.use_connected_components_checkbox.setChecked(True)
st.connected_components_spin.setValue(10)
st.erosion_spin.setValue(1)
pump()
lines = st.status.text().split("\n")
ok(len(lines) == 2 and lines[0].startswith("Current slice: ")
   and lines[1] == "Erosion / connected components: whole volume only, see Create Preview Layer",
   f"status: one line per topic, erosion/CC note on its own line: {lines}")
wait(lambda: st.create_btn.isEnabled(), what="seeds button ready")
st.create_btn.click()
wait(lambda: "dog_preview_100-255_ero1_cc10" in viewer.layers, what="seeds create")
pump()
lines = st.status.text().split("\n")
ok(lines[-2] == "Created: dog_preview_100-255_ero1_cc10" and lines[-1].endswith("10 components")
   and lines[0].startswith("Current slice: "),
   f"status after Create: current slice, note, then 'Created: <name>' and the counts: {lines}")
exp = (dog >= 100) & (dog <= 255)
exp = erosion_binary_img_on_sub(exp)
exp, _ = get_ccomps_with_size_order(exp, 10)
ok(np.array_equal(viewer.layers["dog_preview_100-255_ero1_cc10"].data, exp.astype(np.uint16)),
   "Seeds layer == threshold + erosion + CC")
st.erosion_spin.setValue(3)
wait(lambda: st.create_btn.isEnabled(), what="seeds button ready")
st.create_btn.click()
wait(lambda: "dog_preview_100-255_ero3_cc10" in viewer.layers, what="seeds create 2")
ok(True, "different settings -> a separate layer to compare (..._ero3_cc10)")
wait(lambda: st.create_btn.isEnabled(), what="seeds button ready")
st.create_btn.click()
wait(lambda: "dog_preview_100-255_ero3_cc10_1" in viewer.layers, what="dup")
ok(True, "same settings again -> '_1', never overwrites")
ok(viewer.layers["dog_preview_100-255_ero1_cc10"].metadata["sprout_threshold_preview"]["erosion_steps"] == 1,
   "parameters stored in layer metadata")

# Seeds: add range to table (replaces the automatic full-range default row)
st.hist.set_range(120, st.hist._xmax)
st.add_btn.click()
st.hist.set_range(140, 200)
st.add_btn.click()
pump()
ok(sw.main_param_widget._get_threshold_params() == ([120, 140], [255, 200]),
   f"Seeds 'Add range to table': {sw.main_param_widget._get_threshold_params()}")

# Grow: same button, threshold only
gt.hist.set_range(100, 200)
wait(lambda: gt.create_btn.isEnabled(), what="grow button ready")
gt.create_btn.click()
wait(lambda: "dog_preview_100-200" in viewer.layers, what="grow create")
ok(np.array_equal(viewer.layers["dog_preview_100-200"].data, ((dog >= 100) & (dog <= 200)).astype(np.uint8)),
   "Grow layer == (img>=100)&(img<=200)")
gt.add_btn.click()
gt.add_btn.click()
pump()
ok(gw.main_param_widget._get_threshold_params() == ([100, 100], [200, 200], [5, 5]),
   f"Grow 'Add range to table': {gw.main_param_widget._get_threshold_params()}")

# set_thresholds before the histogram is ready (YAML import)
d2 = tifffile.imread(ROOT / "demo_data" / "simple_demo.tif")
viewer.add_image(d2, name="simple2d")
pump()
sw.image_combo.setCurrentText("simple2d")
st.set_thresholds(80, 150)
wait(lambda: st.hist._ready() and st._layer is not None and st._layer.name == "simple2d", what="hist 2d")
ok(st.get_thresholds() == (80.0, 150.0), f"set_thresholds before histogram ready applied later: {st.get_thresholds()}")
st.hist.set_range(90, 150)
pump()
ok(np.array_equal(viewer.layers[PV].data.astype(bool), (d2 >= 90) & (d2 <= 150)),
   "2D image live preview")

# float image handed to the widget directly (the parameter tables still crash on float: roadmap R1)
f = dog.astype(np.float32) / 255
viewer.add_image(f, name="float_img")
pump()
pump()
st.set_image(viewer.layers["float_img"])
wait(lambda: st.hist._ready() and st._layer.name == "float_img", what="float hist")
st.hist.set_range(0.4, st.hist._xmax)
pump()
lo, hi = st.get_thresholds()
ok(np.isinf(hi) and abs(lo - 0.4) < 1e-6, f"float image: thresholds ({lo}, {hi})")

# ---- combo selections survive layer changes; boundary really reaches Original seeds
import tempfile  # noqa: E402

block = np.zeros((20, 30, 60), np.uint8)
block[4:16, 5:25, 5:55] = 200                    # one solid block
cut = np.zeros(block.shape, np.uint8)
cut[:, :, 30] = 1                                # a full plane cutting it in half
viewer.add_image(block, name="block")
viewer.add_labels(cut, name="cut_plane")
pump()
sw.image_combo.setCurrentText("block")
sw.output_folder_line.setText(tempfile.mkdtemp())
sw.boundary_combo.setCurrentText("cut_plane")
wait(lambda: st.hist._ready() and st._layer.name == "block", what="block hist")
st.hist.set_range(100, st.hist._xmax)
pump()
ok(sw.boundary_combo.currentText() == "cut_plane", "Seeds: boundary still selected after dragging the threshold")
viewer.add_labels(np.zeros(block.shape, np.uint8), name="unrelated")
pump()
ok(sw.boundary_combo.currentText() == "cut_plane", "Seeds: boundary still selected after adding another layer")
gw.seeds_combo.setCurrentText("cut_plane")
gw.boundary_combo.setCurrentText("unrelated")
viewer.add_labels(np.zeros(block.shape, np.uint8), name="unrelated2")
pump()
ok(gw.seeds_combo.currentText() == "cut_plane" and gw.boundary_combo.currentText() == "unrelated",
   "Grow: seeds and boundary selections survive adding a layer")


def run_original(boundary_name):
    sw.boundary_combo.setCurrentText(boundary_name)
    sw.main_param_widget.clean_table()
    st.add_btn.click()                           # table row = current range [100, max]
    pump()
    before = {l.name for l in viewer.layers}
    sw.generate_btn.click()
    wait(lambda: {l.name for l in viewer.layers} - before - {PV}, what="generate")
    pump()
    new = [l for l in viewer.layers if l.name not in before and l.name != PV]
    return len(np.unique(new[0].data)) - 1


sw.main_param_widget.erosion_spin.setValue(0)
sw.main_param_widget.segments_spin.setValue(10)
n_without, n_with = run_original("None"), run_original("cut_plane")
ok(n_without == 1 and n_with == 2, f"Original seeds: without boundary {n_without} region(s), with boundary {n_with}")

# switching back to an image already seen reuses its histogram (cache)
n_calls = len(HIST_CALLS)
sw.image_combo.setCurrentText("dog")
wait(lambda: st.hist._ready() and st._layer.name == "dog", what="back to dog")
ok(len(HIST_CALLS) == n_calls, f"switching back to 'dog' recomputes nothing ({len(HIST_CALLS) - n_calls} computed)")

wait(lambda: not st._workers and not gt._workers and not main.boundary_widget._workers, timeout=60)
viewer.close()
print("viewer closed cleanly")
print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
sys.exit(1 if FAILED else 0)
