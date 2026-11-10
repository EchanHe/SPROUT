"""
Headless check of the Data Info panel (napari_sprout/widgets/info_widget.py) for
ROI layers and plain Shapes layers; Image / Labels output unchanged.

Run in the `sprout` env:
    python tests/check_info_widget.py
Prints PASS/FAIL per check; exits non-zero if anything failed.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "napari-sprout")]

import napari  # noqa: E402
import _headless_gl  # noqa: E402,F401  (napari >= 0.7 offscreen: no GL context)
import numpy as np  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from napari_sprout._widget import SPROUTWidget  # noqa: E402
from napari_sprout.utils import roi_layers as RL  # noqa: E402
from napari_sprout.widgets.info_widget import LayerInfoWidget, roi_layer_summary, shapes_summary  # noqa: E402

FAILED = []


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILED.append(msg)


def wait(cond, timeout=60, what=""):
    t = time.time()
    while not cond():
        QApplication.processEvents()
        time.sleep(0.005)
        if time.time() - t > timeout:
            raise TimeoutError(what)


def pump(n=40):
    for _ in range(n):
        QApplication.processEvents()
        time.sleep(0.005)


v = napari.Viewer(show=False)
img = (np.random.default_rng(0).random((40, 60, 80)) * 255).astype(np.uint8)
il = v.add_image(img, name="img", scale=(2.0, 1.0, 1.0))
lab = np.zeros(img.shape, np.uint16)
lab[5:10, 5:20, 5:20] = 3
lab[20:30, 30:50, 40:70] = 7
ll = v.add_labels(lab, name="cells")
main = SPROUTWidget(v)
v.window.add_dock_widget(main)
iw = LayerInfoWidget(v)
pump(100)


def show(layer):
    v.layers.selection.active = layer
    iw.on_active_layer_changed()
    # several workers may be running for the same layer (napari's selection event + this call)
    wait(lambda: iw.info_box.toPlainText().startswith(f"Name: {layer.name}"), what=layer.name)
    pump(20)
    return iw.info_box.toPlainText()


# ROI layer with a set z range, picked by Seeds and Prompt
r1 = RL.add_roi_layer(v, il, (4, 10, 20), (30, 50, 75))
main.seed_widget.roi_select.select(r1)
main.prompt_tab.roi_select.select(r1)
pump()
txt = show(r1)
lines = txt.splitlines()
exp = ["Name: ROI_1", "Layer type: ROI (Shapes)", "Range z 4–29 · y 10–49 · x 20–74",
       "Size 26 × 40 × 55 voxels (z × y × x)", "Voxels 57,200 (30% of the reference shape)",
       "Reference image: img (40 × 60 × 80)", "Used by: Seeds, Prompt"]
ok(lines == exp, f"ROI layer: {lines}")

# a new ROI whose z range was not chosen: 'z: all slices (not set)'; no tab picks it
r2 = RL.add_roi_layer(v, il, (0, 0, 0), (40, 20, 20), z_all=True)
lines = show(r2).splitlines()
ok(lines[-1] == "z: all slices (not set)" and "Used by: no tab" in lines and "Range z 0–39 · y 0–19 · x 0–19" in lines,
   f"ROI with z_all: {lines}")

# an empty ROI layer (nothing drawn yet)
r3 = RL.add_roi_layer(v, il)
lines = show(r3).splitlines()
ok(lines[:3] == ["Name: ROI_3", "Layer type: ROI (Shapes)", "ROI_3: no box drawn yet"], f"empty ROI layer: {lines}")

# 2D ROI
v2 = v.add_image(img[0], name="slice")
r4 = RL.add_roi_layer(v, v2, (5, 6), (15, 26))
lines = roi_layer_summary(r4, v)
ok("Range y 5–14 · x 6–25" in lines and "Size 10 × 20 voxels (y × x)" in lines
   and "Reference image: slice (60 × 80)" in lines, f"2D ROI: {lines}")

# plain Shapes layer
sh = v.add_shapes([np.array([[0, 0], [0, 10], [12, 10], [12, 0]], float),
                   np.array([[20, 5], [30, 40]], float)], shape_type=["rectangle", "line"], name="drawings")
lines = show(sh).splitlines()
ok(lines == ["Name: drawings", "Layer type: Shapes", "Shapes: 2", "Bounding box y 0.0–30.0 · x 0.0–40.0 (data coordinates)"],
   f"plain Shapes: {lines}")
empty = v.add_shapes(name="nothing")
ok(show(empty).splitlines() == ["Name: nothing", "Layer type: Shapes", "Shapes: 0"], "empty Shapes layer")
ok(shapes_summary(sh)[2] == "Shapes: 2" and "Unsupported" not in show(sh), "no 'Unsupported layer type' for Shapes")

# Image / Labels unchanged
txt = show(ll)
ok(txt.splitlines()[:4] == ["Name: cells", "Layer type: Labels", f"Shape: {lab.shape}", "Data type: uint16"]
   and "Label count: 3" in txt and "Labels: [0 3 7]" in txt, "Labels output unchanged")
txt = show(il)
ok(txt.splitlines()[:4] == ["Name: img", "Layer type: Image", f"Shape: {img.shape}", "Data type: uint8"]
   and "Mean:" in txt, "Image output unchanged")

v.close()
print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
sys.exit(1 if FAILED else 0)
