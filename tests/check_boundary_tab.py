"""
Headless end-to-end check of the Boundary tab.

Drives a real napari viewer (offscreen) and the real SPROUT main widget.
Uses demo_data plus two skull scans from <skull_dir>/input, skull_dir from tests/local_paths.yaml
(Arctictis_binturong, uint16; Orycteropus_afer, uint8, ~935M voxels). Without them a synthetic
uint16 volume stands in for the first and the speed run on the second is skipped.
Run in the `sprout` env:
    python tests/check_boundary_tab.py
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
import tifffile  # noqa: E402
from qtpy.QtGui import QShowEvent  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from napari_sprout._widget import SPROUTWidget  # noqa: E402
from sprout_core import boundary as B  # noqa: E402
from local_paths import local_path, not_set  # noqa: E402

SKULL = local_path("skull_dir") / "input" if local_path("skull_dir") else None
if SKULL is None:
    print(f"SKIP skull scans: {not_set('skull_dir')}")
FAILED = []


def wait(cond, timeout=120, what=""):
    t = time.time()
    while not cond():
        QApplication.processEvents()
        time.sleep(0.005)
        if time.time() - t > timeout:
            raise TimeoutError(what)


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILED.append(msg)


viewer = napari.Viewer(show=False)
main = SPROUTWidget(viewer)
bw = main.boundary_widget
viewer.window.add_dock_widget(main)
_bi = main.roi_boundary_tabs.indexOf(bw)
ok(_bi >= 0 and main.roi_boundary_tabs.tabText(_bi) == "Boundary"
   and main.tabs.tabText(main.tabs.indexOf(main.roi_boundary_tabs)) == "ROI && Boundary",
   "Boundary is the 'Boundary' sub-tab of 'ROI & Boundary'")

# GPU detection (imports torch) only when GPU is picked or on Compute: not when the tab is shown
ok(bw.device_label.text() == "GPU not checked yet", "GPU not probed while the tab is hidden")
bw.showEvent(QShowEvent())
QApplication.processEvents()
ok(bw.device_label.text() == "GPU not checked yet" and not bw._workers, "GPU not probed when the tab is shown")

# ---- 1. 3D image, auto device
dog = tifffile.imread(ROOT / "demo_data" / "dog_img.tif")
viewer.add_image(dog, name="dog")
QApplication.processEvents()
wait(lambda: bw.input_hist._ready(), what="input hist")
ok(bw.image_combo.currentText() == "dog", "image picked up automatically")

bw.panel.set_params({"kind": "black", "shape": "ball", "size": 3})
bw._compute()
ok(bw.resp_hist._message == "Computing…", "response histogram shows 'Computing…' while computing")
wait(lambda: bw._resp is not None, what="compute")
wait(lambda: not bw._computing)
ref, _ = B.tophat(dog, "black", "ball", 3, "cpu")
ok(np.array_equal(bw._resp, ref), f"widget response == core CPU result ({bw.compute_status.text()})")
ok(bw.device_label.text().startswith(("GPU: ", "CPU only (")), f"Compute (Auto) probed the GPU: {bw.device_label.text()}")

# threshold drag -> preview of the displayed slice only
bw.resp_hist.set_range(30, float(bw._resp.max()))
QApplication.processEvents()
ilo, ihi = bw._intensity_range()
z = int(round(viewer.dims.point[0]))
exp = (ref[z] >= 30) & B.intensity_mask(dog[z], ilo, ihi)
buf = bw._preview_buf
ok(np.array_equal(buf[z].astype(bool), exp), f"preview slice z={z} correct ({int(exp.sum())} voxels)")
ok(buf.sum() == exp.sum(), "nothing written outside the displayed slice")
viewer.dims.set_point(0, z + 50)
QApplication.processEvents()
ok(buf[z].sum() == 0 and buf[z + 50].sum() > 0, "moving the slice clears the old one and draws the new one")
ok(viewer.layers.selection.active is not None and viewer.layers.selection.active.name != "Boundary preview",
   "preview layer does not steal the active-layer selection")

# create
bw._create()
wait(lambda: "dog_boundary" in viewer.layers, what="create")
QApplication.processEvents()
out = viewer.layers["dog_boundary"].data
lo_b, hi_b = bw.resp_hist.get_bounds()
expf = B.threshold_mask(ref, lo_b, hi_b, dog, ilo, ihi)
ok(np.array_equal(out, expf), f"created layer == core threshold_mask ({int(out.sum())} voxels)")
ok("Boundary preview" not in viewer.layers, "preview layer removed after create")
ok(main.seed_widget.boundary_combo.currentText() == "dog_boundary"
   and main.grow_widget.boundary_combo.currentText() == "dog_boundary", "set as boundary in Seeds and Grow")
ok("sprout_boundary" in viewer.layers["dog_boundary"].metadata, "parameters stored in layer metadata: "
   + str(viewer.layers["dog_boundary"].metadata.get("sprout_boundary")))

# re-create overwrites our own layer, does not duplicate
bw.resp_hist.set_range(60, float(bw._resp.max()))
bw._create()
wait(lambda: viewer.layers["dog_boundary"].data.sum() < out.sum(), what="overwrite")
ok(len([l for l in viewer.layers if l.name.startswith("dog_boundary")]) == 1, "re-create overwrites own layer")

# ---- 2. parameters changed during compute -> result discarded
bw._compute()
bw.panel.size_spin.setValue(4)
wait(lambda: not bw._computing)
ok(bw._resp is None and "Parameters changed" in bw.resp_hist._message,
   "stale result discarded after a parameter change")

# ---- 3. CPU device
bw.device_combo.setCurrentIndex(bw.device_combo.findData("cpu"))
bw._compute()
wait(lambda: not bw._computing)
ref4, _ = B.tophat(dog, "black", "ball", 4, "cpu")
ok(np.array_equal(bw._resp, ref4) and "CPU" in bw.compute_status.text(), f"CPU path: {bw.compute_status.text()}")

# ---- 4. intensity range committed -> response histogram recomputed, preview respects it
before = bw.resp_hist._counts.sum()
bw.input_hist.set_range(150, bw.input_hist._xmax)
wait(lambda: bw.resp_hist._ready() and bw.resp_hist._counts.sum() != before, what="resp hist")
z = int(round(viewer.dims.point[0]))
ok(not (bw._preview_buf[z].astype(bool) & (dog[z] < 150)).any(),
   "preview excludes voxels outside the intensity range")

# ---- 5. Stop on a bigger uint16 image
if SKULL is not None and SKULL.is_dir():
    big = tifffile.imread(SKULL / "Arctictis_binturong.tif")
else:
    zz, yy, xx = np.mgrid[:192, :192, :192]
    big = (((zz - 96) ** 2 + (yy - 96) ** 2 + (xx - 96) ** 2 < 80 ** 2) * 30000
           + np.random.default_rng(0).integers(0, 2000, zz.shape)).astype(np.uint16)
    big[:, :, 96:98] //= 4                           # a dark seam across the ball
viewer.add_image(big, name="arctictis")
QApplication.processEvents()
bw.image_combo.setCurrentText("arctictis")
wait(lambda: bw.input_hist._ready(), what="input hist big")
lo, hi = bw.input_hist.get_range()
print(f"     arctictis default intensity range [{lo}, {hi}] (zero background excluded)")
bw.panel.set_params({"size": 5})
bw._compute()
wait(lambda: bw.progress.maximum() > 1 and bw.progress.value() >= 1, what="first chunk")
t = time.time()
bw._stop()
wait(lambda: not bw._computing)
ok(bw._resp is None and bw.compute_status.text().startswith("Stopped after ")
   and bw.compute_status.text().endswith("; nothing kept"),
   f"Stop works ({time.time() - t:.1f}s to stop): {bw.compute_status.text()!r}")

# ---- 5b. uint16: a handle at the right end means "no upper limit"
bw.panel.set_params({"size": 3})
bw._compute()
wait(lambda: not bw._computing, timeout=300)
ok(bw.resp_hist.upper_note.text() == "(no limit)", f"upper starts at 'no limit' (axis max {bw.resp_hist._xmax:.0f})")
ilo, ihi = bw._intensity_range()
ok(ihi is None, f"intensity upper at axis end -> None (axis max {bw.input_hist._xmax:.0f}, true max {big.max()})")
lo_b, hi_b = bw.resp_hist.get_bounds()
resp_big = bw._resp
bw._create()
wait(lambda: "arctictis_boundary" in viewer.layers, timeout=300)
QApplication.processEvents()
ok(np.array_equal(viewer.layers["arctictis_boundary"].data, B.threshold_mask(resp_big, lo_b, hi_b, big, ilo, ihi)),
   "uint16 boundary == threshold_mask with 'no limit' bounds")

# ---- 6. switching image while its histogram is computing -> no mix-up
bw.image_combo.setCurrentText("dog")
bw.image_combo.setCurrentText("arctictis")
bw.image_combo.setCurrentText("dog")
wait(lambda: bw.input_hist._ready(), what="hist after switching")
time.sleep(1)
QApplication.processEvents()
ok(bw.input_hist._xmax == 255 and bw.input_hist._xmin == 0,
   f"histogram belongs to the last selected image, range {bw.input_hist._xmin}..{bw.input_hist._xmax}")

# ---- 7. 2D image
d2 = tifffile.imread(ROOT / "demo_data" / "simple_demo.tif")
viewer.add_image(d2, name="simple2d")
QApplication.processEvents()
bw.image_combo.setCurrentText("simple2d")
wait(lambda: bw.input_hist._ready())
bw.device_combo.setCurrentIndex(0)
bw._compute()
wait(lambda: not bw._computing)
ok(bw._resp is not None and bw._resp.shape == d2.shape, f"2D image works ({bw.compute_status.text()})")
bw.resp_hist.set_range(bw.resp_hist._xmax / 3, bw.resp_hist._xmax)
QApplication.processEvents()
ok(bw._preview_buf is not None and bw._preview_buf.sum() > 0, "2D preview drawn")

# ---- 8. speed on the big skull through the widget
if SKULL is not None and SKULL.is_dir():
    oryc = tifffile.imread(SKULL / "Orycteropus_afer.tif")
    viewer.add_image(oryc, name="orycteropus")
    QApplication.processEvents()
    t = time.perf_counter()
    bw.image_combo.setCurrentText("orycteropus")
    wait(lambda: bw.input_hist._ready())
    t_hist = time.perf_counter() - t
    bw.panel.set_params({"kind": "black", "shape": "ball", "size": 3})
    t = time.perf_counter()
    bw._compute()
    wait(lambda: not bw._computing, timeout=300)
    t_comp = time.perf_counter() - t
    t = time.perf_counter()
    for v in np.linspace(bw.resp_hist._xmin, bw.resp_hist._xmax, 20):
        bw.resp_hist.set_range(v, bw.resp_hist._xmax)
        QApplication.processEvents()
    t_drag = (time.perf_counter() - t) / 20
    t = time.perf_counter()
    bw._create()
    wait(lambda: "orycteropus_boundary" in viewer.layers, timeout=300)
    t_create = time.perf_counter() - t
    print(f"     orycteropus (935M voxels): input histogram {t_hist:.2f}s | compute {t_comp:.1f}s | "
          f"per threshold drag step {t_drag * 1000:.0f} ms | create {t_create:.1f}s")

wait(lambda: len(bw._workers) == 0, timeout=60, what="workers drain")
viewer.close()
print("viewer closed cleanly")
print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
sys.exit(1 if FAILED else 0)
