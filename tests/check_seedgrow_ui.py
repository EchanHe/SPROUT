"""
Headless check of the Seeds / Grow layout round and the plugin-wide message style.

    layout: Input / Threshold Preview / Parameters / Output in both tabs, same labels,
            YAML buttons in the Parameters header, tables stacked full width (headers fit,
            height fits the rows, capped at 8 with an inner scrollbar), histogram ~50 px,
            no footer line, no Refresh button, fits a 500 px panel
    layers: the Input combos follow napari by themselves (add / remove / rename)
    YAML:   export -> import restores every parameter (Seeds and Grow); a failed import is
            a red line in the Parameters group, no dialog
    boundary: a multi-valued boundary gives a red line under the combo, no dialog, no run
    status: StatusLine levels -> 4 colours, "ok" starts with ✓; samples from Seeds, Prompt,
            Boundary

Run in the `sprout` env:
    python tests/check_seedgrow_ui.py
Prints PASS/FAIL per check; exits non-zero if anything failed.
"""
import os
import re
import sys
sys.stdout.reconfigure(encoding="utf-8")   # status texts contain ✓
import tempfile
import time
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
WINDOWS_FONTS = os.path.isdir(r"C:/Windows/Fonts")   # widths only mean something with the real font
if WINDOWS_FONTS:
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:/Windows/Fonts")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "napari_sprout"), str(Path(__file__).resolve().parent)]

import napari  # noqa: E402
import _headless_gl  # noqa: E402,F401  (napari >= 0.7 offscreen: no GL context)
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from napari.qt import get_stylesheet  # noqa: E402
from qtpy.QtWidgets import (QApplication, QFormLayout, QGroupBox, QLabel, QMessageBox, QPushButton,  # noqa: E402
                            QScrollArea, QScrollBar)

from napari_sprout._widget import SPROUTWidget  # noqa: E402

FAILED = []
TMP = Path(tempfile.mkdtemp(prefix="sprout_seedgrow_ui_"))
UNCAUGHT = []
def _hook(*a):
    UNCAUGHT.append("".join(traceback.format_exception(*a)))
    print(UNCAUGHT[-1], flush=True)


sys.excepthook = _hook


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILED.append(msg)


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


def safe(fn, default=None):
    try:
        return fn()
    except Exception as e:                      # an attribute missing on an older version
        print(f"     ({type(e).__name__}: {e})")
        return default


def lev(w):
    return getattr(w, "level", None)


def yaml_io(fn, path):
    """Call import_yaml / export_yaml as the buttons do, the file dialog answering `path`."""
    from qtpy.QtWidgets import QFileDialog
    saved = QFileDialog.getOpenFileName, QFileDialog.getSaveFileName
    QFileDialog.getOpenFileName = QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (str(path), ""))
    try:
        fn()
    finally:
        QFileDialog.getOpenFileName, QFileDialog.getSaveFileName = saved


def items(c):
    return [c.itemText(i) for i in range(c.count())]


# dialogs: none may open (purely informational boxes became inline lines)
DIALOGS = []
for _k in ("warning", "critical", "information", "question"):
    setattr(QMessageBox, _k, staticmethod(lambda *a, _k=_k, **kw: DIALOGS.append((_k, a[2] if len(a) > 2 else a))
                                          or QMessageBox.Ok))

dog = tifffile.imread(ROOT / "demo_data" / "dog_img.tif")[200:420]
dseed = tifffile.imread(ROOT / "demo_data" / "dog_seed.tif")[200:420]
viewer = napari.Viewer(show=False)
dlay = viewer.add_image(dog, name="dog_img")
viewer.add_labels(dseed, name="dog_seed")
main = SPROUTWidget(viewer)
main.setStyleSheet(get_stylesheet("dark"))
main.resize(500, 1000)
main.show()                     # a shown top-level panel 500 px wide: real geometry
sw, gw, bw, pt = main.seed_widget, main.grow_widget, main.boundary_widget, main.prompt_tab
st, gt = sw.threshold_widget, gw.threshold_widget
wait(lambda: st.hist._ready() and gt.hist._ready(), what="histograms")
pump()
sw.output_folder_line.setText(str(TMP / "seeds"))
gw.output_folder_line.setText(str(TMP / "grow"))

# ======================================================================
print("== layout")


def groups(w):
    cont = w.findChild(QScrollArea).widget()
    gs = [g for g in cont.findChildren(QGroupBox) if g.parentWidget() is cont]
    return sorted(gs, key=lambda g: g.mapTo(cont, g.rect().topLeft()).y())


def form_labels(w):
    g = groups(w)[0]
    form = g.layout()
    out = []
    if isinstance(form, QFormLayout):
        for r in range(form.rowCount()):
            it = form.itemAt(r, QFormLayout.LabelRole)
            if it is not None and isinstance(it.widget(), QLabel) and it.widget().text():
                out.append(it.widget().text())
    return out


for w, name, labels in ((sw, "Seeds", ["Image", "Boundary", "ROI"]), (gw, "Grow", ["Image", "Seeds", "Boundary", "ROI"])):
    main.tabs.setCurrentWidget(w)
    pump()
    titles = [g.title() for g in groups(w)]
    ok(titles == ["Input", "Threshold Preview", "Parameters", "Output"], f"{name}: sections {titles}")
    ok(form_labels(w) == labels, f"{name}: Input labels {form_labels(w)}")
    texts = {b.text() for b in w.findChildren(QPushButton)}
    ok("Refresh Layers" not in texts and not hasattr(w, "refresh_btn") and callable(getattr(w, "refresh_layers", None)),
       f"{name}: no Refresh Layers button (refresh_layers() kept)")
    mp = w.main_param_widget
    ok(w.import_btn.parentWidget() is mp and w.export_btn.parentWidget() is mp
       and w.import_btn.mapTo(mp, w.import_btn.rect().topLeft()).y()
       < mp.threshold_table.mapTo(mp, mp.threshold_table.rect().topLeft()).y(),
       f"{name}: Import / Export YAML in the Parameters header row")
    ok(w.show_advanced_checkbox.parentWidget() is mp and w.show_advanced_checkbox.text() == "Show advanced options",
       f"{name}: 'Show advanced options' inside Parameters")
    out_group = groups(w)[-1]
    ok(w.output_folder_line.parentWidget() is out_group, f"{name}: the folder row is in the Output group")
    ok(all(hasattr(w, a) for a in ("main_param_widget", "output_folder_line", "image_combo", "boundary_combo",
                                   "results_label", "status_label", "threshold_widget"))
       and (hasattr(w, "generate_btn") or hasattr(w, "grow_btn")), f"{name}: attribute names kept")

labs = [lab.text() for lab in main.findChildren(QLabel)]
ok(not any("Workflow" in t for t in labs), "footer 'Workflow: …' removed")
ok(st.hist.canvas.height() <= 55 and gt.hist.canvas.height() <= 55, f"histogram height {st.hist.canvas.height()} px (<= 55)")
main.tabs.setCurrentWidget(main.roi_boundary_tabs)
main.roi_boundary_tabs.setCurrentWidget(bw)
pump()
ok(bw.input_hist.canvas.height() <= 55 and bw.resp_hist.canvas.height() <= 55,
   f"Boundary histograms {bw.input_hist.canvas.height()} / {bw.resp_hist.canvas.height()} px")

# tables: stacked, full width, headers fit, height = rows, capped at 8
main.resize(500, 1000)
main.tabs.setCurrentWidget(sw)
pump(40)
mp = sw.main_param_widget
tt, ft = mp.threshold_table, mp.footprint_table
top = lambda w: w.mapTo(mp, w.rect().topLeft()).y()   # noqa: E731
ok(top(ft) >= top(tt) + tt.height(), "Seeds: Footprints table below the Thresholds table")
inner = mp.contentsRect().width()
ok(tt.width() >= 0.85 * inner and ft.width() >= 0.85 * inner and abs(tt.width() - ft.width()) <= 2,
   f"Seeds: both tables full width ({tt.width()} / {ft.width()} of {inner})")


def headers_fit(t):
    h = t.horizontalHeader()
    return all(h.sectionSizeHint(i) <= h.sectionSize(i) for i in range(t.columnCount()))


def fitted(t):
    n = min(t.rowCount(), 8)
    want = 2 * t.frameWidth() + t.horizontalHeader().sizeHint().height() + sum(t.rowHeight(i) for i in range(n))
    return abs(t.height() - want) <= 1 + (t.horizontalScrollBar().sizeHint().height() if t.horizontalScrollBar().isVisible() else 0)


gm = gw.main_param_widget
main.tabs.setCurrentWidget(gw)
pump(20)
for t, what in ((tt, "Seeds thresholds"), (ft, "Seeds footprints"), (gm.threshold_table, "Grow thresholds")):
    if not WINDOWS_FONTS or os.environ.get("CI"):    # text is a third wider on the Windows CI machines
        print(f"SKIP {what}: header texts fit: needs the fonts of a Windows desktop")
        continue
    ok(headers_fit(t), f"{what}: header texts fit ({[t.horizontalHeader().sectionSizeHint(i) for i in range(t.columnCount())]} "
                       f"<= {[t.horizontalHeader().sectionSize(i) for i in range(t.columnCount())]})")
main.tabs.setCurrentWidget(sw)
pump(20)
mp.populate_thresholds_from_list([100, 110, 120], [255, 255, 255])
pump(20)
h3 = tt.height()
ok(fitted(tt), f"3 rows: the table is as high as its rows ({h3} px)")
mp.populate_thresholds_from_list(list(range(100, 112)), [255] * 12)
pump(20)
ok(fitted(tt) and tt.height() > h3 and tt.verticalScrollBar().maximum() > 0,
   f"12 rows: capped at 8 ({tt.height()} px), scrolls inside")
mp.populate_thresholds_from_list([140], [255])
pump(20)
ok(fitted(tt) and tt.height() < h3, f"1 row: shrinks back ({tt.height()} px)")
main.tabs.setCurrentWidget(gw)
gm.populate_thresholds_from_list([150, 120], [255, 255], [3, 3])
pump(20)
ok(fitted(gm.threshold_table), f"Grow table fits its 2 rows ({gm.threshold_table.height()} px)")
gm.remove_all_threshold_btn.click()
pump()
ok(gm.threshold_table.rowCount() == 0 and not UNCAUGHT, "Grow 'Remove All Thresholds' clears the table")
gm.populate_thresholds_from_list([150, 120], [255, 255], [3, 3])

# 500 px
for w, name in ((sw, "Seeds"), (gw, "Grow")):
    main.tabs.setCurrentWidget(w)
    w.show_advanced_checkbox.setChecked(True)
    pump(30)
    sa = w.findChild(QScrollArea)
    avail = sa.viewport().width()
    need = sa.widget().minimumSizeHint().width()
    if WINDOWS_FONTS:
        ok(need <= avail, f"{name}: fits a 500 px panel (content needs {need} px of {avail})")
    else:
        print(f"SKIP {name}: fits a 500 px panel: needs the Windows fonts ({need} px of {avail})")
    w.show_advanced_checkbox.setChecked(False)

# output folder: shown middle-elided, real value unchanged
long_path = str(TMP / ("a_very_long_folder_name_" * 4) / "out")
sw.output_folder_line.setText(long_path)
ok(sw.output_folder_line.text() == long_path and sw.output_folder_line.toolTip() == long_path,
   "Output folder: full path is the value and the tooltip")
sw.output_folder_line.setText(str(TMP / "seeds"))

# ======================================================================
print("\n== the Input combos follow napari (add / remove / rename)")
main.tabs.setCurrentWidget(sw)
viewer.add_image(dog[:20], name="img_b")
viewer.add_labels(dseed[:20], name="lab_b")
pump()
ok("img_b" in items(sw.image_combo) and "img_b" in items(gw.image_combo) and "lab_b" in items(sw.boundary_combo)
   and "lab_b" in items(gw.seeds_combo) and "lab_b" in items(gw.boundary_combo), "added layers appear")
viewer.layers["img_b"].name = "img_c"
viewer.layers["lab_b"].name = "lab_c"
pump()
ok("img_c" in items(sw.image_combo) and "img_b" not in items(sw.image_combo) and "img_c" in items(gw.image_combo)
   and "lab_c" in items(sw.boundary_combo) and "lab_c" in items(gw.seeds_combo) and "lab_b" not in items(gw.boundary_combo),
   "renamed layers are renamed in the combos")
gw.seeds_combo.setCurrentText("lab_c")
rows_before = sw.main_param_widget.get_params()["thresholds"]
dlay.name = "dog_renamed"
viewer.layers["lab_c"].name = "lab_d"
pump()
ok(sw.image_combo.currentText() == "dog_renamed" and gw.image_combo.currentText() == "dog_renamed"
   and gw.seeds_combo.currentText() == "lab_d" and sw.main_param_widget.get_params()["thresholds"] == rows_before,
   "renaming the selected layers keeps them selected (and the threshold table)")
dlay.name = "dog_img"
pump()
viewer.layers.remove("img_c")
viewer.layers.remove("lab_d")
pump()
ok("img_c" not in items(sw.image_combo) and "lab_d" not in items(gw.seeds_combo) and "lab_d" not in items(sw.boundary_combo)
   and sw.image_combo.currentText() == "dog_img" and gw.seeds_combo.currentText() == "dog_seed",
   "removed layers disappear; the selections fall back")

# ======================================================================
print("\n== YAML round trip, failures inline")
try:
    from napari_sprout.utils import status as S  # noqa: E402
except ImportError as e:                        # older version: the status checks fail, the rest runs
    print(f"     ({e})")
    S = None

mp = sw.main_param_widget
ap = sw.advanced_params_box
mp.seed_method_combo.setCurrentText("Adaptive (Erosion)")
mp.thread_spin.setValue(2)
mp.erosion_spin.setValue(2)
mp.segments_spin.setValue(7)
mp.populate_thresholds_from_list([120, 140], [250, 255])
safe(lambda: mp.set_footprints(["ball", "cube"]))
ap.set_params({"sort": False, "min_size": 7, "no_split_max_iter": 4, "min_split_ratio": 0.05,
               "min_split_total_ratio": 0.1, "split_size_limit": [10, 5000], "split_convex_hull_limit": [None, 300]})
sw.save_every_iter_checkbox.setChecked(True)


def seed_state():
    return (mp.get_params(), ap.get_params(), sw.output_folder_line.text(), sw.save_every_iter_checkbox.isChecked(),
            sw.image_combo.currentText(), sw.boundary_combo.currentText())


want = seed_state()
yml = TMP / "seeds.yaml"
yaml_io(sw.export_yaml, yml)
mp.seed_method_combo.setCurrentText("Original")
mp.thread_spin.setValue(1)
mp.erosion_spin.setValue(1)
mp.segments_spin.setValue(3)
mp.populate_thresholds_from_list([10], [20])
mp.footprint_table.setRowCount(0)
ap.set_params({"sort": True, "min_size": 1, "no_split_max_iter": 3, "min_split_ratio": 0.0,
               "min_split_total_ratio": 0.0, "split_size_limit": [None, None], "split_convex_hull_limit": [None, None]})
sw.output_folder_line.setText(str(TMP / "elsewhere"))
sw.save_every_iter_checkbox.setChecked(False)
yaml_io(sw.import_yaml, yml)
pump()
got = seed_state()
ok(got == want, "Seeds: export -> change everything -> import restores every parameter"
   + ("" if got == want else f"\n       want {want}\n       got  {got}"))
ok(getattr(sw, "params_status", None) is not None and sw.params_status.text().startswith("✓ Parameters successfully loaded")
   and sw.params_status.level == "ok", f"Seeds: import -> green line in Parameters ({safe(lambda: sw.params_status.text())!r})")

gp, ga = gw.main_param_widget, gw.advanced_params_box
gw.seeds_combo.setCurrentText("dog_seed")
gp.thread_spin.setValue(3)
gp.touch_rule_combo.setCurrentText("overwrite")
gp.populate_thresholds_from_list([150, 120], [255, 250], [3, 4])
ga.set_params({"save_every_n_iters": 5, "grow_to_end": True, "is_sort": True, "to_grow_ids": [1, 2],
               "min_growth_size": 20, "no_growth_max_iter": 6})


def grow_state():
    return (gp.get_params(), ga.get_params(), gw.output_folder_line.text(), gw.image_combo.currentText(),
            gw.seeds_combo.currentText(), gw.boundary_combo.currentText())


want = grow_state()
yml = TMP / "grow.yaml"
yaml_io(gw.export_yaml, yml)
gp.thread_spin.setValue(1)
gp.touch_rule_combo.setCurrentText("stop")
gp.populate_thresholds_from_list([10], [20], [1])
ga.set_params({"save_every_n_iters": 2, "grow_to_end": False, "is_sort": False, "to_grow_ids": [9],
               "min_growth_size": 50, "no_growth_max_iter": None})
gw.output_folder_line.setText(str(TMP / "elsewhere"))
yaml_io(gw.import_yaml, yml)
pump()
got = grow_state()
ok(got == want, "Grow: export -> change everything -> import restores every parameter"
   + ("" if got == want else f"\n       want {want}\n       got  {got}"))

bad = TMP / "bad.yaml"
bad.write_text("thresholds: [1, 2\nseed_method: : :\n")
n0 = len(DIALOGS)
yaml_io(sw.import_yaml, bad)
yaml_io(gw.import_yaml, bad)
pump()
ok(len(DIALOGS) == n0 and safe(lambda: sw.params_status.level) == "error"
   and safe(lambda: sw.params_status.text(), "").startswith("Failed to load parameters:")
   and safe(lambda: gw.params_status.level) == "error" and not sw.params_status.isHidden(),
   f"YAML import failure: red line in Parameters, no dialog ({safe(lambda: sw.params_status.text())!r})")

# ======================================================================
print("\n== boundary that cannot be used: red line under the combo, no dialog, no run")
three = np.zeros(dog.shape, np.uint8)
three[:, :, 60] = 1
three[:, :, 61] = 2
viewer.add_labels(three, name="three_values")
pump()
main.tabs.setCurrentWidget(sw)
mp.seed_method_combo.setCurrentText("Original")
mp.erosion_spin.setValue(0)
mp.segments_spin.setValue(3)
mp.footprint_table.setRowCount(0)
mp.populate_thresholds_from_list([140], [255])
sw.boundary_combo.setCurrentText("three_values")
before = {l.name for l in viewer.layers}
n0 = len(DIALOGS)
sw.generate_btn.click()
pump(60)
err = getattr(sw, "boundary_error", None)
ok(len(DIALOGS) == n0 and err is not None and not err.isHidden() and err.level == "error"
   and "more than two values" in err.text() and "three_values" in err.text(),
   f"Seeds: red line under Boundary ({safe(lambda: err.text())!r}), no dialog ({DIALOGS[n0:]})")
ok({l.name for l in viewer.layers} == before and sw.generate_btn.isEnabled() and sw.worker is None,
   "Seeds: no run started")
main.tabs.setCurrentWidget(gw)
gw.boundary_combo.setCurrentText("three_values")
n0 = len(DIALOGS)
gw.grow_btn.click()
pump(60)
gerr = getattr(gw, "boundary_error", None)
ok(len(DIALOGS) == n0 and gerr is not None and not gerr.isHidden() and gerr.level == "error"
   and {l.name for l in viewer.layers} == before and gw.grow_btn.isEnabled() and gw.worker is None,
   "Grow: red line under Boundary, no dialog, no run")
viewer.add_image(dog[:5], name="tmp_layer")      # an unrelated layer change keeps the line
pump()
ok(safe(lambda: not err.isHidden() and bool(err.text()), False), "the red line stays while the same boundary is selected")
sw.boundary_combo.setCurrentText("None")
gw.boundary_combo.setCurrentText("None")
pump()
ok(safe(lambda: err.isHidden() and gerr.isHidden(), False), "choosing another boundary clears it")
viewer.layers.remove("tmp_layer")

# ======================================================================
print("\n== StatusLine levels, per-tab samples")
cols = {}
if S is not None:
    line = S.StatusLine()
    for lv in ("info", "ok", "warn", "error"):
        line.set("text", lv)
        cols[lv] = (line.level, S.level_of(line), line.text())
ok(S is not None and [cols[k][1] for k in cols] == ["info", "ok", "warn", "error"] and cols["ok"][2] == "✓ text"
   and cols["info"][2] == "text" and len(set(S.COLORS.values())) == 4,
   f"StatusLine: 4 levels, 4 colours, 'ok' starts with ✓ ({getattr(S, 'COLORS', None)})")

# Seeds completion
main.tabs.setCurrentWidget(sw)
ok(sw.results_label.text() == "No seeds generated yet" and lev(sw.results_label) == "info", "Seeds: 'No seeds generated yet' grey")
ok(gw.status_label.text() == "Ready to grow seeds" and lev(gw.status_label) == "info", "Grow: 'Ready to grow seeds' grey")
sw.generate_btn.click()
wait(lambda: sw.generate_btn.isEnabled() and sw.worker is not None and not sw.worker.isRunning(), what="seeds")
pump()
ok(re.fullmatch(r"✓ Seeds generated: \d+ layers? \(\d+\.\d s\)", sw.results_label.text()) is not None
   and lev(sw.results_label) == "ok", f"Seeds completion: {sw.results_label.text()!r} (green)")
# after a Seeds run Grow names the seeds it picked (polish E4), grey
ok(gw.status_label.text() == f"Seeds: {gw.seeds_combo.currentText()} (last Seeds run)" and lev(gw.status_label) == "info",
   f"Grow after the run: {gw.status_label.text()!r} grey")

# Prompt: no model -> red; Done -> green
from napari_sprout.widgets.prompt_tab import PromptTab  # noqa: E402
import napari_sprout.widgets.prompt_tab as _pt  # noqa: E402
LOAD_FIRST = getattr(_pt, "LOAD_FIRST", "Load the model first (3. Model)")
from qtpy.QtCore import QSettings  # noqa: E402
from sprout_core.nni_engine import NniEngine  # noqa: E402
from _prompt_check_utils import FakeSession26  # noqa: E402

_pt.SETTINGS = ("SPROUT-checks", "napari-sprout-checks")
QSettings(*_pt.SETTINGS).clear()
fake = PromptTab(viewer, engine_factory=lambda: NniEngine(session_factory=lambda dev: FakeSession26(device=dev),
                                                          buffer_factory=lambda s: np.zeros(s, np.uint8)),
                 status_fn=lambda: (True, "fake"))
pump()
fake.image_combo.setCurrentText("dog_img")
fake.seeds_combo.setCurrentText("dog_seed")
wait(lambda: fake.table.rowCount() >= 1 and fake.image_layer is not None and fake.seeds_layer is not None, what="prompt table")
fake.set_checked(fake.class_ids()[:1])
fake.run_checked()
pump()
ok(fake.run_status.text() == LOAD_FIRST and lev(fake.run_status) == "error", f"Prompt: run before Load -> red {LOAD_FIRST!r}")
fake.device_combo.setCurrentIndex(fake.device_combo.findData("cpu"))
ok(safe(lambda: S.level_of(fake.cpu_label)) == "warn", "Prompt: CPU note orange")
fake.model_edit.setText("fake_model_dir")
fake.load_model()
wait(lambda: not any(fake._busy.values()) and fake.engine.loaded, what="load")
ok(fake.run_status.text() == "", "Prompt: the 'load first' line is cleared after Load")
fake.generate()
wait(lambda: not any(fake._busy.values()), what="sample")
fake.run_checked()
wait(lambda: not any(fake._busy.values()), what="run")
pump()
ok(fake.run_status.text().startswith("✓ Done: 1 classes in") and lev(fake.run_status) == "ok",
   f"Prompt: {fake.run_status.text()!r} (green)")
fake.deleteLater()

# Boundary: Created -> green inline (+ the notification)
main.tabs.setCurrentWidget(main.roi_boundary_tabs)
main.roi_boundary_tabs.setCurrentWidget(bw)
bw.image_combo.setCurrentText("dog_img")
wait(lambda: bw.input_hist._ready(), what="boundary hist")
bw.device_combo.setCurrentIndex(bw.device_combo.findData("cpu"))
bw.panel.set_params({"kind": "black", "shape": "ball", "size": 2})
bw._create()
ok(safe(lambda: bw.output_status.text()) == "Compute first" and safe(lambda: bw.output_status.level) == "error",
   "Boundary: Create before Compute -> red 'Compute first'")
bw._compute()
wait(lambda: bw._resp is not None and not bw._computing, what="boundary compute")
bw.resp_hist.set_range(25, bw.resp_hist._xmax)
bw._create()
wait(lambda: "dog_img_boundary" in viewer.layers, what="boundary create")
pump()
ok(safe(lambda: bw.output_status.text(), "").startswith("✓ Created 'dog_img_boundary':")
   and safe(lambda: bw.output_status.level) == "ok", f"Boundary: {safe(lambda: bw.output_status.text())!r} (green)")
bw.panel.set_params({"kind": "black", "shape": "ball", "size": 3})
pump()
ok(bw.compute_status.text() == "Parameters changed: click Compute" and safe(lambda: S.level_of(bw.compute_status)) == "warn",
   "Boundary: 'Parameters changed: click Compute' orange")

ok(main.width() == 500, f"panel width {main.width()} px")
ok(not DIALOGS, f"no dialog opened ({DIALOGS})")
ok(not UNCAUGHT, f"no uncaught errors ({len(UNCAUGHT)})" + (f": {UNCAUGHT[0][-500:]}" if UNCAUGHT else ""))
wait(lambda: not (st._workers or gt._workers or bw._workers), timeout=60)
viewer.close()
print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
sys.exit(1 if FAILED else 0)
