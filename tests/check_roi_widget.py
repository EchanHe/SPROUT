"""
Headless check of the napari side of ROI v3: ROI layers picked per tab ("ROI" combo in
Seeds / Grow / Prompt), the ROI tools (ROI & Boundary -> ROI), and every tab running inside
the ROI it picked.

Drives a real napari viewer (offscreen) and the real SPROUT main widget.
Run in the `sprout` env:
    python tests/check_roi_widget.py
Prints PASS/FAIL per check; exits non-zero if anything failed.
"""
import contextlib
import gc
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
import weakref
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.path.isdir(r"C:/Windows/Fonts"):
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:/Windows/Fonts")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "napari-sprout")]

import napari  # noqa: E402
import _headless_gl  # noqa: E402,F401  (napari >= 0.7 offscreen: no GL context)
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from sprout_core import make_adaptive_seed as MA  # noqa: E402
from sprout_core import make_grow as MG  # noqa: E402
from sprout_core import make_seeds as MS  # noqa: E402
import napari_sprout.utils.histogram_threshold_widget as HTM  # noqa: E402
import napari_sprout.widgets.boundary_widget as BWM  # noqa: E402
import napari_sprout.widgets.grow_widget as GWM  # noqa: E402
import napari_sprout.widgets.prompt_tab as PTM  # noqa: E402
import napari_sprout.widgets.seed_widget as SWM  # noqa: E402
from napari_sprout._widget import SPROUTWidget  # noqa: E402
from napari_sprout.utils import roi_layers as RL  # noqa: E402
from napari_sprout.utils.roi_widget import RoiTools  # noqa: E402
from skimage.filters import threshold_otsu  # noqa: E402
from sprout_core import boundary as B  # noqa: E402
from sprout_core import roi as R  # noqa: E402
from sprout_core.roi import Roi  # noqa: E402
from sprout_core.sprout_core import cast_labels  # noqa: E402

FAILED = []
TMP = tempfile.mkdtemp(prefix="sprout_roi_widget_")
UNCAUGHT = []
sys.excepthook = lambda *a: UNCAUGHT.append("".join(traceback.format_exception(*a)))
ERRORS = []
DASH, DOT, X = "–", "·", "×"


def record_error(msg):
    ERRORS.append(str(msg))
    print(f"     show_error: {msg}")


for mod in (SWM, GWM, BWM, HTM, PTM):
    mod.show_error = record_error


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILED.append(msg)


def wait(cond, timeout=300, what=""):
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


def pump_for(sec):
    t = time.time()
    while time.time() - t < sec:
        QApplication.processEvents()
        time.sleep(0.005)


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def combo_texts(c):
    return [c.itemText(i) for i in range(c.count())]


def run_seeds(sw):
    sw.generate_seeds()
    wait(lambda: sw.generate_btn.isEnabled(), what="seeds")
    pump()


def run_grow(gw):
    gw.start_growth()
    wait(lambda: gw.grow_btn.isEnabled(), what="grow")
    pump()


def idle(*ws):
    wait(lambda: not any(w._workers for w in ws), what="workers")
    pump()


class Count:
    """Count histogram work: boundary.histogram calls, and the threshold widgets' cache
    lookups with / without an ROI."""

    def __init__(self):
        self.hist, self.cache_roi, self.cache_whole = 0, 0, 0
        self._h, self._c = B.histogram, HTM._cached_histogram
        me = self

        def h(*a, **k):
            me.hist += 1
            return me._h(*a, **k)

        def c(img, roi):
            if roi is None:
                me.cache_whole += 1
            else:
                me.cache_roi += 1
            return me._c(img, roi)

        B.histogram, HTM._cached_histogram = h, c

    def restore(self):
        B.histogram, HTM._cached_histogram = self._h, self._c


def finish():
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
    sys.exit(1 if FAILED else 0)


# ======================================================================
print("== layout, defaults")
viewer = napari.Viewer(show=False)
dog = tifffile.imread(ROOT / "demo_data" / "dog_img.tif")[200:420]
dseed = tifffile.imread(ROOT / "demo_data" / "dog_seed.tif")[200:420]
dlay = viewer.add_image(dog, name="dog")
viewer.add_labels(dseed, name="dog_seed")
olay = viewer.add_image(np.zeros((30, 40, 50), np.uint8), name="other")
viewer.layers.selection.active = dlay
pump()
main = SPROUTWidget(viewer)
viewer.window.add_dock_widget(main)
hub = RL.get_roi_hub(viewer)
sw, gw, pt, bw, rt = main.seed_widget, main.grow_widget, main.prompt_tab, main.boundary_widget, main.roi_tools
st, gt = sw.threshold_widget, gw.threshold_widget
for w in (sw, gw, bw):
    w.image_combo.setCurrentText("dog")
pt.image_combo.setCurrentText("dog")
pt.seeds_combo.setCurrentText("dog_seed")
wait(lambda: st.hist._ready() and gt.hist._ready() and bw.input_hist._ready(), what="hists")
wait(lambda: pt._stats, what="prompt stats")
idle(st, gt, bw, pt)
ok([main.tabs.tabText(i) for i in range(main.tabs.count())] == ["1. Seeds", "2. Grow", "3. Prompt (AI)", "ROI && Boundary"]   # "&&" shows as "&"
   and [main.roi_boundary_tabs.tabText(i) for i in range(2)] == ["ROI", "Boundary"]
   and main.roi_boundary_tabs.widget(0) is rt and main.roi_boundary_tabs.widget(1) is bw
   and main.tabs.currentIndex() == 0, "tabs: 1. Seeds | 2. Grow | 3. Prompt (AI) | ROI & Boundary (ROI | Boundary); opens on Seeds")
ok(not hasattr(main, "roi_manager") and not hasattr(bw, "roi_use") and not hasattr(sw, "roi_use"),
   "no ROI Manager, no 'Use current ROI', no ROI in Boundary")
sels = {"Seeds": sw.roi_select, "Grow": gw.roi_select, "Prompt": pt.roi_select}
ok(all(combo_texts(s.combo) == ["None"] and s.combo.currentIndex() == 0 and s.state() == "none"
       for s in sels.values()), "every tab: ROI combo with only 'None', selected")
ok(sw.generate_btn.text() == "Generate Seeds" and gw.grow_btn.text() == "Start Growth" and pt.run_btn.text() == "Run checked",
   "run buttons: plain texts")

# ROI sub-tab layout: four titled blocks
from qtpy.QtWidgets import QAbstractSpinBox, QGroupBox, QLineEdit  # noqa: E402
boxes = [b for b in rt.findChildren(QGroupBox) if b.parentWidget() is rt]
boxes.sort(key=lambda b: rt.layout().indexOf(b))
ok([b.title() for b in boxes] == ["Draw a box", "From labels", "Edit ROI", "Save / Load"],
   f"ROI sub-tab blocks: {[b.title() for b in boxes]}")
ok(not hasattr(rt, "name_edit") and not [e for e in rt.findChildren(QLineEdit) if e is not rt.ids_edit
                                         and not isinstance(e.parentWidget(), QAbstractSpinBox)],
   "no name field (rename in the napari layer list)")
ok(rt.whole_btn.text() == "Full image (then set z)" and "z range" in rt.whole_btn.toolTip()
   and "200–400" in rt.whole_btn.toolTip() and rt.draw_btn.text() == "Draw box"
   and rt.create_btn.text() == "Create ROI", f"buttons: {rt.draw_btn.text()!r}, {rt.whole_btn.text()!r}, "
                                             f"{rt.create_btn.text()!r}")
ok(len(boxes) == 4 and rt.draw_btn.parentWidget() is boxes[0] and rt.whole_btn.parentWidget() is boxes[0]
   and boxes[0].layout().getItemPosition(boxes[0].layout().indexOf(rt.draw_btn))[0]
   != boxes[0].layout().getItemPosition(boxes[0].layout().indexOf(rt.whole_btn))[0],
   "Draw box and Full image: separate rows in 'Draw a box'")
ok(len(boxes) == 4 and not boxes[2].isEnabled() and not rt.empty_note.isHidden()
   and rt.empty_note.text() == "No ROI yet: draw a box or create one from labels above",
   "no ROI layer: the Edit ROI block is disabled, one grey line")
ok(rt.roi_combo.toolTip() == "Rename in the napari layer list", f"ROI combo tooltip: {rt.roi_combo.toolTip()!r}")
# width as napari shows it: its stylesheet; real fonts (offscreen Qt has none by default)
from napari.qt import get_stylesheet  # noqa: E402
main.tabs.setCurrentIndex(3)
main.roi_boundary_tabs.setCurrentIndex(0)
main.setStyleSheet(get_stylesheet("dark"))
pump()
w_min = rt.minimumSizeHint().width()
main.setStyleSheet("")
main.tabs.setCurrentIndex(0)
pump()
ok(w_min <= 470, f"fits in a 500 px panel (ROI sub-tab needs {w_min} px of the ~470 inside the tabs)")
sw.output_folder_line.setText(os.path.join(TMP, "seeds"))
gw.output_folder_line.setText(os.path.join(TMP, "grow"))
mp = sw.main_param_widget
mp.seed_method_combo.setCurrentText("Original")
mp.populate_thresholds_from_list([140, 160], [255, 255])
mp.erosion_spin.setValue(2)
mp.segments_spin.setValue(6)
gw.seeds_combo.setCurrentText("dog_seed")
gw.main_param_widget.populate_thresholds_from_list([150, 120], [255, 255], [3, 3])
pt.set_checked(pt.class_ids()[:2])

# ======================================================================
print("\n== picking ROIs in the tabs")
cnt = Count()
calls = {"boundary": 0, "prompt": 0}
_cs = PTM.PromptTab.compute_stats
PTM.PromptTab.compute_stats = lambda self, *a, **k: (calls.__setitem__("prompt", calls["prompt"] + 1),
                                                     _cs(self, *a, **k))[1]
l1 = RL.add_roi_layer(viewer, dlay, (30, 20, 10), (150, 140, 110))
l2 = RL.add_roi_layer(viewer, olay, (0, 0, 0), (10, 10, 10))
pump_for(0.6)
ok(all(combo_texts(s.combo) == ["None", "ROI_1", f"ROI_2 (30{X}40{X}50)"] and s.state() == "none"
       for s in sels.values()), f"new ROI layers are listed, never picked by themselves: {combo_texts(sw.roi_select.combo)}")
ok(cnt.hist == 0 and cnt.cache_roi == 0 and cnt.cache_whole == 0 and calls["prompt"] == 0,
   "ROI layers exist but no tab picks one: nothing recomputed")
ok(all(not combo_texts(c) or not any(t.startswith("ROI_") for t in combo_texts(c))
       for c in (sw.boundary_combo, gw.boundary_combo, gw.seeds_combo, pt.seeds_combo, sw.image_combo, bw.image_combo)),
   "ROI layers are not offered as image / seeds / boundary")
sw.roi_select.combo.setCurrentIndex(1)
sw.roi_select.combo.activated.emit(1)
pump_for(0.4)
roi1 = hub.roi_of(l1)
ok(sw.generate_btn.text() == "Generate Seeds in ROI_1" and gw.grow_btn.text() == "Start Growth",
   f"run button names the ROI: {sw.generate_btn.text()!r} (Grow untouched)")
ok(sw.roi_select.info.text() == f"z 30{DASH}149 {DOT} y 20{DASH}139 {DOT} x 10{DASH}109\n"
   f"120 {X} 120 {X} 100 = 1.44 M voxels (33%)" and sw.roi_select.error.isHidden(),
   f"info lines: {sw.roi_select.info.text()!r}")
ok(st.roi_reminder.text() == f"Preview inside ROI_1: 120 {X} 120 {X} 100" and not st.roi_reminder.isHidden(),
   f"threshold preview line: {st.roi_reminder.text()!r}")
wait(lambda: st.hist._ready() and np.array_equal(st.hist._counts, B.histogram(roi1.sample_values(dog))[0]), timeout=60)
ok(cnt.cache_roi == 1, f"Seeds histogram inside ROI_1 ({cnt.cache_roi} lookup)")
n_hist = cnt.hist
gw.roi_select.select(l1)
wait(lambda: gt.hist._ready() and np.array_equal(gt.hist._counts, st.hist._counts), timeout=60)
pump()
ok(cnt.hist == n_hist and cnt.cache_roi == 2, "two tabs on the same ROI -> one histogram computation (cache by value)")
otsu = threshold_otsu(B.subsample(roi1.sample_values(dog)))
ok(abs(st.hist.get_range()[0] - otsu) < 1e-9, "untouched range moves to the ROI's Otsu")

# mismatch / removed
gw.roi_select.select(l2)
pump()
mis = f"ROI_2 is 30{X}40{X}50; image is 220{X}161{X}122"
ok(gw.roi_select.error.text() == mis and not gw.roi_select.error.isHidden() and gw.roi_select.state() == "mismatch",
   f"shape mismatch: red {gw.roi_select.error.text()!r}")
ERRORS[:] = []
gw.start_growth()
pump()
ok(gw.worker is None and not ERRORS and gw.status_label.text() == mis and gw.status_label.level == "error",
   f"Grow refused: red status {gw.status_label.text()!r}, no notification {ERRORS}")
gw.roi_select.select(l1)
pump_for(0.4)
cnt2_roi = cnt.cache_roi
pt.roi_select.select(l1)
wait(lambda: calls["prompt"] >= 1, timeout=30)
pump_for(0.3)
idle(st, gt, pt)

# drag storm: only the tabs on that layer recompute, once, after the release
calls["prompt"] = 0
c0_roi, c0_whole = cnt.cache_roi, cnt.cache_whole
moved = []
for i in range(20):
    l1._data_view.shift(0, np.array([0.25, 0.0]))
    l1.events.set_data()
    QApplication.processEvents()
pump_for(0.15)
ok(st.hist._message == "ROI changed…" and gt.hist._message == "ROI changed…" and cnt.cache_roi == c0_roi
   and calls["prompt"] == 0, "while dragging: histograms show 'ROI changed…', nothing computed")
ok(sw.roi_select.info.text().startswith(f"z 30{DASH}149 {DOT} y 25{DASH}144"), f"info follows the drag live: "
                                                                             f"{sw.roi_select.info.text().splitlines()[0]!r}")
l1.events.data(value=l1.data, action="changed", data_indices=(0,), vertex_indices=((),))
wait(lambda: st.hist._ready() and gt.hist._ready(), timeout=60, what="after release")
pump_for(0.6)
idle(st, gt, pt)
roi1 = hub.roi_of(l1)
ok(roi1.lo == (30, 25, 10) and cnt.cache_roi - c0_roi == 2 and cnt.cache_whole == c0_whole and calls["prompt"] == 1,
   f"release: one recompute per tab on ROI_1 (threshold lookups {cnt.cache_roi - c0_roi}, prompt {calls['prompt']})")
ok(np.array_equal(st.hist._counts, B.histogram(roi1.sample_values(dog))[0]), "histogram of the moved ROI")
hub.set_yx(l1, (20, 10), (140, 110))
pump_for(0.6)
idle(st, gt, pt)
roi1 = hub.roi_of(l1)

# removed layer
ltmp = RL.add_roi_layer(viewer, dlay, (0, 0, 0), (5, 5, 5))
pump()
sw.roi_select.select(ltmp)
pump_for(0.4)
viewer.layers.remove(ltmp)
pump_for(0.4)
rem = "ROI 'ROI_3' was removed — choose another ROI or None"
ok(sw.roi_select.error.text() == rem and sw.roi_select.combo.currentText() == "ROI_3 (removed)"
   and sw.generate_btn.text() == "Generate Seeds", f"picked layer removed: red {sw.roi_select.error.text()!r}")
ERRORS[:] = []
w0 = sw.worker
sw.generate_seeds()
st._create_preview_layer()
pump()
ok(sw.worker is w0 and not ERRORS and sw.results_label.text() == rem and sw.results_label.level == "error"
   and st.error_line.text() == rem and not st.error_line.isHidden(),
   "Seeds and Create Preview Layer refuse inline, in red (no silent whole-image run, no notification)")
sw.roi_select.select(l1)
pump_for(0.5)
idle(st, gt, pt)
cnt.restore()
PTM.PromptTab.compute_stats = _cs

# ======================================================================
print("\n== runs inside ROI_1")
st.hist.set_range(140, st.hist._xmax)
st.use_connected_components_checkbox.setChecked(True)
st.connected_components_spin.setValue(8)
st.erosion_spin.setValue(1)
wait(lambda: st.create_btn.isEnabled())
st.create_btn.click()
name = "dog_preview_140-255_ero1_cc8_ROI_1"
wait(lambda: name in viewer.layers, what="preview layer")
pl = viewer.layers[name]
ok(same(pl.data, HTM.threshold_erode_cc_in_roi(dog, roi1, 140, None, 1, 8))
   and pl.metadata["sprout_threshold_preview"]["roi"]["name"] == "ROI_1",
   f"Create Preview Layer: '{name}' == threshold_erode_cc_in_roi, ROI name in metadata")
st.use_connected_components_checkbox.setChecked(False)
st.erosion_spin.setValue(0)

before = set(viewer.layers)
sw.generate_seeds()
hub.set_yx(l1, (0, 0), (50, 50))             # edited right after the click: the run keeps its snapshot
wait(lambda: sw.generate_btn.isEnabled(), what="seeds")
pump()
new = [l for l in viewer.layers if l not in before]
ref = quiet(R.seeds_in_roi, MS.make_seeds, roi1, dog, None, thresholds=[140, 160], upper_thresholds=[255, 255],
            erosion_steps=2, segments=6, output_folder=os.path.join(TMP, "ref_seeds"),
            num_threads=mp.thread_spin.value(), footprints=["ball"] * 2, base_name="dog_ROI_1",
            return_for_napari=True)[0]
got = {l.name: l for l in new}
ok(sorted(got) == sorted(R.roi_name(k, "ROI_1") for k in ref) and len(ref) == 6 and all(n.endswith("_ROI_1.tif") for n in got),
   f"Seeds: {len(new)} layers named *_ROI_1 ({sorted(got)[0]}, ...)")
ok(all(same(got[R.roi_name(k, 'ROI_1')].data, v) for k, v in ref.items()),
   "Seeds layers == seeds_in_roi with the ROI as clicked (edited during the run: snapshot kept)")
md0 = new[0].metadata.get(R.META_KEY, {})
ok(md0.get("name") == "ROI_1" and md0.get("bbox_start") == list(roi1.lo) and md0.get("bbox_stop") == list(roi1.hi)
   and md0.get("key_hash") == roi1.key_hash, f"output metadata: name + geometry ({md0.get('description')})")
od = Path(TMP) / "seeds" / "dog_ROI_1"
tifs = sorted(od.rglob("*.tif"))
side = json.loads((od / "roi.json").read_text())
ok(len(tifs) == 6 and all(tifffile.imread(t).shape == roi1.crop_shape(1) for t in tifs) and side["name"] == "ROI_1"
   and side["crop_offset"] == list(roi1.offset(1)), "disk: crop-sized TIFs in 'dog_ROI_1' + roi.json with the name")
hub.set_yx(l1, (20, 10), (140, 110))
pump_for(0.5)
idle(st, gt, pt)
roi1 = hub.roi_of(l1)

gw.advanced_params_box.sort_checkbox.setChecked(True)
# a Seeds run selects its last seeds layer in Grow (polish E4); this Grow run is on dog_seed
ok(gw.seeds_combo.currentText() == new[-1].name, f"Grow seeds after the Seeds run: {gw.seeds_combo.currentText()}")
gw.seeds_combo.setCurrentText("dog_seed")
before = set(viewer.layers)
run_grow(gw)
new = {l.name: l for l in viewer.layers if l not in before}
gp, ga = gw.main_param_widget.get_params(), gw.advanced_params_box.get_params()
ref, _ = quiet(R.grow_in_roi, MG.grow_mp, roi1, dog, dseed, None, base_name="dog_ROI_1", thresholds=gp["thresholds"],
               upper_thresholds=gp["upper_thresholds"], dilation_steps=gp["dilation_steps"],
               touch_rule=gp["touch_rule"], output_folder=os.path.join(TMP, "ref_grow"), num_threads=gp["num_threads"],
               save_every_n_iters=ga["save_every_n_iters"], grow_to_end=ga["grow_to_end"], is_sort=True,
               to_grow_ids=ga["to_grow_ids"], min_growth_size=ga["min_growth_size"],
               no_growth_max_iter=ga["no_growth_max_iter"], return_for_napari=True)
# the Grow layers are at least uint16 (option c); grow_in_roi keeps the smallest dtype
ok(sorted(new) == sorted(ref) and "FINAL_GROW_dog_ROI_1" in new
   and all(same(new[k].data, v.astype(np.promote_types(v.dtype, np.uint16))) for k, v in ref.items()),
   f"Grow in ROI_1 == grow_in_roi for every output (layers >= uint16): {sorted(new)}")
out = ~roi1.mask_full()
ok(all(np.array_equal(l.data[out], cast_labels(dseed)[out]) for l in new.values())
   and "Sort IDs is ignored" in gw.status_label.text(), "Grow: seeds outside unchanged; Sort IDs note")
gw.advanced_params_box.sort_checkbox.setChecked(False)

# Grow: seeds made in an ROI
seeds_roi_name = sorted(got)[-1]
gw.roi_select.select(None)
gw.refresh_layers()
gw.seeds_combo.setCurrentText(seeds_roi_name)
pump()
ok(gw.seed_roi_hint.text() == "Seeds were made in ROI_1" and gw.seed_roi_btn.text() == "Use ROI_1"
   and not gw.seed_roi_btn.isHidden(), "Grow: 'Seeds were made in ROI_1 [Use ROI_1]'")
gw.seed_roi_btn.click()
pump()
ok(gw.roi_select.layer is l1, "[Use ROI_1] picks it")
hub.set_z_range(l1, 40, 150)
pump()
ok(gw.seed_roi_hint.text() == f"ROI_1 changed since the seeds (was z 30{DASH}149 {DOT} y 20{DASH}139 {DOT} x 10{DASH}109)",
   f"warning when the ROI changed: {gw.seed_roi_hint.text()!r}")
hub.set_z_range(l1, 30, 150)
gw.seeds_combo.setCurrentText("dog_seed")
pump_for(0.5)
idle(st, gt, pt, bw)

# ======================================================================
print("\n== ROI tools")
main.tabs.setCurrentIndex(3)
main.roi_boundary_tabs.setCurrentIndex(0)
ok(rt.edit_layer is not None and rt.roi_combo.currentText() == rt.edit_layer.name and getattr(rt, "edit_box", rt).isEnabled()
   and getattr(rt, "empty_note", rt).isHidden(), "the tools edit an ROI layer; Edit ROI block enabled")
rt._pick(weakref.ref(l1))
pump()
ok(rt.info.text() == f"120 {X} 120 {X} 100 {DOT} 1.44 M voxels {DOT} 33%"
   and getattr(rt, "users_line", rt.info).text() == "Used by: Seeds, Grow, Prompt",
   f"two info lines: {rt.info.text()!r} / {getattr(rt, 'users_line', rt.info).text()!r}")
ok([(r[1].value(), r[2].value()) for r in rt.axis_rows.values()] == [(30, 149), (20, 139), (10, 109)],
   "from / to show the box (inclusive)")
viewer.dims.set_point(0, 60)
pump()
rt.axis_rows["z"][3][0].click()                 # z from = slice
pump_for(0.5)
ok(hub.roi_of(l1).lo[0] == 60, "z from '= slice' (60)")
rt.axis_rows["y"][1].setValue(40)
rt.axis_rows["y"][1].editingFinished.emit()
pump()
r = hub.roi_of(l1)
ok((r.lo, r.hi) == ((60, 40, 10), (150, 140, 110)), f"typing y from moves the rectangle: {r.lo}..{r.hi}")
l1.name = "lungs"                                # renamed in the napari layer list
pump()
ok("lungs" in combo_texts(sw.roi_select.combo) and rt.roi_combo.currentText() == "lungs"
   and sw.generate_btn.text() == "Generate Seeds in lungs" and hub.label_of(l1).startswith("lungs ")
   and rt.roi_combo.toolTip().startswith("lungs"), "rename in napari: combos, run button and canvas label follow")
l1.name = "ROI_1"
pump()
n = len(RL.roi_layers(viewer))
names0 = [l.name for l in RL.roi_layers(viewer)]
viewer.layers.selection.active = dlay
rt.image_combo.setCurrentIndex(rt.image_combo.findData("dog"))
edit0 = rt.edit_layer


def draw_layers(v=viewer):
    return [l for l in v.layers if l.name.startswith("SPROUT draw")]


def roi_names(v=viewer):
    return [l.name for l in RL.roi_layers(v)]


def all_combos():
    cs = [s_.combo for s_ in sels.values()] + [rt.roi_combo, rt.image_combo, rt.labels_combo]
    cs += [sw.image_combo, sw.boundary_combo, gw.image_combo, gw.seeds_combo, gw.boundary_combo, bw.image_combo,
           pt.image_combo, pt.seeds_combo]
    return cs


from napari_sprout.widgets.info_widget import LayerInfoWidget  # noqa: E402
iw = LayerInfoWidget(viewer)
rt.draw_btn.click()
pump()
td = draw_layers()
ok(roi_names() == names0 and len(td) == 1 and not RL.is_roi_layer(td[0]) and RL.MARKER not in td[0].metadata
   and str(td[0].mode) == "add_rectangle" and viewer.layers.selection.active is td[0],
   f"Draw box: no ROI layer yet, one temporary 'SPROUT draw' layer in add_rectangle mode ({roi_names()}, "
   f"{[l.name for l in td]})")
ok(td and td[0].ndim == 2 and np.allclose(td[0].scale, dlay.scale[-2:]) and np.allclose(td[0].translate, dlay.translate[-2:])
   and rt.edit_layer is edit0, "the draw layer: 2D, the image's y / x frame; the editor keeps its ROI")
rt._refresh_timer.start(0)
pump()
ok(all(not any(t.startswith("SPROUT draw") for t in combo_texts(c)) for c in all_combos()),
   "no SPROUT combo lists the draw layer")
iw.on_active_layer_changed()
pump()
ok("ROI" not in iw.info_box.toPlainText().splitlines()[1], f"Data Info: not an ROI ({iw.info_box.toPlainText().splitlines()[:2]})")
rt.draw_btn.click()
pump()
ok(len(draw_layers()) == 1 and roi_names() == names0 and viewer.layers.selection.active is draw_layers()[0]
   and str(draw_layers()[0].mode) == "add_rectangle", "Draw box twice: still one draw layer, no ROI")
(draw_layers() or [viewer.layers.selection.active])[0].add_rectangles(np.array([[30.0, 40.0]] * 4))           # a click, no drag
pump(40)
ok(roi_names() == names0 and len(draw_layers()) == 1 and len(draw_layers()[0].data) == 0
   and str(draw_layers()[0].mode) == "add_rectangle", "a click (no drag): no ROI, still drawing")
(draw_layers() or [viewer.layers.selection.active])[0].add_rectangles(RL.rect_corners((10, 20), (60, 70)))
pump(40)
ld = rt.edit_layer
ok(roi_names() == names0 + ["ROI_3"] and not draw_layers() and ld is not None and ld.name == "ROI_3"
   and rt.roi_combo.currentText() == "ROI_3", f"first rectangle: ROI_3 made, draw layer gone, ROI_3 in the editor "
                                              f"({roi_names()}, {[l.name for l in draw_layers()]})")
r3 = hub.roi_of(ld)
ok((r3.lo, r3.hi) == ((0, 10, 20), (220, 60, 70)) and str(ld.mode) == "pan_zoom" and viewer.layers.selection.active is dlay
   and not rt.z_note.isHidden(), f"drawn: z = all slices, pan_zoom, previous layer active again ({r3.lo}..{r3.hi})")
rt.axis_rows["z"][2].setValue(99)
rt.axis_rows["z"][2].editingFinished.emit()
pump()
ok(hub.roi_of(ld).hi[0] == 100 and rt.z_note.isHidden(), "setting the z range clears 'z: all'")
rt._delete()
pump()
ok(ld not in viewer.layers and len(RL.roi_layers(viewer)) == n, "Delete removes the layer")

# leaving the drawing in any way: no ROI layer, no draw layer, no ROI number used
order0 = tuple(viewer.dims.order)


def _ndisplay3():
    viewer.dims.ndisplay = 3


def _transpose():
    viewer.dims.order = (order0[1], order0[0], order0[2])


abandon = [("another mode", lambda l: setattr(l, "mode", "pan_zoom"), True),
           ("another active layer", lambda l: setattr(viewer.layers.selection, "active", olay), False),
           ("3D view", lambda l: _ndisplay3(), True),
           ("transposed view", lambda l: _transpose(), True),
           ("draw layer deleted", lambda l: viewer.layers.remove(l), False)]
for what, act, restores in abandon:
    viewer.layers.selection.active = dlay
    rt.draw_btn.click()
    pump()
    tl = draw_layers()
    if tl:
        act(tl[0])
    else:
        act(rt.edit_layer)
    pump(40)
    viewer.dims.ndisplay = 2
    viewer.dims.order = order0
    pump(40)
    ok(roi_names() == names0 and not draw_layers() and (not restores or viewer.layers.selection.active is dlay),
       f"abandon ({what}): nothing made, draw layer removed{', previous layer active' if restores else ''} "
       f"({roi_names()}, {[l.name for l in draw_layers()]})")
viewer.layers.selection.active = dlay
rt.draw_btn.click()
pump()
(draw_layers() or [rt.edit_layer])[0].add_rectangles(RL.rect_corners((5, 5), (25, 35)))
pump(40)
ok(roi_names() == names0 + ["ROI_3"] and not draw_layers(), f"next Draw box: ROI_3 (no number skipped): {roi_names()}")
viewer.layers.remove("ROI_3")
pump()

# deleting the rectangle of an existing ROI keeps that ROI layer
rt._pick(weakref.ref(l2))
pump()
l2.selected_data = {0}
l2.remove_selected()
pump(40)
ok(l2 in viewer.layers and RL.is_roi_layer(l2) and len(l2.data) == 0 and rt.edit_layer is l2
   and rt.info.text() == "ROI_2: no box yet — drag a rectangle on the 2D view" and not draw_layers(),
   f"rectangle deleted from ROI_2: the layer stays, 'no box yet' ({rt.info.text()!r})")
hub.set_yx(l2, (0, 0), (10, 10))
pump()
ok(hub.roi_of(l2).key == Roi.box(olay.data.shape, (0, 0, 0), (10, 10, 10)).key, "ROI_2 box back")
iw.deleteLater()
rt.whole_btn.click()
pump()
lw = rt.edit_layer
ok(hub.roi_of(lw).key == Roi.box(dog.shape, (0, 0, 0), dog.shape).key, "Whole image")
viewer.layers.remove(lw)
labs = np.zeros(dog.shape, np.uint8)
labs[40:60, 30:50, 20:40] = 4
labs[100:110, 90:100, 50:60] = 5
labs[200:220, 0:10, 0:10] = 9
ll = viewer.add_labels(labs, name="regions")
pump()
rt.labels_combo.setCurrentIndex(rt.labels_combo.findData("regions"))
rt.ids_edit.setText("4,5")
rt.margin_spin.setValue(5)
rt.create_btn.click()
wait(lambda: rt.create_btn.isEnabled() and rt.create_btn.text() in ("Create ROI", "Create"), timeout=30)
pump()
lf = rt.edit_layer
rf = hub.roi_of(lf)
ok((rf.lo, rf.hi) == ((35, 25, 15), (115, 105, 65)) and lf.name.startswith("ROI_"),
   f"From labels ids 4,5 + margin 5 -> {lf.name} {rf.lo}..{rf.hi}")
rt.ids_edit.setText("99")
rt.create_btn.click()
wait(lambda: rt.create_btn.isEnabled() and rt.create_error.text(), timeout=30)
ok(rt.create_error.text() == "label 99 does not occur in 'regions'", f"error inline: {rt.create_error.text()!r}")
rt.ids_edit.setText("4,x")
rt.create_btn.click()
ok(rt.create_error.text() == "ids: enter numbers like 3,5-7 (got '4,x')", "bad ids text")

# save / load
fn = os.path.join(TMP, "rois.json")
rt.save_rois(fn)
saved = json.loads(Path(fn).read_text())
ok(saved["schema"] == 1 and [r["name"] for r in saved["rois"]] == [l.name for l in RL.roi_layers(viewer)]
   and saved["rois"][0]["ref_image"] == "dog", f"Save ROIs: {[r['name'] for r in saved['rois']]}")
for l in RL.roi_layers(viewer):
    viewer.layers.remove(l)
pump()
ok(not getattr(rt, "edit_box", rt).isEnabled() and not getattr(rt, "empty_note", rt).isHidden()
   and rt.info.isHidden() and all(r[1].isHidden() for r in rt.axis_rows.values()),
   "all ROI layers removed: Edit ROI disabled again, only the grey line")
made = rt.load_rois(fn)
pump()
ok(getattr(rt, "edit_box", rt).isEnabled() and getattr(rt, "empty_note", rt).isHidden() and not rt.info.isHidden(),
   "loaded: Edit ROI enabled")
pump()
ok([l.name for l in made] == [r["name"] for r in saved["rois"]]
   and all(hub.roi_of(l).key == Roi.from_metadata(r).key for l, r in zip(made, saved["rois"])),
   "Load ROIs: same names and boxes")
l1 = next(l for l in made if l.name == "ROI_1")
viewer.layers.remove(l1)
pump()
(back,) = rt.load_rois(str(Path(TMP) / "seeds" / "dog_ROI_1" / "roi.json"))
pump()
ok(hub.roi_of(back).key == Roi.box(dog.shape, (30, 20, 10), (150, 140, 110)).key and back.name == "ROI_1",
   "Load ROIs accepts a run's roi.json: that run's ROI back")
l1 = back
v_before = viewer.dims.ndisplay
viewer.dims.ndisplay = 3
pump()
ok(not rt.view_note.isHidden() and rt.view_note.text() == RL.HIDDEN_NOTE and not l1.visible, "3D view: note + boxes hidden")
viewer.dims.ndisplay = 2
pump()
ok(rt.view_note.isHidden() and l1.visible, "2D again")
for s_ in sels.values():
    s_.select(None)
pump_for(0.5)
idle(st, gt, pt, bw)
main.tabs.setCurrentIndex(0)
viewer.close()

# ======================================================================
print("\n== lifecycle")
v = napari.Viewer(show=False)
img_ = (np.random.default_rng(0).random((40, 60, 80)) * 255).astype(np.uint8)
il = v.add_image(img_, name="img")
mw = SPROUTWidget(v)
v.window.add_dock_widget(mw)
pump_for(1.0)
lr = RL.add_roi_layer(v, il, (5, 5, 5), (20, 30, 40))
mw.seed_widget.roi_select.select(lr)
pump_for(0.5)
n0 = len(UNCAUGHT)
v.layers.selection.active = il
mw.roi_tools.draw_btn.click()                    # close the panel while drawing
pump()
ok(len(draw_layers(v)) == 1 or len(RL.roi_layers(v)) == 2, "drawing when the panel closes")
dw = [d for d in v.window._dock_widgets.values() if d.widget() is mw][0]
v.window.remove_dock_widget(dw)
del mw, dw
gc.collect()
pump_for(1.0)
hub = RL.get_roi_hub(v)
ok(not draw_layers(v) and len(RL.roi_layers(v)) == 1 and getattr(hub, "_draw", None) is None,
   f"closed panel: draw layer removed, nothing made ({[l.name for l in v.layers]})")
steps_draw = [("select a layer", lambda: setattr(v.layers.selection, "active", lr)),
              ("a user Shapes layer", lambda: v.add_shapes(RL.rect_corners((1, 1), (5, 5)), name="mine")),
              ("rectangle on it", lambda: v.layers["mine"].add_rectangles(RL.rect_corners((2, 2), (6, 6)))),
              ("its mode", lambda: setattr(v.layers["mine"], "mode", "add_rectangle"))]
steps = [("page z", lambda: v.dims.set_current_step(0, 5)),
         ("add layer", lambda: v.add_image(img_[::2], name="img2")),
         ("remove layer", lambda: v.layers.remove("img2")),
         ("ndisplay 3", lambda: setattr(v.dims, "ndisplay", 3)),
         ("ndisplay 2", lambda: setattr(v.dims, "ndisplay", 2)),
         ("edit the ROI", lambda: hub.set_yx(lr, (1, 1), (9, 9))),
         ("drag the ROI", lambda: (lr._data_view.shift(0, np.array([1.0, 1.0])), lr.events.set_data(),
                                   lr.events.data(value=lr.data, action="changed", data_indices=(0,),
                                                  vertex_indices=((),)))),
         ("rename the ROI", lambda: setattr(lr, "name", "kept")),
         ("new ROI", lambda: RL.add_roi_layer(v, il, (0, 0, 0), (3, 3, 3)))]
raised = []
for what, fn_ in steps_draw + steps:
    try:
        fn_()
        pump_for(0.3)
    except Exception as e:
        raised.append(f"{what}: {type(e).__name__}: {e}")
errs = UNCAUGHT[n0:]
ok(not raised and not errs, f"after closing the SPROUT dock: paging / layers / 3D / ROI edits raise nothing "
                            f"({len(raised)} raised, {len(errs)} uncaught)"
   + (f": {(raised or errs)[0][-300:]}" if raised or errs else ""))
ok(lr in v.layers and RL.is_roi_layer(lr) and len(RL.roi_layers(v)) == 2 and not draw_layers(v),
   "ROI layers are user data: they stay")
ok(hub.used_by(lr) == [], "the closed panel's ROI pickers no longer count as users")
mw2 = SPROUTWidget(v)
pump_for(0.5)
ok(combo_texts(mw2.seed_widget.roi_select.combo) == ["None", "kept", "ROI_1"] and mw2.roi_tools.edit_layer is not None,
   "reopened panel finds the ROI layers again (by their metadata marker)")
v.close()

# ======================================================================
print("\n== anisotropic image, 2D image")
va = napari.Viewer(show=False)
sc, tr = (2.0, 0.5, 0.5), (10.0, 0.0, 0.0)
vol2 = (np.random.default_rng(5).random((40, 60, 80)) * 255).astype(np.uint8)
alay = va.add_image(vol2, name="aniso", scale=sc, translate=tr)
va.add_labels(dseed[:40, :60, :80].astype(np.uint8), name="seed_a", scale=sc, translate=tr)
pump()
main_a = SPROUTWidget(va)
la = RL.add_roi_layer(va, alay, (5, 10, 10), (30, 50, 70))
swa, gwa = main_a.seed_widget, main_a.grow_widget
swa.image_combo.setCurrentText("aniso")
gwa.image_combo.setCurrentText("aniso")
gwa.seeds_combo.setCurrentText("seed_a")
swa.output_folder_line.setText(os.path.join(TMP, "a_seeds"))
gwa.output_folder_line.setText(os.path.join(TMP, "a_grow"))
swa.main_param_widget.populate_thresholds_from_list([140], [255])
swa.main_param_widget.erosion_spin.setValue(1)
gwa.main_param_widget.populate_thresholds_from_list([150], [255], [2])
swa.roi_select.select(la)
gwa.roi_select.select(la)
pump()
ok(np.allclose(la.scale, sc[1:]) and np.allclose(la.translate, tr[1:]), "anisotropic ROI layer: the image's y / x frame")
before = set(va.layers)
run_seeds(swa)
run_grow(gwa)
new = [l for l in va.layers if l not in before and not l.name.startswith("Threshold Preview")]
ra = RL.get_roi_hub(va).roi_of(la)
ok(len(new) > 2 and all("_ROI_1" in l.name and np.allclose(l.scale, sc) and np.allclose(l.translate, tr) for l in new)
   and all(l.metadata[R.META_KEY]["bbox_start"] == list(ra.lo) for l in new),
   "anisotropic: *_ROI_1 layers with the image's scale / translate, the ROI's voxel box in the metadata")
idle(main_a.boundary_widget, swa.threshold_widget, gwa.threshold_widget)
va.close()

v2 = napari.Viewer(show=False)
cell = tifffile.imread(ROOT / "demo_data" / "2d_cell.tif")[:800, :800]
c2 = v2.add_image(cell, name="cell")
pump()
main_2 = SPROUTWidget(v2)
s2 = main_2.seed_widget
s2.output_folder_line.setText(os.path.join(TMP, "seeds2d"))
s2.main_param_widget.populate_thresholds_from_list([120], [255])
s2.main_param_widget.erosion_spin.setValue(1)
s2.main_param_widget.footprint_table.setRowCount(0)          # preset 2D footprints
l2d = RL.add_roi_layer(v2, c2, (100, 150), (600, 700))
s2.roi_select.select(l2d)
pump()
ok(s2.roi_select.info.text() == f"y 100{DASH}599 {DOT} x 150{DASH}699\n500 {X} 550 = 275,000 voxels (43%)",
   f"2D info: {s2.roi_select.info.text()!r}")
r2 = RL.get_roi_hub(v2).roi_of(l2d)
before = set(v2.layers)
run_seeds(s2)
new = {l.name: l for l in v2.layers if l not in before and not l.name.startswith("Threshold Preview")}
ref = quiet(R.seeds_in_roi, MS.make_seeds, r2, cell, None, thresholds=[120], upper_thresholds=[255], erosion_steps=1,
            segments=s2.main_param_widget.segments_spin.value(), output_folder=os.path.join(TMP, "ref2d"),
            num_threads=s2.main_param_widget.thread_spin.value(), footprints=None, base_name="cell_ROI_1",
            return_for_napari=True)[0]
ok(len(new) == len(ref) > 0 and all(same(new[R.roi_name(k, "ROI_1")].data, v_) for k, v_ in ref.items()),
   f"2D image: Seeds in a box ROI == seeds_in_roi ({len(new)} layers)")
idle(main_2.boundary_widget, s2.threshold_widget, main_2.grow_widget.threshold_widget)
v2.close()
ok(not UNCAUGHT, f"no uncaught errors ({len(UNCAUGHT)})" + (f": {UNCAUGHT[0][-400:]}" if UNCAUGHT else ""))
sys.excepthook = sys.__excepthook__
finish()
