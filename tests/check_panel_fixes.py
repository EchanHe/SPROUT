"""
Headless check of the PANEL fixes from the napari-sprout audit (Seeds / Grow / Prompt /
ROI & Boundary tabs):

    A6   Grow "IDs to grow": 3,5-7 ranges; invalid input -> red line, no run; empty = all
    ren  a renamed layer is renamed in place in the Prompt, ROI tools and Boundary combos
    hub  the panel connects nothing to napari's own events (all through the RoiHub)
    heal each tab re-syncs its layer combos when it is shown (a missed event is repaired)
    D2   hidden tabs do no work: Boundary input histogram, Prompt class counts, the live
         slice of a hidden Threshold Preview; done when the tab is shown
    D3   Boundary probes the GPU only when GPU is picked (or on Compute), once
    E1   no empty status line takes space
    E6   advanced Seeds labels are not truncated
    E7   labels without colons, sentence case; Prompt "Browse"
    E9   tooltips (Method, Segments, Touch rule, CC N, Save mid results, All / None,
         positive / negative, Stop); "Preview erosions"
    E10  Boundary: YAML buttons in the Method group header; Compute styled like Generate Seeds
    E11  a disabled Stop looks like any disabled button
    E13  keyboard Tab order follows the layout (YAML buttons); grey hints >= 4.5:1

SPROUT_ROOT=<folder> runs the same checks against another copy of the code (e.g. the
base commit, to see them fail). Run in the `sprout` env:
    python tests/check_panel_fixes.py
Prints PASS/FAIL per check; exits non-zero if anything failed.
"""
import os
import sys
import time
import traceback
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.path.isdir(r"C:/Windows/Fonts"):
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:/Windows/Fonts")
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ROOT = Path(os.environ.get("SPROUT_ROOT") or REPO)
sys.path[:0] = [str(ROOT), str(ROOT / "napari_sprout"), str(HERE)]

import napari  # noqa: E402
import _headless_gl  # noqa: E402,F401
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from napari.qt import get_stylesheet  # noqa: E402
from qtpy.QtCore import Qt  # noqa: E402
from qtpy.QtGui import QColor  # noqa: E402
from qtpy.QtWidgets import (QAbstractButton, QApplication, QComboBox, QFormLayout, QLabel,  # noqa: E402
                            QPushButton, QWidget)

import napari_sprout  # noqa: E402
from napari_sprout._widget import SPROUTWidget  # noqa: E402
import napari_sprout.widgets.boundary_widget as BWM  # noqa: E402
import napari_sprout.widgets.prompt_tab as PTM  # noqa: E402
from napari_sprout.utils import status as S  # noqa: E402
from napari_sprout.utils.roi_layers import get_roi_hub  # noqa: E402

print(f"napari {napari.__version__} | napari_sprout from {Path(napari_sprout.__file__).parent}")
PTM.SETTINGS = ("SPROUT-checks", "napari-sprout-checks")      # not the user's settings
FAILED = []
UNCAUGHT = []


def _hook(*a):
    UNCAUGHT.append("".join(traceback.format_exception(*a)))
    print(UNCAUGHT[-1], flush=True)


sys.excepthook = _hook


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILED.append(msg)


def safe(fn, default=None):
    try:
        return fn()
    except Exception as e:                     # a missing attribute on the base commit = FAIL, not a crash
        return default if default is not None else f"<{type(e).__name__}: {e}>"


def pump(n=20):
    for _ in range(n):
        QApplication.processEvents()
        time.sleep(0.005)


def pump_for(sec):
    t = time.time()
    while time.time() - t < sec:
        QApplication.processEvents()
        time.sleep(0.005)


def wait(cond, timeout=60, what=""):
    t = time.time()
    while not cond():
        QApplication.processEvents()
        time.sleep(0.005)
        if time.time() - t > timeout:
            print(f"     (timeout: {what})")
            return False
    return True


def items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def datas(combo):
    return [combo.itemData(i) for i in range(combo.count())]


SS = get_stylesheet("dark")
dog = tifffile.imread(REPO / "demo_data" / "dog_img.tif")[200:300]
dseed = tifffile.imread(REPO / "demo_data" / "dog_seed.tif")[200:300]


def shown_panel(viewer):
    main = SPROUTWidget(viewer)
    main.setStyleSheet(SS)
    main.resize(500, 1000)
    main.show()
    pump(40)
    return main


def show_tab(main, w):
    if w in (main.roi_tools, main.boundary_widget):
        main.tabs.setCurrentWidget(main.roi_boundary_tabs)
        main.roi_boundary_tabs.setCurrentWidget(w)
    else:
        main.tabs.setCurrentWidget(w)
    pump(10)


# ======================================================================
print("== A6 Grow: IDs to grow")
v = napari.Viewer(show=False)
small = (np.random.default_rng(0).random((12, 40, 40)) * 255).astype(np.uint8)
sm_seed = np.zeros(small.shape, np.uint8)
sm_seed[4:8, 5:15, 5:15], sm_seed[4:8, 20:30, 20:30] = 1, 2
v.add_image(small, name="small")
v.add_labels(sm_seed, name="small_seed")
main = SPROUTWidget(v)
gw = main.grow_widget
box = gw.advanced_params_box
gw.image_combo.setCurrentText("small")
gw.seeds_combo.setCurrentText("small_seed")
gw.output_folder_line.setText(str(Path(os.environ.get("TEMP", ".")) / "sprout_panel_fixes_grow"))
gw.main_param_widget.populate_thresholds_from_list([100], [255], [2])
for text, want in (("3,5-7", [3, 5, 6, 7]), (" 3, 5 - 7 ", [3, 5, 6, 7]), ("", None), ("4", [4])):
    box.id_list_line.setText(text)
    got = safe(lambda: box.get_params()["to_grow_ids"])
    ok(got == want, f"'{text}' -> {got!r} (want {want!r})")


def ids_error_line():
    """The visible red line(s) of the advanced Grow box (or of the tab)."""
    return [lab.text() for lab in gw.findChildren(QLabel)
            if lab.text() and S.level_of(lab) == "error" and lab.isVisibleTo(gw)]


gw.show_advanced_checkbox.setChecked(True)
for bad in ("1,a,3", "1 2 3", "3-x"):
    box.id_list_line.setText(bad)
    gw.worker = None
    safe(gw.start_growth)
    started = gw.worker is not None
    if started:
        wait(lambda: not gw.worker.isRunning(), what="grow run")
    pump()
    errs = ids_error_line()
    ok(not started and any("IDs" in e or "ids" in e for e in errs),
       f"'{bad}': run refused, red line {errs} (run started: {started})")
box.id_list_line.setText("1")
ok(not ids_error_line() or safe(lambda: box.ids_error.isHidden(), True), "a valid entry clears the red line")
wait(lambda: not gw.worker or not gw.worker.isRunning(), what="grow idle")
v.close()

# ======================================================================
print("\n== renamed layers: renamed in place in every combo (selection kept)")
v = napari.Viewer(show=False)
v.add_image(dog, name="img_a")
v.add_labels(dseed, name="lab_a")
v.add_image(dog[:, ::2, ::2].copy(), name="img_b")
main = SPROUTWidget(v)
pump(20)
pt, rt, bw = main.prompt_tab, main.roi_tools, main.boundary_widget
pt.image_combo.setCurrentText("img_a")
pt.seeds_combo.setCurrentText("lab_a")
pt.init_combo.setCurrentIndex(pt.init_combo.findData("lab_a"))
rt.image_combo.setCurrentIndex(rt.image_combo.findData("img_a"))
bw.image_combo.setCurrentText("img_a")
pump(20)
v.layers["img_a"].name = "img_renamed"
v.layers["lab_a"].name = "lab_renamed"
pump(20)
ok(items(pt.image_combo) == ["img_renamed", "img_b"] and pt.image_combo.currentText() == "img_renamed",
   f"Prompt image combo {items(pt.image_combo)} current {pt.image_combo.currentText()!r}")
ok(items(pt.seeds_combo) == ["lab_renamed"] and pt.seeds_combo.currentText() == "lab_renamed",
   f"Prompt seeds combo {items(pt.seeds_combo)}")
ok(items(pt.init_combo) == ["None", "lab_renamed"] and pt.init_combo.currentData() == "lab_renamed",
   f"Prompt initial mask combo {items(pt.init_combo)} data {pt.init_combo.currentData()!r}")
ok(datas(rt.image_combo) == ["img_renamed", "img_b"] and rt.image_combo.currentData() == "img_renamed",
   f"ROI tools image combo {datas(rt.image_combo)} current {rt.image_combo.currentData()!r}")
ok(datas(rt.labels_combo) == ["lab_renamed"], f"ROI tools labels combo {datas(rt.labels_combo)}")
ok(items(bw.image_combo) == ["img_renamed", "img_b"] and bw.image_combo.currentText() == "img_renamed",
   f"Boundary image combo {items(bw.image_combo)} current {bw.image_combo.currentText()!r}")
ok(pt.image_layer is v.layers["img_renamed"] and bw._img_layer is v.layers["img_renamed"],
   "the tabs still use the same layer objects")
v.close()

# ======================================================================
print("\n== napari connections: only through the RoiHub")
v = napari.Viewer(show=False)
v.add_image(dog, name="img")
get_roi_hub(v)
EMS = {"layers.inserted": v.layers.events.inserted, "layers.removed": v.layers.events.removed,
       "dims.current_step": v.dims.events.current_step, "dims.ndisplay": v.dims.events.ndisplay}
before = {k: len(e.callbacks) for k, e in EMS.items()}
main = SPROUTWidget(v)
pump()
grown = {k: len(e.callbacks) - before[k] for k, e in EMS.items()}
ok(not any(grown.values()), f"a SPROUT panel adds no napari callbacks of its own: {grown}")
v.close()

# ======================================================================
print("\n== self-healing: a tab re-syncs its combos when shown")
v = napari.Viewer(show=False)
v.add_image(dog, name="img")
v.add_labels(dseed, name="lab")
main = shown_panel(v)
hub = get_roi_hub(v)
hub.blockSignals(True)                    # the panel misses these events
v.add_image(dog[:, ::2, ::2].copy(), name="missed_img")
v.add_labels(dseed.copy(), name="missed_lab")
v.layers["lab"].name = "lab_renamed"
pump(10)
hub.blockSignals(False)
sw, gw, pt, rt, bw = main.seed_widget, main.grow_widget, main.prompt_tab, main.roi_tools, main.boundary_widget
EXP_I, EXP_L = ["img", "missed_img"], ["lab_renamed", "missed_lab"]
for w, combos in ((gw, lambda: [(items(gw.image_combo), EXP_I), (items(gw.seeds_combo), EXP_L),
                                (items(gw.boundary_combo), ["None"] + EXP_L)]),
                  (pt, lambda: [(items(pt.image_combo), EXP_I), (items(pt.seeds_combo), EXP_L),
                                (items(pt.init_combo), ["None"] + EXP_L)]),
                  (rt, lambda: [(datas(rt.image_combo), EXP_I), (datas(rt.labels_combo), EXP_L)]),
                  (bw, lambda: [(items(bw.image_combo), EXP_I)]),
                  (sw, lambda: [(items(sw.image_combo), EXP_I), (items(sw.boundary_combo), ["None"] + EXP_L)])):
    show_tab(main, w)
    got = combos()
    ok(all(g == e for g, e in got), f"{type(w).__name__} shown: {[g for g, _ in got]}")
ok(sw.image_combo.currentText() == "img", "selection kept")
v.close()

# ======================================================================
print("\n== D2 hidden tabs do no work")
v = napari.Viewer(show=False)
main = shown_panel(v)
sw, gw, pt, bw = main.seed_widget, main.grow_widget, main.prompt_tab, main.boundary_widget
calls = {"stats": 0}
_cs = PTM.PP.class_stats


def _count_stats(*a, **k):
    calls["stats"] += 1
    return _cs(*a, **k)


PTM.PP.class_stats = _count_stats
show_tab(main, sw)
v.add_image(dog, name="dog")
v.add_labels(dseed, name="dog_seed")
pump_for(1.5)
ok(not bw.input_hist._ready() and not bw._workers,
   f"Boundary hidden: no input histogram ({bw.input_hist._message!r})")
ok(calls["stats"] == 0, f"Prompt hidden: no class counts ({calls['stats']} calls)")
show_tab(main, bw)
ok(wait(lambda: bw.input_hist._ready(), what="boundary hist"), "Boundary shown: input histogram computed")
show_tab(main, pt)
ok(wait(lambda: pt.table.rowCount() > 0, what="prompt table"),
   f"Prompt shown: class counts ({pt.table.rowCount()} classes, {calls['stats']} call)")
PTM.PP.class_stats = _cs
# a hidden Threshold Preview does not follow paging
show_tab(main, gw)
gt = gw.threshold_widget
wait(lambda: gt.hist._ready(), what="grow hist")
gt.hist.set_range(100, gt.hist._xmax)
pump()
glay = gt._preview.layer
z = int(round(v.dims.point[0]))
ok(glay is not None and glay.data[z].any(), f"Grow live slice drawn at z={z}")
show_tab(main, sw)
v.dims.set_point(0, z + 7)
pump()
ok(glay.data[z].any() and not glay.data[z + 7].any(), "Grow hidden: paging does not redraw its live slice")
show_tab(main, gw)
pump()
ok(glay.data[z + 7].any() and not glay.data[z].any(), "Grow shown again: its live slice follows the current z")
wait(lambda: not bw._workers and not pt._workers and not gt._workers, what="idle")
v.close()

# ======================================================================
print("\n== D3 Boundary: GPU probed only when needed, once")
probes = []
_gs = BWM.B.gpu_status


def _fake_gpu():
    probes.append(1)
    return False, "fake: no GPU"


BWM.B.gpu_status = _fake_gpu
if hasattr(BWM, "_GPU_STATUS"):
    BWM._GPU_STATUS = None
v = napari.Viewer(show=False)
v.add_image(dog, name="dog")
main = shown_panel(v)
bw = main.boundary_widget
show_tab(main, bw)
pump_for(1.0)
ok(len(probes) == 0, f"showing the Boundary tab probes no GPU ({len(probes)} probes)")
bw.device_combo.setCurrentIndex(bw.device_combo.findData("cpu"))
pump_for(0.3)
ok(len(probes) == 0, "picking CPU probes nothing")
gi = bw.device_combo.findData("gpu")
ok(bw.device_combo.model().item(gi).isEnabled(), "GPU can be picked before any probe")
bw.device_combo.setCurrentIndex(gi)
wait(lambda: probes and not bw._workers, what="probe")
pump()
ok(len(probes) == 1 and "fake: no GPU" in bw.device_label.text(),
   f"picking GPU probes once: {bw.device_label.text()!r}")
ok(bw.device_combo.currentData() == "auto" and not bw.device_combo.model().item(gi).isEnabled(),
   f"no GPU: back to {bw.device_combo.currentData()!r}, GPU greyed out")
bw.device_combo.setCurrentIndex(bw.device_combo.findData("cpu"))
bw.device_combo.setCurrentIndex(bw.device_combo.findData("auto"))
pump_for(0.3)
ok(len(probes) == 1, f"cached: no second probe ({len(probes)})")
BWM.B.gpu_status = _gs
v.close()

# ======================================================================
print("\n== E1 / E6 / E7 / E9 / E10 / E11 / E13 layout")
v = napari.Viewer(show=False)
v.add_image(dog, name="dog_img")
v.add_labels(dseed, name="dog_seed")
main = shown_panel(v)
sw, gw, pt, rt, bw = main.seed_widget, main.grow_widget, main.prompt_tab, main.roi_tools, main.boundary_widget
TABS = {"Seeds": sw, "Grow": gw, "Prompt": pt, "ROI": rt, "Boundary": bw}


def own_row(lab):
    """A label alone on its row (no visible sibling beside it): empty, it is a blank line."""
    if lab.height() <= 0 or isinstance(lab.parentWidget(), QAbstractButton):
        return False
    y0, y1 = lab.y(), lab.y() + lab.height()
    return not any(sib is not lab and isinstance(sib, QWidget) and sib.isVisibleTo(lab.parentWidget())
                   and sib.y() < y1 and sib.y() + sib.height() > y0
                   for sib in lab.parentWidget().children())


for name, w in TABS.items():
    show_tab(main, w)
    if w is bw:
        wait(lambda: bw.input_hist._ready(), what="boundary hist")
    if w is pt:
        wait(lambda: pt.table.rowCount() > 0, what="prompt table")
    pump(20)
    empty = [f"{type(lab).__name__}@{lab.parentWidget().__class__.__name__}:y={lab.y()}"
             for lab in w.findChildren(QLabel) if lab.isVisibleTo(w) and not lab.text() and own_row(lab)]
    ok(not empty, f"E1 {name}: no empty line takes vertical space ({len(empty)}: {empty[:6]})")

# E6: advanced Seeds labels
show_tab(main, sw)
sw.show_advanced_checkbox.setChecked(True)
pump(20)
ab = sw.advanced_params_box
trunc = [(lab.text(), lab.width(), lab.fontMetrics().horizontalAdvance(lab.text()))
         for lab in ab.findChildren(QLabel) if lab.isVisibleTo(ab) and lab.text()
         and lab.fontMetrics().horizontalAdvance(lab.text()) > lab.width()]
ok(not trunc, f"E6 advanced Seeds labels fit (text, width, needed): {trunc}")
sw.show_advanced_checkbox.setChecked(False)
gw.show_advanced_checkbox.setChecked(True)
pump()

# E7: labels without colons, sentence case
ACRONYM_OK = {"ROI", "YAML", "IDs", "ID", "N", "CC", "GPU", "CPU", "Z"}


def form_label_texts(w):
    out = []
    for form in w.findChildren(QFormLayout):
        for r in range(form.rowCount()):
            it = form.itemAt(r, QFormLayout.LabelRole)
            if it is not None and isinstance(it.widget(), QLabel) and it.widget().text():
                out.append(it.widget().text())
    return out


def caption_texts(w):
    """Short, single-line labels (captions), not status lines / hints."""
    return [lab.text() for lab in w.findChildren(QLabel)
            if lab.text() and not lab.wordWrap() and len(lab.text()) <= 40 and "\n" not in lab.text()
            and S.level_of(lab) is None]


bad_colon, bad_case = [], []
for name, w in TABS.items():
    texts = set(form_label_texts(w)) | set(caption_texts(w))
    for t in sorted(texts):
        if t.rstrip().endswith(":"):
            bad_colon.append(f"{name}: {t!r}")
        words = t.replace("(", " ").replace(")", " ").split()
        if any(wd[:1].isupper() and not wd.isupper() and wd not in ACRONYM_OK for wd in words[1:]):
            bad_case.append(f"{name}: {t!r}")
ok(not bad_colon, f"E7 no label ends with ':' ({bad_colon})")
ok(not bad_case, f"E7 labels in sentence case ({bad_case})")
ok(pt.browse_btn.text() == "Browse", f"E7 Prompt weights button: {pt.browse_btn.text()!r}")

# E9 tooltips
mp, gp, st = sw.main_param_widget, gw.main_param_widget, sw.threshold_widget
tips = {"Seeds Method": mp.seed_method_combo, "Seeds Segments": mp.segments_spin,
        "Grow Touch rule": gp.touch_rule_combo, "Seeds CC N": st.connected_components_spin,
        "Seeds Save mid results": sw.save_every_iter_checkbox, "Prompt All": pt.all_btn, "Prompt None": pt.none_btn,
        "Prompt positive": pt.pos_rb, "Prompt negative": pt.neg_rb, "Seeds Stop": sw.stop_btn,
        "Grow Stop": gw.stop_btn, "Prompt Stop": pt.stop_btn, "Boundary Stop": bw.stop_btn,
        "Seeds preview erosions": st.erosion_spin}
missing = [k for k, wdg in tips.items() if not wdg.toolTip().strip()]
ok(not missing, f"E9 tooltips present ({missing})")
ero_labels = [lab for lab in st.findChildren(QLabel) if "rosion" in lab.text() and not lab.wordWrap()]
ok([lab.text() for lab in ero_labels] == ["Preview erosions"] and "preview" in ero_labels[0].toolTip().lower(),
   f"E9 'Preview erosions' with a tooltip: {[(lab.text(), lab.toolTip()) for lab in ero_labels]}")

# E10 Boundary: YAML buttons in the Method group header, Compute like the other run buttons
show_tab(main, bw)
pump()
g2 = bw.method_combo.parentWidget()
ok(bw.import_btn.parentWidget() is g2 and bw.export_btn.parentWidget() is g2
   and bw.import_btn.mapTo(g2, bw.import_btn.rect().topLeft()).y()
   < bw.method_combo.mapTo(g2, bw.method_combo.rect().topLeft()).y(),
   f"E10 Import / Export YAML in the header row of {safe(lambda: g2.title())!r}")
ok(bw.compute_btn.styleSheet() == sw.generate_btn.styleSheet() != "",
   f"E10 Compute styled like Generate Seeds ({bw.compute_btn.styleSheet()!r})")


# E11 a disabled Stop looks like a disabled button
def mean_rgb(btn):
    img = btn.grab().toImage()
    w_, h_ = img.width(), img.height()
    px = [QColor(img.pixel(x, y)) for x in range(4, w_ - 4, 3) for y in range(4, h_ - 4, 3)]
    return np.mean([[c.red(), c.green(), c.blue()] for c in px], axis=0)


show_tab(main, sw)
pump()
ref = QPushButton("Stop", sw.stop_btn.parentWidget())
ref.setEnabled(False)
ref.resize(sw.stop_btn.size())
ref.show()
pump()
for name, btn in (("Seeds Stop", sw.stop_btn), ("Seeds Generate (running)", sw.generate_btn)):
    was = btn.isEnabled()
    btn.setEnabled(False)
    pump()
    d = float(np.abs(mean_rgb(btn) - mean_rgb(ref)).max())
    ok(d < 12, f"E11 {name} disabled looks disabled (colour difference {d:.0f} from a plain disabled button)")
    btn.setEnabled(was)
ref.deleteLater()
sw.stop_btn.setEnabled(True)
pump()
ok(mean_rgb(sw.stop_btn)[0] > 150, f"E11 an enabled Stop is still red {mean_rgb(sw.stop_btn).round()}")
sw.stop_btn.setEnabled(False)


# E13 tab order follows the layout; grey hints readable
def focus_chain(top):
    out, w = [], top.nextInFocusChain()
    while w is not top and len(out) < 2000:
        if w.focusPolicy() & Qt.TabFocus and top.isAncestorOf(w):
            out.append(w)
        w = w.nextInFocusChain()
    return out


for name, w, first in (("Seeds", sw, sw.main_param_widget.seed_method_combo),
                       ("Grow", gw, gw.main_param_widget.thread_spin),
                       ("Boundary", bw, bw.method_combo),
                       ("Prompt", pt, pt.scribble_rb)):
    chain = focus_chain(w)
    pos = {k: (chain.index(x) if x in chain else -1) for k, x in
           (("import", w.import_btn), ("export", w.export_btn), ("first", first))}
    ok(-1 < pos["import"] < pos["export"] < pos["first"],
       f"E13 {name}: Tab reaches Import / Export YAML before the parameters below them {pos}")


def lum(c):
    def ch(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    c = QColor(c)
    return 0.2126 * ch(c.red()) + 0.7152 * ch(c.green()) + 0.0722 * ch(c.blue())


bg = QColor("#262930")                      # napari dark background
cr = (lum(S.INFO) + 0.05) / (lum(bg) + 0.05)
ok(cr >= 4.5, f"E13 grey hints {S.INFO!r}: contrast {cr:.1f}:1 on napari dark (>= 4.5)")

v.close()

print(f"\nuncaught exceptions: {len(UNCAUGHT)}")
ok(not UNCAUGHT, "no uncaught exception")
print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
sys.exit(1 if FAILED else 0)
