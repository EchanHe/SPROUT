"""
Headless check of the UI-polish round, panel side (Seeds / Grow / Boundary / Prompt tabs).

    E4    a successful Seeds run selects its last seeds layer in Grow's seeds combo, with a grey
          "Seeds: <name> (last Seeds run)" line; not after Stop, a failure or another Labels
          layer; the user can pick another layer (the line goes)
    E12   a new image resets the threshold tables only (footprints kept; fitted to a 2D image),
          with an info note; the live preview layers are "Threshold Preview (Seeds)" / "(Grow)"
    E13   the info grey follows napari's theme (>= 4.5:1 on dark and light), for status lines and
          grey hints, also when the viewer is light before the panel is built; level_of knows both
    stop  Boundary and Prompt Stop red while enabled (STOP_BUTTON), Prompt Run green (RUN_BUTTON)
    width the Seeds tab has no horizontal scrollbar at 500 px (vertical scrollbar shown, advanced
          options shown) and at napari's default dock width, Windows font (Segoe UI 9 pt); no text cut

    python tests/check_polish_panel.py [--root <tree>]     (--root: e.g. the base commit)
Prints PASS/FAIL per check; exits non-zero if anything failed.
"""
import os
import sys
import tempfile
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _docks_common import pick_root, pump, wait  # noqa: E402

ROOT = pick_root(sys.argv)

import napari  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from napari.qt import get_stylesheet  # noqa: E402
from napari.utils.theme import get_theme  # noqa: E402
from qtpy.QtGui import QColor, QFont, QFontDatabase  # noqa: E402
from qtpy.QtWidgets import QApplication, QComboBox, QLabel, QScrollArea  # noqa: E402

import napari_sprout.widgets.prompt_tab as ptm  # noqa: E402
ptm.SETTINGS = ("SPROUT-checks", "napari-sprout-checks")
import napari_sprout.widgets.seed_widget as swm  # noqa: E402
from napari_sprout.utils import status as S  # noqa: E402
from napari_sprout._widget import SPROUTWidget  # noqa: E402
from sprout_core.boundary import Cancelled  # noqa: E402

FAILED = []
UNCAUGHT = []
TMP = tempfile.mkdtemp(prefix="sprout_polish_panel_")


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
    except Exception as e:
        print(f"     ({type(e).__name__}: {e})")
        return default


def items(c):
    return [c.itemText(i) for i in range(c.count())]


def lum(c):
    def ch(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    c = QColor(c)
    return 0.2126 * ch(c.red()) + 0.7152 * ch(c.green()) + 0.0722 * ch(c.blue())


def contrast(a, b):
    la, lb = sorted((lum(a), lum(b)))
    return (lb + 0.05) / (la + 0.05)


def css_grey(w):
    """The grey a label's style sheet sets (lower case), or None."""
    import re
    m = re.search(r"(?<![\w-])color:\s*([#\w]+)", w.styleSheet())
    return None if m is None else QColor(m.group(1)).name()


dog = tifffile.imread(ROOT / "demo_data" / "dog_img.tif")[200:420]
dseed = tifffile.imread(ROOT / "demo_data" / "dog_seed.tif")[200:420]

# ======================================================================
print("== E4: Grow picks the last seeds layer of a Seeds run")
viewer = napari.Viewer(show=False)
viewer.add_image(dog, name="dog")
viewer.add_labels(dseed, name="dog_seed")
viewer.add_labels(np.zeros(dog.shape, np.uint8), name="other")
main = SPROUTWidget(viewer)
main.setStyleSheet(get_stylesheet("dark"))
main.resize(500, 900)
main.show()
sw, gw, bw, pt = main.seed_widget, main.grow_widget, main.boundary_widget, main.prompt_tab
st, gt = sw.threshold_widget, gw.threshold_widget
wait(lambda: st.hist._ready() and gt.hist._ready(), what="histograms")
pump()
sw.output_folder_line.setText(os.path.join(TMP, "seeds"))
mp = sw.main_param_widget
mp.seed_method_combo.setCurrentText("Original")
mp.populate_thresholds_from_list([140, 160], [255, 255])
mp.erosion_spin.setValue(1)
mp.segments_spin.setValue(4)
gw.seeds_combo.setCurrentText("dog_seed")
gw.status_label.set("before", "info")

added = []
viewer.layers.events.inserted.connect(lambda e: added.append(e.value.name))


def run_seeds(what):
    added.clear()
    sw.generate_seeds()
    wait(lambda: sw.generate_btn.isEnabled(), what=what)
    pump(20)
    print(f"     {what}: {sw.results_label.text()} | added {added}")


run_seeds("seeds run")
labels_added = [n for n in added if n in viewer.layers and type(viewer.layers[n]).__name__ == "Labels"
                and not n.startswith("Threshold Preview")]
last = labels_added[-1] if labels_added else None
ok(len(labels_added) == 4 and gw.seeds_combo.currentText() == last,
   f"successful run: Grow seeds = the last layer it added ({gw.seeds_combo.currentText()!r}, added {labels_added})")
ok(gw.status_label.text() == f"Seeds: {last} (last Seeds run)" and S.level_of(gw.status_label) == "info",
   f"Grow status line: {gw.status_label.text()!r} ({S.level_of(gw.status_label)})")
ok(sw.results_label.text().startswith("✓ Seeds generated: 4 layers"), f"Seeds status unchanged: {sw.results_label.text()!r}")

# the user picks another layer: it stays, the note goes
gw.seeds_combo.setCurrentText("dog_seed")
gw.seeds_combo.activated.emit(gw.seeds_combo.currentIndex())
pump()
ok(gw.seeds_combo.currentText() == "dog_seed" and gw.status_label.text() == "",
   f"user picks dog_seed: kept, the 'last Seeds run' line cleared ({gw.status_label.text()!r})")

# another Labels layer added (not by a Seeds run): no change
viewer.add_labels(np.zeros(dog.shape, np.uint8), name="unrelated")
pump()
ok(gw.seeds_combo.currentText() == "dog_seed", f"another Labels layer: Grow seeds stay dog_seed ({gw.seeds_combo.currentText()!r})")

# Stop and failure: no change
real_make_seeds = swm.make_seeds


def slow(**kw):
    while not kw["cancel"].is_set():
        time.sleep(0.01)
    raise Cancelled()


def boom(**kw):
    raise RuntimeError("boom")


gw.status_label.set("before", "info")
swm.make_seeds = slow
sw.generate_seeds()
pump(10)
sw.stop_btn.click()
wait(lambda: sw.generate_btn.isEnabled(), what="stopped run")
pump(20)
ok(S.level_of(sw.results_label) == "warn" and gw.seeds_combo.currentText() == "dog_seed"
   and gw.status_label.text() == "before",
   f"Stop: Grow untouched ({gw.seeds_combo.currentText()!r}, {gw.status_label.text()!r}; seeds: {sw.results_label.text()!r})")
swm.make_seeds = boom
sw.generate_seeds()
wait(lambda: sw.generate_btn.isEnabled(), what="failed run")
pump(20)
ok(S.level_of(sw.results_label) == "error" and gw.seeds_combo.currentText() == "dog_seed"
   and gw.status_label.text() == "before",
   f"failure: Grow untouched ({gw.seeds_combo.currentText()!r}; seeds: {sw.results_label.text()!r})")
swm.make_seeds = real_make_seeds
UNCAUGHT.clear()        # the failed run's traceback is printed by the job, not raised

# ======================================================================
print("\n== E12: new image resets thresholds only; preview layer names")
ok(mp.footprint_table.rowCount() == 1, f"first image: one default footprint row ({mp.footprint_table.rowCount()})")
viewer.add_image((dog // 2).astype(np.uint16), name="dog2")
viewer.add_image(dog[100], name="dog_2d")
pump()
mp.populate_thresholds_from_list([140, 160], [255, 255])
safe(lambda: mp.set_footprints(["ball", "ball_XY"]))
sw.image_combo.setCurrentText("dog2")
pump(20)
lows = mp._get_threshold_params()[0]
fps = mp._get_footprints()
ok(len(lows) == 1 and lows[0] == 0, f"Seeds: threshold table reset to one default row ({mp._get_threshold_params()})")
ok(fps == ["ball", "ball_XY"], f"Seeds: footprint table kept ({fps})")
ok(sw.params_status.text() == "Thresholds reset for the new image" and S.level_of(sw.params_status) == "info"
   and sw.params_status.isVisibleTo(sw), f"Seeds: info note {sw.params_status.text()!r}")
sw.image_combo.setCurrentText("dog_2d")
pump(20)
fps = mp._get_footprints()
combo_items = items(mp.footprint_table.cellWidget(0, 0)) if mp.footprint_table.rowCount() else []
ok(fps[:1] == ["ball"] and len(fps) == 2 and fps[1] != "ball_XY" and combo_items[:4] == ["ball", "cube", "disk", "square"],
   f"Seeds 2D image: rows kept, 'ball_XY' (3D only) replaced, 2D choices ({fps}, {combo_items})")
ok(sw.params_status.text() == "Thresholds reset for the new image; 1 footprint changed to fit it",
   f"Seeds 2D note: {sw.params_status.text()!r}")
sw.image_combo.setCurrentText("dog")
pump(20)
gm = gw.main_param_widget
gm.populate_thresholds_from_list([150, 120], [255, 255], [3, 3])
gw.image_combo.setCurrentText("dog2")
pump(20)
ok(len(gm._get_threshold_params()[0]) == 1 and gw.params_status.text() == "Thresholds reset for the new image"
   and S.level_of(gw.params_status) == "info",
   f"Grow: thresholds reset + note ({gm._get_threshold_params()}, {gw.params_status.text()!r})")
gw.image_combo.setCurrentText("dog")
pump(20)
wait(lambda: st.hist._ready() and gt.hist._ready(), what="histograms back")
ok(safe(lambda: (st._preview.name, gt._preview.name)) == ("Threshold Preview (Seeds)", "Threshold Preview (Grow)"),
   f"live preview names: {safe(lambda: (st._preview.name, gt._preview.name))}")
st.hist.set_range(140, st.hist._xmax)
gt.hist.set_range(120, 220)
pump(20)
prev = sorted(l.name for l in viewer.layers if l.name.startswith("Threshold Preview"))
ok(prev == ["Threshold Preview (Grow)", "Threshold Preview (Seeds)"], f"preview layers: {prev}")
tip = st.live_cb.toolTip()
ok("'Threshold Preview (Seeds)'" in tip, f"Seeds live tooltip names its layer: {tip!r}")

# ======================================================================
print("\n== E13: info grey follows the theme")
greys = {}
for theme in ("dark", "light"):
    g = safe(lambda: S.grey_for(theme))
    bg = get_theme(theme).background.as_hex()
    greys[theme] = g
    cr = contrast(g, bg) if g else 0
    ok(cr >= 4.5, f"{theme}: info grey {g} on {bg}: {cr:.2f}:1 (>= 4.5)")
ok(greys["dark"] != greys["light"], f"two greys {greys}")
probes = {"Seeds status": sw.results_label, "Grow status": gw.status_label,
          "Boundary device hint": bw.device_label, "Prompt hint": pt.classes_label}
roi_hints = [l for l in main.roi_tools.findChildren(QLabel) if "color" in l.styleSheet() and S.level_of(l) == "info"]
gw.status_label.set("text", "info")
sw.results_label.set("text", "info")
ok(roi_hints and all(css_grey(w) == QColor(greys["dark"]).name() for w in list(probes.values()) + roi_hints),
   f"dark: all info greys {greys['dark']} ({ {k: css_grey(w) for k, w in probes.items()} }, {len(roi_hints)} ROI hints)")
viewer.theme = "light"
pump(10)
ok(S.INFO == greys["light"], f"viewer.theme = light: status.INFO {S.INFO}")
ok(all(css_grey(w) == QColor(greys["light"]).name() for w in list(probes.values()) + roi_hints),
   f"light: recoloured ({ {k: css_grey(w) for k, w in probes.items()} }, ROI hints {set(css_grey(w) for w in roi_hints)})")
ok(all(S.level_of(w) == "info" for w in probes.values()) and S.level_of(gw.boundary_error) in (None, "info"),
   "level_of: 'info' for the light grey")
line = S.StatusLine("new", "info")
ok(css_grey(line) == QColor(greys["light"]).name(), f"a line made now: {css_grey(line)}")
sw.results_label.set("x", "warn")
ok(S.level_of(sw.results_label) == "warn", "other levels untouched")
viewer.theme = "dark"
pump(10)
ok(S.INFO == greys["dark"] and css_grey(line) == QColor(greys["dark"]).name()
   and S.level_of(line) == "info", f"back to dark: {S.INFO}, {css_grey(line)}")
# light before the panel is built: its hints (ROI tools included) start light
v2 = napari.Viewer(show=False)
v2.theme = "light"
v2.add_image(dog, name="dog")
m2 = SPROUTWidget(v2)
pump(10)
hints2 = [m2.seed_widget.results_label, m2.boundary_widget.device_label, m2.prompt_tab.classes_label] + \
    [l for l in m2.roi_tools.findChildren(QLabel) if "color" in l.styleSheet() and S.level_of(l) == "info"]
ok(len(hints2) > 3 and all(css_grey(w) == QColor(greys["light"]).name() for w in hints2),
   f"panel built on a light viewer: {len(hints2)} greys {set(css_grey(w) for w in hints2)}")
m2.close()
v2.close()
viewer.theme = "dark"
pump(10)

# ======================================================================
print("\n== Stop / Run buttons")
ok(bw.stop_btn.styleSheet() == S.STOP_BUTTON and pt.stop_btn.styleSheet() == S.STOP_BUTTON,
   "Boundary and Prompt Stop: STOP_BUTTON")
ok(pt.run_btn.styleSheet() == S.RUN_BUTTON and bw.compute_btn.styleSheet() == S.RUN_BUTTON,
   "Prompt Run checked and Boundary Compute: RUN_BUTTON")


def centre_colour(b):
    img = b.grab().toImage()
    return QColor(img.pixel(4, img.height() // 2))


for name, b in (("Boundary", bw.stop_btn), ("Prompt", pt.stop_btn)):
    tab = main.roi_boundary_tabs if b is bw.stop_btn else pt
    main.tabs.setCurrentWidget(tab)
    if b is bw.stop_btn:
        main.roi_boundary_tabs.setCurrentWidget(bw)
    pump(10)
    was = b.isEnabled()
    b.setEnabled(True)
    pump(5)
    on = centre_colour(b)
    b.setEnabled(False)
    pump(5)
    off = centre_colour(b)
    b.setEnabled(was)
    ok(on.red() > 180 and on.green() < 120 and not (off.red() > 180 and off.green() < 120),
       f"{name} Stop: red while enabled {on.name()}, not when disabled {off.name()}")

# ======================================================================
print("\n== Seeds tab width: no horizontal scrollbar")
try:
    families = QFontDatabase.families()          # Qt6: static
except TypeError:
    families = QFontDatabase().families()        # Qt5
segoe = "Segoe UI" in families
if segoe:
    QApplication.setFont(QFont("Segoe UI", 9))        # Windows' UI font (offscreen defaults to Helvetica)
main.setStyleSheet(get_stylesheet("dark", extra_variables={"font_size": "9pt"}))
main.resize(500, 700)                                # short: the vertical scrollbar shows
pump(40)


def fits(w, what):
    main.tabs.setCurrentWidget(w)
    pump(40)
    sa = w.findChild(QScrollArea)
    hb = sa.horizontalScrollBar()
    need, avail = sa.widget().minimumSizeHint().width(), sa.viewport().width()
    if not segoe:                                    # widths only mean something with the real font
        print(f"SKIP {what}: needs Segoe UI (content needs {need} px of {avail})")
        return
    ok(not hb.isVisible() and need <= avail,
       f"{what}: no horizontal scrollbar (content needs {need} px of {avail}; vbar {sa.verticalScrollBar().isVisible()})")


font = f"{QApplication.font().family()} {QApplication.font().pointSize()} pt"
for w, name in ((sw, "Seeds"), (gw, "Grow")):
    fits(w, f"{name} 500 px, {font}")
    w.show_advanced_checkbox.setChecked(True)
    pump(20)
    fits(w, f"{name} 500 px + advanced options")
    w.show_advanced_checkbox.setChecked(False)
main.tabs.setCurrentWidget(sw)
pump(20)
cut = [(type(x).__name__, x.text() if isinstance(x, QLabel) else x.currentText(), x.width(), x.sizeHint().width())
       for x in mp.findChildren((QLabel, QComboBox)) if x.isVisibleTo(mp) and x.width() < x.minimumSizeHint().width()]
ok(not cut, f"Seeds Parameters: no label / combo narrower than its text {cut}")
mp.seed_method_combo.setCurrentText("Adaptive (Thresholds)")
pump()
c = mp.seed_method_combo
ok(c.width() >= c.sizeHint().width(), f"Method combo shows 'Adaptive (Thresholds)' ({c.width()} >= {c.sizeHint().width()} px)")
mp.seed_method_combo.setCurrentText("Original")

# napari's default dock width
main.close()
v3 = napari.Viewer(show=True)
v3.window._qt_window.resize(1400, 900)
v3.add_image(dog, name="dog")
m3 = SPROUTWidget(v3)
dock = v3.window.add_dock_widget(m3, name="SPROUT", area="right")
pump(60)
for w, name in ((m3.seed_widget, "Seeds"), (m3.grow_widget, "Grow")):
    m3.tabs.setCurrentWidget(w)
    pump(40)
    sa = w.findChild(QScrollArea)
    ok(not sa.horizontalScrollBar().isVisible(),
       f"{name} docked at napari's default width ({dock.width()} px): no horizontal scrollbar")
v3.close()

viewer.close()
print(f"\nuncaught exceptions: {len(UNCAUGHT)}")
ok(not UNCAUGHT, "no uncaught exception")
print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
sys.exit(1 if FAILED else 0)
