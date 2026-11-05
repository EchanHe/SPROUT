"""
Headless checks of the UI-polish items in the docks:

    D7   the five SPROUT docks never show as their own top-level window while napari docks them
         (application event filter: a Show event of a plugin widget that is still a window);
         after add_plugin_dock_widget all five are visible and docked
    E2   Ctrl+click on the active Labels layer with Edit and Label Mapping both open: only the dock
         used last (mouse press / keyboard focus inside it; opening counts) handles it; a hidden or
         closed dock never does; with one dock open that one handles it

Run in the `sprout` env (add --root <tree> to run against another source tree, e.g. the base commit):
    python tests/check_polish_docks.py
Prints PASS/FAIL per check; exits non-zero if anything failed.
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _docks_common as C  # noqa: E402

ROOT = C.pick_root(sys.argv)

import napari  # noqa: E402
import numpy as np  # noqa: E402
from qtpy.QtCore import QEvent, QObject, Qt  # noqa: E402
from qtpy.QtGui import QFocusEvent  # noqa: E402
from qtpy.QtTest import QTest  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

import napari_sprout.widgets.prompt_tab as PT  # noqa: E402
PT.SETTINGS = ("SPROUT-checks", "napari-sprout-checks")
import napari_sprout._widget as W  # noqa: E402

print("source tree:", Path(W.__file__).resolve().parents[2], flush=True)
FAILED = []
UNCAUGHT = []
sys.excepthook = lambda *a: UNCAUGHT.append(traceback.format_exception_only(a[0], a[1])[-1].strip())
DLG = C.Dialogs().install()


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILED.append(msg)


def section(name, fn):
    print(f"\n== {name}", flush=True)
    try:
        fn()
    except Exception:
        ok(False, f"{name}: raised\n{traceback.format_exc()}")


class ShowSpy(QObject):
    """Every Show event of a napari_sprout widget that is a top-level window at that moment."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def eventFilter(self, obj, ev):
        if (ev.type() == QEvent.Show and type(obj).__module__.startswith("napari_sprout")
                and obj.isWindow()):
            self.seen.append(type(obj).__name__)
        return False


DOCKS = ["SPROUT", "Edit", "Label Mapping", "Batch Builder", "Data Info"]


# ------------------------------------------------------------------ D7
def check_d7():
    viewer = napari.Viewer()
    app = QApplication.instance()
    spy = ShowSpy()
    app.installEventFilter(spy)
    C.pump(30)
    try:
        # the spy itself sees a top-level Show (so a 0 below means something)
        from qtpy.QtWidgets import QWidget
        probe = type("napari_sprout_probe", (QWidget,), {"__module__": "napari_sprout.probe"})()
        probe.show()
        C.pump(5)
        ok(spy.seen == ["napari_sprout_probe"], f"D7 setup: the filter sees a top-level Show ({spy.seen})")
        probe.close()
        spy.seen.clear()
        for name in DOCKS:
            n0 = len(spy.seen)
            dock, w = viewer.window.add_plugin_dock_widget("napari-sprout", name)
            C.pump(30)
            ok(len(spy.seen) == n0, f"D7: {name}: never shown as its own window ({spy.seen[n0:]})")
            ok(w.isVisible() and not w.isWindow() and dock.widget() is w and not dock.isFloating(),
               f"D7: {name}: visible and docked (visible={w.isVisible()}, window={w.isWindow()})")
            ok(type(w).__module__.startswith("napari_sprout"), f"D7: {name}: the widget is SPROUT's ({type(w)})")
    finally:
        app.removeEventFilter(spy)
        viewer.close()
        C.pump(10)


# ------------------------------------------------------------------ E2
SYN = C.synthetic_3d()


def check_e2():
    viewer = napari.Viewer()
    L = viewer.add_labels(SYN.copy(), name="L")
    viewer.layers.selection.active = L
    C.pump(10)
    dock_e, ew = viewer.window.add_plugin_dock_widget("napari-sprout", "Edit")
    dock_m, mw = viewer.window.add_plugin_dock_widget("napari-sprout", "Label Mapping")
    C.pump(30)
    for name in ["head", "body", "legs"]:
        mw.class_input.setText(name)
        mw.add_class()
    mw.radio_manual.setChecked(True)
    ok(ew.isVisible() and mw.isVisible(), "E2 setup: Edit and Label Mapping open")

    def click(lab):
        pos = tuple(int(v) for v in np.argwhere(L.data == lab)[0])
        viewer.dims.set_point(0, pos[0])
        ev = C.FakeClick(pos, ndim=3)
        for cb in list(L.mouse_drag_callbacks):   # what napari does on a mouse press
            g = cb(L, ev)
            if hasattr(g, "__next__"):
                for _ in g:
                    pass
        C.pump(3)

    def state():
        return set(ew.selected_labels), dict(mw.label2class)

    def press_in(widget):
        QTest.mouseClick(widget, Qt.LeftButton)
        C.pump(3)

    def focus_in(widget):
        QApplication.sendEvent(widget, QFocusEvent(QEvent.FocusIn, Qt.MouseFocusReason))
        C.pump(3)

    # opened last: Label Mapping answers
    click(1)
    ok(state() == (set(), {1: "head"}), f"E2: both open, Mapping opened last: only Mapping handles ({state()})")
    # mouse press in Edit
    press_in(ew.layer_label)
    click(2)
    ok(state() == ({2}, {1: "head"}), f"E2: after a press in Edit only Edit handles ({state()})")
    # keyboard focus in Label Mapping
    focus_in(mw.class_input)
    click(5)
    ok(state() == ({2}, {1: "head", 5: "head"}), f"E2: after focus in Mapping only Mapping handles ({state()})")
    # press on a child deep inside Edit (a button in a group box)
    press_in(ew.duplicate_checkbox)
    ew.duplicate_checkbox.setChecked(False)
    click(9)
    ok(state() == ({2, 9}, {1: "head", 5: "head"}), f"E2: a press on any child of Edit counts ({state()})")
    # Mapping used last, then hidden (the dock's hide button): Edit handles
    focus_in(mw.class_dropdown)
    dock_m.close()
    C.pump(10)
    click(12)
    ok(not mw.isVisible() and state() == ({2, 9, 12}, {1: "head", 5: "head"}),
       f"E2: Mapping used last but hidden: Edit handles ({state()})")
    dock_m.show()
    C.pump(10)
    click(12)
    ok(state() == ({2, 9, 12}, {1: "head", 5: "head", 12: "head"}),
       f"E2: Mapping shown again (still used last): Mapping handles ({state()})")
    # Mapping closed (the dock's close button): Edit handles, Mapping never again
    viewer.window.remove_dock_widget(dock_m)
    C.pump(10)
    click(1)
    ok(state() == ({2, 9, 12, 1}, {1: "head", 5: "head", 12: "head"}),
       f"E2: Mapping closed: Edit handles ({state()})")
    focus_in(mw.class_input)        # a closed dock that gets an event still does not take over
    click(2)
    ok(state() == ({9, 12, 1}, {1: "head", 5: "head", 12: "head"}),
       f"E2: ... also after an event in the closed Mapping ({state()})")
    # only Edit open: handles as before; Edit closed too: nobody
    viewer.window.remove_dock_widget(dock_e)
    C.pump(10)
    click(5)
    ok(state() == ({9, 12, 1}, {1: "head", 5: "head", 12: "head"}), f"E2: both closed: nobody handles ({state()})")

    # only Label Mapping open (fresh viewer): it handles, also after presses in it
    viewer.close()
    C.pump(10)
    viewer = napari.Viewer()
    L = viewer.add_labels(SYN.copy(), name="L")
    viewer.layers.selection.active = L
    C.pump(10)
    dock_m, mw = viewer.window.add_plugin_dock_widget("napari-sprout", "Label Mapping")
    C.pump(20)
    mw.class_input.setText("head")
    mw.add_class()
    mw.radio_manual.setChecked(True)
    press_in(mw.layer_label)
    click(2)
    ok(dict(mw.label2class) == {2: "head"}, f"E2: only Mapping open: it handles ({dict(mw.label2class)})")
    # a dock never shown (built in code, as the other checks do) handles as before
    import napari_sprout.widgets.edit_widget as EW
    ew2 = EW.QtLabelSelector(viewer)
    C.pump(5)
    ew2.on_click(L, C.FakeClick(tuple(int(v) for v in np.argwhere(L.data == 5)[0]), ndim=3))
    ok(ew2.selected_labels == {5}, f"E2: a never-shown Edit called directly handles ({ew2.selected_labels})")
    viewer.close()
    C.pump(10)


def check_hints():
    import napari_sprout.widgets.edit_widget as EW
    import napari_sprout.widgets.map_widget as MW
    src = Path(EW.__file__).read_text(encoding="utf-8") + Path(MW.__file__).read_text(encoding="utf-8")
    ok("used last" in src, "E2: the Ctrl+click hints say the dock used last answers")


section("D7", check_d7)
section("E2", check_e2)
section("E2 hints", check_hints)


def check_e2_tabs():
    """Edit and Label Mapping dragged into one tab group: the tab behind stays isVisible() (Qt only
    covers it), so only the tab in front may take Ctrl+click, whichever dock was used last."""
    viewer = napari.Viewer()
    L = viewer.add_labels(SYN.copy(), name="L")
    viewer.layers.selection.active = L
    C.pump(10)
    dock_e, ew = viewer.window.add_plugin_dock_widget("napari-sprout", "Edit")
    dock_m, mw = viewer.window.add_plugin_dock_widget("napari-sprout", "Label Mapping")
    C.pump(30)
    mw.class_input.setText("head")
    mw.add_class()
    mw.radio_manual.setChecked(True)
    viewer.window._qt_window.tabifyDockWidget(dock_e, dock_m)
    C.pump(30)

    def click(lab):
        pos = tuple(int(v) for v in np.argwhere(L.data == lab)[0])
        viewer.dims.set_point(0, pos[0])
        ev = C.FakeClick(pos, ndim=3)
        for cb in list(L.mouse_drag_callbacks):
            g = cb(L, ev)
            if hasattr(g, "__next__"):
                for _ in g:
                    pass
        C.pump(3)

    def state():
        return set(ew.selected_labels), dict(mw.label2class)

    # the user's steps: Edit tab in front and used, then the Mapping tab clicked (the tab bar is
    # not part of Mapping: Edit stays the dock used last)
    dock_e.raise_()
    C.pump(20)
    QTest.mouseClick(ew.layer_label, Qt.LeftButton)
    C.pump(3)
    dock_m.raise_()
    C.pump(20)
    ok(ew.isVisible(), "E2 tabs: the Edit tab behind still reports isVisible() (why the check exists)")
    click(1)
    ok(state() == (set(), {1: "head"}), f"E2 tabs: Mapping tab in front, Edit used last: Mapping handles ({state()})")
    # and back: Edit tab in front, Mapping used last
    QTest.mouseClick(mw.layer_label, Qt.LeftButton)
    C.pump(3)
    dock_e.raise_()
    C.pump(20)
    click(2)
    ok(state() == ({2}, {1: "head"}), f"E2 tabs: Edit tab in front, Mapping used last: Edit handles ({state()})")
    viewer.close()
    C.pump(10)


section("E2 tabs", check_e2_tabs)
ok(not UNCAUGHT, f"no uncaught exceptions ({UNCAUGHT[:3]})")
print(f"\n{'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED'}", flush=True)
sys.exit(1 if FAILED else 0)
