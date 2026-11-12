"""
Input layers of another shape than the image (or than the layer being edited) are greyed out in
the combos of Seeds, Grow, Prompt (AI) and Edit > Merge; a picked one that stops fitting is
dropped with a short note. Merge keeps every id (a uint8 layer does not wrap 260 to 4).

    python tests/check_input_shapes.py [--root <tree>]
Prints PASS/FAIL per check; exits non-zero if anything failed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _docks_common as C  # noqa: E402

ROOT = C.pick_root(sys.argv)

import time  # noqa: E402

import napari  # noqa: E402
import numpy as np  # noqa: E402
from napari.layers import Image  # noqa: E402
from qtpy.QtCore import Qt  # noqa: E402

from napari_sprout._widget import SPROUTWidget  # noqa: E402
import napari_sprout.widgets.edit_widget as EW  # noqa: E402
import napari_sprout.widgets.prompt_tab as PT  # noqa: E402

print("source tree:", Path(PT.__file__).resolve().parents[3], flush=True)
PT.SETTINGS = ("SPROUT-checks", "napari-sprout-checks")      # not the user's
FAILED = []
DLG = C.Dialogs().install()


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILED.append(msg)


def texts(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def usable(combo, text):
    """True / False: the item with this text can / cannot be picked; None: no such item."""
    i = combo.findText(text)
    if i < 0:
        return None
    return bool(combo.model().item(i).flags() & Qt.ItemIsEnabled)


def pt_note():
    note = getattr(pt, "input_note", None)          # None before the shape checks
    return note.text() if note is not None else ""


def greyed(combo, name):
    return usable(combo, f"{name} (shape mismatch)") is False and usable(combo, name) is None


viewer = napari.Viewer(show=False)
main = SPROUTWidget(viewer)
sw, gw, pt = main.seed_widget, main.grow_widget, main.prompt_tab
viewer.add_image(np.zeros((6, 8, 8), np.uint8), name="I")
viewer.add_labels(np.zeros((6, 8, 8), np.uint8), name="fits")
viewer.add_labels(np.zeros((6, 8, 9), np.uint8), name="other")
C.pump(20)

# ---- greyed out, still listed
combos = {"Seeds boundary": sw.boundary_combo, "Grow seeds": gw.seeds_combo, "Grow boundary": gw.boundary_combo,
          "Prompt seeds": pt.seeds_combo, "Prompt initial mask": pt.init_combo}
for what, combo in combos.items():
    ok(greyed(combo, "other") and usable(combo, "fits"), f"{what}: 'other' greyed, 'fits' usable {texts(combo)}")
ok(sw._combos_in_sync() and gw._combos_in_sync() and pt._combos_in_sync(),
   "the panels see their combos in sync (layer names, not item texts)")

# ---- the picked layer stops fitting: dropped, with a note
sw.boundary_combo.setCurrentText("fits")
gw.seeds_combo.setCurrentText("fits")
gw.boundary_combo.setCurrentText("fits")
pt.seeds_combo.setCurrentText("fits")
C.pump(10)
viewer.add_image(np.zeros((6, 8, 9), np.uint8), name="I9")
C.pump(20)
sw.image_combo.setCurrentText("I9")
C.pump(5)
ok(sw.boundary_combo.currentText() == "None" and sw.boundary_error.text() == "Boundary cleared: shape mismatch.",
   f"Seeds: new image of another shape -> boundary None ({sw.boundary_combo.currentText()!r}, "
   f"{sw.boundary_error.text()!r})")
ok(greyed(sw.boundary_combo, "fits") and usable(sw.boundary_combo, "other"), "Seeds: the greyed layers follow the image")
gw.image_combo.setCurrentText("I9")
C.pump(5)
ok(gw.seeds_combo.currentText() == "" and gw.boundary_combo.currentText() == "None"
   and gw.boundary_error.text() == "Seeds and Boundary cleared: shape mismatch.",
   f"Grow: seeds empty, boundary None ({gw.seeds_combo.currentText()!r}, {gw.boundary_error.text()!r})")
pt.image_combo.setCurrentText("I9")
C.pump(10)
ok(pt.seeds_combo.currentText() == "" and pt.seeds_layer is None
   and pt_note() == "Seeds cleared: shape mismatch.",
   f"Prompt: seeds empty ({pt.seeds_combo.currentText()!r}, {pt_note()!r})")

# ---- same shape, other placement: usable, with a warning
viewer.add_labels(np.zeros((6, 8, 9), np.uint8), name="moved", translate=(2, 0, 0))
C.pump(20)
sw.boundary_combo.setCurrentText("moved")
C.pump(5)
ok(sw.boundary_combo.currentText() == "moved"
   and sw.boundary_error.text() == "Boundary: scale/translate differs from the image.",
   f"Seeds: placement warning ({sw.boundary_error.text()!r})")
sw.boundary_combo.setCurrentText("other")
C.pump(5)
ok(sw.boundary_error.text() == "", "Seeds: the warning goes with another boundary")
pt.seeds_combo.setCurrentText("moved")
C.pump(10)
ok(pt_note() == "Seeds: scale/translate differs from the image.",
   f"Prompt: placement warning ({pt_note()!r})")

# ---- a greyed layer renamed keeps its mark
viewer.layers["fits"].name = "fits2"
C.pump(10)
ok(greyed(sw.boundary_combo, "fits2") and greyed(gw.seeds_combo, "fits2") and greyed(pt.seeds_combo, "fits2"),
   f"rename: still greyed under the new name {texts(sw.boundary_combo)}")
ok(sw._combos_in_sync() and gw._combos_in_sync() and pt._combos_in_sync(), "rename: combos still in sync")

# ---- YAML import: a layer of another shape is named as such, not "not found"
import tempfile  # noqa: E402
import yaml  # noqa: E402


def import_yaml(tab, values):
    path = Path(tempfile.mkdtemp()) / "params.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    tab.import_yaml(str(path))
    C.pump(10)
    return tab.params_status.text()


sw.boundary_combo.setCurrentText("None")
note = import_yaml(sw, {"img_path": "I9", "boundary_path": "fits2"})
ok("Boundary image 'fits2': shape mismatch." in note and sw.boundary_combo.currentText() == "None",
   f"Seeds YAML import: a boundary of another shape is not picked ({note!r})")
note = import_yaml(sw, {"img_path": "I9", "boundary_path": "other"})
ok("other" not in note and sw.boundary_combo.currentText() == "other",
   f"Seeds YAML import: a boundary of the image's shape is picked ({note!r})")
note = import_yaml(gw, {"img_path": "I9", "seg_path": "fits2", "boundary_path": "other"})
ok("Seeds layer 'fits2': shape mismatch." in note and gw.boundary_combo.currentText() == "other",
   f"Grow YAML import: seeds of another shape are named, the boundary is picked ({note!r})")

# ---- multiscale: the full-resolution shape, without stacking the levels
try:
    from napari_sprout.utils.layer_shapes import layer_shape
except ImportError:                                        # before the shape checks
    layer_shape = None
ms = Image([np.zeros((6, 8, 8)), np.zeros((3, 4, 4))], multiscale=True)
ok(layer_shape is not None and layer_shape(ms) == (6, 8, 8), "multiscale: the full-resolution shape")

# ---- Edit > Merge
viewer.layers.clear()
C.pump(5)
ew = EW.QtLabelSelector(viewer)


def merge(into, add, shift, dup=False):
    viewer.layers.clear()
    C.pump(3)
    viewer.add_labels(add, name="B")
    lay = viewer.add_labels(into.copy(), name="A")
    viewer.layers.selection.active = lay
    C.pump(5)
    ew.duplicate_checkbox.setChecked(dup)
    ew.merge_source_combo.setCurrentText("B")
    if hasattr(ew, "merge_ids_combo"):
        ew.merge_ids_combo.setCurrentText("Shift up" if shift else "Keep")
    else:                                                  # before the new UI
        ew.merge_offset_checkbox.setChecked(shift)
    ew.merge_labels_operation()
    t0 = time.time()
    while ew._busy and time.time() - t0 < 60:
        C.pump(2)
    C.pump(5)
    return viewer.layers.selection.active.data


A = np.array([[[1, 1, 0, 0, 0]]], np.uint8)
B = np.array([[[0, 2, 2, 1, 0]]], np.uint8)
r = merge(A, B, False)
ok(r.ravel().tolist() == [1, 1, 2, 1, 0], f"merge Keep: the Into layer stays where both have a region {r.ravel()}")
r = merge(A, B, True)
ok(r.ravel().tolist() == [1, 1, 3, 2, 0], f"merge Shift up: added ids above the Into layer's {r.ravel()}")
r = merge(np.array([[[200, 0, 0]]], np.uint8), np.array([[[0, 60, 100]]], np.uint8), True)
ok(r.ravel().tolist() == [200, 260, 300], f"merge Shift up in uint8: 260 and 300, no wrap to 4 and 44 {r.ravel()}")
r = merge(np.array([[[7, 0]]], np.uint8), np.array([[[0, 300]]], np.uint16), False)
ok(r.ravel().tolist() == [7, 300], f"merge Keep: a uint16 id into a uint8 layer stays 300 {r.ravel()}")
ok(getattr(getattr(ew, "merge_into_label", None), "text", lambda: "")() == "A",
   "merge: the Into row names the selected layer")
into = viewer.layers.selection.active
viewer.add_labels(np.zeros((1, 1, 6), np.uint8), name="wide")
viewer.layers.selection.active = into                     # adding a layer selects it
C.pump(5)
ok(greyed(ew.merge_source_combo, "wide") and usable(ew.merge_source_combo, "B"),
   f"merge: a layer of another shape is greyed {texts(ew.merge_source_combo)}")

# a rename: Into and Add follow, and Run finds the layer
ew.merge_source_combo.setCurrentText("B")
viewer.layers["B"].name = "B2"
into.name = "A2"
C.pump(10)
into_text = getattr(getattr(ew, "merge_into_label", None), "text", lambda: "")()   # "" before the Into row
ok(ew.merge_source_combo.currentText() == "B2" and into_text == "A2" and ew.layer_label.text().endswith("A2"),
   f"merge rename: Add {ew.merge_source_combo.currentText()!r}, Into {into_text!r}, {ew.layer_label.text()!r}")
DLG.take()
ew.merge_labels_operation()
t0 = time.time()
while ew._busy and time.time() - t0 < 60:
    C.pump(2)
C.pump(5)
ok(not DLG.take(), "merge rename: Run finds the renamed layer (no dialog)")

DLG.take()
viewer.close()
print("ALL PASSED" if not FAILED else f"{len(FAILED)} FAILED", flush=True)
sys.exit(1 if FAILED else 0)
