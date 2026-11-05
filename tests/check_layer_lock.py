"""
While an Edit operation or a Label Mapping Apply runs in the background, the Labels layers
whose own array the worker reads are locked (layer.editable = False: napari greys out the
paint tools), so brush strokes cannot end up partly in the result and in the Undo state.
Afterwards each layer gets back its own editable state (also a layer that was locked
before; napari resets editable when layer.data is set). Edit Undo is disabled meanwhile.
Results and Undo are unchanged.

    python tests/check_layer_lock.py [--root <tree>]
Prints PASS/FAIL per check; exits non-zero if anything failed.
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _docks_common as C  # noqa: E402

ROOT = C.pick_root(sys.argv)

import napari  # noqa: E402
import numpy as np  # noqa: E402

import napari_sprout.widgets.edit_widget as EW  # noqa: E402
import napari_sprout.widgets.map_widget as MW  # noqa: E402

print("source tree:", Path(EW.__file__).resolve().parents[3], flush=True)
FAILED = []
DLG = C.Dialogs().install()
NOTES = C.Notes().patch(EW)
# napari 0.7 re-raises a worker error in the main thread; outside napari.run() nothing handles it
# and the process exits (in napari.run() it is handled: the dialog shows, napari stays open)
UNCAUGHT = []
sys.excepthook = lambda *a: UNCAUGHT.append(repr(a[1]))


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILED.append(msg)


def gated(fn, fail=False):
    """fn run by the worker, held until go is set (so the test can look at the UI mid-run)."""
    started, go = threading.Event(), threading.Event()

    def run(*a, **k):
        started.set()
        go.wait(60)
        if fail:
            raise RuntimeError("injected failure")
        return fn(*a, **k)
    return run, started, go


def wait_started(started):
    t = time.time()
    while not started.is_set() and time.time() - t < 30:
        C.pump(2)
    C.pump(3)


def settle(w, timeout=60):
    t = time.time()
    while getattr(w, "_busy", 0) and time.time() - t < timeout:
        C.pump(2)
    C.pump(8)


def status(w):
    return w.status.text()


viewer = napari.Viewer(show=False)
ew = EW.QtLabelSelector(viewer)
undo_btn = getattr(ew, "undo_btn", None)
DATA = C.synthetic_3d()


def fresh(name="L", data=None):
    viewer.layers.clear()
    C.pump(3)
    for cb in (ew.duplicate_checkbox, ew.split_all_checkbox, ew.fill_all_checkbox):
        cb.setChecked(False)
    L = viewer.add_labels((DATA if data is None else data).copy(), name=name)
    viewer.layers.selection.active = L
    C.pump(3)
    DLG.take()
    NOTES.take()
    return L


# ---- fill holes, in place: L locked while running, restored after; result and Undo unchanged
L = fresh()
L.mode = "paint"
orig = ew.fill_holes_in_labels
want = orig(DATA.copy(), 64, 2, False)
run, started, go = gated(orig)
ew.fill_holes_in_labels = run
ew.fill_target_spin.setValue(2)
ew.fill_area_spin.setValue(64)
ew.run_fill_holes_operation()
wait_started(started)
mid = (L.editable, L.mode, undo_btn.isEnabled() if undo_btn else True, status(ew))
go.set()
settle(ew)
ok(mid[0] is False and str(mid[1]) == "pan_zoom",
   f"fill holes: L is locked while it runs (editable {mid[0]}, mode {mid[1]})")
ok(mid[2] is False, f"fill holes: Undo is disabled while it runs (enabled: {mid[2]})")
ok(mid[3] == "Running… L is locked until it finishes.", f"fill holes: status while running {mid[3]!r}")
ok(L.editable is True and (undo_btn is None or undo_btn.isEnabled()) and status(ew) == "",
   f"fill holes: afterwards L is editable again, Undo enabled, status cleared ({L.editable}, {status(ew)!r})")
ok(np.array_equal(L.data, want), "fill holes: the result is the same as without the lock")
ew.undo_last_operation()
ok(np.array_equal(L.data, DATA), "fill holes: Undo restores the original exactly")
ew.fill_holes_in_labels = orig

# ---- a layer that was locked before stays locked (napari resets editable when data is set)
L = fresh()
L.editable = False
ew.fill_target_spin.setValue(2)
ew.run_fill_holes_operation()
settle(ew)
ok(L.editable is False, f"a layer locked before the run stays locked afterwards (editable {L.editable})")

# ---- the worker fails: unlocked again, Undo enabled
L = fresh()
run, started, go = gated(orig, fail=True)
ew.fill_holes_in_labels = run
ew.fill_target_spin.setValue(2)
ew.run_fill_holes_operation()
wait_started(started)
mid = L.editable
go.set()
settle(ew)
dialogs = DLG.take()
ok(mid is False and L.editable is True and (undo_btn is None or undo_btn.isEnabled()),
   f"failed run: locked while running ({mid}), editable again after ({L.editable}); dialog {len(dialogs)}")
ew.fill_holes_in_labels = orig

# ---- "Duplicate layer" split: the worker reads a copy, so the source is not locked
L = fresh()
orig_split = ew._do_split_class
run, started, go = gated(orig_split)
ew._do_split_class = run
ew.duplicate_checkbox.setChecked(True)
ew.split_class_spin.setValue(9)
ew.split_class_operation()
wait_started(started)
mid = L.editable
go.set()
settle(ew)
ok(mid is True, f"Duplicate layer split: the source layer is not locked (it reads a copy; editable {mid})")
ew._do_split_class = orig_split
ew.duplicate_checkbox.setChecked(False)

# ---- merge: both the target and the source layer are read, both locked
viewer.layers.clear()
C.pump(3)
src = np.zeros_like(DATA)
src[:8] = 3
M = viewer.add_labels(src, name="M")
L = viewer.add_labels(DATA.copy(), name="L")
viewer.layers.selection.active = L
C.pump(3)
orig_merge = ew._do_merge_labels
run, started, go = gated(orig_merge)
ew._do_merge_labels = run
ew.merge_source_combo.setCurrentText("M")
ew.merge_labels_operation()
wait_started(started)
mid = (L.editable, M.editable, status(ew))
go.set()
settle(ew)
ok(mid[0] is False and mid[1] is False, f"merge: target L and source M locked while running {mid[:2]}")
ok(L.editable and M.editable, "merge: both editable again afterwards")
ew._do_merge_labels = orig_merge

# ---- Label Mapping Apply: the source layer is locked while it runs
viewer.layers.clear()
C.pump(3)
S = viewer.add_labels(DATA.copy(), name="S")
mw = MW.LabelMapperWidget(viewer)
viewer.layers.selection.active = S
C.pump(3)
mw.class_input.setText("head")
mw.add_class()
mw.radio_manual.setChecked(True)
pos = tuple(int(v) for v in np.argwhere(DATA == 2)[0])
viewer.dims.set_point(0, pos[0])
mw.on_click(S, C.FakeClick(pos, ndim=3))
orig_map = mw._do_apply_mapping
run, started, go = gated(orig_map)
mw._do_apply_mapping = run
n0 = len(viewer.layers)
mw.apply_mapping_to_segmentation()
wait_started(started)
mid = (S.editable, mw.status.text())
go.set()
C.wait(lambda: len(viewer.layers) > n0, timeout=60, what="apply")
C.pump(8)
ok(mid[0] is False and mid[1] == "Running… S is locked until it finishes.",
   f"Label Mapping Apply: S locked while running ({mid[0]}, {mid[1]!r})")
ok(S.editable is True and "S_mapped" in viewer.layers and mw.status.text().startswith("✓ Created S_mapped"),
   f"Label Mapping Apply: S editable again, the mapped layer and its status as before ({mw.status.text()!r})")

# ---- reference counting across overlapping runs
try:
    from napari_sprout.utils.layer_lock import lock_layers, unlock_layers
    a = lock_layers(viewer, [S.data])
    b = lock_layers(viewer, [S.data])
    unlock_layers(a)
    mid = S.editable
    unlock_layers(b)
    ok(mid is False and S.editable is True, "two overlapping runs: unlocked only when the last one ends")
except ImportError as e:
    ok(False, f"two overlapping runs: utils.layer_lock missing ({e})")

viewer.close()
print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
sys.exit(1 if FAILED else 0)
