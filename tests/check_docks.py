"""
Headless checks of the fixes in the side docks (Edit / Label Mapping / Batch Builder / Data Info):

    A2   Edit Undo restores the layer the last operation changed, never the selected one;
         if that layer was removed: a message, nothing written
    A3   Label Mapping keeps one mapping per Labels layer; selecting another layer clears nothing
    A10  Edit and Label Mapping bind the active layer at construction; Edit split with
         "Duplicate layer" makes no "_split" copy when there is nothing to split
    A7   Batch Builder: a failed CSV write re-enables Generate CSV / Preview, the error is a red
         line; the path stays after a successful write
    A8   Data Info: one computation per selection, cached, stale jobs stop, chunked (memory),
         dask arrays work, texts unchanged
    qt_viewer  Label Mapping does not touch viewer.window.qt_viewer
    E2/E3/E7   no per-button colours, the target in the white-on-blue header, complete Ctrl+click hints, empty Target
         Region disables its run button, Batch Builder clean-up, no emoji, no debug prints

Run in the `sprout` env (add --root <tree> to run against another source tree, e.g. the base commit):
    python tests/check_docks.py
Prints PASS/FAIL per check; exits non-zero if anything failed.
"""
import ast
import gc
import io
import contextlib
import sys
import threading
import time
import tracemalloc
import traceback
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _docks_common as C  # noqa: E402

ROOT = C.pick_root(sys.argv)

import napari  # noqa: E402
import numpy as np  # noqa: E402
from qtpy.QtWidgets import (QApplication, QCheckBox, QFormLayout, QGroupBox, QLabel, QLineEdit,  # noqa: E402
                            QPushButton, QSpinBox, QComboBox)

import napari_sprout.widgets.edit_widget as EW  # noqa: E402
import napari_sprout.widgets.info_widget as IW  # noqa: E402
import napari_sprout.widgets.map_widget as MW  # noqa: E402
import napari_sprout.widgets.tools_widget as TW  # noqa: E402

print("source tree:", Path(EW.__file__).resolve().parents[3], flush=True)
FAILED = []
DLG = C.Dialogs().install()
NOTES = C.Notes().patch(EW, MW)
UNCAUGHT = []
sys.excepthook = lambda *a: UNCAUGHT.append(traceback.format_exception_only(a[0], a[1])[-1].strip())


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


viewer = napari.Viewer(show=False)
SYN = C.synthetic_3d()


def status_text(w):
    st = getattr(w, "status", None)
    return (st.text(), getattr(st, "level", None)) if st is not None else ("", None)


def settle_edit(ew, timeout=60):
    """Wait until no background operation of the Edit dock is running."""
    t = time.time()
    while any(not b.isEnabled() for b in ew.all_edit_buttons[4:]) and time.time() - t < timeout:
        QApplication.processEvents()
        time.sleep(0.003)
    C.pump(10)


def reselect(layer):
    """Select another layer and then `layer` again (binds it in the base tree too, which binds
    only on a selection change), so A2 / A3 are tested independently of A10."""
    other = [l for l in viewer.layers if l is not layer]
    if other:
        viewer.layers.selection.active = other[0]
        C.pump(2)
    viewer.layers.selection.active = layer
    C.pump(2)


def fresh_viewer_layers():
    viewer.layers.clear()
    C.pump(3)


# ------------------------------------------------------------------ A2
def check_a2():
    fresh_viewer_layers()
    A = viewer.add_labels(SYN.copy(), name="A")
    B = viewer.add_labels(np.zeros((5, 6, 7), np.uint8) + 4, name="B")
    viewer.layers.selection.active = A
    ew = EW.QtLabelSelector(viewer)
    C.pump()
    reselect(A)
    a0, b0 = A.data.copy(), B.data.copy()
    ew.src_spin.setValue(1)
    ew.dst_spin.setValue(5)
    ew.set_class_operation()
    ok(not np.array_equal(A.data, a0), "A2 setup: Replace changed layer A")
    a1 = A.data.copy()
    viewer.layers.selection.active = B
    C.pump()
    ew.undo_last_operation()
    C.pump()
    ok(np.array_equal(B.data, b0) and B.data.shape == (5, 6, 7),
       f"A2: Undo with B selected leaves B unchanged (B shape {B.data.shape})")
    ok(np.array_equal(A.data, a0), "A2: Undo restores A, the layer the operation changed")
    txt, level = status_text(ew)
    ok(txt.startswith("✓ Undo: layer A restored (the layer of the last operation, not the selected one)")
       and level == "ok", f"A2: Undo with B selected names the restored layer A ({txt!r})")
    ew.undo_last_operation()
    ok(np.array_equal(A.data, a0) and np.array_equal(B.data, b0), "A2: a second Undo does nothing")

    # the layer of the last operation was removed -> message, nothing written
    viewer.layers.selection.active = A
    C.pump()
    ew.set_class_operation()
    viewer.layers.remove(A)
    viewer.layers.selection.active = B
    C.pump()
    ew.undo_last_operation()
    C.pump()
    txt, level = status_text(ew)
    ok(np.array_equal(B.data, b0), "A2: Undo after the layer was removed writes nothing into B")
    ok("removed" in txt and level == "warn", f"A2: ... and says so (status: {txt!r}, {level})")

    # Undo of a background operation (split) with another layer selected
    fresh_viewer_layers()
    A = viewer.add_labels(SYN.copy(), name="A")
    B = viewer.add_labels(SYN.copy() * 0 + 2, name="B")
    viewer.layers.selection.active = A
    C.pump()
    a0, b0 = A.data.copy(), B.data.copy()
    ew.split_class_spin.setValue(9)
    ew.split_class_operation()
    settle_edit(ew)
    ok(not np.array_equal(A.data, a0), "A2 setup: split changed A")
    viewer.layers.selection.active = B
    C.pump()
    ew.undo_last_operation()
    C.pump()
    ok(np.array_equal(B.data, b0) and np.array_equal(A.data, a0), "A2: Undo of a split restores A, B untouched")
    disconnect_edit(ew)


def disconnect_edit(ew):
    for em, cb in [(viewer.layers.selection.events.active, ew.update_active_label_layer_binding),
                   (viewer.layers.events.inserted, ew._refresh_merge_source_combo),
                   (viewer.layers.events.removed, ew._refresh_merge_source_combo)]:
        try:
            em.disconnect(cb)
        except Exception:
            pass
    ew.close()


# ------------------------------------------------------------------ A10 (Edit)
def check_a10_edit():
    fresh_viewer_layers()
    A = viewer.add_labels(SYN.copy(), name="A")
    viewer.add_labels(SYN.copy(), name="other")
    viewer.layers.selection.active = A
    C.pump()
    ew = EW.QtLabelSelector(viewer)
    C.pump()
    ok(ew.layer_label.text() == "Active Label Layer: A" and ew.last_bound_layer is A,
       f"A10: Edit opened with A active shows and binds A ({ew.layer_label.text()!r})")
    combo = [ew.merge_source_combo.itemText(i) for i in range(ew.merge_source_combo.count())]
    ok(combo == ["other"], f"A10: the merge source list is filled at opening ({combo})")
    ok(ew.on_click in A.mouse_drag_callbacks, "A10: Ctrl+click works without re-selecting the layer")

    # split with "Duplicate layer": no "_split" copy when there is nothing to split
    reselect(A)
    names0 = [l.name for l in viewer.layers]
    ew.duplicate_checkbox.setChecked(True)
    ew.split_class_spin.setValue(5)                  # label 5: one component
    DLG.take()
    ew.split_class_operation()
    settle_edit(ew)
    names = [l.name for l in viewer.layers]
    dialogs = DLG.take()
    ok(names == names0, f"A10: split of a one-component region makes no _split copy ({names})")
    ok(any("No split needed" in d[2] for d in dialogs), f"A10: ... and still says 'No split needed' ({dialogs})")
    viewer.layers.selection.active = A
    C.pump()
    ew.split_class_spin.setValue(9)                  # label 9: three components
    ew.split_class_operation()
    settle_edit(ew)
    names = [l.name for l in viewer.layers]
    ok(names == names0 + ["A_split"] and np.array_equal(A.data, SYN), f"A10: a real split still makes A_split ({names})")
    split = viewer.layers["A_split"].data.copy()
    ew.undo_last_operation()
    C.pump()
    ok(np.array_equal(viewer.layers["A_split"].data, SYN) and not np.array_equal(split, SYN),
       "A10: Undo after a duplicated split restores the _split layer (as before)")
    ew.duplicate_checkbox.setChecked(False)
    disconnect_edit(ew)


# ------------------------------------------------------------------ A3 / A10 (Label Mapping) / qt_viewer
def check_map():
    fresh_viewer_layers()
    A = viewer.add_labels(SYN.copy(), name="A")
    B = viewer.add_labels(C.synthetic_3d()[::-1].copy(), name="B")
    img = viewer.add_image(np.zeros(SYN.shape, np.uint8), name="img")
    viewer.layers.selection.active = A
    C.pump()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mw = MW.LabelMapperWidget(viewer)
    qt_warn = [str(x.message) for x in w if "qt_viewer" in str(x.message)]
    ok(not qt_warn, f"qt_viewer: Label Mapping does not use viewer.window.qt_viewer ({qt_warn[:1]})")
    src = Path(MW.__file__).read_text(encoding="utf-8")
    ok("qt_viewer" not in src, "qt_viewer: the deprecated key_press connection is gone from map_widget.py")
    C.pump()
    ok(mw.label_layer is A and mw.layer_label.text() == "Active Label Layer: A",
       f"A10: Label Mapping opened with A active binds A ({mw.layer_label.text()!r})")
    reselect(A)

    def click(layer, lab):
        pos = tuple(int(v) for v in np.argwhere(layer.data == lab)[0])
        viewer.dims.set_point(0, pos[0])
        mw.on_click(layer, C.FakeClick(pos, ndim=3))

    for name in ["head", "body"]:
        mw.class_input.setText(name)
        mw.add_class()
    mw.radio_manual.setChecked(True)
    click(A, 1)
    click(A, 2)
    mapA = dict(mw.label2class)
    ok(mapA == {1: "head", 2: "head"}, f"A3 setup: mapping on A {mapA}")
    viewer.layers.selection.active = img
    C.pump()
    ok(dict(mw.label2class) == mapA and mw.mapping_summary.count() == 2,
       f"A3: selecting an Image layer keeps the mapping ({dict(mw.label2class)})")
    viewer.layers.selection.active = B
    C.pump()
    ok(dict(mw.label2class) == {} and mw.label_layer is B, f"A3: B has its own (empty) mapping ({dict(mw.label2class)})")
    mw.class_dropdown.setCurrentIndex(1)
    click(B, 5)
    viewer.layers.selection.active = A
    C.pump()
    summary = [mw.mapping_summary.item(i).text() for i in range(mw.mapping_summary.count())]
    ok(dict(mw.label2class) == mapA and summary == ["Label 1 → head", "Label 2 → head"],
       f"A3: back on A, A's mapping is shown again ({summary})")
    mw.undo()
    ok(dict(mw.label2class) == {1: "head"}, f"A3: Undo works on A's own history ({dict(mw.label2class)})")
    viewer.layers.selection.active = B
    C.pump()
    ok(dict(mw.label2class) == {5: "body"}, f"A3: B kept its mapping ({dict(mw.label2class)})")
    # apply uses the shown layer's mapping
    n0 = len(viewer.layers)
    mw.apply_mapping_to_segmentation()
    C.wait(lambda: len(viewer.layers) > n0, timeout=60, what="apply")
    C.pump(5)
    out = viewer.layers["B_mapped"].data
    ok(set(np.unique(out).tolist()) == {0, 2} and np.array_equal(out == 2, B.data == 5),
       "A3: Apply maps B with B's mapping")
    txt, level = status_text(mw)
    ok(viewer.layers.selection.active is viewer.layers["B_mapped"] and mw.label_layer is viewer.layers["B_mapped"]
       and txt == "✓ Created B_mapped. The mapping of B is kept: select B to see or change it." and level == "ok",
       f"Apply: the new layer is selected and the status says B's mapping is kept ({txt!r})")
    viewer.layers.selection.active = B
    C.pump()
    ok(dict(mw.label2class) == {5: "body"}, "Apply: selecting B again shows its mapping")
    # a removed layer takes its mapping along
    viewer.layers.remove(A)
    del A
    C.pump(5)
    gc.collect()
    ok(len(mw._mappings) == 2 if hasattr(mw, "_mappings") else False,
       f"A3: mappings are held weakly (removed layer dropped; {len(getattr(mw, '_mappings', {}))} left)")
    viewer.layers.selection.events.active.disconnect(mw.update_active_label_layer_binding)
    mw.close()


# ------------------------------------------------------------------ A7
def check_a7(tmp):
    tw = TW.SPROUTToolWidget(viewer)
    g = tw.csv_aligner
    img_dir = tmp / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    import tifffile
    tifffile.imwrite(img_dir / "a.tif", np.zeros((3, 8, 8), np.uint8))
    g.img_folder_line.setText(str(img_dir))
    g.save_csv_line.setText(str(tmp / "missing_dir" / "out.csv"))
    DLG.take()
    UNCAUGHT.clear()
    g.run_alignment()
    try:
        C.wait(lambda: g.run_btn.isEnabled() or UNCAUGHT, timeout=20)
    except TimeoutError:
        pass
    C.pump(30)
    ok(g.run_btn.isEnabled() and g.preview_btn.isEnabled(),
       f"A7: after a failed CSV write Generate CSV / Preview are enabled again "
       f"(run {g.run_btn.isEnabled()}, preview {g.preview_btn.isEnabled()}, uncaught {UNCAUGHT[:1]})")
    txt, level = status_text(g)
    ok(level == "error" and "failed" in txt.lower(), f"A7: the error is a red line in the panel ({txt!r}, {level})")
    ok(not UNCAUGHT, f"A7: no uncaught exception ({UNCAUGHT[:1]})")
    # success: the path stays, a green line
    out = tmp / "ok.csv"
    g.save_csv_line.setText(str(out))
    g.run_alignment()
    C.wait(lambda: out.exists() and g.run_btn.isEnabled(), timeout=20)
    C.pump(10)
    txt, level = status_text(g)
    ok(g.save_csv_line.text() == str(out), f"A7: the CSV path is kept after a successful write ({g.save_csv_line.text()!r})")
    ok(level == "ok" and str(out) in txt and not DLG.take(), f"A7: success is a green line, no dialog ({txt!r})")
    g.save_csv_line.setText("")
    g.run_alignment()
    txt, level = status_text(g)
    ok(level == "error" and not DLG.take(), f"A7: no output file -> red line, no dialog ({txt!r})")
    return tw


# ------------------------------------------------------------------ A8
def wait_info(iw, name, timeout=120):
    C.wait(lambda: iw.info_box.toPlainText().startswith((f"Name: {name}", "[Error]")), timeout=timeout, what=name)
    C.pump(10)
    return iw.info_box.toPlainText()


class WorkerCounter:
    """Count the workers Data Info starts (wraps the module's thread_worker)."""

    def __init__(self):
        self.n = 0
        self.orig = IW.thread_worker

    def __enter__(self):
        def tw(fn):
            factory = self.orig(fn)

            def start(*a, **k):
                self.n += 1
                return factory(*a, **k)
            return start
        IW.thread_worker = tw
        return self

    def __exit__(self, *a):
        IW.thread_worker = self.orig


def check_a8():
    import dask.array as da
    fresh_viewer_layers()
    iw = IW.LayerInfoWidget(viewer)
    rng = np.random.default_rng(5)
    img = (rng.random((60, 70, 80)) * 1000).astype(np.float32)
    lab = SYN.copy()

    # texts: numpy (reference) vs dask
    L = viewer.add_labels(lab, name="lab")
    viewer.layers.selection.active = L
    iw.on_active_layer_changed()
    t_np_lab = wait_info(iw, "lab")
    I = viewer.add_image(img, name="img")
    viewer.layers.selection.active = I
    iw.on_active_layer_changed()
    t_np_img = wait_info(iw, "img")
    LD = viewer.add_labels(da.from_array(lab, chunks=(5, 40, 48)), name="lab")
    LD.name = "lab"
    viewer.layers.selection.active = LD
    iw.on_active_layer_changed()
    t_da_lab = wait_info(iw, LD.name)
    ID = viewer.add_image(da.from_array(img, chunks=(7, 70, 80)), name="img")
    viewer.layers.selection.active = ID
    iw.on_active_layer_changed()
    t_da_img = wait_info(iw, ID.name)
    strip = lambda t: t.split("\n", 1)[1]  # noqa: E731  (the names differ: lab / lab [1])
    ok(not t_da_img.startswith("[Error]") and strip(t_da_img) == strip(t_np_img).replace("float32", "float32"),
       f"A8: a dask Image layer gives the same text as numpy ({t_da_img.splitlines()[:6]})")
    ok(not t_da_lab.startswith("[Error]") and strip(t_da_lab) == strip(t_np_lab),
       f"A8: a dask Labels layer gives the same text as numpy ({t_da_lab.splitlines()[4:6]})")

    # one computation per selection (napari sends several selection events), cache
    fresh_viewer_layers()
    C.pump(20)
    with WorkerCounter() as wc:
        L2 = viewer.add_labels(lab.copy(), name="lab2")    # napari selects it (events) ...
        iw.on_active_layer_changed()                       # ... and the dock's own call
        wait_info(iw, "lab2")
        n_select = wc.n
        viewer.layers.selection.active = L2
        iw.on_active_layer_changed()
        C.pump(20)
        n_again = wc.n - n_select
        I2 = viewer.add_image(img, name="img2")
        iw.on_active_layer_changed()
        wait_info(iw, "img2")
        viewer.layers.selection.active = L2
        iw.on_active_layer_changed()
        txt = wait_info(iw, "lab2")
        n_back = wc.n
        getattr(iw, "refresh", iw.update_info)()
        C.wait(lambda: wc.n > n_back, timeout=10, what="refresh")
        wait_info(iw, "lab2")
        n_refresh = wc.n - n_back
        L2.data = lab[::-1].copy()                         # new data -> computed again
        iw.on_active_layer_changed()
        txt2 = wait_info(iw, "lab2")
        n_new_data = wc.n - n_back - n_refresh
    ok(n_select == 1, f"A8: adding / selecting a layer starts one computation ({n_select})")
    ok(n_again == 0, f"A8: re-selecting the same layer (same data) starts none ({n_again})")
    ok(n_back == 2 and txt.startswith("Name: lab2"), f"A8: switching back to an already computed layer uses the cache "
                                                      f"({n_back} computations in total)")
    ok(n_refresh == 1, f"A8: Refresh recomputes ({n_refresh})")
    ok(n_new_data == 1 and txt2.startswith("Name: lab2"), f"A8: new layer data is recomputed ({n_new_data})")

    # a stale job stops at its next chunk
    if hasattr(IW, "class_stats"):
        fresh_viewer_layers()
        big = np.zeros((256, 512, 512), np.uint8)           # 16 chunks of 4 Mi voxels
        big[::3] = 1
        gate = threading.Event()
        calls = {"big": 0}
        orig = IW.class_stats

        def slow_stats(block, *a, **k):
            if block.shape[1:] == (512, 512):
                calls["big"] += 1
                gate.wait(10)
            return orig(block, *a, **k)
        IW.class_stats = slow_stats
        try:
            LB = viewer.add_labels(big, name="big")
            iw.on_active_layer_changed()
            C.wait(lambda: calls["big"] >= 1, timeout=20, what="big started")
            LS = viewer.add_labels(lab.copy(), name="small")
            iw.on_active_layer_changed()
            gate.set()
            wait_info(iw, "small")
            C.pump(50)
            time.sleep(0.3)
            C.pump(20)
        finally:
            IW.class_stats = orig
        n_chunks = len(IW._chunks(big.shape))
        ok(calls["big"] <= 2 and iw.info_box.toPlainText().startswith("Name: small"),
           f"A8: the job of a layer no longer selected stops ({calls['big']} of {n_chunks} chunks counted)")
    else:
        ok(False, "A8: a job of a layer no longer selected stops (no chunked counting in this tree)")

    # memory: chunked counting (no full-size int64 copy)
    fresh_viewer_layers()
    big = np.zeros((128, 512, 512), np.uint8)               # 32 Mi voxels; an int64 copy = 256 MiB
    big[::2, ::2] = 3
    LB = viewer.add_labels(big, name="mem")
    C.pump(5)
    tracemalloc.start()
    getattr(iw, "refresh", iw.update_info)()
    txt = wait_info(iw, "mem", timeout=300)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    ok(peak < 64 * 2 ** 20 and "Label count: 2" in txt,
       f"A8: counting a 32 Mi-voxel uint8 Labels layer peaks at {peak / 2 ** 20:.0f} MiB (< 64 MiB)")
    iw.hide()
    iw.close()


# ------------------------------------------------------------------ E2 / E3 / E7
def check_ui():
    fresh_viewer_layers()
    L = viewer.add_labels(SYN.copy(), name="L")
    viewer.layers.selection.active = L
    ew = EW.QtLabelSelector(viewer)
    mw = MW.LabelMapperWidget(viewer)
    C.pump()
    for name, w in [("Edit", ew), ("Label Mapping", mw)]:
        styles = {b.text(): b.styleSheet() for b in w.findChildren(QPushButton) if b.styleSheet()}
        colours = sorted({s for s in styles.values()})
        ok(len(colours) <= 1 and all("#007ACC" not in s for s in colours),
           f"E2: {name}: one run-button style, no per-button colours ({len(colours)} styles: {list(styles)})")
        ok("#007ACC" in w.layer_label.styleSheet() and "white" in w.layer_label.styleSheet()
           and w.layer_label.text() == "Active Label Layer: L",
           f"E2: {name}: the target layer in the white-on-blue header ({w.layer_label.text()!r})")
        ok("Ctrl+click" in " ".join(l.text() for l in w.findChildren(QLabel)),
           f"E2: {name}: the panel says to Ctrl+click in the viewer")
    viewer.layers.selection.active = viewer.add_image(np.zeros(SYN.shape, np.uint8), name="img_hdr")
    C.pump()
    for name, w in [("Edit", ew), ("Label Mapping", mw)]:
        ok(w.layer_label.text() == "Active Label Layer: (none) - select a Labels layer in the layer list",
           f"E2: {name}: an Image selected -> the header says to select a Labels layer ({w.layer_label.text()!r})")
    viewer.layers.selection.active = L
    C.pump()
    tips = [ew.label_display.toolTip(), mw.class_dropdown.toolTip(), mw.radio_auto.toolTip()]
    ok(all("Ctrl+click" in t for t in tips), f"E2: Ctrl+click in the related tooltips ({tips})")
    targets = [("split", ew.split_class_spin, ew.split_all_checkbox),
               ("filter", ew.filter_target_label_spin, ew.filter_all_checkbox),
               ("fill", ew.fill_target_spin, ew.fill_all_checkbox),
               ("morph", ew.morph_target_spin, ew.morph_all_checkbox)]
    btns = {b.text(): b for b in ew.findChildren(QPushButton)}
    run_of = {"split": btns["Run Split Region"], "filter": btns["Run Filtering"], "fill": btns["Run Fill Holes"],
              "morph": btns["Run Morphology"]}
    empty = all(s.text() in ("", "—") or s.specialValueText() == s.text() for _, s, _ in targets)
    ok(empty and all(s.value() != 0 for _, s, _ in targets),
       f"E2: Target Region boxes start empty, not at 0 ({[s.text() for _, s, _ in targets]})")
    ok(all(not run_of[n].isEnabled() for n, _, _ in targets), "E2: their run buttons are disabled while empty")
    ew.split_class_spin.setValue(9)
    ew.fill_all_checkbox.setChecked(True)
    ok(run_of["split"].isEnabled() and run_of["fill"].isEnabled() and not run_of["morph"].isEnabled(),
       "E2: a region ID (or 'all') enables the run button")
    ew.split_class_spin.setValue(0)
    ok(run_of["split"].isEnabled(), "E2: region 0 can still be entered (asks before running)")
    # running keeps them disabled, finishing re-enables only those with a target
    ew.split_class_spin.setValue(9)
    ew.split_class_operation()
    busy = not run_of["fill"].isEnabled()
    settle_edit(ew)
    ok(busy and run_of["split"].isEnabled() and run_of["fill"].isEnabled() and not run_of["morph"].isEnabled(),
       "E2: during a run all are disabled; afterwards only those with a target are enabled")
    disconnect_edit(ew)
    viewer.layers.selection.events.active.disconnect(mw.update_active_label_layer_binding)
    mw.close()

    tw = TW.SPROUTToolWidget(viewer)
    g = tw.csv_aligner
    tw.show()
    C.pump()
    texts = [b.text() for b in tw.findChildren(QPushButton)]
    ok("test params" not in texts, f"E3: no 'test params' button ({texts})")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        tw.hide()
        tw.show()
        C.pump()
    ok(out.getvalue() == "", f"E3: showing / hiding prints nothing ({out.getvalue()!r})")
    grp = [b for b in tw.findChildren(QGroupBox) if "threshold" in b.title().lower()]
    ok(grp and not grp[0].isVisible(), "E3: the adaptive threshold box is hidden while Auto is off")
    g.auto_threshold_checkbox.setChecked(True)
    C.pump()
    ok(grp and grp[0].isVisible() and g.threshold_param_box.isVisible(), "E3: ... and shown when Auto is ticked")
    g.auto_threshold_checkbox.setChecked(False)
    labels = [l.text() for l in tw.findChildren(QLabel)]
    ok(any(t.startswith("Seeds folder") for t in labels) and not any("Segmentation" in t for t in labels),
       f"E3: the segmentation folder is called Seeds ({[t for t in labels if 'older' in t]})")
    forms = tw.findChildren(QFormLayout)
    ok(len(forms) >= 2, f"E3: folder rows are aligned in a form layout ({len(forms)} QFormLayouts)")
    notip = [type(c).__name__ + ":" + (c.text() if hasattr(c, "text") else "") for c in tw.findChildren(QWidget_types())
             if c.isVisibleTo(tw) and not c.toolTip()]
    ok(not notip, f"E3: every input / button has a tooltip (missing: {notip})")
    tw.close()

    iw = IW.LayerInfoWidget(viewer)
    ok(all(ord(ch) < 0x2000 for ch in iw.refresh_btn.text()), f"E7: Data Info button without emoji ({iw.refresh_btn.text()!r})")
    iw.close()
    prints = {}
    for mod in (EW, MW, TW, IW):
        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        n = [node.lineno for node in ast.walk(tree)
             if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print"]
        if n:
            prints[Path(mod.__file__).name] = n
    ok(not prints, f"E7: no print() calls left in the four dock files ({prints})")


def QWidget_types():
    return (QPushButton, QLineEdit, QComboBox, QSpinBox, QCheckBox)


import tempfile  # noqa: E402
TMP = Path(tempfile.mkdtemp(prefix="sprout_docks_"))
section("A2 Edit Undo", check_a2)
section("A10 Edit", check_a10_edit)
section("A3 / A10 Label Mapping, qt_viewer", check_map)
section("A7 Batch Builder", lambda: check_a7(TMP))
section("A8 Data Info", check_a8)
section("E2 / E3 / E7", check_ui)
viewer.close()
print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
sys.exit(1 if FAILED else 0)
