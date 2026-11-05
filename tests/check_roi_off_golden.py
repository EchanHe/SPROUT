"""
Golden check: with the ROI off, the napari tabs give exactly what they gave
before the ROI feature was added.

    python tests/check_roi_off_golden.py --record   # once, BEFORE the ROI edits
    python tests/check_roi_off_golden.py --check    # after every step

A scripted headless session (Boundary Compute + Create, Threshold Preview live
slice + Create Preview Layer in Seeds and Grow, Seeds Original / Adaptive
(Erosion) / Adaptive (Thresholds), Grow) stores, for every layer: sha256 of the
data, dtype, shape, scale, translate and the metadata as JSON; the histograms
and ranges of the threshold widgets; and the decoded contents of every .tif the
runs wrote (JSON / CSV / YAML logs carry timestamps and are left out).

--check runs the session three times; each must equal the golden:
  untouched  no ROI layer at all
  pick       ROI layers are made (ROI tools: Draw box, From labels), picked in the
             Seeds, Grow and Prompt tabs and set back to None, then the session runs
  exist      ROI layers exist but no tab picks one; also checks that making them
             makes no tab recompute anything (counted)
ROI layers (found by their metadata marker) are not part of the comparison. Images
have scale 1, so copying the image's scale / translate onto the result layers changes
nothing here.

Run in the `sprout` env. Prints PASS/FAIL per check; exits non-zero on failure.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
sys.stdout.reconfigure(encoding="utf-8")   # status texts contain ✓
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "napari_sprout")]
GOLDEN = Path(__file__).resolve().parent / "data" / "golden_roi_off.json"

import napari  # noqa: E402
import _headless_gl  # noqa: E402,F401  (napari >= 0.7 offscreen: no GL context)
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

FAILED = []


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


def h_array(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(a.tobytes() + str(a.dtype).encode() + str(a.shape).encode()).hexdigest()


def jsonable(x):
    return json.loads(json.dumps(x, sort_keys=True, default=str))


def hist_state(h):
    if not h._ready():
        return {"message": h._message}
    return {"counts": h_array(h._counts), "edges": h_array(h._edges),
            "range": [float(v) for v in h.get_range()], "bounds": [None if v is None else float(v)
                                                                    for v in h.get_bounds()]}


def layer_state(lay):
    return {"type": type(lay).__name__, "data": h_array(lay.data), "dtype": str(np.asarray(lay.data).dtype),
            "shape": list(np.asarray(lay.data).shape), "scale": [float(v) for v in lay.scale],
            "translate": [float(v) for v in lay.translate], "metadata": jsonable(dict(lay.metadata))}


def disk_state(root):
    out = {}
    for p in sorted(Path(root).rglob("*.tif")):
        out[p.relative_to(root).as_posix()] = h_array(tifffile.imread(p))
    return out


def make_rois(viewer, main):
    """Two ROI layers made with the ROI tools: a drawn box and a box around label ids."""
    rt = main.roi_tools
    viewer.layers.selection.active = viewer.layers["dog"]
    rt.image_combo.setCurrentIndex(rt.image_combo.findData("dog"))
    rt.draw_btn.click()
    pump()
    viewer.layers.selection.active.add_rectangles(np.array([[20, 10], [20, 100], [120, 100], [120, 10]], float))   # the Draw box layer
    pump(40)
    rt.labels_combo.setCurrentIndex(rt.labels_combo.findData("dog_seed"))
    rt.ids_edit.setText("")                     # all non-zero seed labels
    rt.create_btn.click()
    wait(lambda: rt.create_btn.isEnabled(), what="ROI from labels")
    pump(40)
    from napari_sprout.utils import roi_layers as RL
    rois = RL.roi_layers(viewer)
    ok(len(rois) == 2 and all(RL.get_roi_hub(viewer).live_roi(l) is not None for l in rois),
       f"ROI layers made: {[l.name for l in rois]}")
    return rois


def pickers(main):
    return (main.seed_widget.roi_select, main.grow_widget.roi_select, main.prompt_tab.roi_select)


def session(tmp, roi_mode=None):
    """The scripted session; returns its record."""
    from napari_sprout._widget import SPROUTWidget
    import napari_sprout.widgets.boundary_widget as bwmod

    rec = {}
    viewer = napari.Viewer(show=False)
    dog = tifffile.imread(ROOT / "demo_data" / "dog_img.tif")[200:420]
    seed = tifffile.imread(ROOT / "demo_data" / "dog_seed.tif")[200:420]
    viewer.add_image(dog, name="dog")
    viewer.add_labels(seed, name="dog_seed")
    pump()
    main = SPROUTWidget(viewer)
    rec_torch = "torch" not in sys.modules
    sw, gw, bw = main.seed_widget, main.grow_widget, main.boundary_widget
    st, gt = sw.threshold_widget, gw.threshold_widget
    wait(lambda: st.hist._ready() and gt.hist._ready() and bw.input_hist._ready(), what="histograms")
    if roi_mode == "pick":
        rois = make_rois(viewer, main)
        for sel, layer in zip(pickers(main), rois + rois[:1]):
            sel.select(layer)
        pump(60)
        wait(lambda: not (st._workers or gt._workers or bw._workers or st._hist_timer.isActive()
                          or gt._hist_timer.isActive()), what="ROI histograms (picked)")
        for sel in pickers(main):
            sel.select(None)
        pump(40)
    elif roi_mode == "exist":
        import napari_sprout.utils.histogram_threshold_widget as htm
        import napari_sprout.widgets.prompt_tab as ptm
        from sprout_core import boundary as B
        calls = []
        saved = htm._cached_histogram, B.histogram, bwmod.BoundaryWidget._run, \
            ptm.PromptTab.compute_stats

        def spy(fn, what):
            def f(*a, **k):
                calls.append(what)
                return fn(*a, **k)
            return f

        htm._cached_histogram, B.histogram = spy(saved[0], "threshold"), spy(saved[1], "histogram")
        bwmod.BoundaryWidget._run = spy(saved[2], "boundary")
        ptm.PromptTab.compute_stats = spy(saved[3], "prompt")
        try:
            make_rois(viewer, main)
            pump(60)
        finally:
            htm._cached_histogram, B.histogram, bwmod.BoundaryWidget._run, \
                ptm.PromptTab.compute_stats = saved
        ok(not calls, f"ROI layers that no tab picks: no tab recomputes anything ({len(calls)} calls: {calls})")
    if roi_mode:
        wait(lambda: not (st._workers or gt._workers or bw._workers or st._hist_timer.isActive()
                          or gt._hist_timer.isActive()), what="ROI histograms")
        wait(lambda: st.hist._ready() and gt.hist._ready() and bw.input_hist._ready(), what="histograms 2")
        pump(40)
    rec["hist_initial"] = {"seeds": hist_state(st.hist), "grow": hist_state(gt.hist),
                           "boundary_input": hist_state(bw.input_hist)}

    # ---- Boundary: Compute (CPU) + live slice preview + Create
    bw.device_combo.setCurrentIndex(bw.device_combo.findData("cpu"))
    bw.panel.set_params({"kind": "black", "shape": "ball", "size": 2})
    bw._compute()
    wait(lambda: bw._resp is not None and not bw._computing, what="boundary compute")
    rec["hist_response"] = hist_state(bw.resp_hist)
    bw.resp_hist.set_range(25, bw.resp_hist._xmax)
    pump()
    rec["boundary_preview"] = h_array(bw._preview_buf)
    rec["boundary_preview_label"] = bw.preview_label.text()
    bw._create()
    wait(lambda: "dog_boundary" in viewer.layers, what="boundary create")
    pump()

    # ---- Threshold Preview (Seeds: erosion + CC; Grow: threshold only)
    st.hist.set_range(140, st.hist._xmax)
    pump()
    rec["seeds_live_slice"] = h_array(viewer.layers[st._preview.name].data)   # "Threshold Preview (Seeds)"
    rec["seeds_live_status"] = st.status.text()
    st.use_connected_components_checkbox.setChecked(True)
    st.connected_components_spin.setValue(8)
    st.erosion_spin.setValue(1)
    wait(lambda: st.create_btn.isEnabled())
    st.create_btn.click()
    wait(lambda: "dog_preview_140-255_ero1_cc8" in viewer.layers, what="seeds preview layer")
    gt.hist.set_range(120, 220)
    pump()
    wait(lambda: gt.create_btn.isEnabled())
    gt.create_btn.click()
    wait(lambda: "dog_preview_120-220" in viewer.layers, what="grow preview layer")

    # ---- Seeds: Original (with the boundary), Adaptive (Erosion) + INTER, Adaptive (Thresholds)
    mp = sw.main_param_widget
    sw.output_folder_line.setText(str(Path(tmp) / "seeds"))
    ok(sw.boundary_combo.currentText() == "dog_boundary", "boundary layer selected in Seeds")

    def run_seeds(what):
        sw.generate_seeds()
        wait(lambda: sw.generate_btn.isEnabled(), what=what)
        pump()
        print(f"     {what}: {sw.results_label.text()}")

    mp.seed_method_combo.setCurrentText("Original")
    mp.populate_thresholds_from_list([140, 160], [255, 255])
    mp.erosion_spin.setValue(2)
    mp.segments_spin.setValue(6)
    run_seeds("seeds original")
    sw.boundary_combo.setCurrentText("None")
    mp.seed_method_combo.setCurrentText("Adaptive (Erosion)")
    mp.populate_thresholds_from_list([140], [255])
    mp.erosion_spin.setValue(3)
    sw.save_every_iter_checkbox.setChecked(True)
    run_seeds("seeds adaptive erosion")
    mp.seed_method_combo.setCurrentText("Adaptive (Thresholds)")
    mp.populate_thresholds_from_list([130, 150, 170], [255, 255, 255])
    mp.erosion_spin.setValue(1)
    sw.save_every_iter_checkbox.setChecked(False)
    run_seeds("seeds adaptive thresholds")

    # ---- Grow (with the boundary)
    gw.refresh_layers()
    gw.seeds_combo.setCurrentText("dog_seed")
    ok(gw.boundary_combo.currentText() == "dog_boundary", "boundary layer selected in Grow")
    gw.main_param_widget.populate_thresholds_from_list([150, 120], [255, 255], [3, 3])
    gw.output_folder_line.setText(str(Path(tmp) / "grow"))
    gw.start_growth()
    wait(lambda: gw.grow_btn.isEnabled(), what="grow")
    pump()
    print(f"     grow: {gw.status_label.text()}")

    from napari_sprout.utils.roi_layers import is_roi_layer
    rec["layers"] = {l.name: layer_state(l) for l in viewer.layers if not is_roi_layer(l)}
    rec["layer_names"] = sorted(rec["layers"])
    rec["disk"] = disk_state(tmp)
    wait(lambda: not bw._workers and not st._workers and not gt._workers, timeout=60)
    viewer.close()
    return rec, rec_torch


def compare(rec, gold, label):
    same = True
    for k in sorted(set(gold) | set(rec)):
        if k == "layers":
            continue
        if rec.get(k) != gold.get(k):
            same = False
            print(f"     {label}: '{k}' differs")
    for name in sorted(set(gold["layers"]) | set(rec["layers"])):
        a, b = rec["layers"].get(name), gold["layers"].get(name)
        if a != b:
            same = False
            diff = [f for f in (b or a) if (a or {}).get(f) != (b or {}).get(f)]
            print(f"     {label}: layer '{name}' differs in {diff}")
    return same


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--record", action="store_true")
    g.add_argument("--check", action="store_true")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="sprout_golden_")
    try:
        rec, no_torch = session(os.path.join(tmp, "a"))
        ok(no_torch, "building the SPROUT panel does not import torch")
        if args.record:
            GOLDEN.write_text(json.dumps(rec, indent=1, sort_keys=True))
            print(f"recorded {len(rec['layers'])} layers, {len(rec['disk'])} tif files -> {GOLDEN}")
        else:
            gold = json.loads(GOLDEN.read_text())
            ok(compare(rec, gold, "untouched"),
               f"ROI never touched: {len(rec['layers'])} layers, {len(rec['disk'])} tif files == golden")
            rec2, _ = session(os.path.join(tmp, "b"), roi_mode="pick")
            ok(compare(rec2, gold, "pick"),
               f"ROIs picked in Seeds / Grow / Prompt and back to None: {len(rec2['layers'])} layers, "
               f"{len(rec2['disk'])} tif files == golden")
            rec3, _ = session(os.path.join(tmp, "c"), roi_mode="exist")
            ok(compare(rec3, gold, "exist"),
               f"ROI layers exist, no tab picks one: {len(rec3['layers'])} layers, {len(rec3['disk'])} tif files "
               f"== golden")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
