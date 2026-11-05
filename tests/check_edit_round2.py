"""
Checks of the round-2 fixes in the Edit dock (napari_sprout/widgets/edit_widget.py):

    A4  "Split all non-background regions" with "Sort by size": every original label is split
        into its connected components, then all regions are renumbered by size once (it used to
        renumber after each split, so later labels of the original list hit other regions).
        Without sorting, or with at most one multi-component label, the result is the old one.
    B2  the "all regions" operations (split, filter, fill holes, morphology) work per label in
        its bounding box (+ margin) instead of scanning the whole image per label: voxel-for-voxel
        identical to the old code (kept below as REF_*) for every op / option, 2D and 3D, labels
        at the image border, holes touching the box, crowded maps, dog_seed; and faster.
    B7  no np.unique on the UI thread, the Undo backup is the old array (no copy) and Undo
        restores it without a deepcopy; the old array is never written; peak memory per op.

    python tests/check_edit_round2.py [--root <tree>] [--quick]
Prints PASS/FAIL per check; exits non-zero if anything failed.
"""
import os
import sys
import threading
import time
import tracemalloc
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _docks_common as C  # noqa: E402

ROOT = C.pick_root(sys.argv)
QUICK = "--quick" in sys.argv
TIMING = not os.environ.get("SPROUT_NO_TIMING")      # shared CI machines: no speed asserts

import napari  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from skimage.measure import label as cc_label  # noqa: E402
from skimage.morphology import ball, closing, dilation, disk, erosion, opening, remove_small_holes  # noqa: E402

import napari_sprout.widgets.edit_widget as EW  # noqa: E402

print("source tree:", Path(EW.__file__).resolve().parents[3], flush=True)
FAILED = []
DLG = C.Dialogs().install()
NOTES = C.Notes().patch(EW)
UNCAUGHT = []
sys.excepthook = lambda *a: UNCAUGHT.append(traceback.format_exception_only(a[0], a[1])[-1].strip())
DOG_SEED = tifffile.imread(C.REPO / "demo_data" / "dog_seed.tif")


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


# ------------------------------------------------------------------ reference: the old code
# Verbatim copies of the round-1 implementation (full-image scan per label), used as the oracle.
def REF_label_cc(mask):
    idx = np.argwhere(mask)
    lo, hi = idx.min(0), idx.max(0) + 1
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    sub = mask[sl]
    lab = cc_label(sub, connectivity=sub.ndim)
    full = np.zeros(mask.shape, dtype=lab.dtype)
    full[sl] = lab
    return full


def REF_split_single(data, target_class, keep_top_n, sort_by_size, start_label_base):
    result = data.copy()
    mask = (result == target_class)
    if not np.any(mask):
        return result, start_label_base, False
    cc = REF_label_cc(mask)
    num_components = cc.max()
    if num_components <= 1:
        return result, start_label_base, False
    if keep_top_n > 0:
        sizes = np.bincount(cc.ravel(), minlength=num_components + 1)[1:]
        sorted_indices = np.argsort(sizes)[::-1]
        keep_ids = set(sorted_indices[:keep_top_n] + 1)
        cc = np.where(np.isin(cc, list(keep_ids)), cc, 0)
        num_components = cc.max()
    result[mask] = 0
    base = start_label_base
    nonzero = cc > 0
    result[nonzero] = cc[nonzero] + base
    if sort_by_size:
        result, _ = EW.sort_labels_by_size(result)
        base = result.max()
    else:
        base = base + num_components
    return result, base, True


def REF_split_all_old(data, keep_top_n, sort_by_size):
    """The old split-all loop (A4: renumbers the map after each split when sorting)."""
    result = data.copy()
    labels = np.unique(result)
    labels = labels[labels != 0]
    if labels.size == 0:
        return result, False
    label_base = result.max()
    any_split = False
    for lbl in labels:
        result, label_base, did_split = REF_split_single(result, lbl, keep_top_n, sort_by_size, label_base)
        any_split = any_split or did_split
    return result, any_split


def REF_split_all_intended(data, keep_top_n, sort_by_size):
    """Intended: split every original label (old no-sort loop, which is correct), sort once."""
    result, any_split = REF_split_all_old(data.astype(np.int64), keep_top_n, False)
    if sort_by_size and any_split:
        result = EW.sort_labels_by_size(result)[0]
    return result.astype(data.dtype), any_split


def REF_fill(label_img, area_threshold=64, target_label=None, apply_in_2d=False):
    result = label_img.copy()
    ndim = result.ndim
    unique_labels = np.unique(label_img)
    for lbl in unique_labels:
        if lbl == 0:
            continue
        mask = (label_img == lbl)
        max_area = np.prod(mask.shape) * 0.25
        current_area_threshold = min(area_threshold, max_area)
        if apply_in_2d and ndim == 3:
            for z in range(mask.shape[0]):
                filled = remove_small_holes(mask[z], area_threshold=current_area_threshold)
                result[z][filled & (~mask[z])] = lbl
        else:
            filled = remove_small_holes(mask, area_threshold=current_area_threshold)
            result[filled & (~mask)] = lbl
    return result


def REF_morph(data, op_name, selem):
    morph_func = {"Erode": erosion, "Dilate": dilation, "Open": opening, "Close": closing}[op_name]
    result = np.zeros_like(data)
    for lbl in np.unique(data):
        if lbl == 0:
            continue
        mask = (data == lbl)
        transformed = morph_func(mask, selem)
        result[transformed] = lbl
    return result


def REF_filter_single(mask, result, label_val, min_size, top_n):
    labeled = REF_label_cc(mask)
    counts = np.bincount(labeled.ravel())
    component_ids = np.nonzero(counts)[0]
    component_ids = component_ids[component_ids != 0]
    sizes = {int(i): int(counts[i]) for i in component_ids}
    keep_ids = [i for i, size in sizes.items() if size >= min_size] if min_size > 0 else list(component_ids)
    if top_n > 0 and len(keep_ids) > top_n:
        keep_ids = sorted(keep_ids, key=lambda x: sizes[x], reverse=True)[:top_n]
    keep_mask = np.isin(labeled, keep_ids)
    result[mask] = 0
    result[keep_mask] = label_val
    return result


def REF_filter(data, min_size, top_n):
    result = data.copy()
    for lbl in np.unique(data):
        if lbl == 0:
            continue
        result = REF_filter_single(data == lbl, result, lbl, min_size, top_n)
    return result


# ------------------------------------------------------------------ test maps
def blobs(shape, n_labels, seed, dtype=np.uint16, fill=0.35, holes=True):
    """Random label map: cubes / balls of random size anywhere (also at the border), some
    labels with several parts, hollow ones and ones with holes at their box edge."""
    rng = np.random.default_rng(seed)
    a = np.zeros(shape, dtype)
    nd = len(shape)
    ids = rng.choice(np.arange(1, max(n_labels * 3, 4)), size=n_labels, replace=False)
    for lab in ids:
        for _ in range(rng.integers(1, 4)):             # 1-3 parts
            c = [rng.integers(0, s) for s in shape]
            h = [rng.integers(1, max(2, int(s * fill / max(1, n_labels ** (1 / nd))) + 2)) for s in shape]
            sl = tuple(slice(max(0, ci - hi), min(s, ci + hi + 1)) for ci, hi, s in zip(c, h, shape))
            a[sl] = lab
            if holes and rng.random() < 0.5:            # a hole, sometimes at the box edge
                inner = tuple(slice(x.start + rng.integers(0, 2), max(x.start + 1, x.stop - rng.integers(1, 3)))
                              for x in sl)
                sub = a[inner]
                sub[(rng.random(sub.shape) < 0.3)] = 0
    return a


def cells(shape, n_labels, seed, dtype=np.uint16):
    """Crowded map of compact regions (like grown seeds): one blob per label, some hollow,
    some with a small second part next to it, many touching each other or the border."""
    rng = np.random.default_rng(seed)
    a = np.zeros(shape, dtype)
    nd = len(shape)
    r0 = max(2, int((np.prod(shape) / n_labels) ** (1 / nd) / 2))
    for lab in range(1, n_labels + 1):
        c = np.array([rng.integers(0, s) for s in shape])
        h = rng.integers(max(1, r0 // 2), r0 + 2, size=nd)
        sl = tuple(slice(max(0, ci - hi), min(s, ci + hi + 1)) for ci, hi, s in zip(c, h, shape))
        a[sl] = lab
        if rng.random() < 0.3:
            sub = a[tuple(slice(x.start + 1, max(x.start + 1, x.stop - 1)) for x in sl)]
            sub[rng.random(sub.shape) < 0.2] = 0
        if rng.random() < 0.2:
            c2 = np.clip(c + rng.integers(-2 * r0, 2 * r0 + 1, size=nd), 0, np.array(shape) - 1)
            a[tuple(slice(ci, min(s, ci + 2)) for ci, s in zip(c2, shape))] = lab
    return a


def maps(quick):
    out = []
    syn3, syn2 = C.synthetic_3d(), C.synthetic_2d()
    out += [("syn3", syn3), ("syn2", syn2)]
    for i in range(4 if quick else 10):
        out.append((f"b2d_{i}", blobs((40 + 3 * i, 50), 6 + i, i, np.uint8)))
        out.append((f"b3d_{i}", blobs((14 + i, 22, 26), 5 + i, 100 + i)))
    out.append(("crowded3d", blobs((40, 60, 60), 300, 7, np.uint16, fill=0.6)))
    out.append(("crowded2d", blobs((200, 220), 250, 8, np.int32, fill=0.6)))
    out.append(("cells3d", cells((50, 70, 70), 400, 9)))
    # whole-image labels, labels touching every border, a wall across the image, tiny slices
    w = np.zeros((10, 30), np.uint8)
    w[5, :] = 3
    w[:, 20] = 4
    out.append(("wall2d", w))
    w3 = np.zeros((30, 5, 5), np.uint16)
    w3[3:9, 1:4, 1:4] = 2
    w3[3:9, 2, 2] = 0
    w3[20:, :, :2] = 5
    out.append(("tinyslices3d", w3))
    full = np.ones((12, 13, 9), np.int32)
    full[4:7, 4:7, 3:5] = 0
    out.append(("full3d", full))
    neg = blobs((20, 24, 18), 6, 33, np.int32)
    neg[neg == neg.max()] = -2
    out.append(("negative_ids", neg))
    big = blobs((16, 20, 22), 5, 44, np.uint32)
    big[big == big.max()] = 2 ** 30
    out.append(("big_ids", big))
    out.append(("dog_seed", DOG_SEED))
    return out


SELF = EW.QtLabelSelector.__new__(EW.QtLabelSelector)   # the worker methods need no Qt state
W = EW.QtLabelSelector


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


# ------------------------------------------------------------------ B2
def check_b2_identical():
    for name, data in maps(QUICK):
        big = data.size > 2_000_000
        bad = []
        radii = [1, 2] if big else ([1, 2, 3] if QUICK else [1, 2, 3, 4, 6])
        for op in ["Erode", "Dilate", "Open", "Close"]:
            for r in radii:
                selem = disk(r) if data.ndim == 2 else ball(r)
                if not same(W._do_morphology(SELF, data, None, op, selem), REF_morph(data, op, selem)):
                    bad.append(f"morph {op} r={r}")
        for thr in ([4, 64] if big else [1, 2, 5, 64, 300, 10 ** 6]):
            for two_d in ([False, True] if data.ndim == 3 else [False]):
                if not same(W.fill_holes_in_labels(SELF, data, thr, None, two_d), REF_fill(data, thr, None, two_d)):
                    bad.append(f"fill thr={thr} 2d={two_d}")
        for mn, tn in [(0, 1), (5, 0), (30, 2), (1, 0)]:
            if not same(W._do_filtering(SELF, data, None, mn, tn, data.ndim), REF_filter(data, mn, tn)):
                bad.append(f"filter min={mn} top={tn}")
        if data.min() >= 0:                       # the sort needs bincount (old code too)
            for tn in [0, 1, 2]:
                got = W._do_split_all_classes(SELF, data, tn, False)
                ref = REF_split_all_old(data, tn, False)
                if not (same(got[0], ref[0]) and got[1] == ref[1]):
                    bad.append(f"split nosort top={tn}")
        ok(not bad, f"B2: {name} {data.shape} {data.dtype}, {len(np.unique(data)) - 1} labels: "
                    f"identical to the old code for morphology / fill / filter / split (differs: {bad})")


def check_b2_speed():
    shape = (64, 96, 96) if QUICK else (100, 160, 160)
    data = cells(shape, 300, 11)
    n = len(np.unique(data)) - 1
    rows = []
    for what, new, ref in [
        ("Close r=2", lambda: W._do_morphology(SELF, data, None, "Close", ball(2)),
         lambda: REF_morph(data, "Close", ball(2))),
        ("Fill holes 64", lambda: W.fill_holes_in_labels(SELF, data, 64, None, False),
         lambda: REF_fill(data, 64, None, False)),
        ("Filter top1", lambda: W._do_filtering(SELF, data, None, 0, 1, 3), lambda: REF_filter(data, 0, 1)),
        ("Split all", lambda: W._do_split_all_classes(SELF, data, 0, False)[0],
         lambda: REF_split_all_old(data, 0, False)[0]),
    ]:
        t = time.perf_counter(); a = new(); t_new = time.perf_counter() - t
        t = time.perf_counter(); b = ref(); t_ref = time.perf_counter() - t
        rows.append((what, t_ref, t_new))
        ok(same(a, b) and (t_ref / max(t_new, 1e-6) >= 5 or not TIMING),
           f"B2 speed {what} on {shape} ({data.size:.1e} voxels, {n} labels): old {t_ref:.2f} s, now {t_new:.2f} s "
           f"({t_ref / max(t_new, 1e-6):.0f}x), identical={same(a, b)}")


# ------------------------------------------------------------------ A4
def check_a4():
    a = np.array([[1, 1, 1, 0, 2, 0, 1, 0, 3, 3, 0, 2, 2]], np.uint8)
    got, did = W._do_split_all_classes(SELF, a, 0, True)
    want = np.array([[1, 1, 1, 0, 5, 0, 4, 0, 2, 2, 0, 3, 3]], np.uint8)
    ok(did and same(got, want), f"A4: {a.tolist()[0]} -> {got.tolist()[0]} (want {want.tolist()[0]}: "
                                f"labels 1 and 2 both split, then IDs by size)")
    n_multi_diff = n_single_same = n_single = n_multi = 0
    bad = []
    for name, data in maps(True) + [(f"r{i}", blobs((30, 40), 8, 500 + i, np.uint8)) for i in range(20)]:
        if data.min() < 0:
            continue
        for tn in [0, 1]:
            got = W._do_split_all_classes(SELF, data, tn, True)
            want = REF_split_all_intended(data, tn, True)
            if not (same(got[0], want[0]) and got[1] == want[1]):
                bad.append(f"{name} top={tn}")
            # how many labels have more than one part?
            multi = sum(1 for lab in np.unique(data) if lab != 0 and cc_label(data == lab, connectivity=data.ndim).max() > 1)
            old = REF_split_all_old(data, tn, True)
            if multi <= 1:
                n_single += 1
                n_single_same += same(old[0], got[0])
            else:
                n_multi += 1
                n_multi_diff += not same(old[0], got[0])
    ok(not bad, f"A4: split all + sort = every label split into its parts, then one renumbering by size (differs: {bad})")
    ok(n_single_same == n_single, f"A4: with at most one multi-part label the result is the old one "
                                  f"({n_single_same}/{n_single} identical)")
    print(f"     info: maps with >= 2 multi-part labels where the old result differs: {n_multi_diff}/{n_multi}")
    # many parts in a uint8 map: the temporary IDs pass 255, the final ones fit
    m = np.zeros((40, 160), np.uint8)
    lab = 1
    for y in range(0, 40, 4):
        for x in range(0, 160, 16):
            m[y, x:x + 3] = lab
            m[y + 2, x + 5:x + 6 + (x // 16) % 3] = lab   # second part
            lab += 1
    got, _ = W._do_split_all_classes(SELF, m, 0, True)
    n_parts = sum(cc_label(m == v, connectivity=2).max() for v in range(1, lab))
    ok(got.dtype == np.uint8 and got.max() == n_parts == len(np.unique(got)) - 1,
       f"A4: uint8 map, {lab - 1} labels x 2 parts: {n_parts} regions with IDs 1..{got.max()}, dtype {got.dtype}")
    nos, _ = W._do_split_all_classes(SELF, a, 0, False)
    ok(same(nos, REF_split_all_old(a, 0, False)[0]), f"A4: without sorting unchanged ({nos.tolist()[0]})")


# ------------------------------------------------------------------ B7
viewer = None


def settle(ew, timeout=300):
    t = time.time()
    while ew._busy and time.time() - t < timeout:
        C.pump(2)
    C.pump(5)


def check_b7():
    global viewer
    viewer = napari.Viewer(show=False)
    ew = EW.QtLabelSelector(viewer)
    main = threading.main_thread()
    calls = []
    orig_unique = np.unique

    def unique_spy(*a, **k):
        f = sys._getframe(1)
        if threading.current_thread() is main and f.f_code.co_filename.endswith("edit_widget.py"):
            calls.append(f.f_code.co_name)
        return orig_unique(*a, **k)

    data0 = C.synthetic_3d()
    big0 = np.tile(DOG_SEED[:, :, :], (1, 1, 1)).astype(np.uint16)   # 1.36e7 voxels, 27 MB
    src = np.zeros_like(data0)
    src[:8] = 3

    def reset(d):
        viewer.layers.clear()
        C.pump(3)
        if d.shape == src.shape:
            viewer.add_labels(src.copy(), name="M")
        L = viewer.add_labels(d.copy(), name="L")
        viewer.layers.selection.active = L
        for cb in (ew.duplicate_checkbox, ew.split_all_checkbox, ew.filter_all_checkbox,
                   ew.fill_all_checkbox, ew.morph_all_checkbox):
            cb.setChecked(False)
        C.pump(3)
        DLG.take(); NOTES.take()
        return L

    ops = {
        "replace": (lambda w: (w.src_spin.setValue(1), w.dst_spin.setValue(5)), "set_class_operation"),
        "select_keep": (lambda w: w.selected_labels.update({2, 5}), "process_selected_operation"),
        "split": (lambda w: w.split_class_spin.setValue(9), "split_class_operation"),
        "split_all": (lambda w: w.split_all_checkbox.setChecked(True), "split_class_operation"),
        "fill": (lambda w: (w.fill_target_spin.setValue(2), w.fill_area_spin.setValue(64)), "run_fill_holes_operation"),
        "fill_all": (lambda w: w.fill_all_checkbox.setChecked(True), "run_fill_holes_operation"),
        "filter": (lambda w: (w.filter_target_label_spin.setValue(9), w.filter_min_size_spin.setValue(20),
                              w.filter_top_n_spin.setValue(0)), "run_filtering_operation"),
        "morph_dilate": (lambda w: (w.morph_target_spin.setValue(5), w.morph_op_combo.setCurrentText("Dilate"),
                                    w.morph_kernel_spin.setValue(1)), "run_morphology_operation"),
        "merge": (lambda w: w.merge_source_combo.setCurrentText("M"), "merge_labels_operation"),
        "keep_top2": (lambda w: w.keep_label_topn_spin.setValue(2), "run_keep_top_n_labels"),
    }
    np.unique = unique_spy
    try:
        for key, (setup, meth) in ops.items():
            for dup in (False, True):
                L = reset(data0)
                setup(ew)
                ew.duplicate_checkbox.setChecked(dup)
                old = L.data
                before = C.sha(old)
                calls.clear()
                getattr(ew, meth)()
                ui_unique = list(calls)
                settle(ew)
                target = viewer.layers[-1]
                changed = not np.array_equal(target.data, data0)
                tag = f"{key}{' (dup)' if dup else ''}"
                ok(not ui_unique, f"B7 {tag}: no np.unique on the UI thread ({ui_unique})")
                ok(C.sha(old) == before, f"B7 {tag}: the old array is not written")
                if not dup:
                    ok(ew.original_data_backup is old,
                       f"B7 {tag}: the Undo backup is the old array itself (no copy)")
                else:
                    ok(ew.original_data_backup is not None and not np.shares_memory(ew.original_data_backup, L.data),
                       f"B7 {tag}: the backup of the duplicate shares no memory with the source layer")
                bk = ew.original_data_backup
                ew.undo_last_operation()
                C.pump(3)
                ok(changed and target.data is bk and C.sha(target.data) == C.sha(data0),
                   f"B7 {tag}: Undo restores the exact old data without a copy (changed={changed})")
                ok(C.sha(L.data) == C.sha(data0), f"B7 {tag}: source layer unchanged")
        # missing region: same dialog text as before, nothing changed
        L = reset(data0)
        ew.split_class_spin.setValue(77)
        calls.clear()
        ew.split_class_operation()
        d = DLG.take()
        ok(d and d[0][1] == "Region Not Found" and "Existing regions: [1, 2, 5, 9, 12]" in d[0][2] and len(viewer.layers) == 2,
           f"B7: a missing region still gives the same warning ({d})")
        # "Split all" ignores the greyed-out Target Region (empty or stale): it used to answer
        # "Region -1 does not exist" and split nothing
        for stale in (-1, 77):
            L = reset(data0)
            ew.split_class_spin.setValue(stale)
            ew.split_all_checkbox.setChecked(True)
            ew.split_class_operation()
            settle(ew)
            d = DLG.take()
            want = W._do_split_all_classes(SELF, data0, ew.top_n_spin.value(), ew.sort_checkbox.isChecked())[0]
            ok(not d and same(L.data, want), f"A4: Split all with Target Region {stale} splits every region ({d})")
    finally:
        np.unique = orig_unique

    # peak memory per operation (tracemalloc; numpy allocations, all threads)
    rows = []
    for key in ["replace", "select_keep", "split", "fill", "filter", "morph_dilate", "keep_top2"]:
        setup, meth = ops[key]
        L = reset(big0)
        # targets that exist in dog_seed
        if key == "replace":
            ew.src_spin.setValue(4); ew.dst_spin.setValue(3)
        elif key == "select_keep":
            ew.selected_labels.update({3, 4})
        elif key == "split":
            ew.split_class_spin.setValue(4)
        elif key == "fill":
            ew.fill_target_spin.setValue(4)
        elif key == "filter":
            ew.filter_target_label_spin.setValue(4); ew.filter_min_size_spin.setValue(20); ew.filter_top_n_spin.setValue(0)
        elif key == "morph_dilate":
            ew.morph_target_spin.setValue(4); ew.morph_op_combo.setCurrentText("Dilate"); ew.morph_kernel_spin.setValue(1)
        else:
            ew.keep_label_topn_spin.setValue(2)
        ew.original_data_backup = None
        C.pump(3)
        tracemalloc.start()
        t = time.perf_counter()
        getattr(ew, meth)()
        t_ui = time.perf_counter() - t
        settle(ew)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        DLG.take(); NOTES.take()
        rows.append((key, peak / big0.nbytes, t_ui))
        print(f"     {key:13s} peak {peak / 2 ** 20:7.1f} MB = {peak / big0.nbytes:4.1f} x image ({big0.nbytes / 2 ** 20:.0f} MB), "
              f"UI thread {t_ui:.2f} s", flush=True)
    # measured (uint16 image, napari 0.6.1): old 2.5 / 4.0 / 5.6 / 10.0 / 9.5 / 5.5 / 6.0 x the image,
    # now 1.5 / 2.0 / 3.6 / 8.0 / 7.5 / 3.5 / 4.0 x (one or two full copies fewer each)
    limits = {"replace": 1.8, "select_keep": 2.3, "split": 3.9, "fill": 8.4, "filter": 7.9,
              "morph_dilate": 3.8, "keep_top2": 4.4}
    over = [f"{k} {p:.1f}x > {limits[k]}x" for k, p, _ in rows if p > limits[k]]
    ok(not over, f"B7: peak memory per op within the new bounds (over: {over})")
    for em, cb in [(viewer.layers.selection.events.active, ew.update_active_label_layer_binding),
                   (viewer.layers.events.inserted, ew._refresh_merge_source_combo),
                   (viewer.layers.events.removed, ew._refresh_merge_source_combo)]:
        try:
            em.disconnect(cb)
        except Exception:
            pass
    ew.close()
    viewer.close()


section("A4 split all + sort", check_a4)
section("B2 identical to the old code", check_b2_identical)
section("B2 speed", check_b2_speed)
section("B7 UI thread, copies, Undo", check_b7)
ok(not UNCAUGHT, f"no uncaught exceptions ({UNCAUGHT})")
print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
sys.exit(1 if FAILED else 0)
