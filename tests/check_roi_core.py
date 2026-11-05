"""
Checks of sprout_core/roi.py against explicit numpy / whole-image references (no viewer).

    1  Roi basics: clipping, bbox margins, mask_crop ring, from_mask vs argwhere,
       several ids, bounding box of a mask + margin (as_box), parse_ids / format_ids,
       mask_window == mask_full()[index], sample_values, contains, paste, metadata,
       roi_name, errors
    2  local operations: top-hat on bbox + halo == whole-image top-hat (negative
       control: halo 2s-1); threshold -> erode -> CC in an ROI == whole image with
       allowed = ROI mask; allowed=None byte-identical to the old code path
    3  Seeds (original + adaptive) in an ROI == whole image with boundary | outside-ROI
       (negative control: crop margin 0)
    4  Grow in an ROI == where(roi, whole-image wall run, seeds)
    5  a full-image box ROI == ROI off
    6  names, value keys, save / load (incl. a run's roi.json), value-keyed histogram cache

Run in the `sprout` env:
    python tests/check_roi_core.py [--quick]
Prints PASS/FAIL per check; exits non-zero if anything failed.
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "napari_sprout")]
os.chdir(ROOT)

from sprout_core import roi as R  # noqa: E402
from sprout_core.roi import Roi, RoiError  # noqa: E402

FAILED = []
QUICK = "--quick" in sys.argv
TMP = tempfile.mkdtemp(prefix="sprout_roi_core_")


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILED.append(msg)


def raises(fn, exc=RoiError, text=None):
    try:
        fn()
    except exc as e:
        return text is None or text in str(e)
    return False


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def same(a, b):
    """Same values, dtype and shape."""
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


def same_dicts(a, b):
    return sorted(a) == sorted(b) and all(same(a[k], b[k]) for k in a)


rng = np.random.default_rng(7)


def box_mask(shape, lo, hi):
    m = np.zeros(shape, bool)
    m[tuple(slice(max(0, l), max(0, min(n, h))) for l, h, n in zip(lo, hi, shape))] = True
    return m


def random_roi(shape, kind, touch=False):
    """A random box or blob-mask ROI, optionally touching the image border."""
    lo = [int(rng.integers(0, n // 2)) for n in shape]
    hi = [int(rng.integers(l + 2, n + 1)) for l, n in zip(lo, shape)]
    if touch:
        ax = int(rng.integers(0, len(shape)))
        if rng.random() < 0.5:
            lo[ax] = 0
        else:
            hi[ax] = shape[ax]
    if kind == "box":
        return Roi.box(shape, lo, hi), box_mask(shape, lo, hi)
    m = box_mask(shape, lo, hi) & (rng.random(shape) > 0.3)
    from scipy import ndimage as ndi
    m = ndi.binary_closing(m, iterations=1) & box_mask(shape, lo, hi)
    if not m.any():
        m[tuple(lo)] = True
    lab = np.zeros(shape, np.uint8)
    lab[m] = 3
    lab[~m & (rng.random(shape) > 0.9)] = 1
    return Roi.from_mask(lab, label=3, name="m"), m


# ======================================================================
print("== 1. Roi basics")
shape = (20, 30, 40)
r = Roi.box(shape, (5, -3, 10), (12, 25, 99))
ok(r.lo == (5, 0, 10) and r.hi == (12, 25, 40) and r.source == {"clipped": True},
   f"box clipped to the image: {r.lo}..{r.hi}")
ok(same(r.mask_full(), box_mask(shape, (5, 0, 10), (12, 25, 40))), "box mask_full == numpy box")
ok(r.n_voxels == 7 * 25 * 30 and abs(r.fraction - 7 * 25 * 30 / (20 * 30 * 40)) < 1e-12, "n_voxels, fraction")
ok(r.bbox(2) == (slice(3, 14), slice(0, 27), slice(8, 40)), f"bbox(2) clamped: {r.bbox(2)}")
ok(r.offset(2) == (3, 0, 8) and r.crop_shape(2) == (11, 27, 32), "offset / crop_shape")
ok(all(type(v) is int for v in r.offset(1)), "offset gives python ints")
ok(r.bbox((1, 0, 3)) == (slice(4, 13), slice(0, 25), slice(7, 40)), "per-axis margin")
ring = r.mask_crop(1)
ok(ring.shape == r.crop_shape(1) and ring.sum() == r.n_voxels and not ring[0].any() and not ring[-1].any()
   and ring[:, 0].any() and not ring[:, -1].any() and not ring[:, :, 0].any() and ring[:, :, -1].any(),
   "mask_crop(1): one-voxel wall ring except where the ROI touches the border")
ok(Roi.box(shape, (0, 0, 0), shape).source == {"clipped": False}, "unclipped full box")
e = Roi.box(shape, (5, 50, 0), (9, 60, 4))
ok(e.empty and raises(lambda: e.check_shape(shape), text="empty"), "box outside the image -> empty, RoiError")
ok(raises(lambda: r.check_shape((20, 30, 41)), text="shape (20, 30, 40), this image has shape (20, 30, 41)"),
   "shape mismatch -> RoiError with both shapes")
ok(raises(lambda: r.crop(np.zeros((3, 3, 3)))), "crop of a wrong-shaped array -> RoiError")
ok(raises(lambda: r.bbox(shape=(1, 2, 3))), "bbox(shape=) validates")
ok(r.check_shape(shape) is r, "check_shape returns the roi")

lab = np.zeros(shape, np.uint16)
lab[2:5, 3:9, 4:6] = 2
lab[10, 20, 30] = 2
lab[15:18, 1:2, 0:3] = 7
for label in (None, 2, 7):
    m = lab != 0 if label is None else lab == label
    rm = Roi.from_mask(lab, label=label, name="lab")
    idx = np.argwhere(m)
    ok(rm.lo == tuple(idx.min(0)) and rm.hi == tuple(idx.max(0) + 1) and same(rm.mask_full(), m)
       and rm.n_voxels == m.sum() and rm.sub_mask.shape == tuple(idx.max(0) + 1 - idx.min(0)),
       f"from_mask label={label}: bbox == argwhere bbox, mask_full == mask, n={rm.n_voxels}")
rm = Roi.from_mask(lab, label=3, name="lab")
ok(rm.empty and raises(lambda: rm.check_shape(shape), text="label 3 does not occur in 'lab'"),
   "label not present -> \"label 3 does not occur in 'lab'\"")
ok(raises(lambda: Roi.from_mask(np.zeros(shape, np.uint8), name="z").check_shape(shape),
          text="'z' has no non-zero voxel"), "empty mask -> RoiError")
src = lab.copy()
rm = Roi.from_mask(src, label=2, name="lab")
src[:] = 0
ok(rm.n_voxels == (lab == 2).sum(), "from_mask copies: later edits of the labels do not change the snapshot")
ok(Roi.box(shape, (0,) * 3, (1,) * 3).token != Roi.box(shape, (0,) * 3, (1,) * 3).token, "token unique per snapshot")

# several label ids, bounding box of a mask (+ margin, clipped), id parsing
ok(R.parse_ids("3,5-7; 9") == {3, 5, 6, 7, 9} and R.parse_ids(" 12 ") == {12} and R.parse_ids("") == set(),
   "parse_ids: ids and ranges")
ok(raises(lambda: R.parse_ids("3,x"), ValueError) and raises(lambda: R.parse_ids("a-4"), ValueError),
   "parse_ids: ValueError on junk")
ok(R.format_ids([7, 3, 5, 6, 9, 10]) == "3,5-7,9-10" and R.format_ids([4]) == "4", "format_ids")
from napari_sprout.widgets.prompt_tab import parse_ids as prompt_parse_ids  # noqa: E402
ok(prompt_parse_ids is R.parse_ids, "the Prompt tab uses the same parse_ids")
lab_m = np.zeros((30, 40, 50), np.uint16)
lab_m[2:6, 3:9, 4:10] = 3
lab_m[20, 30, 40] = 5
lab_m[25:28, 1:4, 45:50] = 7
lab_m[10:12, 10:12, 10:12] = 9
for ids in ([3, 5, 6, 7], [5], [3, 9], [7, 3]):
    rm_ = Roi.from_mask(lab_m, ids=ids, name="m")
    m_ = np.isin(lab_m, ids)
    idx = np.argwhere(m_)
    ok(same(rm_.mask_full(), m_) and rm_.lo == tuple(idx.min(0)) and rm_.hi == tuple(idx.max(0) + 1)
       and rm_.n_voxels == m_.sum(), f"from_mask(ids={ids}) == np.isin mask, bbox == argwhere bbox")
ok(Roi.from_mask(lab_m, ids=[5], name="m").source["label"] == 5 and
   Roi.from_mask(lab_m, ids=[7, 3, 5], name="m").source["ids"] == [3, 5, 7], "a single id is stored as 'label'")
ok(Roi.from_mask(lab_m, ids=[3, 5, 6, 7], name="m").describe().startswith("mask 'm' ids 3,5-7 ("),
   f"describe: {Roi.from_mask(lab_m, ids=[3, 5, 6, 7], name='m').describe()}")
e_ = Roi.from_mask(lab_m, ids=[11, 12], name="m")
ok(e_.empty and raises(lambda: e_.check_shape(lab_m.shape), text="none of the ids 11-12 occurs in 'm'"),
   "absent ids -> RoiError naming them")
lo_, hi_ = R.bbox_of_mask(lab_m == 9)
ok((lo_, hi_) == ([10, 10, 10], [12, 12, 12]) and R.bbox_of_mask(np.zeros((3, 4), bool)) == ([0, 0], [0, 0]),
   "bbox_of_mask (and of an empty mask)")
src = Roi.from_mask(lab_m, ids=[3, 7], name="m")
for margin in (0, 5):
    b_ = src.as_box(margin)
    exp_lo = [max(0, l - margin) for l in src.lo]
    exp_hi = [min(n, h + margin) for h, n in zip(src.hi, lab_m.shape)]
    ok(b_.kind == "box" and b_.sub_mask is None and list(b_.lo) == exp_lo and list(b_.hi) == exp_hi
       and same(b_.mask_full(), box_mask(lab_m.shape, exp_lo, exp_hi))
       and b_.source["from_mask"]["ids"] == [3, 7] and b_.source["margin"] == margin,
       f"as_box(margin={margin}): {b_.lo}..{b_.hi}, clipped={b_.source['clipped']}")
ok(src.as_box(5).source["clipped"] and not src.as_box(0).source["clipped"], "as_box: 'clipped' when the margin hits the border")
ok(src.as_box(5).describe().startswith("bbox of 'm' ids 3,7 + margin 5: z 0-29") and
   src.as_box(0).describe().startswith("bbox of 'm' ids 3,7: z 2-27"), f"as_box describe: {src.as_box(5).describe()}")
ok(json.loads(json.dumps(src.as_box(5).to_metadata())) == src.as_box(5).to_metadata(), "as_box metadata JSON-able")
eb = e_.as_box(3)
ok(eb.empty and raises(lambda: eb.check_shape(lab_m.shape), text="none of the ids 11-12 occurs in 'm'"),
   "as_box of an empty mask stays empty, same reason")

# mask_window == mask_full()[index] for random windows and 2D slices, 3D and 2D, box and mask
n = bad = 0
for trial in range(120):
    shp = (int(rng.integers(3, 15)), int(rng.integers(3, 15)), int(rng.integers(3, 15))) if trial % 3 \
        else (int(rng.integers(3, 30)), int(rng.integers(3, 30)))
    roi, full = random_roi(shp, "box" if trial % 2 else "mask", touch=trial % 4 == 0)
    for _ in range(6):
        idx = []
        for k, s in enumerate(shp):
            c = rng.random()
            if c < 0.25:
                idx.append(int(rng.integers(0, s)))
            elif c < 0.4:
                idx.append(slice(None))
            else:
                a = int(rng.integers(0, s))
                idx.append(slice(a, int(rng.integers(a, s + 1))))
        idx = tuple(idx)
        n += 1
        bad += not same(roi.mask_window(idx), full[idx])
    n += 1
    bad += not same(roi.mask_full(), full)
    n += 1
    m1 = int(rng.integers(0, 4))
    bad += not same(roi.mask_crop(m1), full[roi.bbox(m1)])
ok(bad == 0, f"mask_window / mask_crop / mask_full == numpy mask[index]: {n - bad}/{n} (3D + 2D, box + mask, "
             f"ints, full and partial slices, border-touching)")
ok(Roi.box((10, 10), (2, 2), (5, 5)).mask_window((slice(None), 3)).shape == (10,), "int index drops the axis")
ok(Roi.box((10, 10), (2, 2), (5, 5)).mask_window((3,)).shape == (10,), "short index padded with slice(None)")
ok(raises(lambda: r.mask_window((slice(0, 10, 2),)), ValueError), "stepped slice rejected")

# sample_values
img = rng.integers(0, 1000, (60, 70, 80)).astype(np.uint16)
roi, full = random_roi(img.shape, "mask")
sv = roi.sample_values(img, target=10 ** 9)
ok(same(np.sort(sv), np.sort(img[full])), "sample_values (no striding) == img[mask]")
sv = roi.sample_values(img, target=2_000)
crop = img[roi.bbox()]
step = max(1, int(round((roi.n_voxels / 2_000) ** (1 / 3))))   # mask: stride from the ROI's voxel count
exp = crop[::step, ::step, ::step][full[roi.bbox()][::step, ::step, ::step]]
ok(step > 1 and same(sv, exp), f"sample_values strided (step {step} from n_voxels), then masked")
other = (img * 3 + 1).astype(np.uint16)
ok(same(roi.sample_values(other, 2_000), (sv * 3 + 1).astype(np.uint16)), "same roi + shape -> aligned samples")
rb = Roi.box(img.shape, (5, 5, 5), (30, 40, 50))
ok(same(rb.sample_values(img, 10 ** 9), img[5:30, 5:40, 5:50].ravel()), "box sample_values == bbox values")

# review fix: a sparse mask ROI in a big bbox must not give 0 samples (histograms then fail)
sparse = np.zeros((60, 60, 60), np.uint8)
for p_ in [(0, 1, 1), (1, 0, 1), (1, 1, 0), (59, 59, 59), (31, 31, 31)]:
    sparse[p_] = 1                       # all off the stride-2 / stride-4 grids
rs = Roi.from_mask(sparse)
big_img = rng.integers(0, 255, sparse.shape).astype(np.uint8)
sv = rs.sample_values(big_img, target=1_000)
ok(same(np.sort(sv), np.sort(big_img[sparse != 0])), f"sparse mask: sample_values falls back to all ROI voxels "
                                                     f"({sv.size} samples, was 0)")
tenth = (rng.random((40, 40, 40)) < 0.1).astype(np.uint8)     # ~6,400 voxels in a 64,000-voxel bbox
tenth[0, 0, 0] = tenth[-1, -1, -1] = 1
rh = Roi.from_mask(tenth)
sv = rh.sample_values(rng.integers(0, 255, tenth.shape).astype(np.uint8), target=1_000)
ok(400 <= sv.size <= 4_000, f"mask: about `target` samples ({sv.size} for target 1000; striding by bbox size gave ~100)")

# contains
pts = np.column_stack([rng.uniform(-2, n + 1, 400) for n in img.shape])
ri = np.rint(pts).astype(int)
inb = np.all((ri >= 0) & (ri < np.array(img.shape)), axis=1)
exp = np.zeros(len(pts), bool)
exp[inb] = full[tuple(ri[inb].T)]
ok(same(roi.contains(pts), exp), "contains (float coords, rounded, some outside the image) == mask lookup")
ok(same(roi.contains(ri), exp), "contains (int coords)")

# paste variants
vals = rng.integers(1, 50, roi.crop_shape(1)).astype(np.uint16)
p = roi.paste(vals, margin=1)
exp = np.zeros(img.shape, np.uint16)
exp[roi.bbox(1)][full[roi.bbox(1)]] = vals[full[roi.bbox(1)]]
ok(same(p, exp), "paste region='roi' into zeros: only ROI voxels written, dtype kept")
p = roi.paste(vals, margin=1, region="bbox", fill=9, dtype=np.uint32)
exp = np.full(img.shape, 9, np.uint32)
exp[roi.bbox(1)] = vals
ok(same(p, exp), "paste region='bbox', fill, dtype")
base = img.copy()
p = roi.paste(vals, margin=1, base=base)
exp = img.copy()
exp[full & box_mask(img.shape, [s.start for s in roi.bbox(1)], [s.stop for s in roi.bbox(1)])] = \
    vals[full[roi.bbox(1)]]
ok(same(p, exp) and same(base, img) and p is not base, "paste into base: copied, outside kept")
p = roi.paste(vals, margin=1, base=base, inplace=True)
ok(p is base and same(base, exp), "paste inplace writes into base")
w = (slice(10, 20), slice(0, 70), 4)
ok(raises(lambda: roi.paste(np.zeros((3, 3)), window=(slice(0, 5), slice(0, 5))), ValueError),
   "paste with a wrong-shaped value -> ValueError")
w = (slice(10, 20), slice(0, 70), slice(4, 5))
v2 = np.full((10, 70, 1), 5, np.uint16)
p = roi.paste(v2, window=w)
exp = np.zeros(img.shape, np.uint16)
exp[w][full[w]] = 5
ok(same(p, exp), "paste with an explicit window")

# metadata, describe, roi_name
md = r.to_metadata()
ok(json.loads(json.dumps(md)) == md and md["bbox_start"] == [5, 0, 10] and md["bbox_stop"] == [12, 25, 40]
   and md["kind"] == "box" and md["shape"] == [20, 30, 40] and md["n_voxels"] == r.n_voxels,
   f"to_metadata JSON-able, exclusive stop: {md}")
md = Roi.from_mask(lab, label=2, name="lab").to_metadata()
ok(json.loads(json.dumps(md)) == md and md["source"] == {"layer": "lab", "label": 2}, "mask metadata")
ok(r.describe() == "box z 5-11, y 0-24, x 10-39 (5,250 voxels, 22%)", f"describe box: {r.describe()}")
big = Roi.box((100, 1000, 1000), (10, 100, 0), (50, 300, 256))
ok(big.describe() == "box z 10-49, y 100-299, x 0-255 (2.05 M voxels, 2%)", f"describe big box: {big.describe()}")
ok(Roi.from_mask(lab, 2, "lab").describe() == "mask 'lab' label 2 (37 voxels, 0.15%) in z 2-10, y 3-20, x 4-30",
   f"describe mask: {Roi.from_mask(lab, 2, 'lab').describe()}")
ok(Roi.box((10, 20), (1, 2), (3, 4)).describe().startswith("box y 1-2, x 2-3"), "describe 2D uses y, x")
ok([R.roi_name(s) for s in ("x.tif", "x", "x_roi.tif", "FINAL_adaptive_seed.tif", "dog_preview_0.5-1.2",
                            "a.nii.gz", "seed.TIFF")]
   == ["x_roi.tif", "x_roi", "x_roi.tif", "FINAL_adaptive_seed_roi.tif", "dog_preview_0.5-1.2_roi",
       "a_roi.nii.gz", "seed_roi.TIFF"], "roi_name")
ok([R.roi_name(s) for s in ("sample_roi1", "sample_roi1.tif", "my_roi_data", "x_roi")]
   == ["sample_roi1_roi", "sample_roi1_roi.tif", "my_roi_data_roi", "x_roi"],
   "roi_name: '_roi' elsewhere in the name still gets the suffix (no clash with whole-image outputs)")
ok(R.tophat_halo(3) == 6, "tophat_halo = 2 * size")
sb = Roi.box((691, 161, 122), (250, 90, 50), (451, 161, 122))
ok(sb.sizes == (201, 71, 72) and sb.kind_word == "box" and sb.short() == "box 201 × 71 × 72 (8%)",
   f"short(): {sb.short()}")
sm = Roi.from_mask(lab_m, ids=[3, 7], name="m")
ok(sm.kind_word == "mask" and sm.as_box(5).kind_word == "bbox" and sm.short().startswith("mask 26 × 8 × 46 (")
   and sm.as_box(5).short().startswith("bbox "), f"short() of a mask / its bbox: {sm.short()} / {sm.as_box(5).short()}")
ok([R.format_percent(f) for f in (0.08, 0.0005, 0, 1, 0.625)] == ["8%", "0.05%", "0%", "100%", "62%"], "format_percent")
ok(Roi.box((10, 20), (1, 2), (3, 4)).axis_names == ("y", "x") and sb.axis_names == ("z", "y", "x"), "axis_names")
sc = R.write_sidecar(r, os.path.join(TMP, "sc"))
d = json.loads(Path(sc).read_text())
ok(d["crop_offset"] == [4, 0, 9] and d["crop_shape"] == [9, 26, 31] and d["bbox_start"] == [5, 0, 10],
   "roi.json sidecar: metadata + crop offset / shape (margin 1)")

if "--basics" in sys.argv:
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
    sys.exit(1 if FAILED else 0)

# ======================================================================
print("\n== 2. local operations")
from sprout_core import boundary as B  # noqa: E402
from sprout_core.sprout_core import erosion_binary_img_on_sub, get_ccomps_with_size_order  # noqa: E402
from napari_sprout.utils.histogram_threshold_widget import (  # noqa: E402
    threshold_erode_cc, threshold_erode_cc_in_roi)

from scipy import ndimage as ndi  # noqa: E402
img3 = ndi.gaussian_filter(rng.random((34, 40, 46)) * 255, 1).astype(np.uint8)
img3 = np.clip(img3.astype(int) * 3 - 200, 0, 255).astype(np.uint8)
img2 = ndi.gaussian_filter(rng.random((70, 80)) * 255, 1).astype(np.uint8)
lab3 = np.zeros(img3.shape, np.uint8)
lab3[10:24, 12:30, 15:38] = (rng.random((14, 18, 23)) > 0.4) * 2
rois = {
    "interior box": Roi.box(img3.shape, (9, 10, 12), (22, 28, 33)),
    "face-touching box": Roi.box(img3.shape, (0, 5, 20), (15, 40, 40)),
    "mask": Roi.from_mask(lab3, 2, "m"),
    "2D box": Roi.box(img2.shape, (10, 20), (50, 61)),
}
noise3 = rng.integers(0, 256, img3.shape).astype(np.uint8)   # makes the 2s reach matter often
noise2 = rng.integers(0, 256, img2.shape).astype(np.uint8)
gpu_ok, gpu_desc = B.gpu_status()
n = bad = neg = nneg = 0
combo_differs = {}
for kind in ("black", "white"):
    for shp in ("ball", "cube", "octahedron"):
        for s in (1, 3):
            combo_differs[(kind, shp, s)] = False
            for rname, roi in rois.items():
                for im in ((img2, noise2) if roi.ndim == 2 else (img3, noise3)):
                    ref, _ = B.tophat(im, kind, shp, s, "cpu")
                    out, info = R.run_local(lambda a: B.tophat(a, kind, shp, s, "cpu"), roi, im, R.tophat_halo(s))
                    n += 1
                    bad += not (same(out, ref[roi.bbox()]) and info["device"] == "cpu")
                # negative control on noise: one voxel less of halo is not enough
                im = noise2 if roi.ndim == 2 else noise3
                ref, _ = B.tophat(im, kind, shp, s, "cpu")
                short = R.run_local(lambda a: B.tophat(a, kind, shp, s, "cpu")[0], roi, im, 2 * s - 1)
                nneg += 1
                d = not np.array_equal(short, ref[roi.bbox()])
                neg += d
                combo_differs[(kind, shp, s)] |= d
                if not d:   # the 2s reach of a ball / octahedron is a single apex voxel per axis
                    print(f"     halo 2s-1 happens to be identical: {kind} {shp} s={s} {rname}")
ok(bad == 0, f"top-hat run_local(bbox + 2s) == whole-image top-hat on the bbox: {n - bad}/{n} "
             f"(black/white x ball/cube/octahedron x size 1,3 x interior/face-touching/mask/2D, smooth + noise)")
ok(all(combo_differs.values()) and neg >= 0.8 * nneg,
   f"negative control: halo 2s-1 differs from the whole image in {neg}/{nneg} cases (noise images), "
   f"and in at least one ROI for each of the {len(combo_differs)} kind x shape x size")
if gpu_ok:
    roi = rois["interior box"]
    ref, _ = B.tophat(img3, "black", "ball", 3, "cpu")
    out, info = R.run_local(lambda a: B.tophat(a, "black", "ball", 3, "gpu"), roi, img3, 6)
    ok(same(out, ref[roi.bbox()]) and info["device"] == "cuda", f"GPU top-hat on bbox + halo == CPU whole image")
else:
    print(f"SKIP GPU top-hat ({gpu_desc})")
out = R.run_local(lambda a: a + 1, rois["mask"], img3, 2)
ok(same(out, (img3 + 1)[rois["mask"].bbox()]), "run_local with a plain (non-tuple) result")


def old_threshold_erode_cc(img, lower=None, upper=None, erosion_steps=0, cc_n=None, cancel=None):
    """Copy of threshold_erode_cc before the ROI was added (reference)."""
    binary = B.threshold_mask(img, lower, upper, cancel=cancel).astype(bool)
    for _ in range(erosion_steps):
        binary = erosion_binary_img_on_sub(binary)
    if not cc_n:
        return binary.astype(np.uint8)
    labels, _ = get_ccomps_with_size_order(binary, cc_n)
    return labels.astype(np.uint16 if cc_n < 65536 else np.uint32)


tc = {"interior box": rois["interior box"], "face-touching box": rois["face-touching box"],
      "mask": rois["mask"], "2D box": rois["2D box"],
      "2D mask": Roi.from_mask((img2 > 120).astype(np.uint8) * (np.indices(img2.shape)[1] > 30), None, "m2")}
n = bad = n0 = bad0 = 0
for ero in range(4):
    for cc in (None, 5):
        for lo, hi in ((100, None), (60, 200)):
            for rname, roi in tc.items():
                im = img2 if roi.ndim == 2 else img3
                ref = threshold_erode_cc(im, lo, hi, ero, cc, allowed=roi.mask_full())
                n += 1
                bad += not same(threshold_erode_cc_in_roi(im, roi, lo, hi, ero, cc), ref)
            for im in (img2, img3):
                n0 += 1
                bad0 += not same(threshold_erode_cc(im, lo, hi, ero, cc), old_threshold_erode_cc(im, lo, hi, ero, cc))
ok(bad == 0, f"threshold_erode_cc_in_roi == threshold_erode_cc(allowed=roi.mask_full()): {n - bad}/{n} "
             f"(erosion 0..3, CC None/5, box/mask, 3D/2D)")
ok(bad0 == 0, f"allowed=None is the old code path, byte-identical: {n0 - bad0}/{n0}")
a = threshold_erode_cc(img3, 100, None, 1, 5, allowed=rois["mask"].mask_full())
b = threshold_erode_cc(img3, 100, None, 1, 5)
ok(not np.array_equal(a, b), "negative control: allowed= changes the result (the ROI acts as a wall)")

if "--local" in sys.argv:
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
    sys.exit(1 if FAILED else 0)

# ======================================================================
print("\n== 3. Seeds in an ROI == whole image with boundary | outside-ROI")
import make_adaptive_seed as MA  # noqa: E402
import make_grow as MG  # noqa: E402
import make_seeds as MS  # noqa: E402
import tifffile  # noqa: E402
from sprout_core.sprout_core import cast_labels  # noqa: E402

_n_out = [0]


def outdir():
    _n_out[0] += 1
    return os.path.join(TMP, f"o{_n_out[0]}")


def wall_of(roi, boundary):
    wall = ~roi.mask_full()
    if boundary is not None:
        wall |= np.asarray(boundary) != 0
    return wall


def seeds_ref(fn, roi, img, boundary, **kw):
    return quiet(fn, img=img, boundary=wall_of(roi, boundary), output_folder=outdir(), **kw)


def seeds_roi(fn, roi, img, boundary, **kw):
    return quiet(R.seeds_in_roi, fn, roi, img, boundary, output_folder=outdir(), **kw)


def seeds_margin0(fn, roi, img, boundary, **kw):
    """Negative control: crop exactly the bbox (no wall ring)."""
    img_c, wall_c, _ = R.wall_inputs(roi, img, boundary, margin=0)
    res = quiet(fn, img=img_c, boundary=wall_c, output_folder=outdir(), **kw)
    return R.paste_seeds(roi, res[0], margin=0)


def blobs(shape, sigma, seed_):
    g = np.random.default_rng(seed_)
    a = ndi.gaussian_filter(g.random(shape), sigma)
    a = (a - a.min()) / (a.max() - a.min())
    return (a * 255).astype(np.uint8)


S3 = (32, 40, 48)
im3 = blobs(S3, 2.5, 11)
im2 = blobs((90, 110), 3, 12)
bnd3 = np.zeros(S3, np.uint8)
bnd3[:, 21, :] = 1
bnd3[16, :, 10:30] = 1
bnd2 = np.zeros(im2.shape, np.uint8)
bnd2[:, 55] = 1
mlab = np.zeros(S3, np.uint8)
zz, yy, xx = np.indices(S3)
mlab[((zz - 14) ** 2 / 100 + (yy - 22) ** 2 / 200 + (xx - 25) ** 2 / 300) < 1] = 4
mlab[mlab == 0] = (rng.random(S3)[mlab == 0] > 0.97) * 1
mlab2 = np.zeros(im2.shape, np.uint8)
mlab2[np.hypot(*(np.indices(im2.shape) - np.array([45, 50])[:, None, None])) < 30] = 1
R3 = {"interior box": Roi.box(S3, (5, 8, 9), (26, 33, 40)),
      "border box": Roi.box(S3, (0, 10, 0), (20, 40, 30)),
      "mask": Roi.from_mask(mlab, 4, "m")}
R2 = {"2D box": Roi.box(im2.shape, (10, 0), (80, 70)),
      "2D mask": Roi.from_mask(mlab2, None, "m2")}

thr = dict(thresholds=[120, 140], upper_thresholds=[255, 255])
FP3 = {"None": None, "ball*2": ["ball"] * 2, "cube": ["cube"] * 2, "ball_XY": ["ball_XY"] * 2,
       "2XZ_1Y": ["2XZ_1Y"] * 2, "mixed planes": ["ball_XY", "ball_XZ"]}
FP2 = {"None": None, "ball (disk)": ["ball"] * 2, "cube (square)": ["cube"] * 2, "X/Y": ["X", "Y"]}
SEGS = (1, 5, None)
n = bad = neg = nneg = 0
for rois_, fps, im, bnd in ((R3, FP3, im3, bnd3), (R2, FP2, im2, bnd2)):
    for rname, roi in rois_.items():
        for fname, fp in fps.items():
            for segs in SEGS:
                if QUICK and segs == 1:
                    continue
                for b in (None, bnd):
                    kw = dict(thr, erosion_steps=2, segments=segs, num_threads=2, footprints=fp,
                              base_name="t", return_for_napari=True)
                    ref = seeds_ref(MS.make_seeds, roi, im, b, **kw)[0]
                    got = seeds_roi(MS.make_seeds, roi, im, b, **kw)[0]
                    n += 1
                    if not (same_dicts(got, ref) and len(ref) > 0):
                        bad += 1
                        print(f"     differs: {rname} footprints {fname} segments {segs} boundary {b is not None}")
                    if fname in ("None", "ball*2", "ball (disk)") and segs == 5:
                        nneg += 1
                        neg += not same_dicts(seeds_margin0(MS.make_seeds, roi, im, b, **kw), ref)
ok(bad == 0, f"make_seeds in ROI == whole image with the ROI as a wall (keys, values, dtype): {n - bad}/{n} "
             f"(3D + 2D; interior / border-touching / mask; footprints None, ball, cube, ball_XY, 2XZ_1Y, mixed, "
             f"2D ball (disk), cube (square), X/Y; segments {'5, None' if QUICK else '1, 5, None'}; with/without boundary)")
ok(neg >= 0.75 * nneg and neg > 0, f"negative control: crop margin 0 differs in {neg}/{nneg} cases")

# every erosion step, outside == 0, sizes
roi = R3["interior box"]
got = seeds_roi(MS.make_seeds, roi, im3, bnd3, thresholds=[110], upper_thresholds=[255], erosion_steps=3,
                segments=None, num_threads=1, footprints=["ball"] * 3, base_name="t", return_for_napari=True)[0]
out = ~roi.mask_full()
ok(len(got) == 4 and all(not v[out].any() for v in got.values()) and any(v.max() > 1 for v in got.values()),
   f"all {len(got)} erosion steps returned; 0 outside the ROI; several regions per step")

# adaptive
KA = dict(num_threads=2, base_name="a", return_for_napari=True, save_every_iter=True, min_size=3)
n = bad = neg = nneg = 0
cases = []
for rname, roi in {**R3, **R2}.items():
    im, bnd = (im2, bnd2) if roi.ndim == 2 else (im3, bnd3)
    fp = ["ball"] * 3
    for b in (None, bnd):
        cases.append((rname, roi, im, b, "ero", MA.make_adaptive_seed_ero,
                      dict(threshold=110, upper_threshold=255, erosion_steps=3, segments=8, footprints=fp)))
        cases.append((rname, roi, im, b, "ero split_size_limit", MA.make_adaptive_seed_ero,
                      dict(threshold=110, upper_threshold=255, erosion_steps=3, segments=8, footprints=fp,
                           split_size_limit=(30, 3000))))
        cases.append((rname, roi, im, b, "thre", MA.make_adaptive_seed_thre,
                      dict(thresholds=[100, 120, 140], upper_thresholds=[255] * 3, erosion_steps=1, segments=8,
                           footprints=["ball"])))
cases.append(("interior box", R3["interior box"], im3, None, "ero split_convex_hull_limit", MA.make_adaptive_seed_ero,
              dict(threshold=110, upper_threshold=255, erosion_steps=3, segments=8, footprints=["ball"] * 3,
                   split_convex_hull_limit=(20, 5000))))
for rname, roi, im, b, what, fn, kw in cases:
    if QUICK and b is not None and what != "thre":
        continue
    ref = seeds_ref(fn, roi, im, b, **KA, **kw)[0]
    got = seeds_roi(fn, roi, im, b, **KA, **kw)[0]
    n += 1
    inter = sum(k.startswith("INTER") for k in ref)
    if not (same_dicts(got, ref) and inter > 0):
        bad += 1
        print(f"     differs: adaptive {what} {rname} boundary {b is not None} ({inter} INTER)")
    if what in ("ero", "thre") and b is None:
        nneg += 1
        neg += not same_dicts(seeds_margin0(fn, roi, im, b, **KA, **kw), ref)
ok(bad == 0, f"adaptive seeds (erosion / thresholds, incl. every INTER output, split_size_limit, one "
             f"split_convex_hull_limit) in ROI == whole-image wall run: {n - bad}/{n}")
ok(neg >= 0.75 * nneg and neg > 0, f"negative control (adaptive): crop margin 0 differs in {neg}/{nneg} cases")
try:
    quiet(R.seeds_in_roi, MA.make_adaptive_seed_ero, Roi.box(S3, (0, 0, 0), (2, 2, 2)), np.zeros(S3, np.uint8),
          None, output_folder=outdir(), threshold=100, upper_threshold=255, erosion_steps=1, segments=3, **KA)
    ok(False, "adaptive with nothing inside the ROI raises")
except RuntimeError as e:
    ok("No components found" in str(e), f"adaptive with nothing inside the ROI raises the usual error: {e}")

# real data: dog, 3 boxes incl. border-touching
dog = tifffile.imread(ROOT / "demo_data" / "dog_img.tif")
dseed = tifffile.imread(ROOT / "demo_data" / "dog_seed.tif")
if QUICK:
    dog, dseed = dog[150:450], dseed[150:450]
DR = {"dog box mid": Roi.box(dog.shape, (100, 20, 15), (260, 140, 110)),
      "dog box border": Roi.box(dog.shape, (0, 0, 30), (120, 161, 122)),
      "dog box high z": Roi.box(dog.shape, (dog.shape[0] - 150, 40, 0), (dog.shape[0], 120, 70))}
n = bad = 0
for rname, roi in DR.items():
    for fname, fp in (("None", None), ("ball_XY", ["ball_XY"] * 2), ("2XZ_1Y", ["2XZ_1Y"] * 2)):
        if QUICK and fname != "ball_XY":
            continue
        kw = dict(thresholds=[150], upper_thresholds=[255], erosion_steps=2, segments=10, num_threads=4,
                  footprints=fp, base_name="d", return_for_napari=True)
        n += 1
        if not same_dicts(seeds_roi(MS.make_seeds, roi, dog, None, **kw)[0],
                          seeds_ref(MS.make_seeds, roi, dog, None, **kw)[0]):
            bad += 1
            print(f"     differs: dog make_seeds {rname} {fname}")
    for fn, kw in ((MA.make_adaptive_seed_ero, dict(threshold=150, upper_threshold=255, erosion_steps=3,
                                                    segments=10, footprints=["ball"] * 3)),
                   (MA.make_adaptive_seed_thre, dict(thresholds=[130, 150, 170], upper_thresholds=[255] * 3,
                                                     erosion_steps=2, segments=10, footprints=["ball"] * 2))):
        if QUICK and fn is MA.make_adaptive_seed_thre:
            continue
        n += 1
        if not same_dicts(seeds_roi(fn, roi, dog, None, **KA, **kw)[0], seeds_ref(fn, roi, dog, None, **KA, **kw)[0]):
            bad += 1
            print(f"     differs: dog {fn.__name__} {rname}")
ok(bad == 0, f"dog end-to-end (3 boxes incl. border-touching): make_seeds + adaptive ero / thre with "
             f"save_every_iter, all INTER + FINAL outputs: {n - bad}/{n}")

# ======================================================================
print("\n== 4. Grow in an ROI == where(roi, whole-image wall run, seeds)")


def grow_kw(**kw):
    d = dict(thresholds=[150, 110], upper_thresholds=[255, 255], dilation_steps=[4, 4], num_threads=4,
             base_name="g", return_for_napari=True, is_sort=False, grow_to_end=False,
             min_growth_size=50, no_growth_max_iter=3)
    d.update(kw)
    return d


def grow_ref(roi, img, seg, boundary, **kw):
    inside = roi.mask_full()
    seg_in = np.where(inside, seg, 0).astype(seg.dtype)
    res, _ = quiet(MG.grow_mp, img=img, seg=seg_in, boundary=wall_of(roi, boundary), output_folder=outdir(), **kw)
    base = cast_labels(seg, np.uint8)
    return {k: np.where(inside, v, base) for k, v in res.items()}


def grow_roi(roi, img, seg, boundary, **kw):
    return quiet(R.grow_in_roi, MG.grow_mp, roi, img, seg, boundary, output_folder=outdir(), **kw)[0]


seg3 = np.zeros(S3, np.uint16)
for lid, (z, y, x) in enumerate([(10, 12, 14), (20, 30, 30), (6, 25, 40), (15, 20, 20), (28, 5, 5)], 1):
    seg3[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2] = lid
seg3[2:4, 36:39, 44:47] = 300            # outside the interior box: forces uint16 in the full seeds
seg2 = np.zeros(im2.shape, np.uint16)
for lid, (y, x) in enumerate([(20, 20), (50, 40), (70, 60), (40, 90)], 1):
    seg2[y - 2:y + 2, x - 2:x + 2] = lid
n = bad = 0
for rname, roi in {**R3, **R2}.items():
    im, sg, bnd = (im2, seg2, bnd2) if roi.ndim == 2 else (im3, seg3, bnd3)
    for threads in (1, 4):
        for b in (None, bnd):
            for extra in ({}, {"to_grow_ids": [1, 3]}, {"min_growth_size": 400, "no_growth_max_iter": 1}):
                if QUICK and threads == 1 and extra:
                    continue
                kw = grow_kw(num_threads=threads, **extra)
                ref = grow_ref(roi, im, sg, b, **kw)
                got = grow_roi(roi, im, sg, b, **kw)
                out = ~roi.mask_full()
                base = cast_labels(sg, np.uint8)
                good = (same_dicts(got, ref) and all(v.dtype == base.dtype for v in got.values())
                        and all(np.array_equal(v[out], base[out]) for v in got.values()))
                n += 1
                if not good:
                    bad += 1
                    print(f"     differs: grow {rname} threads {threads} boundary {b is not None} {extra}")
ok(bad == 0, f"grow_mp in ROI == where(roi, wall run with seeds outside cleared, seeds) for every key, "
             f"dtype == cast_labels(seeds), outside unchanged: {n - bad}/{n} (threads 1/4, boundary, "
             f"to_grow_ids, early stop)")
kw = grow_kw(min_growth_size=400, no_growth_max_iter=1)
got = grow_roi(R3["interior box"], im3, seg3, None, **kw)
normal = grow_roi(R3["interior box"], im3, seg3, None, **grow_kw(no_growth_max_iter=100))
ok(sorted(got) != sorted(normal), f"early stop triggered in the ROI run: {sorted(got)} vs {sorted(normal)}")
ok(got[[k for k in got if k.startswith("FINAL")][0]].dtype == np.uint16, "seed id 300 outside the ROI keeps uint16")

# dog grow: 3 boxes x {normal, aggressive early stop}, seed id 300 forced
dseed300 = cast_labels(dseed, np.uint8).astype(np.uint16)
dseed300[dseed300 == 6] = 300
n = bad = 0
for rname, roi in DR.items():
    for extra in ({}, {"min_growth_size": 3000, "no_growth_max_iter": 1}):
        if QUICK and extra:
            continue
        kw = grow_kw(thresholds=[180, 150, 120], upper_thresholds=[255] * 3, dilation_steps=[3, 3, 3], **extra)
        n += 1
        if not same_dicts(grow_roi(roi, dog, dseed300, None, **kw), grow_ref(roi, dog, dseed300, None, **kw)):
            bad += 1
            print(f"     differs: dog grow {rname} {extra}")
ok(bad == 0, f"dog grow (3 boxes x normal / aggressive early stop, seed id 300): {n - bad}/{n} identical")

# ======================================================================
print("\n== 5. a full-image box ROI == ROI off")
full3, full2 = Roi.box(S3, (0, 0, 0), S3), Roi.box(im2.shape, (0, 0), im2.shape)
kw = dict(thr, erosion_steps=2, segments=5, num_threads=2, footprints=None, base_name="t", return_for_napari=True)
ok(all(same_dicts(seeds_roi(MS.make_seeds, fr, im, b, **kw)[0],
                  quiet(MS.make_seeds, img=im, boundary=b, output_folder=outdir(), **kw)[0])
       for fr, im, b in ((full3, im3, None), (full3, im3, bnd3), (full2, im2, bnd2))),
   "make_seeds: full-image box == no ROI (3D, 3D + boundary, 2D + boundary)")
kwa = dict(KA, threshold=110, upper_threshold=255, erosion_steps=3, segments=8, footprints=["ball"] * 3)
ok(same_dicts(seeds_roi(MA.make_adaptive_seed_ero, full3, im3, bnd3, **kwa)[0],
              quiet(MA.make_adaptive_seed_ero, img=im3, boundary=bnd3, output_folder=outdir(), **kwa)[0]),
   "adaptive (erosion): full-image box == no ROI")
kwt = dict(KA, thresholds=[100, 120, 140], upper_thresholds=[255] * 3, erosion_steps=1, segments=8, footprints=["ball"])
ok(same_dicts(seeds_roi(MA.make_adaptive_seed_thre, full2, im2, None, **kwt)[0],
              quiet(MA.make_adaptive_seed_thre, img=im2, output_folder=outdir(), **kwt)[0]),
   "adaptive (thresholds), 2D: full-image box == no ROI")
kw = grow_kw()
ok(all(same_dicts(grow_roi(fr, im, sg, b, **kw),
                  quiet(MG.grow_mp, img=im, seg=sg, boundary=b, output_folder=outdir(), **kw)[0])
       for fr, im, sg, b in ((full3, im3, seg3, bnd3), (full3, im3, seg3, None), (full2, im2, seg2, None))),
   "grow_mp: full-image box == no ROI")
ref, _ = B.tophat(im3, "black", "ball", 3, "cpu")
ok(same(R.run_local(lambda a: B.tophat(a, "black", "ball", 3, "cpu")[0], full3, im3, 6), ref),
   "top-hat: full-image box == no ROI")

# ======================================================================
print("\n== 6. names, value keys, save / load, histogram cache")
a1 = Roi.box(S3, (5, 8, 9), (26, 33, 40), name="ROI_1")
a2 = Roi.box(S3, (5, 8, 9), (26, 33, 40))
ok(a1.key == a2.key and a1.token != a2.token and a1.key != Roi.box(S3, (5, 8, 9), (26, 33, 41)).key,
   "Roi.key: equal for equal boxes (tokens differ), different otherwise")
mk = Roi.from_mask(mlab, 4, "m")
ok(mk.key != Roi.from_mask(mlab, 4, "m").key, "Roi.key of a mask includes its token")
ok(a1.describe().startswith("ROI_1: box z 5-25") and a1.to_metadata()["name"] == "ROI_1"
   and "name" not in a2.to_metadata() and a1.to_metadata()["key_hash"] == a2.key_hash
   and a1.to_metadata()["wall_margin"] == R.WALL_MARGIN, "name in describe() / metadata; key_hash; wall_margin")
ok(same(a1.mask_full(), a2.mask_full()) and a1.with_name("x").token == a1.token and a1.with_name("x").name == "x",
   "the name changes nothing else")
b1 = Roi.from_metadata(a1.to_metadata())
ok(b1.key == a1.key and b1.name == "ROI_1", "from_metadata(to_metadata()) round trip")
ok(raises(lambda: Roi.from_metadata(mk.to_metadata()), text="only box ROIs") and
   raises(lambda: Roi.from_metadata({"kind": "box", "shape": [1, 2]}), text="no 'bbox_start'"),
   "from_metadata: masks / incomplete records -> RoiError")
ok([R.roi_name(n, s) for n, s in (("seed.tif", "ROI_1"), ("dog_img", "ROI_2"), ("x_ROI_1", "ROI_1"),
                                   ("dog", "my roi/2"))]
   == ["seed_ROI_1.tif", "dog_img_ROI_2", "x_ROI_1", "dog_my_roi_2"], "roi_name with the ROI name as suffix")
ok(R.safe_name("a b/c\\d:e") == "a_b_c_d_e" and R.safe_name("ROI_1") == "ROI_1", "safe_name")
fn = os.path.join(TMP, "rois.json")
R.save_rois(fn, [(a1, {"ref_image": "dog_img", "scale": (2, 0.5, 0.5), "translate": (10, 0, 0)}),
                 (Roi.box((90, 110), (1, 2), (30, 40), name="ROI_2"), {})])
back = R.load_rois(fn)
d = json.loads(Path(fn).read_text())
ok(len(back) == 2 and back[0][0].key == a1.key and back[0][0].name == "ROI_1" and back[1][0].name == "ROI_2"
   and back[0][1] == {"ref_image": "dog_img", "scale": [2.0, 0.5, 0.5], "translate": [10.0, 0.0, 0.0]}
   and d["schema"] == 1 and d["rois"][0]["kind"] == "box", "save_rois / load_rois round trip (name, ref, scale)")
side = R.write_sidecar(a1, os.path.join(TMP, "run_ROI_1"))
(sc_roi, _), = R.load_rois(side)
kw = dict(thr, erosion_steps=2, segments=5, num_threads=2, footprints=["ball"] * 2, base_name="t",
          return_for_napari=True)
ok(sc_roi.key == a1.key and same_dicts(seeds_roi(MS.make_seeds, sc_roi, im3, bnd3, **kw)[0],
                                       seeds_roi(MS.make_seeds, a1, im3, bnd3, **kw)[0]),
   "a run's roi.json loads back to the same ROI: Seeds output byte-identical")

import napari_sprout.utils.histogram_threshold_widget as HTM  # noqa: E402
calls = []
_bh = B.histogram
B.histogram = lambda *a, **k: (calls.append(1), _bh(*a, **k))[1]
try:
    h1 = HTM._cached_histogram(im3, a1)
    h2 = HTM._cached_histogram(im3, a2)            # another Roi object, same box
    n_same = len(calls)
    for i in range(17):
        HTM._cached_histogram(im3, Roi.box(S3, (0, 0, 0), (10 + i, 20, 20)))
    n_many = len(calls)
    HTM._cached_histogram(im3, a2)                # least recently used: dropped
    n_after = len(calls)
finally:
    B.histogram = _bh
ok(n_same == 1 and h1 is h2, "histogram cache keyed by value: two Roi objects of the same box -> one computation")
ok(n_many == 18 and n_after == 19, f"LRU of {HTM._HIST_PER_IMAGE} ROIs per image: the oldest one is computed again")

shutil.rmtree(TMP, ignore_errors=True)
print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
sys.exit(1 if FAILED else 0)
