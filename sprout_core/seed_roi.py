"""
Seed-derived ROIs for nnInteractive prediction.

For each class, the ROI is the bounding box of that class's seed voxels, grown by a
margin and clipped to the image. nnInteractive then only sees this crop, so look-alike
structures far from the seed (e.g. a leg when predicting an antenna) are not in its
input. Geometry is in voxel indices of the (z, y, x) array; boxes are half-open.

ROIs are returned as box Roi objects (sprout_core.roi), so callers use
roi.bbox() / roi.offset() / roi.crop(arr) / roi.paste(values, region="bbox", ...).
"""
import math

import numpy as np
from scipy import ndimage

from sprout_core.roi import Roi

MARGIN_UNITS = ("voxel", "percent")


def _per_axis(value, ndim, name):
    v = np.asarray(value, dtype=float)
    if v.ndim == 0:
        v = np.full(ndim, float(v))
    if v.shape != (ndim,):
        raise ValueError(f"{name} must be a number or {ndim} numbers (z, y, x), got {value!r}")
    if np.any(v < 0):
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return v


def margin_voxels(lo, hi, margin, unit="voxel"):
    """Margin per axis in voxels.

    unit 'voxel'  : margin is in voxels
    unit 'percent': margin is a percentage of the seed bbox size on that axis (rounded up)
    """
    if unit not in MARGIN_UNITS:
        raise ValueError(f"roi_margin_unit must be one of {MARGIN_UNITS}, got {unit!r}")
    m = _per_axis(margin, len(lo), "roi_margin")
    if unit == "voxel":
        return tuple(int(round(k)) for k in m)
    # round before ceil: 7 / 100 * 100 is 7.000000000000001 in floating point
    return tuple(int(math.ceil(round(k * (h - l) / 100.0, 9))) for k, l, h in zip(m, lo, hi))


def expand_to_min_size(lo, hi, shape, min_size):
    """Grow [lo, hi) symmetrically so each axis is at least min_size (capped at the image
    size); at an image border the box is shifted inwards instead of being cut."""
    ms = _per_axis(min_size, len(shape), "roi_min_size")
    lo, hi = list(lo), list(hi)
    for i, (n, want) in enumerate(zip(shape, ms)):
        want = min(int(want), n)
        size = hi[i] - lo[i]
        if size >= want:
            continue
        extra = want - size
        lo[i] -= extra // 2
        hi[i] += extra - extra // 2
        if lo[i] < 0:
            hi[i] -= lo[i]
            lo[i] = 0
        if hi[i] > n:
            lo[i] -= hi[i] - n
            hi[i] = n
        lo[i] = max(lo[i], 0)
    return tuple(lo), tuple(hi)


def seed_bboxes(seed, labels=None):
    """{label: (lo, hi)} of the seed voxels of each label, in one pass over the seed.
    Labels that do not occur are left out."""
    seed = np.asarray(seed)
    if labels is None:
        labels = [int(v) for v in np.unique(seed) if v != 0]
    labels = sorted({int(v) for v in labels if int(v) != 0})
    if not labels:
        return {}
    # find_objects allocates one slot per id up to the max label; relabel when ids are huge
    if labels[-1] > 10_000:
        lut = {lab: i + 1 for i, lab in enumerate(labels)}
        sub = np.zeros(seed.shape, np.int32)
        for lab, i in lut.items():
            sub[seed == lab] = i
        objs, ids = ndimage.find_objects(sub), labels
    else:
        objs = ndimage.find_objects(seed.astype(np.int64, copy=False) if seed.dtype.kind not in "iu" else seed,
                                    max_label=labels[-1])
        ids = list(range(1, labels[-1] + 1))
    out = {}
    for lab, obj in zip(ids, objs):
        if obj is None or lab not in labels:
            continue
        out[lab] = (tuple(s.start for s in obj), tuple(s.stop for s in obj))
    return out


def seed_roi_from_bbox(shape, lo, hi, margin=32, unit="voxel", min_size=0):
    """Box Roi = seed bbox [lo, hi) + margin, at least min_size per axis, clipped to shape."""
    m = margin_voxels(lo, hi, margin, unit)
    ms = [int(v) for v in _per_axis(min_size, len(shape), "roi_min_size")]
    glo = [l - k for l, k in zip(lo, m)]
    ghi = [h + k for h, k in zip(hi, m)]
    # clipped: the image border cut the margin, or min_size is larger than the image
    clipped = (any(v < 0 for v in glo) or any(v > n for v, n in zip(ghi, shape))
               or any(w > n for w, n in zip(ms, shape)))
    glo = [max(0, v) for v in glo]
    ghi = [min(n, v) for n, v in zip(shape, ghi)]
    glo, ghi = expand_to_min_size(glo, ghi, shape, ms)
    roi = Roi.box(shape, glo, ghi)
    return Roi(roi.shape, roi.lo, roi.hi, None, "box",
               dict(roi.source, clipped=bool(clipped), seed_lo=list(map(int, lo)), seed_hi=list(map(int, hi)),
                    margin=list(m), margin_unit=unit, min_size=ms))


def seed_rois(seed, labels=None, mode="class", margin=32, unit="voxel", min_size=0):
    """ROIs from a seed label map.

    mode 'class': {label: Roi} with one ROI per label
    mode 'union': {label: Roi} where every label maps to the same ROI around all seeds
    Labels without seed voxels are left out (the caller predicts them without a crop).
    """
    if mode not in ("class", "union"):
        raise ValueError(f"roi_mode must be 'class' or 'union', got {mode!r}")
    seed = np.asarray(seed)
    shape = tuple(int(s) for s in seed.shape)
    boxes = seed_bboxes(seed, labels)
    if not boxes:
        return {}
    if mode == "class":
        return {lab: seed_roi_from_bbox(shape, lo, hi, margin, unit, min_size) for lab, (lo, hi) in boxes.items()}
    lo = tuple(min(b[0][i] for b in boxes.values()) for i in range(seed.ndim))
    hi = tuple(max(b[1][i] for b in boxes.values()) for i in range(seed.ndim))
    roi = seed_roi_from_bbox(shape, lo, hi, margin, unit, min_size)
    return {lab: roi for lab in boxes}


def inner_face_fraction(mask_crop, roi):
    """Fraction of foreground voxels of mask_crop (shaped like roi.bbox()) that lie on a
    face of the ROI which is not also an image border. A high value means the prediction
    was probably cut by the ROI (margin too small)."""
    mask_crop = np.asarray(mask_crop) != 0
    total = int(mask_crop.sum())
    if total == 0:
        return 0.0
    on_face = np.zeros(mask_crop.shape, bool)
    for ax, (l, h, n) in enumerate(zip(roi.lo, roi.hi, roi.shape)):
        idx = [slice(None)] * mask_crop.ndim
        if l > 0:
            idx[ax] = 0
            on_face[tuple(idx)] = True
        if h < n:
            idx[ax] = mask_crop.shape[ax] - 1
            on_face[tuple(idx)] = True
    return float((mask_crop & on_face).sum()) / total


def roi_log(roi):
    """JSON/YAML-friendly description of a seed ROI for the run log."""
    s = roi.source
    return {
        "bbox_start": [int(v) for v in roi.lo],
        "bbox_stop": [int(v) for v in roi.hi],
        "crop_shape": [int(h - l) for l, h in zip(roi.lo, roi.hi)],
        "seed_bbox_start": s.get("seed_lo"),
        "seed_bbox_stop": s.get("seed_hi"),
        "margin_voxels": s.get("margin"),
        "margin_unit": s.get("margin_unit"),
        "min_size": s.get("min_size"),
        "clipped": bool(s.get("clipped", False)),
    }
