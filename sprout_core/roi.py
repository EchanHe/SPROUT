"""
Region of interest (ROI) for SPROUT (no Qt here; usable from napari, CLI or batch).

An ROI is either a box or a mask; a box is just a box-shaped mask. Geometry is
in voxel indices of the image array (array axis order), boxes are half-open
[lo, hi) and always clipped to the image. Works for 2D and 3D.

How each operation uses the ROI
-------------------------------
Local operations (threshold, top-hat): computed on the ROI's bounding box plus a
halo, so the result inside the ROI is identical to the whole-image result. For a
mask ROI the image outside the mask is used as context; only the output is
restricted to the mask.  -> run_local

Non-local operations (Seeds: erosion + connected components + top-N; Grow): "the
ROI is a wall", i.e. the same as running on the whole image with
boundary = user boundary | outside-ROI. Implemented by cropping the bounding box
plus WALL_MARGIN, running the unchanged core function on the crop and pasting
the result back. One voxel of margin is enough: skimage's binary_erosion never
erodes from the array border, so the ring of wall voxels around the ROI makes the
ROI edge erode exactly like a boundary.  -> wall_inputs, seeds_in_roi, grow_in_roi

Results are always image-sized: Seeds outside the ROI = 0, Grow outside the ROI
= the input seeds unchanged.
"""
import itertools
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np

META_KEY = "sprout_roi"
WALL_MARGIN = 1

_tokens = itertools.count(1)


class RoiError(ValueError):
    """User-facing reason an ROI cannot be used."""


def tophat_halo(size):
    """Halo that makes a top-hat on a crop exact: closing / opening reach 2 * size."""
    return 2 * int(size)


def _axis_names(ndim):
    return ("z", "y", "x")[-ndim:] if ndim <= 3 else tuple(f"axis{i}" for i in range(ndim))


def _margins(margin, ndim):
    m = np.broadcast_to(np.asarray(margin, int), (ndim,))
    return tuple(int(v) for v in m)


@dataclass(frozen=True, eq=False)
class Roi:
    """An immutable snapshot of a region of interest.

    shape    : shape of the image the ROI was defined on
    lo, hi   : bounding box, half-open, clipped to shape
    sub_mask : bool array of shape hi - lo, or None for a solid box
    kind     : 'box' | 'mask'
    source   : box: {'clipped': bool}, plus {'from_mask': <mask source>, 'margin': m}
               for the bounding box of a mask (as_box); mask: {'layer': name,
               'label': int | None} (+ 'ids': [int, ...] for several ids)
    token    : unique per snapshot
    """
    shape: tuple
    lo: tuple
    hi: tuple
    sub_mask: object = None
    kind: str = "box"
    source: dict = field(default_factory=dict)
    token: int = field(default_factory=lambda: next(_tokens))

    def __post_init__(self):
        if self.sub_mask is None:
            n = int(np.prod([h - l for l, h in zip(self.lo, self.hi)], dtype=np.int64))
        else:
            n = int(np.count_nonzero(self.sub_mask))
        object.__setattr__(self, "_n_voxels", n)

    # ------------------------------------------------------------ creation
    @classmethod
    def box(cls, shape, lo, hi):
        """Box [lo, hi) clipped to shape; may be empty."""
        shape = tuple(int(s) for s in shape)
        if len(lo) != len(shape) or len(hi) != len(shape):
            raise ValueError(f"box needs {len(shape)} coordinates, got lo={lo}, hi={hi}")
        lo_c = tuple(int(min(max(int(l), 0), n)) for l, n in zip(lo, shape))
        hi_c = tuple(int(min(max(int(h), l), n)) for h, l, n in zip(hi, lo_c, shape))
        clipped = lo_c != tuple(int(v) for v in lo) or hi_c != tuple(int(v) for v in hi)
        return cls(shape, lo_c, hi_c, None, "box", {"clipped": bool(clipped)})

    @classmethod
    def from_mask(cls, labels, label=None, name=None, ids=None):
        """Voxels of a label image that are != 0 (default), == label, or in ids
        (several label ids); may be empty."""
        labels = np.asarray(labels)
        if ids is not None:
            ids = sorted({int(i) for i in ids})
            if len(ids) == 1:
                label, ids = ids[0], None
        if ids is not None:
            m = np.isin(labels, ids)
        else:
            m = labels != 0 if label is None else labels == label
        shape = tuple(int(s) for s in labels.shape)
        lo, hi = bbox_of_mask(m)
        sub = m[tuple(slice(l, h) for l, h in zip(lo, hi))].copy()
        source = {"layer": name, "label": None if label is None else int(label)}
        if ids is not None:
            source["ids"] = ids
        return cls(shape, tuple(lo), tuple(hi), sub, "mask", source)

    def as_box(self, margin=0):
        """The bounding box of this ROI grown by margin voxels (clipped to the image),
        as a box ROI. Empty stays empty."""
        if self.empty:
            return Roi(self.shape, self.lo, self.lo, None, "box",
                       {"clipped": False, "from_mask": dict(self.source), "margin": int(margin)})
        m = _margins(margin, self.ndim)
        box = Roi.box(self.shape, [l - k for l, k in zip(self.lo, m)], [h + k for h, k in zip(self.hi, m)])
        src = dict(box.source, from_mask=dict(self.source) if self.kind == "mask" else None, margin=int(margin))
        return Roi(box.shape, box.lo, box.hi, None, "box", src)

    # ---------------------------------------------------------- properties
    @property
    def ndim(self):
        return len(self.shape)

    @property
    def n_voxels(self):
        return self._n_voxels

    @property
    def empty(self):
        return self._n_voxels == 0

    @property
    def fraction(self):
        total = int(np.prod(self.shape, dtype=np.int64))
        return self._n_voxels / total if total else 0.0

    def _empty_reason(self):
        src = self.source.get("from_mask") or (self.source if self.kind == "mask" else None)
        if src:
            name, label, ids = src.get("layer"), src.get("label"), src.get("ids")
            if ids:
                return f"none of the ids {format_ids(ids)} occurs in '{name}'"
            if label is not None:
                return f"label {label} does not occur in '{name}'"
            return f"'{name}' has no non-zero voxel"
        return "ROI is empty (the box lies outside the image)"

    def check_shape(self, shape):
        """Raise RoiError unless the ROI can be used on an image of this shape."""
        shape = tuple(int(s) for s in shape)
        if shape != self.shape:
            raise RoiError(f"ROI was set on an image of shape {self.shape}, this image has shape {shape}")
        if self.empty:
            raise RoiError(self._empty_reason())
        return self

    # ------------------------------------------------------------ geometry
    def bbox(self, margin=0, *, shape=None):
        """Bounding box grown by margin (int or one per axis), clamped to the image."""
        if shape is not None:
            self.check_shape(shape)
        m = _margins(margin, self.ndim)
        return tuple(slice(max(0, l - k), min(n, h + k))
                     for l, h, k, n in zip(self.lo, self.hi, m, self.shape))

    def offset(self, margin=0):
        return tuple(int(s.start) for s in self.bbox(margin))

    def crop_shape(self, margin=0):
        return tuple(int(s.stop - s.start) for s in self.bbox(margin))

    def _check_array(self, arr):
        if tuple(arr.shape) != self.shape:
            raise RoiError(f"ROI was set on an image of shape {self.shape}, "
                           f"this image has shape {tuple(arr.shape)}")

    def crop(self, arr, margin=0):
        """View of arr on bbox(margin)."""
        self._check_array(arr)
        return arr[self.bbox(margin)]

    def _window(self, index):
        """index -> (starts, stops, kept axes) on the full grid."""
        index = tuple(index)
        if len(index) > self.ndim:
            raise ValueError(f"index {index} has more than {self.ndim} entries")
        index = index + (slice(None),) * (self.ndim - len(index))
        starts, stops, keep = [], [], []
        for i, n in zip(index, self.shape):
            if isinstance(i, slice):
                a, b, step = i.indices(n)
                if step != 1:
                    raise ValueError("mask_window: slices with a step are not supported")
                b = max(a, b)
                keep.append(True)
            else:
                a = int(i)
                if a < 0:
                    a += n
                if not 0 <= a < n:
                    raise IndexError(f"index {i} out of range for size {n}")
                b = a + 1
                keep.append(False)
            starts.append(a)
            stops.append(b)
        return starts, stops, keep

    def mask_window(self, index, *, shape=None):
        """ROI membership on a window of the full grid, without building the full mask.

        index: tuple of ints / slices (step None), e.g. a crop window, bbox(m) or a
        2D display slice. The result has the shape that arr[index] would have.
        """
        if shape is not None:
            self.check_shape(shape)
        starts, stops, keep = self._window(index)
        out = np.zeros([b - a for a, b in zip(starts, stops)], bool)
        lo = [max(a, l) for a, l in zip(starts, self.lo)]
        hi = [min(b, h) for b, h in zip(stops, self.hi)]
        if all(h > l for l, h in zip(lo, hi)):
            dst = tuple(slice(l - a, h - a) for l, h, a in zip(lo, hi, starts))
            if self.sub_mask is None:
                out[dst] = True
            else:
                out[dst] = self.sub_mask[tuple(slice(l - r, h - r) for l, h, r in zip(lo, hi, self.lo))]
        return out[tuple(slice(None) if k else 0 for k in keep)]

    def mask_crop(self, margin=0):
        return self.mask_window(self.bbox(margin))

    def mask_full(self, *, shape=None):
        """Whole-image mask: for tests and small images only."""
        return self.mask_window((slice(None),) * self.ndim, shape=shape)

    # ------------------------------------------------------------ data
    def paste(self, values, *, window=None, margin=0, base=None, fill=0, dtype=None,
              region="roi", inplace=False):
        """Write values (shaped like window) into an image-sized array.

        window  : tuple of slices, default bbox(margin)
        base    : image-sized array to write into (copied unless inplace), or None
                  for np.full(shape, fill, dtype or values.dtype)
        region  : 'roi' writes only ROI voxels of the window, 'bbox' the whole window
        """
        if region not in ("roi", "bbox"):
            raise ValueError("region must be 'roi' or 'bbox'")
        window = self.bbox(margin) if window is None else tuple(window)
        values = np.asarray(values)
        wshape = tuple(len(range(*s.indices(n))) for s, n in zip(window, self.shape))
        if tuple(values.shape) != wshape:
            raise ValueError(f"values have shape {values.shape}, the window has shape {wshape}")
        if base is None:
            out = np.full(self.shape, fill, dtype or values.dtype)
        else:
            self._check_array(base)
            out = base if inplace else np.array(base, copy=True)
        if region == "bbox":
            out[window] = values
        else:
            np.copyto(out[window], values, where=self.mask_window(window))
        return out

    def sample_values(self, arr, target=15_000_000):
        """Values of arr inside the ROI, roughly `target` of them (strided like
        boundary.subsample). The same ROI on arrays of the same shape gives aligned
        samples, so they can be compared element by element.

        A mask ROI is strided by its voxel count, not its bbox; if the stride grid
        misses (nearly) all of a sparse mask, every ROI voxel is taken instead
        (thinned to `target`).
        """
        crop = self.crop(arr)
        basis = crop.size if self.sub_mask is None else self._n_voxels
        step = max(1, int(round((max(basis, 1) / target) ** (1 / crop.ndim))))
        sl = (slice(None, None, step),) * crop.ndim
        sub = crop[sl]
        if self.sub_mask is None:
            return sub.ravel()
        vals = sub[self.sub_mask[sl]]
        if vals.size < min(self._n_voxels, 1000) and step > 1:
            vals = crop[self.sub_mask]
            vals = vals[::max(1, vals.size // target)]
        return vals

    def contains(self, points):
        """Bool per point, (N, ndim) int or float coordinates (rounded)."""
        pts = np.rint(np.asarray(points, float).reshape(-1, self.ndim)).astype(np.int64)
        lo, hi = np.asarray(self.lo), np.asarray(self.hi)
        inside = np.all((pts >= lo) & (pts < hi), axis=1)
        if self.sub_mask is not None and inside.any():
            rel = pts[inside] - lo
            inside[inside] = self.sub_mask[tuple(rel.T)]
        return inside

    # ------------------------------------------------------------ text
    def _extent_text(self):
        return ", ".join(f"{a} {l}-{h - 1}" for a, l, h in zip(_axis_names(self.ndim), self.lo, self.hi))

    def _count_text(self):
        n = self._n_voxels
        count = f"{n / 1e6:.2f} M voxels" if n >= 1_000_000 else f"{n:,} voxels"
        pct = 100 * self.fraction
        return f"{count}, {pct:.0f}%" if pct >= 1 or pct == 0 else f"{count}, {pct:.2g}%"

    def describe(self):
        """One line for logs and status: ends are INCLUSIVE in the text."""
        if self.empty:
            return self._empty_reason()
        if self.kind == "box":
            src = self.source.get("from_mask")
            if src:
                margin = self.source.get("margin") or 0
                extra = f" + margin {margin}" if margin else ""
                return (f"bbox of '{src.get('layer')}' {_which(src)}{extra}: {self._extent_text()} "
                        f"({self._count_text()})")
            return f"box {self._extent_text()} ({self._count_text()})"
        return f"mask '{self.source.get('layer')}' {_which(self.source)} ({self._count_text()}) in {self._extent_text()}"

    def to_metadata(self):
        """JSON-able description (bbox_stop exclusive)."""
        return {"kind": self.kind, "shape": [int(s) for s in self.shape],
                "bbox_start": [int(v) for v in self.lo], "bbox_stop": [int(v) for v in self.hi],
                "n_voxels": int(self._n_voxels), "source": json.loads(json.dumps(self.source, default=str)),
                "description": self.describe()}


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def bbox_of_mask(mask):
    """(lo, hi) half-open bounding box of the True voxels; zeros for an empty mask."""
    mask = np.asarray(mask, bool)
    lo, hi = [], []
    for ax in range(mask.ndim):
        idx = np.flatnonzero(np.any(mask, axis=tuple(a for a in range(mask.ndim) if a != ax)))
        if idx.size == 0:
            return [0] * mask.ndim, [0] * mask.ndim
        lo.append(int(idx[0]))
        hi.append(int(idx[-1]) + 1)
    return lo, hi


def parse_ids(text):
    """'3, 5-9' -> {3, 5, 6, 7, 8, 9} (shared by the ROI Manager and the Prompt tab).
    ValueError if a part is not a number or a range."""
    out = set()
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def format_ids(ids):
    """[3, 5, 6, 7] -> '3,5-7'."""
    ids = sorted({int(i) for i in ids})
    parts, i = [], 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        parts.append(str(ids[i]) if i == j else f"{ids[i]}-{ids[j]}")
        i = j + 1
    return ",".join(parts)


def _which(src):
    ids, label = src.get("ids"), src.get("label")
    if ids:
        return f"ids {format_ids(ids)}"
    return f"label {label}" if label is not None else "non-zero"


_EXT = re.compile(r"(\.(?:tif|tiff|png|jpg|jpeg|npy|nii|nii\.gz|h5|zarr))$", re.IGNORECASE)


def roi_name(name):
    """'x.tif' -> 'x_roi.tif', 'x' -> 'x_roi'; unchanged if the stem already ends in '_roi'.

    Only a trailing '_roi' counts: 'sample_roi1' -> 'sample_roi1_roi', so an ROI run
    never writes to the folder / layer name of a whole-image run.
    """
    m = _EXT.search(name)
    stem, ext = (name[: m.start()], m.group(1)) if m else (name, "")
    if stem.endswith("_roi"):
        return name
    return stem + "_roi" + ext


def write_sidecar(roi, folder, margin=WALL_MARGIN, name="roi.json"):
    """roi.json next to the crop-sized files a run wrote: where the crop sits in the image."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    rec = dict(roi.to_metadata(), crop_offset=list(roi.offset(margin)),
               crop_shape=list(roi.crop_shape(margin)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    return path


def run_local(fn, roi, img, halo):
    """Run a local operation on bbox + halo.

    fn(img_crop) returns out or (out, *extra); out is returned restricted to the
    ROI's bbox (plus *extra, unchanged). Image data outside a mask ROI is used.
    """
    outer = roi.bbox(halo)
    res = fn(roi.crop(img, halo))
    out, extra = (res[0], tuple(res[1:])) if isinstance(res, tuple) else (res, None)
    inner = tuple(slice(b.start - o.start, b.stop - o.start) for b, o in zip(roi.bbox(), outer))
    out = out[inner]
    return out if extra is None else (out, *extra)


def wall_inputs(roi, img, boundary=None, seg=None, margin=WALL_MARGIN):
    """Inputs for a "ROI as a wall" run on the crop bbox(margin).

    Returns (img_c view, wall_c = outside-ROI | boundary != 0, seg_c with the seeds
    outside the ROI cleared, or None).
    """
    img_c = roi.crop(img, margin)
    inside = roi.mask_crop(margin)
    wall_c = ~inside
    if boundary is not None:
        wall_c |= roi.crop(np.asarray(boundary), margin) != 0
    seg_c = None
    if seg is not None:
        seg_c = np.array(roi.crop(np.asarray(seg), margin), copy=True)
        seg_c[~inside] = 0
    return img_c, wall_c, seg_c


def paste_seeds(roi, arrays, margin=WALL_MARGIN):
    """Crop results -> image-sized, 0 outside the ROI, dtype kept."""
    return {k: roi.paste(v, margin=margin, fill=0) for k, v in arrays.items()}


def paste_grow(roi, arrays, seg_full, margin=WALL_MARGIN):
    """Crop results -> image-sized; outside the ROI = the input seeds unchanged.

    The dtype is that of cast_labels(seg_full), which may be wider than the crop
    run's (the crop never holds an id larger than the full seeds).
    """
    from sprout_core.sprout_core import cast_labels
    base = cast_labels(np.asarray(seg_full), np.uint8, copy=False)
    out = {}
    for k, v in arrays.items():
        dt = np.promote_types(base.dtype, np.asarray(v).dtype)
        out[k] = roi.paste(np.asarray(v).astype(dt, copy=False), margin=margin,
                           base=base.astype(dt, copy=False))
    return out


def seeds_in_roi(fn, roi, img, boundary=None, **kw):
    """fn = make_seeds / make_adaptive_seed_* (called with img=, boundary=, **kw);
    the first returned value (dict of seeds) is pasted back to image size."""
    img_c, wall_c, _ = wall_inputs(roi, img, boundary)
    res = fn(img=img_c, boundary=wall_c, **kw)
    return (paste_seeds(roi, res[0]), *res[1:])


def grow_in_roi(grow_fn, roi, img, seg, boundary=None, **kw):
    """grow_fn = make_grow.grow_mp; returns (grows_dict pasted back, log_dict).
    is_sort is forced off: sorting would renumber the seeds inside the ROI only."""
    img_c, wall_c, seg_c = wall_inputs(roi, img, boundary, seg=seg)
    kw["is_sort"] = False
    grows, log = grow_fn(img=img_c, seg=seg_c, boundary=wall_c, **kw)
    return paste_grow(roi, grows, seg), log
