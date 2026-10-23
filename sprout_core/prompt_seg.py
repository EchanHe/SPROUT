"""
Prompt-based segmentation: per-class orchestration, result store, overlap rules
(numpy only; no Qt, no torch).

The model runs on a crop of the image (the whole image, or bbox(ROI) + margin).
Each class is predicted on its own; results are kept per class (tight bounding
box + mask) in a ResultStore and composed into one label image with an overlap
rule. Coordinates: voxel indices (z, y, x) of the full image unless stated.
"""
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from sprout_core.prompt_engine import required_capabilities


# --------------------------------------------------------------------------
# Crop / ROI helpers
# --------------------------------------------------------------------------
def inference_crop(shape, roi=None, margin=32):
    """Window the model sees: the whole image, or roi.bbox(margin) (clamped)."""
    if roi is None:
        return tuple(slice(0, int(s)) for s in shape)
    return tuple(roi.bbox(margin, shape=tuple(shape)))


def roi_mask_in(roi, shape, crop):
    """ROI mask over the crop window (None without ROI)."""
    if roi is None:
        return None
    return roi.mask_window(crop, shape=tuple(shape))


def _crop_start(crop):
    return tuple(int(s.start or 0) for s in crop)


def _crop_shape(crop, shape=None):
    out = []
    for i, s in enumerate(crop):
        stop = s.stop if s.stop is not None else shape[i]
        out.append(int(stop) - int(s.start or 0))
    return tuple(out)


def _rint(v):
    return int(np.rint(float(v)))


def _class_rows(df, class_id):
    if df is None or len(df) == 0:
        return []
    sub = df[df['class_id'] == class_id]
    return [(_rint(z), _rint(y), _rint(x), bool(lb))
            for z, y, x, lb in zip(sub['z'].tolist(), sub['y'].tolist(), sub['x'].tolist(), sub['label'].tolist())]


def _filter_points(rows, crop, valid=None, roi=None):
    """Full-image (z, y, x, label) rows -> crop coordinates; drops points outside the ROI,
    outside the crop or not valid for the model."""
    if not rows:
        return []
    if roi is not None:
        inside = np.asarray(roi.contains(np.array([r[:3] for r in rows], dtype=np.int64)), dtype=bool).reshape(-1)
        rows = [r for r, k in zip(rows, inside) if k]
    start = _crop_start(crop)
    shape = _crop_shape(crop)
    kept = []
    for z, y, x, lb in rows:
        c = (z - start[0], y - start[1], x - start[2])
        if not all(0 <= v < n for v, n in zip(c, shape)):
            continue
        if valid is not None and not valid(*c):
            continue
        kept.append((c[0], c[1], c[2], lb))
    return kept


def class_points(df, class_id, crop, valid=None, roi=None):
    """Points of one class for the model: [(z, y, x, positive)] in CROP coordinates, df order.

    Drops points outside the ROI (roi.contains), outside the crop, or not valid for
    the model (valid(z, y, x) in crop coordinates). Returns (points, n_dropped).
    """
    rows = _class_rows(df, class_id)
    kept = _filter_points(rows, crop, valid=valid, roi=roi)
    return kept, len(rows) - len(kept)


def point_key(df, class_id):
    """Counter of (z, y, x, label) of one class (rounded), to detect point edits."""
    return Counter(_class_rows(df, class_id))


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
@dataclass
class ClassResult:
    class_id: int
    slices: tuple                 # tight bbox, full-image coordinates
    mask: np.ndarray              # bool, shape of `slices`
    n_voxels: int
    points_key: Counter = field(default_factory=Counter)
    mode: str = "scratch"         # 'scratch' | 'refine_warm' | 'refine_cold'
    image_key: Any = None
    prompt_type: str = "points"
    roi_desc: Optional[str] = None
    seconds: float = 0.0
    n_points: int = 0             # points given to the model
    n_dropped: int = 0            # points outside ROI / crop / model's valid region
    mask_key: Any = None          # hash of the mask prompts (initial mask / scribbles); None: none

    def full_mask_in(self, window):
        """This class's mask over `window` (tuple of slices with explicit start/stop)."""
        out = np.zeros(tuple(s.stop - s.start for s in window), bool)
        inter = _intersect(self.slices, window)
        if inter is not None:
            out[_shift(inter, window)] = self.mask[_shift(inter, self.slices)]
        return out


def _intersect(a, b):
    out = []
    for s, t in zip(a, b):
        lo, hi = max(s.start, t.start), min(s.stop, t.stop)
        if lo >= hi:
            return None
        out.append(slice(lo, hi))
    return tuple(out)


def _shift(sl, origin):
    return tuple(slice(s.start - o.start, s.stop - o.start) for s, o in zip(sl, origin))


def _union(a, b):
    if a is None or _empty(a):
        return b
    if b is None or _empty(b):
        return a
    return tuple(slice(min(s.start, t.start), max(s.stop, t.stop)) for s, t in zip(a, b))


def _empty(sl):
    return any(s.stop <= s.start for s in sl)


def _tight_bbox(mask):
    sl = []
    for ax in range(mask.ndim):
        nz = np.flatnonzero(np.any(mask, axis=tuple(a for a in range(mask.ndim) if a != ax)))
        if not len(nz):
            return None
        sl.append(slice(int(nz[0]), int(nz[-1]) + 1))
    return tuple(sl)


def result_from_buffer(class_id, buf, crop, roi_mask=None, **meta):
    """Model output over the crop -> ClassResult (clipped to the ROI mask, tight bbox)."""
    m = np.asarray(buf).astype(bool)          # copy
    if roi_mask is not None:
        m &= roi_mask
    start = _crop_start(crop)
    bb = _tight_bbox(m)
    if bb is None:
        slices = tuple(slice(s, s) for s in start)
        mask = np.zeros((0,) * m.ndim, bool)
    else:
        slices = tuple(slice(b.start + s, b.stop + s) for b, s in zip(bb, start))
        mask = np.ascontiguousarray(m[bb])
    return ClassResult(class_id=int(class_id), slices=slices, mask=mask, n_voxels=int(mask.sum()), **meta)


# --------------------------------------------------------------------------
# Overlap rules
# --------------------------------------------------------------------------
def _rule_min_id(claims, seeds_region):
    """Smaller id wins (== nninter_predict: classes painted in descending id order)."""
    out = None
    for c in sorted(claims, reverse=True):
        m = claims[c]
        if out is None:
            out = np.zeros(m.shape, np.int64)
        out[m] = c
    return out


def _rule_seed(claims, seeds_region):
    """A contested voxel inside the seed of a claiming class keeps that class; else smaller id."""
    out = _rule_min_id(claims, seeds_region)
    if seeds_region is not None and out is not None:
        for c, m in claims.items():
            out[m & (seeds_region == c)] = c
    return out


OVERLAP_RULES = {"seed": _rule_seed, "min_id": _rule_min_id}


def register_overlap_rule(name, fn):
    """fn(claims: {class_id: bool array}, seeds_region or None) -> label array (0 = none).

    compose() calls it on the bounding box of the contested voxels (voxels claimed by
    more than one class) with the claims of every class there; its result replaces
    the output on the claimed voxels of that box.
    """
    OVERLAP_RULES[name] = fn


def _label_dtype(max_id):
    from sprout_core.sprout_core import label_dtype
    return label_dtype(max_id)


class ResultStore:
    """Per-class results (one current + one previous for Revert)."""

    def __init__(self):
        self.results = {}
        self._previous = {}

    def __contains__(self, c):
        return int(c) in self.results

    def __len__(self):
        return len(self.results)

    def get(self, c):
        return self.results.get(int(c))

    def classes(self):
        return sorted(self.results)

    def can_revert(self, c):
        return int(c) in self._previous

    def set(self, r):
        """Store a result; returns the changed region (union of old and new bbox) or None."""
        old = self.results.get(r.class_id)
        self._previous[r.class_id] = old
        self.results[r.class_id] = r
        return _region(old, r)

    def revert(self, c):
        """Back to the result before the last set() (or no result); returns the changed region."""
        c = int(c)
        if c not in self._previous:
            return None
        prev = self._previous.pop(c)
        cur = self.results.pop(c, None)
        if prev is not None:
            self.results[c] = prev
        return _region(cur, prev)

    def remove(self, c):
        c = int(c)
        self._previous.pop(c, None)
        cur = self.results.pop(c, None)
        return _region(cur, None)

    def clear(self):
        self.results.clear()
        self._previous.clear()

    def bbox(self):
        reg = None
        for r in self.results.values():
            reg = _union(reg, r.slices)
        return reg

    def compose(self, shape, dtype=None, rule="seed", seeds=None, region=None):
        """Label image of all stored classes (full `shape`, or only `region` (slices)).

        dtype None -> label_dtype(largest stored id). Classes not stored are 0.
        """
        if rule not in OVERLAP_RULES:
            raise ValueError(f"unknown overlap rule {rule!r}; known: {', '.join(OVERLAP_RULES)}")
        if region is None:
            window = tuple(slice(0, int(s)) for s in shape)
        else:
            window = tuple(slice(int(s.start or 0), int(s.stop if s.stop is not None else n))
                           for s, n in zip(region, shape))
        if dtype is None:
            dtype = _label_dtype(max(self.results) if self.results else 0)
        out = np.zeros(tuple(max(s.stop - s.start, 0) for s in window), dtype)
        if not self.results or _empty(window):
            return out
        count = np.zeros(out.shape, np.uint16)
        hits = []
        for c in sorted(self.results, reverse=True):       # descending: smaller id wins
            r = self.results[c]
            inter = _intersect(r.slices, window)
            if inter is None:
                continue
            m = r.mask[_shift(inter, r.slices)]
            w = _shift(inter, window)
            out[w][m] = c
            count[w] += m
            hits.append(c)
        if rule == "min_id":
            return out
        contested = count > 1
        cb = _tight_bbox(contested)
        if cb is None:
            return out
        sub = tuple(slice(s.start + w.start, s.stop + w.start) for s, w in zip(cb, window))   # full coords
        claims = {}
        for c in hits:
            m = self.results[c].full_mask_in(sub)
            if m.any():
                claims[c] = m
        seeds_region = None if seeds is None else np.asarray(seeds[sub])
        lab = OVERLAP_RULES[rule](claims, seeds_region)
        if lab is not None:
            any_claim = np.zeros(lab.shape, bool)
            for m in claims.values():
                any_claim |= m
            view = out[cb]
            view[any_claim] = lab[any_claim].astype(dtype, copy=False)
        return out

    def overlap_voxels(self):
        """Number of voxels claimed by more than one stored class."""
        reg = self.bbox()
        if reg is None or len(self.results) < 2:
            return 0
        count = np.zeros(tuple(s.stop - s.start for s in reg), np.uint16)
        for r in self.results.values():
            inter = _intersect(r.slices, reg)
            if inter is not None:
                count[_shift(inter, reg)] += r.mask[_shift(inter, r.slices)]
        return int((count > 1).sum())


def _region(a, b):
    sa = None if a is None or _empty(a.slices) else a.slices
    sb = None if b is None or _empty(b.slices) else b.slices
    return _union(sa, sb)


# --------------------------------------------------------------------------
# Update planning
# --------------------------------------------------------------------------
_UNSET = object()


def plan_update(prev, cur_key, active_ok, *, prompt_type=_UNSET, image_key=_UNSET, roi_desc=_UNSET,
                mask_key=_UNSET):
    """How to bring a class up to date after point edits -> (mode, added_points Counter).

    'scratch'      no previous result, points removed or moved, or (when given) the
                   prompt type / image / ROI / mask prompts differ from the previous run
    'refine_warm'  only points added and the engine still holds this class's state
    'refine_cold'  only points added; restart from the previous result as initial mask
    'nothing'      unchanged
    """
    cur_key = Counter(cur_key)
    if prev is None:
        return "scratch", cur_key
    for name, val in (("prompt_type", prompt_type), ("image_key", image_key), ("roi_desc", roi_desc),
                      ("mask_key", mask_key)):
        if val is not _UNSET and getattr(prev, name) != val:
            return "scratch", cur_key
    removed = prev.points_key - cur_key
    added = cur_key - prev.points_key
    if removed:
        return "scratch", cur_key
    if not added:
        return "nothing", Counter()
    return ("refine_warm" if active_ok else "refine_cold"), added


def _take(points_full, added):
    """Rows of points_full (df order) that are in the `added` multiset."""
    left = Counter(added)
    out = []
    for p in points_full:
        if left[p] > 0:
            left[p] -= 1
            out.append(p)
    return out


def _array_key(*arrays):
    import hashlib
    h = hashlib.sha1()
    for a in arrays:
        if a is None:
            h.update(b"none")
            continue
        a = np.ascontiguousarray(a)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.view(np.uint8).data if a.size else b"")
    return h.hexdigest()


# --------------------------------------------------------------------------
# The orchestration loop
# --------------------------------------------------------------------------
def run_classes(engine, img, seeds, df, classes, *, crop, image_key, roi=None, mode="scratch", store=None,
                prompt_type="points", scribbles=None, neg_scribble=False, base_seed=0, cancel=None,
                on_point=None):
    """Predict classes one by one; yields ClassResult or ('skipped', class_id, reason).

    engine     a PromptEngine (loaded)
    img, seeds full-size 3D arrays (seeds are copied for the crop at the start)
    df         prompt table (POINT_COLUMNS), full-image coordinates; may be None/empty
    classes    ids to run, in this order
    crop       window the model sees (inference_crop(...)); image_key identifies the
               image (the engine sees (image_key, crop))
    roi        optional Roi: points outside are dropped, outputs clipped to it
    mode       'scratch' (every class from scratch) or 'update' (store required:
               classes whose points only grew are refined, unchanged ones skipped)
    prompt_type 'points' | 'points+init' (seed of the class as initial mask) |
               'scribble' (scribbles=None: the seeds; else a drawn label array; points
               in df are added after the scribbles). neg_scribble: True = the other
               classes of the scribble source as negative scribble; an array = a drawn
               "not this class" label array.
    base_seed  np.random.seed(base_seed + class_id) before each class (None: no seeding)

    The store is NOT modified here (the caller stores what it receives), so a Stop
    (boundary.Cancelled from the engine) leaves it as it was for the running class.
    """
    from sprout_core.boundary import Cancelled, _is_cancelled

    req = required_capabilities(prompt_type)
    engine.require_prompt_type(prompt_type)
    if mode not in ("scratch", "update"):
        raise ValueError(f"mode must be 'scratch' or 'update', got {mode!r}")
    if mode == "update" and store is None:
        raise ValueError("mode='update' needs the ResultStore of the previous run")
    img = np.asarray(img)
    if img.ndim != 3:
        raise ValueError(f"Prompt segmentation needs a 3D image, got {img.ndim}D")
    if seeds is not None and np.shape(seeds) != img.shape:
        raise ValueError(f"seeds shape {np.shape(seeds)} does not match the image {img.shape}")
    crop = tuple(slice(int(s.start or 0), int(s.stop if s.stop is not None else n)) for s, n in zip(crop, img.shape))
    shape = img.shape

    img_c = np.ascontiguousarray(img[crop])
    seeds_c = None if seeds is None else np.array(seeds[crop], copy=True)
    roi_m = roi_mask_in(roi, shape, crop)
    roi_desc = roi.describe() if roi is not None else None

    use_scribble = "scribble" in req
    scr_c = neg_c = None
    if use_scribble:
        if scribbles is None:
            if seeds_c is None:
                raise ValueError("seed scribbles need seeds")
            scr_c = seeds_c
        else:
            if np.shape(scribbles) != shape:
                raise ValueError(f"scribbles shape {np.shape(scribbles)} does not match the image {shape}")
            scr_c = np.array(scribbles[crop], copy=True)
        if isinstance(neg_scribble, np.ndarray):
            if neg_scribble.shape != shape:
                raise ValueError(f"negative scribbles shape {neg_scribble.shape} does not match the image {shape}")
            neg_c = np.array(neg_scribble[crop], copy=True)
    use_init = prompt_type == "points+init"
    if use_init and seeds_c is None:
        raise ValueError("'points+init' needs seeds")

    engine_key = (image_key, tuple((s.start, s.stop) for s in crop))
    engine.set_image(img_c, engine_key)

    def masked(m):
        if m is None:
            return None
        return (m & roi_m) if roi_m is not None else m

    for c in classes:
        c = int(c)
        if _is_cancelled(cancel):
            raise Cancelled()
        t0 = time.perf_counter()
        rows = _class_rows(df, c)
        key = Counter(rows)
        pts = _filter_points(rows, crop, valid=engine.valid_point, roi=roi)
        n_dropped = len(rows) - len(pts)

        init = pos_s = neg_s = None
        if use_init:
            init = masked(seeds_c == c)
        if use_scribble:
            pos_s = masked(scr_c == c)
            if isinstance(neg_scribble, np.ndarray):
                neg_s = masked(neg_c == c)
            elif neg_scribble:
                neg_s = masked((scr_c != c) & (scr_c != 0))
                if scribbles is None and not pos_s.any():
                    neg_s = None          # as nninter_predict: negatives only with a positive scribble
        mkey = _array_key(init, pos_s, neg_s) if (use_init or use_scribble) else None

        run_mode, run_pts, run_init, warm = "scratch", pts, init, False
        if mode == "update":
            prev = store.get(c)
            plan, added = plan_update(prev, key, engine.active == (engine_key, c), prompt_type=prompt_type,
                                      image_key=image_key, roi_desc=roi_desc, mask_key=mkey)
            if plan == "nothing":
                yield ("skipped", c, "unchanged")
                continue
            if plan != "scratch":
                run_pts = _filter_points(_take(rows, added), crop, valid=engine.valid_point, roi=roi)
                if not run_pts:
                    yield ("skipped", c, "the added points are outside the ROI / crop")
                    continue
                run_mode = plan
                pos_s = neg_s = None
                if plan == "refine_warm":
                    warm, run_init = True, None
                else:
                    run_init = prev.full_mask_in(crop)
                    if roi_m is not None:
                        run_init &= roi_m

        has_prompt = bool(run_pts) or (pos_s is not None and pos_s.any()) or \
            (neg_s is not None and neg_s.any()) or (run_init is not None and run_init.any())
        if not has_prompt:
            yield ("skipped", c, "no prompt inside the ROI / crop" if (rows or use_init or use_scribble)
                   else "no prompt")
            continue

        seed = None if base_seed is None else int(base_seed) + c
        buf = engine.predict_class(run_pts, init_mask=run_init, scribble=pos_s, neg_scribble=neg_s, warm=warm,
                                   cancel=cancel, on_point=on_point, seed=seed, class_id=c)
        yield result_from_buffer(c, buf, crop, roi_m, points_key=key, mode=run_mode, image_key=image_key,
                                 prompt_type=prompt_type, roi_desc=roi_desc,
                                 seconds=round(time.perf_counter() - t0, 3), n_points=len(run_pts),
                                 n_dropped=n_dropped, mask_key=mkey)
