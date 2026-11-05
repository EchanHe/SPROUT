"""
Shared helpers for check_prompt_core.py / check_prompt_nni.py (not a check itself).

Roi: the real sprout_core.roi.Roi once it is importable, else a small stub with the
signatures of design_roi.md section 1 (only what the Prompt code uses).
FakeSession11 / FakeSession26: record calls; translation-equivariant fake prediction
(balls around positive points minus balls around negatives, applied in order, plus
scribbles and the initial mask).
"""
import itertools
import os

import numpy as np

try:
    if os.environ.get("SPROUT_ROI_STUB"):          # force the stub (to test it)
        raise ImportError("stub forced")
    from sprout_core.roi import Roi, WALL_MARGIN  # noqa: F401
    ROI_IS_STUB = False
except ImportError:
    ROI_IS_STUB = True
    WALL_MARGIN = 1
    _tokens = itertools.count(1)

    class Roi:
        """Stub of sprout_core.roi.Roi (design_roi.md section 1)."""

        def __init__(self, shape, lo, hi, sub_mask, kind, source, token):
            self.shape, self.lo, self.hi = tuple(shape), tuple(lo), tuple(hi)
            self.sub_mask, self.kind, self.source, self.token = sub_mask, kind, source, token

        @classmethod
        def box(cls, shape, lo, hi):
            lo2 = tuple(int(min(max(a, 0), n)) for a, n in zip(lo, shape))
            hi2 = tuple(int(min(max(b, a), n)) for a, b, n in zip(lo2, hi, shape))
            return cls(shape, lo2, hi2, None, "box", {"clipped": lo2 != tuple(lo) or hi2 != tuple(hi)},
                       next(_tokens))

        @classmethod
        def from_mask(cls, labels, label=None, name=None):
            m = (labels != 0) if label is None else (labels == label)
            lo, hi = [], []
            for ax in range(m.ndim):
                nz = np.flatnonzero(np.any(m, axis=tuple(a for a in range(m.ndim) if a != ax)))
                if not len(nz):
                    lo, hi = [0] * m.ndim, [0] * m.ndim
                    break
                lo.append(int(nz[0]))
                hi.append(int(nz[-1]) + 1)
            sub = m[tuple(slice(a, b) for a, b in zip(lo, hi))].copy()
            return cls(m.shape, lo, hi, sub, "mask", {"layer": name, "label": label}, next(_tokens))

        @property
        def ndim(self):
            return len(self.shape)

        @property
        def n_voxels(self):
            if self.sub_mask is None:
                return int(np.prod([b - a for a, b in zip(self.lo, self.hi)]))
            return int(self.sub_mask.sum())

        @property
        def empty(self):
            return self.n_voxels == 0

        def check_shape(self, shape):
            if tuple(shape) != self.shape:
                raise ValueError(f"ROI was set on an image of shape {self.shape}, this image has shape {tuple(shape)}")

        def bbox(self, margin=0, *, shape=None):
            if shape is not None:
                self.check_shape(shape)
            return tuple(slice(max(a - margin, 0), min(b + margin, n)) for a, b, n in zip(self.lo, self.hi, self.shape))

        def offset(self, margin=0):
            return tuple(int(s.start) for s in self.bbox(margin))

        def crop_shape(self, margin=0):
            return tuple(s.stop - s.start for s in self.bbox(margin))

        def crop(self, arr, margin=0):
            self.check_shape(arr.shape)
            return arr[self.bbox(margin)]

        def mask_window(self, index, *, shape=None):
            if shape is not None:
                self.check_shape(shape)
            sl = []
            for i, n in zip(index, self.shape):
                if isinstance(i, slice):
                    a, b, _ = i.indices(n)
                    sl.append(slice(a, b))
                else:
                    sl.append(int(i))
            out_shape = tuple(s.stop - s.start for s in sl if isinstance(s, slice))
            full_win = tuple(s if isinstance(s, slice) else slice(s, s + 1) for s in sl)
            out = np.zeros(tuple(s.stop - s.start for s in full_win), bool)
            inter = []
            for s, a, b in zip(full_win, self.lo, self.hi):
                lo, hi = max(s.start, a), min(s.stop, b)
                if lo >= hi:
                    return out.reshape(out_shape)
                inter.append((lo, hi))
            dst = tuple(slice(lo - s.start, hi - s.start) for (lo, hi), s in zip(inter, full_win))
            if self.sub_mask is None:
                out[dst] = True
            else:
                out[dst] = self.sub_mask[tuple(slice(lo - a, hi - a) for (lo, hi), a in zip(inter, self.lo))]
            return out.reshape(out_shape)

        def mask_crop(self, margin=0):
            return self.mask_window(self.bbox(margin))

        def mask_full(self, *, shape=None):
            return self.mask_window(tuple(slice(0, n) for n in self.shape), shape=shape)

        def contains(self, points):
            p = np.rint(np.asarray(points, dtype=float)).astype(np.int64).reshape(-1, self.ndim)
            inb = np.all((p >= np.array(self.lo)) & (p < np.array(self.hi)), axis=1)
            if self.sub_mask is None:
                return inb
            out = np.zeros(len(p), bool)
            q = p[inb] - np.array(self.lo)
            out[inb] = self.sub_mask[tuple(q.T)]
            return out

        def describe(self):
            rng = ", ".join(f"{ax} {a}-{b - 1}" for ax, a, b in zip("zyx"[-self.ndim:], self.lo, self.hi))
            return f"{self.kind} {rng} ({self.n_voxels:,} voxels) [stub]"

        def to_metadata(self):
            return {"kind": self.kind, "shape": list(self.shape), "bbox_start": list(self.lo),
                    "bbox_stop": list(self.hi), "n_voxels": self.n_voxels, "source": self.source}


# --------------------------------------------------------------------------
# Fake nnInteractive sessions
# --------------------------------------------------------------------------
BALL_R = 4


def _ball_into(m, center, r, value):
    lo = [max(c - r, 0) for c in center]
    hi = [min(c + r + 1, n) for c, n in zip(center, m.shape)]
    if any(a >= b for a, b in zip(lo, hi)):
        return
    zz, yy, xx = np.ogrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    b = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2 <= r * r
    view = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    if value:
        view |= b
    else:
        view &= ~b


def _h(a):
    a = np.ascontiguousarray(np.asarray(a).astype(np.uint8))
    return hash(a.tobytes()) ^ hash(a.shape)


class _FakeBase:
    instances = []

    def _init(self):
        self.calls = []
        self.shape = None
        self.target_buffer = None
        self.init = None
        self.inter = []
        self.loaded_from = None
        self.license = "fake license"
        _FakeBase.instances.append(self)

    def initialize_from_trained_model_folder(self, folder, *a, **k):
        self.loaded_from = folder
        self.calls.append(("init_weights",))

    def set_image(self, image, image_properties=None):
        assert image.ndim == 4
        self.calls.append(("set_image", tuple(image.shape), str(image.dtype)))
        self.image = image
        self.shape = image.shape[1:]
        self.target_buffer = None
        self.init = None
        self.inter = []

    def set_target_buffer(self, buf):
        self.calls.append(("set_target_buffer",))
        self.target_buffer = buf

    def reset_interactions(self):
        self.calls.append(("reset",))
        self.init = None
        self.inter = []
        self._write(np.zeros(self.shape, bool))

    def add_initial_seg_interaction(self, initial_seg, run_prediction=False):
        self.calls.append(("init_seg", _h(initial_seg)))
        self.init = None
        self.inter = []
        self.init = np.asarray(initial_seg) > 0
        self._write(self.init)
        if run_prediction:
            self._predict()

    def add_point_interaction(self, coordinates, include_interaction, run_prediction=True):
        c = tuple(int(round(v)) for v in coordinates)
        self.calls.append(("point", c, bool(include_interaction)))
        self.inter.append(("point", bool(include_interaction), c))
        if run_prediction:
            self._predict()

    def add_scribble_interaction(self, scribble_image, include_interaction, run_prediction=True, **kw):
        self.calls.append(("scribble", _h(scribble_image), bool(include_interaction)))
        self.inter.append(("scribble", bool(include_interaction), np.asarray(scribble_image) > 0))
        if run_prediction:
            self._predict()

    def _predict(self):
        self.calls.append(("predict",))
        m = np.zeros(self.shape, bool) if self.init is None else self.init.copy()
        for kind, inc, data in self.inter:
            if kind == "point":
                _ball_into(m, data, BALL_R, inc)
            elif inc:
                m |= data
            else:
                m &= ~data
        self._write(m)

    def _write(self, m):
        b = self.target_buffer
        if b is None:
            return
        if hasattr(b, "numpy") and not isinstance(b, np.ndarray):
            import torch
            b.copy_(torch.from_numpy(m.astype(np.uint8)))
        else:
            b[...] = m


class FakeSession11(_FakeBase):
    """__init__ signature copied from nnInteractive 1.1.3."""

    def __init__(self, device=None, use_torch_compile=False, verbose=False, torch_n_threads=8,
                 do_autozoom=True, use_pinned_memory=True):
        self._init()
        self.device = device
        self.kwargs = dict(device=device, use_torch_compile=use_torch_compile, verbose=verbose,
                           torch_n_threads=torch_n_threads, do_autozoom=do_autozoom,
                           use_pinned_memory=use_pinned_memory)


class FakeSession26(_FakeBase):
    """__init__ signature copied from nnInteractive 2.6.0."""

    def __init__(self, device=None, use_torch_compile=False, verbose=False, torch_n_threads=8,
                 do_autozoom=True, interactions_storage="auto", enable_undo=True):
        self._init()
        self.device = device
        self.supports_undo = enable_undo
        self.kwargs = dict(device=device, use_torch_compile=use_torch_compile, verbose=verbose,
                           torch_n_threads=torch_n_threads, do_autozoom=do_autozoom,
                           interactions_storage=interactions_storage, enable_undo=enable_undo)

    def undo(self):
        self.calls.append(("undo",))
        return True


def normalize_calls(calls):
    """Engine calls (run_prediction=True) and script calls (False + _predict) -> per-class lists."""
    per, cur = [], None
    for c in calls:
        if c[0] in ("set_image", "set_target_buffer", "init_weights"):
            continue
        if c[0] == "reset":
            cur = []
            per.append(cur)
            continue
        cur.append(c)
    return [c for c in per if c]       # nninter_predict ends with a reset_interactions()


def dice(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s
