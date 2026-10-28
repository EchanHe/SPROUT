"""The current region of interest (ROI) of a napari viewer, shared by the SPROUT tabs.

There is exactly one *current ROI* per viewer (or none). The ROI Manager edits a
*draft* (a box, or a Labels layer + ids used as a mask or as its bounding box) and
`apply()` ("Set as current ROI") turns the draft into the current ROI; `clear()`
("Clear ROI") sets it to none. Nothing is computed while the draft is edited: a
mask is snapshot only by apply() / update(), in the background.

Tabs opt in one by one ("Use current ROI", roi_widget.RoiUse) and read
`model.current`:

    model = get_roi_model(viewer)        # created on first use, works headless
    model.current                        # sprout_core.roi.Roi or None
    model.changed.connect(...)           # the current ROI was set / cleared / updated

A run keeps the Roi it got: Roi objects are immutable, a new Set makes a new one.

Labels draft preview: while the ROI Manager is open on its Labels page it asks
compute_preview() (debounced) for the draft's mask snapshot: bbox, size and voxel
count to show and draw. The result is cached per (layer, data version, ids) and a
Set with the same key reuses it instead of computing again; a job counter drops
stale results. Nothing here ever makes a tab recompute.

The model lives as long as its viewer and owns the napari event connections the
ROI widgets need; it re-emits them as Qt signals (layer_inserted, layer_removed,
dims_changed), so a closed SPROUT panel leaves no callbacks behind in napari.
Layers are only held by weak reference.
"""
import weakref

import numpy as np
from napari.qt.threading import thread_worker
from qtpy.QtCore import QObject, Signal

from sprout_core.roi import Roi, RoiError, parse_ids

_MODELS = weakref.WeakKeyDictionary()      # viewer -> RoiModel


def get_roi_model(viewer):
    """The viewer's RoiModel, created on first use."""
    model = _MODELS.get(viewer)
    if model is None:
        model = _MODELS[viewer] = RoiModel(viewer)
    return model


def _parse_ids(ids_text):
    """None (all non-zero) or sorted ids; ValueError with a message for the user."""
    if ids_text is None:
        return None
    try:
        ids = sorted(parse_ids(ids_text))
    except ValueError:
        raise ValueError(f"ids: enter numbers like 3,5-7 (got '{ids_text}')") from None
    if not ids:
        raise ValueError("ids: enter at least one id, e.g. 3,5-7")
    return ids


def _weak(obj):
    return None if obj is None else weakref.ref(obj)


def _deref(ref):
    return None if ref is None else ref()


class RoiModel(QObject):
    changed = Signal()            # the current ROI was set / cleared / updated
    draft_changed = Signal()      # the manager's draft was edited
    status_changed = Signal()     # busy / error / "source edited since Set"
    layer_inserted = Signal(object)    # napari events, re-emitted for widgets
    layer_removed = Signal(object)
    dims_changed = Signal()
    preview_changed = Signal()    # the Labels draft preview started / finished

    def __init__(self, viewer):
        super().__init__()
        self._viewer = weakref.ref(viewer)
        # current ROI
        self._current = None
        self._current_spec = None
        self._source = None              # weakref: Labels layer the current ROI was made from
        self._current_ref = None         # weakref: layer whose transform places the current ROI
        self._source_name = None
        self.current_ref_name = None
        self.source_edited = False       # ...edited since Set
        self.source_removed = False
        self.busy = False                # a Set / Update is computing
        self.error = None                # why the last Set / Update failed
        self._job = 0
        self._workers = set()
        # draft
        self.draft_kind = "box"
        self.draft_box = None            # Roi.box in the voxel grid of draft_ref
        self._draft_ref = None           # weakref: image layer giving coordinates / shape
        self.draft_ref_name = None
        self._draft_labels = None        # weakref: Labels layer
        self.draft_ids_text = None       # None = all non-zero, else e.g. "3,5-7"
        self.draft_use = "mask"          # 'mask' | 'bbox'
        self.draft_margin = 0
        self.version = 0
        # Labels draft preview (cached mask snapshot), data versions of watched Labels layers
        self.preview = None              # {'key', 'state': 'computing'|'ok'|'error', 'roi', 'error'}
        self._pjob = 0
        self._after_preview = None       # (job, spec) of a Set waiting for the running preview
        self._versions = weakref.WeakKeyDictionary()
        self._tracked = weakref.WeakSet()

        viewer.layers.events.inserted.connect(self._on_layer_inserted)
        viewer.layers.events.removed.connect(self._on_layer_removed)
        for name in ("current_step", "order", "ndisplay"):
            getattr(viewer.dims.events, name).connect(self._on_dims)

    # ------------------------------------------------------------ current
    @property
    def viewer(self):
        return self._viewer()

    @property
    def current(self):
        """The current ROI (sprout_core.roi.Roi) or None."""
        return self._current

    @property
    def source_layer(self):
        return _deref(self._source)

    @property
    def current_ref(self):
        """The image a box was drawn on / the labels layer a mask came from (or None)."""
        return _deref(self._current_ref)

    def _set_current(self, roi, spec, source=None, ref=None):
        ref = ref if ref is not None else source
        self._current_ref = _weak(ref)
        self.current_ref_name = None if ref is None else ref.name
        if source is not self.source_layer:
            self._watch_source(source)
        self.source_edited = self.source_removed = False
        if roi is self._current and spec == self._current_spec:
            self.status_changed.emit()
            return
        self._current, self._current_spec = roi, spec
        self.version += 1
        self.changed.emit()
        self.status_changed.emit()
        self.draft_changed.emit()        # the "edited" note depends on the current ROI

    def clear(self):
        """'Clear ROI': no current ROI."""
        self._job += 1
        self.busy, self.error = False, None
        self._set_current(None, None)

    # ------------------------------------------------------------ draft
    @property
    def draft_ref(self):
        return _deref(self._draft_ref)

    @property
    def draft_labels(self):
        return _deref(self._draft_labels)

    def _draft_edited(self):
        self.draft_changed.emit()

    def set_draft_kind(self, kind):
        if kind not in ("box", "labels"):
            raise ValueError("kind must be 'box' or 'labels'")
        if kind != self.draft_kind:
            self.draft_kind = kind
            self._draft_edited()

    def set_draft_box(self, shape, lo, hi, ref=None):
        """Box [lo, hi) in the voxel grid of `ref` (an image layer of this shape)."""
        roi = Roi.box(shape, lo, hi)
        if ref is not None:
            self._draft_ref, self.draft_ref_name = _weak(ref), ref.name
        old = self.draft_box
        if old is not None and (old.shape, old.lo, old.hi) == (roi.shape, roi.lo, roi.hi):
            return
        self.draft_box = roi
        self._draft_edited()

    def clear_draft_box(self):
        if self.draft_box is not None:
            self.draft_box = None
            self._draft_edited()

    def set_draft_labels(self, layer=..., ids_text=..., use=..., margin=...):
        """Labels draft: layer, ids_text (None = all non-zero), use 'mask' | 'bbox', margin."""
        if layer is not ...:
            self._draft_labels = _weak(layer)
            self._track(layer)
        if ids_text is not ...:
            self.draft_ids_text = None if ids_text is None else str(ids_text)
        if use is not ...:
            if use not in ("mask", "bbox"):
                raise ValueError("use must be 'mask' or 'bbox'")
            self.draft_use = use
        if margin is not ...:
            self.draft_margin = int(margin)
        self._draft_edited()

    def draft_spec(self):
        """Hashable description of the draft, or None while it is incomplete."""
        if self.draft_kind == "box":
            b = self.draft_box
            return None if b is None else ("box", b.shape, b.lo, b.hi)
        lay = self.draft_labels
        if lay is None:
            return None
        return ("labels", weakref.ref(lay), self.draft_ids_text,
                self.draft_use, self.draft_margin if self.draft_use == "bbox" else 0)

    @property
    def draft_edited(self):
        """True if the draft is complete and differs from the current ROI."""
        spec = self.draft_spec()
        return spec is not None and spec != self._current_spec

    # ------------------------------------------------------------ Set / Update
    def apply(self):
        """'Set as current ROI'. A box is set at once; a Labels draft is snapshot in the
        background (busy meanwhile). Returns False if the draft cannot be used (error)."""
        spec = self.draft_spec()
        if spec is None:
            self._fail("draw a box or click Full extent first" if self.draft_kind == "box"
                       else "choose a Labels layer first")
            return False
        if spec[0] == "box":
            self._job += 1
            self.busy, self.error = False, None
            roi = self.draft_box
            if roi.empty:
                self._fail(roi._empty_reason())
                return False
            self._set_current(roi, spec, ref=self.draft_ref)
            return True
        return self._snapshot(spec)

    def update(self):
        """Re-snapshot the current ROI from its (edited) Labels layer."""
        if self._current_spec is None or self._current_spec[0] != "labels":
            return False
        return self._snapshot(self._current_spec)

    def _fail(self, msg):
        self.busy, self.error = False, msg
        self.status_changed.emit()

    def _snapshot(self, spec):
        _, lref, ids_text, use, margin = spec
        layer = lref()
        if layer is None or self.viewer is None or layer not in self.viewer.layers:
            self._fail("the Labels layer was removed")
            return False
        try:
            ids = _parse_ids(ids_text)
        except ValueError as e:
            self._fail(str(e))
            return False
        key = self._labels_key(layer, ids)
        p = self.preview
        if p is not None and p["key"] == key and p["state"] in ("ok", "error"):   # reuse the preview
            self._job += 1
            self.busy, self.error = False, None
            if p["state"] == "error":
                self._fail(p["error"])
                return False
            roi = p["roi"]
            self._set_current(roi.as_box(margin) if use == "bbox" else roi, spec, source=layer)
            return True
        if p is not None and p["key"] == key and p["state"] == "computing":   # wait for it
            self._job += 1
            self._after_preview = (self._job, spec)
            self.busy, self.error = True, None
            self.status_changed.emit()
            return True
        self._job += 1
        job, data, name = self._job, layer.data, layer.name
        self.busy, self.error = True, None
        self.status_changed.emit()

        def work():
            roi = Roi.from_mask(np.asarray(data), ids=ids, name=name)
            if roi.empty:
                raise RoiError(roi._empty_reason())
            return roi

        def done(roi):
            if self._pjob_key() != key:      # also serves as the preview of this key
                self._pjob += 1
                self.preview = {"key": key, "state": "ok", "roi": roi, "error": None}
                self.preview_changed.emit()
            if job == self._job and self.busy:
                self.busy = False
                self._set_current(roi.as_box(margin) if use == "bbox" else roi, spec, source=lref())

        def failed(e):
            if job == self._job:
                msg = str(e) if isinstance(e, RoiError) else f"cannot use '{name}': {type(e).__name__}: {e}"
                self._fail(msg)

        w = thread_worker(work, ignore_errors=True)()
        w.returned.connect(done)
        w.errored.connect(failed)
        self._workers.add(w)
        w.finished.connect(lambda: self._workers.discard(w))
        w.start()
        return True

    # ------------------------------------------------------------ source watch
    def _watch_source(self, layer):
        old = self.source_layer
        if old is not None:
            for name in ("data", "paint"):
                ev = getattr(old.events, name, None)
                if ev is not None:
                    try:
                        ev.disconnect(self._on_source_edited)
                    except Exception:
                        pass
        self._source = _weak(layer)
        self._source_name = None if layer is None else layer.name
        self._track(layer)
        if layer is not None:
            layer.events.data.connect(self._on_source_edited)
            if hasattr(layer.events, "paint"):
                layer.events.paint.connect(self._on_source_edited)

    def _on_source_edited(self, event=None):
        if not self.source_edited:       # nothing is recomputed: the user clicks Update
            self.source_edited = True
            self.status_changed.emit()

    # ------------------------------------------------------------ Labels draft preview
    def _track(self, layer):
        """Count edits of a Labels layer (its data version: part of the preview cache key)."""
        if layer is None or layer in self._tracked:
            return
        self._tracked.add(layer)
        layer.events.data.connect(self._on_labels_edited)
        if hasattr(layer.events, "paint"):
            layer.events.paint.connect(self._on_labels_edited)

    def _on_labels_edited(self, event=None):
        layer = getattr(event, "source", None)
        if layer is None:
            return
        self._versions[layer] = self._versions.get(layer, 0) + 1
        if layer is self.draft_labels:
            self.draft_changed.emit()        # the preview is stale now (no computing here)

    def _labels_key(self, layer, ids):
        return (weakref.ref(layer), self._versions.get(layer, 0), None if ids is None else tuple(ids))

    def _pjob_key(self):
        p = self.preview
        return None if p is None else p["key"]

    def draft_labels_key(self):
        """Cache key of the Labels draft; ValueError (user text) if it cannot be made."""
        layer = self.draft_labels
        if layer is None or self.viewer is None or layer not in self.viewer.layers:
            raise ValueError("choose a Labels layer")
        return self._labels_key(layer, _parse_ids(self.draft_ids_text))

    def draft_preview(self):
        """The preview of the current Labels draft (state 'computing' / 'ok' / 'error'), or
        None if there is none for it yet (stale or never computed)."""
        try:
            key = self.draft_labels_key()
        except ValueError:
            return None
        p = self.preview
        return p if p is not None and p["key"] == key else None

    def compute_preview(self):
        """Snapshot the Labels draft's mask in the background (no-op if cached / running)."""
        try:
            key = self.draft_labels_key()
            ids = _parse_ids(self.draft_ids_text)
        except ValueError:
            return False
        if self._pjob_key() == key:
            return False
        layer = self.draft_labels
        self._pjob += 1
        job, data, name = self._pjob, layer.data, layer.name
        self.preview = {"key": key, "state": "computing", "roi": None, "error": None}
        self.preview_changed.emit()

        def work():
            return Roi.from_mask(np.asarray(data), ids=ids, name=name)

        def done(roi):
            if job == self._pjob:
                err = roi._empty_reason() if roi.empty else None
                self.preview = {"key": key, "state": "error" if err else "ok", "roi": roi, "error": err}
                self.preview_changed.emit()
                self._finish_after_preview()

        def failed(e):
            if job == self._pjob:
                self.preview = {"key": key, "state": "error", "roi": None,
                                "error": f"cannot use '{name}': {type(e).__name__}: {e}"}
                self.preview_changed.emit()
                self._finish_after_preview()

        w = thread_worker(work, ignore_errors=True)()
        w.returned.connect(done)
        w.errored.connect(failed)
        self._workers.add(w)
        w.finished.connect(lambda: self._workers.discard(w))
        w.start()
        return True

    def _finish_after_preview(self):
        """A Set clicked while the same preview was computing: finish it with the preview."""
        pending, self._after_preview = self._after_preview, None
        if pending is not None and pending[0] == self._job and self.busy:
            self.busy = False
            self._snapshot(pending[1])

    def draft_roi(self):
        """(Roi, layer giving its coordinates) of the draft for display, or None: the Box
        draft, or the previewed Labels draft (its mask, or its bounding box + margin)."""
        if self.draft_kind == "box":
            return None if self.draft_box is None else (self.draft_box, self.draft_ref)
        p = self.draft_preview()
        if p is None or p["state"] != "ok":
            return None
        roi = p["roi"]
        return (roi.as_box(self.draft_margin) if self.draft_use == "bbox" else roi), self.draft_labels

    # ------------------------------------------------------------ napari events
    def _on_layer_inserted(self, event):
        self.layer_inserted.emit(event.value)

    def _on_dims(self, event=None):
        self.dims_changed.emit()

    def _on_layer_removed(self, event):
        lay = event.value
        if lay is not None and lay is self.source_layer:
            self.source_removed = True   # the current ROI is a snapshot: it stays valid
            self.status_changed.emit()
        self.layer_removed.emit(lay)

    # ------------------------------------------------------------ text
    def summary(self):
        return "none" if self._current is None else self._current.describe()

    def source_note(self):
        """'' or a note about the Labels layer the current ROI was made from."""
        if self.source_removed:
            return f"layer '{self._source_name}' was removed since the ROI was set (the ROI is kept)"
        if self.source_edited:
            return "mask layer changed since the ROI was set"
        return ""
