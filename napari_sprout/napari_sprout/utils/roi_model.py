"""One region of interest (ROI) per napari viewer, shared by every SPROUT tab.

The model holds what the user chose (on/off, Box or Mask, the box, the mask
layer and label) and hands out immutable sprout_core.roi.Roi snapshots:

    model = get_roi_model(viewer)        # created on first use, works headless
    roi = model.roi_for(image.shape)     # for a RUN: None if off, else a usable
                                         # Roi, else RoiError with the reason
    roi = model.display_roi(shape)       # for a DISPLAY: None instead of raising
    model.changed.connect(...)           # after any state change

`changed` only fires when the user (or a test) touches the ROI, so tabs that
listen to it behave exactly as before while the ROI is never used.

A run keeps the Roi it got: later edits make a new snapshot and never change a
running job. Mask snapshots are computed in the background (pending meanwhile)
and recomputed, debounced, when the mask layer is painted or replaced.
"""
import weakref

import numpy as np
from napari.qt.threading import thread_worker
from qtpy.QtCore import QObject, QTimer, Signal

from sprout_core.roi import Roi, RoiError

_MODELS = weakref.WeakKeyDictionary()      # viewer -> RoiModel


def get_roi_model(viewer):
    """The viewer's RoiModel, created on first use."""
    model = _MODELS.get(viewer)
    if model is None:
        model = _MODELS[viewer] = RoiModel(viewer)
    return model


def _deref(ref):
    return ref() if ref is not None else None


class RoiModel(QObject):
    changed = Signal()

    MASK_DEBOUNCE_MS = 300

    def __init__(self, viewer):
        super().__init__()
        self._viewer = weakref.ref(viewer)
        self._enabled = False
        self._kind = "box"
        self._ref = None             # weakref to the image layer the box is drawn on
        self._box_roi = None         # Roi snapshot of the box
        self._mask_ref = None        # weakref to the Labels layer
        self._mask_name = None
        self._label = None
        self._mask_roi = None
        self._mask_error = None
        self._pending = False
        self._job = 0
        self._workers = set()
        self.version = 0

        self._mask_timer = QTimer(self)
        self._mask_timer.setSingleShot(True)
        self._mask_timer.setInterval(self.MASK_DEBOUNCE_MS)
        self._mask_timer.timeout.connect(self._start_mask_snapshot)
        viewer.layers.events.removed.connect(self._on_layer_removed)

    # ------------------------------------------------------------ state
    @property
    def viewer(self):
        return self._viewer()

    @property
    def enabled(self):
        return self._enabled

    @property
    def kind(self):
        return self._kind

    @property
    def box(self):
        """(lo, hi) of the box (half-open, clipped), or None."""
        r = self._box_roi
        return None if r is None else (r.lo, r.hi)

    @property
    def box_shape(self):
        r = self._box_roi
        return None if r is None else r.shape

    @property
    def reference(self):
        return _deref(self._ref)

    @property
    def mask_layer(self):
        return _deref(self._mask_ref)

    @property
    def label(self):
        return self._label

    @property
    def pending(self):
        return self._pending

    def _emit(self):
        self.version += 1
        self.changed.emit()

    # ------------------------------------------------------------ setters
    def set_enabled(self, on):
        on = bool(on)
        if on != self._enabled:
            self._enabled = on
            self._emit()

    def set_kind(self, kind):
        if kind not in ("box", "mask"):
            raise ValueError("kind must be 'box' or 'mask'")
        if kind != self._kind:
            self._kind = kind
            self._emit()

    def set_reference(self, layer):
        """Image layer the box is drawn on. A box of another shape is cleared.

        The reference is not part of the ROI itself, so `changed` only fires
        when the box is cleared.
        """
        if layer is self.reference:
            return
        self._ref = None if layer is None else weakref.ref(layer)
        if layer is not None and self._box_roi is not None \
                and tuple(np.shape(layer.data)) != self._box_roi.shape:
            self._box_roi = None
            self._emit()

    def set_box(self, shape, lo, hi):
        roi = Roi.box(shape, lo, hi)
        old = self._box_roi
        if old is not None and (old.shape, old.lo, old.hi) == (roi.shape, roi.lo, roi.hi):
            return
        self._box_roi = roi
        self._emit()

    def clear_box(self):
        if self._box_roi is not None:
            self._box_roi = None
            self._emit()

    def set_mask(self, layer, label=None):
        """Labels layer (all non-zero voxels if label is None, else == label), or None."""
        label = None if label is None else int(label)
        if layer is self.mask_layer and label == self._label and layer is not None:
            return
        old = self.mask_layer
        if old is not None and old is not layer:
            self._disconnect_mask(old)
        self._mask_ref = None if layer is None else weakref.ref(layer)
        self._mask_name = None if layer is None else layer.name
        self._label = label
        self._mask_roi = None
        self._mask_error = None
        self._mask_timer.stop()
        if layer is None:
            self._job += 1
            self._pending = False
        else:
            if old is not layer:
                layer.events.data.connect(self._on_mask_edited)
                if hasattr(layer.events, "paint"):
                    layer.events.paint.connect(self._on_mask_edited)
            self._start_mask_snapshot(emit=False)
        self._emit()

    def _disconnect_mask(self, layer):
        for name in ("data", "paint"):
            ev = getattr(layer.events, name, None)
            if ev is not None:
                try:
                    ev.disconnect(self._on_mask_edited)
                except Exception:
                    pass

    # ------------------------------------------------------- mask snapshot
    def _on_mask_edited(self, event=None):
        self._mask_timer.start()          # debounce painting
        if not self._pending:
            self._pending = True
            self._emit()

    def _start_mask_snapshot(self, emit=True):
        layer = self.mask_layer
        if layer is None:
            return
        self._job += 1
        job, data, label, name = self._job, layer.data, self._label, layer.name
        was_pending, self._pending = self._pending, True

        def work():
            return Roi.from_mask(np.asarray(data), label=label, name=name)

        def done(roi):
            if job == self._job:
                self._mask_roi, self._mask_error, self._pending = roi, None, False
                self._emit()

        def failed(e):
            if job == self._job:
                self._mask_roi, self._pending = None, False
                self._mask_error = f"cannot use mask '{name}': {type(e).__name__}: {e}"
                self._emit()

        w = thread_worker(work, ignore_errors=True)()
        w.returned.connect(done)
        w.errored.connect(failed)
        self._workers.add(w)
        w.finished.connect(lambda: self._workers.discard(w))
        w.start()
        if emit and not was_pending:
            self._emit()

    def _on_layer_removed(self, event):
        lay = event.value
        if lay is self.mask_layer:
            self._job += 1
            self._mask_timer.stop()
            self._pending = False
            self._emit()
        elif lay is self.reference:
            self._ref = None             # the box itself stays valid

    def _mask_present(self):
        lay, viewer = self.mask_layer, self.viewer
        return lay is not None and viewer is not None and lay in viewer.layers

    # ------------------------------------------------------------ snapshots
    def _current(self):
        """The Roi for the current choice; RoiError if it cannot be used (ignores on/off)."""
        if self._kind == "box":
            roi = self._box_roi
            if roi is None:
                raise RoiError("no box drawn yet")
        else:
            if self._mask_ref is None:
                raise RoiError("no mask layer chosen")
            if not self._mask_present():
                raise RoiError(f"mask layer '{self._mask_name}' was removed")
            if self._pending:
                raise RoiError("ROI mask is still being computed")
            if self._mask_error:
                raise RoiError(self._mask_error)
            roi = self._mask_roi
            if roi is None:
                raise RoiError("ROI mask is still being computed")
        if roi.empty:
            raise RoiError(roi._empty_reason())
        return roi

    @property
    def roi(self):
        """Roi, or None if off OR unusable. Never raises."""
        if not self._enabled:
            return None
        try:
            return self._current()
        except RoiError:
            return None

    def roi_for(self, shape):
        """For a run: None if off; the Roi if usable on this shape; else RoiError."""
        if not self._enabled:
            return None
        return self._current().check_shape(shape)

    def display_roi(self, shape):
        """For a display: like roi_for, but None instead of raising."""
        try:
            return self.roi_for(shape)
        except RoiError:
            return None

    @property
    def error(self):
        """Why the ROI cannot be used, or None (also None when off)."""
        if not self._enabled:
            return None
        try:
            self._current()
        except RoiError as e:
            return str(e)
        return None

    def summary(self):
        if not self._enabled:
            return "ROI: off (whole image)"
        try:
            return "ROI: " + self._current().describe()
        except RoiError as e:
            return f"ROI: {e}"
