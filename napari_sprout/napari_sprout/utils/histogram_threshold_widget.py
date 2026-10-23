"""Threshold Preview with a histogram, for the Seeds and Grow tabs.

Drop-in replacement for util_widget.ThresholdWidget (same public methods), with:

* a histogram of the image (computed in the background, "Computing..." meanwhile)
  and a draggable [lower, upper] range; the axis follows the data, not the dtype
* "Live preview on current slice" (checkbox): while dragging, the displayed slice
  shows the threshold instantly (threshold only: erosion and connected components
  are 3D operations and would be wrong on a single slice)
* "Create Preview Layer": threshold (+ erosion + connected components in Seeds)
  on the whole volume, as a NEW Labels layer whose name records the parameters,
  so several settings can be created and compared side by side
* "Add range to table": appends the current range as a row of the parameter table

A handle parked at an axis end means "no limit"; get_thresholds() then returns
the dtype's min/max (what the old widget defaulted to), or +-inf for floats.
"""
import numpy as np
from napari.layers import Image
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_error, show_info
from qtpy.QtCore import QTimer, Signal
from qtpy.QtWidgets import (
    QCheckBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout,
)

from sprout_core import boundary as B
from sprout_core import roi as R
from sprout_core.roi import RoiError
from sprout_core.sprout_core import erosion_binary_img_on_sub, get_ccomps_with_size_order

from .histogram_widget import HistogramRangeWidget
from .roi_model import get_roi_model
from .roi_widget import RoiReminder
from .slice_preview import SlicePreview

PREVIEW_NAME = "Threshold Preview"


def _is_default_full_range_row(param_widget, params):
    """True if the table holds only the row clean_table() adds: [dtype min, dtype max]."""
    if len(params[0]) != 1:
        return False
    try:
        name = param_widget.image_combo.currentText()
        dtype = np.asarray(param_widget.viewer.layers[name].data).dtype
        info = np.iinfo(dtype)
    except Exception:
        return False
    return params[0][0] == info.min and params[1][0] == info.max


def append_threshold_row(param_widget, lower, upper, default_dilation=5):
    """Append [lower, upper] as a new row of a Seeds/Grow parameter table.

    If the table only holds the automatic full-range default row (which would
    select the whole image), that row is replaced instead.
    """
    params = param_widget._get_threshold_params()
    if _is_default_full_range_row(param_widget, params):
        params = tuple([] for _ in params)
    lows, ups = list(params[0]) + [int(round(lower))], list(params[1]) + [int(round(upper))]
    if len(params) == 3:   # Grow table: lower, upper, dilation steps
        steps = list(params[2]) + [params[2][-1] if params[2] else default_dilation]
        param_widget.populate_thresholds_from_list(lows, ups, steps)
    else:
        param_widget.populate_thresholds_from_list(lows, ups)


def threshold_erode_cc(img, lower=None, upper=None, erosion_steps=0, cc_n=None, cancel=None, allowed=None):
    """Same steps as the old 3D preview (util_widget.PreviewWorker):
    threshold -> erosion x n -> keep the cc_n largest connected components.

    lower/upper None = no limit. allowed: optional bool mask, voxels outside it
    act as a wall (ROI). Returns uint8 (0/1) without CC, else a label image.
    """
    binary = B.threshold_mask(img, lower, upper, cancel=cancel).astype(bool)
    if allowed is not None:
        binary &= allowed
    for _ in range(erosion_steps):
        binary = erosion_binary_img_on_sub(binary)
    if not cc_n:
        return binary.astype(np.uint8)
    labels, _ = get_ccomps_with_size_order(binary, cc_n)
    return labels.astype(np.uint16 if cc_n < 65536 else np.uint32)


def threshold_erode_cc_in_roi(img, roi, lower=None, upper=None, erosion_steps=0, cc_n=None, cancel=None):
    """threshold_erode_cc with the ROI as a wall, computed on the ROI's bbox + 1.

    Equal to threshold_erode_cc(img, ..., allowed=roi.mask_full()); image-sized, 0 outside the ROI.
    """
    res = threshold_erode_cc(roi.crop(img, R.WALL_MARGIN), lower, upper, erosion_steps, cc_n, cancel,
                             allowed=roi.mask_crop(R.WALL_MARGIN))
    return roi.paste(res, margin=R.WALL_MARGIN)


class HistogramThresholdWidget(QGroupBox):
    preview_requested = Signal(float, float)       # kept for compatibility; not emitted
    add_to_table_requested = Signal(float, float)

    def __init__(self, viewer, title="Threshold Preview", add_connected_components=False):
        super().__init__(title)
        self.viewer = viewer
        self.add_connected_components = add_connected_components
        self._layer = None
        self._img = None
        self._job = 0
        self._pending = None
        self._last_slice = None
        self._workers = set()
        self._preview = SlicePreview(viewer, PREVIEW_NAME, opacity=0.5)
        # Histogram jobs start on the next event-loop turn: image switches that
        # happen while napari closes (layers removed one by one) start nothing,
        # because the timer dies with the widget.
        self._hist_timer = QTimer(self)
        self._hist_timer.setSingleShot(True)
        self._hist_timer.timeout.connect(self._start_histogram)
        # ROI: histogram, live slice and Create Preview Layer only look inside it
        self._roi_model = get_roi_model(viewer)
        self._roi_timer = QTimer(self)          # next event-loop turn, like the histogram
        self._roi_timer.setSingleShot(True)
        self._roi_timer.timeout.connect(self._on_roi_changed)
        self._roi_model.changed.connect(self._roi_timer.start)
        self._hist_roi = None
        self._auto_range = None
        self._roi_refresh = False
        self._slice_roi = False

        lay = QVBoxLayout(self)
        self.hist = HistogramRangeWidget()
        lay.addWidget(self.hist)
        self.live_cb = QCheckBox("Live preview on current slice")
        self.live_cb.setChecked(True)
        self.live_cb.setToolTip("While dragging, show the threshold on the displayed slice "
                                f"(layer '{PREVIEW_NAME}'). Untick to remove it.")
        lay.addWidget(self.live_cb)
        self.status = QLabel("")
        self.status.setStyleSheet("color: gray")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        if add_connected_components:
            # same attribute names as the old ThresholdWidget
            self.use_connected_components_checkbox = QCheckBox("Connected Components")
            self.connected_components_spin = QSpinBox()
            self.connected_components_spin.setRange(1, 100000)
            self.connected_components_spin.setValue(100)
            self.connected_components_spin.setEnabled(False)
            self.use_connected_components_checkbox.toggled.connect(self.connected_components_spin.setEnabled)
            self.erosion_spin = QSpinBox()
            self.erosion_spin.setRange(0, 100)
            row = QHBoxLayout()
            row.addWidget(self.use_connected_components_checkbox)
            row.addWidget(QLabel("N:"))
            row.addWidget(self.connected_components_spin)
            row.addWidget(QLabel("Erosions"))
            row.addWidget(self.erosion_spin)
            row.addStretch()
            lay.addLayout(row)
            self.use_connected_components_checkbox.toggled.connect(lambda *_: self._refresh_status())
            self.erosion_spin.valueChanged.connect(lambda *_: self._refresh_status())

        self.roi_reminder = RoiReminder(viewer, "Preview inside ROI",
                                        shape=lambda: None if self._img is None else self._img.shape)
        lay.addWidget(self.roi_reminder)
        row = QHBoxLayout()
        self.create_btn = QPushButton("Create Preview Layer")
        self.create_btn.setToolTip(
            "Apply the settings above to the whole volume and add the result as a new layer. "
            "Each click makes a new layer named after its parameters, so you can compare settings.")
        self.add_btn = QPushButton("Add range to table")
        self.add_btn.setToolTip("Append the current range as a new row of the threshold table below")
        row.addWidget(self.create_btn)
        row.addWidget(self.add_btn)
        lay.addLayout(row)
        self.preview_btn = self.create_btn   # old name, used by the host widgets

        self.hist.range_changed.connect(lambda *_: self._update_slice())
        self.viewer.dims.events.current_step.connect(lambda e: self._on_dims())
        self.viewer.dims.events.ndisplay.connect(lambda e: self._on_dims())
        self.live_cb.toggled.connect(self._on_live_toggled)
        self.create_btn.clicked.connect(self._create_preview_layer)
        self.add_btn.clicked.connect(lambda: self.add_to_table_requested.emit(*self.get_thresholds()))

        self.hist.show_message("Select an image")
        self._set_enabled(False)

    # ------------------------------------------------------------ public API
    def set_image(self, layer):
        """Image layer to threshold (None or a non-Image clears the widget)."""
        if not isinstance(layer, Image) or layer.rgb or layer.multiscale:
            layer = None
        if layer is self._layer:
            return
        self._layer = layer
        self._img = None if layer is None else np.asarray(layer.data)
        self.roi_reminder.refresh()
        self._preview.set_reference(layer)
        self._last_slice = None
        self._job += 1
        self._set_enabled(False)
        self.status.setText("")
        if layer is None:
            self.hist.show_message("Select an image")
            return
        self.hist.show_message("Computing...")
        self._hist_timer.start(0)

    def _start_histogram(self):
        if self._img is None:
            return
        job, img = self._job, self._img
        roi = self._hist_roi = self._roi_model.display_roi(img.shape)

        def work():
            vals = img if roi is None else roi.sample_values(img)   # ROI: only voxels inside it
            counts, edges = B.histogram(vals)
            from skimage.filters import threshold_otsu
            sub = B.subsample(vals)
            return counts, edges, float(threshold_otsu(sub)) if sub.size and sub.min() < sub.max() else None

        def done(res):
            if job != self._job or roi is not self._hist_roi:
                return
            counts, edges, otsu = res
            lo, hi = self._pending or (otsu, None)   # start at Otsu, no upper limit
            self._pending = None
            self.hist.set_data(counts, edges, integer=img.dtype.kind in "ui", lower=lo, upper=hi)
            self._auto_range = self.hist.get_range() if (lo, hi) == (otsu, None) else None
            if self._roi_refresh:
                self._roi_refresh = False
                self._on_dims()        # redraw the live slice if it is shown
            self._set_enabled(True)
            self.status.setText("Drag the range to preview the current slice"
                                if self.live_cb.isChecked() else "")

        self._run(work, done)

    def _on_roi_changed(self):
        """New ROI: recompute the histogram inside it; redraw the live slice."""
        if self._img is None:
            return
        if self._roi_model.display_roi(self._img.shape) is self._hist_roi:
            return                     # same ROI for this image (e.g. off -> still off)
        if self.hist._ready() and self._pending is None and self.hist.get_range() != self._auto_range:
            self._pending = self.hist.get_bounds()   # keep the user's range ("no limit" stays)
        self._roi_refresh = True
        self._hist_timer.start(0)
        self._on_dims()                # redraw the live slice if it is shown

    def set_range_by_dtype(self, dtype):
        """Compatibility no-op: the range now comes from the data (see set_image)."""

    def get_thresholds(self):
        lo, hi = self.hist.get_bounds()
        dtype = self._img.dtype if self._img is not None else np.dtype(np.uint16)
        if dtype.kind in "ui":
            info = np.iinfo(dtype)
            lo = info.min if lo is None else lo
            hi = info.max if hi is None else hi
        else:
            lo = -np.inf if lo is None else lo
            hi = np.inf if hi is None else hi
        return float(lo), float(hi)

    def set_thresholds(self, lower, upper):
        if self.hist._ready():
            self.hist.set_range(lower, upper)
        else:
            self._pending = (lower, upper)   # applied when the histogram arrives

    # -------------------------------------------------------------- internals
    def _run(self, fn, on_return, on_finish=None):
        w = thread_worker(fn, ignore_errors=True)()
        w.returned.connect(on_return)
        if on_finish:
            w.finished.connect(on_finish)
        w.errored.connect(lambda e: show_error(f"Threshold preview: {type(e).__name__}: {e}"))
        self._workers.add(w)
        w.finished.connect(lambda: self._workers.discard(w))
        w.start()

    def _set_enabled(self, on):
        self.create_btn.setEnabled(on)
        self.add_btn.setEnabled(on)

    def _uses_3d_ops(self):
        return self.add_connected_components and (
            self.use_connected_components_checkbox.isChecked() or self.erosion_spin.value() > 0)

    # ---- live slice preview
    def _on_live_toggled(self, on):
        if on:
            self._update_slice()
        else:
            self._preview.remove()
            self._last_slice = None
            self.status.setText("Live preview off")

    def _on_dims(self):
        if self.live_cb.isChecked() and self._preview.active():
            self._update_slice()

    def _update_slice(self):
        if not self.live_cb.isChecked() or self._img is None or not self.hist._ready():
            return
        lo, hi = self.hist.get_bounds()
        img = self._img
        roi = self._roi_model.display_roi(img.shape)
        inside = []

        def mask(sl):
            a = img[sl]
            m = np.ones(a.shape, bool) if lo is None else a >= lo
            if hi is not None:
                m &= a <= hi
            if roi is not None:
                inside.append(roi.mask_window(sl))
                m &= inside[0]
            return m

        m = self._preview.show_slice(mask)
        if m is None:
            self._last_slice = None
            self.status.setText("Live preview: switch to a 2D view")
            return
        self._last_slice = (int(m.sum()), m.size)
        self._slice_roi = roi is not None
        if roi is not None:   # count against the part of the slice inside the ROI
            self._last_slice = (int(m.sum()), int(inside[0].sum()))
        self._refresh_status()

    def _refresh_status(self):
        if not self.live_cb.isChecked() or self._last_slice is None:
            return
        n, size = self._last_slice
        if self._slice_roi:
            text = f"Current slice, inside the ROI: {n:,} voxels ({n / max(size, 1):.2%})"
        else:
            text = f"Current slice: {n:,} voxels ({n / size:.2%})"
        if self._uses_3d_ops():
            text += (". Live preview shows the threshold only; "
                     "Create Preview Layer to see erosion / connected components.")
        self.status.setText(text)

    # ---- whole-volume result as a new layer
    def _create_preview_layer(self):
        if self._img is None:
            show_error("Please select an image first")
            return
        try:
            roi = self._roi_model.roi_for(self._img.shape)
        except RoiError as e:
            show_error(f"ROI: {e}")
            return
        lo_b, hi_b = self.hist.get_bounds()
        lo, hi = self.get_thresholds()
        ero = self.erosion_spin.value() if self.add_connected_components else 0
        cc_n = (self.connected_components_spin.value()
                if self.add_connected_components and self.use_connected_components_checkbox.isChecked()
                else None)
        layer, img, job = self._layer, self._img, self._job
        self.create_btn.setEnabled(False)
        self.create_btn.setText("Creating...")

        def work():
            if roi is not None:
                return threshold_erode_cc_in_roi(img, roi, lo_b, hi_b, ero, cc_n)
            return threshold_erode_cc(img, lo_b, hi_b, ero, cc_n)

        def done(result):
            if job != self._job:
                return
            base = f"{layer.name}_preview_{lo:g}-{hi:g}"
            base += f"_ero{ero}" if ero else ""
            base += f"_cc{cc_n}" if cc_n else ""
            if roi is not None:
                base = R.roi_name(base)
            name, i = base, 1
            while name in self.viewer.layers:
                name, i = f"{base}_{i}", i + 1
            meta = {"image": layer.name, "lower": lo, "upper": hi,
                    "erosion_steps": ero, "connected_components": cc_n}
            if roi is not None:
                meta["roi"] = roi.to_metadata()
            self.viewer.add_labels(result, name=name, scale=layer.scale, translate=layer.translate,
                                   metadata={"sprout_threshold_preview": meta})
            n = int(np.count_nonzero(result))
            extra = f", {int(result.max())} components" if cc_n else ""
            msg = f"Created '{name}': {n:,} voxels ({n / result.size:.2%}){extra}"
            show_info(msg)
            self.status.setText(msg)

        def finished():
            self.create_btn.setText("Create Preview Layer")
            self.create_btn.setEnabled(self._img is not None and self.hist._ready())

        self._run(work, done, on_finish=finished)
