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
from sprout_core.sprout_core import erosion_binary_img_on_sub, get_ccomps_with_size_order

from .histogram_widget import HistogramRangeWidget
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


def threshold_erode_cc(img, lower=None, upper=None, erosion_steps=0, cc_n=None, cancel=None):
    """Same steps as the old 3D preview (util_widget.PreviewWorker):
    threshold -> erosion x n -> keep the cc_n largest connected components.

    lower/upper None = no limit. Returns uint8 (0/1) without CC, else a label image.
    """
    binary = B.threshold_mask(img, lower, upper, cancel=cancel).astype(bool)
    for _ in range(erosion_steps):
        binary = erosion_binary_img_on_sub(binary)
    if not cc_n:
        return binary.astype(np.uint8)
    labels, _ = get_ccomps_with_size_order(binary, cc_n)
    return labels.astype(np.uint16 if cc_n < 65536 else np.uint32)


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

        def work():
            counts, edges = B.histogram(img)
            from skimage.filters import threshold_otsu
            sub = B.subsample(img)
            return counts, edges, float(threshold_otsu(sub)) if sub.min() < sub.max() else None

        def done(res):
            if job != self._job:
                return
            counts, edges, otsu = res
            lo, hi = self._pending or (otsu, None)   # start at Otsu, no upper limit
            self._pending = None
            self.hist.set_data(counts, edges, integer=img.dtype.kind in "ui", lower=lo, upper=hi)
            self._set_enabled(True)
            self.status.setText("Drag the range to preview the current slice"
                                if self.live_cb.isChecked() else "")

        self._run(work, done)

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

        def mask(sl):
            a = img[sl]
            m = np.ones(a.shape, bool) if lo is None else a >= lo
            if hi is not None:
                m &= a <= hi
            return m

        m = self._preview.show_slice(mask)
        if m is None:
            self._last_slice = None
            self.status.setText("Live preview: switch to a 2D view")
            return
        self._last_slice = (int(m.sum()), m.size)
        self._refresh_status()

    def _refresh_status(self):
        if not self.live_cb.isChecked() or self._last_slice is None:
            return
        n, size = self._last_slice
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
            return threshold_erode_cc(img, lo_b, hi_b, ero, cc_n)

        def done(result):
            if job != self._job:
                return
            base = f"{layer.name}_preview_{lo:g}-{hi:g}"
            base += f"_ero{ero}" if ero else ""
            base += f"_cc{cc_n}" if cc_n else ""
            name, i = base, 1
            while name in self.viewer.layers:
                name, i = f"{base}_{i}", i + 1
            meta = {"image": layer.name, "lower": lo, "upper": hi,
                    "erosion_steps": ero, "connected_components": cc_n}
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
