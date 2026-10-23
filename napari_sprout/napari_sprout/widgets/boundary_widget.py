"""Boundary tab for napari-sprout.

Builds a boundary mask (e.g. skull sutures) that Seeds / Grow can use to
enforce separation. Workflow, top to bottom:

    1. Input      pick an image; optionally restrict by ORIGINAL intensity
    2. Method     Top-hat parameters -> Compute (background, with progress)
    3. Threshold  drag a range on the response histogram; the current slice
                  previews live
    4. Output     Create Boundary Layer (whole volume, background)

All heavy work runs off the UI thread. The only work on the UI thread is the
single-slice preview, which takes milliseconds.
"""
import threading
import time

import numpy as np
from napari.layers import Image, Labels
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_error, show_info, show_warning
from qtpy.QtCore import QObject, QTimer, Signal
from qtpy.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QProgressBar, QPushButton, QRadioButton, QScrollArea, QSpinBox,
    QStackedWidget, QVBoxLayout, QWidget,
)

from sprout_core import boundary as B

from ..utils.histogram_widget import HistogramRangeWidget
from ..utils.slice_preview import SlicePreview

PREVIEW_NAME = "Boundary preview"
META_KEY = "sprout_boundary"


class _Progress(QObject):
    """Created on the UI thread; emitting from a worker thread is queued to the UI."""
    changed = Signal(int, int)


# --------------------------------------------------------------------------
# Method panels. A method turns (image, params) into a response image of the
# same shape; sections 1, 3 and 4 are shared by every method.
# --------------------------------------------------------------------------
class TophatPanel(QWidget):
    """Parameters of the grey top-hat."""

    name = "Top-hat"
    params_changed = Signal()

    SHAPES = [
        ("Ball (exact sphere)", "ball",
         "Exact Euclidean ball of radius = size."),
        ("Cube", "cube",
         "Cube of side 2*size+1. Same as Avizo: Cube, neighbourhood 26, size."),
        ("Octahedron", "octahedron",
         "|dz|+|dy|+|dx| <= size. Same as Avizo: Cube, neighbourhood 6, size."),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)

        self.black = QRadioButton("black")
        self.white = QRadioButton("white")
        self.black.setChecked(True)
        self.black.setToolTip("closing - image: small DARK structures (sutures, gaps)")
        self.white.setToolTip("image - opening: small BRIGHT structures")
        grp = QButtonGroup(self)
        grp.addButton(self.black)
        grp.addButton(self.white)
        row = QHBoxLayout()
        row.addWidget(self.black)
        row.addWidget(self.white)
        row.addStretch()
        form.addRow("Type:", row)

        self.shape_combo = QComboBox()
        for i, (label, key, tip) in enumerate(self.SHAPES):
            self.shape_combo.addItem(label, key)
            self.shape_combo.setItemData(i, tip, 3)  # 3 == Qt.ToolTipRole
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 50)
        self.size_spin.setValue(3)
        self.size_spin.setToolTip("Footprint radius in voxels. Structures narrower than "
                                  "the footprint are extracted.")
        row = QHBoxLayout()
        row.addWidget(self.shape_combo)
        row.addWidget(QLabel("Size:"))
        row.addWidget(self.size_spin)
        form.addRow("Shape:", row)

        self.black.toggled.connect(self.params_changed)
        self.shape_combo.currentIndexChanged.connect(self.params_changed)
        self.size_spin.valueChanged.connect(self.params_changed)

    def params(self):
        return {"kind": "black" if self.black.isChecked() else "white",
                "shape": self.shape_combo.currentData(),
                "size": self.size_spin.value()}

    def set_params(self, p):
        if "kind" in p:
            (self.black if p["kind"] == "black" else self.white).setChecked(True)
        if "shape" in p:
            i = self.shape_combo.findData(p["shape"])
            if i >= 0:
                self.shape_combo.setCurrentIndex(i)
        if "size" in p:
            self.size_spin.setValue(int(p["size"]))

    @staticmethod
    def describe(p):
        return f"{p['kind']} top-hat, {p['shape']}, size {p['size']}"

    @staticmethod
    def compute(img, p, device, progress, cancel):
        return B.tophat(img, p["kind"], p["shape"], p["size"], device, progress, cancel)


# --------------------------------------------------------------------------
class BoundaryWidget(QWidget):
    """Boundary tab: image -> response -> threshold -> boundary Labels layer."""

    boundary_created = Signal(str)   # layer name; connect to Seeds / Grow

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._img_layer = None
        self._img = None
        self._resp = None            # cached response for the current image + params
        self._resp_key = None
        self._resp_desc = ""
        self._job = {"hist": 0, "compute": 0, "resp_hist": 0, "create": 0}
        self._computing = False
        self._cancel = None
        self._workers = set()
        self._slice_preview = SlicePreview(napari_viewer, PREVIEW_NAME,
                                           color=(1, 0.15, 0.15, 1), opacity=0.7)
        self._pending_threshold = None

        # Layer changes are handled on the next event-loop turn and coalesced.
        # This also means nothing is started while napari is closing: the timer
        # dies with the widget.
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self.refresh_layers)

        self._build_ui()
        self._connect()
        self.refresh_layers()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        layout = QVBoxLayout()

        # 1. Input
        g1 = QGroupBox("1. Input")
        l1 = QVBoxLayout(g1)
        form = QFormLayout()
        self.image_combo = QComboBox()
        form.addRow("Image:", self.image_combo)
        l1.addLayout(form)
        self.input_hist = HistogramRangeWidget(
            "Original intensity: only voxels in this range can become boundary")
        self.input_hist.setToolTip("A zero background is excluded by default.")
        l1.addWidget(self.input_hist)
        layout.addWidget(g1)

        # 2. Method
        g2 = QGroupBox("2. Method")
        l2 = QVBoxLayout(g2)
        form = QFormLayout()
        self.method_combo = QComboBox()
        self.method_stack = QStackedWidget()
        self.panels = [TophatPanel()]
        for p in self.panels:
            self.method_combo.addItem(p.name)
            self.method_stack.addWidget(p)
        form.addRow("Method:", self.method_combo)
        l2.addLayout(form)
        l2.addWidget(self.method_stack)

        form = QFormLayout()
        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto", "auto")
        self.device_combo.addItem("GPU", "gpu")
        self.device_combo.addItem("CPU", "cpu")
        # GPU detection imports torch (~3 s), so it is done in the background the
        # first time this tab is shown, not when the SPROUT panel opens.
        self.device_combo.model().item(1).setEnabled(False)
        self.device_label = QLabel("checking for a GPU...")
        self.device_label.setStyleSheet("color: gray")
        self._gpu_checked = False
        row = QHBoxLayout()
        row.addWidget(self.device_combo)
        row.addWidget(self.device_label)
        row.addStretch()
        form.addRow("Device:", row)
        self.response_layer_cb = QCheckBox("Also add the response as an image layer")
        self.response_layer_cb.setToolTip("Useful to look at what the method highlights. "
                                          "Costs one more image worth of memory.")
        form.addRow(self.response_layer_cb)
        l2.addLayout(form)

        row = QHBoxLayout()
        self.compute_btn = QPushButton("Compute")
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        row.addWidget(self.compute_btn)
        row.addWidget(self.progress, 1)
        row.addWidget(self.stop_btn)
        l2.addLayout(row)
        self.compute_status = QLabel("")
        self.compute_status.setStyleSheet("color: gray")
        self.compute_status.setWordWrap(True)
        l2.addWidget(self.compute_status)
        layout.addWidget(g2)

        # 3. Threshold
        g3 = QGroupBox("3. Threshold")
        l3 = QVBoxLayout(g3)
        self.resp_hist = HistogramRangeWidget(
            "Response (eligible voxels only): drag the range; the current slice previews live")
        l3.addWidget(self.resp_hist)
        self.preview_label = QLabel("")
        self.preview_label.setStyleSheet("color: gray")
        l3.addWidget(self.preview_label)
        layout.addWidget(g3)

        # 4. Output
        g4 = QGroupBox("4. Output")
        form = QFormLayout(g4)
        self.name_edit = QLineEdit()
        form.addRow("Name:", self.name_edit)
        self.use_cb = QCheckBox("Use as boundary in Seeds / Grow")
        self.use_cb.setChecked(True)
        form.addRow(self.use_cb)
        row = QHBoxLayout()
        self.create_btn = QPushButton("Create Boundary Layer")
        self.import_btn = QPushButton("Import YAML")
        self.export_btn = QPushButton("Export YAML")
        row.addWidget(self.create_btn, 1)
        row.addWidget(self.import_btn)
        row.addWidget(self.export_btn)
        form.addRow(row)
        layout.addWidget(g4)
        layout.addStretch()

        container = QWidget()
        container.setLayout(layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self.input_hist.show_message("Select an image")
        self.resp_hist.show_message("Click Compute")
        self._update_buttons()

    def _connect(self):
        self.viewer.layers.events.inserted.connect(lambda e: self._refresh_timer.start(0))
        self.viewer.layers.events.removed.connect(lambda e: self._refresh_timer.start(0))
        self.viewer.dims.events.current_step.connect(lambda e: self._update_preview())
        self.viewer.dims.events.ndisplay.connect(lambda e: self._update_preview())

        self.image_combo.currentTextChanged.connect(self._on_image_changed)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        for p in self.panels:
            p.params_changed.connect(self._on_params_changed)
        self.device_combo.currentIndexChanged.connect(self._on_params_changed)

        self.input_hist.range_changed.connect(lambda *_: self._update_preview())
        self.input_hist.range_committed.connect(lambda *_: self._on_intensity_range_committed())
        self.resp_hist.range_changed.connect(lambda *_: self._update_preview())

        self.compute_btn.clicked.connect(self._compute)
        self.stop_btn.clicked.connect(self._stop)
        self.create_btn.clicked.connect(self._create)
        self.import_btn.clicked.connect(self._import_yaml)
        self.export_btn.clicked.connect(self._export_yaml)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._gpu_checked:
            self._gpu_checked = True
            self._run(B.gpu_status, self._on_gpu_status)

    def _on_gpu_status(self, res):
        ok, desc = res
        self.device_combo.model().item(1).setEnabled(ok)
        self.device_label.setText(f"GPU: {desc}" if ok else f"CPU only ({desc})")

    # ------------------------------------------------------------- helpers
    @property
    def panel(self):
        return self.method_stack.currentWidget()

    def _run(self, fn, on_return, on_error=None, on_finish=None):
        """Run fn() on a worker thread; callbacks run on the UI thread.

        Errors are handled here (ignore_errors=True stops napari from also
        dumping a traceback, e.g. every time the user presses Stop).
        """
        w = thread_worker(fn, ignore_errors=True)()
        w.returned.connect(on_return)
        w.errored.connect(on_error or self._report_error)
        if on_finish:
            w.finished.connect(on_finish)
        self._workers.add(w)
        w.finished.connect(lambda: self._workers.discard(w))
        w.start()
        return w

    @staticmethod
    def _report_error(e):
        import traceback
        traceback.print_exception(type(e), e, e.__traceback__)   # full detail in the console
        show_error(f"Boundary: {type(e).__name__}: {e}")

    def _intensity_range(self):
        """(lower, upper) on the ORIGINAL image, or (None, None) if unrestricted."""
        if not self.input_hist._ready():
            return None, None
        return self.input_hist.get_bounds()

    def _current_key(self):
        name = self._img_layer.name if self._img_layer is not None else None
        return (id(self._img_layer), name, self.method_combo.currentIndex(),
                tuple(sorted(self.panel.params().items())), self.device_combo.currentData())

    def _layer_kwargs(self):
        lay = self._img_layer
        return {"scale": lay.scale, "translate": lay.translate}

    def _update_buttons(self):
        has_img = self._img is not None
        has_resp = self._resp is not None
        self.compute_btn.setEnabled(has_img and not self._computing)
        self.stop_btn.setEnabled(self._computing)
        self.create_btn.setEnabled(has_resp and not self._computing)
        self.export_btn.setEnabled(has_img)

    # --------------------------------------------------------------- input
    def refresh_layers(self):
        current = self.image_combo.currentText()
        names = [l.name for l in self.viewer.layers if isinstance(l, Image)]
        self.image_combo.blockSignals(True)
        self.image_combo.clear()
        self.image_combo.addItems(names)
        if current in names:
            self.image_combo.setCurrentText(current)
        self.image_combo.blockSignals(False)
        self._on_image_changed(self.image_combo.currentText())

    def _on_image_changed(self, name):
        layer = self.viewer.layers[name] if name and name in self.viewer.layers else None
        if layer is self._img_layer:
            return
        self._stop()
        self._reset_response("Click Compute")
        self._img_layer, self._img = None, None
        self._job["hist"] += 1

        if layer is None:
            self.input_hist.show_message("Select an image")
            self._update_buttons()
            return
        if layer.multiscale or layer.rgb or layer.ndim not in (2, 3):
            self.input_hist.show_message("Unsupported image (needs 2D or 3D, single channel)")
            self._update_buttons()
            return

        self._img_layer = layer
        self._img = np.asarray(layer.data)
        self.name_edit.setText(f"{name}_boundary")
        self.input_hist.show_message("Computing...")
        job, img = self._job["hist"], self._img

        def work():
            counts, edges = B.histogram(img)
            lo, hi, _, _ = B.default_intensity_range(img)
            return counts, edges, lo, hi

        def done(res):
            if job != self._job["hist"]:
                return  # the user picked another image meanwhile
            counts, edges, lo, hi = res
            self.input_hist.set_data(counts, edges, integer=img.dtype.kind in "ui", lower=lo, upper=hi)

        self._run(work, done)
        self._update_buttons()

    def _on_intensity_range_committed(self):
        if self._resp is None:
            return
        self._compute_response_histogram(keep_range=True)

    # -------------------------------------------------------------- method
    def _on_method_changed(self, i):
        self.method_stack.setCurrentIndex(i)
        self._on_params_changed()

    def _on_params_changed(self, *_):
        if self._computing:
            return  # the running result is discarded on return (stale key)
        if self._resp is not None:
            self._reset_response("Parameters changed: click Compute")

    def _reset_response(self, message):
        self._resp = None
        self._resp_key = None
        self._slice_preview.remove()
        self.resp_hist.show_message(message)
        self.preview_label.setText("")
        self._update_buttons()

    def _compute(self):
        if self._img is None:
            show_error("Select an image first")
            return
        self._stop()
        self._job["compute"] += 1
        job, img, key = self._job["compute"], self._img, self._current_key()
        params, device, panel = self.panel.params(), self.device_combo.currentData(), self.panel
        ilo, ihi = self._intensity_range()
        cancel = self._cancel = threading.Event()

        # One progress object per job, kept alive by the worker closure below, and
        # ignored once a newer job has started.
        prog = _Progress()
        prog.changed.connect(lambda d, t: job == self._job["compute"] and self._on_progress(d, t))
        self._computing = True
        self._reset_response("Computing...")
        self.progress.setRange(0, 0)  # busy until the first chunk reports
        self.compute_status.setText("")
        self._update_buttons()
        def emit(done_, total):
            prog.changed.emit(done_, total)

        def work():
            t0 = time.perf_counter()
            resp, info = panel.compute(img, params, device, emit, cancel)
            info["seconds"] = time.perf_counter() - t0
            if cancel.is_set():
                raise B.Cancelled()
            counts, edges = B.histogram(resp, img=img, lower=ilo, upper=ihi)
            suggest = B.suggest_threshold(resp, img, ilo, ihi)
            return resp, info, counts, edges, suggest

        def done(res):
            if job != self._job["compute"]:
                return
            if cancel.is_set():  # Stop pressed: never show a result the user cancelled
                failed(B.Cancelled())
                return
            if key != self._current_key():
                self._reset_response("Parameters changed during computing: click Compute")
                return
            resp, info, counts, edges, suggest = res
            self._resp, self._resp_key = resp, key
            self._resp_desc = panel.describe(params)
            self._resp_info = info
            dev = "GPU" if info["device"] == "cuda" else "CPU"
            note = "" if dev == "GPU" or device == "cpu" else f" ({info['device_reason']})"
            self.compute_status.setText(f"{self._resp_desc}: {info['seconds']:.1f} s on {dev}{note}")
            lo, hi = self._pending_threshold or (suggest, None)   # None -> axis end = no limit
            self._pending_threshold = None
            self.resp_hist.set_data(counts, edges, integer=resp.dtype.kind in "ui", lower=lo, upper=hi)
            if self.response_layer_cb.isChecked():
                self._show_response_layer()
            self._update_preview()

        def failed(e):
            if isinstance(e, B.Cancelled):
                self.resp_hist.show_message("Stopped: click Compute")
                self.compute_status.setText("Stopped")
            else:
                self.resp_hist.show_message("Failed: see notification")
                self._report_error(e)

        def finished():
            if job == self._job["compute"]:
                self._computing = False
                self.progress.setRange(0, 1)
                self.progress.setValue(1 if self._resp is not None else 0)
                self._update_buttons()

        self._run(work, done, failed, finished)

    def _on_progress(self, done, total):
        self.progress.setRange(0, total)
        self.progress.setValue(done)

    def _stop(self):
        if self._cancel is not None:
            self._cancel.set()

    def _compute_response_histogram(self, keep_range):
        self._job["resp_hist"] += 1
        job, resp, img = self._job["resp_hist"], self._resp, self._img
        ilo, ihi = self._intensity_range()
        old = self.resp_hist.get_bounds()   # keep "no limit" as no limit
        self.resp_hist.show_message("Computing...")

        def work():
            return B.histogram(resp, img=img, lower=ilo, upper=ihi)

        def done(res):
            if job != self._job["resp_hist"] or resp is not self._resp:
                return
            counts, edges = res
            lo, hi = old if keep_range else (None, None)
            self.resp_hist.set_data(counts, edges, integer=resp.dtype.kind in "ui", lower=lo, upper=hi)
            self._update_preview()

        self._run(work, done)

    def _show_response_layer(self):
        name = f"{self._img_layer.name}_response"
        md = {META_KEY + "_response": self._resp_desc}
        if name in self.viewer.layers and META_KEY + "_response" in self.viewer.layers[name].metadata:
            lay = self.viewer.layers[name]
            lay.data = self._resp
            lay.metadata = md
        else:
            self.viewer.add_image(self._resp, name=name, colormap="magma", metadata=md,
                                  **self._layer_kwargs())

    # ------------------------------------------------------------- preview
    @property
    def _preview_buf(self):
        return self._slice_preview._buf

    def _update_preview(self):
        """Threshold only the displayed slice: milliseconds, so it runs on the UI thread."""
        if self._resp is None or not self.resp_hist._ready() or self._img is None:
            return
        self._slice_preview.set_reference(self._img_layer)
        lo, hi = self.resp_hist.get_bounds()
        ilo, ihi = self._intensity_range()
        resp, img = self._resp, self._img

        def mask(sl):
            r = resp[sl]
            m = np.ones(r.shape, bool) if lo is None else r >= lo
            if hi is not None:
                m &= r <= hi
            if ilo is not None or ihi is not None:
                m &= B.intensity_mask(img[sl], ilo, ihi)
            return m

        m = self._slice_preview.show_slice(mask)
        if m is None:
            self.preview_label.setText("Preview: switch to a 2D view")
            return
        n = int(m.sum())
        self.preview_label.setText(f"Current slice: {n:,} boundary voxels ({n / m.size:.2%})")

    # -------------------------------------------------------------- output
    def _params_record(self):
        lo, hi = self.resp_hist.get_bounds()   # None = no limit
        ilo, ihi = self._intensity_range()
        rec = {"method": self.method_combo.currentText(), **self.panel.params(),
               "device": self.device_combo.currentData(),
               "intensity_lower": ilo, "intensity_upper": ihi,
               "threshold_lower": lo, "threshold_upper": hi}
        return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in rec.items()}

    def _create(self):
        if self._resp is None:
            show_error("Compute first")
            return
        self._job["create"] += 1
        job, resp, img = self._job["create"], self._resp, self._img
        rec = self._params_record()
        rec["image"] = self._img_layer.name
        lo, hi = rec["threshold_lower"], rec["threshold_upper"]
        ilo, ihi = rec["intensity_lower"], rec["intensity_upper"]
        self.create_btn.setEnabled(False)
        self.create_btn.setText("Creating...")

        def work():
            mask = B.threshold_mask(resp, lo, hi, img, ilo, ihi)
            return mask, int(np.count_nonzero(mask))

        def done(res):
            if job != self._job["create"]:
                return
            mask, n = res
            if n == 0:
                show_error("No voxel is in the threshold range: lower the threshold")
                return
            if n == mask.size:
                show_error("Every voxel is in the threshold range: raise the threshold")
                return
            name = self._add_or_update_output(mask, rec)
            self._slice_preview.remove()
            show_info(f"Created '{name}': {n:,} voxels ({n / mask.size:.2%})")
            if self.use_cb.isChecked():
                self.boundary_created.emit(name)

        def finished():
            self.create_btn.setText("Create Boundary Layer")
            self._update_buttons()

        self._run(work, done, on_finish=finished)

    def _add_or_update_output(self, mask, rec):
        name = self.name_edit.text().strip() or f"{self._img_layer.name}_boundary"
        if name in self.viewer.layers:
            lay = self.viewer.layers[name]
            if isinstance(lay, Labels) and META_KEY in lay.metadata:
                lay.data = mask            # our own earlier result: overwrite
                lay.metadata = {META_KEY: rec}
                return name
            base, i = name, 1              # someone else's layer: never overwrite
            while name in self.viewer.layers:
                name = f"{base}_{i}"
                i += 1
            show_warning(f"A layer named '{base}' already exists; created '{name}'")
        self.viewer.add_labels(mask, name=name, metadata={META_KEY: rec}, **self._layer_kwargs())
        return name

    # ---------------------------------------------------------------- YAML
    def _export_yaml(self):
        import yaml
        path, _ = QFileDialog.getSaveFileName(self, "Export boundary parameters", "boundary.yaml",
                                              "YAML (*.yaml *.yml)")
        if not path:
            return
        rec = self._params_record()
        rec["image"] = self._img_layer.name if self._img_layer is not None else None
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(rec, f, sort_keys=False)
        show_info(f"Saved {path}")

    def _import_yaml(self):
        import yaml
        path, _ = QFileDialog.getOpenFileName(self, "Import boundary parameters", "", "YAML (*.yaml *.yml)")
        if not path:
            return
        with open(path, encoding="utf-8") as f:
            rec = yaml.safe_load(f) or {}
        i = self.method_combo.findText(rec.get("method", ""))
        if i >= 0:
            self.method_combo.setCurrentIndex(i)
        self.panel.set_params(rec)
        j = self.device_combo.findData(rec.get("device"))
        if j >= 0 and self.device_combo.model().item(j).isEnabled():
            self.device_combo.setCurrentIndex(j)
        if self.input_hist._ready():
            h = self.input_hist
            ilo, ihi = rec.get("intensity_lower"), rec.get("intensity_upper")
            h.set_range(h._xmin if ilo is None else ilo, h._xmax if ihi is None else ihi)
        if "threshold_lower" in rec or "threshold_upper" in rec:
            self._pending_threshold = (rec.get("threshold_lower"), rec.get("threshold_upper"))
        show_info("Parameters loaded: click Compute")
