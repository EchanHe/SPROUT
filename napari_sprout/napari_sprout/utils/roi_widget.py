"""ROI editor for the top of the SPROUT panel, its box overlay, and reminder lines.

    RoiWidget       "Region of interest (ROI)" group, checkable, off by default.
                    Box: draw a rectangle on the 2D view (the two displayed axes),
                    set the third axis with "= current" or the From / To numbers.
                    Mask: a Labels layer, all non-zero voxels or one label.
    RoiBoxOverlay   the "SPROUT ROI" Shapes layer showing the box on the current
                    slice: yellow inside the box's range on the third axis, grey outside
    RoiReminder     a line above a tab's run button: hidden when the ROI is off
    RoiSummaryLine  read-only summary + "use ROI" checkbox for panels outside the
                    main panel (not embedded anywhere yet)

All state lives in the viewer's RoiModel (roi_model.py); these widgets only edit
and display it. napari 0.6.1 limits: a Shapes rectangle lives on one slice, and
nothing can be drawn in the 3D view.
"""
import math

import numpy as np
from napari.layers import Image, Labels
from napari.utils.notifications import show_info
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QPushButton, QRadioButton, QSpinBox, QStackedWidget, QVBoxLayout, QWidget,
)

from .roi_model import get_roi_model

OVERLAY_NAME = "SPROUT ROI"
YELLOW = "yellow"
GREY = "#99999999"           # (0.6, 0.6, 0.6, 0.6)
OK_STYLE = "color: gray"
ERROR_STYLE = "color: #d9534f"
ROI_STYLE = "color: #e08a00"
# SPROUT's own live-preview Labels layers are never offered as an ROI mask
_PREVIEW_LAYERS = ("Threshold Preview", "Boundary preview")


def _displayed_axes(viewer, ref):
    """Data axes of `ref` that the viewer displays."""
    offset = viewer.dims.ndim - ref.ndim
    return [a - offset for a in viewer.dims.displayed if a >= offset]


def _current_index(viewer, ref, ax):
    return int(round(ref.world_to_data(viewer.dims.point)[ax]))


# --------------------------------------------------------------------------
class RoiBoxOverlay:
    """The box as a Shapes rectangle on the current 2D slice.

    The layer gets the image's scale / translate / rotate / shear / affine, so its
    data coordinates are the image's voxel indices (anisotropic images included).
    Every programmatic change of the layer runs under `_syncing`, so that only the
    user's own drawing / editing / deleting reaches the model.
    """

    def __init__(self, viewer, model):
        self.viewer = viewer
        self.model = model
        self.layer = None
        self.ref = None
        self._syncing = False
        self._prev_active = None
        self._deleted_by_user = False

    # ---- layer
    def _alive(self):
        return self.layer is not None and self.layer in self.viewer.layers

    def _matches(self, ref):
        lay = self.layer
        return (lay.ndim == ref.ndim and np.allclose(lay.scale, ref.scale)
                and np.allclose(lay.translate, ref.translate)
                and np.allclose(lay.affine.affine_matrix, ref.affine.affine_matrix))

    def ensure_layer(self, ref):
        if self._alive() and self._matches(ref):
            return self.layer
        if self._alive():
            self._syncing = True
            try:
                self.viewer.layers.remove(self.layer)
            finally:
                self._syncing = False
        shape = np.asarray(ref.data.shape)
        width = max(1.0, 0.004 * float(shape[_displayed_axes(self.viewer, ref)].max()))
        active = self.viewer.layers.selection.active
        self._syncing = True
        try:
            self.layer = self.viewer.add_shapes(
                ndim=ref.ndim, name=OVERLAY_NAME, shape_type="rectangle", edge_color=YELLOW,
                face_color="transparent", edge_width=width, scale=ref.scale, translate=ref.translate,
                rotate=ref.rotate, shear=ref.shear, affine=ref.affine.affine_matrix)
        finally:
            self._syncing = False
        self.layer.events.data.connect(self._on_data)
        if active is not None and active in self.viewer.layers:   # don't steal the selection
            self.viewer.layers.selection.active = active
        self._deleted_by_user = False
        return self.layer

    def on_layer_removed(self, layer):
        if layer is self.layer:
            self.layer = None
            self._deleted_by_user = not self._syncing

    # ---- drawing
    def start_draw(self, ref):
        """Let the user drag a rectangle on the displayed slice."""
        if ref is None:
            show_info("Select an image for the ROI first")
            return False
        if self.viewer.dims.ndisplay != 2:
            show_info("Switch to the 2D view to draw")
            return False
        self.ref = ref
        prev = self.viewer.layers.selection.active
        lay = self.ensure_layer(ref)
        self._prev_active = prev if prev is not lay else None
        self._syncing = True
        try:
            lay.data = []
            lay.edge_color = YELLOW
            lay.visible = True
        finally:
            self._syncing = False
        self.viewer.layers.selection.active = lay
        lay.mode = "add_rectangle"
        return True

    def _on_data(self, event):
        if self._syncing:
            return
        action = str(getattr(event, "action", ""))
        if action not in ("added", "changed", "removed"):
            return
        # ADDED fires inside napari's mouse-release handler: change the layer later
        QTimer.singleShot(0, lambda: self._apply_from_shapes(action))

    def _apply_from_shapes(self, action):
        lay, ref = self.layer, self.ref if self.ref is not None else self.model.reference
        if not self._alive() or ref is None:
            return
        if len(lay.data) == 0:
            if action == "removed":
                self.model.clear_box()
            self._finish_drawing()
            return
        verts = np.asarray(lay.data[-1], float)
        shape = tuple(ref.data.shape)
        old = self.model.box if self.model.box_shape == shape else None
        lo = list(old[0]) if old else [0] * len(shape)
        hi = list(old[1]) if old else list(shape)
        for ax in _displayed_axes(self.viewer, ref):
            # a voxel is inside if its centre is (pixel-centre rule; idempotent on our corners)
            lo[ax] = min(max(int(math.ceil(verts[:, ax].min())), 0), shape[ax])
            hi[ax] = min(max(int(math.floor(verts[:, ax].max())) + 1, lo[ax]), shape[ax])
        self.model.set_reference(ref)
        self.model.set_box(shape, lo, hi)
        self._finish_drawing()
        self.sync()

    def _finish_drawing(self):
        lay = self.layer
        if self._alive() and str(lay.mode).startswith("add"):
            lay.mode = "pan_zoom"
            prev, self._prev_active = self._prev_active, None
            if prev is not None and prev in self.viewer.layers:
                self.viewer.layers.selection.active = prev

    # ---- display
    def sync(self):
        """Show the model's box on the current slice (or hide it)."""
        m = self.model
        ref = m.reference if m.reference is not None else self.ref
        show = (m.enabled and m.kind == "box" and m.box is not None and ref is not None
                and m.box_shape == tuple(ref.data.shape) and self.viewer.dims.ndisplay == 2)
        if not self._alive() or (show and not self._matches(ref)):
            if not show or self._deleted_by_user:
                return
            self.ensure_layer(ref)
        lay = self.layer
        if str(lay.mode).startswith("add"):
            return   # the user is drawing
        if not show:
            if lay.visible:
                self._syncing = True
                try:
                    lay.visible = False
                finally:
                    self._syncing = False
            return
        self.ref = ref
        lo, hi = m.box
        shown = _displayed_axes(self.viewer, ref)
        if len(shown) != 2:
            return
        a, b = shown
        p = np.zeros(ref.ndim)
        inside = True
        for ax in range(ref.ndim):
            if ax not in shown:
                p[ax] = _current_index(self.viewer, ref, ax)
                inside &= lo[ax] <= p[ax] < hi[ax]
        verts = np.repeat(p[None], 4, axis=0)
        verts[:, a] = [lo[a] - 0.5, lo[a] - 0.5, hi[a] - 0.5, hi[a] - 0.5]
        verts[:, b] = [lo[b] - 0.5, hi[b] - 0.5, hi[b] - 0.5, lo[b] - 0.5]
        self._syncing = True
        try:
            lay.data = [(verts, "rectangle")]
            lay.edge_color = YELLOW if inside or ref.ndim == 2 else GREY
            lay.visible = True
        finally:
            self._syncing = False


# --------------------------------------------------------------------------
class RoiWidget(QGroupBox):
    """'Region of interest (ROI)': one ROI for every SPROUT tab of this viewer."""

    def __init__(self, napari_viewer, parent=None):
        super().__init__("Region of interest (ROI)", parent)
        self.viewer = napari_viewer
        self.model = get_roi_model(napari_viewer)
        self.overlay = RoiBoxOverlay(napari_viewer, self.model)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self.refresh_layers)

        self.setCheckable(True)
        self.setChecked(False)
        self.setToolTip("Off: every tab works on the whole image. On: Seeds, Grow, Boundary and the "
                        "threshold previews only work inside the ROI; results stay image-sized.")
        outer = QVBoxLayout(self)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        outer.addWidget(self.summary)

        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.box_rb = QRadioButton("Box")
        self.mask_rb = QRadioButton("Mask")
        self.box_rb.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.box_rb)
        grp.addButton(self.mask_rb)
        row.addWidget(QLabel("Kind:"))
        row.addWidget(self.box_rb)
        row.addWidget(self.mask_rb)
        row.addStretch()
        body.addLayout(row)
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_box_page())
        self.stack.addWidget(self._build_mask_page())
        body.addWidget(self.stack)
        outer.addWidget(self.body)
        self.body.setVisible(False)

        self.toggled.connect(self._on_toggled)
        self.box_rb.toggled.connect(self._on_kind)
        self.model.changed.connect(self._on_model_changed)
        self.viewer.layers.events.inserted.connect(lambda e: self._refresh_timer.start(0))
        self.viewer.layers.events.removed.connect(self._on_layer_removed)
        self.viewer.dims.events.current_step.connect(lambda e: self._on_dims())
        self.viewer.dims.events.order.connect(lambda e: self._on_dims())
        self.viewer.dims.events.ndisplay.connect(lambda e: self._on_dims())
        self.refresh_layers()
        self._on_model_changed()

    # ------------------------------------------------------------- pages
    def _build_box_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self.image_combo = QComboBox()
        self.image_combo.setToolTip("The image whose voxel grid the box is drawn on")
        form.addRow("Image:", self.image_combo)
        lay.addLayout(form)
        row = QHBoxLayout()
        self.draw_btn = QPushButton("Draw box")
        self.draw_btn.setToolTip("Drag a rectangle on the 2D view: it sets the two displayed axes. "
                                 "The third axis keeps its range (the whole image the first time).")
        self.full_btn = QPushButton("Full extent")
        self.clear_btn = QPushButton("Clear")
        row.addWidget(self.draw_btn)
        row.addWidget(self.full_btn)
        row.addWidget(self.clear_btn)
        row.addStretch()
        lay.addLayout(row)
        self.hint = QLabel("drag a rectangle on the 2D view")
        self.hint.setStyleSheet(OK_STYLE)
        lay.addWidget(self.hint)

        grid = QGridLayout()
        grid.addWidget(QLabel("From"), 0, 1)
        grid.addWidget(QLabel("To"), 0, 3)
        self.axis_rows = []
        for i, name in enumerate(("z", "y", "x"), 1):
            lab = QLabel(name)
            lo, hi = QSpinBox(), QSpinBox()
            lo_cur, hi_cur = QPushButton("= current"), QPushButton("= current")
            for w in (lo_cur, hi_cur):
                w.setToolTip("Set to the slice shown in the viewer")
            for sp in (lo, hi):
                sp.setToolTip("Voxel index, inclusive")
                sp.editingFinished.connect(self._commit_spins)
            lo_cur.clicked.connect(lambda _=False, s=lo: self._set_current(s))
            hi_cur.clicked.connect(lambda _=False, s=hi: self._set_current(s))
            for c, w in enumerate((lab, lo, lo_cur, hi, hi_cur)):
                grid.addWidget(w, i, c)
            self.axis_rows.append((lab, lo, lo_cur, hi, hi_cur))
        lay.addLayout(grid)

        self.draw_btn.clicked.connect(self._draw)
        self.full_btn.clicked.connect(self._full_extent)
        self.clear_btn.clicked.connect(self.model.clear_box)
        self.image_combo.currentTextChanged.connect(self._on_image_changed)
        return page

    def _build_mask_page(self):
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(0, 0, 0, 0)
        self.mask_combo = QComboBox()
        form.addRow("Labels layer:", self.mask_combo)
        self.all_rb = QRadioButton("All non-zero")
        self.label_rb = QRadioButton("Label")
        self.all_rb.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.all_rb)
        grp.addButton(self.label_rb)
        self.label_spin = QSpinBox()
        self.label_spin.setRange(1, 2 ** 31 - 1)
        self.label_spin.setValue(1)
        row = QHBoxLayout()
        row.addWidget(self.all_rb)
        row.addWidget(self.label_rb)
        row.addWidget(self.label_spin)
        row.addStretch()
        form.addRow(row)
        self.mask_status = QLabel("")
        self.mask_status.setStyleSheet(OK_STYLE)
        form.addRow(self.mask_status)
        self.mask_combo.activated.connect(lambda *_: self._apply_mask())
        self.all_rb.toggled.connect(lambda *_: self._apply_mask())
        self.label_spin.valueChanged.connect(lambda *_: self.label_rb.isChecked() and self._apply_mask())
        return page

    # ------------------------------------------------------------- helpers
    def _ref(self):
        name = self.image_combo.currentText()
        lay = self.viewer.layers[name] if name and name in self.viewer.layers else None
        return lay if isinstance(lay, Image) else None

    def _mask_layer(self):
        name = self.mask_combo.currentText()
        lay = self.viewer.layers[name] if name and name in self.viewer.layers else None
        return lay if isinstance(lay, Labels) else None

    def refresh_layers(self):
        imgs = [l.name for l in self.viewer.layers if isinstance(l, Image) and not l.rgb and not l.multiscale]
        labs = [l.name for l in self.viewer.layers
                if isinstance(l, Labels) and not l.name.startswith(_PREVIEW_LAYERS)]
        cur = self.image_combo.currentText()
        ref = self.model.reference
        active = self.viewer.layers.selection.active
        pick = (ref.name if ref is not None and ref.name in imgs else
                cur if cur in imgs else
                active.name if isinstance(active, Image) and active.name in imgs else
                imgs[0] if imgs else "")
        self.image_combo.blockSignals(True)
        self.image_combo.clear()
        self.image_combo.addItems(imgs)
        self.image_combo.setCurrentText(pick)
        self.image_combo.blockSignals(False)
        self._on_image_changed(self.image_combo.currentText())

        mcur = self.model.mask_layer.name if self.model.mask_layer is not None else self.mask_combo.currentText()
        if mcur not in labs and isinstance(active, Labels):
            mcur = active.name
        self.mask_combo.blockSignals(True)
        self.mask_combo.clear()
        self.mask_combo.addItems(labs)
        if mcur in labs:
            self.mask_combo.setCurrentText(mcur)
        self.mask_combo.blockSignals(False)
        if self.model.kind == "mask" and self.model.mask_layer is None and self._mask_layer() is not None:
            self._apply_mask()

    def _on_layer_removed(self, event):
        self.overlay.on_layer_removed(event.value)
        self._refresh_timer.start(0)

    def _on_image_changed(self, _name):
        ref = self._ref()
        self.model.set_reference(ref)
        self._update_axis_rows()

    # ------------------------------------------------------------- actions
    def _on_toggled(self, on):
        self.body.setVisible(on)
        self.model.set_enabled(on)

    def _on_kind(self, *_):
        kind = "box" if self.box_rb.isChecked() else "mask"
        self.stack.setCurrentIndex(0 if kind == "box" else 1)
        self.model.set_kind(kind)
        if kind == "mask" and self.model.mask_layer is None:
            self._apply_mask()

    def _apply_mask(self):
        if self.model.kind != "mask":
            return
        lay = self._mask_layer()
        self.model.set_mask(lay, None if self.all_rb.isChecked() else self.label_spin.value())

    def _draw(self):
        self.overlay.start_draw(self._ref())

    def _full_extent(self):
        ref = self._ref()
        if ref is None:
            show_info("Select an image for the ROI first")
            return
        shape = tuple(ref.data.shape)
        self.model.set_reference(ref)
        self.model.set_box(shape, (0,) * len(shape), shape)

    def _rows(self, ref):
        """(axis, row widgets) for the axes of ref: z, y, x or y, x."""
        rows = self.axis_rows[-ref.ndim:] if ref is not None and ref.ndim <= 3 else []
        return list(zip(range(len(rows)), rows))

    def _commit_spins(self, force=False):
        ref = self._ref()
        if ref is None:
            return
        shape = tuple(ref.data.shape)
        lo, hi = [], []
        for ax, (_, a, _, b, _) in self._rows(ref):
            x, y = a.value(), b.value()
            lo.append(min(x, y))
            hi.append(max(x, y) + 1)   # From / To are inclusive, swapped if From > To
        no_box = self.model.box is None or self.model.box_shape != shape
        if no_box and not force and lo == [0] * len(shape) and hi == list(shape):
            return                     # focus left an untouched row: don't create a box
        self.model.set_reference(ref)
        self.model.set_box(shape, lo, hi)
        self._update_axis_rows()       # shows the swap

    def _set_current(self, spin):
        """'= current': From / To of the non-displayed axis := the slice shown."""
        ref = self._ref()
        if ref is None:
            return
        for ax, row in self._rows(ref):
            if spin in row:
                spin.setValue(_current_index(self.viewer, ref, ax))
        self._commit_spins(force=True)

    # ------------------------------------------------------------- display
    def _on_dims(self):
        self._update_axis_rows()
        self.overlay.sync()

    def _update_axis_rows(self):
        ref = self._ref()
        rows = dict(self._rows(ref))
        box = self.model.box if ref is not None and self.model.box_shape == tuple(ref.data.shape) else None
        displayed = _displayed_axes(self.viewer, ref) if ref is not None else []
        for r in self.axis_rows:
            for w in r:
                w.setVisible(False)
        for ax, (lab, a, a_cur, b, b_cur) in rows.items():
            n = int(ref.data.shape[ax])
            for sp, v in ((a, box[0][ax] if box else 0), (b, box[1][ax] - 1 if box else n - 1)):
                sp.blockSignals(True)
                sp.setRange(0, max(0, n - 1))
                sp.setValue(int(v))
                sp.blockSignals(False)
            cur_ok = ref.ndim == 3 and self.viewer.dims.ndisplay == 2 and ax not in displayed
            for w in (a_cur, b_cur):
                w.setEnabled(cur_ok)
            for w in (lab, a, a_cur, b, b_cur):
                w.setVisible(True)
        for w in (self.draw_btn, self.full_btn):
            w.setEnabled(ref is not None)

    def _on_model_changed(self):
        m = self.model
        if self.isChecked() != m.enabled:
            self.blockSignals(True)
            self.setChecked(m.enabled)
            self.blockSignals(False)
            self.body.setVisible(m.enabled)
        if self.box_rb.isChecked() != (m.kind == "box"):
            self.box_rb.blockSignals(True)
            (self.box_rb if m.kind == "box" else self.mask_rb).setChecked(True)
            self.box_rb.blockSignals(False)
            self.stack.setCurrentIndex(0 if m.kind == "box" else 1)
        self.summary.setText(m.summary())
        self.summary.setStyleSheet(ERROR_STYLE if m.error else OK_STYLE)
        self.mask_status.setText("computing..." if m.pending else "")
        self._update_axis_rows()
        self.overlay.sync()


# --------------------------------------------------------------------------
class RoiReminder(QLabel):
    """One line above a run button: hidden when the ROI is off."""

    def __init__(self, viewer, text="Runs inside ROI", parent=None):
        super().__init__(parent)
        self.model = get_roi_model(viewer)
        self.text_ = text
        self.setWordWrap(True)
        self.model.changed.connect(self.refresh)
        self.refresh()

    def refresh(self):
        m = self.model
        if not m.enabled:
            self.setVisible(False)
            return
        err = m.error
        if err:
            self.setText(f"ROI is on but cannot be used: {err}")
            self.setStyleSheet(ERROR_STYLE)
        else:
            self.setText(f"{self.text_}: {m.roi.describe()}")
            self.setStyleSheet(ROI_STYLE)
        self.setVisible(True)


class RoiSummaryLine(QWidget):
    """Read-only summary + 'use ROI' checkbox, for panels outside the main panel."""

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.model = get_roi_model(viewer)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel("")
        self.label.setStyleSheet(OK_STYLE)
        self.label.setWordWrap(True)
        self.use_cb = QCheckBox("use ROI")
        self.use_cb.setChecked(True)
        row.addWidget(self.label, 1)
        row.addWidget(self.use_cb)
        self.model.changed.connect(self.refresh)
        self.refresh()

    def refresh(self):
        m = self.model
        self.label.setText(m.summary())
        self.label.setStyleSheet(ERROR_STYLE if m.error else OK_STYLE)
        self.use_cb.setEnabled(m.enabled)

    def roi_for(self, shape):
        """None when unticked or off; else like RoiModel.roi_for (may raise RoiError)."""
        if not self.use_cb.isChecked():
            return None
        return self.model.roi_for(shape)
