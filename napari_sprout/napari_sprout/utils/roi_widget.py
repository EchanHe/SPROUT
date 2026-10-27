"""ROI Manager (bottom of the SPROUT panel), its box overlay, and the per-tab opt-in.

    RoiManager      collapsible "ROI Manager — Current ROI: ..." section. Edits a draft
                    (a box, or a Labels layer + ids used as a mask or as its bounding
                    box); "Set as current ROI" applies it, "Clear ROI" removes it.
                    Nothing is computed while the draft is edited.
    RoiBoxOverlay   the "SPROUT ROI" Shapes layer: the current ROI's bounding box on the
                    current slice (yellow inside its range on the third axis, grey
                    outside), plus an edited draft box in cyan while the Manager is open
    RoiUse          a tab's "Use current ROI" checkbox (off by default) + status line;
                    resolves the ROI for that tab's runs / displays
    RoiReminder     a line above a run button, shown only while the tab runs inside an ROI

All ROI state lives in the viewer's RoiModel (roi_model.py); the widgets connect only
to its Qt signals. napari 0.6.1 limits: a Shapes rectangle lives on one slice, and
nothing can be drawn in the 3D view.
"""
import math
import weakref

import numpy as np
from napari.layers import Image, Labels
from napari.utils.notifications import show_info
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QRadioButton, QSizePolicy, QSpinBox, QStackedWidget, QToolButton,
    QVBoxLayout, QWidget,
)

from sprout_core.roi import RoiError

from .roi_model import get_roi_model

OVERLAY_NAME = "SPROUT ROI"
YELLOW = "yellow"
GREY = "#99999999"           # (0.6, 0.6, 0.6, 0.6)
DRAFT = "cyan"               # a draft box that is not applied yet
DRAFT_DIM = "#00ffff66"      # ...outside its range on the non-displayed axis
OK_STYLE = "color: gray"
ERROR_STYLE = "color: #d9534f"
ROI_STYLE = "color: #e08a00"
# SPROUT's own live-preview Labels layers are never offered as an ROI mask
_PREVIEW_LAYERS = ("Threshold Preview", "Boundary preview")

NO_ROI_ERROR = "No current ROI: set one in the ROI Manager (bottom of the panel), or untick Use current ROI"
NO_ROI_HINT = "No current ROI — set one in the ROI Manager below"
EDITED_NOTE = "Edited — click Set as current ROI to apply"


def _displayed_axes(viewer, ref):
    """Data axes of `ref` that the viewer displays."""
    offset = viewer.dims.ndim - ref.ndim
    return [a - offset for a in viewer.dims.displayed if a >= offset]


def _current_index(viewer, ref, ax):
    return int(round(ref.world_to_data(viewer.dims.point)[ax]))


def _weak(obj):
    return None if obj is None else weakref.ref(obj)


def _deref(ref):
    return None if ref is None else ref()


def _usable_image(layer):
    return isinstance(layer, Image) and not layer.rgb and not layer.multiscale


def shape_mismatch_text(roi, shape):
    return f"ROI is for shape {tuple(roi.shape)}; this image is {tuple(int(s) for s in shape)}"


# --------------------------------------------------------------------------
class RoiBoxOverlay:
    """The "SPROUT ROI" Shapes layer: rectangles on the displayed 2D slice.

        current ROI   its bounding box, whenever there is a current ROI: yellow while the
                      slice is inside its range on the non-displayed axis, grey outside
        draft box     only while the Manager is open on Box and the draft differs from the
                      current ROI: cyan (dim cyan outside its range)

    Purely visual: nothing is computed here. The layer gets the reference layer's scale /
    translate / rotate / shear / affine, so its data coordinates are voxel indices
    (anisotropic images included): for the current ROI the image it was drawn on, or
    the labels layer it was made from (else an Image layer of the same shape, else
    identity); while a draft is shown, the draft's image. A rectangle in another
    frame is not drawn. Every programmatic change of the layer runs under `_syncing`,
    so that only the user's own drawing / editing / deleting reaches the model (as a
    draft edit: the current ROI only changes with Set / Clear). Other layers are only
    held by weak reference.
    """

    def __init__(self, viewer, model, draft_shown=lambda: True):
        self.viewer = viewer
        self.model = model
        self.draft_shown = draft_shown   # callable: may the draft box be shown now?
        self.layer = None
        self._ref = None                 # image the user draws on
        self._frame = None               # (weakref to the layer giving the frame, shape) of the last sync
        self._kinds = []                 # what each rectangle is: 'current' | 'draft'
        self._syncing = False
        self._prev = None
        self._deleted_by_user = False

    @property
    def ref(self):
        return _deref(self._ref)

    @ref.setter
    def ref(self, layer):
        self._ref = _weak(layer)

    @property
    def _prev_active(self):
        return _deref(self._prev)

    @_prev_active.setter
    def _prev_active(self, layer):
        self._prev = _weak(layer)

    # ---- frames
    @staticmethod
    def _transform(ref, ndim):
        if ref is None:
            return np.ones(ndim), np.zeros(ndim), np.eye(ndim + 1)
        return np.asarray(ref.scale), np.asarray(ref.translate), np.asarray(ref.affine.affine_matrix)

    def _same_frame(self, a, b, ndim):
        ta, tb = self._transform(a, ndim), self._transform(b, ndim)
        return all(x.shape == y.shape and np.allclose(x, y) for x, y in zip(ta, tb))

    def _current_ref(self):
        """Layer whose transform places the current ROI: its own reference, else an Image
        layer of the same shape, else None (identity)."""
        roi = self.model.current
        ref = self.model.current_ref
        if ref is not None and ref in self.viewer.layers and tuple(np.shape(ref.data)) == roi.shape:
            return ref
        return next((l for l in self.viewer.layers
                     if _usable_image(l) and tuple(np.shape(l.data)) == roi.shape), None)

    # ---- layer
    def _alive(self):
        return self.layer is not None and self.layer in self.viewer.layers

    def _matches(self, ref, ndim):
        return self.layer.ndim == ndim and self._same_frame(self.layer, ref, ndim)

    def ensure_layer(self, ref, shape=None):
        shape = tuple(ref.data.shape) if shape is None else tuple(shape)
        ndim = len(shape)
        if self._alive() and self._matches(ref, ndim):
            return self.layer
        if self._alive():
            self._syncing = True
            try:
                self.viewer.layers.remove(self.layer)
            finally:
                self._syncing = False
        scale, translate, affine = self._transform(ref, ndim)
        kw = {} if ref is None else {"rotate": ref.rotate, "shear": ref.shear}
        offset = self.viewer.dims.ndim - ndim
        shown = [a - offset for a in self.viewer.dims.displayed if a >= offset] or list(range(ndim))
        width = max(1.0, 0.004 * float(max(shape[a] for a in shown)))
        active = self.viewer.layers.selection.active
        self._syncing = True
        try:
            self.layer = self.viewer.add_shapes(
                ndim=ndim, name=OVERLAY_NAME, shape_type="rectangle", edge_color=YELLOW,
                face_color="transparent", edge_width=width, scale=scale, translate=translate,
                affine=affine, **kw)
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
        if layer is self.ref:
            self.ref = None
        if layer is self._prev_active:
            self._prev_active = None

    def allow_again(self):
        """A new Set / Clear / Draw: a layer the user deleted may come back."""
        self._deleted_by_user = False

    def remove_layer(self):
        """Take the overlay out of the viewer (the panel is closing)."""
        lay, self.layer = self.layer, None
        try:
            if lay is not None and lay in self.viewer.layers:
                self._syncing = True
                self.viewer.layers.remove(lay)
        except Exception:
            pass                     # the viewer is closing too
        finally:
            self._syncing = False

    # ---- drawing
    def start_draw(self, ref):
        """Let the user drag a rectangle on the displayed slice (it becomes the draft box)."""
        if ref is None:
            show_info("Load an image first: the box is drawn in its voxel grid")
            return False
        if self.viewer.dims.ndisplay != 2:
            show_info("Switch to the 2D view to draw")
            return False
        self.ref = ref
        self.allow_again()
        prev = self.viewer.layers.selection.active
        lay = self.ensure_layer(ref)
        self._prev_active = prev if prev is not lay else None
        self._syncing = True
        try:
            lay.data = []
            lay.edge_color = DRAFT
            lay.visible = True
        finally:
            self._syncing = False
        self._kinds = []
        self._frame = (weakref.ref(ref), tuple(ref.data.shape))
        self.viewer.layers.selection.active = lay
        lay.mode = "add_rectangle"
        return True

    def _on_data(self, event):
        if self._syncing:
            return
        action = str(getattr(event, "action", ""))
        if action not in ("added", "changed", "removed"):
            return
        idx = getattr(event, "data_indices", None)
        idx = [int(i) for i in idx] if idx is not None and len(idx) and int(idx[0]) >= 0 else None
        # ADDED fires inside napari's mouse-release handler: change the layer later
        QTimer.singleShot(0, lambda: self._apply_from_shapes(action, idx))

    def _apply_from_shapes(self, action, idx=None):
        """The user drew / edited / deleted a rectangle: that is a draft edit."""
        lay = self.layer
        if not self._alive():
            return
        drawing = str(lay.mode).startswith("add")
        if drawing:
            ref, shape = self.ref, (tuple(self.ref.data.shape) if self.ref is not None else None)
        else:
            fref, shape = self._frame if self._frame else (None, None)
            ref = fref() if fref is not None else None
        if shape is None:
            self._finish_drawing()
            return
        if action == "removed":
            gone = [self._kinds[i] for i in (idx or []) if i < len(self._kinds)] or ["draft"]
            if "draft" in gone:
                self.model.clear_draft_box()
            self._finish_drawing()
            self.sync()
            return
        if len(lay.data) == 0:
            self._finish_drawing()
            return
        i = len(lay.data) - 1
        if action == "changed" and idx:
            i = min(idx[0], i)
        verts = np.asarray(lay.data[i], float)
        old = self.model.draft_box
        old = old if old is not None and old.shape == shape else None
        lo = list(old.lo) if old else [0] * len(shape)
        hi = list(old.hi) if old else list(shape)
        for ax in _displayed_axes(self.viewer, lay):
            # a voxel is inside if its centre is (pixel-centre rule; idempotent on our corners)
            lo[ax] = min(max(int(math.ceil(verts[:, ax].min())), 0), shape[ax])
            hi[ax] = min(max(int(math.floor(verts[:, ax].max())) + 1, lo[ax]), shape[ax])
        self.model.set_draft_box(shape, lo, hi, ref=ref)
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
    def _rect(self, lo, hi):
        """(corners, inside) of the box [lo, hi) on the displayed slice, or None."""
        lay = self.layer
        shown = _displayed_axes(self.viewer, lay)
        if len(shown) != 2:
            return None
        a, b = shown
        p = np.zeros(lay.ndim)
        inside = True
        coord = lay.world_to_data(self.viewer.dims.point)
        for ax in range(lay.ndim):
            if ax not in shown:
                p[ax] = int(round(coord[ax]))
                inside &= lo[ax] <= p[ax] < hi[ax]
        verts = np.repeat(p[None], 4, axis=0)
        verts[:, a] = [lo[a] - 0.5, lo[a] - 0.5, hi[a] - 0.5, hi[a] - 0.5]
        verts[:, b] = [lo[b] - 0.5, hi[b] - 0.5, hi[b] - 0.5, lo[b] - 0.5]
        return verts, inside or lay.ndim == 2

    def _hide(self):
        if self._alive() and (self.layer.visible or len(self.layer.data)):
            self._syncing = True
            try:
                self.layer.data = []
                self.layer.visible = False
            finally:
                self._syncing = False
        self._kinds = []

    def sync(self):
        """Draw the current ROI (and an edited draft box) on the current slice, or hide."""
        if self._syncing:
            return   # re-entered from napari events while the layer is being added / changed
        m = self.model
        if self._alive() and str(self.layer.mode).startswith("add"):
            return   # the user is drawing
        cur = m.current
        draft, dref = m.draft_box, m.draft_ref
        if not (self.draft_shown() and m.draft_kind == "box" and draft is not None and m.draft_edited
                and dref is not None and dref in self.viewer.layers and tuple(dref.data.shape) == draft.shape):
            draft = None
        if self.viewer.dims.ndisplay != 2 or (cur is None and draft is None):
            self._hide()
            return
        cref = self._current_ref() if cur is not None else None
        # the layer's frame: the draft's while it is edited, else the current ROI's
        fref, fshape = (dref, draft.shape) if draft is not None else (cref, cur.shape)
        if not self._alive() or not self._matches(fref, len(fshape)):
            if self._deleted_by_user:
                return
            self.ensure_layer(fref, fshape)
        self._frame = (None if fref is None else weakref.ref(fref), tuple(fshape))
        rects, colors, kinds = [], [], []
        if cur is not None and cur.ndim == len(fshape) and (draft is None or self._same_frame(cref, fref, cur.ndim)):
            r = self._rect(cur.lo, cur.hi)
            if r is not None:
                rects.append(r[0]), colors.append(YELLOW if r[1] else GREY), kinds.append("current")
        if draft is not None:
            r = self._rect(draft.lo, draft.hi)
            if r is not None:
                rects.append(r[0]), colors.append(DRAFT if r[1] else DRAFT_DIM), kinds.append("draft")
        if not rects:
            self._hide()
            return
        lay = self.layer
        self._syncing = True
        try:
            lay.data = [(v, "rectangle") for v in rects]
            lay.edge_color = colors
            lay.visible = True
        finally:
            self._syncing = False
        self._kinds = kinds


# --------------------------------------------------------------------------
class RoiManager(QWidget):
    """'ROI Manager': defines the one current ROI of this viewer (collapsed by default)."""

    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent)
        self.viewer = napari_viewer
        self.model = get_roi_model(napari_viewer)
        self.overlay = RoiBoxOverlay(napari_viewer, self.model, draft_shown=lambda: self.expanded)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self.refresh_layers)
        # dims events also fire while napari adds a layer: redraw on the next turn
        self._dims_timer = QTimer(self)
        self._dims_timer.setSingleShot(True)
        self._dims_timer.timeout.connect(self._on_dims_settled)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.header = QToolButton()
        self.header.setCheckable(True)
        self.header.setChecked(False)
        self.header.setArrowType(Qt.RightArrow)
        self.header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.header.setStyleSheet("QToolButton { border: none; text-align: left; }")
        self.header.setToolTip("One current ROI for every tab that ticks 'Use current ROI'")
        outer.addWidget(self.header)

        self.body = QFrame()
        self.body.setFrameShape(QFrame.StyledPanel)
        body = QVBoxLayout(self.body)
        row = QHBoxLayout()
        self.box_rb = QRadioButton("Box")
        self.labels_rb = QRadioButton("Labels")
        self.box_rb.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.box_rb)
        grp.addButton(self.labels_rb)
        row.addWidget(QLabel("Define from:"))
        row.addWidget(self.box_rb)
        row.addWidget(self.labels_rb)
        row.addStretch()
        body.addLayout(row)
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_box_page())
        self.stack.addWidget(self._build_labels_page())
        body.addWidget(self.stack)

        row = QHBoxLayout()
        self.set_btn = QPushButton("Set as current ROI")
        self.set_btn.setToolTip("Use the box / labels above as the current ROI")
        self.clear_btn = QPushButton("Clear ROI")
        self.clear_btn.setToolTip("No current ROI: every tab works on the whole image")
        row.addWidget(self.set_btn)
        row.addWidget(self.clear_btn)
        row.addStretch()
        body.addLayout(row)
        self.edited_note = QLabel(EDITED_NOTE)
        self.edited_note.setStyleSheet(OK_STYLE)
        body.addWidget(self.edited_note)
        row = QHBoxLayout()
        self.source_note = QLabel("")
        self.source_note.setStyleSheet(ROI_STYLE)
        self.source_note.setWordWrap(True)
        self.update_btn = QPushButton("Update")
        self.update_btn.setToolTip("Make the current ROI again from the edited layer")
        row.addWidget(self.source_note, 1)
        row.addWidget(self.update_btn)
        body.addLayout(row)
        self.error_label = QLabel("")
        self.error_label.setStyleSheet(ERROR_STYLE)
        self.error_label.setWordWrap(True)
        body.addWidget(self.error_label)
        outer.addWidget(self.body)
        self.body.setVisible(False)

        self.header.toggled.connect(self._on_expanded)
        self.box_rb.toggled.connect(self._on_kind)
        self.set_btn.clicked.connect(self._set)
        self.clear_btn.clicked.connect(self.model.clear)
        self.update_btn.clicked.connect(self.model.update)
        # Only Qt signals of the model (which lives with the viewer): when this
        # widget is deleted, Qt drops the connections, nothing stays behind in napari.
        self.model.changed.connect(self._on_current_changed)
        self.model.draft_changed.connect(self._on_draft_changed)
        self.model.status_changed.connect(self._on_status_changed)
        self.model.layer_inserted.connect(self._on_layer_inserted)
        self.model.layer_removed.connect(self._on_layer_removed)
        self.model.dims_changed.connect(self._on_dims)
        overlay = self.overlay
        self.destroyed.connect(lambda *_: overlay.remove_layer())   # the box goes with the panel
        self.refresh_layers()
        self._on_current_changed()
        self._on_draft_changed()
        self._on_status_changed()

    # ------------------------------------------------------------- pages
    def _build_box_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.draw_btn = QPushButton("Draw box")
        self.draw_btn.setToolTip("Drag a rectangle on the 2D view: it sets the two displayed axes. "
                                 "The third axis keeps its range (the whole image the first time).")
        self.full_btn = QPushButton("Full extent")
        self.full_btn.setToolTip("The whole image")
        row.addWidget(self.draw_btn)
        row.addWidget(self.full_btn)
        row.addStretch()
        lay.addLayout(row)
        self.coords_note = QLabel("")
        self.coords_note.setStyleSheet(OK_STYLE)
        self.coords_note.setToolTip("Voxel indices are those of this image: the active Image layer "
                                    "when you draw (else the first one)")
        lay.addWidget(self.coords_note)

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
            lo_cur.clicked.connect(lambda _=False, s=lo: self._set_current_slice(s))
            hi_cur.clicked.connect(lambda _=False, s=hi: self._set_current_slice(s))
            for c, w in enumerate((lab, lo, lo_cur, hi, hi_cur)):
                grid.addWidget(w, i, c)
            self.axis_rows.append((lab, lo, lo_cur, hi, hi_cur))
        lay.addLayout(grid)

        self.draw_btn.clicked.connect(self._draw)
        self.full_btn.clicked.connect(self._full_extent)
        return page

    def _build_labels_page(self):
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)
        self.labels_combo = QComboBox()
        grid.addWidget(QLabel("Labels layer:"), 0, 0)
        grid.addWidget(self.labels_combo, 0, 1, 1, 3)
        self.all_rb = QRadioButton("all non-zero")
        self.ids_rb = QRadioButton("ids")
        self.all_rb.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.all_rb)
        grp.addButton(self.ids_rb)
        self.ids_edit = QLineEdit()
        self.ids_edit.setPlaceholderText("e.g. 3,5-7")
        self.ids_edit.setToolTip("Label ids: single ids and ranges, separated by commas")
        grid.addWidget(QLabel("Ids:"), 1, 0)
        grid.addWidget(self.all_rb, 1, 1)
        grid.addWidget(self.ids_rb, 1, 2)
        grid.addWidget(self.ids_edit, 1, 3)
        self.mask_rb = QRadioButton("mask")
        self.bbox_rb = QRadioButton("bounding box, with margin")
        self.mask_rb.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.mask_rb)
        grp.addButton(self.bbox_rb)
        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(0, 100000)
        self.margin_spin.setValue(0)
        self.margin_spin.setSuffix(" voxels")
        grid.addWidget(QLabel("Use as:"), 2, 0)
        grid.addWidget(self.mask_rb, 2, 1)
        row = QHBoxLayout()
        row.addWidget(self.bbox_rb)
        row.addWidget(self.margin_spin)
        row.addStretch()
        grid.addLayout(row, 2, 2, 1, 2)
        self.labels_combo.currentTextChanged.connect(lambda *_: self._labels_to_draft())
        self.all_rb.toggled.connect(lambda *_: self._labels_to_draft())
        self.ids_edit.textChanged.connect(lambda *_: self._labels_to_draft())
        self.mask_rb.toggled.connect(lambda *_: self._labels_to_draft())
        self.margin_spin.valueChanged.connect(lambda *_: self._labels_to_draft())
        return page

    # ------------------------------------------------------------- helpers
    @property
    def expanded(self):
        return self.header.isChecked()

    def _pick_ref(self):
        """The image the user works on: the active Image layer, else the first one."""
        active = self.viewer.layers.selection.active
        if _usable_image(active):
            return active
        return next((l for l in self.viewer.layers if _usable_image(l)), None)

    def _box_ref(self):
        """Coordinates for editing the draft box: the image it was drawn on, else _pick_ref()."""
        ref = self.model.draft_ref
        return ref if ref is not None and ref in self.viewer.layers else self._pick_ref()

    def _labels_layer(self):
        name = self.labels_combo.currentText()
        lay = self.viewer.layers[name] if name and name in self.viewer.layers else None
        return lay if isinstance(lay, Labels) else None

    def refresh_layers(self):
        labs = [l.name for l in self.viewer.layers
                if isinstance(l, Labels) and not l.name.startswith(_PREVIEW_LAYERS)]
        cur = self.model.draft_labels.name if self.model.draft_labels is not None else self.labels_combo.currentText()
        active = self.viewer.layers.selection.active
        if cur not in labs and isinstance(active, Labels) and active.name in labs:
            cur = active.name
        self.labels_combo.blockSignals(True)
        self.labels_combo.clear()
        self.labels_combo.addItems(labs)
        if cur in labs:
            self.labels_combo.setCurrentText(cur)
        self.labels_combo.blockSignals(False)
        if self._labels_layer() is not self.model.draft_labels:
            self._labels_to_draft()
        self._update_box_page()

    def _on_layer_inserted(self, layer):
        self._refresh_timer.start(0)

    def _on_layer_removed(self, layer):
        self.overlay.on_layer_removed(layer)
        self._refresh_timer.start(0)

    # ------------------------------------------------------------- actions
    def _on_expanded(self, on):
        self.body.setVisible(on)
        self.header.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self.overlay.sync()

    def _on_kind(self, *_):
        kind = "box" if self.box_rb.isChecked() else "labels"
        self.stack.setCurrentIndex(0 if kind == "box" else 1)
        self.model.set_draft_kind(kind)
        self.overlay.sync()

    def _labels_to_draft(self):
        self.ids_edit.setEnabled(self.ids_rb.isChecked())
        self.margin_spin.setEnabled(self.bbox_rb.isChecked())
        self.model.set_draft_labels(
            layer=self._labels_layer(),
            ids_text=None if self.all_rb.isChecked() else self.ids_edit.text().strip(),
            use="mask" if self.mask_rb.isChecked() else "bbox",
            margin=self.margin_spin.value())

    def _set(self):
        self.model.apply()

    def _draw(self):
        self.overlay.start_draw(self._pick_ref())

    def _full_extent(self):
        ref = self._pick_ref()
        if ref is None:
            show_info("Load an image first: the box is drawn in its voxel grid")
            return
        shape = tuple(ref.data.shape)
        self.model.set_draft_box(shape, (0,) * len(shape), shape, ref=ref)

    def _rows(self, ref):
        """(axis, row widgets) for the axes of ref: z, y, x or y, x."""
        rows = self.axis_rows[-ref.ndim:] if ref is not None and ref.ndim <= 3 else []
        return list(zip(range(len(rows)), rows))

    def _commit_spins(self, force=False):
        ref = self._box_ref()
        if ref is None:
            return
        shape = tuple(ref.data.shape)
        lo, hi = [], []
        for ax, (_, a, _, b, _) in self._rows(ref):
            x, y = a.value(), b.value()
            lo.append(min(x, y))
            hi.append(max(x, y) + 1)   # From / To are inclusive, swapped if From > To
        box = self.model.draft_box
        no_box = box is None or box.shape != shape
        if no_box and not force and lo == [0] * len(shape) and hi == list(shape):
            return                     # focus left an untouched row: don't create a box
        self.model.set_draft_box(shape, lo, hi, ref=ref)
        self._update_box_page()        # shows the swap

    def _set_current_slice(self, spin):
        """'= current': From / To of the non-displayed axis := the slice shown."""
        ref = self._box_ref()
        if ref is None:
            return
        for ax, row in self._rows(ref):
            if spin in row:
                spin.setValue(_current_index(self.viewer, ref, ax))
        self._commit_spins(force=True)

    # ------------------------------------------------------------- display
    def _on_dims(self):
        self._dims_timer.start(0)

    def _on_dims_settled(self):
        if self.expanded:
            self._update_box_page()
        self.overlay.sync()

    def _update_box_page(self):
        ref = self._box_ref()
        box = self.model.draft_box
        if box is not None and (ref is None or box.shape != tuple(ref.data.shape)):
            box = None
        if self.model.draft_box is not None and self.model.draft_ref_name:
            self.coords_note.setText(f"coords from: {self.model.draft_ref_name}")
        elif ref is not None:
            self.coords_note.setText(f"coords from: {ref.name}")
        else:
            self.coords_note.setText("coords from: (no image layer yet)")
        rows = dict(self._rows(ref))
        displayed = _displayed_axes(self.viewer, ref) if ref is not None else []
        for r in self.axis_rows:
            for w in r:
                w.setVisible(False)
        for ax, (lab, a, a_cur, b, b_cur) in rows.items():
            n = int(ref.data.shape[ax])
            for sp, v in ((a, box.lo[ax] if box else 0), (b, box.hi[ax] - 1 if box else n - 1)):
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

    def _on_current_changed(self):
        self.header.setText(f"ROI Manager — Current ROI: {self.model.summary()}")
        self.overlay.allow_again()
        self.overlay.sync()

    def _on_draft_changed(self):
        m = self.model
        if self.box_rb.isChecked() != (m.draft_kind == "box"):
            self.box_rb.blockSignals(True)
            (self.box_rb if m.draft_kind == "box" else self.labels_rb).setChecked(True)
            self.box_rb.blockSignals(False)
            self.stack.setCurrentIndex(0 if m.draft_kind == "box" else 1)
        self.edited_note.setVisible(m.draft_edited)
        self._update_box_page()
        self.overlay.sync()

    def _on_status_changed(self):
        m = self.model
        self.set_btn.setText("Setting..." if m.busy else "Set as current ROI")
        self.set_btn.setEnabled(not m.busy)
        self.error_label.setText(f"Cannot set the ROI: {m.error}" if m.error else "")
        self.error_label.setVisible(bool(m.error))
        note = m.source_note()
        self.source_note.setText(note)
        self.source_note.setVisible(bool(note))
        self.update_btn.setVisible(m.source_edited and not m.source_removed)
        self.update_btn.setEnabled(not m.busy)


# --------------------------------------------------------------------------
class RoiUse(QWidget):
    """A tab's 'Use current ROI' checkbox (off by default) and its status line.

    shape: callable giving the shape of the tab's image (or None). The tab calls
    refresh() when its image changes. `changed` fires when the ROI this tab uses may
    have changed: the box was (un)ticked, or the current ROI changed while ticked.
    """
    changed = Signal()
    refreshed = Signal()

    def __init__(self, viewer, shape, description="", parent=None):
        super().__init__(parent)
        self.model = get_roi_model(viewer)
        self.shape = shape
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        row = QHBoxLayout()
        self.use_cb = QCheckBox("Use current ROI")
        self.use_cb.setChecked(False)
        self.desc = QLabel(description)
        self.desc.setStyleSheet(OK_STYLE)
        self.desc.setWordWrap(True)
        row.addWidget(self.use_cb)
        row.addWidget(self.desc, 1)
        lay.addLayout(row)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.use_cb.toggled.connect(self._on_toggled)
        self.model.changed.connect(self._on_current_changed)
        self.refresh()

    @property
    def checked(self):
        return self.use_cb.isChecked()

    def _shape(self, shape=None):
        if shape is None and self.shape is not None:
            shape = self.shape()
        return None if shape is None else tuple(int(s) for s in shape)

    def state(self, shape=None):
        """'off' | 'none' | 'mismatch' | 'ok'."""
        if not self.checked:
            return "off"
        roi = self.model.current
        if roi is None:
            return "none"
        shape = self._shape(shape)
        if shape is not None and shape != tuple(roi.shape):
            return "mismatch"
        return "ok"

    def roi_for_run(self, shape=None):
        """None when unticked; the current ROI when usable; else RoiError (shown to the user)."""
        st = self.state(shape)
        if st == "off":
            return None
        if st == "none":
            raise RoiError(NO_ROI_ERROR)
        if st == "mismatch":
            raise RoiError(shape_mismatch_text(self.model.current, self._shape(shape)))
        return self.model.current

    def roi_for_display(self, shape=None):
        try:
            return self.roi_for_run(shape)
        except RoiError:
            return None

    def _on_toggled(self, _on):
        self.refresh()
        self.changed.emit()

    def _on_current_changed(self):
        self.refresh()
        if self.checked:
            self.changed.emit()

    def refresh(self):
        st = self.state()
        if st == "off":
            self.status.setVisible(False)
        else:
            if st == "none":
                text, style = NO_ROI_HINT, ROI_STYLE
            elif st == "mismatch":
                text, style = shape_mismatch_text(self.model.current, self._shape()), ERROR_STYLE
            else:
                text, style = f"Runs inside ROI: {self.model.current.describe()}", ROI_STYLE
            self.status.setText(text)
            self.status.setStyleSheet(style)
            self.status.setVisible(True)
        self.refreshed.emit()


class RoiReminder(QLabel):
    """One line above a run button, shown only while its tab runs inside an ROI."""

    def __init__(self, use, text="Runs inside ROI", parent=None):
        super().__init__(parent)
        self.use = use
        self.text_ = text
        self.setWordWrap(True)
        self.setStyleSheet(ROI_STYLE)
        use.refreshed.connect(self.refresh)
        self.refresh()

    def refresh(self):
        if self.use.state() != "ok":
            self.setVisible(False)
            return
        self.setText(f"{self.text_}: {self.use.model.current.describe()}")
        self.setVisible(True)
