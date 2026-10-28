"""ROI Manager (bottom of the SPROUT panel), its box overlay, and the per-tab opt-in.

    RoiManager      collapsible "ROI Manager — box 201 × 71 × 72 (8%)" section: details of
                    the current ROI and of the draft; edits the draft (a box, or a Labels
                    layer + ids used as a mask or as its bounding box); "Set as current
                    ROI" applies it, "Clear ROI" removes it. Only a Labels draft is
                    previewed (debounced, while the Manager is open on Labels).
    RoiBoxOverlay   the "SPROUT ROI" Shapes layer: the current ROI's bounding box on the
                    current slice (yellow + "ROI 201×71×72" inside its range on the
                    paging axis, orange-red + "outside ROI (z 250–450)" outside), plus an
                    edited draft in cyan ("draft …") while the Manager is open
    RoiUse          a tab's "Use current ROI" checkbox (off by default) + status line;
                    resolves the ROI for that tab's runs / displays
    RoiReminder     a line above a run button, shown only while the tab runs inside an ROI

All ROI state lives in the viewer's RoiModel (roi_model.py); the widgets connect only
to its Qt signals. napari 0.6.1 limits: a Shapes rectangle lives on one slice, and
nothing can be drawn in the 3D view.
"""
import math
import warnings
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

from sprout_core.roi import TIMES, RoiError, format_ids, format_percent, parse_ids

from .roi_model import get_roi_model

OVERLAY_NAME = "SPROUT ROI"
YELLOW = "yellow"            # the current ROI, slice inside its range
DRAFT = "cyan"               # a draft that is not applied yet, slice inside its range
OUTSIDE = "#ff5a36"          # either, slice outside its range on the paging axis
OK_STYLE = "color: gray"
ERROR_STYLE = "color: #d9534f"
ROI_STYLE = "color: #e08a00"
DRAFT_STYLE = "color: cyan"
DASH = "\u2013"              # en dash in ranges
TEXT = {"string": "{label}", "anchor": "upper_left", "size": 10, "color": {"feature": "tcolor"}}
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


def elide(name, n=32):
    """Middle-elide a long layer name: 'very_long…_name'."""
    name = str(name)
    return name if len(name) <= n else name[: (n - 1) // 2] + "\u2026" + name[-(n // 2):]


def range_text(roi, sep=" \u00b7 "):
    """'z 250–450 · y 90–160 · x 50–121' (inclusive)."""
    return sep.join(f"{a} {l}{DASH}{h - 1}" for a, l, h in zip(roi.axis_names, roi.lo, roi.hi))


def size_text(roi):
    """'201 × 71 × 72 voxels (z × y × x)'."""
    return f"{TIMES.join(str(v) for v in roi.sizes)} voxels ({TIMES.join(roi.axis_names)})"


def which_text(src):
    """'ids 3,5-7' / 'all non-zero' from a mask source."""
    ids = src.get("ids") or ([src["label"]] if src.get("label") is not None else None)
    return f"ids {format_ids(ids)}" if ids else "all non-zero"


def origin_text(roi, ref_name=None):
    """'box, coords from dog_img' / "mask of 'cells' ids 5" / "bbox of 'cells' ids 3,5 + margin 5"."""
    if roi.kind == "mask":
        return f"mask of '{elide(roi.source.get('layer'))}' {which_text(roi.source)}"
    src = roi.source.get("from_mask")
    if src:
        margin = roi.source.get("margin") or 0
        return f"bbox of '{elide(src.get('layer'))}' {which_text(src)}" + (f" + margin {margin}" if margin else "")
    return f"box, coords from {elide(ref_name)}" if ref_name else "box"


def voxels_text(roi):
    """'1,027,512 (8% of 691 × 161 × 122)'."""
    return f"{roi.n_voxels:,} ({format_percent(roi.fraction)} of {TIMES.join(str(s) for s in roi.shape)})"


def shape_mismatch_text(roi, shape):
    return f"ROI is for shape {tuple(roi.shape)}; this image is {tuple(int(s) for s in shape)}"


# --------------------------------------------------------------------------
class RoiBoxOverlay:
    """The "SPROUT ROI" Shapes layer: rectangles on the displayed 2D slice.

        current ROI   its bounding box, whenever there is a current ROI: yellow and
                      "ROI 201×71×72" while the slice is inside its range on the paging
                      axis; orange-red and "outside ROI (z 250–450)" outside
        draft         only while the Manager is open and the draft differs from the current
                      ROI: a Box draft, or the previewed Labels draft's bbox (in the labels
                      layer's frame): cyan and "draft 253×24×22"; orange-red and
                      "draft: outside (z 202–454)" outside its range

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
        self.labels = []                 # the text label of each rectangle (for checks)
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
            with warnings.catch_warnings():      # the text encoding has no features yet
                warnings.simplefilter("ignore", RuntimeWarning)
                self.layer = self.viewer.add_shapes(
                    ndim=ndim, name=OVERLAY_NAME, shape_type="rectangle", edge_color=YELLOW,
                    face_color="transparent", edge_width=width, scale=scale, translate=translate,
                    affine=affine, text=TEXT, features={"label": [], "tcolor": []}, **kw)
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
        if not drawing and self.model.draft_kind != "box":
            self.sync()          # a Labels draft / the current ROI: not editable here
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
    def _rect(self, roi, what):
        """(corners, colour, label) of roi's bbox on the displayed slice, or None.
        what: 'current' | 'draft'."""
        lay = self.layer
        lo, hi = roi.lo, roi.hi
        shown = _displayed_axes(self.viewer, lay)
        if len(shown) != 2:
            return None
        a, b = shown
        p = np.zeros(lay.ndim)
        outside = []
        coord = lay.world_to_data(self.viewer.dims.point)
        names = roi.axis_names
        for ax in range(lay.ndim):
            if ax not in shown:
                p[ax] = int(round(coord[ax]))
                if not lo[ax] <= p[ax] < hi[ax]:
                    outside.append(f"{names[ax]} {lo[ax]}{DASH}{hi[ax] - 1}")
        verts = np.repeat(p[None], 4, axis=0)
        verts[:, a] = [lo[a] - 0.5, lo[a] - 0.5, hi[a] - 0.5, hi[a] - 0.5]
        verts[:, b] = [lo[b] - 0.5, hi[b] - 0.5, hi[b] - 0.5, lo[b] - 0.5]
        size = "\u00d7".join(str(v) for v in roi.sizes)
        if outside:
            where = ", ".join(outside)
            return verts, OUTSIDE, (f"outside ROI ({where})" if what == "current" else f"draft: outside ({where})")
        return verts, (YELLOW if what == "current" else DRAFT), (f"ROI {size}" if what == "current" else f"draft {size}")

    def _hide(self):
        if self._alive() and (self.layer.visible or len(self.layer.data)):
            self._syncing = True
            try:
                self.layer.data = []
                self.layer.features = {"label": [], "tcolor": []}
                self.layer.visible = False
            finally:
                self._syncing = False
        self._kinds, self.labels = [], []

    def sync(self):
        """Draw the current ROI (and an edited draft box) on the current slice, or hide."""
        if self._syncing:
            return   # re-entered from napari events while the layer is being added / changed
        m = self.model
        if self._alive() and str(self.layer.mode).startswith("add"):
            return   # the user is drawing
        cur = m.current
        draft = dref = None
        if self.draft_shown() and m.draft_edited:
            d = m.draft_roi()
            if d is not None and d[1] is not None and d[1] in self.viewer.layers \
                    and tuple(np.shape(d[1].data)) == d[0].shape:
                draft, dref = d
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
        rects, colors, labels, kinds = [], [], [], []
        if cur is not None and cur.ndim == len(fshape) and (draft is None or self._same_frame(cref, fref, cur.ndim)):
            r = self._rect(cur, "current")
            if r is not None:
                rects.append(r[0]), colors.append(r[1]), labels.append(r[2]), kinds.append("current")
        if draft is not None:
            r = self._rect(draft, "draft")
            if r is not None:
                rects.append(r[0]), colors.append(r[1]), labels.append(r[2]), kinds.append("draft")
        if not rects:
            self._hide()
            return
        lay = self.layer
        self._syncing = True
        try:
            lay.data = [(v, "rectangle") for v in rects]
            lay.features = {"label": labels, "tcolor": colors}
            lay.edge_color = colors
            lay.visible = True
        finally:
            self._syncing = False
        self._kinds, self.labels = kinds, labels


# --------------------------------------------------------------------------
class _InfoBlock(QWidget):
    """A compact grid: bold labels left, values right (the current ROI or the draft)."""

    ROWS = ("Range", "Size", "Voxels")

    def __init__(self, first, title=None, style="", parent=None):
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(1)
        grid.setColumnStretch(1, 1)
        row = 0
        self.title = None
        if title:
            self.title = QLabel(title)
            f = self.title.font()
            f.setBold(True)
            self.title.setFont(f)
            self.title.setStyleSheet(style)
            grid.addWidget(self.title, row, 0, 1, 2)
            row += 1
        self.keys, self.values = {}, {}
        for name in (first,) + self.ROWS:
            k, v = QLabel(name), QLabel("")
            f = k.font()
            f.setBold(True)
            k.setFont(f)
            v.setWordWrap(True)
            v.setTextInteractionFlags(Qt.TextSelectableByMouse)
            if style:
                k.setStyleSheet(style)
                v.setStyleSheet(style)
            grid.addWidget(k, row, 0, Qt.AlignLeft | Qt.AlignTop)
            grid.addWidget(v, row, 1)
            self.keys[name], self.values[name] = k, v
            row += 1
        self.first = first

    def show_roi(self, origin, roi):
        """origin text + the roi's Range / Size / Voxels; roi None: only origin, other rows hidden."""
        self.values[self.first].setText(origin)
        self.values[self.first].setToolTip(roi.describe() if roi is not None else "")
        for name in self.ROWS:
            self.keys[name].setVisible(roi is not None)
            self.values[name].setVisible(roi is not None)
        if roi is not None:
            self.values["Range"].setText(range_text(roi))
            self.values["Size"].setText(size_text(roi))
            self.values["Voxels"].setText(voxels_text(roi))

    def text(self, name):
        return self.values[name].text()


class _Combo(QComboBox):
    """A layer combo showing middle-elided names; the full name is the item data and tooltip."""

    def set_names(self, names, current=None):
        self.blockSignals(True)
        self.clear()
        for n in names:
            self.addItem(elide(n), n)
            self.setItemData(self.count() - 1, n, Qt.ToolTipRole)
        if current in names:
            self.setCurrentIndex(names.index(current))
        self.blockSignals(False)
        self.setToolTip(self.name() or "")

    def name(self):
        return self.currentData()


class RoiManager(QWidget):
    """'ROI Manager': defines the one current ROI of this viewer (collapsed by default)."""

    PREVIEW_DEBOUNCE_MS = 500

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
        # Labels draft preview: after the user stops editing, only while it is shown
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(self.PREVIEW_DEBOUNCE_MS)
        self._preview_timer.timeout.connect(self._compute_preview)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.header = QToolButton()
        self.header.setCheckable(True)
        self.header.setChecked(False)
        self.header.setArrowType(Qt.RightArrow)
        self.header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.header.setStyleSheet("QToolButton { border: none; text-align: left; }")
        outer.addWidget(self.header)

        self.body = QFrame()
        self.body.setFrameShape(QFrame.StyledPanel)
        body = QVBoxLayout(self.body)
        body.setSpacing(4)
        self.current_block = _InfoBlock("Current ROI")
        body.addWidget(self.current_block)
        self.draft_block = _InfoBlock("Draft", title="Draft (not applied)", style=DRAFT_STYLE)
        body.addWidget(self.draft_block)
        self.edited_note = QLabel(EDITED_NOTE)
        self.edited_note.setStyleSheet(DRAFT_STYLE)
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

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        body.addWidget(line)
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
        self.stack.currentChanged.connect(self._fit_stack)
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
        self.error_label = QLabel("")
        self.error_label.setStyleSheet(ERROR_STYLE)
        self.error_label.setWordWrap(True)
        body.addWidget(self.error_label)
        outer.addWidget(self.body)
        self.body.setVisible(False)
        self._fit_stack(0)
        # only as tall as its content: the tabs above get the rest of the panel
        self.stack.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

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
        self.model.preview_changed.connect(self._on_preview_changed)
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
        lay.setSpacing(4)
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
        grid.setVerticalSpacing(2)
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
        lay.addStretch()

        self.draw_btn.clicked.connect(self._draw)
        self.full_btn.clicked.connect(self._full_extent)
        return page

    def _build_labels_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        grid = QGridLayout()
        grid.setVerticalSpacing(2)
        grid.setColumnStretch(3, 1)
        self.labels_combo = _Combo()
        self.labels_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.labels_combo.setMinimumContentsLength(12)
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
        self.bbox_rb = QRadioButton("bounding box, margin")
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
        grid.addWidget(self.bbox_rb, 2, 2)
        grid.addWidget(self.margin_spin, 2, 3)
        lay.addLayout(grid)
        self.preview_label = QLabel("")
        self.preview_label.setWordWrap(True)
        self.preview_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.preview_label)
        lay.addStretch()
        self.labels_combo.currentIndexChanged.connect(lambda *_: self._labels_to_draft())
        self.all_rb.toggled.connect(lambda *_: self._labels_to_draft())
        self.ids_edit.textChanged.connect(lambda *_: self._labels_to_draft())
        self.mask_rb.toggled.connect(lambda *_: self._labels_to_draft())
        self.margin_spin.valueChanged.connect(lambda *_: self._labels_to_draft())
        return page

    def _fit_stack(self, index):
        """The stack is as tall as the page shown (no empty space under the shorter one)."""
        for i in range(self.stack.count()):
            pol = QSizePolicy.Preferred if i == index else QSizePolicy.Ignored
            self.stack.widget(i).setSizePolicy(pol, pol)
        self.stack.updateGeometry()

    # ------------------------------------------------------------- helpers
    @property
    def expanded(self):
        return self.header.isChecked()

    @property
    def on_labels_page(self):
        return self.expanded and self.model.draft_kind == "labels"

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
        name = self.labels_combo.name()
        lay = self.viewer.layers[name] if name and name in self.viewer.layers else None
        return lay if isinstance(lay, Labels) else None

    def refresh_layers(self):
        labs = [l.name for l in self.viewer.layers
                if isinstance(l, Labels) and not l.name.startswith(_PREVIEW_LAYERS)]
        cur = self.model.draft_labels.name if self.model.draft_labels is not None else self.labels_combo.name()
        active = self.viewer.layers.selection.active
        if cur not in labs and isinstance(active, Labels) and active.name in labs:
            cur = active.name
        self.labels_combo.set_names(labs, cur)
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
        self._schedule_preview()
        self._update_preview_info()
        self._show_draft()
        self.overlay.sync()

    def _on_kind(self, *_):
        kind = "box" if self.box_rb.isChecked() else "labels"
        self.stack.setCurrentIndex(0 if kind == "box" else 1)
        self.model.set_draft_kind(kind)
        self.overlay.sync()

    def _labels_to_draft(self):
        self.ids_edit.setEnabled(self.ids_rb.isChecked())
        self.margin_spin.setEnabled(self.bbox_rb.isChecked())
        self.labels_combo.setToolTip(self.labels_combo.name() or "")
        self.model.set_draft_labels(
            layer=self._labels_layer(),
            ids_text=None if self.all_rb.isChecked() else self.ids_edit.text().strip(),
            use="mask" if self.mask_rb.isChecked() else "bbox",
            margin=self.margin_spin.value())

    def _set(self):
        self._preview_timer.stop()
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

    # ------------------------------------------------------------- Labels preview
    def _schedule_preview(self):
        """Debounce: the preview is computed only while the Labels page is shown."""
        if self.on_labels_page and self.model.draft_preview() is None:
            self._preview_timer.start()
        elif not self.on_labels_page:
            self._preview_timer.stop()

    def _compute_preview(self):
        if self.on_labels_page:
            self.model.compute_preview()

    def _which_draft(self, single="label"):
        """'label 5' (single='label') or 'ids 5', 'ids 3,5-7', 'all non-zero'; + ' + margin m'."""
        m = self.model
        which = "all non-zero" if m.draft_ids_text is None else f"ids {m.draft_ids_text}"
        if m.draft_ids_text is not None:
            try:
                ids = sorted(parse_ids(m.draft_ids_text))
                which = f"{single} {ids[0]}" if len(ids) == 1 else f"ids {format_ids(ids)}"
            except ValueError:
                pass
        if m.draft_use == "bbox" and m.draft_margin:
            which += f" + margin {m.draft_margin}"
        return which

    def _update_preview_info(self):
        """'Draft  label 5 → bbox z 202–454, y 130–153, x 81–102' + sizes, or the reason why not."""
        m = self.model
        style, text = OK_STYLE, ""
        try:
            m.draft_labels_key()
        except ValueError as e:
            text, style = f"Draft  {e}", (OK_STYLE if m.draft_labels is None else ERROR_STYLE)
        else:
            p = m.draft_preview()
            if p is None or p["state"] == "computing":
                text = "Draft  computing…" if self.expanded else ""
            elif p["state"] == "error":
                text, style = f"Draft  {p['error']}", ERROR_STYLE
            else:
                roi = p["roi"]
                box = roi.as_box(m.draft_margin) if m.draft_use == "bbox" else roi
                rng = range_text(box, sep=", ")
                vol = int(np.prod(box.sizes, dtype=np.int64))
                text = (f"Draft  {self._which_draft()} → bbox {rng}\n"
                        f"{TIMES.join(str(v) for v in box.sizes)} voxels ({TIMES.join(box.axis_names)})"
                        f" · {vol:,} voxels")
                if m.draft_use == "mask":
                    text += f" · mask {roi.n_voxels:,} voxels"
                style = DRAFT_STYLE
        self.preview_label.setText(text)
        self.preview_label.setStyleSheet(style)
        self.preview_label.setVisible(bool(text))

    def _on_preview_changed(self):
        self._update_preview_info()
        self._show_draft()
        self.overlay.sync()

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
            self.coords_note.setText(f"coords from: {elide(self.model.draft_ref_name)}")
            self.coords_note.setToolTip(self.model.draft_ref_name)
        elif ref is not None:
            self.coords_note.setText(f"coords from: {elide(ref.name)}")
            self.coords_note.setToolTip(ref.name)
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
        cur = self.model.current
        if cur is None:
            self.header.setText("ROI Manager — none")
            self.header.setToolTip("No current ROI: every tab works on the whole image")
            self.current_block.show_roi("none", None)
        else:
            self.header.setText(f"ROI Manager — {cur.short()}")
            self.header.setToolTip(cur.describe())
            self.current_block.show_roi(origin_text(cur, self.model.current_ref_name), cur)
        self._show_draft()
        self.overlay.allow_again()
        self.overlay.sync()

    def _show_draft(self):
        """The 'Draft (not applied)' block, while the draft differs from the current ROI."""
        m = self.model
        show = m.draft_edited
        self.draft_block.setVisible(show)
        self.edited_note.setVisible(show)
        if not show:
            return
        d = m.draft_roi()
        if m.draft_kind == "box":
            self.draft_block.show_roi(f"box, coords from {elide(m.draft_ref_name)}" if m.draft_ref_name else "box",
                                      d[0] if d else None)
            return
        lay = m.draft_labels
        kind = "bbox" if m.draft_use == "bbox" else "mask"
        which = self._which_draft(single="ids")
        origin = f"{kind} of '{elide(lay.name if lay is not None else '')}' {which}"
        if d is None:
            p = m.draft_preview()
            origin += "  (" + (p["error"] if p is not None and p["state"] == "error" else "computing…") + ")"
        self.draft_block.show_roi(origin, d[0] if d else None)

    def _on_draft_changed(self):
        m = self.model
        if self.box_rb.isChecked() != (m.draft_kind == "box"):
            self.box_rb.blockSignals(True)
            (self.box_rb if m.draft_kind == "box" else self.labels_rb).setChecked(True)
            self.box_rb.blockSignals(False)
            self.stack.setCurrentIndex(0 if m.draft_kind == "box" else 1)
        self._update_box_page()
        self._schedule_preview()
        self._update_preview_info()
        self._show_draft()
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
                text, style = f"Runs inside ROI: {self.model.current.short()}", ROI_STYLE
            self.status.setText(text)
            self.status.setToolTip(self.model.current.describe() if st == "ok" else "")
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
        cur = self.use.model.current
        self.setText(f"{self.text_}: {cur.short()}")
        self.setToolTip(cur.describe())
        self.setVisible(True)
