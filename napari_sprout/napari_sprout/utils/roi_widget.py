"""ROI widgets: the per-tab ROI picker and the ROI tools (ROI & Boundary -> ROI).

    RoiSelect   "ROI [None ▾ | ROI_1 | ...]" in a tab's input area, with two info lines
                (range; size, voxels, share) and a red line when the ROI cannot be used.
                Resolves the ROI for that tab: roi_for_run(shape) / roi_for_display(shape).
                Default None; never picks a new ROI by itself.
    RoiReminder one line by a run / preview button, shown while its tab uses an ROI
    RoiTools    four blocks: Draw a box (draw, full image), From labels (ids + margin),
                Edit ROI (z / y / x range, delete; rename in napari's layer list), Save / Load

ROIs are napari Shapes layers (roi_layers.py); every napari event reaches these widgets
through the viewer's RoiHub as a Qt signal, so a closed panel leaves nothing behind.
"""
import os
import weakref

import numpy as np
from napari.layers import Image, Labels
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

from sprout_core import roi as R
from sprout_core.roi import RoiError, format_percent, parse_ids

from .roi_layers import (
    DASH, HIDDEN_NOTE, add_roi_layer, get_roi_hub, is_roi_layer, layer_extra, layer_meta, sizes_text,
)

OK_STYLE = "color: gray"
ERROR_STYLE = "color: #d9534f"
ROI_STYLE = "color: #e08a00"
TIMES = " × "
RENAME_TIP = "Rename in the napari layer list"
# SPROUT's own live-preview Labels layers are never offered as a source of label ids
_PREVIEW_LAYERS = ("Threshold Preview", "Boundary preview")


def elide(name, n=32):
    """Middle-elide a long layer name: 'very_long…_name'."""
    name = str(name)
    return name if len(name) <= n else name[: (n - 1) // 2] + "…" + name[-(n // 2):]


def range_text(roi, sep=" · "):
    """'z 250–450 · y 90–160 · x 50–121' (inclusive)."""
    return sep.join(f"{a} {l}{DASH}{h - 1}" for a, l, h in zip(roi.axis_names, roi.lo, roi.hi))


def voxels_text(n):
    return f"{n / 1e6:.2f} M voxels" if n >= 1_000_000 else f"{n:,} voxels"


def size_line(roi):
    """'201 × 71 × 72 = 1.03 M voxels (8%)'."""
    return f"{TIMES.join(str(v) for v in roi.sizes)} = {voxels_text(roi.n_voxels)} ({format_percent(roi.fraction)})"


def mismatch_text(roi, shape):
    return f"{roi.name} is {sizes_text(roi.shape)}; image is {sizes_text(shape)}"


# --------------------------------------------------------------------------
class RoiSelect(QWidget):
    """A tab's ROI picker: combo + info lines. `changed`: the ROI this tab uses changed
    (picked, removed, or a committed edit of the picked layer, debounced 250 ms);
    `moving`: the picked ROI is being dragged (texts only)."""
    changed = Signal()
    moving = Signal()
    refreshed = Signal()

    DEBOUNCE_MS = 250

    def __init__(self, viewer, shape, tab_name="", run_button=None, run_text=None, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.hub = get_roi_hub(viewer)
        self.shape = shape
        self.tab_name = tab_name
        self._run = (weakref.ref(run_button), run_text) if run_button is not None else None
        self._layer = None                 # weakref to the picked layer
        self._layer_name = None
        self._removed = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(1)
        self.combo = QComboBox()
        self.combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo.setMinimumContentsLength(10)
        self.combo.setToolTip("Region of interest: runs only look inside it. None = the whole image.\n"
                              "Create and edit ROIs in 'ROI & Boundary' -> ROI.")
        lay.addWidget(self.combo)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        self.error = QLabel("")
        self.error.setWordWrap(True)
        self.error.setStyleSheet(ERROR_STYLE)
        lay.addWidget(self.error)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(self.DEBOUNCE_MS)
        self._debounce.timeout.connect(self.changed)
        self.combo.activated.connect(self._on_activated)
        self.hub.rois_changed.connect(self._rebuild)
        self.hub.roi_edited.connect(self._on_edited)
        self.hub.roi_moving.connect(self._on_moving)
        self.hub.register_user(self, tab_name)
        self._rebuild()

    # ---- state
    @property
    def layer(self):
        lay = None if self._layer is None else self._layer()
        return lay if lay is not None and lay in self.viewer.layers and is_roi_layer(lay) else None

    def _shape(self, shape=None):
        if shape is None and self.shape is not None:
            shape = self.shape()
        return None if shape is None else tuple(int(s) for s in shape)

    def state(self, shape=None):
        """'none' | 'ok' | 'mismatch' | 'removed' | 'invalid'."""
        if self._layer is None:
            return "none"
        layer = self.layer
        if layer is None:
            return "removed"
        try:
            roi = self.hub.roi_of(layer)
        except RoiError:
            return "invalid"
        shape = self._shape(shape)
        if shape is not None and shape != tuple(roi.shape):
            return "mismatch"
        return "ok"

    def roi_for_run(self, shape=None):
        """None when no ROI is picked; the picked ROI (a snapshot) if usable; else RoiError."""
        st = self.state(shape)
        if st == "none":
            return None
        if st == "removed":
            raise RoiError(f"ROI '{self._layer_name}' was removed — choose another ROI or None")
        roi = self.hub.roi_of(self.layer)              # RoiError if the layer has no box
        if st == "mismatch":
            raise RoiError(mismatch_text(roi, self._shape(shape)))
        return roi

    def roi_for_display(self, shape=None):
        try:
            return self.roi_for_run(shape)
        except RoiError:
            return None

    def set_run_button(self, button, text):
        """The button whose text names the ROI: '<text> in ROI_1' (plain text with None)."""
        self._run = (weakref.ref(button), text)
        self.refresh()

    def select(self, layer):
        """Pick an ROI layer (or None) as if chosen in the combo."""
        self._layer = None if layer is None else weakref.ref(layer)
        self._layer_name = None if layer is None else layer.name
        self._rebuild()
        self.hub.users_changed.emit()
        self.changed.emit()

    # ---- combo
    def _rebuild(self):
        layers = self.hub.layers()
        cur = self.layer
        if cur is not None:
            self._layer_name = cur.name
        shape = self._shape()
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("None", None)
        for l in layers:
            text = elide(l.name)
            md = layer_meta(l)
            if shape is not None and tuple(md.get("shape", ())) != shape:
                text += f" ({sizes_text(md.get('shape', ()))})"
            self.combo.addItem(text, weakref.ref(l))
            self.combo.setItemData(self.combo.count() - 1, l.name, Qt.ToolTipRole)
        if self._layer is not None and cur is None:          # picked layer removed: keep it visible
            self.combo.addItem(f"{elide(self._layer_name)} (removed)", "removed")
            self.combo.setCurrentIndex(self.combo.count() - 1)
        elif cur is not None:
            self.combo.setCurrentIndex(1 + layers.index(cur))
        else:
            self.combo.setCurrentIndex(0)
        self.combo.blockSignals(False)
        removed_now = self._layer is not None and cur is None
        if removed_now and not self._removed:
            self._removed = True
            self.changed.emit()
        self._removed = removed_now
        self.refresh()

    def _on_activated(self, index):
        data = self.combo.itemData(index)
        if data == "removed":
            return
        layer = None if data is None else data()
        if layer is self.layer and (layer is not None or self._layer is None):
            return
        self.select(layer)

    def _on_edited(self, layer):
        if layer is self.layer:
            self.refresh()
            self._debounce.start()

    def _on_moving(self, layer):
        if layer is self.layer:
            self.refresh(live=True)
            self.moving.emit()

    # ---- texts
    def refresh(self, live=False):   # live: the same texts (the box is read from the layer anyway)
        """Info lines + red line + the run button text."""
        st = self.state()
        layer = self.layer
        roi = None
        if layer is not None:
            roi = self.hub.live_roi(layer)
        if roi is not None:
            extra = "  (z: all)" if layer_meta(layer).get("z_all") and roi.ndim == 3 else ""
            self.info.setText(f"{range_text(roi)}{extra}\n{size_line(roi)}")
            self.info.setToolTip(roi.describe())
        else:
            self.info.setText("")
        self.info.setVisible(roi is not None)
        err = ""
        if st in ("removed", "mismatch", "invalid"):
            try:
                self.roi_for_run()
            except RoiError as e:
                err = str(e)
        self.error.setText(err)
        self.error.setVisible(bool(err))
        if self._run is not None:
            btn, text = self._run[0](), self._run[1]
            if btn is not None:
                btn.setText(f"{text} in {self._layer_name}" if self._layer is not None and st != "removed" else text)
        self.refreshed.emit()


class RoiReminder(QLabel):
    """'<text> ROI_1: 201 × 71 × 72' by a button, shown while its tab uses a usable ROI."""

    def __init__(self, select, text="Runs inside", shape=None, parent=None):
        super().__init__(parent)
        self.select = select
        self.text_ = text
        self.shape = shape
        self.setWordWrap(True)
        self.setStyleSheet(ROI_STYLE)
        select.refreshed.connect(self.refresh)
        self.refresh()

    def refresh(self):
        shape = self.shape() if self.shape is not None else None
        roi = self.select.roi_for_display(shape)
        if roi is None:
            self.setVisible(False)
            return
        self.setText(f"{self.text_} {roi.name}: {TIMES.join(str(v) for v in roi.sizes)}")
        self.setToolTip(roi.describe())
        self.setVisible(True)


# --------------------------------------------------------------------------
class _LayerCombo(QComboBox):
    """Layer names, middle-elided; the full name is the item data and tooltip."""

    def set_names(self, names):
        cur = self.name()
        self.blockSignals(True)
        self.clear()
        for n in names:
            self.addItem(elide(n), n)
            self.setItemData(self.count() - 1, n, Qt.ToolTipRole)
        if cur in names:
            self.setCurrentIndex(names.index(cur))
        self.blockSignals(False)

    def name(self):
        return self.currentData()


class RoiTools(QWidget):
    """The ROI sub-tab: create, edit, delete, save and load ROI layers."""

    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent)
        self.viewer = napari_viewer
        self.hub = get_roi_hub(napari_viewer)
        self._edit = None                   # weakref: ROI layer being edited
        self._job = 0
        self._workers = set()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self.refresh_layers)

        outer = QVBoxLayout(self)
        outer.setSpacing(8)
        self._form_labels = []

        # Draw a box
        box = self.draw_box = QGroupBox("Draw a box")
        g = self._grid(box)
        g.addWidget(self._label("On image"), 0, 0)
        self.image_combo = _LayerCombo()
        self.image_combo.setToolTip("The image whose voxel grid the box uses")
        g.addWidget(self.image_combo, 0, 1)
        self.draw_btn = QPushButton("Draw box")
        self.draw_btn.setToolTip("Drag a rectangle on the 2D view (y / x); z = all slices until you set it below")
        g.addWidget(self.draw_btn, 1, 0, 1, 2)
        self.whole_btn = QPushButton("Full image (then set z)")
        self.whole_btn.setToolTip("An ROI with the full y / x extent and all slices. Use it to limit only the z range:\n"
                                  "set z from / to below, e.g. process slices 200–400 with full x / y.")
        g.addWidget(self.whole_btn, 2, 0, 1, 2)
        g.addWidget(self._hint("Drag a rectangle on the 2D view; the z range starts as all slices."), 3, 0, 1, 2)
        outer.addWidget(box)

        # From labels
        box = self.labels_box = QGroupBox("From labels")
        g = self._grid(box)
        g.addWidget(self._label("Labels"), 0, 0)
        self.labels_combo = _LayerCombo()
        self.labels_combo.setToolTip("A labels layer: the box goes around the label ids below")
        g.addWidget(self.labels_combo, 0, 1)
        g.addWidget(self._label("Ids"), 1, 0)
        self.ids_edit = QLineEdit()
        self.ids_edit.setPlaceholderText("e.g. 3,5-7 (empty: all)")
        self.ids_edit.setToolTip("Label ids, e.g. 3,5-7; empty: all non-zero labels")
        g.addWidget(self.ids_edit, 1, 1)
        g.addWidget(self._label("Margin"), 2, 0)
        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(0, 100000)
        self.margin_spin.setToolTip("Voxels added on every side of the labels' bounding box")
        row = QHBoxLayout()
        row.addWidget(self.margin_spin)
        row.addWidget(QLabel("voxels"))
        row.addStretch()
        g.addLayout(row, 2, 1)
        self.create_btn = QPushButton("Create ROI")
        self.create_btn.setToolTip("A box around these label ids (+ margin), computed once")
        g.addWidget(self.create_btn, 3, 0, 1, 2)
        self.create_error = QLabel("")
        self.create_error.setStyleSheet(ERROR_STYLE)
        self.create_error.setWordWrap(True)
        g.addWidget(self.create_error, 4, 0, 1, 2)
        self.create_error.setVisible(False)
        outer.addWidget(box)

        # Edit ROI
        box = self.edit_box = QGroupBox("Edit ROI")
        v = QVBoxLayout(box)
        v.setSpacing(4)
        self.empty_note = self._hint("No ROI yet: draw a box or create one from labels above")
        v.addWidget(self.empty_note)
        g = QGridLayout()
        g.setVerticalSpacing(4)
        g.addWidget(self._label("ROI"), 0, 0)
        self.roi_combo = QComboBox()
        self.roi_combo.setToolTip(RENAME_TIP)
        self.roi_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.roi_combo.setMinimumContentsLength(8)
        g.addWidget(self.roi_combo, 0, 1, 1, 3)
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setToolTip("Remove this ROI layer")
        g.addWidget(self.delete_btn, 0, 4)
        self.from_to = (QLabel("from"), QLabel("to"))
        g.addWidget(self.from_to[0], 1, 1)
        g.addWidget(self.from_to[1], 1, 3)
        self.axis_rows = {}
        for i, ax in enumerate(("z", "y", "x"), 2):
            lab = self._label(ax)
            lo, hi = QSpinBox(), QSpinBox()
            for sp in (lo, hi):
                sp.setToolTip("Voxel index, inclusive")
                sp.setKeyboardTracking(False)
                sp.editingFinished.connect(self._commit_spins)
            g.addWidget(lab, i, 0)
            g.addWidget(lo, i, 1)
            g.addWidget(hi, i, 3)
            cur = []
            if ax == "z":
                for col, sp in ((2, lo), (4, hi)):
                    b = QPushButton("= slice")
                    b.setToolTip("Set to the z slice shown in the viewer")
                    b.clicked.connect(lambda _=False, s=sp: self._set_slice(s))
                    g.addWidget(b, i, col)
                    cur.append(b)
            self.axis_rows[ax] = (lab, lo, hi, cur)
        for col in (1, 3):
            g.setColumnStretch(col, 1)
        v.addLayout(g)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        v.addWidget(self.info)
        self.users_line = QLabel("")
        self.users_line.setWordWrap(True)
        v.addWidget(self.users_line)
        self.z_note = QLabel("z: all — set the z range with from / to or = slice")
        self.z_note.setStyleSheet(ROI_STYLE)
        self.z_note.setWordWrap(True)
        v.addWidget(self.z_note)
        self.view_note = QLabel(HIDDEN_NOTE)
        self.view_note.setStyleSheet(ROI_STYLE)
        self.view_note.setWordWrap(True)
        v.addWidget(self.view_note)
        outer.addWidget(box)

        # Save / Load
        box = self.file_box = QGroupBox("Save / Load")
        v = QVBoxLayout(box)
        v.setSpacing(4)
        row = QHBoxLayout()
        row.setSpacing(12)
        self.save_btn = QPushButton("Save ROIs…")
        self.save_btn.setToolTip("All ROI layers to one JSON file")
        self.load_btn = QPushButton("Load ROIs…")
        self.load_btn.setToolTip("ROIs from a JSON file saved here, or from a run's roi.json")
        row.addWidget(self.save_btn)
        row.addWidget(self.load_btn)
        row.addStretch()
        v.addLayout(row)
        v.addWidget(self._hint("Load also accepts a run's roi.json"))
        self.file_note = QLabel("")
        self.file_note.setWordWrap(True)
        self.file_note.setVisible(False)
        v.addWidget(self.file_note)
        outer.addWidget(box)
        outer.addStretch()
        self._align_labels()

        self.draw_btn.clicked.connect(self._draw)
        self.whole_btn.clicked.connect(self._whole)
        self.create_btn.clicked.connect(self._from_labels)
        self.roi_combo.activated.connect(lambda i: self._pick(self.roi_combo.itemData(i)))
        self.delete_btn.clicked.connect(self._delete)
        self.save_btn.clicked.connect(lambda: self.save_rois())
        self.load_btn.clicked.connect(lambda: self.load_rois())
        # only the hub's Qt signals: nothing stays connected in napari when the panel closes
        self.hub.rois_changed.connect(self._on_rois_changed)
        self.hub.roi_edited.connect(self._on_edited)
        self.hub.roi_moving.connect(self._on_moving)
        self.hub.users_changed.connect(self._show)
        self.hub.layer_inserted.connect(self._on_layers_changed)
        self.hub.layer_removed.connect(self._on_layers_changed)
        self.hub.dims_changed.connect(self._on_dims)
        self.hub.draw_finished.connect(self._on_drawn)
        # a closed panel takes its unfinished drawing (the "SPROUT draw" helper layer) along
        self.destroyed.connect(self.hub.cancel_draw)
        self.refresh_layers()

    # ---- helpers
    def _align_labels(self):
        """One label column width for every block (form-style alignment)."""
        for lab in self._form_labels:
            lab.setMinimumWidth(0)
            lab.setMaximumWidth(16777215)
        w = max(lab.sizeHint().width() for lab in self._form_labels)
        for lab in self._form_labels:
            lab.setFixedWidth(w)

    def showEvent(self, event):
        self._align_labels()          # the font may have changed with the napari stylesheet
        super().showEvent(event)

    def _label(self, text):
        lab = QLabel(text)
        self._form_labels.append(lab)
        return lab

    @staticmethod
    def _hint(text):
        lab = QLabel(text)
        lab.setStyleSheet(OK_STYLE)
        lab.setWordWrap(True)
        return lab

    @staticmethod
    def _grid(box):
        g = QGridLayout(box)
        g.setVerticalSpacing(4)
        g.setColumnStretch(1, 1)
        return g

    def _file_msg(self, text, style=OK_STYLE):
        self.file_note.setText(text)
        self.file_note.setStyleSheet(style)
        self.file_note.setVisible(bool(text))

    def _create_err(self, text):
        self.create_error.setText(text)
        self.create_error.setVisible(bool(text))

    def _on_layers_changed(self, *_):
        self._refresh_timer.start(0)

    @property
    def edit_layer(self):
        lay = None if self._edit is None else self._edit()
        return lay if lay is not None and lay in self.viewer.layers else None

    def _ref(self):
        name = self.image_combo.name()
        lay = self.viewer.layers[name] if name and name in self.viewer.layers else None
        return lay if isinstance(lay, Image) else None

    def refresh_layers(self):
        imgs = [l.name for l in self.viewer.layers
                if isinstance(l, Image) and not l.rgb and not l.multiscale and l.ndim in (2, 3)]
        active = self.viewer.layers.selection.active
        self.image_combo.set_names(imgs)
        if self.image_combo.name() is None and imgs:
            self.image_combo.setCurrentIndex(imgs.index(active.name) if isinstance(active, Image) and
                                             active.name in imgs else 0)
        labs = [l.name for l in self.viewer.layers if isinstance(l, Labels) and not l.name.startswith(_PREVIEW_LAYERS)]
        self.labels_combo.set_names(labs)
        self._on_rois_changed()

    def _on_rois_changed(self):
        layers = self.hub.layers()
        cur = self.edit_layer
        if cur is None and layers:
            cur = layers[-1]
            self._edit = weakref.ref(cur)
        self.roi_combo.blockSignals(True)
        self.roi_combo.clear()
        for l in layers:
            self.roi_combo.addItem(elide(l.name), weakref.ref(l))
            self.roi_combo.setItemData(self.roi_combo.count() - 1, l.name, Qt.ToolTipRole)
        if cur is not None:
            self.roi_combo.setCurrentIndex(layers.index(cur))
        self.roi_combo.blockSignals(False)
        self._show()

    def _pick(self, ref):
        layer = None if ref is None else ref()
        self._edit = None if layer is None else weakref.ref(layer)
        self._show()

    # ---- create
    def _draw(self):
        ref = self._ref()
        if ref is None:
            show_info("Load an image first: the box is drawn in its voxel grid")
            return
        if not self.hub.view_ok():
            show_info("Switch to the 2D view (z slices) to draw")
            return
        # a temporary helper layer: the ROI layer is made from the first rectangle drawn
        self.hub.start_draw(ref)

    def _on_drawn(self, layer):
        self._edit = weakref.ref(layer)
        self._on_rois_changed()

    def _whole(self):
        ref = self._ref()
        if ref is None:
            show_info("Load an image first")
            return
        shape = np.shape(ref.data)
        layer = add_roi_layer(self.viewer, ref, (0,) * len(shape), shape)
        self._edit = weakref.ref(layer)
        self._on_rois_changed()

    def _from_labels(self):
        name = self.labels_combo.name()
        lay = self.viewer.layers[name] if name and name in self.viewer.layers else None
        self._create_err("")
        if not isinstance(lay, Labels):
            self._create_err("Choose a Labels layer")
            return
        text = self.ids_edit.text().strip()
        try:
            ids = sorted(parse_ids(text)) if text else None
        except ValueError:
            self._create_err(f"ids: enter numbers like 3,5-7 (got '{text}')")
            return
        if ids is not None and not ids:
            self._create_err("ids: enter at least one id, e.g. 3,5-7")
            return
        margin = self.margin_spin.value()
        self._job += 1
        job, data = self._job, lay.data
        self.create_btn.setEnabled(False)
        self.create_btn.setText("Creating…")

        def work():
            roi = R.Roi.from_mask(np.asarray(data), ids=ids, name=name)
            if roi.empty:
                raise RoiError(roi._empty_reason())
            return roi.as_box(margin)

        def done(box):
            if job == self._job and lay in self.viewer.layers:
                layer = add_roi_layer(self.viewer, lay, box.lo, box.hi)
                self._edit = weakref.ref(layer)
                self._on_rois_changed()

        def failed(e):
            if job == self._job:
                self._create_err(str(e) if isinstance(e, RoiError) else f"{type(e).__name__}: {e}")

        def finished():
            self.create_btn.setEnabled(True)
            self.create_btn.setText("Create ROI")

        w = thread_worker(work, ignore_errors=True)()
        w.returned.connect(done)
        w.errored.connect(failed)
        w.finished.connect(finished)
        self._workers.add(w)
        w.finished.connect(lambda: self._workers.discard(w))
        w.start()

    # ---- edit
    def _on_edited(self, layer):
        if layer is self.edit_layer:
            self._show()

    def _on_moving(self, layer):
        if layer is self.edit_layer:
            self._show(live=True)

    def _delete(self):
        layer = self.edit_layer
        if layer is not None:
            self._edit = None
            self.viewer.layers.remove(layer)

    def _commit_spins(self):
        layer = self.edit_layer
        if layer is None:
            return
        md = layer_meta(layer)
        shape = md["shape"]
        vals = {ax: (r[1].value(), r[2].value()) for ax, r in self.axis_rows.items()}
        y0, y1 = sorted(vals["y"])
        x0, x1 = sorted(vals["x"])
        roi = self.hub.live_roi(layer)
        if roi is None or (roi.lo[-2:], roi.hi[-2:]) != ((y0, x0), (y1 + 1, x1 + 1)):
            self.hub.set_yx(layer, (y0, x0), (y1 + 1, x1 + 1))
        if len(shape) == 3:
            z0, z1 = sorted(vals["z"])
            if (z0, z1 + 1) != tuple(md.get("z", (0, shape[0]))) or md.get("z_all"):
                if roi is None or (z0, z1 + 1) != (roi.lo[0], roi.hi[0]) or md.get("z_all"):
                    self.hub.set_z_range(layer, z0, z1 + 1)
        self._show()

    def _set_slice(self, spin):
        layer = self.edit_layer
        if layer is None:
            return
        z = self.hub._z_index(layer)
        if z is not None:
            spin.setValue(max(0, min(z, spin.maximum())))
            self._commit_spins()

    # ---- display
    def _on_dims(self):
        self.view_note.setVisible(self.hub.view_hidden)
        has = self.edit_layer is not None and len(layer_meta(self.edit_layer).get("shape", ())) == 3
        for b in self.axis_rows["z"][3]:
            b.setEnabled(has and self.viewer.dims.ndisplay == 2)

    def _show(self, live=False):
        layer = self.edit_layer
        none = layer is None
        self.edit_box.setEnabled(not none)
        self.empty_note.setVisible(none)
        for w in (*self.from_to, self.info, self.users_line):
            w.setVisible(not none)
        self.view_note.setVisible(self.hub.view_hidden)
        self.roi_combo.setToolTip(RENAME_TIP if none else f"{layer.name}\n{RENAME_TIP}")
        if none:
            self.info.setText("")
            self.users_line.setText("")
            self.z_note.setVisible(False)
            for r in self.axis_rows.values():
                for w in (r[0], r[1], r[2], *r[3]):
                    w.setVisible(False)
            return
        md = layer_meta(layer)
        shape = md["shape"]
        roi = self.hub.live_roi(layer)
        axes = ("z", "y", "x")[-len(shape):]
        for ax, (lab, lo, hi, cur) in self.axis_rows.items():
            vis = ax in axes
            for w in (lab, lo, hi, *cur):
                w.setVisible(vis)
            if not vis:
                continue
            i = axes.index(ax)
            n = int(shape[i])
            for sp, val in ((lo, roi.lo[i] if roi else 0), (hi, roi.hi[i] - 1 if roi else n - 1)):
                if sp.hasFocus():
                    continue
                sp.blockSignals(True)
                sp.setRange(0, max(0, n - 1))
                sp.setValue(int(val))
                sp.blockSignals(False)
        self.z_note.setVisible(bool(md.get("z_all")) and len(shape) == 3)
        users = self.hub.used_by(layer)
        self.users_line.setText(f"Used by: {', '.join(users) if users else 'no tab'}")
        if roi is None:
            self.info.setText(f"{layer.name}: no box yet — drag a rectangle on the 2D view")
            self.info.setStyleSheet(ROI_STYLE)
            return
        self.info.setText(f"{TIMES.join(str(v) for v in roi.sizes)} · {voxels_text(roi.n_voxels)} · "
                          f"{format_percent(roi.fraction)}")
        self.info.setToolTip(roi.describe())
        self.info.setStyleSheet("")
        self._on_dims()

    # ---- files
    def save_rois(self, path=None):
        layers = [l for l in self.hub.layers() if self.hub.live_roi(l) is not None]
        if not layers:
            self._file_msg("No ROI to save", ERROR_STYLE)
            return None
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Save ROIs", "rois.json", "ROI JSON (*.json)")
            if not path:
                return None
        R.save_rois(path, [(self.hub.roi_of(l), layer_extra(l)) for l in layers])
        self._file_msg(f"Saved {len(layers)} ROI{'s' if len(layers) != 1 else ''} to {os.path.basename(path)}")
        return path

    def load_rois(self, path=None):
        """Add the ROIs of a file (ours, or a run's roi.json) as ROI layers."""
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Load ROIs", "", "ROI JSON (*.json)")
            if not path:
                return []
        try:
            entries = R.load_rois(path)
        except (OSError, ValueError) as e:
            self._file_msg(f"Cannot load {os.path.basename(path)}: {e}", ERROR_STYLE)
            return []
        made = []
        for roi, extra in entries:
            ref = self._ref_for(roi, extra)
            name = roi.name if roi.name and roi.name not in self.viewer.layers else None
            if ref is not None:
                layer = add_roi_layer(self.viewer, ref, roi.lo, roi.hi, name=name)
            else:
                layer = add_roi_layer(self.viewer, None, roi.lo, roi.hi, name=name, shape=roi.shape,
                                      scale=extra.get("scale"), translate=extra.get("translate"),
                                      ref_name=extra.get("ref_image"))
            made.append(layer)
        if made:
            self._edit = weakref.ref(made[-1])
            self._on_rois_changed()
        self._file_msg(f"Loaded {len(made)} ROI{'s' if len(made) != 1 else ''} from {os.path.basename(path)}")
        return made

    def _ref_for(self, roi, extra):
        """The image the ROI belongs to: by name, else an Image of the same shape (None: use the file)."""
        name = extra.get("ref_image")
        lay = self.viewer.layers[name] if name and name in self.viewer.layers else None
        if lay is not None and tuple(np.shape(lay.data)) == roi.shape:
            return lay
        if extra.get("scale") is not None:
            return None
        return next((l for l in self.viewer.layers
                     if isinstance(l, (Image, Labels)) and tuple(np.shape(l.data)) == roi.shape), None)
