"""Prompt (AI) tab for napari-sprout: seeds -> prompts -> interactive model.

Paper Fig. 1d: instead of threshold-guided growing, the seeds are turned into
prompts (points, scribbles, an initial mask) for a promptable model
(nnInteractive now). Workflow, top to bottom:

    1. Input     image + seeds ("Use current ROI": sampling and the run stay inside the ROI)
    2. Classes   which seed classes to work on (checked) and the current class
    3. Prompts   input type (only what the model supports), Generate points for
                 the checked classes into one Points layer "{seeds}_prompts"
                 (edit them there), drawn scribble layers, CSV import / export
    4. Model     folder, device, Load (cached), Release GPU
    5. Run       Run checked / Update current class / Revert class / Stop
    6. Output    one full-size Labels layer "{seeds}_nni" ("_roi" with an ROI)

The model runs in the background, one class after the other; each finished
class is pasted into the output layer right away. Heavy imports (torch,
nnInteractive) happen only when the model is loaded.
"""
import functools
import threading
import time
import weakref

import numpy as np
import pandas as pd
from napari.layers import Image, Labels, Points
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_error, show_info, show_warning
from qtpy.QtCore import QObject, QSettings, Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QProgressBar, QPushButton, QRadioButton, QScrollArea,
    QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from sprout_core import prompt_points as PP
from sprout_core import prompt_seg as PS
from sprout_core import roi as R
from sprout_core.boundary import Cancelled
from sprout_core.nni_engine import NniEngine, nninteractive_status
from sprout_core.prompt_engine import is_oom
from sprout_core.roi import RoiError, parse_ids  # noqa: F401  (parse_ids: also importable from here)

from ..utils.roi_model import get_roi_model
from ..utils.roi_widget import RoiReminder, RoiUse

META_KEY = "sprout_prompt"
SETTINGS = ("SPROUT", "napari-sprout")
POS_COLOR, NEG_COLOR = "lime", "magenta"
TOOLTIP_ROLE = 3            # Qt.ToolTipRole

# (key, prompt_type, scribble source, label, hint shown under the combo)
INPUT_TYPES = [
    ("points", "points", None, "Points", ""),
    ("points+init", "points+init", None, "Points + seed as initial mask",
     "The seed of each class is the initial mask. Usually combine with points: the mask "
     "alone under-segments badly."),
    ("seed_scribble", "scribble", "seed", "Seed as scribble (+points)",
     "The seed of each class is a scribble. Usually combine with points: the scribble "
     "alone under-segments badly."),
    ("drawn_scribble", "scribble", "drawn", "Drawn scribbles (+points)",
     "Paint the class id into '{seeds}_scribbles' (and, optionally, 'not this class' into "
     "'{seeds}_scribbles_neg'). Points of the class are added after the scribbles."),
]
OVERLAP = [("seed", "Seed class wins, else smaller id"), ("min_id", "Smaller id wins (as the scripts)")]
NEG_SLOW = ("Negative points are slow to sample on big images (5-14 s on the 13 M-voxel dog demo; "
            "they draw from all background / other-class voxels).")
COLUMNS = ["", "id", "seed voxels", "points + / -", "status", "result voxels"]


class _Progress(QObject):
    """Created on the UI thread; emits from worker threads are queued to it."""
    changed = Signal(int, int)


def _on_destroyed(life, *_):
    """The tab's Qt object is gone: stop its jobs and drop its napari layer callbacks.

    Module-level (holds no reference to the tab) and connected to `destroyed`.
    """
    life["alive"] = False
    for ev in list(life["cancel"]):
        ev.set()
    for emitter, cb in list(life["conns"]):
        try:
            emitter.disconnect(cb)
        except Exception:
            pass
    life["conns"].clear()


def _shutdown_engine(engine):
    try:
        engine.close() if hasattr(engine, "close") else engine.release()
    except Exception:
        pass


# parse_ids now lives in sprout_core.roi (shared with the ROI Manager); imported above.


class PromptTab(QWidget):
    """The 'Prompt (AI)' tab. engine_factory / status_fn are injectable for tests."""

    output_updated = Signal(str)          # output layer name

    def __init__(self, napari_viewer, engine_factory=NniEngine, status_fn=nninteractive_status):
        super().__init__()
        self.viewer = napari_viewer
        self.engine = engine_factory()
        self._finalizer = weakref.finalize(self, _shutdown_engine, self.engine)
        # lifecycle: worker callbacks check life["alive"]; napari layer callbacks are listed
        # in life["conns"] and disconnected when the Qt object is destroyed
        self._life = {"alive": True, "cancel": [], "conns": []}
        self.destroyed.connect(functools.partial(_on_destroyed, self._life))
        try:
            self._status = status_fn()
        except Exception as e:                       # never break the panel
            self._status = (False, f"cannot check nnInteractive: {e}")
        self.available = bool(self._status[0])
        self._roi_model = get_roi_model(napari_viewer)

        self._img_layer = None
        self._seeds_layer = None
        self._stats = {}                 # class id -> seed voxels (inside the ROI)
        self._extra = set()              # classes added with "+ New class"
        self._checked = {}               # class id -> bool
        self._class_status = {}          # class id -> text
        self._stores = {}                # (image, seeds, output name) -> ResultStore
        self._ctx = None                 # context of the last run (output layer, rule, seeds, ...)
        self._job = {"stats": 0, "gen": 0, "load": 0, "run": 0}
        self._busy = {"gen": False, "load": False, "run": False}
        self._cancel = None
        self._workers = set()
        self._current = None             # current class id
        self._points_conn = None         # Points layer whose events we listen to

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self.refresh_layers)
        self._points_timer = QTimer(self)
        self._points_timer.setSingleShot(True)
        self._points_timer.setInterval(100)
        self._points_timer.timeout.connect(self._on_points_settled)
        self._defaults_timer = QTimer(self)
        self._defaults_timer.setSingleShot(True)
        self._defaults_timer.timeout.connect(self._apply_feature_defaults)
        # ROI changes are handled on the next event-loop turn (nothing starts while
        # napari is closing: the timer dies with the widget)
        self._roi_timer = QTimer(self)
        self._roi_timer.setSingleShot(True)
        self._roi_timer.timeout.connect(self._on_roi_changed)
        self._last_roi = None            # display ROI the class counts were made for

        self._build_ui()
        self._connect()
        self.refresh_layers()

    # ================================================================== UI
    def _build_ui(self):
        layout = QVBoxLayout()

        self.banner = QLabel(self._status[1] if not self.available else "")
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet("QLabel { background: #5a3d00; color: #ffd27f; padding: 6px; }")
        self.banner.setVisible(not self.available)
        layout.addWidget(self.banner)

        # 1. Input
        g1 = QGroupBox("1. Input")
        f1 = QFormLayout(g1)
        self.image_combo = QComboBox()
        self.seeds_combo = QComboBox()
        f1.addRow("Image:", self.image_combo)
        f1.addRow("Seeds:", self.seeds_combo)
        self.roi_use = RoiUse(self.viewer, self._shape,
                              "Sample points and run the model only inside the ROI")
        f1.addRow(self.roi_use)
        layout.addWidget(g1)

        # 2. Classes
        g2 = QGroupBox("2. Classes")
        l2 = QVBoxLayout(g2)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(140)
        l2.addWidget(self.table)
        row = QHBoxLayout()
        self.all_btn = QPushButton("All")
        self.none_btn = QPushButton("None")
        self.min_spin = QSpinBox()
        self.min_spin.setRange(0, 2_000_000_000)
        self.min_spin.setValue(100)
        self.min_btn = QPushButton("Check >= N voxels")
        self.ids_edit = QLineEdit()
        self.ids_edit.setPlaceholderText("ids, e.g. 3,5-9")
        self.ids_btn = QPushButton("Check ids")
        self.new_btn = QPushButton("+ New class")
        self.new_btn.setToolTip("A class without a seed: place its points by hand.")
        for w in (self.all_btn, self.none_btn, self.min_spin, self.min_btn, self.ids_edit, self.ids_btn,
                  self.new_btn):
            row.addWidget(w)
        l2.addLayout(row)
        self.classes_label = QLabel("")
        self.classes_label.setStyleSheet("color: gray")
        l2.addWidget(self.classes_label)
        layout.addWidget(g2)

        # 3. Prompts
        g3 = QGroupBox("3. Prompts")
        l3 = QVBoxLayout(g3)
        f3 = QFormLayout()
        self.type_combo = QComboBox()
        for key, ptype, src, label, hint in INPUT_TYPES:
            self.type_combo.addItem(label, key)
        f3.addRow("Input type:", self.type_combo)
        self.type_hint = QLabel("")
        self.type_hint.setWordWrap(True)
        self.type_hint.setStyleSheet("color: #d08000")
        f3.addRow(self.type_hint)
        self.neg_seed_cb = QCheckBox("Other classes as negative scribble")
        f3.addRow(self.neg_seed_cb)
        row = QHBoxLayout()
        self.scrib_btn = QPushButton("+ Scribble layer")
        self.scrib_btn.setToolTip("Labels layer '{seeds}_scribbles': paint the class id (positive scribble).")
        self.scrib_neg_btn = QPushButton("+ Negative scribble layer")
        self.scrib_neg_btn.setToolTip("Labels layer '{seeds}_scribbles_neg': paint the class id = "
                                      "'not this class'. Optional.")
        row.addWidget(self.scrib_btn)
        row.addWidget(self.scrib_neg_btn)
        row.addStretch()
        self.scrib_row = QWidget()
        self.scrib_row.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        f3.addRow(self.scrib_row)
        l3.addLayout(f3)

        gs = QGroupBox("Sample points from the seeds")
        fs = QFormLayout(gs)
        self.npos_spin = QSpinBox()
        self.npos_spin.setRange(1, 1000)
        self.npos_spin.setValue(1)
        self.method_combo = QComboBox()
        for m in ("random", "grid", "center_edge", "kmeans"):
            self.method_combo.addItem(m)
        self.method_combo.setItemData(3, "Needs scikit-learn.", TOOLTIP_ROLE)
        row = QHBoxLayout()
        row.addWidget(self.npos_spin)
        row.addWidget(QLabel("method:"))
        row.addWidget(self.method_combo)
        row.addStretch()
        fs.addRow("Positive per class:", row)
        self.nneg_spin = QSpinBox()
        self.nneg_spin.setRange(0, 1000)
        self.nneg_spin.setValue(0)
        self.neg_bg_cb = QCheckBox("background")
        self.neg_bg_cb.setChecked(True)
        self.neg_other_cb = QCheckBox("other classes")
        self.neg_other_cb.setChecked(True)
        self.neg_each_cb = QCheckBox("n from each other class")
        row = QHBoxLayout()
        row.addWidget(self.nneg_spin)
        row.addWidget(QLabel("from:"))
        row.addWidget(self.neg_bg_cb)
        row.addWidget(self.neg_other_cb)
        row.addWidget(self.neg_each_cb)
        row.addStretch()
        fs.addRow("Negative per class:", row)
        self.neg_slow_label = QLabel(NEG_SLOW)
        self.neg_slow_label.setWordWrap(True)
        self.neg_slow_label.setStyleSheet("color: #d08000")
        fs.addRow(self.neg_slow_label)
        self.zero_cb = QCheckBox("Exclude zero-intensity voxels from negatives")
        self.zero_cb.setChecked(True)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2 ** 31 - 1)
        self.seed_spin.setValue(0)
        self.seed_spin.setToolTip("Random seed: the same seed gives the same points; also seeds the model "
                                  "per class (seed + class id).")
        row = QHBoxLayout()
        row.addWidget(self.zero_cb)
        row.addStretch()
        row.addWidget(QLabel("Random seed:"))
        row.addWidget(self.seed_spin)
        fs.addRow(row)
        self.gen_reminder = RoiReminder(self.roi_use, "Samples inside ROI")
        fs.addRow(self.gen_reminder)
        row = QHBoxLayout()
        self.gen_btn = QPushButton("Generate for checked classes")
        self.gen_btn.setToolTip("Replaces the generated points of the checked classes; points you added "
                                "by hand are kept.")
        row.addWidget(self.gen_btn, 1)
        fs.addRow(row)
        l3.addWidget(gs)

        ge = QGroupBox("Edit points")
        fe = QFormLayout(ge)
        self.pos_rb = QRadioButton("positive")
        self.neg_rb = QRadioButton("negative")
        self.pos_rb.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.pos_rb)
        grp.addButton(self.neg_rb)
        self.edit_btn = QPushButton("Add points")
        self.edit_btn.setToolTip("Selects the prompts layer in 'add' mode; new points belong to the "
                                 "current class.")
        self.only_cb = QCheckBox("Show only the current class")
        self.only_cb.setChecked(False)
        row = QHBoxLayout()
        row.addWidget(QLabel("New points:"))
        row.addWidget(self.pos_rb)
        row.addWidget(self.neg_rb)
        row.addWidget(self.edit_btn)
        row.addStretch()
        fe.addRow(row)
        fe.addRow(self.only_cb)
        row = QHBoxLayout()
        self.import_btn = QPushButton("Import CSV")
        self.export_btn = QPushButton("Export CSV")
        self.clear_pts_btn = QPushButton("Clear points of checked")
        row.addWidget(self.import_btn)
        row.addWidget(self.export_btn)
        row.addWidget(self.clear_pts_btn)
        row.addStretch()
        fe.addRow(row)
        l3.addWidget(ge)
        self.prompt_status = QLabel("")
        self.prompt_status.setStyleSheet("color: gray")
        self.prompt_status.setWordWrap(True)
        l3.addWidget(self.prompt_status)
        layout.addWidget(g3)

        # 4. Model
        g4 = QGroupBox(f"4. Model ({self.engine.name})")
        f4 = QFormLayout(g4)
        self.model_edit = QLineEdit(str(QSettings(*SETTINGS).value("prompt/model_dir", "") or ""))
        self.model_edit.setPlaceholderText("folder of the nnInteractive weights (e.g. nnInteractive_v1.0)")
        self.browse_btn = QPushButton("...")
        row = QHBoxLayout()
        row.addWidget(self.model_edit, 1)
        row.addWidget(self.browse_btn)
        f4.addRow("Model folder:", row)
        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto (GPU if available)", None)
        self.device_combo.addItem("GPU (cuda)", "cuda")
        self.device_combo.addItem("CPU (very slow)", "cpu")
        self.cpu_label = QLabel("")
        self.cpu_label.setStyleSheet("color: #d08000")
        row = QHBoxLayout()
        row.addWidget(self.device_combo)
        row.addWidget(self.cpu_label, 1)
        f4.addRow("Device:", row)
        row = QHBoxLayout()
        self.load_btn = QPushButton("Load model")
        self.release_btn = QPushButton("Release GPU")
        row.addWidget(self.load_btn, 1)
        row.addWidget(self.release_btn)
        f4.addRow(row)
        self.model_status = QLabel("not loaded" if self.available else "nnInteractive is not installed")
        self.model_status.setWordWrap(True)
        self.model_status.setStyleSheet("color: gray")
        f4.addRow(self.model_status)
        layout.addWidget(g4)

        # 5. Run
        g5 = QGroupBox("5. Run")
        f5 = QFormLayout(g5)
        self.overlap_combo = QComboBox()
        for key, label in OVERLAP:
            self.overlap_combo.addItem(label, key)
        f5.addRow("Overlaps:", self.overlap_combo)
        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(0, 1024)
        self.margin_spin.setValue(32)
        self.margin_spin.setToolTip("Image context around the ROI the model sees (voxels).")
        f5.addRow("ROI margin:", self.margin_spin)
        self.run_reminder = RoiReminder(self.roi_use, "Runs inside ROI")
        f5.addRow(self.run_reminder)
        row = QHBoxLayout()
        self.run_btn = QPushButton("Run checked")
        self.update_btn = QPushButton("Update current class")
        self.update_btn.setToolTip("Only added points: refines the previous result; otherwise re-runs it.")
        self.revert_btn = QPushButton("Revert class")
        self.stop_btn = QPushButton("Stop")
        for w in (self.run_btn, self.update_btn, self.revert_btn, self.stop_btn):
            row.addWidget(w)
        f5.addRow(row)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        f5.addRow(self.progress)
        self.run_status = QLabel("")
        self.run_status.setWordWrap(True)
        self.run_status.setStyleSheet("color: gray")
        f5.addRow(self.run_status)
        layout.addWidget(g5)

        # 6. Output
        g6 = QGroupBox("6. Output")
        f6 = QFormLayout(g6)
        self.output_label = QLabel("")
        self.output_label.setWordWrap(True)
        f6.addRow("Layer:", self.output_label)
        note = QLabel("Full image size; classes not run are 0.")
        note.setStyleSheet("color: gray")
        f6.addRow(note)
        layout.addWidget(g6)
        layout.addStretch()

        container = QWidget()
        container.setLayout(layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._apply_capabilities()
        self._on_type_changed()
        self._update_buttons()

    def _connect(self):
        # napari events reach the tab only through the RoiModel's Qt signals (bound
        # methods: Qt disconnects them when the tab is destroyed)
        self._roi_model.layer_inserted.connect(self._schedule_refresh)
        self._roi_model.layer_removed.connect(self._schedule_refresh)
        self.image_combo.currentTextChanged.connect(lambda *_: self._on_inputs_changed())
        self.seeds_combo.currentTextChanged.connect(lambda *_: self._on_inputs_changed())
        self.roi_use.changed.connect(self._roi_timer.start)

        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._on_selection)
        self.all_btn.clicked.connect(lambda: self._check(lambda c: True))
        self.none_btn.clicked.connect(lambda: self._check(lambda c: False))
        self.min_btn.clicked.connect(lambda: self._check(lambda c: self._stats.get(c, 0) >= self.min_spin.value()))
        self.ids_btn.clicked.connect(self._check_ids)
        self.new_btn.clicked.connect(self.new_class)

        self.type_combo.currentIndexChanged.connect(lambda *_: self._on_type_changed())
        self.scrib_btn.clicked.connect(lambda: self.create_scribble_layer(negative=False))
        self.scrib_neg_btn.clicked.connect(lambda: self.create_scribble_layer(negative=True))
        self.gen_btn.clicked.connect(self.generate)
        self.pos_rb.toggled.connect(lambda *_: self._apply_feature_defaults())
        self.edit_btn.clicked.connect(self.start_adding_points)
        self.only_cb.toggled.connect(lambda *_: self._apply_shown())
        self.import_btn.clicked.connect(self._import_dialog)
        self.export_btn.clicked.connect(self._export_dialog)
        self.clear_pts_btn.clicked.connect(self.clear_points_of_checked)

        self.browse_btn.clicked.connect(self._browse_model)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        self.load_btn.clicked.connect(self.load_model)
        self.release_btn.clicked.connect(self.release_model)

        self.overlap_combo.currentIndexChanged.connect(lambda *_: self._recompose())
        self.run_btn.clicked.connect(self.run_checked)
        self.update_btn.clicked.connect(self.update_current)
        self.revert_btn.clicked.connect(self.revert_current)
        self.stop_btn.clicked.connect(self.stop)

    # ------------------------------------------------------------- helpers
    def _alive(self):
        return self._life["alive"]

    def _guard(self, fn):
        """fn, called only while the tab exists (worker signals can outlive it)."""
        life = self._life

        def call(*a):
            if life["alive"]:
                return fn(*a)
        return call

    def _schedule_refresh(self, *_):
        self._refresh_timer.start(0)

    def _connect_layer_event(self, emitter, cb):
        emitter.connect(cb)
        self._life["conns"].append((emitter, cb))

    def _disconnect_layer_event(self, emitter, cb):
        try:
            emitter.disconnect(cb)
        except Exception:
            pass
        conns = self._life["conns"]
        conns[:] = [(e, c) for e, c in conns if not (e is emitter and c == cb)]

    def _run_bg(self, fn, on_return, on_error=None, on_finish=None):
        w = thread_worker(fn, ignore_errors=True)()
        w.returned.connect(self._guard(on_return))
        w.errored.connect(self._guard(on_error or self._report_error))
        if on_finish:
            w.finished.connect(self._guard(on_finish))
        self._workers.add(w)
        w.finished.connect(lambda: self._workers.discard(w))
        w.start()
        return w

    @staticmethod
    def _report_error(e):
        import traceback
        traceback.print_exception(type(e), e, e.__traceback__)
        show_error(f"Prompt (AI): {type(e).__name__}: {e}")

    def _layer(self, name, kind):
        if name and name in self.viewer.layers:
            lay = self.viewer.layers[name]
            if isinstance(lay, kind):
                return lay
        return None

    @property
    def image_layer(self):
        return self._img_layer if self._img_layer in self.viewer.layers else None

    @property
    def seeds_layer(self):
        return self._seeds_layer if self._seeds_layer in self.viewer.layers else None

    def _seeds_name(self):
        lay = self.seeds_layer
        return lay.name if lay is not None else ""

    def points_name(self):
        return f"{self._seeds_name()}_prompts"

    def scribble_name(self, negative=False):
        return f"{self._seeds_name()}_scribbles" + ("_neg" if negative else "")

    def points_layer(self):
        return self._layer(self.points_name(), Points) if self.seeds_layer is not None else None

    def _layer_kwargs(self):
        lay = self.image_layer
        if lay is None:
            return {}
        return {"scale": np.array(lay.scale), "translate": np.array(lay.translate)}

    def _shape(self):
        lay = self.image_layer
        return None if lay is None else tuple(np.shape(lay.data))

    def _display_roi(self):
        shape = self._shape()
        return None if shape is None else self.roi_use.roi_for_display(shape)

    def input_type(self):
        """(key, prompt_type, scribble source)."""
        key = self.type_combo.currentData()
        for k, ptype, src, _, _ in INPUT_TYPES:
            if k == key:
                return k, ptype, src
        return INPUT_TYPES[0][:3]

    def _update_buttons(self):
        busy = any(self._busy.values())
        have_inputs = self.image_layer is not None and self.seeds_layer is not None
        is3d = self._shape() is not None and len(self._shape()) == 3
        self.gen_btn.setEnabled(have_inputs and not self._busy["gen"])
        loaded = self.available and self.engine.loaded
        self.load_btn.setEnabled(self.available and not self._busy["load"] and not self._busy["run"])
        self.release_btn.setEnabled(loaded and not self._busy["run"] and not self._busy["load"])
        can_run = loaded and have_inputs and is3d and not busy
        self.run_btn.setEnabled(can_run)
        self.update_btn.setEnabled(can_run and self._current is not None)
        store = self._store_for_current_output(create=False)
        self.revert_btn.setEnabled(not self._busy["run"] and store is not None and self._current is not None
                                   and store.can_revert(self._current))
        self.stop_btn.setEnabled(self._busy["run"] or self._busy["gen"])
        if not self.available:
            tip = "nnInteractive is not installed in this Python environment"
            for b in (self.load_btn, self.run_btn, self.update_btn):
                b.setToolTip(tip)

    # --------------------------------------------------------- capabilities
    def _apply_capabilities(self):
        model = self.type_combo.model()
        caps = self.engine.capabilities
        for i, (key, ptype, src, label, hint) in enumerate(INPUT_TYPES):
            ok = self.engine.supports_prompt_type(ptype)
            item = model.item(i)
            item.setEnabled(ok)
            if ok:
                self.type_combo.setItemData(i, hint.replace("{seeds}", "seeds") or label, TOOLTIP_ROLE)
            else:
                missing = sorted(PS.required_capabilities(ptype) - caps)
                self.type_combo.setItemData(
                    i, f"{self.engine.name} does not support {', '.join(missing)} prompts", TOOLTIP_ROLE)
        if not model.item(self.type_combo.currentIndex()).isEnabled():
            self.type_combo.setCurrentIndex(0)

    def type_enabled(self, key):
        i = self.type_combo.findData(key)
        return i >= 0 and self.type_combo.model().item(i).isEnabled()

    def _on_type_changed(self):
        key, ptype, src = self.input_type()
        hint = [h for k, _, _, _, h in INPUT_TYPES if k == key][0]
        self.type_hint.setText(hint.replace("{seeds}", self._seeds_name() or "seeds"))
        self.type_hint.setVisible(bool(hint))
        self.neg_seed_cb.setVisible(src == "seed")
        self.scrib_row.setVisible(src == "drawn")

    # ================================================================ input
    def refresh_layers(self):
        img_cur, seeds_cur = self.image_combo.currentText(), self.seeds_combo.currentText()
        images = [l.name for l in self.viewer.layers if isinstance(l, Image)]
        derived = ("_nni", "_scribbles", "_scribbles_neg", "_prompts")
        labels = [l.name for l in self.viewer.layers if isinstance(l, Labels)]
        prefer = [n for n in labels if not any(n.endswith(s) or n.endswith(s + "_roi") for s in derived)
                  and META_KEY not in getattr(self.viewer.layers[n], "metadata", {})]
        for combo, names, cur, pref in ((self.image_combo, images, img_cur, images),
                                        (self.seeds_combo, labels, seeds_cur, prefer)):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            if cur in names:
                combo.setCurrentText(cur)
            elif pref:
                combo.setCurrentText(pref[0])
            combo.blockSignals(False)
        self._on_inputs_changed()

    def _on_inputs_changed(self):
        img = self._layer(self.image_combo.currentText(), Image)
        seeds = self._layer(self.seeds_combo.currentText(), Labels)
        if img is self._img_layer and seeds is self._seeds_layer:
            self._connect_points_layer()
            self._update_buttons()
            return
        self._img_layer, self._seeds_layer = img, seeds
        self._class_status = {}
        self._current = None
        if seeds is not None and img is not None and np.shape(seeds.data) != np.shape(img.data):
            self.classes_label.setText(f"Seeds {np.shape(seeds.data)} and image {np.shape(img.data)} "
                                       f"differ in shape")
            self._seeds_layer = None
        self._connect_points_layer()
        self._on_type_changed()
        self._update_output_label()
        self.roi_use.refresh()           # shows an ROI / image shape mismatch
        self.compute_stats()
        self._update_buttons()

    def _on_roi_changed(self):
        if not self._alive():
            return
        roi = self._display_roi()
        if roi is self._last_roi:
            return                       # e.g. None -> None: ROI off, a kind change, an unused mask painted
        self.compute_stats()
        self._update_output_label()
        self._update_buttons()

    # ============================================================== classes
    def compute_stats(self):
        """Seed voxels per class (inside the ROI if on), in the background."""
        self._job["stats"] += 1
        job = self._job["stats"]
        seeds = self.seeds_layer
        if seeds is None or self.image_layer is None:
            self._stats = {}
            self._rebuild_table()
            return
        data = np.asarray(seeds.data)
        roi = self._last_roi = self._display_roi()

        def work():
            if roi is None:
                st = PP.class_stats(data)
            else:
                st = PP.class_stats(roi.crop(data), allowed=roi.mask_crop())
            return dict(zip(st.class_id.tolist(), st.voxels.tolist()))

        def done(stats):
            if job != self._job["stats"]:
                return
            self._stats = stats
            self._rebuild_table()

        self.classes_label.setText("counting seed voxels...")
        self._run_bg(work, done)

    def class_ids(self):
        ids = set(self._stats) | set(self._extra)
        df = self.prompts_df()
        if len(df):
            ids |= set(int(c) for c in df.class_id.unique())
        store = self._store_for_current_output(create=False)
        if store is not None:
            ids |= set(store.classes())
        return sorted(ids)

    def _rebuild_table(self):
        ids = self.class_ids()
        df = self.prompts_df()
        counts = {}
        if len(df):
            for (c, lb), n in df.groupby(["class_id", "label"]).size().items():
                counts.setdefault(int(c), [0, 0])[0 if lb else 1] += int(n)
        store = self._store_for_current_output(create=False)
        self.table.blockSignals(True)
        sel = self._current
        self.table.setRowCount(len(ids))
        for r, c in enumerate(ids):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            chk.setCheckState(Qt.Checked if self._checked.setdefault(c, True) else Qt.Unchecked)
            self.table.setItem(r, 0, chk)
            pos, neg = counts.get(c, [0, 0])
            res = store.get(c) if store is not None else None
            vals = [str(c), f"{self._stats.get(c, 0):,}", f"{pos} / {neg}",
                    self._class_status.get(c, ""), "" if res is None else f"{res.n_voxels:,}"]
            for k, v in enumerate(vals, start=1):
                it = QTableWidgetItem(v)
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                if k in (1, 2, 5):
                    it.setTextAlignment(int(Qt.AlignRight | Qt.AlignVCenter))
                self.table.setItem(r, k, it)
        self.table.blockSignals(False)
        roi = self._display_roi()
        self.table.horizontalHeaderItem(2).setText("seed voxels in ROI" if roi is not None else "seed voxels")
        if sel in ids:
            self.select_class(sel)
        elif ids and self._current is None:
            self.select_class(ids[0])
        n_chk = sum(1 for c in ids if self._checked.get(c))
        self.classes_label.setText(f"{len(ids)} classes, {n_chk} checked" if ids else
                                   ("Select an image and seeds" if self.seeds_layer is None else "No classes"))
        self._update_buttons()

    def _row_of(self, c):
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 1)
            if it is not None and int(it.text()) == c:
                return r
        return -1

    def select_class(self, c):
        r = self._row_of(int(c))
        if r >= 0:
            self.table.selectRow(r)
            if self._current != int(c):         # selectRow does not emit when already selected
                self._set_current(int(c))

    def _on_selection(self):
        rows = {i.row() for i in self.table.selectedItems()}
        if not rows:
            return
        it = self.table.item(min(rows), 1)
        if it is not None:
            self._set_current(int(it.text()))

    def _set_current(self, c):
        self._current = c
        self._apply_feature_defaults()
        self._apply_shown()
        for neg in (False, True):
            lay = self._layer(self.scribble_name(neg), Labels)
            if lay is not None:
                lay.selected_label = c
        self._update_buttons()

    def _on_item_changed(self, item):
        if item.column() != 0:
            return
        idt = self.table.item(item.row(), 1)
        if idt is not None:
            self._checked[int(idt.text())] = item.checkState() == Qt.Checked
            self.classes_label.setText(f"{self.table.rowCount()} classes, "
                                       f"{len(self.checked_classes())} checked")

    def _check(self, pred):
        for c in self.class_ids():
            self._checked[c] = bool(pred(c))
        self._rebuild_table()

    def _check_ids(self):
        try:
            ids = parse_ids(self.ids_edit.text())
        except ValueError:
            show_error("Class ids: use numbers and ranges, e.g. 3,5-9")
            return
        self._check(lambda c: c in ids)

    def checked_classes(self):
        return [c for c in self.class_ids() if self._checked.get(c)]

    def set_checked(self, ids):
        ids = set(int(i) for i in ids)
        self._check(lambda c: c in ids)

    def new_class(self):
        ids = self.class_ids()
        c = (max(ids) + 1) if ids else 1
        self._extra.add(c)
        self._checked[c] = True
        self._rebuild_table()
        self.select_class(c)
        return c

    # ============================================================== points
    def prompts_df(self):
        """The prompt table from the Points layer (voxel coordinates of the image)."""
        lay = self.points_layer()
        img = self.image_layer
        if lay is None or img is None or len(lay.data) == 0:
            return PP.empty_points_df()
        nd = img.ndim
        data = np.asarray(lay.data, dtype=float)[:, -nd:]
        vox = PP.points_data_to_voxels(data, np.asarray(lay.scale)[-nd:], np.asarray(lay.translate)[-nd:],
                                       np.asarray(img.scale), np.asarray(img.translate),
                                       rotate=lay.rotate, shear=lay.shear)
        return PP.points_to_df(vox, lay.features, image_name=img.name)

    def set_prompts(self, df):
        """Replace the Points layer content by a prompt table (creates the layer if needed)."""
        img = self.image_layer
        if img is None or self.seeds_layer is None:
            raise RuntimeError("select an image and seeds first")
        nd = img.ndim
        vox, feats = PP.df_to_points(df, ndim=nd)
        lay = self.points_layer()
        if lay is None:
            size = max(3.0, round(max(np.shape(img.data)) * 0.012))
            lay = self.viewer.add_points(
                vox, ndim=nd, features=feats, name=self.points_name(), size=size,
                face_color="label", face_color_cycle={True: POS_COLOR, False: NEG_COLOR},
                border_color="white", border_width=0.05, **self._layer_kwargs())
            lay.metadata[META_KEY] = {"role": "prompts"}
            self._connect_points_layer()
        else:
            si, ti = np.asarray(img.scale), np.asarray(img.translate)
            sp, tp = np.asarray(lay.scale)[-nd:], np.asarray(lay.translate)[-nd:]
            data = ((vox * si + ti) - tp) / sp            # voxels -> this layer's data coordinates
            lay.data = data
            lay.features = feats
            lay.refresh_colors()
        self._apply_feature_defaults()
        self._apply_shown()
        self._rebuild_table()
        return lay

    def _connect_points_layer(self):
        lay = self.points_layer()
        if lay is self._points_conn:
            return
        old = self._points_conn
        if old is not None:
            self._disconnect_layer_event(old.events.data, self._on_points_data)
            self._disconnect_layer_event(old.selected_data.events.items_changed, self._on_points_selection)
        self._points_conn = lay
        if lay is not None:
            self._connect_layer_event(lay.events.data, self._on_points_data)
            # napari copies the features of a selected point into feature_defaults; put ours back
            self._connect_layer_event(lay.selected_data.events.items_changed, self._on_points_selection)
            self._apply_feature_defaults()
            self._apply_shown()

    def _on_points_data(self, event=None):
        if self._alive():
            self._points_timer.start()    # coalesce; also runs after napari finished adding

    def _on_points_selection(self, event=None):
        if self._alive():
            self._defaults_timer.start(0)

    def _on_points_settled(self):
        self._apply_shown()
        self._rebuild_table()

    def _apply_feature_defaults(self):
        lay = self.points_layer()
        if lay is None or self._current is None:
            return
        pos = self.pos_rb.isChecked()
        lay.feature_defaults = {"class_id": int(self._current), "label": pos,
                                "point_type": "positive" if pos else "negative", "sample_method": "manual"}

    def _apply_shown(self):
        lay = self.points_layer()
        if lay is None or len(lay.data) == 0:
            return
        if self.only_cb.isChecked() and self._current is not None:
            shown = lay.features["class_id"].to_numpy() == self._current
        else:
            shown = np.ones(len(lay.data), bool)
        if not np.array_equal(np.asarray(lay.shown), shown):
            lay.shown = shown

    def start_adding_points(self):
        if self._current is None:
            show_error("Select a class first")
            return
        lay = self.points_layer()
        if lay is None:
            lay = self.set_prompts(PP.empty_points_df())
        self.viewer.layers.selection.active = lay
        self._apply_feature_defaults()
        lay.mode = "add"

    def clear_points_of_checked(self):
        df = self.prompts_df()
        if not len(df):
            return
        self.set_prompts(df[~df.class_id.isin(self.checked_classes())].reset_index(drop=True))

    def generate(self):
        """Sample points for the checked classes (background); replaces their generated points."""
        img, seeds = self.image_layer, self.seeds_layer
        if img is None or seeds is None:
            show_error("Select an image and seeds first")
            return
        classes = self.checked_classes()
        if not classes:
            show_error("Check at least one class")
            return
        try:
            roi = self.roi_use.roi_for_run(np.shape(img.data))
        except RoiError as e:
            show_error(str(e))
            return
        opts = dict(default_n_pos=self.npos_spin.value(), default_n_neg=self.nneg_spin.value(),
                    default_method=self.method_combo.currentText(), negative_from_bg=self.neg_bg_cb.isChecked(),
                    negative_from_other_classes=self.neg_other_cb.isChecked(),
                    negative_per_other_class=self.neg_each_cb.isChecked(),
                    exclude_zero_intensity=self.zero_cb.isChecked(), image_name=img.name)
        seed = self.seed_spin.value()
        sdata, idata = np.asarray(seeds.data), np.asarray(img.data)
        self._job["gen"] += 1
        job = self._job["gen"]
        cancel = self._cancel = threading.Event()
        self._life["cancel"][:] = [cancel]
        logs = []
        self._busy["gen"] = True
        self._update_buttons()
        self.prompt_status.setText(f"Sampling points for {len(classes)} classes...")
        t0 = time.perf_counter()

        def work():
            kw = dict(classes=classes, rng=np.random.RandomState(seed), log=logs.append, cancel=cancel, **opts)
            if roi is None:
                return PP.seeds_to_point_prompts(sdata, idata, **kw)
            return PP.seeds_to_point_prompts_in_roi(sdata, idata, roi, **kw)

        def done(gen):
            if job != self._job["gen"]:
                return
            cur = self.prompts_df()
            chk = cur.class_id.isin(classes)
            manual = chk & (cur.sample_method == "manual")
            new = pd.concat([cur[~chk], gen, cur[manual]], ignore_index=True)
            new = new[PP.POINT_COLUMNS] if len(new) else PP.empty_points_df()
            self.set_prompts(new.reset_index(drop=True))
            missing = sorted(set(classes) - set(gen.class_id.tolist())) if len(gen) else classes
            msg = f"{len(gen)} points for {len(classes) - len(missing)} classes in {time.perf_counter() - t0:.1f} s"
            if missing:
                msg += f"; no points for {', '.join(map(str, missing[:20]))}" + (" ..." if len(missing) > 20 else "")
                msg += " (no seed voxel" + (" inside the ROI)" if roi is not None else ")")
            self.prompt_status.setText(msg)

        def failed(e):
            if job != self._job["gen"]:
                return
            if isinstance(e, Cancelled):
                self.prompt_status.setText("Sampling stopped")
                return
            self.prompt_status.setText(f"Sampling failed: {e}")
            self._report_error(e)

        def finished():
            if job == self._job["gen"]:
                self._busy["gen"] = False
                self._update_buttons()

        self._run_bg(work, done, failed, finished)

    # ------------------------------------------------------------------ CSV
    def export_csv(self, path):
        df = self.prompts_df()
        df.to_csv(path, index=False)
        return len(df)

    def import_csv(self, path):
        df = pd.read_csv(path)
        need = {"x", "y", "z", "label", "class_id"}
        if not need <= set(df.columns):
            raise ValueError(f"CSV is missing columns: {sorted(need - set(df.columns))}")
        df = df.copy()
        df["label"] = df["label"].map(lambda v: str(v).strip().lower() in ("true", "1", "1.0")
                                      if not isinstance(v, (bool, np.bool_)) else bool(v))
        for col, default in (("image_name", ""), ("point_type", None), ("sample_method", "imported")):
            if col not in df:
                df[col] = default
        df["point_type"] = [pt if isinstance(pt, str) and pt else ("positive" if lb else "negative")
                            for pt, lb in zip(df["point_type"], df["label"])]
        shape = self._shape()
        if shape is not None:
            z_ok = (df.z >= 0) & (df.z < (shape[0] if len(shape) == 3 else 1))
            inside = z_ok & (df.y >= 0) & (df.y < shape[-2]) & (df.x >= 0) & (df.x < shape[-1])
            if not inside.all():
                show_warning(f"{int((~inside).sum())} points outside the image were skipped")
                df = df[inside]
        self.set_prompts(df.reset_index(drop=True))
        return len(df)

    def _import_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import prompts", "", "CSV (*.csv)")
        if path:
            try:
                n = self.import_csv(path)
                self.prompt_status.setText(f"Imported {n} points from {path}")
            except Exception as e:
                self._report_error(e)

    def _export_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export prompts", f"{self.points_name()}.csv", "CSV (*.csv)")
        if path:
            n = self.export_csv(path)
            self.prompt_status.setText(f"Exported {n} points to {path}")

    # ------------------------------------------------------------ scribbles
    def create_scribble_layer(self, negative=False):
        img, seeds = self.image_layer, self.seeds_layer
        if img is None or seeds is None:
            show_error("Select an image and seeds first")
            return None
        name = self.scribble_name(negative)
        lay = self._layer(name, Labels)
        if lay is None:
            ids = self.class_ids()
            dtype = PS._label_dtype(max(ids) if ids else 1)
            lay = self.viewer.add_labels(np.zeros(np.shape(img.data), dtype), name=name, **self._layer_kwargs())
            lay.metadata[META_KEY] = {"role": "negative scribbles" if negative else "scribbles"}
        if self._current is not None:
            lay.selected_label = self._current
        self.viewer.layers.selection.active = lay
        lay.mode = "paint"
        return lay

    # ================================================================ model
    def _browse_model(self):
        d = QFileDialog.getExistingDirectory(self, "nnInteractive model folder", self.model_edit.text())
        if d:
            self.model_edit.setText(d)

    def _on_device_changed(self):
        cpu = self.device_combo.currentData() == "cpu"
        self.cpu_label.setText("very slow: minutes per class" if cpu else "")

    def load_model(self):
        if not self.available:
            show_error(self._status[1])
            return
        folder = self.model_edit.text().strip()
        if not folder:
            show_error("Choose the model folder first")
            return
        device = self.device_combo.currentData()
        self._job["load"] += 1
        job = self._job["load"]
        self._busy["load"] = True
        self.model_status.setText("loading...")
        self._update_buttons()

        def done(info):
            if job != self._job["load"]:
                return
            QSettings(*SETTINGS).setValue("prompt/model_dir", folder)
            dev = info.get("device", "?")
            warn = " - CPU is very slow" if str(dev).startswith("cpu") else ""
            lic = f", license {info['license']}" if info.get("license") else ""
            self.model_status.setText(f"{self.engine.name} {info.get('version', '?')} on {dev}, "
                                      f"loaded in {info.get('seconds', 0):.1f} s{lic}{warn}")

        def failed(e):
            if job != self._job["load"]:
                return
            self.model_status.setText(f"loading failed: {type(e).__name__}: {e}")
            self._report_error(e)

        def finished():
            if job == self._job["load"]:
                self._busy["load"] = False
                self._update_buttons()

        self._run_bg(lambda: self.engine.load(folder, device), done, failed, finished)

    def release_model(self):
        if self._busy["run"]:
            show_error("Stop the run first")
            return
        self.engine.release()
        self.model_status.setText("released" if self.available else self.model_status.text())
        self._update_buttons()

    # ================================================================== run
    def output_name(self, roi=None):
        name = f"{self._seeds_name()}_nni"
        return R.roi_name(name) if roi is not None else name

    def _store_key(self, out_name):
        return (id(self.image_layer), id(self.seeds_layer), out_name)

    def _store_for_current_output(self, create=True):
        if self.image_layer is None or self.seeds_layer is None:
            return None
        key = self._store_key(self.output_name(self._display_roi()))
        if create:
            return self._stores.setdefault(key, PS.ResultStore())
        return self._stores.get(key)

    def _update_output_label(self):
        if self.seeds_layer is None:
            self.output_label.setText("")
            return
        self.output_label.setText(self.output_name(self._display_roi()))

    def run_checked(self):
        return self._start_run(self.checked_classes(), "scratch")

    def update_current(self):
        if self._current is None:
            show_error("Select a class first")
            return None
        return self._start_run([self._current], "update")

    def _start_run(self, classes, mode):
        img, seeds = self.image_layer, self.seeds_layer
        if not self.available:
            show_error(self._status[1])
            return None
        if img is None or seeds is None:
            show_error("Select an image and seeds first")
            return None
        if not self.engine.loaded:
            show_error("Load the model first")
            return None
        if img.ndim != 3:
            show_error("Prompt (AI) needs a 3D image (nnInteractive is a 3D model)")
            return None
        if not classes:
            show_error("Check at least one class")
            return None
        shape = tuple(np.shape(img.data))
        try:
            roi = self.roi_use.roi_for_run(shape)
        except RoiError as e:
            show_error(str(e))
            return None
        key, ptype, src = self.input_type()
        if not self.engine.supports_prompt_type(ptype):
            show_error(f"{self.engine.name} does not support this input type")
            return None
        scribbles, neg = None, False
        if src == "seed":
            neg = self.neg_seed_cb.isChecked()
        elif src == "drawn":
            lay = self._layer(self.scribble_name(False), Labels)
            if lay is None:
                show_error(f"Create and paint '{self.scribble_name(False)}' first")
                return None
            scribbles = np.asarray(lay.data)
            nl = self._layer(self.scribble_name(True), Labels)
            neg = np.asarray(nl.data) if nl is not None else False
        df = self.prompts_df()
        seeds_snap = np.array(seeds.data, copy=True)
        img_data = np.asarray(img.data)
        crop = PS.inference_crop(shape, roi, self.margin_spin.value())
        out_name = self.output_name(roi)
        store = self._stores.setdefault(self._store_key(out_name), PS.ResultStore())
        image_key = (id(img), img.name, id(img.data))
        base_seed = self.seed_spin.value()
        ctx = dict(out=out_name, store=store, seeds=seeds_snap, shape=shape, roi=roi, ptype=ptype, input=key,
                   layer_kwargs=self._layer_kwargs(), image=img.name, seeds_name=seeds.name,
                   margin=self.margin_spin.value(), base_seed=base_seed, mode=mode, classes=list(classes),
                   t0=time.perf_counter(), done=0)
        self._ctx = ctx

        self._job["run"] += 1
        job = self._job["run"]
        cancel = self._cancel = threading.Event()
        self._life["cancel"][:] = [cancel]
        prog = _Progress()
        prog.changed.connect(self._guard(lambda i, n: job == self._job["run"] and self._on_point_progress(i, n)))
        engine = self.engine
        for c in classes:
            self._class_status[c] = "queued"
        self._class_status[classes[0]] = "running"
        self._busy["run"] = True
        self.progress.setRange(0, len(classes))
        self.progress.setValue(0)
        self.progress.setFormat(f"%v / %m classes")
        self.run_status.setText(f"Running {len(classes)} classes ({mode})...")
        self._rebuild_table()

        def gen():
            yield from PS.run_classes(
                engine, img_data, seeds_snap, df, classes, crop=crop, image_key=image_key, roi=roi,
                mode="update" if mode == "update" else "scratch", store=store, prompt_type=ptype,
                scribbles=scribbles, neg_scribble=neg, base_seed=base_seed, cancel=cancel,
                on_point=lambda i, n: prog.changed.emit(i, n))

        w = thread_worker(gen, ignore_errors=True)()
        w.yielded.connect(self._guard(lambda r: job == self._job["run"] and self._on_result(r, ctx)))
        w.errored.connect(self._guard(lambda e: job == self._job["run"] and self._on_run_error(e, ctx)))
        w.finished.connect(self._guard(lambda: job == self._job["run"] and self._on_run_finished(ctx)))
        w._prog = prog           # keep the progress object alive with the worker
        self._workers.add(w)
        w.finished.connect(lambda: self._workers.discard(w))
        self._update_buttons()
        w.start()
        return w

    def _on_point_progress(self, i, n):
        if self._ctx is not None:
            c = self._next_running(self._ctx)
            if c is not None:
                self._class_status[c] = f"running: point {i}/{n}"
                r = self._row_of(c)
                if r >= 0 and self.table.item(r, 4) is not None:
                    self.table.item(r, 4).setText(self._class_status[c])

    def _next_running(self, ctx):
        i = ctx["done"]
        return ctx["classes"][i] if i < len(ctx["classes"]) else None

    def _on_result(self, r, ctx):
        ctx["done"] += 1
        if isinstance(r, tuple):
            _, c, reason = r
            self._class_status[c] = f"skipped: {reason}"
        else:
            c = r.class_id
            region = ctx["store"].set(r)
            extra = f", {r.n_dropped} points outside" if r.n_dropped else ""
            self._class_status[c] = f"done ({r.mode.replace('_', ' ')}, {r.seconds:.1f} s{extra})"
            self._write_output(ctx, region)
        nxt = self._next_running(ctx)
        if nxt is not None:
            self._class_status[nxt] = "running"
        self.progress.setValue(ctx["done"])
        self._rebuild_table()

    def _on_run_error(self, e, ctx):
        for c in ctx["classes"][ctx["done"]:]:
            self._class_status[c] = "stopped" if isinstance(e, Cancelled) else "failed"
        if isinstance(e, Cancelled):
            self.run_status.setText(f"Stopped after {ctx['done']} of {len(ctx['classes'])} classes")
        elif is_oom(e):
            self.run_status.setText(str(e))
            show_error(f"Prompt (AI): {e}")
        else:
            self.run_status.setText(f"Failed: {type(e).__name__}: {e}")
            self._report_error(e)
        ctx["error"] = e

    def _on_run_finished(self, ctx):
        self._busy["run"] = False
        if "error" not in ctx:
            self.run_status.setText(f"Done: {ctx['done']} classes in {time.perf_counter() - ctx['t0']:.1f} s"
                                    + (f"; {ctx['store'].overlap_voxels():,} voxels claimed by more than one "
                                       f"class" if len(ctx["store"]) > 1 else ""))
        self._rebuild_table()
        self._update_buttons()

    def stop(self):
        if self._cancel is not None:
            self._cancel.set()

    def revert_current(self):
        store = self._store_for_current_output(create=False)
        if store is None or self._current is None or not store.can_revert(self._current):
            show_error("Nothing to revert for this class")
            return
        region = store.revert(self._current)
        ctx = self._output_ctx(store)
        self._class_status[self._current] = "reverted"
        if ctx is not None:
            self._write_output(ctx, region)
        self._rebuild_table()

    def _output_ctx(self, store):
        ctx = self._ctx
        if ctx is not None and ctx["store"] is store:
            return ctx
        return None

    def _recompose(self):
        ctx = self._ctx
        if ctx is None or ctx["out"] not in self.viewer.layers:
            return
        self._write_output(ctx, None)

    # --------------------------------------------------------------- output
    def _write_output(self, ctx, region):
        store, shape = ctx["store"], ctx["shape"]
        rule = self.overlap_combo.currentData()
        ids = store.classes()
        dtype = PS._label_dtype(max(ids) if ids else 0)
        lay = self._layer(ctx["out"], Labels)
        if lay is not None and (tuple(lay.data.shape) != shape or np.dtype(lay.data.dtype).itemsize < dtype.itemsize):
            lay = None
            self.viewer.layers.remove(ctx["out"])
        if lay is None:
            data = store.compose(shape, dtype=dtype, rule=rule, seeds=ctx["seeds"])
            lay = self.viewer.add_labels(data, name=ctx["out"], **ctx["layer_kwargs"])
        elif region is None:
            lay.data = store.compose(shape, dtype=lay.data.dtype, rule=rule, seeds=ctx["seeds"])
        else:
            lay.data[region] = store.compose(shape, dtype=lay.data.dtype, rule=rule, seeds=ctx["seeds"], region=region)
            lay.refresh()
        meta = dict(lay.metadata)
        info = self.engine.info
        meta[META_KEY] = {
            "engine": {k: info.get(k) for k in ("version", "device", "license")} | {"name": self.engine.name},
            "image": ctx["image"], "seeds": ctx["seeds_name"], "input_type": ctx["input"],
            "prompt_type": ctx["ptype"], "overlap_rule": rule, "roi_margin": ctx["margin"],
            "base_seed": ctx["base_seed"],
            "classes": {int(c): {"mode": r.mode, "voxels": r.n_voxels, "points": r.n_points,
                                 "points_dropped": r.n_dropped, "seconds": r.seconds,
                                 "prompt_type": r.prompt_type}
                        for c, r in ((c, store.get(c)) for c in ids)},
        }
        if ctx["roi"] is not None:
            meta[R.META_KEY] = ctx["roi"].to_metadata()
        lay.metadata = meta
        self.output_updated.emit(ctx["out"])
