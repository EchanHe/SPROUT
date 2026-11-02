"""Prompt (AI) tab for napari-sprout: seeds -> prompts -> interactive model.

Paper Fig. 1d: instead of threshold-guided growing, the seeds are turned into
prompts for a promptable model (nnInteractive now; the model list is a registry so
SAM can be added). The tab follows SPROUT's nnInteractive pipeline
(`python sprout.py --prompt --nninteractive`, sprout_core/nninteractive_predict.py):

    1. Input     image, seeds, ROI (the run stays inside the ROI)
    2. Classes   which seed classes to run (checked); the highlighted row is the class
                 new points belong to
    3. Model     model, weights folder, device, Load / Release
    4. Prompts   prompt type as in the YAML: "Seeds as scribbles" (prompt_type: scribble,
                 default) or "Points sampled from seeds" (prompt_type: point); optional
                 initial mask (init_seg_path); manual points; Import / Export YAML
    5. Run       Run checked / Stop -> one full-size Labels layer "{seeds}_nni"
                 ("{seeds}_nni_ROI_1" with an ROI); optionally one layer per class and
                 the pipeline's files in a folder

Overlaps: like the scripts, classes are composed so that the smaller id wins ('min_id').
The model runs in the background, one class after the other; each finished class is
pasted into the output right away. Heavy imports (torch, nnInteractive) happen only when
the model is loaded.

Kept in code but not in this UI (sprout_core and the methods below): drawn scribble
layers, seed as initial mask, other overlap rules, ROI margin (32), Update / Revert one
class, CSV import / export.
"""
import functools
import os
import threading
import time
import weakref
from pathlib import Path

import numpy as np
import pandas as pd
from napari.layers import Image, Labels, Points
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_error
from qtpy.QtCore import QEvent, QObject, QSettings, Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QProgressBar, QPushButton, QRadioButton, QScrollArea,
    QSizePolicy, QSpinBox, QTableWidgetItem, QVBoxLayout, QWidget,
)

from sprout_core import config_core
from sprout_core import prompt_points as PP
from sprout_core import prompt_seg as PS
from sprout_core import roi as R
from sprout_core.boundary import Cancelled
from sprout_core.nni_engine import NniEngine, nninteractive_status
from sprout_core.prompt_engine import is_oom
from sprout_core.roi import RoiError, parse_ids  # noqa: F401  (parse_ids: also importable from here)
from sprout_core.sprout_core import label_dtype

from ..utils import jobs
from ..utils.fit_table import FitTable
from ..utils.status import StatusLine, style, tab_hidden
from ..utils.roi_layers import get_roi_hub
from ..utils.roi_widget import RoiSelect
from ..utils.util_widget import create_output_folder_row

META_KEY = "sprout_prompt"
SETTINGS = ("SPROUT", "napari-sprout")
POS_COLOR, NEG_COLOR = "lime", "magenta"
TOOLTIP_ROLE = 3            # Qt.ToolTipRole
GREY = style("info")
LOAD_FIRST = "Load the model first (3. Model)"

# Promptable models offered in "3. Model": (label, engine class, availability check).
# The engine CLASS declares `name` and `capabilities`, so the UI knows what a model
# supports without loading weights or importing torch. Add SAM here later.
MODELS = [("nnInteractive", NniEngine, nninteractive_status)]

CAPABILITY_TEXT = {"points": "points", "scribble": "scribbles", "init_mask": "initial mask",
                   "bbox": "boxes", "lasso": "lasso"}

# Prompt types of the pipeline YAML (prompt_type) -> (core prompt type, label, tooltip)
PROMPT_MODES = {
    "scribble": ("scribble", "Seeds as scribbles",
                 "prompt_type: scribble - for each class, its whole seed is a positive scribble "
                 "(as `sprout.py --prompt --nninteractive`)."),
    "point": ("points", "Points sampled from seeds",
              "prompt_type: point - points sampled from each class's seed (Generate points), "
              "plus the points you add by hand."),
}
SCRIBBLE_HINT = "The seeds are the scribbles: paint or erase in the seeds Labels layer to change them, then Run."
MANUAL_HINT = ("Add points to fix a class, check only that class and Run again. "
               "To change a scribble, edit the seeds Labels layer directly.")
OVERLAP_RULE = "min_id"         # as nninter_predict: classes in descending id, last write wins
ROI_MARGIN = 32                 # image context (voxels) the model sees around the ROI
BASE_SEED = 0                   # random seed of the point sampling (and of the model, + class id)
COLUMNS = ["", "id", "seed voxels", "points + / -", "status", "result voxels"]
NEG_SLOW = "(slow on big images)"
YAML_RULES = config_core.input_val_nninteractive_run


def layer_dtype(max_id):
    """dtype of the result layers: at least uint16 (as the seeds), so a brushed id above 255 does not
    wrap; saved files keep the smallest dtype."""
    return label_dtype(max_id, np.uint16)


def yaml_default(key):
    return YAML_RULES[key].get("default")


def capabilities_text(caps):
    """'points · scribbles · initial mask' in a fixed order."""
    order = [c for c in CAPABILITY_TEXT if c in caps] + sorted(c for c in caps if c not in CAPABILITY_TEXT)
    return " · ".join(CAPABILITY_TEXT.get(c, c) for c in order)


def _source_path(layer):
    p = getattr(getattr(layer, "source", None), "path", None) if layer is not None else None
    return str(p) if p else None


def _same_path(a, b):
    try:
        return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))
    except Exception:
        return False


class _FitTable(FitTable):
    """The classes table: fitted by hand after each rebuild (see utils/fit_table.py)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, auto_fit=False, **kwargs)
        # the horizontal scrollbar can come and go after a fit (the vertical one appears,
        # the columns settle): fit again so its height is not counted when it is gone
        self.horizontalScrollBar().rangeChanged.connect(self._refit)
        self.horizontalHeader().sectionResized.connect(self._refit)

    def _refit(self, *_):
        self.fit()

    def viewportEvent(self, event):
        res = super().viewportEvent(event)
        if event.type() == QEvent.Resize:         # e.g. the vertical scrollbar appeared
            self.fit()
        return res


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


class PromptTab(QWidget):
    """The 'Prompt (AI)' tab.

    models: list of (label, engine class or factory, status_fn); default MODELS.
    engine_factory / status_fn (tests): one model built by engine_factory().
    """

    output_updated = Signal(str)          # output layer name

    def __init__(self, napari_viewer, engine_factory=None, status_fn=None, models=None):
        super().__init__()
        self.viewer = napari_viewer
        if engine_factory is not None:
            cls = engine_factory if isinstance(engine_factory, type) else None
            name = cls.name if cls is not None else None
            models = [(name, engine_factory, status_fn or nninteractive_status)]
        self.models = list(models or MODELS)
        self._model_index = 0
        self.engine = self._make_engine(0)
        if self.models[0][0] is None:                      # label from the engine built by a factory
            self.models[0] = (self.engine.name,) + tuple(self.models[0][1:])
        self._finalizer = weakref.finalize(self, _shutdown_engine, self.engine)
        # lifecycle: worker callbacks check life["alive"]; napari layer callbacks are listed
        # in life["conns"] and disconnected when the Qt object is destroyed
        self._life = {"alive": True, "cancel": [], "conns": []}
        self.destroyed.connect(functools.partial(_on_destroyed, self._life))
        self._status = self._check_status(0)
        self.available = bool(self._status[0])
        self._roi_model = get_roi_hub(napari_viewer)   # napari layer events as Qt signals

        self._img_layer = None
        self._seeds_layer = None
        self._stats = {}                 # class id -> seed voxels (inside the ROI)
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
        self._out_auto = True            # output name follows seeds / ROI until the user edits it
        self.class_config = None         # per-class sampling options from an imported YAML
        self.base_seed = int(yaml_default("seed") or BASE_SEED)   # YAML 'seed': sampling + seed + class id

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
        self._stats_pending = False      # class counts wait for the tab to be shown

        self._build_ui()
        self._connect()
        self.refresh_layers()

    # ------------------------------------------------------------- models
    def _make_engine(self, i):
        factory = self.models[i][1]
        return factory()

    def _check_status(self, i):
        fn = self.models[i][2]
        try:
            return fn() if fn is not None else (True, "")
        except Exception as e:                       # never break the panel
            return (False, f"cannot check {self.models[i][0]}: {e}")

    def model_class(self, i=None):
        """The engine class of a registry entry (capabilities known without loading)."""
        f = self.models[self._model_index if i is None else i][1]
        return f if isinstance(f, type) else type(self.engine)

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
        for c in (self.image_combo, self.seeds_combo):
            c.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            c.setMinimumContentsLength(12)
        f1.addRow("Image", self.image_combo)
        f1.addRow("Seeds", self.seeds_combo)
        self.roi_select = RoiSelect(self.viewer, self._shape, "Prompt")
        f1.addRow("ROI", self.roi_select)
        layout.addWidget(g1)

        # 2. Classes
        g2 = QGroupBox("2. Classes")
        l2 = QVBoxLayout(g2)
        self.table = _FitTable(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)     # height: see _FitTable.fit
        self.table.fit()
        self.table.setToolTip("Tick the classes to run. The highlighted row is the class new points "
                              "belong to.")
        l2.addWidget(self.table)
        row = QHBoxLayout()
        self.all_btn = QPushButton("All")
        self.all_btn.setToolTip("Tick every class")
        self.none_btn = QPushButton("None")
        self.none_btn.setToolTip("Untick every class")
        self.classes_label = QLabel("")
        self.classes_label.setStyleSheet(GREY)
        row.addWidget(self.all_btn)
        row.addWidget(self.none_btn)
        row.addWidget(self.classes_label, 1)
        l2.addLayout(row)
        layout.addWidget(g2)

        # 3. Model
        g3 = QGroupBox("3. Model")
        f3 = QFormLayout(g3)
        self.model_combo = QComboBox()
        for label, _, _ in self.models:
            self.model_combo.addItem(label)
        f3.addRow("Model", self.model_combo)
        self.supports_label = QLabel("")
        self.supports_label.setStyleSheet(GREY)
        f3.addRow(self.supports_label)
        self.model_edit = QLineEdit(str(QSettings(*SETTINGS).value("prompt/model_dir", "") or ""))
        self.model_edit.setPlaceholderText("folder of the model weights")
        self.model_edit.setToolTip("Folder of the weights, e.g. nnInteractive_v1.0 "
                                   "(remembered for the next session; model_path in the YAML).")
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setToolTip("Choose the weights folder")
        row = QHBoxLayout()
        row.addWidget(self.model_edit, 1)
        row.addWidget(self.browse_btn)
        f3.addRow("Weights", row)
        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto", None)
        self.device_combo.addItem("GPU", "cuda")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.setToolTip("Auto: the GPU if torch sees one, else the CPU.")
        self.cpu_label = QLabel("")
        self.cpu_label.setStyleSheet(style("warn"))
        row = QHBoxLayout()
        row.addWidget(self.device_combo)
        row.addWidget(self.cpu_label, 1)
        f3.addRow("Device", row)
        row = QHBoxLayout()
        self.load_btn = QPushButton("Load")
        self.load_btn.setToolTip("Load the weights (once; kept until Release or another folder / device).")
        self.release_btn = QPushButton("Release")
        self.release_btn.setToolTip("Free the model and its GPU memory.")
        row.addWidget(self.load_btn, 1)
        row.addWidget(self.release_btn)
        f3.addRow(row)
        self.model_status = StatusLine(hide_empty=True)
        f3.addRow(self.model_status)
        layout.addWidget(g3)

        # 4. Prompts
        g4 = QGroupBox("4. Prompts")
        l4 = QVBoxLayout(g4)
        row = QHBoxLayout()
        row.addStretch()
        self.import_btn = QPushButton("Import YAML")
        self.import_btn.setToolTip("Read a `sprout.py --prompt --nninteractive` config: model, device, prompt "
                                   "options, output folder; layers whose file is img_path / seg_path / "
                                   "init_seg_path are selected.")
        self.export_btn = QPushButton("Export YAML")
        self.export_btn.setToolTip("Write this setup as a config for `python sprout.py --prompt --nninteractive "
                                   "--config x.yaml`.")
        row.addWidget(self.import_btn)
        row.addWidget(self.export_btn)
        l4.addLayout(row)
        self.yaml_status = StatusLine(hide_empty=True)
        l4.addWidget(self.yaml_status)
        f4 = QFormLayout()
        self.scribble_rb = QRadioButton(PROMPT_MODES["scribble"][1])
        self.points_rb = QRadioButton(PROMPT_MODES["point"][1])
        self.scribble_rb.setToolTip(PROMPT_MODES["scribble"][2])
        self.points_rb.setToolTip(PROMPT_MODES["point"][2])
        self.scribble_rb.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.scribble_rb)
        grp.addButton(self.points_rb)
        box = QVBoxLayout()
        box.setSpacing(2)
        box.addWidget(self.scribble_rb)
        box.addWidget(self.points_rb)
        f4.addRow("Prompt type", box)
        l4.addLayout(f4)

        # -- Seeds as scribbles
        self.scribble_box = QWidget()
        ls = QVBoxLayout(self.scribble_box)
        ls.setContentsMargins(0, 0, 0, 0)
        self.neg_scribble_cb = QCheckBox("Other classes' seeds as negative scribbles")
        self.neg_scribble_cb.setChecked(bool(yaml_default("use_negative_scribble")))
        self.neg_scribble_cb.setToolTip("use_negative_scribble: for each class, the seeds of all other classes "
                                        "are added as a negative scribble.")
        ls.addWidget(self.neg_scribble_cb)
        self.scribble_hint = QLabel(SCRIBBLE_HINT)
        self.scribble_hint.setWordWrap(True)
        self.scribble_hint.setStyleSheet(GREY)
        ls.addWidget(self.scribble_hint)
        l4.addWidget(self.scribble_box)

        # -- Points sampled from seeds
        self.points_box = QWidget()
        fp = QFormLayout(self.points_box)
        fp.setContentsMargins(0, 0, 0, 0)
        self.npos_spin = QSpinBox()
        self.npos_spin.setRange(1, 1000)
        self.npos_spin.setValue(int(yaml_default("default_n_pos")))
        self.npos_spin.setToolTip("default_n_pos: positive points sampled inside each checked class's seed.")
        self.method_combo = QComboBox()
        for m, tip in (("random", "Uniformly at random inside the seed."),
                       ("grid", "One point per cell of a grid over the seed (spread out)."),
                       ("center_edge", "The deepest voxel of the seed, then points near its edge."),
                       ("kmeans", "Cluster centres of the seed voxels (needs scikit-learn).")):
            self.method_combo.addItem(m)
            self.method_combo.setItemData(self.method_combo.count() - 1, tip, TOOLTIP_ROLE)
        self.method_combo.setCurrentText(yaml_default("default_method"))
        self.method_combo.setToolTip("default_method: how the points are placed inside the seed.")
        self.nneg_spin = QSpinBox()
        self.nneg_spin.setRange(0, 1000)
        self.nneg_spin.setValue(int(yaml_default("default_n_neg")))
        self.nneg_spin.setToolTip("default_n_neg: negative points per class (zero-intensity voxels excluded).")
        self.neg_slow_label = QLabel(NEG_SLOW)
        self.neg_slow_label.setStyleSheet(GREY)
        self.neg_slow_label.setToolTip("Negatives are drawn from all background / other-class voxels: "
                                       "5-14 s on the 13 M-voxel dog demo.")
        row = QHBoxLayout()
        row.addWidget(self.npos_spin)
        row.addWidget(QLabel("method"))
        row.addWidget(self.method_combo)
        row.addStretch()
        fp.addRow("Positive", row)
        row = QHBoxLayout()
        row.addWidget(self.nneg_spin)
        row.addWidget(self.neg_slow_label)
        row.addStretch()
        fp.addRow("Negative", row)
        self.neg_bg_cb = QCheckBox("background")
        self.neg_bg_cb.setChecked(bool(yaml_default("negative_from_bg")))
        self.neg_bg_cb.setToolTip("negative_from_bg")
        self.neg_other_cb = QCheckBox("other classes")
        self.neg_other_cb.setChecked(bool(yaml_default("negative_from_other_classes")))
        self.neg_other_cb.setToolTip("negative_from_other_classes")
        self.neg_each_cb = QCheckBox("each other class")
        self.neg_each_cb.setChecked(bool(yaml_default("negative_per_other_class")))
        self.neg_each_cb.setToolTip("negative_per_other_class: n negatives from EACH other class "
                                    "(else from all other classes together).")
        row = QHBoxLayout()
        row.addWidget(self.neg_bg_cb)
        row.addWidget(self.neg_other_cb)
        row.addWidget(self.neg_each_cb)
        row.addStretch()
        fp.addRow("Negatives from", row)
        self.gen_btn = QPushButton("Generate points")
        self.gen_btn.setToolTip("Samples points for the checked classes into '{seeds}_prompts' (inside the ROI "
                                "picked in 1. Input). Replaces their generated points; points added by hand "
                                "are kept.")
        fp.addRow(self.gen_btn)
        l4.addWidget(self.points_box)

        # -- both
        f4b = QFormLayout()
        self.init_combo = QComboBox()
        self.init_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.init_combo.setMinimumContentsLength(12)
        self.init_combo.setToolTip("init_seg_path: an initial segmentation (any Labels layer, e.g. a Grow "
                                   "result). For each class, its voxels in this layer are the starting mask.")
        f4b.addRow("Initial mask", self.init_combo)
        l4.addLayout(f4b)

        # -- manual points
        gm = QGroupBox("Manual points")
        lm = QVBoxLayout(gm)
        self.pos_rb = QRadioButton("positive")
        self.pos_rb.setToolTip("New points mark voxels that belong to the class")
        self.neg_rb = QRadioButton("negative")
        self.neg_rb.setToolTip("New points mark voxels that do not belong to the class")
        self.pos_rb.setChecked(True)
        grp2 = QButtonGroup(self)
        grp2.addButton(self.pos_rb)
        grp2.addButton(self.neg_rb)
        self.edit_btn = QPushButton("Add points")
        self.edit_btn.setToolTip("Selects '{seeds}_prompts' in add mode: click in the image to add points to "
                                 "the class highlighted in the table.")
        row = QHBoxLayout()
        row.addWidget(QLabel("New points"))
        row.addWidget(self.pos_rb)
        row.addWidget(self.neg_rb)
        row.addWidget(self.edit_btn)
        row.addStretch()
        lm.addLayout(row)
        self.manual_hint = QLabel(MANUAL_HINT)
        self.manual_hint.setWordWrap(True)
        self.manual_hint.setStyleSheet(GREY)
        lm.addWidget(self.manual_hint)
        l4.addWidget(gm)
        self.prompt_status = StatusLine(hide_empty=True)
        l4.addWidget(self.prompt_status)
        layout.addWidget(g4)

        # 5. Run
        g5 = QGroupBox("5. Run")
        f5 = QFormLayout(g5)
        row = QHBoxLayout()
        self.run_btn = QPushButton("Run checked")
        self.run_btn.setToolTip("Runs the model class by class on the checked classes; each result is pasted "
                                "into the output layer as soon as it is done.")
        self.roi_select.set_run_button(self.run_btn, "Run checked")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setToolTip("Stop the run (the classes already done are kept), or stop generating points")
        row.addWidget(self.run_btn, 1)
        row.addWidget(self.stop_btn)
        f5.addRow(row)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        f5.addRow(self.progress)
        self.run_status = StatusLine(hide_empty=True)
        f5.addRow(self.run_status)
        self.output_edit = QLineEdit("")
        self.output_edit.setToolTip("Labels layer the results go to (full image size; classes not run are 0). "
                                    "Overlaps: a voxel claimed by two classes goes to the smaller id, as in "
                                    "the scripts.")
        f5.addRow("Output", self.output_edit)
        self.per_class_cb = QCheckBox("Also add each class as a separate layer")
        self.per_class_cb.setChecked(bool(yaml_default("return_per_class_masks")))
        self.per_class_cb.setToolTip("return_per_class_masks: one Labels layer per class "
                                     "('{output}_class_<id>'), before overlaps are resolved.")
        f5.addRow(self.per_class_cb)
        self.save_cb = QCheckBox("Save results to folder")
        self.save_cb.setToolTip("After a run, write <image>_segmentation.tif (+ <image>_class_<id>.tif), "
                                "used_config.yaml and, for points, <image>_point_prompts.csv - "
                                "the files of the pipeline.")
        f5.addRow(self.save_cb)
        folder_row, self.folder_edit = create_output_folder_row(label="Folder")
        self.folder_browse_btn = folder_row.browse_btn
        f5.addRow(folder_row)
        layout.addWidget(g5)
        layout.addStretch()

        container = QWidget()
        container.setLayout(layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        self.scroll = scroll
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._apply_model()
        self._on_mode_changed()
        self._on_nneg_changed()
        self._update_buttons()

    def _connect(self):
        # napari events reach the tab only through the RoiHub's Qt signals (bound
        # methods: Qt disconnects them when the tab is destroyed)
        self._roi_model.layer_inserted.connect(self._schedule_refresh)
        self._roi_model.layer_removed.connect(self._schedule_refresh)
        self._roi_model.layer_renamed.connect(self._on_layer_renamed)
        self.image_combo.currentTextChanged.connect(lambda *_: self._on_inputs_changed())
        self.seeds_combo.currentTextChanged.connect(lambda *_: self._on_inputs_changed())
        self.roi_select.changed.connect(self._roi_timer.start)

        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._on_selection)
        self.all_btn.clicked.connect(lambda: self._check(lambda c: True))
        self.none_btn.clicked.connect(lambda: self._check(lambda c: False))

        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.browse_btn.clicked.connect(self._browse_model)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        self.load_btn.clicked.connect(self.load_model)
        self.release_btn.clicked.connect(self.release_model)

        self.scribble_rb.toggled.connect(lambda *_: self._on_mode_changed())
        self.nneg_spin.valueChanged.connect(lambda *_: self._on_nneg_changed())
        self.gen_btn.clicked.connect(self.generate)
        self.pos_rb.toggled.connect(lambda *_: self._apply_feature_defaults())
        self.edit_btn.clicked.connect(self.start_adding_points)
        self.import_btn.clicked.connect(lambda: self.import_yaml())
        self.export_btn.clicked.connect(lambda: self.export_yaml())

        self.run_btn.clicked.connect(self.run_checked)
        self.stop_btn.clicked.connect(self.stop)
        self.output_edit.textEdited.connect(self._on_output_edited)

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

    def _on_layer_renamed(self, layer, old):
        """Rename in place: the selections (and the class table) stay."""
        for combo in (self.image_combo, self.seeds_combo, self.init_combo):
            i = combo.findData(old) if combo is self.init_combo else combo.findText(old)
            if i >= 0:
                combo.blockSignals(True)
                combo.setItemText(i, layer.name)
                if combo is self.init_combo:
                    combo.setItemData(i, layer.name)
                combo.blockSignals(False)

    def _combos_in_sync(self):
        images = [l.name for l in self.viewer.layers if isinstance(l, Image)]
        labels = [l.name for l in self.viewer.layers if isinstance(l, Labels)]
        texts = [[c.itemText(i) for i in range(c.count())]
                 for c in (self.image_combo, self.seeds_combo, self.init_combo)]
        return texts == [images, labels, ["None"] + labels]

    def showEvent(self, event):
        super().showEvent(event)
        if not self._combos_in_sync():     # a missed layer event: re-sync, keeping the selection
            self.refresh_layers()
        if self._stats_pending:            # the inputs changed while the tab was hidden
            self.compute_stats()

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

    @property
    def init_layer(self):
        name = self.init_combo.currentData()
        return self._layer(name, Labels) if name else None

    def _seeds_name(self):
        lay = self.seeds_layer
        return lay.name if lay is not None else ""

    def points_name(self):
        return f"{self._seeds_name()}_prompts"

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
        return None if shape is None else self.roi_select.roi_for_display(shape)

    def prompt_mode(self):
        """'scribble' | 'point' (the YAML prompt_type)."""
        return "scribble" if self.scribble_rb.isChecked() else "point"

    def set_prompt_mode(self, mode):
        (self.scribble_rb if mode == "scribble" else self.points_rb).setChecked(True)

    def input_type(self):
        """(key, core prompt type, scribble source) of the current prompt type."""
        if self.prompt_mode() == "scribble":
            return "seed_scribble", "scribble", "seed"
        return "points", "points", None

    def type_enabled(self, key):
        """Whether an input type can be chosen ('seed_scribble' / 'points' / 'init_mask')."""
        if key in ("seed_scribble", "scribble"):
            return self.scribble_rb.isEnabled()
        if key in ("points", "point"):
            return self.points_rb.isEnabled()
        if key == "init_mask":
            return self.init_combo.isEnabled()
        return False

    def _update_buttons(self):
        busy = any(self._busy.values())
        have_inputs = self.image_layer is not None and self.seeds_layer is not None
        is3d = self._shape() is not None and len(self._shape()) == 3
        self.gen_btn.setEnabled(have_inputs and not self._busy["gen"])
        loaded = self.available and self.engine.loaded
        self.load_btn.setEnabled(self.available and not self._busy["load"] and not self._busy["run"])
        self.release_btn.setEnabled(loaded and not self._busy["run"] and not self._busy["load"])
        # Run stays clickable without a loaded model: the click says what to do
        self.run_btn.setEnabled(self.available and have_inputs and is3d and not busy)
        self.stop_btn.setEnabled(self._busy["run"] or self._busy["gen"])
        if not self.available:
            tip = f"{self.engine.name} is not installed in this Python environment"
            for b in (self.load_btn, self.run_btn):
                b.setToolTip(tip)

    # --------------------------------------------------------------- model
    def _apply_model(self):
        """Supports line + prompt types for the selected model (from its class: nothing is loaded)."""
        cls = self.model_class()
        caps = frozenset(getattr(cls, "capabilities", self.engine.capabilities))
        self.supports_label.setText(f"Supports: {capabilities_text(caps)}")
        self.model_status.set("not loaded" if self.available else f"{self.engine.name} is not installed", "info")
        name = self.models[self._model_index][0]
        for rb, mode in ((self.scribble_rb, "scribble"), (self.points_rb, "point")):
            ptype, label, tip = PROMPT_MODES[mode]
            req = PS.required_capabilities(ptype)
            ok = req <= caps
            rb.setEnabled(ok)
            rb.setToolTip(tip if ok else f"{name} does not support "
                                         f"{', '.join(CAPABILITY_TEXT.get(c, c) for c in sorted(req - caps))}")
        if not self.scribble_rb.isEnabled() and self.scribble_rb.isChecked():
            self.points_rb.setChecked(True)                 # points: the fallback
        init_ok = "init_mask" in caps
        self.init_combo.setEnabled(init_ok)
        if not init_ok:
            self.init_combo.setCurrentIndex(0)
            self.init_combo.setToolTip(f"{name} does not support an initial mask")

    def _on_model_changed(self, i):
        if i == self._model_index or i < 0:
            return
        if self._busy["run"] or self._busy["load"]:
            self.model_combo.blockSignals(True)
            self.model_combo.setCurrentIndex(self._model_index)
            self.model_combo.blockSignals(False)
            self.model_status.set("Stop the run first", "error")
            return
        self._finalizer()                                  # release the old model
        self._model_index = i
        self.engine = self._make_engine(i)
        self._finalizer = weakref.finalize(self, _shutdown_engine, self.engine)
        self._status = self._check_status(i)
        self.available = bool(self._status[0])
        self.banner.setText(self._status[1] if not self.available else "")
        self.banner.setVisible(not self.available)
        self._apply_model()
        self._update_buttons()

    def _on_mode_changed(self):
        scribble = self.prompt_mode() == "scribble"
        self.scribble_box.setVisible(scribble)
        self.points_box.setVisible(not scribble)

    def _on_nneg_changed(self):
        self.neg_slow_label.setVisible(self.nneg_spin.value() > 0)

    # ================================================================ input
    def refresh_layers(self):
        img_cur, seeds_cur = self.image_combo.currentText(), self.seeds_combo.currentText()
        init_cur = self.init_combo.currentData()
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
        self.init_combo.blockSignals(True)
        self.init_combo.clear()
        self.init_combo.addItem("None", None)
        for n in labels:
            self.init_combo.addItem(n, n)
        i = self.init_combo.findData(init_cur) if init_cur else 0
        self.init_combo.setCurrentIndex(max(i, 0))
        self.init_combo.blockSignals(False)
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
        self._out_auto = True
        if seeds is not None and img is not None and np.shape(seeds.data) != np.shape(img.data):
            self.classes_label.setText(f"Seeds {np.shape(seeds.data)} and image {np.shape(img.data)} "
                                       f"differ in shape")
            self._seeds_layer = None
        self._connect_points_layer()
        self._update_output_name()
        self.roi_select.refresh()        # shows an ROI / image shape mismatch
        self.compute_stats()
        self._update_buttons()

    def _on_roi_changed(self):
        if not self._alive():
            return
        roi = self._display_roi()
        if (None if roi is None else roi.key) == (None if self._last_roi is None else self._last_roi.key):
            return                       # e.g. None -> None, or an unused ROI edited
        self.compute_stats()
        self._update_output_name()
        self._update_buttons()

    # ============================================================== classes
    def compute_stats(self):
        """Seed voxels per class (inside the ROI if one is picked), in the background."""
        self._job["stats"] += 1
        job = self._job["stats"]
        seeds = self.seeds_layer
        self._stats_pending = False
        if seeds is None or self.image_layer is None:
            self._stats = {}
            self._rebuild_table()
            return
        if tab_hidden(self):             # a hidden tab counts when it is shown (showEvent)
            self._stats_pending = True
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

        self.classes_label.setText("counting seed voxels…")
        self._run_bg(work, done)

    def class_ids(self):
        ids = set(self._stats)
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
        self.table.fit()
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

    def checked_classes(self):
        return [c for c in self.class_ids() if self._checked.get(c)]

    def set_checked(self, ids):
        ids = set(int(i) for i in ids)
        self._check(lambda c: c in ids)

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

    def run_points(self, ptype=None):
        """(points a run uses, number left out). Seeds as scribbles uses the manual points only:
        points generated earlier in the points mode stay in the layer but are not sent."""
        df = self.prompts_df()
        if (ptype or self.input_type()[1]) != "scribble" or not len(df):
            return df, 0
        manual = (df["sample_method"] == "manual").to_numpy()
        return df[manual].reset_index(drop=True), int((~manual).sum())

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

    def _on_points_data(self, event=None):
        if self._alive():
            self._points_timer.start()    # coalesce; also runs after napari finished adding

    def _on_points_selection(self, event=None):
        if self._alive():
            self._defaults_timer.start(0)

    def _on_points_settled(self):
        self._rebuild_table()

    def _apply_feature_defaults(self):
        lay = self.points_layer()
        if lay is None or self._current is None:
            return
        pos = self.pos_rb.isChecked()
        lay.feature_defaults = {"class_id": int(self._current), "label": pos,
                                "point_type": "positive" if pos else "negative", "sample_method": "manual"}

    def start_adding_points(self):
        if self._current is None:
            self.prompt_status.set("Select a class first", "error")
            return
        lay = self.points_layer()
        if lay is None:
            lay = self.set_prompts(PP.empty_points_df())
        self.viewer.layers.selection.active = lay
        self._apply_feature_defaults()
        lay.mode = "add"

    def point_options(self):
        """The point sampling options with the YAML's names."""
        return dict(default_n_pos=self.npos_spin.value(), default_n_neg=self.nneg_spin.value(),
                    default_method=self.method_combo.currentText(), negative_from_bg=self.neg_bg_cb.isChecked(),
                    negative_from_other_classes=self.neg_other_cb.isChecked(),
                    negative_per_other_class=self.neg_each_cb.isChecked())

    def generate(self):
        """Sample points for the checked classes (background); replaces their generated points."""
        img, seeds = self.image_layer, self.seeds_layer
        if img is None or seeds is None:
            self.prompt_status.set("Select an image and seeds first", "error")
            return
        classes = self.checked_classes()
        if not classes:
            self.prompt_status.set("Check at least one class", "error")
            return
        try:
            roi = self.roi_select.roi_for_run(np.shape(img.data))
        except RoiError as e:
            self.prompt_status.set(str(e), "error")
            return
        opts = dict(self.point_options(), class_config=self.class_config, exclude_zero_intensity=True,
                    image_name=img.name)
        sdata, idata = np.asarray(seeds.data), np.asarray(img.data)
        self._job["gen"] += 1
        job = self._job["gen"]
        cancel = self._cancel = threading.Event()
        self._life["cancel"][:] = [cancel]
        logs = []
        self._busy["gen"] = True
        self._update_buttons()
        self.prompt_status.set(f"Sampling points for {len(classes)} classes…", "info")
        t0 = time.perf_counter()

        def work():
            kw = dict(classes=classes, rng=np.random.RandomState(self.base_seed), log=logs.append, cancel=cancel,
                      **opts)
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
                msg += f"; no points for {', '.join(map(str, missing[:20]))}" + (" …" if len(missing) > 20 else "")
                msg += " (no seed voxel" + (" inside the ROI)" if roi is not None else ")")
            self.prompt_status.set(msg, "info")

        def failed(e):
            if job != self._job["gen"]:
                return
            if isinstance(e, Cancelled):
                self.prompt_status.set("Stopped: no points sampled", "warn")
                return
            self.prompt_status.set(f"Sampling failed: {jobs.reason(e)}", "error")
            self._report_error(e)

        def finished():
            if job == self._job["gen"]:
                self._busy["gen"] = False
                self._update_buttons()

        self._run_bg(work, done, failed, finished)

    # ----------------------------------------------- kept, not in the UI
    def clear_points_of_checked(self):
        df = self.prompts_df()
        if not len(df):
            return
        self.set_prompts(df[~df.class_id.isin(self.checked_classes())].reset_index(drop=True))

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
                self.prompt_status.set(f"{int((~inside).sum())} points outside the image were skipped", "warn")
                df = df[inside]
        self.set_prompts(df.reset_index(drop=True))
        return len(df)

    # ================================================================ YAML
    def yaml_config(self):
        """(config dict for `sprout.py --prompt --nninteractive`, list of placeholder keys)."""
        missing = []

        def path_of(layer, key, what):
            p = _source_path(layer)
            if p:
                return p
            missing.append(key)
            return f"<path to {what}>"

        model = self.model_edit.text().strip()
        if not model:
            missing.append("model_path")
        dev = self.device_combo.currentData()
        cfg = {
            "model_path": model or "<path to the nnInteractive weights folder>",
            "device": "cpu" if dev == "cpu" else "cuda:0",
            "img_path": path_of(self.image_layer, "img_path",
                                f"{self.image_layer.name}.tif" if self.image_layer is not None else "the image"),
            "seg_path": path_of(self.seeds_layer, "seg_path",
                                f"{self.seeds_layer.name}.tif" if self.seeds_layer is not None else "the seeds"),
            "output_folder": self.folder_edit.text().strip() or "<output folder>",
        }
        if self.init_layer is not None:
            cfg["init_seg_path"] = path_of(self.init_layer, "init_seg_path", f"{self.init_layer.name}.tif")
        cfg["prompt_type"] = self.prompt_mode()
        cfg.update(self.point_options())
        if self.class_config:
            cfg["class_config"] = self.class_config
        cfg["use_negative_scribble"] = self.neg_scribble_cb.isChecked()
        cfg["return_per_class_masks"] = self.per_class_cb.isChecked()
        cfg["seed"] = int(self.base_seed)
        return cfg, missing

    def export_yaml(self, file_path=None):
        import yaml
        if not file_path:
            file_path, _ = QFileDialog.getSaveFileName(self, "Export nnInteractive YAML", "nninteractive.yaml",
                                                       "YAML Files (*.yaml *.yml);;All Files (*)")
            if not file_path:
                return None
        cfg, missing = self.yaml_config()
        with open(file_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        notes = []
        if missing:
            notes.append(f"fill in {', '.join(missing)} (no file behind the layer)")
        if self.roi_select.layer is not None:
            notes.append("the ROI is not part of the YAML (the script runs on the whole image)")
        if len(self.checked_classes()) != len(self.class_ids()):
            notes.append("the script runs all classes, not only the checked ones")
        if self.prompt_mode() == "scribble" and len(self.run_points("scribble")[0]):
            notes.append("manual points are not part of a scribble YAML")
        if notes:
            self.yaml_status.set(f"Exported {Path(file_path).name}; " + "; ".join(notes), "warn")
        else:
            self.yaml_status.set(f"Exported {Path(file_path).name}", "ok")
        return cfg

    def import_yaml(self, file_path=None):
        import yaml
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(self, "Import nnInteractive YAML", "",
                                                       "YAML Files (*.yaml *.yml);;All Files (*)")
            if not file_path:
                return None
        with open(file_path) as f:
            cfg = yaml.safe_load(f) or {}
        notes = []
        unknown = [k for k in cfg if k not in YAML_RULES]
        if unknown:
            notes.append(f"ignored: {', '.join(unknown)}")
        if cfg.get("model_path"):
            self.model_edit.setText(str(cfg["model_path"]))
        dev = str(cfg.get("device", yaml_default("device")) or "")
        self.device_combo.setCurrentIndex(self.device_combo.findData("cpu" if dev.startswith("cpu") else "cuda"))
        pt = cfg.get("prompt_type")
        if pt in ("scribble", "point"):
            if pt == "scribble" and not self.scribble_rb.isEnabled():
                notes.append(f"{self.engine.name} cannot use scribbles: points are used")
            else:
                self.set_prompt_mode(pt)
        elif pt is not None:
            notes.append(f"unknown prompt_type '{pt}'")
        self.npos_spin.setValue(int(cfg.get("default_n_pos", yaml_default("default_n_pos"))))
        self.nneg_spin.setValue(int(cfg.get("default_n_neg", yaml_default("default_n_neg"))))
        meth = cfg.get("default_method", yaml_default("default_method"))
        if self.method_combo.findText(str(meth)) >= 0:
            self.method_combo.setCurrentText(str(meth))
        for cb, key in ((self.neg_bg_cb, "negative_from_bg"), (self.neg_other_cb, "negative_from_other_classes"),
                        (self.neg_each_cb, "negative_per_other_class"),
                        (self.neg_scribble_cb, "use_negative_scribble"),
                        (self.per_class_cb, "return_per_class_masks")):
            cb.setChecked(bool(cfg.get(key, yaml_default(key))))
        self.base_seed = int(cfg.get("seed", yaml_default("seed")) or 0)
        cc = cfg.get("class_config")
        self.class_config = {int(k): dict(v) for k, v in cc.items()} if isinstance(cc, dict) and cc else None
        if self.class_config:
            notes.append(f"class_config for {len(self.class_config)} classes is used by Generate points")
        if cfg.get("do_autozoom") is False:
            notes.append("do_autozoom: false is not supported here (autozoom stays on)")
        if cfg.get("output_folder"):
            self.folder_edit.setText(str(cfg["output_folder"]))
            self.save_cb.setChecked(True)
        # layers whose file is the YAML's path
        for key, combo, kind in (("img_path", self.image_combo, Image), ("seg_path", self.seeds_combo, Labels)):
            p = cfg.get(key)
            if not p:
                continue
            lay = next((l for l in self.viewer.layers if isinstance(l, kind) and _same_path(_source_path(l), p)), None)
            if lay is not None:
                combo.setCurrentText(lay.name)
            else:
                notes.append(f"{key} {Path(str(p)).name} is not open: open it and pick it in 1. Input")
        p = cfg.get("init_seg_path")
        if p:
            lay = next((l for l in self.viewer.layers if isinstance(l, Labels) and _same_path(_source_path(l), p)), None)
            if lay is not None and self.init_combo.isEnabled():
                self.init_combo.setCurrentIndex(self.init_combo.findData(lay.name))
            else:
                notes.append(f"init_seg_path {Path(str(p)).name} is not open: open it and pick it as Initial mask")
        else:
            self.init_combo.setCurrentIndex(0)
        self._on_inputs_changed()
        if notes:
            self.yaml_status.set(f"Imported {Path(file_path).name}; " + "; ".join(notes), "warn")
        else:
            self.yaml_status.set(f"Imported {Path(file_path).name}", "ok")
        return cfg

    # ================================================================ model
    def _browse_model(self):
        d = QFileDialog.getExistingDirectory(self, "Model weights folder", self.model_edit.text())
        if d:
            self.model_edit.setText(d)

    def _on_device_changed(self):
        cpu = self.device_combo.currentData() == "cpu"
        self.cpu_label.setText("very slow: minutes per class" if cpu else "")

    def load_model(self):
        if not self.available:
            self.model_status.set(self._status[1], "error")
            return
        folder = self.model_edit.text().strip()
        if not folder:
            self.model_status.set("Choose the weights folder first", "error")
            return
        device = self.device_combo.currentData()
        self._job["load"] += 1
        job = self._job["load"]
        self._busy["load"] = True
        self.model_status.set("loading…", "info")
        self._update_buttons()

        def done(info):
            if job != self._job["load"]:
                return
            QSettings(*SETTINGS).setValue("prompt/model_dir", folder)
            dev = info.get("device", "?")
            self.model_status.set(f"{self.engine.name} {info.get('version', '?')} on {dev}, "
                                  f"loaded in {info.get('seconds', 0):.1f} s", "info")
            if info.get("license"):
                self.model_status.setToolTip(f"Model license: {info['license']}")
            if self.run_status.text() == LOAD_FIRST:              # "load the model first" is done
                self._set_run_status("")

        def failed(e):
            if job != self._job["load"]:
                return
            self.model_status.set(f"Loading failed: {jobs.reason(e)}", "error")
            self._report_error(e)

        def finished():
            if job == self._job["load"]:
                self._busy["load"] = False
                self._update_buttons()

        self._run_bg(lambda: self.engine.load(folder, device), done, failed, finished)

    def release_model(self):
        if self._busy["run"]:
            self.model_status.set("Stop the run first", "error")
            return
        self.engine.release()
        if self.available:
            self.model_status.set("released", "info")
        self._update_buttons()

    # ================================================================== run
    def output_name(self, roi=None):
        name = f"{self._seeds_name()}_nni"
        return R.roi_name(name, roi.name or "roi") if roi is not None else name

    def class_layer_name(self, out_name, class_id, roi=None):
        """'{seeds}_nni_class_3' ('..._class_3_ROI_1' with an ROI) for an automatic output name."""
        base = out_name
        if roi is not None and roi.name and base.endswith(f"_{roi.name}"):
            return R.roi_name(f"{base[: -len(roi.name) - 1]}_class_{class_id}", roi.name)
        return f"{base}_class_{class_id}"

    def _update_output_name(self):
        if self._out_auto:
            self.output_edit.setText(self.output_name(self._display_roi()) if self.seeds_layer is not None else "")

    def _on_output_edited(self, text):
        self._out_auto = not text.strip()
        if self._out_auto:
            self._update_output_name()

    def current_output_name(self):
        text = self.output_edit.text().strip()
        return text or self.output_name(self._display_roi())

    def _store_key(self, out_name):
        return (id(self.image_layer), id(self.seeds_layer), out_name)

    def _store_for_current_output(self, create=True):
        if self.image_layer is None or self.seeds_layer is None:
            return None
        key = self._store_key(self.current_output_name())
        if create:
            return self._stores.setdefault(key, PS.ResultStore())
        return self._stores.get(key)

    def _set_run_status(self, text, level="info"):
        self.run_status.set(text, level)

    def run_checked(self):
        return self._start_run(self.checked_classes(), "scratch")

    def update_current(self):
        """Refine the highlighted class with its added points (no button in this UI)."""
        if self._current is None:
            self._set_run_status("Select a class first", "error")
            return None
        return self._start_run([self._current], "update")

    def _start_run(self, classes, mode):
        img, seeds = self.image_layer, self.seeds_layer
        if not self.available:
            self._set_run_status(self._status[1], "error")
            return None
        if img is None or seeds is None:
            self._set_run_status("Select an image and seeds first", "error")
            return None
        if not self.engine.loaded:
            self._set_run_status(LOAD_FIRST, "error")
            return None
        if img.ndim != 3:
            self._set_run_status(f"Prompt (AI) needs a 3D image ({self.engine.name} is a 3D model)", "error")
            return None
        if not classes:
            self._set_run_status("Check at least one class", "error")
            return None
        shape = tuple(np.shape(img.data))
        try:
            roi = self.roi_select.roi_for_run(shape)
        except RoiError as e:
            self._set_run_status(str(e), "error")
            return None
        key, ptype, src = self.input_type()
        if not self.engine.supports_prompt_type(ptype):
            self._set_run_status(f"{self.engine.name} does not support this prompt type", "error")
            return None
        init_lay = self.init_layer
        init_data = None
        if init_lay is not None:
            if tuple(np.shape(init_lay.data)) != shape:
                self._set_run_status(f"Initial mask '{init_lay.name}' is {np.shape(init_lay.data)}; image is "
                                     f"{shape}", "error")
                return None
            init_data = np.array(init_lay.data, copy=True)
        save_dir = None
        if self.save_cb.isChecked():
            save_dir = self.folder_edit.text().strip()
            if not save_dir:
                self._set_run_status("Choose the output folder first (or untick 'Save results to folder')", "error")
                return None
        df, n_unused = self.run_points(ptype)
        seeds_snap = np.array(seeds.data, copy=True)
        img_data = np.asarray(img.data)
        crop = PS.inference_crop(shape, roi, ROI_MARGIN)
        if self._out_auto:
            self.output_edit.setText(self.output_name(roi))
        out_name = self.current_output_name()
        store = self._stores.setdefault(self._store_key(out_name), PS.ResultStore())
        image_key = (id(img), img.name, id(img.data))
        stem = Path(_source_path(img)).stem if _source_path(img) else img.name
        if roi is not None:
            stem = R.roi_name(stem, roi.name or "roi")
        ctx = dict(out=out_name, store=store, seeds=seeds_snap, shape=shape, roi=roi, ptype=ptype, input=key,
                   layer_kwargs=self._layer_kwargs(), image=img.name, seeds_name=seeds.name,
                   margin=ROI_MARGIN, base_seed=self.base_seed, mode=mode, classes=list(classes),
                   t0=time.perf_counter(), done=0, per_class=self.per_class_cb.isChecked(),
                   save_dir=save_dir, stem=stem, df=df, unused_points=n_unused, config=self.yaml_config()[0],
                   init=None if init_lay is None else init_lay.name)
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
        self.progress.setFormat("%v / %m classes")
        self._set_run_status(f"Running {len(classes)} classes…")
        self._rebuild_table()
        neg = self.neg_scribble_cb.isChecked() if src == "seed" else False

        def gen():
            yield from PS.run_classes(
                engine, img_data, seeds_snap, df, classes, crop=crop, image_key=image_key, roi=roi,
                mode="update" if mode == "update" else "scratch", store=store, prompt_type=ptype,
                scribbles=None, neg_scribble=neg, base_seed=self.base_seed, cancel=cancel,
                on_point=lambda i, n: prog.changed.emit(i, n), init_masks=init_data)

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
            self._class_status[c] = f"done ({r.seconds:.1f} s{extra})"
            self._write_output(ctx, region)
            if ctx["per_class"]:
                self._write_class_layer(ctx, r)
        nxt = self._next_running(ctx)
        if nxt is not None:
            self._class_status[nxt] = "running"
        self.progress.setValue(ctx["done"])
        self._rebuild_table()

    def _on_run_error(self, e, ctx):
        for c in ctx["classes"][ctx["done"]:]:
            self._class_status[c] = "stopped" if isinstance(e, Cancelled) else "failed"
        if isinstance(e, Cancelled):
            self._set_run_status(f"Stopped after {ctx['done']} of {len(ctx['classes'])} classes"
                                 + ("; nothing saved" if ctx["save_dir"] else ""), "warn")
        elif is_oom(e):
            self._set_run_status(str(e), "error")
            show_error(f"Prompt (AI): {e}")
        else:
            self._set_run_status(f"Run failed: {jobs.reason(e)}", "error")
            self._report_error(e)
        ctx["error"] = e

    def _on_run_finished(self, ctx):
        self._busy["run"] = False
        if "error" not in ctx:
            msg = (f"Done: {ctx['done']} classes in {time.perf_counter() - ctx['t0']:.1f} s"
                   + (f"; {ctx['store'].overlap_voxels():,} voxels claimed by more than one class (smaller id "
                      f"wins)" if len(ctx["store"]) > 1 else "")
                   + (f"; {ctx['unused_points']} generated points not used (scribble mode uses manual "
                      f"points only)" if ctx.get("unused_points") else ""))
            level = "ok"
            if ctx["save_dir"]:
                try:
                    files = self.save_results(ctx)
                    msg += f"; saved {len(files)} files to {ctx['save_dir']}"
                except Exception as e:
                    msg += f"; saving failed: {type(e).__name__}: {e}"
                    level = "error"
            self._set_run_status(msg, level)
        self._rebuild_table()
        self._update_buttons()

    def stop(self):
        if self._cancel is not None:
            self._cancel.set()

    def revert_current(self):
        """Back to the highlighted class's previous result (no button in this UI)."""
        store = self._store_for_current_output(create=False)
        if store is None or self._current is None or not store.can_revert(self._current):
            self._set_run_status("Nothing to revert for this class", "error")
            return
        region = store.revert(self._current)
        ctx = self._ctx if self._ctx is not None and self._ctx["store"] is store else None
        self._class_status[self._current] = "reverted"
        if ctx is not None:
            self._write_output(ctx, region)
        self._rebuild_table()

    # --------------------------------------------------------------- output
    def _write_output(self, ctx, region):
        store, shape = ctx["store"], ctx["shape"]
        rule = OVERLAP_RULE
        ids = store.classes()
        dtype = layer_dtype(max(ids) if ids else 0)
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
            "prompt_type": ctx["ptype"], "initial_mask": ctx["init"], "overlap_rule": rule,
            "roi_margin": ctx["margin"], "base_seed": ctx["base_seed"], "config": ctx["config"],
            "classes": {int(c): {"mode": r.mode, "voxels": r.n_voxels, "points": r.n_points,
                                 "points_dropped": r.n_dropped, "seconds": r.seconds,
                                 "prompt_type": r.prompt_type}
                        for c, r in ((c, store.get(c)) for c in ids)},
        }
        if ctx["roi"] is not None:
            meta[R.META_KEY] = ctx["roi"].to_metadata()
        lay.metadata = meta
        self.output_updated.emit(ctx["out"])

    def _write_class_layer(self, ctx, r):
        """One Labels layer per class (its own mask, before overlaps; value = class id)."""
        name = self.class_layer_name(ctx["out"], r.class_id, ctx["roi"])
        dtype = layer_dtype(r.class_id)
        data = np.zeros(ctx["shape"], dtype)
        data[r.slices][r.mask] = r.class_id
        lay = self._layer(name, Labels)
        if lay is not None and tuple(lay.data.shape) == ctx["shape"] and lay.data.dtype == dtype:
            lay.data = data
        else:
            if lay is not None:
                self.viewer.layers.remove(name)
            lay = self.viewer.add_labels(data, name=name, **ctx["layer_kwargs"])
        lay.metadata = {**lay.metadata, META_KEY: {"role": "class", "class_id": int(r.class_id),
                                                    "output": ctx["out"]}}
        if ctx["roi"] is not None:
            lay.metadata = {**lay.metadata, R.META_KEY: ctx["roi"].to_metadata()}

    def save_results(self, ctx=None):
        """Write the pipeline's files for the current output (as nninter_main): returns the paths."""
        import tifffile
        import yaml
        ctx = ctx or self._ctx
        folder = Path(ctx["save_dir"])
        folder.mkdir(parents=True, exist_ok=True)
        store, stem = ctx["store"], ctx["stem"]
        ids = store.classes()
        total = store.compose(ctx["shape"], dtype=PS._label_dtype(max(ids) if ids else 0), rule=OVERLAP_RULE,
                              seeds=ctx["seeds"])
        files = []
        p = folder / f"{stem}_segmentation.tif"
        tifffile.imwrite(p, total, compression="zlib")
        files.append(p)
        if ctx["per_class"]:
            full = tuple(slice(0, n) for n in ctx["shape"])
            for c in ids:
                p = folder / f"{stem}_class_{c}.tif"
                tifffile.imwrite(p, store.get(c).full_mask_in(full).astype(np.uint8) * 255, compression="zlib")
                files.append(p)
        cfg = dict(ctx["config"])
        cfg["output_folder"] = str(folder)
        p = folder / "used_config.yaml"
        with open(p, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        files.append(p)
        if ctx["ptype"] == "points":
            p = folder / f"{stem}_point_prompts.csv"
            ctx["df"].to_csv(p, index=False)
            files.append(p)
        return files
