"""Widget for seed generation in SPROUT workflow."""

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox, QCheckBox,
    QSlider, QFormLayout, QMessageBox, QLineEdit, QFileDialog,
    QScrollArea, QSizePolicy, QProgressBar
)
from qtpy.QtCore import QObject, Qt, Signal
import numpy as np
from typing import Optional
from napari.layers import Image, Labels
from napari.utils.notifications import show_info, show_error

from ..utils.util_widget import (SWEEP_EROSION, SWEEP_METHODS, SWEEP_THRESHOLDS, as_list, create_output_folder_row, SeedOptionalParamGroupBox,
                                 MainSeedParamWidget, ThresholdWidget,apply_threshold_preview)
from ..utils.histogram_threshold_widget import PREVIEW_NAME, HistogramThresholdWidget, append_threshold_row
from ..utils.boundary_check import boundary_ok
from ..utils.layer_files import layer_name, saved_note, yaml_path
from ..utils.roi_widget import RoiSelect
from ..utils.status import RUN_BUTTON, STOP_BUTTON, StatusLine, follow_theme
from ..utils import jobs

import yaml
import sys
import os
import time
import weakref
# get the current file's absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
    
from make_seeds import make_seeds
from make_adaptive_seed import make_adaptive_seed_thre, make_adaptive_seed_ero
from make_sweep_adaptive_seed import make_sweep_adaptive_seed
from sprout_core import roi as R
from sprout_core.roi import RoiError

class SeedWorker:
    """The parameters of one Seeds run; run(cancel, progress) is the job (utils/jobs.py)."""

    def __init__(self, seed_mode, img, boundary, thresholds, upper_thresholds,
                 erosion_steps, segments, output_folder, num_threads,
                    footprints=None, base_name="seeds",
                    sort=True, no_split_max_iter=3,
                    min_size=5, min_split_ratio=0.01, min_split_total_ratio=0,
                    save_every_iter=True, init_segments=None,last_segments=None,
                    split_size_limit=(None, None), split_convex_hull_limit=(None, None),
                    roi=None
                    ):
        self.roi = roi   # sprout_core.roi.Roi or None (whole image)
        ## Parameters for both original and adaptive seed making
        self.seed_mode = seed_mode  # "Original", "Adaptive (Erosion)", "Adaptive (Thresholds)", or one of SWEEP_METHODS
        self.img = img
        self.boundary = boundary
        self.thresholds = thresholds
        self.upper_thresholds = upper_thresholds
        self.erosion_steps = erosion_steps
        self.segments = segments
        self.output_folder = output_folder
        self.num_threads = num_threads
        self.footprints = footprints
        self.base_name = base_name
        
        ## parameters for adaptive seed making
        self.sort = sort
        self.no_split_max_iter = no_split_max_iter
        self.min_split_ratio = min_split_ratio
        self.min_split_total_ratio = min_split_total_ratio
        
        self.min_size = min_size
        self.save_every_iter = save_every_iter
        self.init_segments = init_segments
        self.last_segments = last_segments
        self.split_size_limit = split_size_limit
        self.split_convex_hull_limit = split_convex_hull_limit
        
        
    @property
    def folder(self):
        """Where the run writes its files."""
        return os.path.join(self.output_folder, self.base_name)

    def run(self, cancel, progress):
        """Runs on the job's worker thread; returns seeds_dict. On Stop, Cancelled carries
        the number of files written (files_written)."""
        watch = jobs.FolderWatch(self.folder)
        try:
            if self.seed_mode not in ["Original", "Adaptive (Erosion)", "Adaptive (Thresholds)", *SWEEP_METHODS]:
                raise ValueError("Invalid seed mode selected.")
            if self.roi is not None:
                # ROI as a wall: run on its bbox + 1, outside the ROI counts as boundary
                self.img, self.boundary, _ = R.wall_inputs(self.roi, self.img, self.boundary)
            if self.seed_mode == "Original":
                # Call the original seed making function
                seeds_dict, _ = make_seeds(
                    img=self.img,
                    boundary=self.boundary,
                    thresholds=self.thresholds,
                    upper_thresholds=self.upper_thresholds,
                    erosion_steps=self.erosion_steps,
                    segments=self.segments,
                    output_folder=self.output_folder,
                    num_threads=self.num_threads,
                    footprints=self.footprints,
                    base_name=self.base_name,

                    return_for_napari=True,
                    cancel=cancel, progress=progress
                )
            elif self.seed_mode == "Adaptive (Erosion)":

                seeds_dict, _, _ = make_adaptive_seed_ero(
                    img=self.img,
                    boundary=self.boundary,
                    threshold=self.thresholds,
                    upper_threshold=self.upper_thresholds,
                    erosion_steps=self.erosion_steps,
                    segments=self.segments,
                    output_folder=self.output_folder,
                    num_threads=self.num_threads,
                    footprints=self.footprints,
                    
                    base_name=self.base_name,
                                        
                    sort=self.sort,
                    no_split_max_iter=self.no_split_max_iter,
                    min_split_ratio=self.min_split_ratio,
                    min_split_total_ratio=self.min_split_total_ratio,
                    min_size=self.min_size,
                    save_every_iter=self.save_every_iter,
                    init_segments=self.init_segments,
                    last_segments=self.last_segments,
                    split_size_limit=self.split_size_limit,
                    split_convex_hull_limit=self.split_convex_hull_limit,


                    return_for_napari=True,
                    cancel=cancel, progress=progress
                )
            elif self.seed_mode == "Adaptive (Thresholds)":
                seeds_dict, _, _ = make_adaptive_seed_thre(
                    img=self.img,
                    boundary=self.boundary,
                    thresholds=self.thresholds,
                    upper_thresholds=self.upper_thresholds,
                    erosion_steps=self.erosion_steps,
                    segments=self.segments,
                    output_folder=self.output_folder,
                    num_threads=self.num_threads,
                    footprints=self.footprints,
                    
                    base_name=self.base_name,
                                        
                    sort=self.sort,
                    no_split_max_iter=self.no_split_max_iter,
                    min_size=self.min_size,
                    min_split_ratio=self.min_split_ratio,
                    min_split_total_ratio=self.min_split_total_ratio,
                    save_every_iter=self.save_every_iter,
                    init_segments=self.init_segments,
                    last_segments=self.last_segments,
                    split_size_limit=self.split_size_limit,
                    split_convex_hull_limit=self.split_convex_hull_limit,

                    return_for_napari=True,
                    cancel=cancel, progress=progress
                )
            elif self.seed_mode in SWEEP_METHODS:
                seeds_dict = make_sweep_adaptive_seed(
                    img=self.img,
                    boundary=self.boundary,
                    thresholds=self.thresholds,
                    upper_thresholds=self.upper_thresholds,
                    erosion_steps=self.erosion_steps,
                    segments=self.segments,
                    output_folder=self.output_folder,
                    num_threads=self.num_threads,
                    footprints=self.footprints,
                    base_name=self.base_name,
                    min_size=self.min_size,
                    cancel=cancel, progress=progress
                )
            jobs.check_cancel(cancel)         # Stop after the last step: no layers either
            if self.roi is not None:
                seeds_dict = R.paste_seeds(self.roi, seeds_dict)
                R.write_sidecar(self.roi, os.path.join(self.output_folder, self.base_name))
            return seeds_dict
        except jobs.Cancelled as e:
            e.files_written = watch.written()
            raise

    def stopped(self, e, done, total):
        """The orange line after Stop."""
        unit = {"Original": "seeds", "Adaptive (Thresholds)": "thresholds"}.get(self.seed_mode, "steps")
        return jobs.stopped_text(done, total, unit, getattr(e, "files_written", None))


class SeedRuns(QObject):
    """Per viewer: a finished Seeds run tells the Grow tab the last seeds layer it added."""
    finished = Signal(str)


_SEED_RUNS = weakref.WeakKeyDictionary()     # viewer -> SeedRuns


def seed_runs(viewer):
    runs = _SEED_RUNS.get(viewer)
    if runs is None:
        runs = _SEED_RUNS[viewer] = SeedRuns()
    return runs


def input_combo():
    """A layer combo for the Input group: long layer names never widen the panel."""
    c = QComboBox()
    c.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    c.setMinimumContentsLength(10)
    return c


def combo_items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def put_in_tab_order(first, *widgets):
    """Keyboard Tab reaches `widgets` (in this order) right before `first`, e.g. the
    Import / Export YAML buttons of a group header before the group's first parameter."""
    prev = first.previousInFocusChain()
    while prev.focusPolicy() == Qt.NoFocus and prev is not first:   # setTabOrder ignores those
        prev = prev.previousInFocusChain()
    for w in widgets:
        QWidget.setTabOrder(prev, w)
        prev = w


def rename_in_combos(layer, old, combos):
    """A layer was renamed in napari: rename its item in place (the selection stays)."""
    for combo in combos:
        i = combo.findText(old)
        if i >= 0 and old != "None":
            combo.blockSignals(True)
            combo.setItemText(i, layer.name)
            combo.blockSignals(False)


def set_min_size_policy_recursive(widget, min_width=40, max_width=300):
    for w in widget.findChildren((QPushButton, QLineEdit)):
        w.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        w.setMinimumWidth(min_width)
        w.setMaximumWidth(max_width)

def set_compact(widget, minw=None, maxw=None):
    widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
    if minw is not None:
        widget.setMinimumWidth(minw)
    if maxw is not None:
        widget.setMaximumWidth(maxw)


class SeedGenerationWidget(QWidget):
    """Widget for interactive seed generation."""
    
    seeds_generated = Signal(np.ndarray, list)  # seeds array, sizes
    
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        follow_theme(napari_viewer, self)       # the info grey of napari's theme

        
        self.worker = None
        
        self.previous_image_name = None
        
        self.current_image = None
        self.current_boundary = None
        self.preview_layer = None
        self.seeds_layer = None
        self._run_roi = None             # ROI and image scale / translate of the running job
        self._run_layer_kwargs = None
        
        self._init_ui()
        self._connect_signals()
        self.refresh_layers()   # pick up layers that were loaded before the panel opened

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # Input
        image_group = QGroupBox("Input")
        image_layout = QFormLayout()

        self.image_combo = input_combo()
        self.boundary_combo = input_combo()
        self.boundary_combo.addItem("None")

        image_layout.addRow("Image", self.image_combo)
        image_layout.addRow("Boundary", self.boundary_combo)
        # a boundary that cannot be used: red line under the combo, the run is refused
        self.boundary_error = StatusLine(hide_empty=True)
        image_layout.addRow(self.boundary_error)
        self.roi_select = RoiSelect(
            self.viewer, lambda: None if self.current_image is None else np.shape(self.current_image), "Seeds")
        image_layout.addRow("ROI", self.roi_select)

        image_group.setLayout(image_layout)
        layout.addWidget(image_group)



        # --- Parameters for Original and Adaptive (Thresholds) methods ---
        
        ## for list params, currently not used
        self.list_params_group = QGroupBox("Threshold List Parameters")
        self.list_params_group.setFocusPolicy(Qt.NoFocus) # Allow children to get focus
        list_params_layout = QFormLayout()
        self.lower_thresholds_list = QLineEdit("130, 140, 150")
        self.upper_thresholds_list = QLineEdit("255, 255, 255")

        # Style to make QLineEdit look editable, text visible, and ensure cursor blinks.
        style = """
            QLineEdit {
                background-color: white;
                border: 1px solid #999;
                color: black;
                padding: 2px;
            }
            QLineEdit:focus {
                border: 1px solid #0078d7; /* Blue border on focus */
            }
        """
        self.lower_thresholds_list.setStyleSheet(style)
        self.upper_thresholds_list.setStyleSheet(style)

        # Add tooltips for clarity
        self.lower_thresholds_list.setToolTip("Enter comma-separated integer values for lower thresholds.")
        self.upper_thresholds_list.setToolTip("Enter comma-separated integer values for upper thresholds.")

        self.use_upper_list = QCheckBox("Use upper thresholds list")
        self.use_upper_list.setChecked(True)
        list_params_layout.addRow("Lower Thresholds (comma-separated):", self.lower_thresholds_list)
        list_params_layout.addRow(self.use_upper_list)
        list_params_layout.addRow("Upper Thresholds (comma-separated):", self.upper_thresholds_list)
        self.list_params_group.setLayout(list_params_layout)
        # layout.addWidget(self.list_params_group)
        ###
        
        
        self.threshold_widget = HistogramThresholdWidget(self.viewer, add_connected_components=True,
                                                         roi_use=self.roi_select,
                                                         preview_name=PREVIEW_NAME + " (Seeds)")
        self.threshold_widget.preview_requested.connect(self.preview_threshold)
        layout.addWidget(self.threshold_widget)

        # --- Main parameters widget ---
        self.main_param_widget = MainSeedParamWidget(title="Parameters",
                                                 image_combo=self.image_combo,viewer=self.viewer)
        self.threshold_widget.add_to_table_requested.connect(
            lambda lo, hi: append_threshold_row(self.main_param_widget, lo, hi))


        layout.addWidget(self.main_param_widget)


        # Import / Export YAML: in the Parameters header row; their messages in its status line
        self.import_btn = QPushButton("Import YAML")
        self.export_btn = QPushButton("Export YAML")
        self.main_param_widget.add_header_widget(self.import_btn)
        self.main_param_widget.add_header_widget(self.export_btn)
        self.params_status = self.main_param_widget.status_line

        # advanced adaptive seed options, at the bottom of Parameters
        self.show_advanced_checkbox = QCheckBox("Show advanced options")
        self.show_advanced_checkbox.setToolTip("Advanced options of the adaptive seed methods")
        self.show_advanced_checkbox.stateChanged.connect(self._toggle_seed_params_visibility)
        self.advanced_params_box = SeedOptionalParamGroupBox()
        self.advanced_params_box.setVisible(False)
        self.main_param_widget.add_bottom_widget(self.show_advanced_checkbox)
        self.main_param_widget.add_bottom_widget(self.advanced_params_box)

        # Output
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        output_dir_layout, self.output_folder_line = create_output_folder_row(label="Folder")
        self.save_every_iter_checkbox = QCheckBox("Save mid results")
        self.save_every_iter_checkbox.setToolTip("Adaptive methods: also save the seed of every iteration "
                                                 "(written to the folder and added as layers)")
        self.save_every_iter_checkbox.setChecked(False)
        output_layout.addLayout(output_dir_layout)
        output_layout.addWidget(self.save_every_iter_checkbox)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Progress bar (shown while running)
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Execute buttons        # Execute buttons
        btn_layout = QHBoxLayout()
        # Generate button
        self.generate_btn = QPushButton("Generate Seeds")
        self.generate_btn.setStyleSheet(RUN_BUTTON)
        # layout.addWidget(self.generate_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(STOP_BUTTON)
        self.stop_btn.setToolTip("Stop generating seeds (files already written stay in the folder)")
        self.stop_btn.setEnabled(False)        
        

        btn_layout.addWidget(self.generate_btn)
        btn_layout.addWidget(self.stop_btn)
        self.roi_select.set_run_button(self.generate_btn, "Generate Seeds")
        # btn_layout.addWidget(self.save_btn)
        layout.addLayout(btn_layout)
        
        # Status: one line, info / ok / warn / error (utils/status.py)
        self.results_label = StatusLine("No seeds generated yet", "info")
        self.status_label = self.results_label          # same name as in the Grow tab
        layout.addWidget(self.results_label)
        layout.addStretch()
        # start / Stop / progress / end messages of a run (utils/jobs.py)
        self.run_control = jobs.RunControl(self, self.results_label, [self.generate_btn], self.stop_btn,
                                           self.progress_bar, notify=self._notify)
        
        
        main_scroll_container = QWidget()
        main_scroll_container.setLayout(layout)


        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(main_scroll_container)
        
        final_layout = QVBoxLayout()
        final_layout.addWidget(scroll)
        self.setLayout(final_layout)
        
        
        for w in [self.generate_btn, self.stop_btn]:
            set_compact(w, minw=100)

        # the Input combos follow napari's layers by themselves (added / removed / renamed),
        # through the viewer's hub: a closed panel leaves no napari callbacks behind
        hub = self.roi_select.hub
        hub.layer_inserted.connect(self._on_layers_changed)
        hub.layer_removed.connect(self._on_layers_changed)
        hub.layer_renamed.connect(self._on_layer_renamed)
        # Tab reaches Import / Export YAML where they are shown: before the parameters
        put_in_tab_order(self.main_param_widget.seed_method_combo, self.import_btn, self.export_btn)
        self.main_param_widget.seed_method_combo.currentTextChanged.connect(self._on_method_changed)

    def _on_method_changed(self, method):
        """Grey what the method does not use."""
        self.advanced_params_box.set_method(method)
        self.save_every_iter_checkbox.setEnabled(method not in SWEEP_METHODS)
        self.main_param_widget.thread_spin.setEnabled(method != SWEEP_EROSION)     # an erosion sweep is sequential
        self.main_param_widget.segments_spin.setToolTip(
            "Number of largest connected components to keep in each step; the result can have more seeds"
            if method in SWEEP_METHODS else "Number of largest connected components to keep in each seed")

    def _sweep_rows_problem(self, method, rows, erosion_steps):
        """Why the threshold table does not fit the sweep, or None."""
        if method == SWEEP_EROSION and rows != 1:
            return f"{SWEEP_EROSION} uses one threshold row (there are {rows}); for several, use {SWEEP_THRESHOLDS}"
        if method == SWEEP_EROSION and erosion_steps < 1:
            return f"{SWEEP_EROSION} needs at least 1 erosion step"
        if method == SWEEP_THRESHOLDS and rows < 2:
            return f"{SWEEP_THRESHOLDS} needs at least 2 threshold rows; for one row, use {SWEEP_EROSION}"
        return None

    def showEvent(self, event):
        super().showEvent(event)
        if not self._combos_in_sync():         # a missed layer event: re-sync, keeping the selection
            self._on_layers_changed()

    def _combos_in_sync(self):
        imgs = [l.name for l in self.viewer.layers if isinstance(l, Image)]
        labs = [l.name for l in self.viewer.layers if isinstance(l, Labels)]
        return combo_items(self.image_combo) == imgs and combo_items(self.boundary_combo) == ["None"] + labs

    def _toggle_seed_params_visibility(self, state: int):
        """Toggle visibility of advanced seed parameters."""
        self.advanced_params_box.setVisible(state == Qt.Checked)  

    def _connect_signals(self):
        """Connect widget signals."""
        # TODO to be delete if thresholdwidget works
        # self.use_upper.toggled.connect(self.upper_threshold.setEnabled)
        # self.lower_threshold.valueChanged.connect(lambda v: self.lower_slider.setValue(int(v)))
        # self.lower_slider.valueChanged.connect(lambda v: self.lower_threshold.setValue(float(v)))
        # self.preview_threshold_btn.clicked.connect(self.preview_threshold)
        self.generate_btn.clicked.connect(self.generate_seeds)
        self.stop_btn.clicked.connect(self.stop_generation)
        
        self.image_combo.currentIndexChanged.connect(self._on_image_changed)

        self.use_upper_list.toggled.connect(self.upper_thresholds_list.setEnabled)
        
        self.import_btn.clicked.connect(lambda: self.import_yaml())
        self.export_btn.clicked.connect(lambda: self.export_yaml())
        self.boundary_combo.currentTextChanged.connect(self._on_boundary_changed)
        
        # self.browse_folder_btn.clicked.connect(self._browse_output_folder)

    def _browse_output_folder(self):
        """Open a dialog to select an output folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder_edit.setText(folder)


    def _on_layers_changed(self, *_):
        self._refreshing = True
        try:
            self.refresh_layers()
        finally:
            self._refreshing = False
        self._on_boundary_changed(self.boundary_combo.currentText())

    def _on_boundary_changed(self, text):
        if getattr(self, "_refreshing", False):              # combo refilled: same selection
            return
        if text != getattr(self, "_bad_boundary", None):    # another mask: its red line goes
            self.boundary_error.clear()

    def _on_layer_renamed(self, layer, old):
        """Rename in place: the selection (and the threshold table) stay as they are."""
        rename_in_combos(layer, old, (self.image_combo, self.boundary_combo))
        if self.previous_image_name == old and isinstance(layer, Image):
            self.previous_image_name = layer.name

    def refresh_layers(self):
        """Refresh the list of available layers."""
        current_image_name = self.image_combo.currentText()
        current_boundary_name = self.boundary_combo.currentText()
        self.image_combo.blockSignals(True)  # prevent triggering img_changed temporarily

        self.image_combo.clear()
        self.boundary_combo.clear()
        self.boundary_combo.addItem("None")

        image_names = [layer.name for layer in self.viewer.layers if isinstance(layer, Image)]
        label_names = [layer.name for layer in self.viewer.layers if isinstance(layer, Labels)]
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                self.image_combo.addItem(layer.name)
            elif isinstance(layer, Labels):
                self.boundary_combo.addItem(layer.name)

        # Restore image selection if it still exists
        if current_image_name in image_names:
            self.image_combo.setCurrentText(current_image_name)
        # Restore the boundary selection too: without this, adding any layer
        # (e.g. the live Threshold Preview) silently reset it to "None"
        if current_boundary_name in label_names:
            self.boundary_combo.setCurrentText(current_boundary_name)
        self.image_combo.blockSignals(False)

        # If current_text changed (e.g., setCurrentText fails), manually trigger img_changed
        if self.image_combo.currentText() != self.previous_image_name:
            self._on_image_changed(self.image_combo.currentText())
    
    
    def _on_image_changed(self , _):
        """Handle image selection change."""
        text = self.image_combo.currentText()
        if text and text != self.previous_image_name:
            had_image = self.previous_image_name is not None
            self.previous_image_name = text 
            changed = self.main_param_widget.new_image()     # thresholds reset, footprints kept
            if had_image:
                note = "Thresholds reset for the new image"
                if changed:
                    note += f"; {changed} footprint{'s' if changed > 1 else ''} changed to fit it"
                self.params_status.set(note, "info")

            layer = self.viewer.layers[self.image_combo.currentText()]
            if isinstance(layer, Image):
                self.current_image = layer.data
                self.threshold_widget.set_image(layer)
                self.roi_select.refresh()
       

    
    def preview_threshold(self, lower=None, upper=None):
        if self.current_image is None:
            show_error("Select an image first")
            return

        try:
            if lower is None:
                lower, upper = self.threshold_widget.get_thresholds()

            self.threshold_widget.run_preview_in_thread(self.current_image, callback=self._on_preview_result)


            # binary = self.threshold_widget.apply_preview(self.current_image)


            # # delete existing preview layer if it exists
            # if self.preview_layer and self.preview_layer in self.viewer.layers:
            #     self.viewer.layers.remove(self.preview_layer)
            #     self.preview_layer = None

            # # delete existing seeds layer if it exists
            # self.preview_layer = self.viewer.add_labels(
            #     binary.astype(np.uint8),
            #     name="Threshold Preview",
            #     opacity=0.5,
            # )


            # show_info(f"Preview updated: {np.sum(binary)} pixels selected")
        except Exception as e:
            show_error(f"Error in preview: {e}")
    
    def _on_preview_result(self, binary):
        """Handle the result of the threshold preview."""
        # delete existing preview layer if it exists
        if self.preview_layer and self.preview_layer in self.viewer.layers:
            self.viewer.layers.remove(self.preview_layer)
            self.preview_layer = None

        # delete existing seeds layer if it exists
        self.preview_layer = self.viewer.add_labels(
            binary.astype(np.uint8),
            name=PREVIEW_NAME + " (Seeds)",
            opacity=0.5,
        )
        show_info(f"Preview updated: {np.sum(binary)} pixels selected")

        self.threshold_widget.preview_btn.setEnabled(True)  # Re-enable button after processing
    
        
    def add_labels_layer(self, seeds_dict, roi=None, layer_kwargs=None):
        for seed_name, seed in seeds_dict.items():
            kwargs = dict(layer_kwargs or {})   # the image's scale / translate
            if roi is not None:
                seed_name = R.roi_name(seed_name, roi.name or "roi")
                kwargs["metadata"] = {R.META_KEY: roi.to_metadata()}
            print(f"Adding seed layer: {seed_name}")
            self.seeds_layer = self.viewer.add_labels(
                seed,
                name=seed_name,
                **kwargs
            )
    
    def generate_seeds(self):
        """Generate seeds with current parameters."""
        if self.current_image is None:
            self.results_label.set("Select an image first", "error")
            return
        if self.run_control.running:
            return
        current_name = self.image_combo.currentText()
        
        try:
        

            
            # Get boundary if selected
            boundary = None
            if self.boundary_combo.currentText() != "None":
                boundary_layer = self.viewer.layers[self.boundary_combo.currentText()]
                if isinstance(boundary_layer, Labels):
                    boundary = boundary_layer.data
            if not boundary_ok(self, self.boundary_combo.currentText(), boundary, self.boundary_error):
                self._bad_boundary = self.boundary_combo.currentText()
                return
            try:
                roi = self.roi_select.roi_for_run(np.shape(self.current_image))
            except RoiError as e:
                self.results_label.set(str(e), "error")
                return
            image_layer = self.viewer.layers[current_name]
            self._run_roi = roi
            self._run_layer_kwargs = {"scale": image_layer.scale, "translate": image_layer.translate}

            output_folder = self.output_folder_line.text()
            folder_note = ""
            if not output_folder:
                folder_note = "no output folder given, writing to 'napari_temp'"
                output_folder = 'napari_temp'
            
            # TODO: @ioannouE
            # I have added interfaces for seed making they are
            # make_seeds that does orignal seed making
            # make_adaptive_seed_thre that does adaptive seed on a list of thresholds/ upper thresholds
            # make_adaptive_seed_ero that does adaptive seed making with erosions
            # So probably make a selection widget that allows user to select which method to use
            
            # They return seeds_dict for {name: seed}, and will be added to labels
            # NOTE: maybe add UI for selecting the output folder
            # NOTE: need to add list of thresholds for adaptive seed making
            
            main_params = self.main_param_widget.get_params()
            selected_method = main_params['seed_method']
            problem = self._sweep_rows_problem(selected_method, len(main_params['thresholds']),
                                               main_params['erosion_steps'])
            if problem:
                self.results_label.set(problem, "error")
                return
            print(f"Generating seeds with method: {selected_method}")
            
            
            advance_params  = self.advanced_params_box.get_params()
            
            
            worker = SeedWorker(
                seed_mode=selected_method,
                img=self.current_image,
                boundary=boundary,
                
                thresholds=main_params['thresholds'],
                upper_thresholds=main_params['upper_thresholds'],
                erosion_steps=main_params['erosion_steps'],
                num_threads= main_params['num_threads'],
                
                segments=main_params['segments'],
                output_folder=output_folder,
                
                footprints=main_params['footprints'],
                base_name= R.roi_name(current_name, roi.name or "roi") if roi is not None else current_name,
                
                sort=advance_params['sort'],
                no_split_max_iter=advance_params['no_split_max_iter'],
                min_size=advance_params['min_size'],
                min_split_ratio=advance_params['min_split_ratio'],
                min_split_total_ratio=advance_params['min_split_total_ratio'],
                save_every_iter=self.save_every_iter_checkbox.isChecked(),
                init_segments=None,  # Add if needed
                last_segments=None,  # Add if needed
                split_size_limit = advance_params['split_size_limit'],
                split_convex_hull_limit = advance_params['split_convex_hull_limit'],
                roi=roi
            )
            self.worker = self.run_control.start(
                worker.run, "Seed generation",
                f"Generating seeds… ({folder_note})" if folder_note else "Generating seeds…",
                done=self._on_seed_finished, stopped=worker.stopped,
                failed=lambda e: ("ROI: " if roi is not None else "") + jobs.reason(e),
                level="warn" if folder_note else "info")
            # self.viewer.window.statusBar().showMessage(
            #     f"Generating seeds using {selected_method} method...")
            
            ##TODO, as we got worker ready, we comment below code
            ## To be delete
            
            # seeds_dict = {}
            # if selected_method == "Original_ignore":
            #     seeds_dict, _ = make_seeds(
            #         img=self.current_image,
            #         boundary=boundary,
            #         thresholds=thresholds,
            #         upper_thresholds=upper_thresholds,
            #         erosion_steps=erosion,
            #         segments=segments,
            #         output_folder=output_folder,
            #         num_threads=4,
            #         footprints=None,
            #         base_name="to_get_name",
            #         return_for_napari=True
            #     )
            # elif selected_method == "Adaptive (Erosion)_ignore":
            # seeds_dict, _, _ = make_adaptive_seed_ero(
            #     img=self.current_image,
            #     boundary=boundary,
            #     threshold=main_params['thresholds'],
            #     upper_threshold=main_params['upper_thresholds'],
            #     erosion_steps=main_params['erosion_steps'],
            #     num_threads= main_params['num_threads'],
                
            #     segments=main_params['segments'],
            #     output_folder=output_folder,

            #     footprints=None,
            #     sort=True,
            #     no_split_max_iter=3,
            #     min_size=5,
            #     min_split_ratio=0.01,
            #     min_split_total_ratio=0,
            #     save_every_iter=True,
            #     init_segments=None,
            #     split_size_limit=(None, None),
            #     split_convex_hull_limit=(None, None),
            #     return_for_napari=True
            # )
            # elif selected_method == "Adaptive (Thresholds)_ignore":
            #     seeds_dict, _, _ = make_adaptive_seed_thre(
            #         img=self.current_image,
            #         boundary=boundary,
            #         thresholds=thresholds,
            #         upper_thresholds=upper_thresholds,
            #         erosion_steps=erosion,
            #         segments=segments,
            #         output_folder=output_folder,
            #         num_threads=4,
            #         footprints=None,
            #         sort=True,
            #         no_split_max_iter=3,
            #         min_size=5,
            #         min_split_ratio=0.01,
            #         min_split_total_ratio=0,
            #         save_every_iter=True,
            #         init_segments=None,
            #         split_size_limit=(None, None),
            #         split_convex_hull_limit=(None, None),
            #         return_for_napari=True
            #     )

            # if not seeds_dict:
            #     show_error("Seed generation returned no seeds.")
            #     return

            # self.add_labels_layer(seeds_dict)
            
            # self.results_label.setText(f"Generated seeds using {selected_method}")
            
            # if seeds_dict:
            #     first_seed_name = list(seeds_dict.keys())[0]
            #     seeds = seeds_dict[first_seed_name]
            #     sizes = [np.sum(seeds == i) for i in np.unique(seeds) if i != 0]
            #     self.seeds_generated.emit(seeds, sizes)
            #     show_info(f"Successfully generated {len(sizes)} seeds using {selected_method}")

        except Exception as e:
            show_error(f"Seed generation failed: {jobs.reason(e)}")
            self.results_label.set(f"Seed generation failed: {jobs.reason(e)}", "error")

    @staticmethod
    def _notify(level, text):
        """The one napari notification of a run (finished / failed)."""
        (show_error if level == "error" else show_info)(text)

    def _on_seed_finished(self, seeds_dict, secs):
        """Handle seed generation completion (UI thread): add the layers, return the summary."""
        # Remove any existing preview layer
        # if self.preview_layer is not None:
        #     self.viewer.layers.remove(self.preview_layer)
        #     self.preview_layer = None
        
        # Add the generated seeds to the viewer
        self.add_labels_layer(seeds_dict, roi=self._run_roi, layer_kwargs=self._run_layer_kwargs)
        if seeds_dict:              # the Grow tab picks the last one
            seed_runs(self.viewer).finished.emit(self.seeds_layer.name)
        
        # Summary for the status line and the notification ("✓ ... (2.5 s)")
        n = len(seeds_dict)
        return f"Seeds generated: {n} layer{'' if n == 1 else 's'}"

    def stop_generation(self):
        """Stop the current seed generation at its next step (cooperative; no partial layers)."""
        self.run_control.stop()
            

    # TODO to implement import/export yaml for seed parameters
    # This is a placeholder for the import/export functionality.

    # TODO: need to consider threshold as a list
    # how to set threshold in the widget?
    def import_yaml(self, file_path=None):
        if file_path is None:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open YAML File", "", "YAML Files (*.yaml *.yml);;All Files (*)"
            )
        if file_path:
            notes = []
            try:
                with open(file_path, "r") as f:
                    params = yaml.safe_load(f)

                # Map flat YAML keys to main and advanced parameter dictionaries
                main_params_to_set = {}
                advanced_params_to_set = {}

                # input parameters
                if 'img_path' in params:
                    input_image_name = layer_name(self.viewer, params['img_path'])
                    if input_image_name in [self.image_combo.itemText(i) for i in range(self.image_combo.count())]:
                        self.image_combo.setCurrentText(input_image_name)
                    else:
                        notes.append(f"Input image '{input_image_name}' not found in current layers.")
                if 'boundary_path' in params:
                    boundary_name = layer_name(self.viewer, params['boundary_path'])
                    if boundary_name in [self.boundary_combo.itemText(i) for i in range(self.boundary_combo.count())]:
                        self.boundary_combo.setCurrentText(boundary_name)
                    else:
                        notes.append(f"Boundary image '{boundary_name}' not found in current layers.")
                
                
                # Main parameters
                main_keys = ["seed_method", "thresholds",
                             "upper_thresholds", "erosion_steps",
                             "segments", "num_threads", "footprints"]
                for key in main_keys:
                    if key in params:
                        main_params_to_set[key] = params[key]
                        
                # Handle single threshold values for erosion method
                if 'threshold' in params and 'thresholds' not in main_params_to_set:
                    main_params_to_set['thresholds'] = [params['threshold']]
                if 'upper_threshold' in params and 'upper_thresholds' not in main_params_to_set:
                    main_params_to_set['upper_thresholds'] = [params['upper_threshold']]
                
                if main_params_to_set:
                    self.main_param_widget.set_params(main_params_to_set)

                # Advanced parameters
                advanced_keys = ["sort", "no_split_max_iter", 
                                 "min_size", "min_split_ratio", 
                                 "min_split_total_ratio", "split_size_limit", 
                                 "split_convex_hull_limit"]
                for key in advanced_keys:
                    if key in params:
                        advanced_params_to_set[key] = params[key]

                if advanced_params_to_set:
                    self.advanced_params_box.set_params(advanced_params_to_set)
                    if params.get('seed_method', "Original") != "Original":
                        self.show_advanced_checkbox.setChecked(True)
                
                # Output parameters
                if 'output_folder' in params:
                    self.output_folder_line.setText(params["output_folder"])
                if 'save_every_iter' in params:
                    self.save_every_iter_checkbox.setChecked(params["save_every_iter"])

                # Populate the thresholds table with the list from YAML
                thresholds_list = as_list(params.get('thresholds', None))
                upper_thresholds_list = as_list(params.get('upper_thresholds', None))
                self.main_param_widget.populate_thresholds_from_list(thresholds_list, upper_thresholds_list)
                if 'footprints' in params:
                    self.main_param_widget.set_footprints(params['footprints'])
                
                # Also set the threshold widget for preview functionality
                if thresholds_list:
                    lower_threshold = thresholds_list[0]
                    if upper_thresholds_list and upper_thresholds_list:
                        upper_threshold = upper_thresholds_list[0]
                    else:
                        upper_threshold = 255 if lower_threshold <= 255 else lower_threshold + 100
                    self.threshold_widget.set_thresholds(lower_threshold, upper_threshold)
                

                if notes:
                    self.params_status.set("\n".join(notes), "warn")
                else:
                    self.params_status.set(f"Parameters successfully loaded from {os.path.basename(file_path)}.", "ok")

                # Currently not used as we use table for thresholds
                # Update threshold line edits, which are part of this widget
                # if 'thresholds' in params:
                #     threshold_str = ", ".join(map(str, params.get('thresholds', [])))
                #     self.lower_thresholds_list.setText(threshold_str)
                    
                # if 'upper_thresholds' in params:
                #     upper_threshold_str = ", ".join(map(str, params.get('upper_thresholds', [])))
                #     self.upper_thresholds_list.setText(upper_threshold_str)

               

            except Exception as e:
                self.params_status.set(f"Failed to load parameters: {e}", "error")

    def export_yaml(self, file_path=None):
        if file_path is None:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save YAML File", "", "YAML Files (*.yaml *.yml);;All Files (*)"
            )
        if file_path:
            try:
                # Get parameters from child widgets
                main_params = self.main_param_widget.get_params()
                advanced_params = self.advanced_params_box.get_params()
                

                
                # Combine all parameters into a single flat dictionary
                all_params = {}
                all_params.update(main_params)
                all_params.update(advanced_params)

                # the files of the input layers
                unsaved = []
                all_params["img_path"] = yaml_path(self.viewer, self.image_combo.currentText(), unsaved)
                all_params["boundary_path"] = yaml_path(self.viewer, self.boundary_combo.currentText(), unsaved)

                # Get and parse threshold lists from the main widget
                # deactivated as the listparam group is not used
                # lower_threshold_str = self.lower_thresholds_list.text()
                # if lower_threshold_str:
                #     all_params['thresholds'] = [int(x.strip()) for x in lower_threshold_str.split(',') if x.strip()]

                # if self.use_upper_list.isChecked():
                #     upper_threshold_str = self.upper_thresholds_list.text()
                #     if upper_threshold_str:
                #         all_params['upper_thresholds'] = [int(x.strip()) for x in upper_threshold_str.split(',') if x.strip()]

                # Add output parameters
                all_params["output_folder"] = self.output_folder_line.text()
                all_params["save_every_iter"] = self.save_every_iter_checkbox.isChecked()

                # turn tuple to list for yaml saving
                if 'split_size_limit' in all_params and isinstance(all_params['split_size_limit'], tuple):
                    all_params['split_size_limit'] = list(all_params['split_size_limit'])  
                if 'split_convex_hull_limit' in all_params and isinstance(all_params['split_convex_hull_limit'], tuple):
                    all_params['split_convex_hull_limit'] = list(all_params['split_convex_hull_limit'])
                
                

                # Write to YAML file with lists in flow style (e.g., [1, 2, 3])
                class FlowStyleList(list):
                    pass

                def represent_flow_style_list(dumper, data):
                    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

                yaml.add_representer(FlowStyleList, represent_flow_style_list)

                # Convert all list values to FlowStyleList for flow style output
                def convert_lists(obj):
                    if isinstance(obj, list):
                        return FlowStyleList([convert_lists(i) for i in obj])
                    elif isinstance(obj, dict):
                        return {k: convert_lists(v) for k, v in obj.items()}
                    else:
                        return obj

                all_params_flow = convert_lists(all_params)

                # Reorder keys to write seed_method, img_path, boundary_path first
                ordered_keys = ["seed_method", "img_path", "boundary_path"]
                rest_keys = [k for k in all_params_flow.keys() if k not in ordered_keys]
                ordered_params = {k: all_params_flow[k] for k in ordered_keys if k in all_params_flow}
                ordered_params.update({k: all_params_flow[k] for k in rest_keys})

                with open(file_path, "w") as f:
                    yaml.dump(ordered_params, f, sort_keys=False, default_flow_style=False)

                self.params_status.set(*saved_note(os.path.basename(file_path), unsaved))
            except Exception as e:
                self.params_status.set(f"Failed to save parameters: {e}", "error")