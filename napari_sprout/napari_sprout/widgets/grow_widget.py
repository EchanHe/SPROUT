"""Widget for seed growth in SPROUT workflow."""

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QFormLayout, QProgressBar,
    QHeaderView, QFileDialog, QMessageBox,QScrollArea
)
from qtpy.QtCore import Qt, Signal
import numpy as np
from typing import Optional, List
from napari.layers import Image, Labels
from napari.utils.notifications import show_info, show_error


from ..utils.util_widget import (create_output_folder_row, GrowOptionalParamGroupBox,
                                 MainGrowParamWidget,apply_threshold_preview,ThresholdWidget)
from ..utils.histogram_threshold_widget import PREVIEW_NAME, HistogramThresholdWidget, append_threshold_row
from ..utils.boundary_check import boundary_ok
from ..utils.roi_layers import is_roi_layer
from ..utils.roi_widget import RoiSelect, range_text
from ..utils.status import RUN_BUTTON, STOP_BUTTON, StatusLine, follow_theme, style
from ..utils import jobs
from .seed_widget import combo_items, input_combo, put_in_tab_order, rename_in_combos, seed_runs

import yaml
import sys
import os
import time
# get the current file's absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from make_grow import grow_mp
from sprout_core import roi as R
from sprout_core.sprout_core import cast_labels
from sprout_core.roi import RoiError


class GrowthWorker:
    """The parameters of one Grow run; run(cancel, progress) is the job (utils/jobs.py)."""

    def __init__(self, image, seeds, thresholds, dilate_iters,
                 n_threads=4,   
                 output_folder=None,
                 upper_thresholds=None, boundary=None, touch_rule='stop',
                 base_name='growth_result',
                 save_every_n_iters = None,
                 grow_to_end=False,
                 is_sort = False,
                 to_grow_ids=None,
                 min_growth_size=50,
                 no_growth_max_iter=3,
                 roi=None
                 ):
        self.roi = roi   # sprout_core.roi.Roi or None (whole image)
        self.image = image
        self.seeds = seeds
        self.thresholds = thresholds
        self.dilate_iters = dilate_iters
        
        

        self.upper_thresholds = upper_thresholds
        self.boundary = boundary
        self.touch_rule = touch_rule
        self.base_name = base_name
        self.output_folder = output_folder if output_folder else 'napari_temp/grow'
        self.n_threads = n_threads
        
        self.save_every_n_iters = save_every_n_iters if save_every_n_iters is not None else None
        self.grow_to_end = grow_to_end
        self.is_sort = is_sort
        
        self.to_grow_ids = to_grow_ids 
        self.min_growth_size = min_growth_size
        self.no_growth_max_iter = no_growth_max_iter
    
    @property
    def folder(self):
        """Where the run writes its files."""
        return os.path.join(self.output_folder, self.base_name)

    def run(self, cancel, progress):
        """Runs on the job's worker thread; returns (grows_dict, log_dict). On Stop, Cancelled
        carries the number of files written (files_written)."""
        watch = jobs.FolderWatch(self.folder)
        try:
            print(f"Starting growth with {self.n_threads} threads")
            if self.roi is not None:
                # ROI as a wall: run on its bbox + 1 with the seeds inside the ROI only;
                # sorting ids would renumber them inside the ROI only, so it is off
                full_seeds = self.seeds
                self.image, self.boundary, self.seeds = R.wall_inputs(
                    self.roi, self.image, self.boundary, seg=self.seeds)
                self.is_sort = False
            grows_dict , log_dict = grow_mp(
                
                img = self.image,
                seg = self.seeds,
                boundary = self.boundary,
                
                base_name = self.base_name,
                                
                dilation_steps = self.dilate_iters,
                thresholds = self.thresholds,
                upper_thresholds = self.upper_thresholds,
                
                # Default is 'stop', actually no other rules for now.
                touch_rule = self.touch_rule, 
                
                output_folder = self.output_folder, 
                num_threads = self.n_threads,
                
                save_every_n_iters = self.save_every_n_iters, 
                grow_to_end = self.grow_to_end,
                is_sort = self.is_sort,
                
                to_grow_ids = self.to_grow_ids,
                # a int for detect min diff and a int for tolerate iterations
                min_growth_size = self.min_growth_size,
                no_growth_max_iter = self.no_growth_max_iter,

                    
                # Fixed para
                return_for_napari = True,
                cancel = cancel, progress = progress

                )
            jobs.check_cancel(cancel)         # Stop after the last step: no layers either
            if self.roi is not None:   # image-sized; outside the ROI = the input seeds
                grows_dict = R.paste_grow(self.roi, grows_dict, full_seeds)
                R.write_sidecar(self.roi, os.path.join(self.output_folder, self.base_name))
            return grows_dict, log_dict
        except jobs.Cancelled as e:
            e.files_written = watch.written()
            raise

    def stopped(self, e, done, total):
        """The orange line after Stop."""
        return jobs.stopped_text(done, total, "iterations", getattr(e, "files_written", None))




class SeedGrowthWidget(QWidget):
    """Widget for interactive seed growth."""
    
    growth_completed = Signal(np.ndarray)
    
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        follow_theme(napari_viewer, self)       # the info grey of napari's theme
        
        self.preview_layer = None
        
        self.current_image = None
        self.current_seeds = None
        self.current_boundary = None
        self.growth_result = None
        self.worker = None
        
        self.previous_image_name = None
        self._run_roi = None             # ROI and image scale / translate of the running job
        self._run_layer_kwargs = None
        self._run_note = ""
        
        self._init_ui()
        self._connect_signals()
        self.refresh_layers()   # pick up layers that were loaded before the panel opened

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # Input
        input_group = QGroupBox("Input")
        input_layout = QFormLayout()

        self.image_combo = input_combo()
        self.seeds_combo = input_combo()
        self.boundary_combo = input_combo()
        self.boundary_combo.addItem("None")

        input_layout.addRow("Image", self.image_combo)
        input_layout.addRow("Seeds", self.seeds_combo)
        input_layout.addRow("Boundary", self.boundary_combo)
        # a boundary that cannot be used: red line under the combo, the run is refused
        self.boundary_error = StatusLine(hide_empty=True)
        input_layout.addRow(self.boundary_error)
        self.roi_select = RoiSelect(
            self.viewer, lambda: None if self.current_image is None else np.shape(self.current_image), "Grow")
        input_layout.addRow("ROI", self.roi_select)
        # seeds made in an ROI: offer that ROI
        self.seed_roi_hint = QLabel("")
        self.seed_roi_hint.setWordWrap(True)
        self.seed_roi_btn = QPushButton("")
        self.seed_roi_btn.clicked.connect(self._use_seed_roi)
        row = QHBoxLayout()
        row.addWidget(self.seed_roi_hint, 1)
        row.addWidget(self.seed_roi_btn)
        input_layout.addRow(row)
        self.seed_roi_hint.setVisible(False)
        self.seed_roi_btn.setVisible(False)
        self.seeds_combo.currentTextChanged.connect(lambda *_: self._update_seed_roi_hint())
        self.roi_select.hub.rois_changed.connect(self._update_seed_roi_hint)
        self.roi_select.hub.roi_edited.connect(self._update_seed_roi_hint)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)


        self.threshold_widget = HistogramThresholdWidget(self.viewer, roi_use=self.roi_select,
                                                         preview_name=PREVIEW_NAME + " (Grow)")
        self.threshold_widget.preview_requested.connect(self.preview_threshold)
        layout.addWidget(self.threshold_widget)


        self.main_param_widget = MainGrowParamWidget(title="Parameters",
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

        # advanced grow options, at the bottom of Parameters
        self.show_advanced_checkbox = QCheckBox("Show advanced options")
        self.show_advanced_checkbox.stateChanged.connect(self._toggle_grow_params_visibility)
        self.advanced_params_box = GrowOptionalParamGroupBox()
        self.advanced_params_box.setVisible(False)
        self.main_param_widget.add_bottom_widget(self.show_advanced_checkbox)
        self.main_param_widget.add_bottom_widget(self.advanced_params_box)

        # Output
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        output_dir_layout, self.output_folder_line = create_output_folder_row(label="Folder")
        output_layout.addLayout(output_dir_layout)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)


        # Control buttons        # Control buttons
        btn_layout = QHBoxLayout()
        self.grow_btn = QPushButton("Start Growth")
        self.grow_btn.setStyleSheet(RUN_BUTTON)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(STOP_BUTTON)
        self.stop_btn.setToolTip("Stop growing (files already written stay in the folder)")
        self.stop_btn.setEnabled(False)

        
        btn_layout.addWidget(self.grow_btn)
        btn_layout.addWidget(self.stop_btn)
        self.roi_select.set_run_button(self.grow_btn, "Start Growth")
        layout.addLayout(btn_layout)

        
        # Status: one line, info / ok / warn / error (utils/status.py)
        self.status_label = StatusLine("Ready to grow seeds", "info")
        self.results_label = self.status_label          # same name as in the Seeds tab
        layout.addWidget(self.status_label)
        layout.addStretch()
        # start / Stop / progress / end messages of a run (utils/jobs.py)
        self.run_control = jobs.RunControl(self, self.status_label, [self.grow_btn], self.stop_btn,
                                           self.progress_bar, notify=self._notify)

        main_scroll_container = QWidget()
        main_scroll_container.setLayout(layout)


        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(main_scroll_container)
        
        final_layout = QVBoxLayout()
        final_layout.addWidget(scroll)
        self.setLayout(final_layout)


        # the Input combos follow napari's layers by themselves (added / removed / renamed),
        # through the viewer's hub: a closed panel leaves no napari callbacks behind
        hub = self.roi_select.hub
        hub.layer_inserted.connect(self._on_layers_changed)
        hub.layer_removed.connect(self._on_layers_changed)
        hub.layer_renamed.connect(self._on_layer_renamed)
        # a finished Seeds run: its last seeds layer is picked (the user can pick another)
        seed_runs(self.viewer).finished.connect(self._on_seeds_run)
        self.seeds_combo.activated.connect(self._on_seeds_picked)
        self._seeds_note = None
        # Tab reaches Import / Export YAML where they are shown: before the parameters
        put_in_tab_order(self.main_param_widget.thread_spin, self.import_btn, self.export_btn)
        
        # Add default threshold
        # self._add_threshold_row(100, None, 5)
    
    def _connect_signals(self):
        """Connect widget signals."""
        self.grow_btn.clicked.connect(self.start_growth)
        self.stop_btn.clicked.connect(self.stop_growth)
        self.image_combo.currentTextChanged.connect(self._on_image_changed)
        

        self.import_btn.clicked.connect(lambda: self.import_yaml())
        self.export_btn.clicked.connect(lambda: self.export_yaml())
        self.boundary_combo.currentTextChanged.connect(self._on_boundary_changed)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._combos_in_sync():         # a missed layer event: re-sync, keeping the selection
            self._on_layers_changed()

    def _combos_in_sync(self):
        imgs = [l.name for l in self.viewer.layers if isinstance(l, Image)]
        labs = [l.name for l in self.viewer.layers if isinstance(l, Labels)]
        return (combo_items(self.image_combo) == imgs and combo_items(self.seeds_combo) == labs
                and combo_items(self.boundary_combo) == ["None"] + labs)

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

    def _on_seeds_run(self, name):
        """A Seeds run finished: pick the last seeds layer it added."""
        if self.seeds_combo.findText(name) < 0:
            self._on_layers_changed()          # its layer event not seen yet
        if self.seeds_combo.findText(name) < 0:
            return
        self.seeds_combo.setCurrentText(name)
        if not self.run_control.running:       # a Grow run keeps its own line
            self._seeds_note = (name, f"Seeds: {name} (last Seeds run)")
            self.status_label.set(self._seeds_note[1], "info")

    def _on_seeds_picked(self, *_):
        """The user picked other seeds: the 'last Seeds run' line no longer applies."""
        note = self._seeds_note
        if note is not None and self.seeds_combo.currentText() != note[0]:
            self._seeds_note = None
            if self.status_label.text() == note[1]:
                self.status_label.clear()

    def _on_layer_renamed(self, layer, old):
        """Rename in place: the selections (and the threshold table) stay as they are."""
        rename_in_combos(layer, old, (self.image_combo, self.seeds_combo, self.boundary_combo))
        if self.previous_image_name == old and isinstance(layer, Image):
            self.previous_image_name = layer.name
        self._update_seed_roi_hint()


    def _on_image_changed(self, _):
        """Handle image selection change."""
        text = self.image_combo.currentText()
        if text and text != self.previous_image_name:
            had_image = self.previous_image_name is not None
            self.previous_image_name = text
            
            self.main_param_widget.clean_table()      # thresholds depend on the image's values
            if had_image:
                self.params_status.set("Thresholds reset for the new image", "info")


            layer = self.viewer.layers[self.image_combo.currentText()]
            if isinstance(layer, Image):
                self.current_image = layer.data
            self.threshold_widget.set_image(layer)
            self.roi_select.refresh()
            
    def _seed_roi(self):
        """(name, recorded metadata, ROI layer or None) if the picked seeds were made in an ROI."""
        name = self.seeds_combo.currentText()
        lay = self.viewer.layers[name] if name and name in self.viewer.layers else None
        md = lay.metadata.get(R.META_KEY) if isinstance(lay, Labels) else None
        if not isinstance(md, dict) or not md.get("name"):
            return None
        roi_name = md["name"]
        roi_layer = self.viewer.layers[roi_name] if roi_name in self.viewer.layers else None
        return roi_name, md, (roi_layer if roi_layer is not None and is_roi_layer(roi_layer) else None)

    def _update_seed_roi_hint(self, *_):
        info = self._seed_roi()
        show = info is not None
        self.seed_roi_hint.setVisible(show)
        self.seed_roi_btn.setVisible(show and info[2] is not None)
        if not show:
            return
        name, md, layer = info
        text, css = f"Seeds were made in {name}", style("info")
        if layer is None:
            text += " (that ROI layer is gone)"
        else:
            self.seed_roi_btn.setText(f"Use {name}")
            roi = self.roi_select.hub.live_roi(layer)
            was = R.Roi.from_metadata(md)
            if roi is not None and roi.key != was.key:
                text = f"{name} changed since the seeds (was {range_text(was)})"
                css = style("warn")
        self.seed_roi_hint.setText(text)
        self.seed_roi_hint.setStyleSheet(css)

    def _use_seed_roi(self):
        info = self._seed_roi()
        if info is not None and info[2] is not None:
            self.roi_select.select(info[2])

    def _toggle_grow_params_visibility(self, state):
        self.advanced_params_box.setVisible(state == Qt.Checked)  
          
    def refresh_layers(self):
        """Refresh the list of available layers."""
        current_image_name = self.image_combo.currentText()
        current_seeds_name = self.seeds_combo.currentText()
        current_boundary_name = self.boundary_combo.currentText()

        self.image_combo.blockSignals(True)  # prevent triggering img_changed temporarily
        self.image_combo.clear()
        self.seeds_combo.clear()
        self.boundary_combo.clear()
        self.boundary_combo.addItem("None")

        image_names = []
        label_names = []
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                image_names.append(layer.name)
                self.image_combo.addItem(layer.name)
            elif isinstance(layer, Labels):
                label_names.append(layer.name)
                self.seeds_combo.addItem(layer.name)
                self.boundary_combo.addItem(layer.name)

        # Restore image selection if it still exists
        if current_image_name in image_names:
            self.image_combo.setCurrentText(current_image_name)
        # Restore the seeds and boundary selections too: without this, adding
        # any layer (e.g. the live Threshold Preview) silently reset them
        if current_seeds_name in label_names:
            self.seeds_combo.setCurrentText(current_seeds_name)
        if current_boundary_name in label_names:
            self.boundary_combo.setCurrentText(current_boundary_name)
        self.image_combo.blockSignals(False)

        # If current_text changed (e.g., setCurrentText fails), manually trigger img_changed
        if self.image_combo.currentText() != self.previous_image_name:
            self._on_image_changed(self.image_combo.currentText())


    def start_growth(self):
        """Start the growth process."""
        # Get inputs
        if not self.image_combo.currentText() or not self.seeds_combo.currentText():
            self.status_label.set("Select an image and a seeds layer first", "error")
            return
        if self.run_control.running:
            return
        try:
            self.advanced_params_box.to_grow_ids()      # a typo: red line in the advanced box, no run
        except ValueError as e:
            self.show_advanced_checkbox.setChecked(True)
            self.status_label.set(str(e), "error")
            return
        
        try:
            # Get data
            self.current_image = self.viewer.layers[self.image_combo.currentText()].data
            self.current_seeds = self.viewer.layers[self.seeds_combo.currentText()].data
            
            # Get boundary if selected
            self.current_boundary = None
            if self.boundary_combo.currentText() != "None":
                self.current_boundary = self.viewer.layers[self.boundary_combo.currentText()].data
            if not boundary_ok(self, self.boundary_combo.currentText(), self.current_boundary, self.boundary_error):
                self._bad_boundary = self.boundary_combo.currentText()
                return
            try:
                roi = self.roi_select.roi_for_run(np.shape(self.current_image))
            except RoiError as e:
                self.status_label.set(str(e), "error")
                return
            image_layer = self.viewer.layers[self.image_combo.currentText()]
            self._run_roi = roi
            self._run_layer_kwargs = {"scale": image_layer.scale, "translate": image_layer.translate}
            
            # Get parameters
            # thresholds, upper_thresholds, dilate_iters = self._get_threshold_params()
            
            main_params = self.main_param_widget.get_params()

            
            # Deprecated: sort thresholds from high to low
            # TODO raise an error instead            
            # # Sort thresholds from high to low
            # sorted_indices = sorted(range(len(thresholds)), key=lambda i: thresholds[i], reverse=True)
            # thresholds = [int(thresholds[i]) for i in sorted_indices]
            # upper_thresholds = [int(upper_thresholds[i]) if upper_thresholds[i] is not None else None for i in sorted_indices]
            # dilate_iters = [dilate_iters[i] for i in sorted_indices]
            
            advance_params = self.advanced_params_box.get_params()
            self._run_note = (" (Sort IDs is ignored with an ROI)"
                              if roi is not None and advance_params['is_sort'] else "")

            # Create and start the job
            worker = GrowthWorker(
                image= self.current_image,
                seeds= self.current_seeds,
                thresholds=  main_params['thresholds'],
                dilate_iters=  main_params['dilation_steps'],
                
                n_threads= main_params['num_threads'],
                
                output_folder= self.output_folder_line.text(),
                upper_thresholds=main_params["upper_thresholds"] if any(u is not None for u in main_params["upper_thresholds"]) else None,
                boundary= self.current_boundary,
                touch_rule= main_params['touch_rule'],
                base_name=R.roi_name(image_layer.name, roi.name or "roi") if roi is not None else image_layer.name,
                save_every_n_iters= advance_params['save_every_n_iters'],
                grow_to_end= advance_params['grow_to_end'],
                is_sort= advance_params['is_sort'],
                to_grow_ids= advance_params['to_grow_ids'],
                min_growth_size= advance_params['min_growth_size'],
                no_growth_max_iter= advance_params['no_growth_max_iter'],
                roi=roi
            )
            
            self.worker = self.run_control.start(
                worker.run, "Growth", "Growing seeds…",
                done=self._on_growth_finished, stopped=worker.stopped,
                failed=lambda e: ("ROI: " if roi is not None else "") + jobs.reason(e))

        except Exception as e:
            show_error(f"Growth failed: {jobs.reason(e)}")
            self.status_label.set(f"Growth failed: {jobs.reason(e)}", "error")

    def stop_growth(self):
        """Stop growing at the next iteration (cooperative; no partial layers)."""
        self.run_control.stop()

    @staticmethod
    def _notify(level, text):
        """The one napari notification of a run (finished / failed)."""
        (show_error if level == "error" else show_info)(text)

    def _on_growth_finished(self, result, secs):
        """Handle growth completion (UI thread): add the layers, return the summary."""
        grows_dict, log_dict = result
        # self.growth_result = result
        
        # Add result to viewer
        roi = self._run_roi
        for key, value in grows_dict.items():
            print(f"{key}: {value.shape}")
            kwargs = dict(self._run_layer_kwargs or {})   # the image's scale / translate
            if roi is not None:   # the names already hold the '<image>_roi' base name
                kwargs["metadata"] = {R.META_KEY: roi.to_metadata()}
            # at least uint16 in napari (as the seeds): a brushed id above 255 would wrap in uint8
            self.viewer.add_labels(
                cast_labels(value, np.uint16, copy=False),
                name=key,
                **kwargs
            )
        
        # Summary for the status line and the notification ("✓ ... (2.5 s)<note>")
        n = len(grows_dict)
        # import psutil, os
        # print("MEM (MB):", psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
        # Emit signal
        # self.growth_completed.emit(result)
        return f"Growth completed: {n} layer{'' if n == 1 else 's'}", self._run_note


    def preview_threshold(self, lower=None, upper=None):
        if self.current_image is None:
            show_error("Select an image first")
            return

        try:
            self.threshold_widget.run_preview_in_thread(self.current_image, callback=self._on_preview_result)
        except Exception as e:
            show_error(f"Error in preview: {e}")
    
    def _on_preview_result(self, binary):
        """Handle the result of the threshold preview."""
        if self.preview_layer and self.preview_layer in self.viewer.layers:
            self.viewer.layers.remove(self.preview_layer)
            self.preview_layer = None

        # delete existing seeds layer if it exists
        self.preview_layer = self.viewer.add_labels(
            binary.astype(np.uint8),
            name=PREVIEW_NAME + " (Grow)",
            opacity=0.5,
        )
        show_info(f"Preview updated: {np.sum(binary)} pixels selected")

        self.threshold_widget.preview_btn.setEnabled(True)  # Re-enable button after processing
        # Optionally, you can connect this to a signal or update other UI elements
        
    def import_yaml(self, file_path=None):
        """Import grow parameters from a YAML file."""
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

                
                ### For inputs
                # Input image layer
                if 'img_path' in params:
                    input_image_name = params['img_path']
                    if input_image_name in [self.image_combo.itemText(i) for i in range(self.image_combo.count())]:
                        self.image_combo.setCurrentText(input_image_name)
                    else:
                        notes.append(f"Input image '{input_image_name}' not found in current layers.")
                
                # Seeds layer
                if 'seg_path' in params:
                    seeds_name = params['seg_path']
                    if seeds_name in [self.seeds_combo.itemText(i) for i in range(self.seeds_combo.count())]:
                        self.seeds_combo.setCurrentText(seeds_name)
                    else:
                        notes.append(f"Seeds layer '{seeds_name}' not found in current layers.")
                
                # Boundary layer
                if 'boundary_path' in params and params['boundary_path'] is not None:
                    boundary_name = params['boundary_path']
                    if boundary_name in [self.boundary_combo.itemText(i) for i in range(self.boundary_combo.count())]:
                        self.boundary_combo.setCurrentText(boundary_name)
                    else:
                        notes.append(f"Boundary layer '{boundary_name}' not found in current layers.")
                
                # Main parameters
                main_keys = ["thresholds", "upper_thresholds", "dilation_steps", 
                             "num_threads", "touch_rule"]
                for key in main_keys:
                    if key in params:
                        main_params_to_set[key] = params[key]
                
                if main_params_to_set:
                    self.main_param_widget.set_params(main_params_to_set)

                # Advanced parameters
                advanced_keys = ["save_every_n_iters", "grow_to_end", 
                                "is_sort", "to_grow_ids", 
                                "min_growth_size", "no_growth_max_iter"]
                for key in advanced_keys:
                    if key in params:
                        advanced_params_to_set[key] = params[key]

                if advanced_params_to_set:
                    self.advanced_params_box.set_params(advanced_params_to_set)
                    # Show advanced parameters if any are set
                    if any(v is not None and v != False for v in advanced_params_to_set.values()):
                        self.show_advanced_checkbox.setChecked(True)
                
                # Output folder
                if 'output_folder' in params:
                    self.output_folder_line.setText(params["output_folder"])

                # Populate the thresholds table with the lists from YAML
                thresholds_list = params.get('thresholds', None)
                upper_thresholds_list = params.get('upper_thresholds', None)
                dilation_steps_list = params.get('dilation_steps', None)
                
                if thresholds_list:
                    self.main_param_widget.populate_thresholds_from_list(
                        thresholds_list, upper_thresholds_list, dilation_steps_list
                    )
                    
                    # Also set the threshold widget for preview functionality
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

            except Exception as e:
                self.params_status.set(f"Failed to load parameters: {e}", "error")

    def export_yaml(self, file_path=None):
        """Export grow parameters to a YAML file."""
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

                # Get input image name - use as img_path
                if self.image_combo.currentText():
                    all_params["img_path"] = self.image_combo.currentText()
                else:
                    all_params["img_path"] = None
                
                # Get seeds layer name - use as seg_path
                if self.seeds_combo.currentText():
                    all_params["seg_path"] = self.seeds_combo.currentText()
                else:
                    all_params["seg_path"] = None
                
                # Get the boundary image name - use as boundary_path
                if self.boundary_combo.currentText() != "None":
                    all_params["boundary_path"] = self.boundary_combo.currentText()
                else:
                    all_params["boundary_path"] = None

                # Add output folder
                all_params["output_folder"] = self.output_folder_line.text() or "./result/grow_output"

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

                # Reorder keys to write img_path, seg_path, boundary_path, thresholds, dilation_steps first
                ordered_keys = ["img_path", "seg_path", "boundary_path", 
                               "thresholds", "dilation_steps", "upper_thresholds",
                               "touch_rule", "output_folder", "num_threads"]
                rest_keys = [k for k in all_params_flow.keys() if k not in ordered_keys]
                ordered_params = {k: all_params_flow[k] for k in ordered_keys if k in all_params_flow}
                ordered_params.update({k: all_params_flow[k] for k in rest_keys})

                with open(file_path, "w") as f:
                    yaml.dump(ordered_params, f, sort_keys=False, default_flow_style=False)
                
                self.params_status.set(f"Parameters successfully saved to {os.path.basename(file_path)}.", "ok")

            except Exception as e:
                self.params_status.set(f"Failed to save parameters: {e}", "error")