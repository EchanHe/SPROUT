"""Widget for seed growth in SPROUT workflow."""

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QFormLayout, QProgressBar,
    QHeaderView, QFileDialog, QMessageBox,QScrollArea
)
from qtpy.QtCore import Qt, Signal, QThread
import numpy as np
from typing import Optional, List
from napari.layers import Image, Labels
from napari.utils.notifications import show_info, show_error


from ..utils.util_widget import (create_output_folder_row, GrowOptionalParamGroupBox,
                                 MainGrowParamWidget,apply_threshold_preview,ThresholdWidget)
from ..utils.histogram_threshold_widget import HistogramThresholdWidget, append_threshold_row
from ..utils.boundary_check import boundary_ok
from ..utils.roi_layers import is_roi_layer
from ..utils.roi_widget import RoiSelect, range_text

import yaml
import sys
import os
# get the current file's absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from make_grow import grow_mp
from sprout_core import roi as R
from sprout_core.roi import RoiError


class GrowthWorker(QThread):
    """Worker thread for seed growth."""
    progress = Signal(int)
    finished = Signal(dict, dict)  # emits grows_dict and log_dict
    error = Signal(str)
    
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
        super().__init__()
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
    
    def run(self):
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
                return_for_napari = True
                                   
                )      
            if self.roi is not None:   # image-sized; outside the ROI = the input seeds
                grows_dict = R.paste_grow(self.roi, grows_dict, full_seeds)
                R.write_sidecar(self.roi, os.path.join(self.output_folder, self.base_name))
            self.finished.emit(grows_dict, log_dict)
        except Exception as e:
            self.error.emit(("ROI: " if self.roi is not None else "") + str(e))




class SeedGrowthWidget(QWidget):
    """Widget for interactive seed growth."""
    
    growth_completed = Signal(np.ndarray)
    
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        
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
        
        # Input selection
        input_group = QGroupBox("Input Options")
        input_layout = QFormLayout()
        
        self.image_combo = QComboBox()
        self.seeds_combo = QComboBox()
        self.boundary_combo = QComboBox()
        self.boundary_combo.addItem("None")
        
        input_layout.addRow("Original Image:", self.image_combo)
        input_layout.addRow("Seeds Layer:", self.seeds_combo)
        input_layout.addRow("Boundary Mask (optional):", self.boundary_combo)
        self.roi_select = RoiSelect(
            self.viewer, lambda: None if self.current_image is None else np.shape(self.current_image), "Grow")
        input_layout.addRow("ROI:", self.roi_select)
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
        
        self.refresh_btn = QPushButton("Refresh Layers")
        input_layout.addRow(self.refresh_btn)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)


        self.threshold_widget = HistogramThresholdWidget(self.viewer, roi_use=self.roi_select)
        self.threshold_widget.preview_requested.connect(self.preview_threshold)
        layout.addWidget(self.threshold_widget)


        self.main_param_widget = MainGrowParamWidget(title="Parameters",
                                                 image_combo=self.image_combo,viewer=self.viewer)
        self.threshold_widget.add_to_table_requested.connect(
            lambda lo, hi: append_threshold_row(self.main_param_widget, lo, hi))
        layout.addWidget(self.main_param_widget)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.show_advanced_checkbox = QCheckBox("Show Advanced Grow Options")
        self.show_advanced_checkbox.stateChanged.connect(self._toggle_grow_params_visibility)
        layout.addWidget(self.show_advanced_checkbox)
        
        self.advanced_params_box = GrowOptionalParamGroupBox()
        layout.addWidget(self.advanced_params_box)   
        self.advanced_params_box.setVisible(False)


        # Import/Export yaml buttons
        button_layout = QHBoxLayout()
        self.import_btn = QPushButton("Import YAML")
        self.export_btn = QPushButton("Export YAML")
        button_layout.addWidget(self.import_btn)
        button_layout.addWidget(self.export_btn)
        layout.addLayout(button_layout)   

        # Output folder selection
        output_dir_layout, self.output_folder_line = create_output_folder_row()
        layout.addLayout(output_dir_layout)


        # Control buttons
        btn_layout = QHBoxLayout()
        self.grow_btn = QPushButton("Start Growth")
        self.grow_btn.setStyleSheet("""QPushButton { font-weight: bold; background-color: #45a049;}""")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                background-color: #d9534f;  
                color: white;
            }
            QPushButton:disabled {
                background-color: #aaa;
                color: #eee;
            }
        """)
        self.stop_btn.setEnabled(False)

        
        btn_layout.addWidget(self.grow_btn)
        btn_layout.addWidget(self.stop_btn)
        self.roi_select.set_run_button(self.grow_btn, "Start Growth")
        layout.addLayout(btn_layout)

        
        # Status
        self.status_label = QLabel("Ready to grow seeds")
        layout.addWidget(self.status_label)

        main_scroll_container = QWidget()
        main_scroll_container.setLayout(layout)


        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(main_scroll_container)
        
        final_layout = QVBoxLayout()
        final_layout.addWidget(scroll)
        self.setLayout(final_layout)

        # refresh layers when layers changed in napari
        self.viewer.layers.events.inserted.connect(lambda e: self.refresh_layers())
        self.viewer.layers.events.removed.connect(lambda e: self.refresh_layers())
        
        # Add default threshold
        # self._add_threshold_row(100, None, 5)
    
    def _connect_signals(self):
        """Connect widget signals."""
        self.refresh_btn.clicked.connect(self.refresh_layers)
        self.grow_btn.clicked.connect(self.start_growth)
        self.stop_btn.clicked.connect(self.stop_growth)
        self.image_combo.currentTextChanged.connect(self._on_image_changed)
        

        self.import_btn.clicked.connect(self.import_yaml)
        self.export_btn.clicked.connect(self.export_yaml)


    def _on_image_changed(self, _):
        """Handle image selection change."""
        text = self.image_combo.currentText()
        if text and text != self.previous_image_name:
            self.previous_image_name = text
            
            self.main_param_widget.clean_table()


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
        text, style = f"Seeds were made in {name}", "color: gray"
        if layer is None:
            text += " (that ROI layer is gone)"
        else:
            self.seed_roi_btn.setText(f"Use {name}")
            roi = self.roi_select.hub.live_roi(layer)
            was = R.Roi.from_metadata(md)
            if roi is not None and roi.key != was.key:
                text = f"{name} changed since the seeds (was {range_text(was)})"
                style = "color: #e08a00"
        self.seed_roi_hint.setText(text)
        self.seed_roi_hint.setStyleSheet(style)

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
            show_error("Please select both an image and seeds layer")
            return
        
        try:
            # Get data
            self.current_image = self.viewer.layers[self.image_combo.currentText()].data
            self.current_seeds = self.viewer.layers[self.seeds_combo.currentText()].data
            
            # Get boundary if selected
            self.current_boundary = None
            if self.boundary_combo.currentText() != "None":
                self.current_boundary = self.viewer.layers[self.boundary_combo.currentText()].data
            if not boundary_ok(self, self.boundary_combo.currentText(), self.current_boundary):
                return
            try:
                roi = self.roi_select.roi_for_run(np.shape(self.current_image))
            except RoiError as e:
                show_error(str(e))
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
            
            # Update UI
            self.grow_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.status_label.setText("Growing seeds...")


            advance_params = self.advanced_params_box.get_params()
            self._run_note = (" (Sort IDs is ignored with an ROI)"
                              if roi is not None and advance_params['is_sort'] else "")

            # Create and start worker
            self.worker = GrowthWorker(
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
            
            self.worker.progress.connect(self.progress_bar.setValue)
            self.worker.finished.connect(self._on_growth_finished)
            self.worker.error.connect(self._on_growth_error)
            
            self.worker.start()
            # 
            
        except Exception as e:
            show_error(f"Error starting growth: {str(e)}")
            self._reset_ui()
    
    def stop_growth(self):
        """Stop the growth process."""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self._reset_ui()
            self.status_label.setText("Growth stopped by user")
    
    def _on_growth_finished(self, grows_dict, log_dict):
        """Handle growth completion."""
        # self.growth_result = result
        
        # Add result to viewer
        roi = self._run_roi
        for key, value in grows_dict.items():
            print(f"{key}: {value.shape}")
            kwargs = dict(self._run_layer_kwargs or {})   # the image's scale / translate
            if roi is not None:   # the names already hold the '<image>_roi' base name
                kwargs["metadata"] = {R.META_KEY: roi.to_metadata()}
            self.viewer.add_labels(
                value,  
                name=key,
                **kwargs
            )
        
        # Update UI
        self._reset_ui()
        self.status_label.setText(f"Growth completed!" + self._run_note)
        # import psutil, os
        # print("MEM (MB):", psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
        # Emit signal
        # self.growth_completed.emit(result)
        
        show_info("Growth completed successfully!")
    
    def _on_growth_error(self, error_msg):
        """Handle growth error."""
        show_error(f"Napari Growth Error: {error_msg}")
        self._reset_ui()
        self.status_label.setText("Growth failed - see error message")
    
    def _reset_ui(self):
        """Reset UI after growth."""
        self.grow_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
    

    def preview_threshold(self, lower=None, upper=None):
        if self.current_image is None:
            show_error("Please select an image first")
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
            name="Threshold Preview",
            opacity=0.5,
        )
        show_info(f"Preview updated: {np.sum(binary)} pixels selected")

        self.threshold_widget.preview_btn.setEnabled(True)  # Re-enable button after processing
        # Optionally, you can connect this to a signal or update other UI elements
        
    def import_yaml(self):
        """Import grow parameters from a YAML file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open YAML File", "", "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if file_path:
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
                        show_error(f"Input image '{input_image_name}' not found in current layers.")
                
                # Seeds layer
                if 'seg_path' in params:
                    seeds_name = params['seg_path']
                    if seeds_name in [self.seeds_combo.itemText(i) for i in range(self.seeds_combo.count())]:
                        self.seeds_combo.setCurrentText(seeds_name)
                    else:
                        show_error(f"Seeds layer '{seeds_name}' not found in current layers.")
                
                # Boundary layer
                if 'boundary_path' in params and params['boundary_path'] is not None:
                    boundary_name = params['boundary_path']
                    if boundary_name in [self.boundary_combo.itemText(i) for i in range(self.boundary_combo.count())]:
                        self.boundary_combo.setCurrentText(boundary_name)
                    else:
                        show_error(f"Boundary layer '{boundary_name}' not found in current layers.")
                
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

                show_info(f"Parameters successfully loaded from {os.path.basename(file_path)}.")

            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"Failed to load parameters: {e}")

    def export_yaml(self):
        """Export grow parameters to a YAML file."""
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
                
                show_info(f"Parameters successfully saved to {os.path.basename(file_path)}.")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to save parameters: {e}")