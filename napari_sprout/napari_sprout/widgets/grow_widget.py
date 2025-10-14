"""Widget for seed growth in SPROUT workflow."""

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QFormLayout, QProgressBar,
    QHeaderView, QFileDialog, QMessageBox
)
from qtpy.QtCore import Qt, Signal, QThread
import numpy as np
from typing import Optional, List
from napari.layers import Image, Labels
from napari.utils.notifications import show_info, show_error


from ..utils.util_widget import (create_output_folder_row, GrowOptionalParamGroupBox,
                                 MainGrowParamWidget,apply_threshold_preview,ThresholdWidget)

import yaml
import sys
import os
# get the current file's absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from make_grow import grow_mp


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
                 no_growth_max_iter=3
                 ):
        super().__init__()
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
            self.finished.emit(grows_dict, log_dict)
        except Exception as e:
            self.error.emit(str(e))




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
        
        self._init_ui()
        self._connect_signals()
    
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
        
        self.refresh_btn = QPushButton("Refresh Layers")
        input_layout.addRow(self.refresh_btn)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)


        self.threshold_widget = ThresholdWidget()
        self.threshold_widget.preview_requested.connect(self.preview_threshold)
        layout.addWidget(self.threshold_widget)        

        
        self.main_param_widget = MainGrowParamWidget(title="Parameters",
                                                 image_combo=self.image_combo,viewer=self.viewer)
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
        layout.addLayout(btn_layout)

        
        # Status
        self.status_label = QLabel("Ready to grow seeds")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        self.setLayout(layout)
        
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
            self.threshold_widget.set_range_by_dtype(self.current_image.dtype)
            
    def _toggle_grow_params_visibility(self, state):
        self.advanced_params_box.setVisible(state == Qt.Checked)  
          
    def refresh_layers(self):
        """Refresh the list of available layers."""
        current_image_name = self.image_combo.currentText()
        
        self.image_combo.blockSignals(True)  # prevent triggering img_changed temporarily
        self.image_combo.clear()
        self.seeds_combo.clear()
        self.boundary_combo.clear()
        self.boundary_combo.addItem("None")
        
        image_names = []
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                image_names.append(layer.name)
                self.image_combo.addItem(layer.name)
            elif isinstance(layer, Labels):
                self.seeds_combo.addItem(layer.name)
                self.boundary_combo.addItem(layer.name)
        
        # Restore image selection if it still exists
        if current_image_name in image_names:
            self.image_combo.setCurrentText(current_image_name)
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
                base_name=self.viewer.layers[self.image_combo.currentText()].name,
                save_every_n_iters= advance_params['save_every_n_iters'],
                grow_to_end= advance_params['grow_to_end'],
                is_sort= advance_params['is_sort'],
                to_grow_ids= advance_params['to_grow_ids'],
                min_growth_size= advance_params['min_growth_size'],
                no_growth_max_iter= advance_params['no_growth_max_iter']
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
        for key, value in grows_dict.items():
            print(f"{key}: {value.shape}")
            self.viewer.add_labels(
                value,  
                name=key
            )
        
        # Update UI
        self._reset_ui()
        self.status_label.setText(f"Growth completed!")
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