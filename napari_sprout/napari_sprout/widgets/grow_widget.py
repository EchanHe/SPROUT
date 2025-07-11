"""Widget for seed growth in SPROUT workflow."""

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QFormLayout, QProgressBar,
    QHeaderView
)
from qtpy.QtCore import Qt, Signal, QThread
import numpy as np
from typing import Optional, List
from napari.layers import Image, Labels
from napari.utils.notifications import show_info, show_error


from ..utils.util_widget import (create_output_folder_row, GrowOptionalParamGroupBox,
                                 MainGrowParamWidget,apply_threshold_preview,ThresholdWidget)


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
        input_group = QGroupBox("Input Selection")
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
        # self.save_btn = QPushButton("Save Result")
        # self.save_btn.setEnabled(False)
        
        btn_layout.addWidget(self.grow_btn)
        btn_layout.addWidget(self.stop_btn)
        # btn_layout.addWidget(self.save_btn)
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
        
        # self.save_btn.clicked.connect(self.save_result)



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
            if lower is None:
                lower, upper = self.threshold_widget.get_thresholds()

            binary = apply_threshold_preview(self.current_image, lower, upper)

            # delete existing preview layer if it exists
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
        except Exception as e:
            show_error(f"Error in preview: {e}")