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


from ..utils.util_widget import create_output_folder_row, GrowOptionalParamGroupBox


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
        self.current_image = None
        self.current_seeds = None
        self.current_boundary = None
        self.growth_result = None
        self.worker = None
        
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
        
        self.n_thread_spin = QSpinBox() 
        self.n_thread_spin.setValue(4)  # Default to 4 threads
        self.n_thread_spin.setRange(1, os.cpu_count() or 8)  # Default to 8 if os.cpu_count() is None
        
        # Growth parameters
        params_group = QGroupBox("Growth Parameters")
        params_layout = QVBoxLayout()
        
        # Touch rule
        touch_layout = QHBoxLayout()
        touch_layout.addWidget(QLabel("Touch Rule:"))
        self.touch_rule_combo = QComboBox()
        self.touch_rule_combo.addItems(["stop", "continue"])
        touch_layout.addWidget(self.touch_rule_combo)
        touch_layout.addStretch()
        params_layout.addLayout(touch_layout)
        
        # Threshold table
        self.threshold_table = QTableWidget()
        self.threshold_table.setColumnCount(3)
        self.threshold_table.setHorizontalHeaderLabels(
            ["Lower Threshold", "Upper Threshold", "Dilate Iterations"]
        )
        self.threshold_table.horizontalHeader().setStretchLastSection(True)
        self.threshold_table.setMaximumHeight(200)
        
        # Add/remove buttons
        table_btn_layout = QHBoxLayout()
        self.add_threshold_btn = QPushButton("Add Threshold")
        self.remove_threshold_btn = QPushButton("Remove Selected")
        table_btn_layout.addWidget(self.add_threshold_btn)
        table_btn_layout.addWidget(self.remove_threshold_btn)
        table_btn_layout.addStretch()
        
        params_layout.addWidget(QLabel("Growth Thresholds (process from high to low):"))
        params_layout.addWidget(self.threshold_table)
        params_layout.addLayout(table_btn_layout)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

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


        self.show_advanced_checkbox = QCheckBox("Show Advanced Grow Options")
        self.show_advanced_checkbox.stateChanged.connect(self.toggle_grow_params_visibility)
        layout.addWidget(self.show_advanced_checkbox)
        
        self.advanced_params_box = GrowOptionalParamGroupBox()
        layout.addWidget(self.advanced_params_box)   
        self.advanced_params_box.setVisible(False)
        
        # Status
        self.status_label = QLabel("Ready to grow seeds")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # Add default threshold
        self._add_threshold_row(100, None, 5)
    
    def _connect_signals(self):
        """Connect widget signals."""
        self.refresh_btn.clicked.connect(self.refresh_layers)
        self.add_threshold_btn.clicked.connect(self._add_threshold_row)
        self.remove_threshold_btn.clicked.connect(self._remove_threshold_row)
        self.grow_btn.clicked.connect(self.start_growth)
        self.stop_btn.clicked.connect(self.stop_growth)
        # self.save_btn.clicked.connect(self.save_result)
        
    def toggle_grow_params_visibility(self, state):
        self.advanced_params_box.setVisible(state == Qt.Checked)  
          
    def refresh_layers(self):
        """Refresh the list of available layers."""
        self.image_combo.clear()
        self.seeds_combo.clear()
        self.boundary_combo.clear()
        self.boundary_combo.addItem("None")
        
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                self.image_combo.addItem(layer.name)
            elif isinstance(layer, Labels):
                self.seeds_combo.addItem(layer.name)
                self.boundary_combo.addItem(layer.name)
    
    def _add_threshold_row(self, lower=100, upper=None, dilate=5):
        """Add a new threshold row to the table."""
        row = self.threshold_table.rowCount()
        self.threshold_table.insertRow(row)
        
        # Lower threshold
        lower_spin = QDoubleSpinBox()
        lower_spin.setRange(0, 65535)
        lower_spin.setValue(lower)
        self.threshold_table.setCellWidget(row, 0, lower_spin)
        
        # Upper threshold
        upper_widget = QWidget()
        upper_layout = QHBoxLayout()
        upper_layout.setContentsMargins(0, 0, 0, 0)
        
        use_upper = QCheckBox()
        upper_spin = QDoubleSpinBox()
        upper_spin.setRange(0, 65535)
        upper_spin.setValue(upper if upper else 500)
        upper_spin.setEnabled(upper is not None)
        
        use_upper.setChecked(upper is not None)
        use_upper.toggled.connect(upper_spin.setEnabled)
        
        upper_layout.addWidget(use_upper)
        upper_layout.addWidget(upper_spin)
        upper_widget.setLayout(upper_layout)
        
        self.threshold_table.setCellWidget(row, 1, upper_widget)
        
        # Dilate iterations
        dilate_spin = QSpinBox()
        dilate_spin.setRange(1, 100)
        dilate_spin.setValue(dilate)
        self.threshold_table.setCellWidget(row, 2, dilate_spin)
    
    def _remove_threshold_row(self):
        """Remove selected threshold row."""
        current_row = self.threshold_table.currentRow()
        if current_row >= 0:
            self.threshold_table.removeRow(current_row)
    
    def _get_threshold_params(self):
        """Get threshold parameters from table."""
        thresholds = []
        upper_thresholds = []
        dilate_iters = []
        
        for row in range(self.threshold_table.rowCount()):
            # Lower threshold
            lower_widget = self.threshold_table.cellWidget(row, 0)
            thresholds.append(lower_widget.value())
            
            # Upper threshold
            upper_widget = self.threshold_table.cellWidget(row, 1)
            upper_layout = upper_widget.layout()
            use_upper = upper_layout.itemAt(0).widget()
            upper_spin = upper_layout.itemAt(1).widget()
            
            if use_upper.isChecked():
                upper_thresholds.append(upper_spin.value())
            else:
                upper_thresholds.append(None)
            
            # Dilate iterations
            dilate_widget = self.threshold_table.cellWidget(row, 2)
            dilate_iters.append(dilate_widget.value())
        
        return thresholds, upper_thresholds, dilate_iters
    
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
            thresholds, upper_thresholds, dilate_iters = self._get_threshold_params()
            
            if not thresholds:
                show_error("Please add at least one threshold")
                return
            
            # Sort thresholds from high to low
            sorted_indices = sorted(range(len(thresholds)), key=lambda i: thresholds[i], reverse=True)
            thresholds = [int(thresholds[i]) for i in sorted_indices]
            upper_thresholds = [int(upper_thresholds[i]) if upper_thresholds[i] is not None else None for i in sorted_indices]
            dilate_iters = [dilate_iters[i] for i in sorted_indices]
            
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
                thresholds=  thresholds,
                dilate_iters=  dilate_iters,
                
                n_threads= self.n_thread_spin.value(),
                
                output_folder= self.output_folder_line.text(),
                upper_thresholds=upper_thresholds if any(u is not None for u in upper_thresholds) else None,
                boundary= self.current_boundary,
                touch_rule= self.touch_rule_combo.currentText(),
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
    
