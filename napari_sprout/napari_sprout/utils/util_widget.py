import os, sys
from qtpy.QtWidgets import (
    QGroupBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,QDoubleSpinBox,
    QSpinBox, QCheckBox, QFileDialog , QVBoxLayout , QComboBox , QTableWidget,QWidget,QSlider,
     QHeaderView
    
)
from qtpy.QtCore import Signal, Qt ,QThreadPool,QObject,QRunnable,Slot
from napari.utils.notifications import show_error, show_info
import numpy as np

# current_dir = os.path.dirname(os.path.abspath(__file__))
# root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
# if root_dir not in sys.path:
#     sys.path.insert(0, root_dir)
from pathlib import Path
sprout_path = Path(__file__).parent.parent.parent.parent
if str(sprout_path) not in sys.path:
    sys.path.insert(0, str(sprout_path))

from sprout_core.config_core import support_footprints, support_footprints_2d
from sprout_core.sprout_core import get_ccomps_with_size_order




def create_output_folder_row(default_folder=None):

    layout = QHBoxLayout()
    label = QLabel("Output Folder:")
    folder_edit = QLineEdit()
    browse_btn = QPushButton("Browse")

    # set default folder if provided, otherwise use current working directory
    if default_folder is None:
        default_folder = os.path.join(os.getcwd(), "sprout_temp")
    os.makedirs(default_folder, exist_ok=True)
    folder_edit.setText(default_folder)

    def browse():
        folder = QFileDialog.getExistingDirectory(None, "Select Output Folder")
        if folder:
            folder_edit.setText(folder)

    browse_btn.clicked.connect(browse)

    layout.addWidget(label)
    layout.addWidget(folder_edit)
    layout.addWidget(browse_btn)

    return layout, folder_edit


class PreviewWorkerSignals(QObject):
    finished = Signal(np.ndarray)
    error = Signal(str)

class PreviewWorker(QRunnable):
    def __init__(self, image, lower, upper, use_cc=False, cc_n=100):
        super().__init__()
        self.image = image
        self.lower = lower
        self.upper = upper
        self.use_cc = use_cc
        self.cc_n = cc_n
        self.signals = PreviewWorkerSignals()

    @Slot()
    def run(self):
        try:
            binary = (self.image >= self.lower) & (self.image <= self.upper)
            if self.use_cc:
                binary, _ = get_ccomps_with_size_order(binary, self.cc_n)

            num_classes = np.unique(binary).size
            if num_classes <=255:
                binary = binary.astype(np.uint8)
            elif num_classes <= 65535:
                binary = binary.astype(np.uint16)
            
            self.signals.finished.emit(binary)

        except Exception as e:
            self.signals.error.emit(str(e))


class ThresholdWidget(QGroupBox):
    preview_requested = Signal(float, float)  # Always emit lower + upper

    def __init__(self, title = 'Threshold Preview', show_preview_button=True,
                 add_connected_components=False):
        super().__init__(title)
        self.add_connected_components = add_connected_components

        layout = QFormLayout()

        # Lower threshold
        self.lower_spin = QDoubleSpinBox()
        self.lower_spin.setDecimals(1)
        self.lower_spin.setRange(0, 65535)
        self.lower_spin.setValue(150)

        self.lower_slider = QSlider(Qt.Horizontal)
        self.lower_slider.setRange(0, 65535)
        self.lower_slider.setValue(150)

        # Upper threshold
        self.upper_spin = QDoubleSpinBox()
        self.upper_spin.setDecimals(1)
        self.upper_spin.setRange(0, 65535)
        self.upper_spin.setValue(255)

        self.upper_slider = QSlider(Qt.Horizontal)
        self.upper_slider.setRange(0, 65535)
        self.upper_slider.setValue(255)

        # link lower
        self.lower_spin.valueChanged.connect(lambda v: self.lower_slider.setValue(int(v)))
        self.lower_slider.valueChanged.connect(lambda v: self.lower_spin.setValue(float(v)))
        self.lower_spin.valueChanged.connect(self._enforce_order)
        self.lower_slider.valueChanged.connect(self._enforce_order)

        # link upper
        self.upper_spin.valueChanged.connect(lambda v: self.upper_slider.setValue(int(v)))
        self.upper_slider.valueChanged.connect(lambda v: self.upper_spin.setValue(float(v)))
        self.upper_spin.valueChanged.connect(self._enforce_order)
        self.upper_slider.valueChanged.connect(self._enforce_order)

        # Lower layout
        lower_layout = QHBoxLayout()
        lower_layout.addWidget(self.lower_spin)
        lower_layout.addWidget(self.lower_slider)
        layout.addRow("Lower Threshold:", lower_layout)

        # Upper layout
        upper_layout = QHBoxLayout()
        upper_layout.addWidget(self.upper_spin)
        upper_layout.addWidget(self.upper_slider)
        layout.addRow("Upper Threshold:", upper_layout)

        # Connected components (optional)
        if self.add_connected_components:
            self.use_connected_components_checkbox = QCheckBox("Use Connected Components")
            self.connected_components_spin = QSpinBox()
            self.connected_components_spin.setRange(1, 1000)
            self.connected_components_spin.setValue(100)
            self.connected_components_spin.setEnabled(False)

            self.use_connected_components_checkbox.toggled.connect(
                self.connected_components_spin.setEnabled
            )

            cc_layout = QHBoxLayout()
            cc_layout.addWidget(self.use_connected_components_checkbox)
            cc_layout.addWidget(QLabel("Max Components:"))
            cc_layout.addWidget(self.connected_components_spin)
            layout.addRow(cc_layout)

        # Preview button
        if show_preview_button:
            self.preview_btn = QPushButton("Preview Threshold")
            layout.addRow(self.preview_btn)
            self.preview_btn.clicked.connect(self._emit_preview)

        self.setLayout(layout)

    def _enforce_order(self):
        """do enforce order for both way"""
        lower = self.lower_spin.value()
        upper = self.upper_spin.value()
        sender = self.sender()

        if lower > upper:
            # if lower is greater than upper, set both to the same value
            if sender in (self.lower_spin, self.lower_slider):
                new_val = lower
            # if upper is greater than lower, set both to the same value
            else:
                new_val = upper

            # refresh the values without triggering signals
            self.lower_spin.blockSignals(True)
            self.lower_slider.blockSignals(True)
            self.upper_spin.blockSignals(True)
            self.upper_slider.blockSignals(True)

            self.lower_spin.setValue(new_val)
            self.lower_slider.setValue(int(new_val))
            self.upper_spin.setValue(new_val)
            self.upper_slider.setValue(int(new_val))

            self.lower_spin.blockSignals(False)
            self.lower_slider.blockSignals(False)
            self.upper_spin.blockSignals(False)
            self.upper_slider.blockSignals(False)
    def _emit_preview(self):
        lower = self.lower_spin.value()
        upper = self.upper_spin.value()
        self.preview_requested.emit(lower, upper)

    def get_thresholds(self):
        return self.lower_spin.value(), self.upper_spin.value()

    def set_thresholds(self, lower, upper):
        self.lower_spin.setValue(lower)
        self.upper_spin.setValue(upper)
        self.lower_slider.setValue(int(lower))
        self.upper_slider.setValue(int(upper))

    def set_range(self, min_val, max_val):
        self.lower_spin.setRange(min_val, max_val)
        self.upper_spin.setRange(min_val, max_val)
        self.lower_slider.setRange(int(min_val), int(max_val))
        self.upper_slider.setRange(int(min_val), int(max_val))

    def set_range_by_dtype(self, dtype):
        """Automatically set range by image dtype."""
        # if np.issubdtype(dtype, np.integer):
        #     max_val = np.iinfo(dtype).max
        # elif np.issubdtype(dtype, np.floating):
        #     max_val = 1.0
        # else:
        #     max_val = 255
        min_val, max_val = get_max_min_value_by_dtype(dtype)

        self.set_range(min_val, max_val)


    def run_preview_in_thread(self, image, callback=None):
        
        self.preview_btn.setEnabled(False)  # Disable button during processing
        
        lower = self.lower_spin.value()
        upper = self.upper_spin.value()
        use_cc = False
        cc_n = 100
        if self.add_connected_components and self.use_connected_components_checkbox.isChecked():
            use_cc = True
            cc_n = self.connected_components_spin.value()

        worker = PreviewWorker(image, lower, upper, use_cc, cc_n)

        if callback:
            worker.signals.finished.connect(callback)
        worker.signals.error.connect(lambda e: show_error(f"Preview error: {e}"))

        QThreadPool.globalInstance().start(worker)
    
        
        
    def apply_preview(self, image: np.ndarray):
        """
        Apply the current thresholds to the given image and return a binary mask.
        
        Parameters
        ----------
        image : np.ndarray
            Input image to apply thresholds on
            
        Returns
        -------
        binary : np.ndarray
            Binary mask after applying thresholds
        """
        lower = self.lower_spin.value()
        upper = self.upper_spin.value()
        
        if upper is None:
            binary =  image >= lower
        else:
            binary =  (image >= lower) & (image <= upper)
        
        
        if self.add_connected_components and self.use_connected_components_checkbox.isChecked():
            n_components = self.connected_components_spin.value()
            binary,_ = get_ccomps_with_size_order(binary, n_components)
        
        return binary


def get_max_min_value_by_dtype(dtype):
    if dtype == np.uint8:
        return 0, 255
    elif dtype == np.uint16:
        return 0, 65535
    elif dtype == np.uint32:
        return 0, 4294967295
    elif dtype == np.int8:
        return -128, 127
    elif dtype == np.int16:
        return -32768, 32767
    elif dtype == np.int32:
        return -2147483648, 2147483647
    else:
        raise ValueError(f"Unsupported image dtype: {dtype}")

def apply_threshold_preview(
    image: np.ndarray,
    threshold,
    upper_threshold = None
) -> np.ndarray:
    """
    Apply threshold to create a binary preview.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    threshold : float
        Lower threshold
    upper_threshold : float, optional
        Upper threshold
        
    Returns
    -------
    binary : np.ndarray
        Binary threshold result
    """
    if upper_threshold is None:
        return image >= threshold
    else:
        return (image >= threshold) & (image <= upper_threshold)

class MainSeedParamWidget(QGroupBox):
    def __init__(self, title="Parameters", viewer=None, image_combo=None):
        super().__init__(title)

        self.viewer = viewer
        self.image_combo = image_combo  # ComboBox for selecting image layer
        # self.show_touch_rule = show_touch_rule
        
        layout = QVBoxLayout()

        # Top rowlayout 
        top_row_layout = QHBoxLayout()
        
        # second row layout
        second_row_layout = QHBoxLayout()

        # seed method
        top_row_layout.addWidget(QLabel("Seed Method:"))
        self.seed_method_combo = QComboBox()
        self.seed_method_combo.addItems(["Original", "Adaptive (Erosion)", "Adaptive (Thresholds)"])
        self.seed_method_combo.setFixedWidth(150)
        top_row_layout.addWidget(self.seed_method_combo)

        # erosion steps
        second_row_layout.addWidget(QLabel("Erosion Steps:"))
        self.erosion_spin = QSpinBox()
        self.erosion_spin.setRange(0, 1000)
        self.erosion_spin.setValue(0)
        self.erosion_spin.setFixedWidth(40)
        second_row_layout.addWidget(self.erosion_spin)
        
        # Threads
        top_row_layout.addWidget(QLabel("Threads:"))
        self.thread_spin = QSpinBox()
        self.thread_spin.setMinimum(1)
        self.thread_spin.setMaximum(os.cpu_count())
        self.thread_spin.setValue(min(4, os.cpu_count() - 1))
        self.thread_spin.setFixedWidth(30)
        top_row_layout.addWidget(self.thread_spin)




        second_row_layout.addWidget(QLabel("Segments:"))
        self.segments_spin = QSpinBox()
        self.segments_spin.setRange(1, 1000)
        self.segments_spin.setValue(10)
        self.segments_spin.setFixedWidth(40)
        second_row_layout.addWidget(self.segments_spin)

    
        

        top_row_layout.addStretch()
        layout.addLayout(top_row_layout)
        second_row_layout.addStretch()
        layout.addLayout(second_row_layout) 


        

        

        # #  Threshold Table
        # self.threshold_table = QTableWidget()
        # self.threshold_table.setColumnCount(2)
            
        # self.threshold_table.setHorizontalHeaderLabels(
        #     ["Lower Threshold", "Upper Threshold"]
        # )
        
        # header = self.threshold_table.horizontalHeader()
        # header.setSectionResizeMode(QHeaderView.Stretch)    
        
        # self.threshold_table.setMaximumHeight(200)
        # layout.addWidget(self.threshold_table)

        # # ▶️ Add/Remove Buttons
        # btn_layout = QHBoxLayout()
        # self.add_threshold_btn = QPushButton("Add Threshold")
        # self.remove_threshold_btn = QPushButton("Remove Selected")
        # btn_layout.addWidget(self.add_threshold_btn)
        # btn_layout.addWidget(self.remove_threshold_btn)
        # btn_layout.addStretch()
        # layout.addLayout(btn_layout)

        # self.footprint_table = QTableWidget()
        # self.footprint_table.setColumnCount(1)
        # self.footprint_table.setHorizontalHeaderLabels(["Footprint"])
        # self.footprint_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  
        
        # layout.addWidget(self.footprint_table)
        # # Add a button to add a new threshold row
        # self.add_footprint_btn = QPushButton("Add Footprint")
        # self.remove_footprint_btn = QPushButton("Remove Selected")
        # self.remove_footprint_btn_all = QPushButton("Remove All Footprints")
        # footprint_btn_layout = QHBoxLayout()
        
        # footprint_btn_layout.addWidget(self.add_footprint_btn)
        # footprint_btn_layout.addWidget(self.remove_footprint_btn)
        # footprint_btn_layout.addWidget(self.remove_footprint_btn_all)
        # footprint_btn_layout.addStretch()
        # layout.addLayout(footprint_btn_layout)
    
        
        # === Threshold Table + Buttons as VBox ===
        threshold_vbox = QVBoxLayout()
        self.threshold_table = QTableWidget()
        self.threshold_table.setColumnCount(2)
        self.threshold_table.setHorizontalHeaderLabels(["Lower Threshold", "Upper Threshold"])
        self.threshold_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.threshold_table.setMaximumHeight(200)

        threshold_vbox.addWidget(self.threshold_table)

        # Buttons for threshold table
        btn_layout = QHBoxLayout()
        self.add_threshold_btn = QPushButton("Add Threshold")
        self.remove_threshold_btn = QPushButton("Remove Selected")
        btn_layout.addWidget(self.add_threshold_btn)
        btn_layout.addWidget(self.remove_threshold_btn)
        btn_layout.addStretch()
        threshold_vbox.addLayout(btn_layout)

        # === Footprint Table + Buttons as VBox ===
        footprint_vbox = QVBoxLayout()

        self.footprint_table = QTableWidget()
        self.footprint_table.setColumnCount(1)
        self.footprint_table.setHorizontalHeaderLabels(["Footprint"])
        self.footprint_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.footprint_table.setMaximumHeight(200)
        self.footprint_table.setMaximumWidth(150)

        footprint_vbox.addWidget(self.footprint_table)

        footprint_btn_layout = QHBoxLayout()
        self.add_footprint_btn = QPushButton("Add Footprint")
        self.remove_footprint_btn = QPushButton("Remove Selected")
        self.remove_footprint_btn_all = QPushButton("Remove All")
        footprint_btn_layout.addWidget(self.add_footprint_btn)
        footprint_btn_layout.addWidget(self.remove_footprint_btn)
        footprint_btn_layout.addWidget(self.remove_footprint_btn_all)
        footprint_btn_layout.addStretch()
        footprint_vbox.addLayout(footprint_btn_layout)

        # === Combine into a HBox ===
        combined_table_layout = QHBoxLayout()
        combined_table_layout.addLayout(threshold_vbox)
        combined_table_layout.addLayout(footprint_vbox)

        # Add to main layout
        layout.addLayout(combined_table_layout)



        self.setLayout(layout)
        
        self.add_threshold_btn.clicked.connect(self._add_threshold_row_adaptive)
        self.remove_threshold_btn.clicked.connect(self._remove_threshold_row)
        
        self.add_footprint_btn.clicked.connect(self._add_footprint_row)
        self.remove_footprint_btn.clicked.connect(self._remove_footprint_row)
        self.remove_footprint_btn_all.clicked.connect(lambda: self.footprint_table.setRowCount(0))
        

    def get_params(self):
        # return parameters as a dictionary
        
        thresholds, upper_thresholds = self._get_threshold_params()
        footprints = self._get_footprints()
        # assert footprints is len 1 or len == erosion_steps

        # Handle footprints: if empty, set to None (uses preset footprints)
        # If provided, must be 1 (repeated) or match erosion_steps count
        if len(footprints) == 0:
            footprints = None
        elif len(footprints) == 1:
            footprints = footprints * self.erosion_spin.value()
        elif len(footprints) != self.erosion_spin.value():
            raise ValueError(
                f"Please add {self.erosion_spin.value()} footprints, got {len(footprints)}"
            )

        return {
            "num_threads": self.thread_spin.value(),
            "segments": self.segments_spin.value(),
            "thresholds": thresholds,
            "upper_thresholds": upper_thresholds,
            "erosion_steps": self.erosion_spin.value(),
            "footprints": footprints,
            "seed_method": self.seed_method_combo.currentText() if self.seed_method_combo else None,
        }

    def set_params(self, params):
        """Set parameters for the widget from a dictionary."""
        if "seed_method" in params:
            index = self.seed_method_combo.findText(params["seed_method"], Qt.MatchFixedString)
            if index >= 0:
                self.seed_method_combo.setCurrentIndex(index)
        
        if "erosion_steps" in params:
            self.erosion_spin.setValue(params["erosion_steps"])

        if "num_threads" in params:
            self.thread_spin.setValue(params["num_threads"])

        if "segments" in params:
            self.segments_spin.setValue(params["segments"])

    def _get_footprints(self):
        """Get footprints from the footprint table."""
        footprints = []
        for row in range(self.footprint_table.rowCount()):
            footprint_widget = self.footprint_table.cellWidget(row, 0)
            footprints.append(footprint_widget.currentText())
        return footprints
    
    def _get_threshold_params(self):
        """Get threshold parameters from table."""
        thresholds = []
        upper_thresholds = []

        
        for row in range(self.threshold_table.rowCount()):
            # Lower threshold
            lower_widget = self.threshold_table.cellWidget(row, 0)
            thresholds.append(lower_widget.value())
            
            # Upper threshold
            upper_widget = self.threshold_table.cellWidget(row, 1)
            upper_thresholds.append(upper_widget.value())

            
        return thresholds, upper_thresholds

    def _add_footprint_row(self):
        """Add a new footprint row to the table."""

        
        if self.image_combo is None or not self.image_combo.currentText():
            show_error("Please select an image layer first")
            return

        row = self.footprint_table.rowCount()
        self.footprint_table.insertRow(row)

        footprint_combo = QComboBox()

        current_image = self.viewer.layers[self.image_combo.currentText()].data
        
        if current_image.ndim == 2:
            footprint_combo.addItems(support_footprints_2d)
        elif current_image.ndim == 3:
            footprint_combo.addItems(support_footprints)
        
        self.footprint_table.setCellWidget(row, 0, footprint_combo)
    
    def _remove_footprint_row(self):
        """Remove selected footprint row."""
        current_row = self.footprint_table.currentRow()
        if current_row >= 0:
            self.footprint_table.removeRow(current_row)

    def _add_threshold_row_adaptive(self, dilate=5):
        """Add a new threshold row to the table, auto-suggesting values from previous row or image dtype."""
 

        # not image
        if self.image_combo is None or not self.image_combo.currentText():
            show_error("Please select an image layer first")
            return

        row = self.threshold_table.rowCount()
        self.threshold_table.insertRow(row)

        current_image = self.viewer.layers[self.image_combo.currentText()].data

        # set default values based on previous row or image dtype
        if row == 0:
            # upper_value = self._get_img_dtype_max(current_image.dtype)
            # lower_value = 0
            lower_value, upper_value = get_max_min_value_by_dtype(current_image.dtype)
        else:
            lower_widget = self.threshold_table.cellWidget(row - 1, 0)
            lower_value = lower_widget.value()
            upper_widget = self.threshold_table.cellWidget(row - 1, 1)
            upper_value = upper_widget.value()

        # set the range for the spinboxes
        lower_min , upper_max = get_max_min_value_by_dtype(current_image.dtype)
        lower_max = upper_value
        upper_min = lower_value
        # upper_max = self._get_img_dtype_max(current_image.dtype)

        # Lower threshold
        lower_spin = QSpinBox()
        lower_spin.setRange(lower_min, lower_max)
        lower_spin.setValue(lower_value)
        self.threshold_table.setCellWidget(row, 0, lower_spin)

        # Upper threshold
        upper_spin = QSpinBox()
        upper_spin.setRange(upper_min, upper_max)
        upper_spin.setValue(upper_value)
        self.threshold_table.setCellWidget(row, 1, upper_spin)

     
        
    

    def _remove_threshold_row(self):
        """Remove selected threshold row."""
        current_row = self.threshold_table.currentRow()
        if current_row >= 0:
            self.threshold_table.removeRow(current_row)

    def _get_img_dtype_max(self, image_dtype):
        """Set the range of the threshold spinboxes based on the image dtype."""
        if image_dtype == np.uint8:
            max_value = 255
        elif image_dtype == np.uint16:
            max_value = 65535
        elif image_dtype == np.uint32 or image_dtype == np.int64:
            max_value = 4294967295
        elif image_dtype == np.float32 or image_dtype == np.float64:
            max_value = 1.0
        else:
            raise ValueError(f"Unsupported image dtype: {image_dtype}")
        return max_value
    
    def clean_table(self):
        """Clear all rows in the threshold table."""
        self.threshold_table.setRowCount(0)
        self._add_threshold_row_adaptive()
        self.footprint_table.setRowCount(0)
        self._add_footprint_row()

    def populate_thresholds_from_list(self, thresholds_list, upper_thresholds_list=None):
        """Populate the thresholds table with a list of thresholds from YAML import.
        
        Parameters
        ----------
        thresholds_list : list
            List of lower threshold values
        upper_thresholds_list : list, optional
            List of upper threshold values. If None, will use 255 or threshold+100
        """
        if not thresholds_list:
            return
            
        # Clear existing rows
        self.threshold_table.setRowCount(0)
        
        # Check if we have an image to get dtype info
        current_image = None
        if self.image_combo and self.image_combo.currentText():
            current_image = self.viewer.layers[self.image_combo.currentText()].data
        
        for i, lower_threshold in enumerate(thresholds_list):
            row = self.threshold_table.rowCount()
            self.threshold_table.insertRow(row)
            
            # Determine upper threshold
            if upper_thresholds_list and i < len(upper_thresholds_list):
                upper_threshold = upper_thresholds_list[i]
            else:
                # Use 255 or threshold+100 if threshold > 255
                upper_threshold = 255 if lower_threshold <= 255 else lower_threshold + 100
            
            # Set range based on image dtype if available
            if current_image is not None:
                lower_min, upper_max = get_max_min_value_by_dtype(current_image.dtype)
            else:
                lower_min, upper_max = 0, 65535  # Default range
            
            # Lower threshold spinbox
            lower_spin = QSpinBox()
            lower_spin.setRange(lower_min, upper_threshold)
            lower_spin.setValue(int(lower_threshold))
            self.threshold_table.setCellWidget(row, 0, lower_spin)
            
            # Upper threshold spinbox
            upper_spin = QSpinBox()
            upper_spin.setRange(int(lower_threshold), upper_max)
            upper_spin.setValue(int(upper_threshold))
            self.threshold_table.setCellWidget(row, 1, upper_spin)


class MainGrowParamWidget(QGroupBox):
    def __init__(self, title="Parameters", viewer=None, image_combo=None,
                 mode = "grow",  # "grow" or "seed"
                 colnames=None):
        super().__init__(title)

        self.viewer = viewer
        self.image_combo = image_combo  # ComboBox for selecting image layer
        # self.show_touch_rule = show_touch_rule
        
        self.mode = mode  # "grow" or "seed"
        
        layout = QVBoxLayout()

        # Top rowlayout 
        top_row_layout = QHBoxLayout()

        # Threads
        top_row_layout.addWidget(QLabel("Threads:"))
        self.thread_spin = QSpinBox()
        self.thread_spin.setMinimum(1)
        self.thread_spin.setMaximum(os.cpu_count())
        self.thread_spin.setValue(min(4, os.cpu_count() - 1))
        self.thread_spin.setFixedWidth(80)
        top_row_layout.addWidget(self.thread_spin)

        # Add some space between the two sections
        top_row_layout.addSpacing(20)

 
        top_row_layout.addWidget(QLabel("Touch Rule:"))
        self.touch_rule_combo = QComboBox()
        self.touch_rule_combo.addItems(["stop", "overwrite"])
        self.touch_rule_combo.setFixedWidth(100)
        top_row_layout.addWidget(self.touch_rule_combo)




        top_row_layout.addStretch()
        layout.addLayout(top_row_layout)

        # ▶️ Threshold Table
        self.threshold_table = QTableWidget()
        self.threshold_table.setColumnCount(3)
        
        
        
        self.threshold_table.setHorizontalHeaderLabels(
            ["Lower Threshold", "Upper Threshold", "Dilation Steps"])


        
        header = self.threshold_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)    
        
        self.threshold_table.setMaximumHeight(200)
        layout.addWidget(self.threshold_table)

        # ▶️ Add/Remove Buttons
        btn_layout = QHBoxLayout()
        self.add_threshold_btn = QPushButton("Add Threshold")
        self.remove_threshold_btn = QPushButton("Remove Selected")
        self.remove_all_threshold_btn = QPushButton("Remove All Thresholds")
        btn_layout.addWidget(self.add_threshold_btn)
        btn_layout.addWidget(self.remove_threshold_btn)
        btn_layout.addWidget(self.remove_all_threshold_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        
        self.add_threshold_btn.clicked.connect(self._add_threshold_row_adaptive)
        self.remove_threshold_btn.clicked.connect(self._remove_threshold_row)
        self.remove_all_threshold_btn.clicked.connect(lambda: self.footprint_table.setRowCount(0))

    def get_params(self):
        # return parameters as a dictionary
        
        thresholds, upper_thresholds, steps = self._get_threshold_params()
        

        return {
            "num_threads": self.thread_spin.value(),
            "touch_rule": self.touch_rule_combo.currentText() if self.touch_rule_combo else None,
            "thresholds": thresholds,
            "upper_thresholds": upper_thresholds,
            "dilation_steps": steps
        }


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
            upper_thresholds.append(upper_widget.value())

            
            # Dilate iterations
            dilate_widget = self.threshold_table.cellWidget(row, 2)
            dilate_iters.append(dilate_widget.value())
            
        
        return thresholds, upper_thresholds, dilate_iters

    def _add_threshold_row_adaptive(self, dilate=5):
        """Add a new threshold row to the table, auto-suggesting values from previous row or image dtype."""
        row = self.threshold_table.rowCount()
        self.threshold_table.insertRow(row)

        # not image
        if self.image_combo is None or not self.image_combo.currentText():
            show_error("Please select an image layer first")
            return

        current_image = self.viewer.layers[self.image_combo.currentText()].data

        # set default values based on previous row or image dtype
        if row == 0:
            # upper_value = self._get_img_dtype_max(current_image.dtype)
            # lower_value = 0
            lower_value, upper_value = get_max_min_value_by_dtype(current_image.dtype)
        else:
            lower_widget = self.threshold_table.cellWidget(row - 1, 0)
            lower_value = lower_widget.value()
            upper_widget = self.threshold_table.cellWidget(row - 1, 1)
            upper_value = upper_widget.value()
            dilate_widget = self.threshold_table.cellWidget(row - 1, 2)
            dilate = dilate_widget.value()

        # set the range for the spinboxes
        # lower_min = 0
        lower_max = upper_value
        upper_min = lower_value
        lower_min, upper_max = get_max_min_value_by_dtype(current_image.dtype)
        # upper_max = self._get_img_dtype_max(current_image.dtype)

        # Lower threshold
        lower_spin = QSpinBox()
        lower_spin.setRange(lower_min, lower_max)
        lower_spin.setValue(lower_value)
        self.threshold_table.setCellWidget(row, 0, lower_spin)

        # Upper threshold
        upper_spin = QSpinBox()
        upper_spin.setRange(upper_min, upper_max)
        upper_spin.setValue(upper_value)
        self.threshold_table.setCellWidget(row, 1, upper_spin)

        # Dilate iterations
        dilate_spin = QSpinBox()
        dilate_spin.setRange(1, 1000)
        dilate_spin.setValue(dilate)
        self.threshold_table.setCellWidget(row, 2, dilate_spin)
        
        
        

    def _remove_threshold_row(self):
        """Remove selected threshold row."""
        current_row = self.threshold_table.currentRow()
        if current_row >= 0:
            self.threshold_table.removeRow(current_row)

    def _get_img_dtype_max(self, image_dtype):
        """Set the range of the threshold spinboxes based on the image dtype."""
        if image_dtype == np.uint8 or image_dtype == np.int8:
            max_value = 255
        elif image_dtype == np.uint16 or image_dtype == np.int16:
            max_value = 65535
        elif image_dtype == np.uint32 or image_dtype == np.int32:
            max_value = 4294967295
        elif image_dtype == np.float32 or image_dtype == np.float64:
            max_value = 1.0
        else:
            raise ValueError(f"Unsupported image dtype: {image_dtype}")
        return max_value
    
    
    def clean_table(self):
        """Clear all rows in the threshold table."""
        self.threshold_table.setRowCount(0)
        self._add_threshold_row_adaptive()





class SeedOptionalParamGroupBox(QGroupBox):
    def __init__(self, default_output=None):
        super().__init__("Advanced Seed Parameters")
        layout = QFormLayout()

        # sort (bool)
        self.sort_checkbox = QCheckBox("Enable sorting")
        self.sort_checkbox.setChecked(True)
        layout.addRow("Sort segments", self.sort_checkbox)

        # min_size (int)
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(1, 10000)
        self.min_size_spin.setValue(5)
        layout.addRow("Min segment size", self.min_size_spin)


        self.no_split_spin = QSpinBox()
        self.no_split_spin.setRange(0, 100)
        self.no_split_spin.setValue(3)
        self.no_split_spin.setToolTip("Max no-split iterations")

        self.min_split_ratio_spin = QDoubleSpinBox()
        self.min_split_ratio_spin.setDecimals(3)
        self.min_split_ratio_spin.setRange(0.0, 1.0)
        self.min_split_ratio_spin.setSingleStep(0.01)
        self.min_split_ratio_spin.setValue(0.01)
        self.min_split_ratio_spin.setToolTip("Min split ratio")

        self.min_split_total_ratio_spin = QDoubleSpinBox()
        self.min_split_total_ratio_spin.setDecimals(3)
        self.min_split_total_ratio_spin.setRange(0.0, 1.0)
        self.min_split_total_ratio_spin.setSingleStep(0.01)
        self.min_split_total_ratio_spin.setValue(0.0)
        self.min_split_total_ratio_spin.setToolTip("Min total split ratio")

        # add to row layout
        no_split_row_layout = QHBoxLayout()
        no_split_row_layout.addWidget(QLabel("No-split Iter:"))
        no_split_row_layout.addWidget(self.no_split_spin)
        # no_split_row_layout.addSpacing(8)
        no_split_row_layout.addWidget(QLabel("Min Ratio:"))
        no_split_row_layout.addWidget(self.min_split_ratio_spin)
        # no_split_row_layout.addSpacing(8)
        no_split_row_layout.addWidget(QLabel("Total Ratio:"))
        no_split_row_layout.addWidget(self.min_split_total_ratio_spin)

        # add to the split parameters row
        layout.addRow("Split Parameters", no_split_row_layout)

        # # min_split_ratio (float)
        # self.min_split_ratio_spin = QDoubleSpinBox()
        # self.min_split_ratio_spin.setDecimals(3)
        # self.min_split_ratio_spin.setRange(0.0, 1.0)
        # self.min_split_ratio_spin.setSingleStep(0.01)
        # self.min_split_ratio_spin.setValue(0.01)
        # layout.addRow("Min split ratio", self.min_split_ratio_spin)

        # # min_split_total_ratio (float)
        # self.min_split_total_ratio_spin = QDoubleSpinBox()
        # self.min_split_total_ratio_spin.setDecimals(3)
        # self.min_split_total_ratio_spin.setRange(0.0, 1.0)
        # self.min_split_total_ratio_spin.setSingleStep(0.01)
        # self.min_split_total_ratio_spin.setValue(0.0)
        # layout.addRow("Min total split ratio", self.min_split_total_ratio_spin)


        # split_size_limit (tuple of float or None)
        self.size_lower_line = QLineEdit()
        self.size_upper_line = QLineEdit()
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Lower"))
        size_layout.addWidget(self.size_lower_line)
        size_layout.addWidget(QLabel("Upper"))
        size_layout.addWidget(self.size_upper_line)
        layout.addRow("Split size limit", size_layout)

        # split_convex_hull_limit (tuple of float or None)
        self.hull_lower_line = QLineEdit()
        self.hull_upper_line = QLineEdit()
        hull_layout = QHBoxLayout()
        hull_layout.addWidget(QLabel("Lower"))
        hull_layout.addWidget(self.hull_lower_line)
        hull_layout.addWidget(QLabel("Upper"))
        hull_layout.addWidget(self.hull_upper_line)
        layout.addRow("Convex hull limit", hull_layout)

        self.setLayout(layout)
    
    def set_params(self, params):
        """Set advanced parameters for the widget from a dictionary."""
        if 'sort' in params:
            self.sort_checkbox.setChecked(params['sort'])
        if 'min_size' in params:
            self.min_size_spin.setValue(params['min_size'])
        if 'no_split_max_iter' in params:
            self.no_split_spin.setValue(params['no_split_max_iter'])
        if 'min_split_ratio' in params:
            self.min_split_ratio_spin.setValue(params['min_split_ratio'])
        if 'min_split_total_ratio' in params:
            self.min_split_total_ratio_spin.setValue(params['min_split_total_ratio'])

        if 'split_size_limit' in params and params['split_size_limit'] is not None:
            lower, upper = params['split_size_limit']
            self.size_lower_line.setText(str(lower) if lower is not None else "")
            self.size_upper_line.setText(str(upper) if upper is not None else "")

        if 'split_convex_hull_limit' in params and params['split_convex_hull_limit'] is not None:
            lower, upper = params['split_convex_hull_limit']
            self.hull_lower_line.setText(str(lower) if lower is not None else "")
            self.hull_upper_line.setText(str(upper) if upper is not None else "")

    def get_params(self):
        def parse_optional_float(text):
            try:
                return float(text)
            except ValueError:
                return None

        return {
            "sort": self.sort_checkbox.isChecked(),
            "no_split_max_iter": self.no_split_spin.value(),
            "min_size": self.min_size_spin.value(),
            "min_split_ratio": self.min_split_ratio_spin.value(),
            "min_split_total_ratio": self.min_split_total_ratio_spin.value(),
            "split_size_limit": (
                parse_optional_float(self.size_lower_line.text()),
                parse_optional_float(self.size_upper_line.text())
            ),
            "split_convex_hull_limit": (
                parse_optional_float(self.hull_lower_line.text()),
                parse_optional_float(self.hull_upper_line.text())
            )
        }

class GrowOptionalParamGroupBox(QGroupBox):
    def __init__(self, default_output=None):
        super().__init__("Advanced Grow Parameters")


        layout = QFormLayout()

        # Save every n iterations
        self.save_iter_checkbox = QCheckBox("Save every N iterations")
        self.save_iter_checkbox.stateChanged.connect(self._toggle_save_iters)
        self.save_every_n_spin = QSpinBox()
        self.save_every_n_spin.setRange(0, 100)
        self.save_every_n_spin.setEnabled(False)
        layout.addRow(self.save_iter_checkbox, self.save_every_n_spin)

        # Grow to end
        self.grow_to_end_checkbox = QCheckBox("Grow to end")
        layout.addRow(self.grow_to_end_checkbox)

        # Sort seed ids
        self.sort_checkbox = QCheckBox("Sort the result by size")
        layout.addRow(self.sort_checkbox)

        # Grow specific IDs
        self.id_list_line = QLineEdit()
        self.id_list_line.setPlaceholderText("e.g. 1,3,5")
        layout.addRow(QLabel("IDs to Grow (optional):"), self.id_list_line)

        self.early_stop_checkbox = QCheckBox("Early stop if no growth")
        self.early_stop_checkbox.setToolTip("Stop growing if no growth occurs for a specified number of iterations.")
        self.early_stop_checkbox.setChecked(True)
        self.early_stop_checkbox.stateChanged.connect(self._toggle_no_growth_spin)
        layout.addRow(self.early_stop_checkbox)
        # Max no-growth iters
        self.no_growth_spin = QSpinBox()
        self.no_growth_spin.setRange(1, 100)
        self.no_growth_spin.setValue(3)
        self.no_growth_spin.setEnabled(True)
        layout.addRow(QLabel("Max No-Growth Iterations:"), self.no_growth_spin)

        # Min growth size
        self.min_growth_spin = QSpinBox()
        self.min_growth_spin.setRange(0, 1000000)
        self.min_growth_spin.setValue(50)
        self.min_growth_spin.setEnabled(True)
        layout.addRow(QLabel("Minimum Growth Size:"), self.min_growth_spin)



        self.setLayout(layout)

    def _toggle_no_growth_spin(self):
        self.no_growth_spin.setEnabled(self.early_stop_checkbox.isChecked())
        # if not self.early_stop_checkbox.isChecked():
        #     self.no_growth_spin.setValue(3)
        self.min_growth_spin.setEnabled(self.early_stop_checkbox.isChecked())
    
    def _browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder_line.setText(folder)

    def _toggle_save_iters(self):
        self.save_every_n_spin.setEnabled(self.save_iter_checkbox.isChecked())

    def get_params(self):
        to_grow_text = self.id_list_line.text().strip()
        to_grow_text.replace(" ", "")  # remove any spaces

        try:
            to_grow_ids = [int(x) for x in to_grow_text.split(",") if x.strip().isdigit()]
        except:
            to_grow_ids = None
            # show error in napari message box
            from napari.utils.notifications import show_error
            show_error("Invalid IDs format. Please enter a comma-separated list of integers.")
       
        # if to_grow_ids is empty, set to None
        if not to_grow_ids:
            to_grow_ids = None
       
        if not self.early_stop_checkbox.isChecked():
            no_growth_max_iter = None
        else:
            no_growth_max_iter = self.no_growth_spin.value()
        
        return {
            "save_every_n_iters": self.save_every_n_spin.value() if self.save_iter_checkbox.isChecked() else None,
            "grow_to_end": self.grow_to_end_checkbox.isChecked(),
            "is_sort": self.sort_checkbox.isChecked(),
            "to_grow_ids": to_grow_ids,
            "min_growth_size": self.min_growth_spin.value(),
            "no_growth_max_iter": no_growth_max_iter
        }
        

## Deprecated classes, kept for reference

class MainParamWidget(QGroupBox):
    def __init__(self, title="Parameters", viewer=None, image_combo=None,
                 mode = "grow",  # "grow" or "seed"
                 colnames=None):
        super().__init__(title)

        self.viewer = viewer
        self.image_combo = image_combo  # ComboBox for selecting image layer
        # self.show_touch_rule = show_touch_rule
        
        self.mode = mode  # "grow" or "seed"
        
        layout = QVBoxLayout()

        # Top rowlayout 
        top_row_layout = QHBoxLayout()

        # Threads
        top_row_layout.addWidget(QLabel("Threads:"))
        self.thread_spin = QSpinBox()
        self.thread_spin.setMinimum(1)
        self.thread_spin.setMaximum(os.cpu_count())
        self.thread_spin.setValue(min(4, os.cpu_count() - 1))
        self.thread_spin.setFixedWidth(80)
        top_row_layout.addWidget(self.thread_spin)

        # Add some space between the two sections
        top_row_layout.addSpacing(20)

        # Touch Rule (if shown)
        if self.mode == "grow":
            top_row_layout.addWidget(QLabel("Touch Rule:"))
            self.touch_rule_combo = QComboBox()
            self.touch_rule_combo.addItems(["stop", "overwrite"])
            self.touch_rule_combo.setFixedWidth(100)
            top_row_layout.addWidget(self.touch_rule_combo)
        else:
            self.touch_rule_combo = None

        if self.mode == "seed":
            top_row_layout.addWidget(QLabel("Segments:"))
            self.segments_spin = QSpinBox()
            self.segments_spin.setRange(1, 1000)
            self.segments_spin.setValue(10)
            self.segments_spin.setMaximumWidth(80)
            top_row_layout.addWidget(self.segments_spin)
        else:
            self.segments_spin = None
            

        top_row_layout.addStretch()
        layout.addLayout(top_row_layout)

        # ▶️ Threshold Table
        self.threshold_table = QTableWidget()
        self.threshold_table.setColumnCount(4)
        
        
        
        if colnames is not None:
            self.threshold_table.setHorizontalHeaderLabels(colnames)
        else:
            self.threshold_table.setHorizontalHeaderLabels(
                ["Lower Threshold", "Upper Threshold", "Steps", "Footprint"]
            )


        
        header = self.threshold_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)    
        
        self.threshold_table.setMaximumHeight(200)
        layout.addWidget(self.threshold_table)

        # ▶️ Add/Remove Buttons
        btn_layout = QHBoxLayout()
        self.add_threshold_btn = QPushButton("Add Threshold")
        self.remove_threshold_btn = QPushButton("Remove Selected")
        btn_layout.addWidget(self.add_threshold_btn)
        btn_layout.addWidget(self.remove_threshold_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        
        self.add_threshold_btn.clicked.connect(self._add_threshold_row_adaptive)
        self.remove_threshold_btn.clicked.connect(self._remove_threshold_row)

    def get_params(self):
        # return parameters as a dictionary
        
        thresholds, upper_thresholds, steps , footprints = self._get_threshold_params()
        
        if self.mode == "grow":
            return {
                "threads": self.thread_spin.value(),
                "touch_rule": self.touch_rule_combo.currentText() if self.touch_rule_combo else None,
                "thresholds": thresholds,
                "upper_thresholds": upper_thresholds,
                "dilation_steps": steps,
                'footprints': footprints
            }
        elif self.mode == "seed":
            return {
                "threads": self.thread_spin.value(),
                "segments": self.segments_spin.value(),
                "thresholds": thresholds,
                "upper_thresholds": upper_thresholds,
                "erosion_steps": steps,
                "footprints": footprints
            }

    def _get_threshold_params(self):
        """Get threshold parameters from table."""
        thresholds = []
        upper_thresholds = []
        dilate_iters = []
        footprints = []
        
        for row in range(self.threshold_table.rowCount()):
            # Lower threshold
            lower_widget = self.threshold_table.cellWidget(row, 0)
            thresholds.append(lower_widget.value())
            
            # Upper threshold
            upper_widget = self.threshold_table.cellWidget(row, 1)
            upper_thresholds.append(upper_widget.value())

            
            # Dilate iterations
            dilate_widget = self.threshold_table.cellWidget(row, 2)
            dilate_iters.append(dilate_widget.value())
            

            footprint_widget = self.threshold_table.cellWidget(row, 3)
            footprints.append(footprint_widget.currentText())
        
        return thresholds, upper_thresholds, dilate_iters , footprints

    def _add_threshold_row_adaptive(self, dilate=5):
        """Add a new threshold row to the table, auto-suggesting values from previous row or image dtype."""
        row = self.threshold_table.rowCount()
        self.threshold_table.insertRow(row)

        # not image
        if self.image_combo is None or not self.image_combo.currentText():
            show_error("Please select an image layer first")
            return

        current_image = self.viewer.layers[self.image_combo.currentText()].data

        # set default values based on previous row or image dtype
        if row == 0:
            upper_value = self._get_img_dtype_max(current_image.dtype)
            lower_value = 0
        else:
            lower_widget = self.threshold_table.cellWidget(row - 1, 0)
            lower_value = lower_widget.value()
            upper_widget = self.threshold_table.cellWidget(row - 1, 1)
            upper_value = upper_widget.value()
            dilate_widget = self.threshold_table.cellWidget(row - 1, 2)
            dilate = dilate_widget.value()

        # set the range for the spinboxes
        lower_min = 0
        lower_max = upper_value
        upper_min = lower_value
        upper_max = self._get_img_dtype_max(current_image.dtype)

        # Lower threshold
        lower_spin = QSpinBox()
        lower_spin.setRange(lower_min, lower_max)
        lower_spin.setValue(lower_value)
        self.threshold_table.setCellWidget(row, 0, lower_spin)

        # Upper threshold
        upper_spin = QSpinBox()
        upper_spin.setRange(upper_min, upper_max)
        upper_spin.setValue(upper_value)
        self.threshold_table.setCellWidget(row, 1, upper_spin)

        # Dilate iterations
        dilate_spin = QSpinBox()
        dilate_spin.setRange(1, 1000)
        dilate_spin.setValue(dilate)
        self.threshold_table.setCellWidget(row, 2, dilate_spin)
        
        # Footprint
        footprint_combo = QComboBox()
        # footprint_combo.addItems("ball")
        
        if current_image.ndim == 2 or current_image.shape[0] == 1:
            footprint_combo.addItems(support_footprints_2d)
        elif current_image.ndim == 3:
            footprint_combo.addItems(support_footprints)
        
        self.threshold_table.setCellWidget(row, 3, footprint_combo)
        
        

    def _remove_threshold_row(self):
        """Remove selected threshold row."""
        current_row = self.threshold_table.currentRow()
        if current_row >= 0:
            self.threshold_table.removeRow(current_row)

    def _get_img_dtype_max(self, image_dtype):
        """Set the range of the threshold spinboxes based on the image dtype."""
        if image_dtype == np.uint8:
            max_value = 255
        elif image_dtype == np.uint16:
            max_value = 65535
        elif image_dtype == np.uint32:
            max_value = 4294967295
        elif image_dtype == np.float32 or image_dtype == np.float64:
            max_value = 1.0
        else:
            max_value = 255
        return max_value
    
    def clean_table(self):
        """Clear all rows in the threshold table."""
        self.threshold_table.setRowCount(0)
        self._add_threshold_row_adaptive()

    def _add_threshold_row(self, lower=100, upper=None, dilate=5):
        """depracated: use _add_threshold_row_adaptive instead
        Add a new threshold row to the table."""
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
        upper_spin.setValue(upper if upper else 65535)
        upper_spin.setEnabled(upper is not None)
        
        # use_upper.setChecked(upper is not None)
        # use_upper.toggled.connect(upper_spin.setEnabled)
        
        # upper_layout.addWidget(use_upper)
        upper_layout.addWidget(upper_spin)
        upper_widget.setLayout(upper_layout)
        
        self.threshold_table.setCellWidget(row, 1, upper_widget)
        
        # Dilate iterations
        dilate_spin = QSpinBox()
        dilate_spin.setRange(1, 1000)
        dilate_spin.setValue(dilate)
        self.threshold_table.setCellWidget(row, 2, dilate_spin)
    
