import os
from qtpy.QtWidgets import (
    QGroupBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,QDoubleSpinBox,
    QSpinBox, QCheckBox, QFileDialog , QVBoxLayout , QComboBox , QTableWidget
)

from napari.utils.notifications import show_error, show_info
import numpy as np

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

        top_row_layout.addStretch()
        layout.addLayout(top_row_layout)

        # ▶️ Threshold Table
        self.threshold_table = QTableWidget()
        self.threshold_table.setColumnCount(3)
        
        if colnames is not None:
            self.threshold_table.setHorizontalHeaderLabels(colnames)
        else:
            self.threshold_table.setHorizontalHeaderLabels(
                ["Lower Threshold", "Upper Threshold", "Steps"]
            )
        self.threshold_table.horizontalHeader().setStretchLastSection(True)
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
        
        thresholds, upper_thresholds, steps = self._get_threshold_params()
        
        if self.mode == "grow":
            return {
                "threads": self.thread_spin.value(),
                "touch_rule": self.touch_rule_combo.currentText() if self.touch_rule_combo else None,
                "thresholds": thresholds,
                "upper_thresholds": upper_thresholds,
                "dilation_steps": steps,
            }
        elif self.mode == "seed":
            return {
                "threads": self.thread_spin.value(),
                "thresholds": thresholds,
                "upper_thresholds": upper_thresholds,
                "erosion_steps": steps,
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