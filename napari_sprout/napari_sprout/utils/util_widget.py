import os, sys
from qtpy.QtWidgets import (
    QGroupBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,QDoubleSpinBox,
    QSpinBox, QCheckBox, QFileDialog , QVBoxLayout , QComboBox , QTableWidget,QWidget,QSlider,
     QHeaderView, QSizePolicy, QGridLayout, QStyle, QStyleOptionFrame

)
from qtpy.QtCore import Signal, Qt ,QThreadPool,QObject,QRunnable,Slot
from qtpy.QtGui import QPainter, QPalette
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
from sprout_core.roi import parse_ids
from .fit_table import FitTable
from .status import StatusLine
from sprout_core.sprout_core import get_ccomps_with_size_order,erosion_binary_img_on_sub




class ElidedLineEdit(QLineEdit):
    """A line edit that shows a long text middle-elided while it has no focus.

    text() / setText() are the real value; the tooltip is the full text. While editing
    (focused) the whole text is shown as usual.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.textChanged.connect(self.setToolTip)

    def paintEvent(self, event):
        text = self.text()
        if self.hasFocus() or not text:
            return super().paintEvent(event)
        opt = QStyleOptionFrame()
        self.initStyleOption(opt)
        rect = self.style().subElementRect(QStyle.SE_LineEditContents, opt, self).adjusted(2, 0, -2, 0)
        if self.fontMetrics().horizontalAdvance(text) <= rect.width():
            return super().paintEvent(event)
        p = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_PanelLineEdit, opt, p, self)
        p.setPen(self.palette().color(QPalette.Text))
        p.drawText(rect, int(Qt.AlignVCenter | Qt.AlignLeft),
                   self.fontMetrics().elidedText(text, Qt.ElideMiddle, rect.width()))
        p.end()


def create_output_folder_row(default_folder=None, label="Output folder"):

    layout = QHBoxLayout()
    label = QLabel(label)
    folder_edit = ElidedLineEdit()
    folder_edit.setMinimumWidth(80)
    browse_btn = QPushButton("Browse")

    # set default folder if provided, otherwise use current working directory. It is only shown
    # here: the writers create it (os.makedirs(..., exist_ok=True)) when a run writes files.
    if default_folder is None:
        default_folder = os.path.join(os.getcwd(), "sprout_temp")
    folder_edit.setText(default_folder)

    def browse():
        folder = QFileDialog.getExistingDirectory(None, "Select Output Folder")
        if folder:
            folder_edit.setText(folder)

    browse_btn.clicked.connect(browse)

    layout.addWidget(label)
    layout.addWidget(folder_edit, 1)
    layout.addWidget(browse_btn)
    layout.label = label            # for hosts that align it with a form
    layout.browse_btn = browse_btn

    return layout, folder_edit


class PreviewWorkerSignals(QObject):
    finished = Signal(np.ndarray)
    error = Signal(str)

class PreviewWorker(QRunnable):
    def __init__(self, image, lower, upper, use_cc=False, cc_n=100, erosion_steps=0):
        super().__init__()
        self.image = image
        self.lower = lower
        self.upper = upper
        self.use_cc = use_cc
        self.cc_n = cc_n
        self.erosion_steps = erosion_steps
        self.signals = PreviewWorkerSignals()

    @Slot()
    def run(self):
        try:
            binary = (self.image >= self.lower) & (self.image <= self.upper)
            
            if self.erosion_steps > 0:
                for _ in range(self.erosion_steps):
                    # Apply erosion on the binary image
                    # Assuming erosion_binary_img_on_sub is a function that applies erosion
                    binary = erosion_binary_img_on_sub(binary)

            if self.use_cc:
                binary, _ = get_ccomps_with_size_order(binary, self.cc_n)

            # Always use 16-bit to avoid overflow (no class counting needed here).
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
        self.lower_slider.valueChanged.connect(lambda v: self.lower_spin.setValue(float(v)))
        self.lower_spin.editingFinished.connect(self._enforce_order)
        self.lower_slider.valueChanged.connect(self._enforce_order)

        # link upper
        self.upper_slider.valueChanged.connect(lambda v: self.upper_spin.setValue(float(v)))
        self.upper_spin.editingFinished.connect(self._enforce_order)
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
            self.use_connected_components_checkbox = QCheckBox("Connected Components")
            self.connected_components_spin = QSpinBox()
            self.connected_components_spin.setRange(1, 100000)
            self.connected_components_spin.setValue(100)
            self.connected_components_spin.setEnabled(False)

            self.use_connected_components_checkbox.toggled.connect(
                self.connected_components_spin.setEnabled
            )

            cc_layout = QHBoxLayout()
            
            self.use_connected_components_checkbox.setMaximumWidth(150)
            cc_layout.addWidget(self.use_connected_components_checkbox)
           
            label = QLabel("N:")
            label.setMaximumWidth(20)
            cc_layout.addWidget(label)
            self.connected_components_spin.setMaximumWidth(60)
            cc_layout.addWidget(self.connected_components_spin)
            
            self.erosion_spin = QSpinBox()
           
            self.erosion_spin.setRange(0, 100)
            self.erosion_spin.setValue(0)
            self.erosion_spin.setMaximumWidth(60)
            
            erosion_label = QLabel("Erosions")
            erosion_label.setMaximumWidth(80)

            cc_layout.addWidget(erosion_label)
            cc_layout.addWidget(self.erosion_spin)
            

            layout.addRow(cc_layout)

            # layout.addRow("Erosion (ball)", self.erosion_spin)

        
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

        if sender == self.lower_spin:
            self.lower_slider.blockSignals(True)
            self.lower_slider.setValue(int(lower))
            self.lower_slider.blockSignals(False)
        elif sender == self.upper_spin:
            self.upper_slider.blockSignals(True)
            self.upper_slider.setValue(int(upper))
            self.upper_slider.blockSignals(False)

        if lower > upper:
            
            
            # refresh the values without triggering signals
            self.lower_spin.blockSignals(True)
            self.lower_slider.blockSignals(True)
            self.upper_spin.blockSignals(True)
            self.upper_slider.blockSignals(True)

            # if lower is greater than upper, set both to the same value
            if sender in (self.lower_spin, self.lower_slider):
                new_val = lower
            # if upper is greater than lower, set both to the same value
            else:
                new_val = upper


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

        ero_step = self.erosion_spin.value() if hasattr(self, 'erosion_spin') else 0

        worker = PreviewWorker(image, lower, upper, use_cc, cc_n, ero_step)

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


def as_list(value):
    """A YAML file may hold one value where the tables take a list."""
    return value if value is None or isinstance(value, list) else [value]


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


def is_float_image(image):
    return image is not None and np.issubdtype(image.dtype, np.floating)


def threshold_limits(image):
    """Limits of a threshold on `image`: those of its dtype; of a float image, those of its values."""
    if not is_float_image(image):
        return get_max_min_value_by_dtype(image.dtype)
    sample = np.asarray(image).ravel()[::max(1, image.size // 1_000_000)]    # enough for a spin box
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return 0.0, 1.0
    return float(np.floor(sample.min() * 1000) / 1000), float(np.ceil(sample.max() * 1000) / 1000)


def threshold_spin(image, low, high, value):
    """A spin box of a threshold table: whole numbers, with decimals for a float image."""
    if is_float_image(image):
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setSingleStep(10 ** np.floor(np.log10(max(high - low, 0.001)) - 1))
        spin.setRange(low, high)
        spin.setValue(value)
        return spin
    spin = QSpinBox()
    spin.setRange(int(low), int(high))
    spin.setValue(int(value))
    return spin

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

def _param_spin(lo, hi, value, width=70):
    sp = QSpinBox()
    sp.setRange(lo, hi)
    sp.setValue(value)
    sp.setMinimumWidth(width)
    return sp


def _table_buttons(*buttons):
    row = QHBoxLayout()
    for b in buttons:
        row.addWidget(b)
    row.addStretch()
    return row


class _ParamGroup(QGroupBox):
    """Parameters group: a header row (hosts add Import / Export YAML), a status line
    (YAML / table messages), then the parameters."""

    def _init_header(self, layout):
        self.header_row = QHBoxLayout()
        self.header_row.addStretch()
        layout.addLayout(self.header_row)
        self.status_line = StatusLine(hide_empty=True)
        layout.addWidget(self.status_line)

    def add_header_widget(self, w):
        self.header_row.addWidget(w)

    def add_bottom_widget(self, w):
        self.layout().addWidget(w)


class MainSeedParamWidget(_ParamGroup):
    def __init__(self, title="Parameters", viewer=None, image_combo=None):
        super().__init__(title)

        self.viewer = viewer
        self.image_combo = image_combo  # ComboBox for selecting image layer

        layout = QVBoxLayout()
        layout.setSpacing(4)
        self._init_header(layout)

        # method / threads / erosion steps / segments
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        self.seed_method_combo = QComboBox()
        self.seed_method_combo.addItems(["Original", "Adaptive (Erosion)", "Adaptive (Thresholds)", SWEEP_METHOD])
        self.seed_method_combo.setToolTip(
            "Original: one seed layer per threshold row and erosion step.\n"
            "Adaptive (Erosion) / Adaptive (Thresholds): combine the seeds across the erosion\n"
            "steps / the threshold rows into one final seed.\n"
            "Sweep (Thresholds): one seed per threshold row, in order, combined from strict to\n"
            "loose. Stays fast with many seeds. One row: one seed per erosion step.")
        self.thread_spin = _param_spin(1, os.cpu_count(), min(4, os.cpu_count() - 1), 60)
        self.erosion_spin = _param_spin(0, 1000, 0)
        self.segments_spin = _param_spin(1, 100000, 10)
        self.segments_spin.setToolTip("Number of largest connected components to keep in each seed")
        self.param_labels = [QLabel("Method"), QLabel("Threads"), QLabel("Erosion steps"), QLabel("Segments")]
        # Method on its own row: next to Threads the panel needed a horizontal scrollbar at 500 px
        grid.addWidget(self.param_labels[0], 0, 0)
        grid.addWidget(self.seed_method_combo, 0, 1, 1, 3)
        grid.addWidget(self.param_labels[1], 1, 0)
        grid.addWidget(self.thread_spin, 1, 1)
        grid.addWidget(self.param_labels[2], 1, 2)
        grid.addWidget(self.erosion_spin, 1, 3)
        grid.addWidget(self.param_labels[3], 2, 0)
        grid.addWidget(self.segments_spin, 2, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(4, 1)
        layout.addLayout(grid)

        # thresholds, then footprints: each full width, as high as its rows
        self.threshold_table = FitTable(0, 2, max_rows=8)
        self.threshold_table.setHorizontalHeaderLabels(["Lower threshold", "Upper threshold"])
        self.threshold_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.threshold_table)
        self.add_threshold_btn = QPushButton("Add Threshold")
        self.remove_threshold_btn = QPushButton("Remove Selected")
        layout.addLayout(_table_buttons(self.add_threshold_btn, self.remove_threshold_btn))

        self.footprint_table = FitTable(0, 1, max_rows=8)
        self.footprint_table.setHorizontalHeaderLabels(["Footprint"])
        self.footprint_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.footprint_table)
        self.add_footprint_btn = QPushButton("Add Footprint")
        self.remove_footprint_btn = QPushButton("Remove Selected")
        self.remove_footprint_btn_all = QPushButton("Remove All")
        layout.addLayout(_table_buttons(self.add_footprint_btn, self.remove_footprint_btn,
                                        self.remove_footprint_btn_all))

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
                f"Add {self.erosion_spin.value()} footprints (one per erosion step), got {len(footprints)}"
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
            self.status_line.set("Select an image layer first", "error")
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
        self._footprints_set = True

    def _fit_footprints(self):
        """Footprint rows offer the footprints of the image's ndim (2D / 3D); a footprint
        the image cannot use becomes the first one. Returns how many were changed."""
        if self.image_combo is None or not self.image_combo.currentText():
            return 0
        ndim = self.viewer.layers[self.image_combo.currentText()].data.ndim
        names = support_footprints_2d if ndim == 2 else support_footprints
        changed = 0
        for row in range(self.footprint_table.rowCount()):
            combo = self.footprint_table.cellWidget(row, 0)
            current = combo.currentText()
            if [combo.itemText(i) for i in range(combo.count())] == list(names):
                continue
            combo.clear()
            combo.addItems(names)
            if current in names:
                combo.setCurrentText(current)
            else:
                changed += 1
        return changed
    
    def set_footprints(self, footprints):
        """Footprint table rows from a list (YAML import); None or [] = no rows."""
        self.footprint_table.setRowCount(0)
        self._footprints_set = True
        for fp in footprints or []:
            self._add_footprint_row()
            combo = self.footprint_table.cellWidget(self.footprint_table.rowCount() - 1, 0)
            if combo is None:
                return
            i = combo.findText(str(fp))
            if i >= 0:
                combo.setCurrentIndex(i)

    def _remove_footprint_row(self):
        """Remove selected footprint row."""
        current_row = self.footprint_table.currentRow()
        if current_row >= 0:
            self.footprint_table.removeRow(current_row)

    def _add_threshold_row_adaptive(self, dilate=5):
        """Add a new threshold row to the table, auto-suggesting values from previous row or image dtype."""
 

        # not image
        if self.image_combo is None or not self.image_combo.currentText():
            self.status_line.set("Select an image layer first", "error")
            return

        row = self.threshold_table.rowCount()
        self.threshold_table.insertRow(row)

        current_image = self.viewer.layers[self.image_combo.currentText()].data

        # set default values based on previous row or image dtype
        if row == 0:
            # upper_value = self._get_img_dtype_max(current_image.dtype)
            # lower_value = 0
            lower_value, upper_value = threshold_limits(current_image)
        else:
            lower_widget = self.threshold_table.cellWidget(row - 1, 0)
            lower_value = lower_widget.value()
            upper_widget = self.threshold_table.cellWidget(row - 1, 1)
            upper_value = upper_widget.value()

        # set the range for the spinboxes
        lower_min , upper_max = threshold_limits(current_image)
        lower_max = upper_value
        upper_min = lower_value
        # upper_max = self._get_img_dtype_max(current_image.dtype)

        # Lower threshold
        lower_spin = threshold_spin(current_image, lower_min, lower_max, lower_value)
        self.threshold_table.setCellWidget(row, 0, lower_spin)

        # Upper threshold
        upper_spin = threshold_spin(current_image, upper_min, upper_max, upper_value)
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

    def new_image(self):
        """Another image: the threshold table is reset (thresholds depend on its values),
        the footprint table kept (its first image: one default row). Returns the number of
        footprints changed to fit the image's ndim."""
        self.threshold_table.setRowCount(0)
        self._add_threshold_row_adaptive()
        if not getattr(self, "_footprints_set", False):
            self._add_footprint_row()
            return 0
        return self._fit_footprints()

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
            elif is_float_image(current_image):
                _, upper_threshold = threshold_limits(current_image)
            else:
                # Use 255 or threshold+100 if threshold > 255
                upper_threshold = 255 if lower_threshold <= 255 else lower_threshold + 100
            
            # Set range based on image dtype if available
            if current_image is not None:
                lower_min, upper_max = threshold_limits(current_image)
            else:
                lower_min, upper_max = 0, 65535  # Default range
            
            # Lower threshold spinbox
            lower_spin = threshold_spin(current_image, lower_min, upper_threshold, lower_threshold)
            self.threshold_table.setCellWidget(row, 0, lower_spin)
            
            # Upper threshold spinbox
            upper_spin = threshold_spin(current_image, lower_threshold, upper_max, upper_threshold)
            self.threshold_table.setCellWidget(row, 1, upper_spin)


class MainGrowParamWidget(_ParamGroup):
    def __init__(self, title="Parameters", viewer=None, image_combo=None,
                 mode = "grow",  # "grow" or "seed"
                 colnames=None):
        super().__init__(title)

        self.viewer = viewer
        self.image_combo = image_combo  # ComboBox for selecting image layer
        self.mode = mode  # "grow" or "seed"

        layout = QVBoxLayout()
        layout.setSpacing(4)
        self._init_header(layout)

        # threads / touch rule
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        self.thread_spin = _param_spin(1, os.cpu_count(), min(4, os.cpu_count() - 1), 60)
        self.touch_rule_combo = QComboBox()
        self.touch_rule_combo.addItems(["stop", "overwrite"])
        self.touch_rule_combo.setToolTip("When a growing region reaches another region:\n"
                                         "stop: it never grows into the other region's voxels\n"
                                         "overwrite: it may take them over")
        self.param_labels = [QLabel("Threads"), QLabel("Touch rule")]
        grid.addWidget(self.param_labels[0], 0, 0)
        grid.addWidget(self.thread_spin, 0, 1)
        grid.addWidget(self.param_labels[1], 0, 2)
        grid.addWidget(self.touch_rule_combo, 0, 3)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(4, 1)
        layout.addLayout(grid)

        # thresholds table: full width, as high as its rows
        self.threshold_table = FitTable(0, 3, max_rows=8)
        self.threshold_table.setHorizontalHeaderLabels(
            ["Lower threshold", "Upper threshold", "Dilation steps"])
        self.threshold_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.threshold_table)

        self.add_threshold_btn = QPushButton("Add Threshold")
        self.remove_threshold_btn = QPushButton("Remove Selected")
        self.remove_all_threshold_btn = QPushButton("Remove All Thresholds")
        layout.addLayout(_table_buttons(self.add_threshold_btn, self.remove_threshold_btn,
                                        self.remove_all_threshold_btn))

        self.setLayout(layout)

        self.add_threshold_btn.clicked.connect(self._add_threshold_row_adaptive)
        self.remove_threshold_btn.clicked.connect(self._remove_threshold_row)
        self.remove_all_threshold_btn.clicked.connect(lambda: self.threshold_table.setRowCount(0))

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


        # not image
        if self.image_combo is None or not self.image_combo.currentText():
            self.status_line.set("Select an image layer first", "error")
            return
        row = self.threshold_table.rowCount()
        self.threshold_table.insertRow(row)
        
        current_image = self.viewer.layers[self.image_combo.currentText()].data

        # set default values based on previous row or image dtype
        if row == 0:
            # upper_value = self._get_img_dtype_max(current_image.dtype)
            # lower_value = 0
            lower_value, upper_value = threshold_limits(current_image)
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
        lower_min, upper_max = threshold_limits(current_image)
        # upper_max = self._get_img_dtype_max(current_image.dtype)

        # Lower threshold
        lower_spin = threshold_spin(current_image, lower_min, lower_max, lower_value)
        self.threshold_table.setCellWidget(row, 0, lower_spin)

        # Upper threshold
        upper_spin = threshold_spin(current_image, upper_min, upper_max, upper_value)
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
    
    def set_params(self, params):
        """Set parameters for the widget from a dictionary."""
        if "num_threads" in params:
            self.thread_spin.setValue(params["num_threads"])
        
        if "touch_rule" in params:
            index = self.touch_rule_combo.findText(params["touch_rule"], Qt.MatchFixedString)
            if index >= 0:
                self.touch_rule_combo.setCurrentIndex(index)
    
    def populate_thresholds_from_list(self, thresholds_list, upper_thresholds_list=None, dilation_steps_list=None):
        """Populate the thresholds table with lists from YAML import.
        
        Parameters
        ----------
        thresholds_list : list
            List of lower threshold values
        upper_thresholds_list : list, optional
            List of upper threshold values. If None, will use max value based on dtype
        dilation_steps_list : list, optional
            List of dilation steps. If None, will use default value of 5
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
                # Use max value based on dtype if available
                if current_image is not None:
                    _, upper_threshold = threshold_limits(current_image)
                else:
                    upper_threshold = 255 if lower_threshold <= 255 else lower_threshold + 100
            
            # Determine dilation steps
            if dilation_steps_list and i < len(dilation_steps_list):
                dilation_steps = dilation_steps_list[i]
            else:
                dilation_steps = 5  # Default value
            
            # Set range based on image dtype if available
            if current_image is not None:
                lower_min, upper_max = threshold_limits(current_image)
            else:
                lower_min, upper_max = 0, 65535  # Default range
            
            # Lower threshold spinbox
            lower_spin = threshold_spin(current_image, lower_min, upper_threshold, lower_threshold)
            self.threshold_table.setCellWidget(row, 0, lower_spin)
            
            # Upper threshold spinbox
            upper_spin = threshold_spin(current_image, lower_threshold, upper_max, upper_threshold)
            self.threshold_table.setCellWidget(row, 1, upper_spin)
            
            # Dilation steps spinbox
            dilate_spin = QSpinBox()
            dilate_spin.setRange(1, 1000)
            dilate_spin.setValue(int(dilation_steps))
            self.threshold_table.setCellWidget(row, 2, dilate_spin)


def set_compact(widget, minw=None, maxw=None):
    widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
    if minw is not None:
        widget.setMinimumWidth(minw)
    if maxw is not None:
        widget.setMaximumWidth(maxw)


SWEEP_METHOD = "Sweep (Thresholds)"


class SeedOptionalParamGroupBox(QGroupBox):
    def __init__(self, default_output=None):
        super().__init__("Advanced seed parameters")
        layout = QFormLayout()
        # a row too wide for the panel puts its field under the label (no horizontal scrollbar)
        layout.setRowWrapPolicy(QFormLayout.WrapLongRows)

        # sort (bool)
        self.sort_checkbox = QCheckBox("Enable sorting")
        self.sort_checkbox.setChecked(True)
        layout.addRow("Sort segments", self.sort_checkbox)

        # min_size (int)
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(1, 100000)
        self.min_size_spin.setValue(1)
        layout.addRow("Min segment size", self.min_size_spin)


        self.no_split_spin = QSpinBox()
        self.no_split_spin.setRange(0, 100)
        self.no_split_spin.setValue(3)
        self.no_split_spin.setToolTip("Max no-split iterations")

        self.min_split_ratio_spin = QDoubleSpinBox()
        self.min_split_ratio_spin.setDecimals(3)
        self.min_split_ratio_spin.setRange(0.0, 1.0)
        self.min_split_ratio_spin.setSingleStep(0.01)
        self.min_split_ratio_spin.setValue(0.0)
        self.min_split_ratio_spin.setToolTip("Min split ratio")

        self.min_split_total_ratio_spin = QDoubleSpinBox()
        self.min_split_total_ratio_spin.setDecimals(3)
        self.min_split_total_ratio_spin.setRange(0.0, 1.0)
        self.min_split_total_ratio_spin.setSingleStep(0.01)
        self.min_split_total_ratio_spin.setValue(0.0)
        self.min_split_total_ratio_spin.setToolTip("Min total split ratio")




        # add to the split parameters row
        layout.addRow("No split iterations", self.no_split_spin)
        
        # one form row each: side by side in one row the labels were cut ("Split Rati")
        self.split_ratio_label = QLabel("Min split ratio")
        self.split_ratio_label.setToolTip("Minimum ratios of a sub-segment to be considered as a split")
        self.min_split_ratio_spin.setToolTip(self.split_ratio_label.toolTip())
        layout.addRow(self.split_ratio_label, self.min_split_ratio_spin)

        self.total_ratio_label = QLabel("Min total split ratio")
        self.total_ratio_label.setToolTip("Minimum proportion of all sub-segments to current to be considered as a split")
        self.min_split_total_ratio_spin.setToolTip(self.total_ratio_label.toolTip())
        layout.addRow(self.total_ratio_label, self.min_split_total_ratio_spin)

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
        self.size_lower_spin = QSpinBox()
        self.size_lower_spin.setRange(-1, 1_000_000_000)
        self.size_lower_spin.setValue(-1)
        self.size_lower_spin.setToolTip("Lower limit (-1 means no limit)")

        self.size_upper_spin = QSpinBox()
        self.size_upper_spin.setRange(-1, 1_000_000_000)
        self.size_upper_spin.setValue(-1)
        self.size_upper_spin.setToolTip("Upper limit (-1 means no limit)")


        size_layout = QHBoxLayout()
        
        self.size_lower_label = QLabel("Lower")
        self.size_lower_label.setToolTip("Lower limit (-1 means no limit)")
        size_layout.addWidget(self.size_lower_label)
        size_layout.addWidget(self.size_lower_spin)
        self.size_upper_label = QLabel("Upper")
        self.size_upper_label.setToolTip("Upper limit (-1 means no limit)")
        size_layout.addWidget(self.size_upper_label)
        size_layout.addWidget(self.size_upper_spin)
        layout.addRow("Split size", size_layout)

        # split_convex_hull_limit (tuple of float or None)
        self.hull_lower_spin = QSpinBox()
        self.hull_lower_spin.setRange(-1, 1_000_000_000)
        self.hull_lower_spin.setValue(-1)
        self.hull_lower_spin.setToolTip("Lower limit (-1 means no limit)")

        self.hull_upper_spin = QSpinBox()
        self.hull_upper_spin.setRange(-1, 1_000_000_000)
        self.hull_upper_spin.setValue(-1)
        self.hull_upper_spin.setToolTip("Upper limit (-1 means no limit)")
        hull_layout = QHBoxLayout()
        self.hull_lower_label = QLabel("Lower")
        self.hull_lower_label.setToolTip("Lower limit (-1 means no limit)")
        hull_layout.addWidget(self.hull_lower_label)
        hull_layout.addWidget(self.hull_lower_spin)
        self.hull_upper_label = QLabel("Upper")
        self.hull_upper_label.setToolTip("Upper limit (-1 means no limit)")
        hull_layout.addWidget(self.hull_upper_label)
        hull_layout.addWidget(self.hull_upper_spin)
        layout.addRow("Convex hull size", hull_layout)

        for w in [
            self.min_size_spin,
            self.no_split_spin,
            self.min_split_ratio_spin,
            self.min_split_total_ratio_spin,
            self.size_lower_spin,
            self.size_upper_spin,
            self.hull_lower_spin,
            self.hull_upper_spin,
            self.size_lower_label,
            self.size_upper_label,
            self.hull_lower_label,
            self.hull_upper_label,
        ]:
            set_compact(w, minw=40, maxw=80)

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
            # valid lower and upper are int or None
            if not (isinstance(lower, int) or lower is None):
                raise ValueError("split_size_limit lower must be int or None")
            if not (isinstance(upper, int) or upper is None):
                raise ValueError("split_size_limit upper must be int or None")
            self.size_lower_spin.setValue(lower if lower is not None else -1)
            self.size_upper_spin.setValue(upper if upper is not None else -1)

        if 'split_convex_hull_limit' in params and params['split_convex_hull_limit'] is not None:
            lower, upper = params['split_convex_hull_limit']
            
            # valid lower and upper are int or None
            if not (isinstance(lower, int) or lower is None):
                raise ValueError("split_convex_hull_limit lower must be int or None")
            if not (isinstance(upper, int) or upper is None):
                raise ValueError("split_convex_hull_limit upper must be int or None")
            self.hull_lower_spin.setValue(lower if lower is not None else -1)
            self.hull_upper_spin.setValue(upper if upper is not None else -1)



    def set_method(self, method):
        """A sweep only uses the minimum size: the rest is greyed out."""
        for widget in (self.sort_checkbox, self.no_split_spin, self.min_split_ratio_spin,
                       self.min_split_total_ratio_spin, self.size_lower_spin, self.size_upper_spin,
                       self.hull_lower_spin, self.hull_upper_spin):
            widget.setEnabled(method != SWEEP_METHOD)

    def get_params(self):
        def parse_optional_spin(spinbox):
            val = spinbox.value()
            return None if val == -1 else val

        return {
            "sort": self.sort_checkbox.isChecked(),
            "no_split_max_iter": self.no_split_spin.value(),
            "min_size": self.min_size_spin.value(),
            "min_split_ratio": self.min_split_ratio_spin.value(),
            "min_split_total_ratio": self.min_split_total_ratio_spin.value(),
            "split_size_limit": (
                parse_optional_spin(self.size_lower_spin),
                parse_optional_spin(self.size_upper_spin)
            ),
            "split_convex_hull_limit": (
                parse_optional_spin(self.hull_lower_spin),
                parse_optional_spin(self.hull_upper_spin)
            )
        }

class GrowOptionalParamGroupBox(QGroupBox):
    def __init__(self, default_output=None):
        super().__init__("Advanced grow parameters")


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

        # Grow specific IDs ("3,5-7", as in the ROI tools); a typo is a red line, the run is refused
        self.id_list_line = QLineEdit()
        self.id_list_line.setPlaceholderText("e.g. 3,5-7 (empty: all)")
        self.id_list_line.setToolTip("Seed ids to grow, e.g. 3,5-7; empty: all seeds")
        layout.addRow(QLabel("IDs to grow"), self.id_list_line)
        self.ids_error = StatusLine(hide_empty=True)
        layout.addRow(self.ids_error)
        self.id_list_line.textChanged.connect(lambda *_: self.ids_error.clear())

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
        layout.addRow(QLabel("Max no-growth iterations"), self.no_growth_spin)

        # Min growth size
        self.min_growth_spin = QSpinBox()
        self.min_growth_spin.setRange(0, 1000000)
        self.min_growth_spin.setValue(50)
        self.min_growth_spin.setEnabled(True)
        layout.addRow(QLabel("Min growth size"), self.min_growth_spin)



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

    def set_params(self, params):
        """Set advanced grow parameters for the widget from a dictionary."""
        if 'save_every_n_iters' in params:
            # None (not saved) unticks the box: a YAML round trip restores every value
            self.save_iter_checkbox.setChecked(params['save_every_n_iters'] is not None)
            if params['save_every_n_iters'] is not None:
                self.save_every_n_spin.setValue(params['save_every_n_iters'])
        
        if 'grow_to_end' in params:
            self.grow_to_end_checkbox.setChecked(params['grow_to_end'])
        
        if 'is_sort' in params:
            self.sort_checkbox.setChecked(params['is_sort'])
        
        if 'to_grow_ids' in params:
            ids = params['to_grow_ids']
            self.id_list_line.setText("" if ids is None else ",".join(map(str, ids)))
        
        if 'no_growth_max_iter' in params:
            if params['no_growth_max_iter'] is None:
                self.early_stop_checkbox.setChecked(False)
            else:
                self.early_stop_checkbox.setChecked(True)
                self.no_growth_spin.setValue(params['no_growth_max_iter'])
        
        if 'min_growth_size' in params:
            self.min_growth_spin.setValue(params['min_growth_size'])

    def to_grow_ids(self):
        """The ids to grow (sorted), None for all. ValueError on a typo, also shown as a red line."""
        text = self.id_list_line.text().strip()
        if not text:
            return None
        try:
            ids = sorted(parse_ids(text))
        except ValueError:
            ids = None
        if not ids:
            msg = f"IDs to grow: enter ids like 3,5-7, or nothing for all (got '{text}')"
            self.ids_error.set(msg, "error")
            raise ValueError(msg)
        return ids

    def get_params(self):
        to_grow_ids = self.to_grow_ids()

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
        
