from napari.layers import Labels, Image
from skimage.morphology import disk, ball
import numpy as np

import os
import pandas as pd

from skimage.filters import threshold_otsu
from tifffile import imread


from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel, QComboBox,
    QSpinBox, QPushButton, QMessageBox, QRadioButton, QButtonGroup, QCheckBox,
    QFileDialog, QLineEdit, QTableWidget, QTableWidgetItem, QDialog, QDialogButtonBox
)
try:
    from napari.qt import thread_worker
except ImportError:
    from napari.utils import thread_worker

class SizeReferenceGroupBox(QGroupBox):
    def __init__(self, viewer):
        super().__init__('Size Reference Preview')
        self.setToolTip("Visualize structuring elements like disk or ball, and preview their size relative to your image.")

        self.viewer = viewer
        self.clicked_position = None  # Store click position if using click mode

        layout = QFormLayout()

        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["disk", "ball"])
        self.shape_combo.currentTextChanged.connect(self.sync_size_from_radius)
        layout.addRow(QLabel("Shape Type:"), self.shape_combo)

        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(1, 100)
        self.radius_spin.setValue(5)
        self.radius_spin.valueChanged.connect(self.sync_size_from_radius)
        layout.addRow(QLabel("Radius (pixels):"), self.radius_spin)

        self.size_display_label = QLabel("Computed size: N/A")
        layout.addRow(self.size_display_label)

        # Show clicked position
        self.click_pos_label = QLabel("Clicked position: N/A")
        layout.addRow(self.click_pos_label)

        # Placement options
        self.place_group = QButtonGroup()
        self.center_radio = QRadioButton("Place at image center")
        self.click_radio = QRadioButton("Place at clicked position")
        self.center_radio.setChecked(True)
        self.place_group.addButton(self.center_radio)
        self.place_group.addButton(self.click_radio)
        layout.addRow(QLabel("Placement:"), self.center_radio)
        layout.addRow(QLabel(""), self.click_radio)

        preview_btn = QPushButton("Add Preview")
        preview_btn.clicked.connect(self.add_preview)
        layout.addRow(preview_btn)

        self.setLayout(layout)

        # Connect mouse click if needed
        # Now add in showEvent
        # self.viewer.mouse_drag_callbacks.append(self.store_click_position)

        self.sync_size_from_radius()

    
    def showEvent(self, event):
        # print("showEvent called")
        try:
            if self.store_click_position not in self.viewer.mouse_drag_callbacks:
                self.viewer.mouse_drag_callbacks.append(self.store_click_position)
        except Exception:
            print("Warning: Could not connect active layer change event.")
            pass
        
        
    def hideEvent(self, event):
        # print("hideEvent called")
        try:
            if self.store_click_position in self.viewer.mouse_drag_callbacks:
                self.viewer.mouse_drag_callbacks.remove(self.store_click_position)
        except Exception:
            print("Warning: Could not disconnect active layer change event.")
            pass
    
    # def closeEvent(self, event):
    #     if self.store_click_position in self.viewer.mouse_drag_callbacks:
    #         self.viewer.mouse_drag_callbacks.remove(self.store_click_position)
    #     print("Callback removed when widget closed.")
    #     super().closeEvent(event)

    def store_click_position(self, viewer, event):
        if not hasattr(self, "click_radio") or self.click_radio is None:
            return
        if not self.click_radio.isChecked():
            return
        if event.type != 'mouse_press':
            return
        self.clicked_position = tuple(int(round(x)) for x in viewer.cursor.position)
        self.click_pos_label.setText(f"Clicked position: {self.clicked_position}")
        print(f"Stored click position: {self.clicked_position}")

    def sync_size_from_radius(self):
        shape = self.shape_combo.currentText()
        r = self.radius_spin.value()
        if shape == 'disk':
            area = np.pi * r * r
            self.size_display_label.setText(f"Computed area: {int(round(area))} px²")
        elif shape == 'ball':
            volume = (4/3) * np.pi * r ** 3
            self.size_display_label.setText(f"Computed volume: {int(round(volume))} px³")
        else:
            self.size_display_label.setText("Computed size: N/A")

    def add_preview(self):
        shape = self.shape_combo.currentText()
        radius = self.radius_spin.value()
        data_layer = self.get_reference_layer()

        if data_layer is None:
            QMessageBox.warning(self, "Error", "No reference image or label layer found.")
            return

        shape_func = disk if shape == 'disk' else ball
        preview_shape = shape_func(radius).astype(np.uint8)

        canvas_shape = data_layer.data.shape
        canvas = np.zeros(canvas_shape, dtype=np.uint8)

        insert_at = self.get_insert_position(canvas_shape, preview_shape.shape)
        if insert_at is None:
            QMessageBox.warning(self, "Error", "Invalid placement position.")
            return

        slices = tuple(slice(s, s + sz) for s, sz in zip(insert_at, preview_shape.shape))
        canvas[slices] = preview_shape

        self.viewer.add_labels(canvas, name=f"{shape}_r{radius}_preview")

    def get_reference_layer(self):
        for layer in self.viewer.layers.selection:
            if isinstance(layer, (Labels, Image)):
                return layer
        return None

    def get_insert_position(self, canvas_shape, patch_shape):
        if self.center_radio.isChecked():
            return tuple((c - p) // 2 for c, p in zip(canvas_shape, patch_shape))
        elif self.click_radio.isChecked():
            if self.clicked_position is None:
                return None
            return tuple(max(0, min(c - p, cp - p // 2)) for cp, c, p in zip(self.clicked_position, canvas_shape, patch_shape))
        else:
            return None


def remove_prefix(filename, prefix):
    """
    Safely remove a prefix from the start of a filename.

    Args:
        filename (str): The filename to process.
        prefix (str): The prefix to remove.

    Returns:
        str: The filename with the prefix removed if it exists, otherwise unchanged.
    """
    if filename.startswith(prefix):
        return filename[len(prefix):]
    return filename

def remove_suffix(filename, suffix):
    """
    Safely remove a suffix from the end of a filename.

    Args:
        filename (str): The filename to process.
        suffix (str): The suffix to remove.

    Returns:
        str: The filename with the suffix removed if it exists, otherwise unchanged.
    """
    if filename.endswith(suffix):
        return filename[: -len(suffix)]
    return filename


def sort_dict_and_extract_values(input_dict):
    """
    Sort a dictionary by keys alphabetically and extract values in that order.

    Args:
        input_dict (dict): The dictionary to process.

    Returns:
        list: A list of values sorted by the keys.
    """
    # Sort the dictionary by keys alphabetically
    sorted_keys = sorted(input_dict.keys())
    
    # Extract values in the sorted order of keys
    sorted_values = [input_dict[key] for key in sorted_keys]
    
    return sorted_values


def align_files_to_df(img_folder, seg_folder=None, boundary_folder=None,
                        seg_prefix="", seg_suffix="", 
                        boundary_prefix="", boundary_suffix="", 
                        match_type="exact"):
    """
    Align files from img, seg, and boundary folders and return a DataFrame.

    Args:
        img_folder (str): Path to the folder containing image files.
        seg_folder (str): Path to the folder containing segmentation files.
        boundary_folder (str): Path to the folder containing boundary files.
        seg_prefix (str): Prefix for segmentation filenames.
        seg_suffix (str): Suffix for segmentation filenames.
        boundary_prefix (str): Prefix for boundary filenames.
        boundary_suffix (str): Suffix for boundary filenames.
        match_type (str): Type of matching: "exact" for identical filenames, or "base" for using img as the base.

    Returns:
        pd.DataFrame: A DataFrame with aligned file paths.
    """
    # Helper function to list and sort files (excluding extensions)
    def list_files(folder):
        return sorted(
            [f for f in os.listdir(folder) if f.lower().endswith(('.tif', '.tiff'))]
        )
    

    def list_files(folder, prefix="", suffix="", must_include=True):
        """
        List and process files in a folder based on prefix and suffix.

        Args:
            folder (str): Path to the folder.
            prefix (str): Prefix to match filenames.
            suffix (str): Suffix to match filenames.

        Returns:
            dict: Dictionary with base filenames as keys and full paths as values.
        """
        # return {
        #     remove_suffix(remove_prefix(os.path.splitext(f)[0], prefix), suffix): sorted(
        #     [f for f in os.listdir(folder) if f.lower().endswith(('.tif', '.tiff'))]
        # )
        # }

        if must_include:
            return {
                remove_prefix(os.path.splitext(f)[0], prefix).replace(suffix, ""): os.path.join(folder, f)
                for f in os.listdir(folder) if f.lower().endswith(('.tif', '.tiff')) and os.path.splitext(f)[0].startswith(prefix)
                and os.path.splitext(f)[0].endswith(suffix)
            }
        else:
            return {
                remove_prefix(os.path.splitext(f)[0], prefix).replace(suffix, ""): os.path.join(folder, f)
                for f in os.listdir(folder) if f.lower().endswith(('.tif', '.tiff'))
            }

    
    # def get_abs_path(folder, file_list):
    #     # return [os.path.abspath(os.path.join(folder, f)) for f in file_list]
    #     return [os.path.abspath(f) for f in file_list]
        
    
    def get_abs_path(file_list):
        return [os.path.abspath(f) for f in file_list]
        

    # Initialize dictionaries for file lists
    file_lists = {}


    if img_folder:
        file_lists['img'] = list_files(img_folder)
    if seg_folder:
        file_lists['seg'] = list_files(seg_folder, prefix = seg_prefix,
                                       suffix=seg_suffix)
    if boundary_folder:
        file_lists['boundary'] = list_files(boundary_folder, prefix = boundary_prefix,
                                       suffix=boundary_suffix)


    data = {}
    
    
    if match_type == "exact":
        # Find common filenames exactly across all folders
        common_files = set(file_lists['img'].keys())
        if seg_folder:
            common_files &= set(file_lists['seg'].keys())
        if boundary_folder:
            common_files &= set(file_lists['boundary'].keys())
    elif match_type == "base":
        # Use img filenames as the base and match with seg and boundary files
        common_files = set(file_lists['img'].keys())

        if seg_folder:
            common_files &= {f.replace(seg_prefix, "").replace(seg_suffix, "") for f in file_lists['seg'].keys()}
        if boundary_folder:
            common_files &= {f.replace(boundary_prefix, "").replace(boundary_suffix, "") for 
                             f in file_lists['boundary'].keys()}    
    
    elif match_type == "sorted":
        if img_folder:
            data["img_path"] = get_abs_path(sort_dict_and_extract_values(file_lists['img']))
            # get_abs_path(img_folder, file_lists['img'])

        if seg_folder:
            
            data["seg_path"] = get_abs_path(sort_dict_and_extract_values(file_lists['seg']))
            # get_abs_path(seg_folder, file_lists['seg'])

        if boundary_folder:
            data["boundary_path"] = get_abs_path(sort_dict_and_extract_values(file_lists['boundary']))
            # get_abs_path(boundary_folder, file_lists['boundary'])
        
        df = pd.DataFrame(data)
        return df
    else:
        raise ValueError("Invalid match_type. Use 'exact' , 'sorted' 'base'.")
    
    sorted_keys = sorted(list(common_files))
    
    if img_folder:
        data["img_path"] = get_abs_path([file_lists['img'][f] for f in sorted_keys])
    if seg_folder:
        data["seg_path"] = get_abs_path([file_lists['seg'][f] for f in sorted_keys])
    if boundary_folder:
        data["boundary_path"] = get_abs_path([file_lists['boundary'][f] for f in sorted_keys])

    df = pd.DataFrame(data)

    return df

label_to_key = {
    "Same Name": "exact",
    "Image Name + Prefix/Suffix": "base",
    "File Order": "sorted"
}

key_to_label = {v: k for k, v in label_to_key.items()}
class CSVAlignerGroupBox(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Create CSV from Folder Alignment", parent)
        self.setLayout(QVBoxLayout())

        # === Input folders ===
        self.img_folder_line = self._add_path_row("Image Folder:")
        self.seg_folder_line = self._add_path_row("Segmentation Folder:")
        self.boundary_folder_line = self._add_path_row("Boundary Folder:")

        # === Prefix/suffix ===
        # self.seg_prefix_line = self._add_kv_row("Seg Prefix:")
        # self.seg_suffix_line = self._add_kv_row("Seg Suffix:")
        # self.boundary_prefix_line = self._add_kv_row("Boundary Prefix:")
        # self.boundary_suffix_line = self._add_kv_row("Boundary Suffix:")

        # === Match type selector ===
        # match_layout = QHBoxLayout()
        # match_layout.addWidget(QLabel("Match Type:"))
        # self.match_type_box = QComboBox()
        # self.match_type_box.addItems(["sorted", "exact", "base"])
        # match_layout.addWidget(self.match_type_box)
        # self.layout().addLayout(match_layout)

        row_layout = QHBoxLayout()
        self.match_type_label = QLabel("Match Type:")
        self.match_type_combo = QComboBox()
        self.match_type_combo.addItems(["Same Name", "Image Name + Prefix/Suffix", "File Order"])
        self.match_type_combo.currentTextChanged.connect(self.toggle_prefix_suffix)

        self.explain_btn = QPushButton("?")
        self.explain_btn.setFixedWidth(20)
        self.explain_btn.clicked.connect(self.show_explanation)

        row_layout.addWidget(self.match_type_label)
        row_layout.addWidget(self.match_type_combo)
        row_layout.addWidget(self.explain_btn)
        self.layout().addLayout(row_layout)

        # === Prefix/Suffix panel ===
        self.prefix_suffix_box = QGroupBox("Prefix/Suffix Settings")
        self.prefix_suffix_box.setLayout(QVBoxLayout())
        self.prefix_suffix_box.setVisible(False)
        self.layout().addWidget(self.prefix_suffix_box)

        # === Prefix/Suffix inputs ===
        self.seg_prefix_line = self._add_kv_row("Seg Prefix:", self.prefix_suffix_box)
        self.seg_suffix_line = self._add_kv_row("Seg Suffix:", self.prefix_suffix_box)
        self.boundary_prefix_line = self._add_kv_row("Boundary Prefix:", self.prefix_suffix_box)
        self.boundary_suffix_line = self._add_kv_row("Boundary Suffix:", self.prefix_suffix_box)

        # === Auto-generate threshold parameters ===
        self.auto_threshold_checkbox = QCheckBox("Auto-generate Thresholds")
        self.auto_threshold_checkbox.setChecked(False)
        self.auto_threshold_checkbox.stateChanged.connect(self.toggle_threshold_widget)
        self.layout().addWidget(self.auto_threshold_checkbox)

        # === Threshold Param Box ===
        self.threshold_param_box = ThresholdParamWidget()
        self.threshold_param_box.setVisible(False)
        self.layout().addWidget(self.threshold_param_box)
        
        threshold_group = QGroupBox("Adaptive Threshold Settings")
        threshold_group.setLayout(QVBoxLayout())
        threshold_group.layout().addWidget(self.threshold_param_box)
        self.layout().addWidget(threshold_group)
        

        # === Save CSV path ===
        self.save_csv_line = self._add_path_row("Save CSV As:", file_dialog=True)

        # === Run + Preview Buttons ===
        # run_layout = QHBoxLayout()
        self.run_btn = QPushButton("Generate CSV")
        self.run_btn.clicked.connect(self.run_alignment)
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.clicked.connect(self.preview_alignment)
        # run_layout.addWidget(run_btn)
        # run_layout.addWidget(preview_btn)
        # self.layout().addLayout(run_layout)
        self.layout().addWidget(self.preview_btn)
        self.layout().addWidget(self.run_btn)
        

    def _add_path_row(self, label_text, file_dialog=False):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        line_edit = QLineEdit()
        btn = QPushButton("Browse")
        if file_dialog:
            btn.clicked.connect(lambda: self._browse_file(line_edit))
        else:
            btn.clicked.connect(lambda: self._browse_folder(line_edit))
        layout.addWidget(label)
        layout.addWidget(line_edit)
        layout.addWidget(btn)
        self.layout().addLayout(layout)
        return line_edit

    def _add_kv_row(self, label_text, parent=None):
        if parent is None:
            parent = self.prefix_suffix_box

        layout = QHBoxLayout()
        label = QLabel(label_text)
        line_edit = QLineEdit()
        layout.addWidget(label)
        layout.addWidget(line_edit)
        parent.layout().addLayout(layout)
        return line_edit
    
    # def _add_kv_row(self, label_text):
    #     layout = QHBoxLayout()
    #     label = QLabel(label_text)
    #     line_edit = QLineEdit()
    #     layout.addWidget(label)
    #     layout.addWidget(line_edit)
    #     self.layout().addLayout(layout)
    #     return line_edit

    def _browse_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            line_edit.setText(folder)

    def _browse_file(self, line_edit):
        file, _ = QFileDialog.getSaveFileName(self, "Save CSV File", filter="CSV Files (*.csv)")
        if file:
            line_edit.setText(file)

    def collect_params(self):
        return {
            "img_folder": self.img_folder_line.text().strip(),
            "seg_folder": self.seg_folder_line.text().strip() or None,
            "boundary_folder": self.boundary_folder_line.text().strip() or None,
            "seg_prefix": self.seg_prefix_line.text().strip(),
            "seg_suffix": self.seg_suffix_line.text().strip(),
            "boundary_prefix": self.boundary_prefix_line.text().strip(),
            "boundary_suffix": self.boundary_suffix_line.text().strip(),
            "match_type": label_to_key[self.match_type_combo.currentText()]
        }
    def collect_threshold_params(self):
        return self.threshold_param_box.get_params()
    
    def run_alignment_no_background_thread(self):
        ### old version without background thread
        try:
            # Collect parameters from input fields
            params = self.collect_params()
            df = align_files_to_df(**params)

            if self.auto_threshold_checkbox.isChecked():
                # Collect threshold parameters
                threshold_params = self.collect_threshold_params()
                print("Threshold Parameters:", threshold_params)
                add_thresholds_to_df(df, threshold_params)

            save_path = self.save_csv_line.text().strip()
            if not save_path.endswith(".csv"):
                save_path += ".csv"
            df.to_csv(save_path, index=False)
            QMessageBox.information(self, "Success", f"CSV saved to:\n{save_path}")
        except Exception as e:
            print(f"[CSV Align Error] {e}")
            QMessageBox.critical(self, "Error", f"CSV creation failed:\n{str(e)}")

    def preview_alignment_no_background_thread(self):
        ### old version without background thread
        try:
            # Collect parameters from input fields
            params = self.collect_params()
            df = align_files_to_df(**params)

            if self.auto_threshold_checkbox.isChecked():
                # Collect threshold parameters
                threshold_params = self.collect_threshold_params()
                print("Threshold Parameters:", threshold_params)
                add_thresholds_to_df(df, threshold_params)
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Preview Aligned DataFrame")
            dialog_layout = QVBoxLayout(dialog)
            table = QTableWidget()
            table.setColumnCount(len(df.columns))
            table.setRowCount(len(df))
            table.setHorizontalHeaderLabels(df.columns.tolist())

            for row in range(len(df)):
                for col, colname in enumerate(df.columns):
                    table.setItem(row, col, QTableWidgetItem(str(df.iloc[row, col])))

            dialog_layout.addWidget(table)
            buttons = QDialogButtonBox(QDialogButtonBox.Ok)
            buttons.accepted.connect(dialog.accept)
            dialog_layout.addWidget(buttons)
            dialog.exec_()

        except Exception as e:
            print(f"[CSV Align Error] {e}")
            QMessageBox.critical(self, "Error", f"CSV preview failed:\n{str(e)}")
            

    def do_alignment_and_threshold(self):
        params = self.collect_params()
        df = align_files_to_df(**params)

        if self.auto_threshold_checkbox.isChecked():
            threshold_params = self.collect_threshold_params()
            df = add_thresholds_to_df(df, threshold_params)

        return df

    def set_buttons_enabled(self, enabled: bool):
        self.run_btn.setEnabled(enabled)
        self.preview_btn.setEnabled(enabled)
        
    @thread_worker
    def run_alignment_worker(self):
        return self.do_alignment_and_threshold()

    def run_alignment(self):
        save_path = self.save_csv_line.text().strip()
        if not save_path:
            QMessageBox.warning(self, "Missing Output File", "Please specify an output file using the Browse button.")
            return
        if not save_path.endswith(".csv"):
            save_path += ".csv"

        self.set_buttons_enabled(False)  
        worker = self.run_alignment_worker()

        def handle_success(df):
            df.to_csv(save_path, index=False)
            self.show_info(f"CSV saved to:\n{save_path}")
            self.set_buttons_enabled(True)

        def handle_error(e):
            self.show_error(f"CSV creation failed:\n{str(e)}")
            self.set_buttons_enabled(True)

        worker.returned.connect(handle_success)
        worker.errored.connect(handle_error)
        worker.start()

    def _on_alignment_done(self, df, save_path):
        df.to_csv(save_path, index=False)
        self.show_info(f"CSV saved to:\n{save_path}")

    def show_info(self, message):
        QMessageBox.information(self, "Info", message)

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)


    @thread_worker
    def preview_alignment_worker(self):
        return self.do_alignment_and_threshold()

    def preview_alignment(self):
        self.set_buttons_enabled(False)
        worker = self.preview_alignment_worker()

        def handle_success(df):
            self._show_preview_dialog(df)
            self.set_buttons_enabled(True)

        def handle_error(e):
            self.show_error(f"Preview failed:\n{str(e)}")
            self.set_buttons_enabled(True)

        worker.returned.connect(handle_success)
        worker.errored.connect(handle_error)
        worker.start()

    def _show_preview_dialog(self, df):
        dialog = QDialog(self)
        dialog.setWindowTitle("Preview Aligned DataFrame")
        dialog_layout = QVBoxLayout(dialog)
        table = QTableWidget()
        table.setColumnCount(len(df.columns))
        table.setRowCount(len(df))
        table.setHorizontalHeaderLabels(df.columns.tolist())

        for row in range(len(df)):
            for col, _ in enumerate(df.columns):
                table.setItem(row, col, QTableWidgetItem(str(df.iloc[row, col])))

        dialog_layout.addWidget(table)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dialog.accept)
        dialog_layout.addWidget(buttons)
        dialog.exec_()


    def toggle_threshold_widget(self):
        self.threshold_param_box.setVisible(self.auto_threshold_checkbox.isChecked())
        # self.setVisible(self.auto_threshold_checkbox.isChecked())
            
    def toggle_prefix_suffix(self, text):
        self.prefix_suffix_box.setVisible(text == "Image Name + Prefix/Suffix")

    def show_explanation(self):
        explanation = (

            "<b>Match Type Options:</b><br><br>"
            
            "<b>Same Name:</b> Match only when filenames are exactly the same (excluding extension). "
            "Use when all files share identical names.<br><br>"
            
            "<b>Image Name + Prefix/Suffix:</b> Use image names as base and strip prefix/suffix from segmentation and boundary files. "
            "Set those in the options below.<br><br>"
            
            "<b>File Order:</b> Match by sorting filenames in each folder and aligning row-by-row. "
            "Use when filenames don't match but order is consistent."


        )
        QMessageBox.information(self, "Match Type Explanation", explanation)



def generate_threshold_pairs(image: np.ndarray,
                             mode: str = "otsu_factor",
                             factor_range: tuple = (0.8, 1.2),
                             upper_factor_range: tuple = None,
                             num_steps: int = 5):
    """
    Generate thresholds and upper_thresholds from an image with range checking and int conversion.

    Parameters:
        image (np.ndarray): Grayscale image.
        mode (str): "otsu_factor", "mean_factor", or "actual_values".
        factor_range (tuple): (min_factor, max_factor) for thresholds or (min_value, max_value) for actual_values.
        upper_factor_range (tuple or None): Same as above for upper thresholds. If None, upper_thresholds = None.
        num_steps (int): Number of thresholds to generate (>= 1). If 1, use left bound only.

    Returns:
        (List[int], List[int] or None): thresholds and upper_thresholds
    """
    # Determine image bit-depth-based max value
    dtype = image.dtype
    if np.issubdtype(dtype, np.integer):
        bit_depth = image.dtype.itemsize * 8
        max_val = 2**bit_depth - 1
    else:
        max_val = 1.0  # fallback for float images
    min_val = 0

    # Handle missing upper_factor_range
    if upper_factor_range is not None and upper_factor_range[0] is None and upper_factor_range[1] is None:
        upper_factor_range = None

    # Sanity checks
    if factor_range[0] is None:
        raise ValueError("factor_range[0] (min value) cannot be None.")

    if num_steps > 1 and factor_range[1] is None:
        raise ValueError("factor_range[1] (max value) cannot be None when num_steps > 1.")

    if num_steps > 1 and upper_factor_range is not None and upper_factor_range[1] is None:
        raise ValueError("upper_factor_range[1] (max upper value) cannot be None when num_steps > 1.")


        # raise ValueError("upper_factor_range[0] cannot be None if upper_factor_range is specified.")


    if mode in {"otsu_factor", "mean_factor"}:
        base_val = threshold_otsu(image) if mode == "otsu_factor" else float(np.mean(image))

        min_factor = factor_range[0]
        max_factor = factor_range[0] if num_steps == 1 else factor_range[1]
        thresholds = np.array([min_factor * base_val] if num_steps == 1 else
                              np.linspace(min_factor * base_val, max_factor * base_val, num_steps))

        if upper_factor_range is not None:
            min_u = upper_factor_range[0]
            max_u = upper_factor_range[0] if num_steps == 1 else upper_factor_range[1]
            upper_thresholds = np.array([min_u * base_val] if num_steps == 1 else
                                        np.linspace(min_u * base_val, max_u * base_val, num_steps))

            if np.any(upper_thresholds <= thresholds):
                raise ValueError("Each upper threshold must be greater than its corresponding threshold.")
        else:
            upper_thresholds = None

    elif mode == "actual_values":
        min_thresh = factor_range[0]
        max_thresh = factor_range[0] if num_steps == 1 else factor_range[1]
        thresholds = np.array([min_thresh] if num_steps == 1 else
                              np.linspace(min_thresh, max_thresh, num_steps))

        if upper_factor_range is not None:
            min_u = upper_factor_range[0]
            max_u = upper_factor_range[0] if num_steps == 1 else upper_factor_range[1]
            upper_thresholds = np.array([min_u] if num_steps == 1 else
                                        np.linspace(min_u, max_u, num_steps))
            if np.any(upper_thresholds <= thresholds):
                raise ValueError("Each upper threshold must be greater than its corresponding threshold.")
        else:
            upper_thresholds = None
    else:
        raise ValueError("Unsupported mode. Use 'otsu_factor', 'mean_factor', or 'actual_values'.")

    # Clip and convert
    thresholds = np.clip(thresholds, min_val, max_val).astype(int).tolist()
    if upper_thresholds is not None:
        upper_thresholds = np.clip(upper_thresholds, min_val, max_val).astype(int).tolist()

    return thresholds, upper_thresholds

def add_thresholds_to_df(df, threshold_params):
    for idx, row in df.iterrows():
        img = imread(row["img_path"])
        thres, upper = generate_threshold_pairs(
            img,
            mode=threshold_params["mode"],
            factor_range=threshold_params["factor_range"],
            upper_factor_range=threshold_params["upper_factor_range"],
            num_steps=threshold_params["num_steps"]
        )
        
        df.loc[idx, "thresholds"] = None
        df.loc[idx, "upper_thresholds"] = None
        
        df.at[idx, "thresholds"] = thres
        df.at[idx, "upper_thresholds"] = upper
    return df  


class ThresholdParamWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Threshold Parameter Controls")
        self.setLayout(QVBoxLayout())

        # === Mode selector + Help button ===
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Threshold Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["otsu_factor", "mean_factor", "actual_values"])
        self.mode_combo.setToolTip(
            "Choose method to compute thresholds:\n"
            "- otsu_factor: Otsu value × factor range\n"
            "- mean_factor: Mean image value × factor range\n"
            "- actual_values: Directly sweep a range of values"
        )

        help_btn = QPushButton("?")
        help_btn.setFixedWidth(20)
        help_btn.setToolTip("Click for explanation of threshold parameters")
        help_btn.clicked.connect(self.show_explanation)

        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addWidget(help_btn)
        self.layout().addLayout(mode_layout)

        # === Threshold range inputs ===
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("Threshold Range (min, max):"))
        self.thresh_min = QLineEdit("1")
        self.thresh_min.setToolTip("Left bound of threshold factor or value")
        self.thresh_max = QLineEdit("")
        self.thresh_max.setToolTip("Right bound of threshold factor or value")
        thresh_layout.addWidget(self.thresh_min)
        thresh_layout.addWidget(self.thresh_max)
        self.layout().addLayout(thresh_layout)

       # === Upper threshold range inputs ===
        upper_layout = QHBoxLayout()
        upper_layout.addWidget(QLabel("Upper Threshold Range (min, max):"))
        self.upper_min = QLineEdit("")
        self.upper_min.setToolTip("Left bound of upper threshold (optional)")
        self.upper_max = QLineEdit("")
        self.upper_max.setToolTip("Right bound of upper threshold (optional)")
        upper_layout.addWidget(self.upper_min)
        upper_layout.addWidget(self.upper_max)
        self.layout().addLayout(upper_layout)

        # Number of steps
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("Number of Steps:"))
        self.num_steps = QSpinBox()
        self.num_steps.setMinimum(1)
        self.num_steps.setValue(1)
        self.num_steps.setToolTip("Number of threshold values to generate. 1 = single value")
        step_layout.addWidget(self.num_steps)
        self.layout().addLayout(step_layout)

        # Run button
        run_btn = QPushButton("test params")
        run_btn.clicked.connect(self.print_params)
        self.layout().addWidget(run_btn)

    def get_params(self):
        """Collect thresholding parameters from UI."""
        mode = self.mode_combo.currentText()
        num_steps = self.num_steps.value()

        def parse_range(min_edit, max_edit):
            min_text = min_edit.text().strip()
            max_text = max_edit.text().strip()
            
            
            try:
                min = float(min_text) if min_text else None
            except ValueError:
                min = None
            try:
                max = float(max_text) if max_text else None
            except ValueError:
                max = None
            return (min, max)
            # if not min_text:
            #     min = None
                
            # if not max_text:
            #     max = None
            # if not min_text or not max_text:
            #     return None
            # try:
            #     return float(min_text), float(max_text)
            # except ValueError:
            #     return None

        factor_range = parse_range(self.thresh_min, self.thresh_max)
        upper_factor_range = parse_range(self.upper_min, self.upper_max)

        return {
            "mode": mode,
            "factor_range": factor_range,
            "upper_factor_range": upper_factor_range,
            "num_steps": num_steps
        }

    def print_params(self):
        params = self.get_params()
        print("Threshold Parameters:", params)

    def show_explanation(self):
        text = (
            "<b>Thresholding Parameters Explained:</b><br><br>"
            "<b>Threshold Mode:</b><br>"
            "<b>・otsu_factor</b>: Use Otsu’s method to get a base threshold, then multiply it by min/max factors.<br>"
            "<b>・mean_factor</b>: Use the image mean intensity as the base threshold, then multiply by factor range.<br>"
            "<b>・actual_values</b>: Use absolute intensity values (e.g. 0–255 for 8-bit images) directly as thresholds.<br><br>"
            "<b>Threshold Range (min, max):</b><br>"
            "Defines the range to generate threshold values from.<br>"
            "• For <b>otsu_factor</b> or <b>mean_factor</b>: values are multipliers on the base threshold.<br>"
            "• For <b>actual_values</b>: values are direct pixel intensities.<br>"
            "• If <b>Number of Steps = 1</b>, only the left (min) value is used.<br><br>"
            "<b>Upper Threshold Range (optional):</b><br>"
            "Used to generate a corresponding upper threshold list.<br>"
            "• Must be the same length as the threshold list.<br>"
            "• Each upper threshold must be strictly greater than the corresponding threshold.<br><br>"
            "<b>Number of Steps:</b><br>"
            "Controls how many values are linearly generated between min and max.<br>"
            "• If 1, only a single value (min) is used.<br><br>"
            "<b>Notes:</b><br>"
            "• Leave 'max' empty if only one value is needed.<br>"
            "• For 8-bit images, actual values must be in the range 0–255."
        )
        QMessageBox.information(self, "Threshold Parameters Help", text)
        
        
class SPROUTToolWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.setLayout(QVBoxLayout())

        # Add size reference section first
        self.size_reference = SizeReferenceGroupBox(viewer)
        self.layout().addWidget(self.size_reference)

        self.csv_aligner = CSVAlignerGroupBox()
        self.layout().addWidget(self.csv_aligner)

        # You can append more QGroupBox widgets for other helper tools below...
        # e.g. self.other_tool = OtherGroupBox(viewer)
        # self.layout().addWidget(self.other_tool)


    def showEvent(self, event):
        print("showEvent called")

        
    def hideEvent(self, event):
        print("hideEvent called")
