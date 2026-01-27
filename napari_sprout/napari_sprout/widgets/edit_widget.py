import numpy as np
from skimage.measure import label as cc_label
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel,
                            QHBoxLayout, QCheckBox, QMessageBox,QSpinBox, QGroupBox,
                            QFormLayout, QScrollArea , QSizePolicy)
from copy import deepcopy
from napari.layers import Labels
from napari.utils.notifications import show_info, show_error
from qtpy.QtWidgets import QFrame

import numpy as np
from skimage.morphology import remove_small_holes
from skimage.morphology import erosion, dilation, opening, closing, disk, ball

def sort_labels_by_size(label_img, ignore_label=0):
    label_img = label_img.copy()
    unique_labels = np.unique(label_img)
    unique_labels = unique_labels[unique_labels != ignore_label]

    sizes = {label: np.sum(label_img == label) for label in unique_labels}

    sorted_labels = sorted(sizes.items(), key=lambda x: -x[1])  # [(old_label, size), ...]

    remap = {old: new + 1 for new, (old, _) in enumerate(sorted_labels)}

    new_img = np.zeros_like(label_img)
    for old, new in remap.items():
        new_img[label_img == old] = new

    return new_img, remap

class QtLabelSelector(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.selected_labels = set()
        self.original_data_backup = None
        self.last_bound_layer = None

        
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        
        # Label layer selection
        self.layer_label = QLabel("Active Label Layer: (none)")
        self.layer_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: white;
                background-color: #007ACC;  
                padding: 4px;
                border-radius: 4px;
            }
        """)
        self.layer_label.setToolTip("Displays the name of the currently active Labels layer in Napari.")
        layout.addWidget(self.layer_label)
        
        # Undo button
        undo_btn = QPushButton("Undo")
        undo_btn.clicked.connect(self.undo_last_operation)
        layout.addWidget(undo_btn)
        
        # Duplicate option
        # self.duplicate_combo = QComboBox()
        # self.duplicate_combo.addItems(["No", "Yes"])
        # self.duplicate_combo.setToolTip("Select 'Yes' to apply processing on a duplicate of the original layer.")
        # layout.addWidget(QLabel("Duplicate layer:"))
        # layout.addWidget(self.duplicate_combo)

        duplicate_layout = QHBoxLayout()

        duplicate_label = QLabel("Duplicate layer:")
        duplicate_label.setToolTip("If checked, processing will apply on a duplicated copy of the current label layer.")

        self.duplicate_checkbox = QCheckBox()
        self.duplicate_checkbox.setChecked(False)  # default unchecked
        self.duplicate_checkbox.setToolTip("Check to apply processing on a duplicated layer instead of the original.")

        duplicate_layout.addWidget(duplicate_label)
        duplicate_layout.addWidget(self.duplicate_checkbox)
        duplicate_layout.addStretch()  

        layout.addLayout(duplicate_layout)
        
        # self.layer_combo = QComboBox()
        # self.layer_combo.currentTextChanged.connect(self.on_layer_combo_changed)
        
        
        
        
        # layout.addWidget(self.layer_combo)
        
        # self.update_layer_list()

        # self.viewer.layers.events.inserted.connect(self.update_layer_list)
        # self.viewer.layers.events.removed.connect(self.update_layer_list)


        label_select_group = QGroupBox("Label Selection")
        label_select_group.setToolTip("Select labels by clicking on the image. Use the buttons below to process selected labels.")
        label_select_layout = QFormLayout()
        
        # Label display
        self.label_display = QLabel("Selected labels: []")
        self.label_display.setToolTip("Displays labels you have selected by clicking on the image.")
        # layout.addWidget(self.label_display)
        label_select_layout.addRow(self.label_display)
        
        # Clear button
        clear_btn = QPushButton("Clear Selection")
        clear_btn.setToolTip("Clear all currently selected labels.")
        clear_btn.clicked.connect(self.clear_selection)

        label_select_layout.addRow(clear_btn)

        # Mode select
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Keep selected labels", "Remove selected labels"])
        self.mode_combo.setToolTip("Choose whether to keep or remove the selected labels during processing.")
        # layout.addWidget(QLabel("Mode:"))
        # layout.addWidget(self.mode_combo)
        
        label_select_layout.addRow(QLabel("Mode:") ,self.mode_combo)

        # Run button
        run_btn = QPushButton("Run processing")
        run_btn.setToolTip("Run the main label selection processing according to the chosen mode and options.")
        run_btn.setStyleSheet("""QPushButton { font-weight: bold; background-color: #45a049;}""")
        run_btn.clicked.connect(self.process_selected_operation)
        # layout.addWidget(run_btn)
        label_select_layout.addRow(run_btn)
        label_select_group.setLayout(label_select_layout)
        layout.addWidget(label_select_group)

        # layout.addWidget(create_separator(horizontal=True))
        # Set class selection operation
        # layout.addWidget(QLabel("Replace Label"))
        
        label_replc_group = QGroupBox("Label Replace")
        label_replc_group.setToolTip("Replace all pixels of a source class label with a target class label.")
        label_replc_layout = QFormLayout()
        
        
        self.src_spin = QSpinBox()
        self.src_spin.setMinimum(0)
        self.src_spin.setMaximum(10000)  
        self.src_spin.setToolTip("Enter the source class label you want to replace.")
        
        self.dst_spin = QSpinBox()
        self.dst_spin.setMinimum(0)
        self.dst_spin.setMaximum(10000)
        self.dst_spin.setToolTip("Enter the target class label that will replace the source.")

        label_replc_layout.addRow(QLabel("Source Class:"), self.src_spin)
        label_replc_layout.addRow(QLabel("Target Class:"), self.dst_spin)
        
        # layout.addWidget(QLabel("Source Class:"))
        # layout.addWidget(self.src_spin)

        # layout.addWidget(QLabel("Target Class:"))
        # layout.addWidget(self.dst_spin)

        set_class_btn = QPushButton("Replace")
        set_class_btn.setStyleSheet("""QPushButton { font-weight: bold; background-color: #45a049;}""")
        set_class_btn.setToolTip("Replace all pixels of the source class with the target class.")
        set_class_btn.clicked.connect(self.set_class_operation)
        # layout.addWidget(set_class_btn)
        label_replc_layout.addRow(set_class_btn)
        label_replc_group.setLayout(label_replc_layout)
        layout.addWidget(label_replc_group)


        label_split_group = QGroupBox("Label Split")

        label_split_group.setToolTip("Split a selected label class into connected components.")
        label_split_layout = QFormLayout()

        self.split_all_checkbox = QCheckBox("Split all non-background labels")
        self.split_all_checkbox.setChecked(False)
        self.split_all_checkbox.setToolTip("If checked, split all labels > 0 instead of only the target class.")
        label_split_layout.addRow(self.split_all_checkbox)        
        
        self.split_class_spin = QSpinBox()
        self.split_class_spin.setMinimum(0)
        self.split_class_spin.setMaximum(10000)
        self.split_class_spin.setToolTip("Enter the label class ID to split into components.")

        label_split_layout.addRow(QLabel("Target Class to Split:"), self.split_class_spin)
        
        self.sort_checkbox = QCheckBox("Sort by size (reassign IDs by size)")
        self.sort_checkbox.setChecked(True)
        self.sort_checkbox.setToolTip("If checked, reassign IDs based on component size.")
        label_split_layout.addRow(self.sort_checkbox)
        # layout.addWidget(self.sort_checkbox)


        self.top_n_spin = QSpinBox()
        self.top_n_spin.setMinimum(0)
        self.top_n_spin.setMaximum(10000)
        self.top_n_spin.setToolTip("If set > 0, keep only the top N largest components.")
        # layout.addWidget(QLabel("Keep Top-N Components (0 = all):"))
        # layout.addWidget(self.top_n_spin)
        label_split_layout.addRow(QLabel("Keep Top-N Components (0 = all):"), self.top_n_spin)

        split_btn = QPushButton("Run Split Class")
        split_btn.setStyleSheet("""QPushButton { font-weight: bold; background-color: #45a049;}""")
        split_btn.setToolTip("Split selected label into connected components.")
        split_btn.clicked.connect(self.split_class_operation)
        label_split_layout.addRow(split_btn)
        label_split_group.setLayout(label_split_layout)
        layout.addWidget(label_split_group)

        # ---- Keep Top-N Labels Group ----
        keep_label_group = QGroupBox("Keep Top-N Labels")
        keep_label_group.setToolTip("Keep the N largest labels based on total area/volume.")

        keep_layout = QFormLayout()

        self.keep_label_topn_spin = QSpinBox()
        self.keep_label_topn_spin.setMinimum(1)
        self.keep_label_topn_spin.setMaximum(10_000)
        self.keep_label_topn_spin.setValue(5)
        self.keep_label_topn_spin.setToolTip("Number of labels to keep (by total pixel count).")
        keep_layout.addRow(QLabel("Top-N Labels:"), self.keep_label_topn_spin)

        keep_label_btn = QPushButton("Run Keep Top-N Labels")
        keep_label_btn.setStyleSheet("QPushButton { font-weight: bold; background-color: #3c8dbc; }")
        keep_label_btn.clicked.connect(self.run_keep_top_n_labels)
        keep_layout.addRow(keep_label_btn)

        keep_label_group.setLayout(keep_layout)
        layout.addWidget(keep_label_group)


        # ---- Filter operations Group ----
        filter_group = QGroupBox("Filter Regions")
        filter_group.setToolTip("Remove small connected components or keep only the largest regions.")

        filter_layout = QFormLayout()

        self.filter_min_size_spin = QSpinBox()
        self.filter_min_size_spin.setMinimum(0)
        self.filter_min_size_spin.setMaximum(1_000_000_000)
        self.filter_min_size_spin.setValue(0)
        self.filter_min_size_spin.setToolTip("Remove all components smaller than this size (0 = skip).")
        filter_layout.addRow(QLabel("Minimum Size:"), self.filter_min_size_spin)

        self.filter_top_n_spin = QSpinBox()
        self.filter_top_n_spin.setMinimum(0)
        self.filter_top_n_spin.setMaximum(10_000)
        self.filter_top_n_spin.setValue(0)
        self.filter_top_n_spin.setToolTip("Keep only the top-N largest components (0 = all).")
        filter_layout.addRow(QLabel("Keep Top-N:"), self.filter_top_n_spin)

        self.filter_target_label_spin = QSpinBox()
        self.filter_target_label_spin.setMinimum(0)
        self.filter_target_label_spin.setMaximum(10_000)
        self.filter_target_label_spin.setToolTip("Only filter the given label (0 = all labels > 0).")
        filter_layout.addRow(QLabel("Target Label (0 = all):"), self.filter_target_label_spin)

        filter_btn = QPushButton("Run Filtering")
        filter_btn.setStyleSheet("QPushButton { font-weight: bold; background-color: #d97d00; }")
        filter_btn.clicked.connect(self.run_filtering_operation)
        filter_layout.addRow(filter_btn)

        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)


        # ---- Fill holes operation ----
        fill_group = QGroupBox("Fill Holes")
        fill_group.setToolTip("Fill small holes inside label regions.")

        fill_layout = QFormLayout()

        self.fill_area_spin = QSpinBox()
        self.fill_area_spin.setMinimum(1)
        self.fill_area_spin.setMaximum(1_000_000_000)
        self.fill_area_spin.setValue(64)
        self.fill_area_spin.setToolTip("Maximum hole area to fill (in pixels/voxels).")
        fill_layout.addRow(QLabel("Area Threshold:"), self.fill_area_spin)

        self.fill_in_2d_checkbox = QCheckBox("Apply per-slice (2D)")
        self.fill_in_2d_checkbox.setChecked(False)
        self.fill_in_2d_checkbox.setToolTip("Apply hole filling on each 2D slice (if 3D).")
        fill_layout.addRow(self.fill_in_2d_checkbox)

        self.fill_target_spin = QSpinBox()
        self.fill_target_spin.setMinimum(0)
        self.fill_target_spin.setMaximum(10_000)
        self.fill_target_spin.setToolTip("Target label to process (0 = all labels > 0).")
        fill_layout.addRow(QLabel("Target Label (0 = all):"), self.fill_target_spin)

        fill_btn = QPushButton("Run Fill Holes")
        fill_btn.setStyleSheet("QPushButton { font-weight: bold; background-color: #0066aa; }")
        fill_btn.clicked.connect(self.run_fill_holes_operation)
        fill_layout.addRow(fill_btn)

        fill_group.setLayout(fill_layout)
        layout.addWidget(fill_group)

        # ---- Morphology Group ----
        

        morph_group = QGroupBox("Morphology Transform")
        morph_group.setToolTip("Apply morphological operations to label regions.")

        morph_layout = QFormLayout()

        self.morph_op_combo = QComboBox()
        self.morph_op_combo.addItems(["Erode", "Dilate", "Open", "Close"])
        self.morph_op_combo.setToolTip("Choose morphological operation to apply.")
        morph_layout.addRow(QLabel("Operation:"), self.morph_op_combo)

        self.morph_kernel_spin = QSpinBox()
        self.morph_kernel_spin.setMinimum(1)
        self.morph_kernel_spin.setMaximum(50)
        self.morph_kernel_spin.setValue(2)
        self.morph_kernel_spin.setToolTip("Kernel radius for the operation.")
        morph_layout.addRow(QLabel("Kernel Radius:"), self.morph_kernel_spin)

        self.morph_target_spin = QSpinBox()
        self.morph_target_spin.setMinimum(0)
        self.morph_target_spin.setMaximum(10_000)
        self.morph_target_spin.setToolTip("Target label to process (0 = all labels > 0).")
        morph_layout.addRow(QLabel("Target Label (0 = all):"), self.morph_target_spin)

        morph_btn = QPushButton("Run Morphology")
        morph_btn.setStyleSheet("QPushButton { font-weight: bold; background-color: #8844aa; }")
        morph_btn.clicked.connect(self.run_morphology_operation)
        morph_layout.addRow(morph_btn)

        morph_group.setLayout(morph_layout)
        layout.addWidget(morph_group)


        content_widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        content_widget.setMinimumWidth(content_widget.sizeHint().width())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content_widget)

        # 3. 最外层 layout 包裹 scroll
        outer_layout = QVBoxLayout()
        outer_layout.addWidget(scroll)
        self.setLayout(outer_layout)

        # self.setLayout(layout)

        # Track layer selection
        self.viewer.layers.selection.events.active.connect(self.update_active_label_layer_binding)
        

        self.all_edit_buttons = [
            fill_btn, morph_btn, split_btn, filter_btn,
            keep_label_btn, run_btn, set_class_btn
        ]

    def update_layer_list(self, *args):
        # deprecated method
        
        # update the combo box with current label layers
        self.layer_combo.blockSignals(True) 
        
        self.layer_combo.clear()
        label_layers = [layer.name for layer in self.viewer.layers if isinstance(layer, Labels)]
        self.layer_combo.addItems(label_layers)

        self.layer_combo.blockSignals(False)


    def on_layer_combo_changed(self, layer_name):
        # If the selected layer is a Labels layer, update the class combos
        layer = self.viewer.layers[layer_name]
        if isinstance(layer, Labels):
            
            # Bind the click handler to the selected layer
            if self.last_bound_layer and self.on_click in self.last_bound_layer.mouse_drag_callbacks:
                self.last_bound_layer.mouse_drag_callbacks.remove(self.on_click)
            if self.on_click not in layer.mouse_drag_callbacks:
                layer.mouse_drag_callbacks.append(self.on_click)
            self.last_bound_layer = layer
        else:
            if self.last_bound_layer and self.on_click in self.last_bound_layer.mouse_drag_callbacks:
                self.last_bound_layer.mouse_drag_callbacks.remove(self.on_click)
            self.last_bound_layer = None
            
   

    def get_selected_layer(self) -> Labels:
        # selected_name = self.layer_combo.currentText()
        # return self.viewer.layers[selected_name]

        layer_name = self.combo.currentText()
        layer = self.viewer.layers.get(layer_name, None)
        if isinstance(layer, Labels):
            return layer
        return None

    def update_display_label(self):
        label_str = ", ".join(str(lbl) for lbl in sorted(self.selected_labels))
        self.label_display.setText(f"Selected labels: {label_str}")

    def on_click(self, layer, event):
        if not isinstance(layer, Labels):
            return
        if "Control" not in event.modifiers:
            return

        label = layer.get_value(event.position,
                                view_direction=event.view_direction,
                                dims_displayed=event.dims_displayed)
        if label is None or label == 0:
            return

        if label in self.selected_labels:
            self.selected_labels.remove(label)
        else:
            self.selected_labels.add(label)

        self.update_display_label()


    def set_class_operation(self):
        label_layer = self.last_bound_layer
        if label_layer is None:
            QMessageBox.warning(self, "Error", "No label layer selected.")
            return

        src = self.src_spin.value()
        dst = self.dst_spin.value()

        # Decide whether to duplicate the layer
        target_layer = label_layer
        if self.duplicate_checkbox.isChecked():
            duplicated_data = label_layer.data.copy()
            new_name = f"{label_layer.name}_set"
            target_layer = self.viewer.add_labels(duplicated_data, name=new_name)
            print(f"Operating on duplicated layer: {new_name}")
            if self.on_click not in target_layer.mouse_drag_callbacks:
                target_layer.mouse_drag_callbacks.append(self.on_click)
            self.last_bound_layer = target_layer
        print(f"Running set_class on layer: {target_layer.name}")
        data = target_layer.data.copy()
        self.original_data_backup = target_layer.data.copy()
        data[data == src] = dst
        target_layer.data = data
        # QMessageBox.information(self, "Success", f"Class {src} changed to {dst}.")
        show_info(f"Class {src} changed to {dst} in layer {target_layer.name}.")

        self.clear_selection()

    ### commented out code for split_class_operation,   see below for the new implementation
    # def split_class_operation(self):
    #     label_layer = self.last_bound_layer
    #     if label_layer is None:
    #         QMessageBox.warning(self, "Error", "No label layer selected.")
    #         return

    #     target_class = self.split_class_spin.value()
    #     keep_top_n = self.top_n_spin.value()
    #     sort_by_size = self.sort_checkbox.isChecked()

    #     data = label_layer.data.copy()
    #     mask = (data == target_class)
    #     if not np.any(mask):
    #         QMessageBox.information(self, "Info", f"Class {target_class} not found.")
    #         return

    #     # Connected component labeling
    #     cc = cc_label(mask, connectivity=2)  
        
    #     num_components = cc.max()

    #     if num_components <= 1:
    #         QMessageBox.information(self, "Info", "No split needed.")
    #         return


    #     # Decide whether to duplicate the layer
    #     target_layer = label_layer
    #     if self.duplicate_checkbox.isChecked():
    #         duplicated_data = label_layer.data.copy()
    #         new_name = f"{label_layer.name}_split"
    #         target_layer = self.viewer.add_labels(duplicated_data, name=new_name)
    #         print(f"Operating on duplicated layer: {new_name}")
    #         if self.on_click not in target_layer.mouse_drag_callbacks:
    #             target_layer.mouse_drag_callbacks.append(self.on_click)
    #         self.last_bound_layer = target_layer

    #     # Optional: keep only top-N components
    #     if keep_top_n > 0:
    #         sizes = [(cc == i).sum() for i in range(1, num_components + 1)]
    #         sorted_indices = np.argsort(sizes)[::-1]
    #         keep_ids = set(sorted_indices[:keep_top_n] + 1)
    #         cc = np.where(np.isin(cc, list(keep_ids)), cc, 0)

    #     # update the num_components after filtering
    #     num_components = cc.max()

    #     data[mask] = 0 
    #     new_label_base = data.max()
    #     for i in range(1, num_components + 1):
    #         data[cc == i] = i + new_label_base
            
    #     if sort_by_size:
    #         data,_ = sort_labels_by_size(data)


    #     # Update layer
    #     self.original_data_backup = target_layer.data.copy()
    #     target_layer.data = data
    #     QMessageBox.information(self, "Success", f"Split class {target_class} into {cc.max()} components.")
        
    #     self.clear_selection()


    def split_class_operation(self):
        label_layer = self.last_bound_layer
        if label_layer is None:
            QMessageBox.warning(self, "Error", "No label layer selected.")
            return

        target_class = self.split_class_spin.value()
        keep_top_n = self.top_n_spin.value()
        sort_by_size = self.sort_checkbox.isChecked()

        data = label_layer.data.copy()

        # Duplicate if needed
        target_layer = label_layer
        if self.duplicate_checkbox.isChecked():
            new_name = f"{label_layer.name}_split"
            target_layer = self.viewer.add_labels(data.copy(), name=new_name)
            print(f"Operating on duplicated layer: {new_name}")
            if self.on_click not in target_layer.mouse_drag_callbacks:
                target_layer.mouse_drag_callbacks.append(self.on_click)
            self.last_bound_layer = target_layer
            data = target_layer.data.copy()
        print(f"Running split_class on layer: {target_layer.name}")
        
        self.original_data_backup = data.copy()
        ##
        
        def on_done(result):
            if target_layer not in self.viewer.layers:
                print("Layer was removed before update.")
                return
            target_layer.data = result
            if self.split_all_checkbox.isChecked():
                msg = f"Split all non-background labels completed in layer {target_layer.name}."
            else:
                msg = f"Split class {target_class} completed in layer {target_layer.name}."
            show_info(msg)
            self.clear_selection()

        if self.split_all_checkbox.isChecked():
            worker_func = self._do_split_all_classes
            worker_args = [data, keep_top_n, sort_by_size]
            print("Split mode: all non-background labels")
        else:
            # Early check if split is needed
            mask = (data == target_class)
            if not np.any(mask):
                QMessageBox.information(self, "Info", f"Class {target_class} not found.")
                return

            cc = cc_label(mask, connectivity=2)
            if cc.max() <= 1:
                QMessageBox.information(self, "Info", "No split needed.")
                return
            worker_func = self._do_split_class
            worker_args = [data, target_class, keep_top_n, sort_by_size]
            print(f"Split mode: single class {target_class}")

        self.run_in_background(
            worker_func,
            worker_args,
            on_done,
            buttons_to_disable=self.all_edit_buttons
        )

        # self.run_in_background(
        #     self._do_split_class,
        #     [data, target_class, keep_top_n, sort_by_size],
        #     on_done,
        #     buttons_to_disable= self.all_edit_buttons
        # )


    def _split_single_class(
        self,
        data: np.ndarray,
        target_class: int,
        keep_top_n: int,
        sort_by_size: bool,
        start_label_base: int,
    ):
        """Process a single target_class on data and return (result, new_label_base).

        - data: original label image
        - start_label_base: starting offset for numbering new components (usually pass current result.max())
        """
        result = data.copy()
        mask = (result == target_class)
        if not np.any(mask):
            return result, start_label_base

        cc = cc_label(mask, connectivity=2)
        num_components = cc.max()
        if num_components <= 1:
            return result, start_label_base

        if keep_top_n > 0:
            sizes = [(cc == i).sum() for i in range(1, num_components + 1)]
            sorted_indices = np.argsort(sizes)[::-1]
            keep_ids = set(sorted_indices[:keep_top_n] + 1)
            cc = np.where(np.isin(cc, list(keep_ids)), cc, 0)
            num_components = cc.max()

        result[mask] = 0
        base = start_label_base
        for i in range(1, num_components + 1):
            result[cc == i] = base + i

        if sort_by_size:
            result, _ = sort_labels_by_size(result)
            base = result.max()
        else:
            base = base + num_components

        return result, base
    
    def _do_split_class(self, data: np.ndarray, target_class: int, keep_top_n: int, sort_by_size: bool) -> np.ndarray:
        result, _ = self._split_single_class(
            data=data,
            target_class=target_class,
            keep_top_n=keep_top_n,
            sort_by_size=sort_by_size,
            start_label_base=data.max()
        )
        return result

    def _do_split_all_classes(self, data: np.ndarray, keep_top_n: int, sort_by_size: bool) -> np.ndarray:
        result = data.copy()
        labels = np.unique(result)
        labels = labels[labels != 0]

        if labels.size == 0:
            return result

        # Starting base: current max label
        label_base = result.max()

        for lbl in labels:
            result, label_base = self._split_single_class(
                data=result,
                target_class=lbl,
                keep_top_n=keep_top_n,
                sort_by_size=sort_by_size,
                start_label_base=label_base
            )

        return result

    def update_active_label_layer_binding(self, event):
        layer = event.value

        # unbind from the last layer if it exists
        if self.last_bound_layer and self.on_click in self.last_bound_layer.mouse_drag_callbacks:
            self.last_bound_layer.mouse_drag_callbacks.remove(self.on_click)
        self.last_bound_layer = None
        if isinstance(layer, Labels):
            if self.on_click not in layer.mouse_drag_callbacks:
                layer.mouse_drag_callbacks.append(self.on_click)
            self.last_bound_layer = layer
            
            
            self.layer_label.setText(f"Active Label Layer: {layer.name}")
            print(f"Bound to active label layer: {layer.name}")
        else:
            self.layer_label.setText("Active Label Layer: (none)")

    def clear_selection(self):
        self.selected_labels.clear()
        self.update_display_label()
        print("Selection cleared.")

    def undo_last_operation(self):
        if self.original_data_backup is not None and self.last_bound_layer is not None:
            self.last_bound_layer.data = deepcopy(self.original_data_backup)
            self.original_data_backup = None
            print("Undo completed.")
        else:
            print("Nothing to undo.")

    def process_selected_operation(self):
        if not self.selected_labels:
            print("No labels selected.")
            return

        label_layer = self.last_bound_layer

        # label_layer = self.get_selected_layer()
        if label_layer is None:
            QMessageBox.warning(self, "Error", "No valid label layer selected.")
            return

        if label_layer is None:
            print("No label layer bound.")
            return

        target_layer = label_layer
        # if self.duplicate_combo.currentText() == "Yes":
        if self.duplicate_checkbox.isChecked():
            duplicated_data = label_layer.data.copy()
            new_name = f"{label_layer.name}_filtered"
            target_layer = self.viewer.add_labels(duplicated_data, name=new_name)
            print(f"Operating on duplicated layer: {new_name}")
            if self.on_click not in target_layer.mouse_drag_callbacks:
                target_layer.mouse_drag_callbacks.append(self.on_click)
            self.last_bound_layer = target_layer
        print(f"Running label selection on layer: {target_layer.name}")

        self.original_data_backup = deepcopy(target_layer.data)

        data = target_layer.data.copy()
        mask = np.isin(data, list(self.selected_labels))

        if self.mode_combo.currentText() == "Keep selected labels":
            data = data * mask
        elif self.mode_combo.currentText() == "Remove selected labels":
            data = data * (~mask)

        target_layer.data = data
        
        kept_or_removed = "kept" if self.mode_combo.currentText() == "Keep selected labels" else "removed"
        label_str = ", ".join(str(lbl) for lbl in sorted(self.selected_labels))
        
        # QMessageBox.information(
        #     self,
        #     "Done",
        #     f"Label selection processing completed.\n"
        #     f"{kept_or_removed.capitalize()} labels: [{label_str}]"
        # )
        
        show_info(
            f"Label selection processing completed in layer {target_layer.name}.\n"
            f"{kept_or_removed.capitalize()} labels: [{label_str}]"
        )
        
        self.clear_selection()

    def run_fill_holes_operation(self):
        label_layer = self.last_bound_layer
        if label_layer is None:
            QMessageBox.warning(self, "Error", "No label layer selected.")
            return

        area_threshold = self.fill_area_spin.value()
        apply_in_2d = self.fill_in_2d_checkbox.isChecked()
        target_label = self.fill_target_spin.value()
        if target_label == 0:
            target_label = None  # means process all labels > 0

        data = label_layer.data.copy()

        # Duplicate if needed
        target_layer = label_layer
        if self.duplicate_checkbox.isChecked():
            new_name = f"{label_layer.name}_fill"
            target_layer = self.viewer.add_labels(data.copy(), name=new_name)
            print(f"Operating on duplicated layer: {new_name}")
            if self.on_click not in target_layer.mouse_drag_callbacks:
                target_layer.mouse_drag_callbacks.append(self.on_click)
            self.last_bound_layer = target_layer
            data = target_layer.data.copy()
        print(f"Running fill_holes on layer: {target_layer.name}")
        self.original_data_backup = data.copy()  # for undo

        # Define a callback to run the long task in background
        def on_done(result):
            if target_layer not in self.viewer.layers:
                print("Layer was removed before completion.")
                return
            target_layer.data = result
            # QMessageBox.information(self, "Done", "Hole filling completed.")
            show_info(f"Hole filling completed in layer {target_layer.name}.")

        # Run the long task in background
        self.run_in_background(
            self.fill_holes_in_labels,
            [data, area_threshold, target_label, apply_in_2d],
            on_done,
            buttons_to_disable=self.all_edit_buttons
        )


    
    def fill_holes_in_labels(self, label_img, area_threshold=64, target_label=None, apply_in_2d=False):
        result = label_img.copy()
        ndim = result.ndim

        if target_label is not None:
            mask = (label_img == target_label)
            if mask.sum() == 0:
                return result
            
            max_area = np.prod(mask.shape) * 0.25
            current_area_threshold = min(area_threshold, max_area)

            if apply_in_2d and ndim == 3:
                for z in range(mask.shape[0]):
                    filled = remove_small_holes(mask[z], area_threshold=current_area_threshold)
                    result[z][filled & (~mask[z])] = target_label
            else:
                filled = remove_small_holes(mask, area_threshold=current_area_threshold)
                result[filled & (~mask)] = target_label
        else:
            unique_labels = np.unique(label_img)
            for lbl in unique_labels:
                if lbl == 0:
                    continue
                mask = (label_img == lbl)
                if mask.sum() == 0:
                    continue
                max_area = np.prod(mask.shape) * 0.25
                current_area_threshold = min(area_threshold, max_area)

                if apply_in_2d and ndim == 3:
                    for z in range(mask.shape[0]):
                        filled = remove_small_holes(mask[z], area_threshold=current_area_threshold)
                        result[z][filled & (~mask[z])] = lbl
                else:
                    filled = remove_small_holes(mask, area_threshold=current_area_threshold)
                    result[filled & (~mask)] = lbl
        return result

    ###commented out code for morphology operation, use the one below instead
    # def run_morphology_operation(self):
    #     label_layer = self.last_bound_layer
    #     if label_layer is None:
    #         QMessageBox.warning(self, "Error", "No label layer selected.")
    #         return

    #     op_name = self.morph_op_combo.currentText()
    #     radius = self.morph_kernel_spin.value()
    #     target_label = self.morph_target_spin.value()
    #     if target_label == 0:
    #         target_label = None  # process all labels

    #     op_map = {
    #         "Erode": erosion,
    #         "Dilate": dilation,
    #         "Open": opening,
    #         "Close": closing
    #     }
    #     morph_func = op_map.get(op_name)
    #     if morph_func is None:
    #         QMessageBox.warning(self, "Error", f"Unsupported operation: {op_name}")
    #         return

    #     data = label_layer.data.copy()
    #     ndim = data.ndim

    #     # Define structuring element
    #     if ndim == 2:
    #         selem = disk(radius)
    #     elif ndim == 3:
    #         selem = ball(radius)
    #     else:
    #         QMessageBox.warning(self, "Error", "Only 2D or 3D data is supported.")
    #         return

    #     # Duplicate if requested
    #     target_layer = label_layer
    #     if self.duplicate_checkbox.isChecked():
    #         new_name = f"{label_layer.name}_morph"
    #         target_layer = self.viewer.add_labels(data.copy(), name=new_name)
    #         print(f"Operating on duplicated layer: {new_name}")
    #         if self.on_click not in target_layer.mouse_drag_callbacks:
    #             target_layer.mouse_drag_callbacks.append(self.on_click)
    #         self.last_bound_layer = target_layer
    #         data = target_layer.data.copy()

    #     self.original_data_backup = data.copy()

    #     result = np.zeros_like(data)

    #     if target_label is not None:
    #         mask = (data == target_label)
    #         if not np.any(mask):
    #             QMessageBox.information(self, "Info", f"Label {target_label} not found.")
    #             return
    #         transformed = morph_func(mask, selem)
    #         result[transformed] = target_label
    #     else:
    #         unique_labels = np.unique(data)
    #         for lbl in unique_labels:
    #             if lbl == 0:
    #                 continue
    #             mask = (data == lbl)
    #             if not np.any(mask):
    #                 continue
    #             transformed = morph_func(mask, selem)
    #             result[transformed] = lbl

    #     target_layer.data = result
    #     QMessageBox.information(self, "Done", f"{op_name} operation completed.")

    

    def run_morphology_operation(self):
        label_layer = self.last_bound_layer
        if label_layer is None:
            QMessageBox.warning(self, "Error", "No label layer selected.")
            return

        op_name = self.morph_op_combo.currentText()
        radius = self.morph_kernel_spin.value()
        target_label = self.morph_target_spin.value()
        if target_label == 0:
            target_label = None

        data = label_layer.data.copy()
        ndim = data.ndim

        if ndim == 2:
            selem = disk(radius)
        elif ndim == 3:
            selem = ball(radius)
        else:
            QMessageBox.warning(self, "Error", "Only 2D or 3D data is supported.")
            return

        print(f"Running {op_name} on layer: {label_layer.name}")
        self.original_data_backup = data.copy()

        # callback to run the long task in background
        def on_done(result):
            if self.duplicate_checkbox.isChecked():
                new_name = f"{label_layer.name}_morph"
                new_layer = self.viewer.add_labels(result, name=new_name)
                print(f"Created new layer: {new_name}")
                if self.on_click not in new_layer.mouse_drag_callbacks:
                    new_layer.mouse_drag_callbacks.append(self.on_click)
                self.last_bound_layer = new_layer
            else:
                if label_layer not in self.viewer.layers:
                    print("Original layer was removed before update.")
                    return
                label_layer.data = result
                self.last_bound_layer = label_layer

            show_info(f"{op_name} operation completed.")

        # background task to perform morphology
        self.run_in_background(
            self._do_morphology,
            [data, target_label, op_name, selem],
            on_done,
            buttons_to_disable= self.all_edit_buttons
        )


    def _do_morphology(self, data: np.ndarray, target_label: int, op_name: str, selem) -> np.ndarray:
        op_map = {
            "Erode": erosion,
            "Dilate": dilation,
            "Open": opening,
            "Close": closing
        }
        morph_func = op_map.get(op_name)
        if morph_func is None:
            raise ValueError(f"Unsupported operation: {op_name}")

        result = np.zeros_like(data)

        
        
        if target_label is not None:
            mask = (data == target_label)
            if not np.any(mask):
                return result
            transformed = morph_func(mask, selem)
            result[transformed] = target_label

            preserve_mask = (data != 0) & (data != target_label)
            result[preserve_mask] = data[preserve_mask]

        else:
            unique_labels = np.unique(data)
            for lbl in unique_labels:
                if lbl == 0:
                    continue
                mask = (data == lbl)
                if not np.any(mask):
                    continue
                transformed = morph_func(mask, selem)
                result[transformed] = lbl

        return result

    ### commented out code for filtering operation, use the one below instead
    # def run_filtering_operation(self):
    #     label_layer = self.last_bound_layer
    #     if label_layer is None:
    #         QMessageBox.warning(self, "Error", "No label layer selected.")
    #         return

    #     min_size = self.filter_min_size_spin.value()
    #     top_n = self.filter_top_n_spin.value()
    #     target_label = self.filter_target_label_spin.value()
    #     if target_label == 0:
    #         target_label = None

    #     if min_size == 0 and top_n == 0:
    #         QMessageBox.information(self, "Info", "No filtering performed (min size and top-N both zero).")
    #         return

    #     data = label_layer.data.copy()
    #     ndim = data.ndim
    #     target_layer = label_layer

    #     if self.duplicate_checkbox.isChecked():
    #         new_name = f"{label_layer.name}_filtered"
    #         target_layer = self.viewer.add_labels(data.copy(), name=new_name)
    #         print(f"Operating on duplicated layer: {new_name}")
    #         if self.on_click not in target_layer.mouse_drag_callbacks:
    #             target_layer.mouse_drag_callbacks.append(self.on_click)
    #         self.last_bound_layer = target_layer
    #         data = target_layer.data.copy()

    #     self.original_data_backup = data.copy()
    #     result = data.copy()

    #     if target_label is not None:
    #         mask = (data == target_label)
    #         result = self._filter_single_label(mask, result, target_label, min_size, top_n, ndim)
    #     else:
    #         unique_labels = np.unique(data)
    #         for lbl in unique_labels:
    #             if lbl == 0:
    #                 continue
    #             mask = (data == lbl)
    #             result = self._filter_single_label(mask, result, lbl, min_size, top_n, ndim)

    #     target_layer.data = result
    #     QMessageBox.information(self, "Done", "Filtering completed.")

    def run_filtering_operation(self):
        label_layer = self.last_bound_layer
        if label_layer is None:
            QMessageBox.warning(self, "Error", "No label layer selected.")
            return

        min_size = self.filter_min_size_spin.value()
        top_n = self.filter_top_n_spin.value()
        target_label = self.filter_target_label_spin.value()
        if target_label == 0:
            target_label = None

        if min_size == 0 and top_n == 0:
            QMessageBox.information(self, "Info", "No filtering performed (min size and top-N both zero).")
            return

        data = label_layer.data.copy()
        ndim = data.ndim

        # Duplicate if needed
        target_layer = label_layer
        if self.duplicate_checkbox.isChecked():
            new_name = f"{label_layer.name}_filtered"
            target_layer = self.viewer.add_labels(data.copy(), name=new_name)
            print(f"Operating on duplicated layer: {new_name}")
            if self.on_click not in target_layer.mouse_drag_callbacks:
                target_layer.mouse_drag_callbacks.append(self.on_click)
            self.last_bound_layer = target_layer
            data = target_layer.data.copy()
        print(f"Running filtering on layer: {target_layer.name}")
        self.original_data_backup = data.copy()

        # callback to run the long task in background
        def on_done(result):
            if target_layer not in self.viewer.layers:
                print("Layer removed before update.")
                return
            target_layer.data = result
            # QMessageBox.information(self, "Done", "Filtering completed.")
            show_info(f"Filtering completed in layer {target_layer.name}.")

        # start the background task
        self.run_in_background(
            self._do_filtering,
            [data, target_label, min_size, top_n, ndim],
            on_done,
            buttons_to_disable= self.all_edit_buttons
        )
    

    def _do_filtering(self, data: np.ndarray, target_label: int, min_size: int, top_n: int, ndim: int) -> np.ndarray:
        result = data.copy()

        if target_label is not None:
            mask = (data == target_label)
            result = self._filter_single_label(mask, result, target_label, min_size, top_n, ndim)
        else:
            unique_labels = np.unique(data)
            for lbl in unique_labels:
                if lbl == 0:
                    continue
                mask = (data == lbl)
                result = self._filter_single_label(mask, result, lbl, min_size, top_n, ndim)

        return result


    def _filter_single_label(self, mask, result, label_val, min_size, top_n, ndim):
        labeled = cc_label(mask, connectivity=1)
        component_ids = np.unique(labeled)[1:]  # skip background 0

        # Compute size of each component
        sizes = {i: np.sum(labeled == i) for i in component_ids}

        # Remove small components
        keep_ids = [i for i, size in sizes.items() if size >= min_size] if min_size > 0 else list(component_ids)

        # Keep only top-N if specified
        if top_n > 0 and len(keep_ids) > top_n:
            sorted_by_size = sorted(keep_ids, key=lambda x: sizes[x], reverse=True)
            keep_ids = sorted_by_size[:top_n]

        keep_mask = np.isin(labeled, keep_ids)
        result[mask] = 0  # remove all current pixels of this label
        result[keep_mask] = label_val
        return result

    def run_keep_top_n_labels(self):
        label_layer = self.last_bound_layer
        if label_layer is None:
            QMessageBox.warning(self, "Error", "No label layer selected.")
            return

        top_n = self.keep_label_topn_spin.value()
        data = label_layer.data.copy()

        target_layer = label_layer
        if self.duplicate_checkbox.isChecked():
            new_name = f"{label_layer.name}_keeplabels"
            target_layer = self.viewer.add_labels(data.copy(), name=new_name)
            print(f"Operating on duplicated layer: {new_name}")
            if self.on_click not in target_layer.mouse_drag_callbacks:
                target_layer.mouse_drag_callbacks.append(self.on_click)
            self.last_bound_layer = target_layer
            data = target_layer.data.copy()
        print(f"Running keep top-N labels on layer: {target_layer.name}")
        
        self.original_data_backup = data.copy()


        unique_labels = np.unique(data)
        unique_labels = unique_labels[unique_labels != 0]
        if top_n >= len(unique_labels):
            QMessageBox.information(self, "Info", f"Only {len(unique_labels)} labels found. No filtering needed.")
            return

        def on_done(result):
            if target_layer not in self.viewer.layers:
                print("Layer was removed before update.")
                return
            target_layer.data = result
            # QMessageBox.information(self, "Done", f"Kept top {top_n} labels.")
            show_info(f"Kept top {top_n} labels in layer {target_layer.name}.")

        self.run_in_background(
            self._do_keep_top_n_labels,
            [data, top_n],
            on_done,
            buttons_to_disable=[self.keep_label_btn] if hasattr(self, "keep_label_btn") else []
        )

    def _do_keep_top_n_labels(self, data: np.ndarray, top_n: int) -> np.ndarray:
        labels = np.unique(data)
        labels = labels[labels != 0]

        if top_n >= len(labels):
            return data  # 不需要处理

        sizes = {lbl: np.sum(data == lbl) for lbl in labels}
        sorted_labels = sorted(sizes.items(), key=lambda x: x[1], reverse=True)

        keep_labels = [lbl for lbl, _ in sorted_labels[:top_n]]

        result = np.where(np.isin(data, keep_labels), data, 0)
        return result



    def run_in_background(self, func, args, done_callback, buttons_to_disable=None):
        from napari.qt.threading import thread_worker

        @thread_worker
        def worker_wrapper(*args):
            return func(*args)

        worker = worker_wrapper(*args)

        if buttons_to_disable is None:
            buttons_to_disable = []

        # set buttons to disabled state during the background task
        for btn in buttons_to_disable:
            btn.setEnabled(False)

        def on_done(result):
            # when the background task is done, re-enable buttons
            for btn in buttons_to_disable:
                btn.setEnabled(True)
            done_callback(result)

        def on_error(e):
            print(f"Background task failed: {e}")
            for btn in buttons_to_disable:
                btn.setEnabled(True)
            QMessageBox.warning(self, "Error", f"Operation failed:\n{e}")

        worker.returned.connect(on_done)
        worker.errored.connect(on_error)
        worker.start()