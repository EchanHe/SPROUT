import numpy as np
from skimage.measure import label as cc_label
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel,
                            QHBoxLayout, QCheckBox, QMessageBox,QSpinBox, QGroupBox,
                            QFormLayout)
from copy import deepcopy
from napari.layers import Labels

from qtpy.QtWidgets import QFrame

import numpy as np

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

        layout = QVBoxLayout()
        
        
        
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
        set_class_btn.setStyleSheet("""QPushButton { font-weight: bold; background-color: #45a049; color: white;}""")
        set_class_btn.setToolTip("Replace all pixels of the source class with the target class.")
        set_class_btn.clicked.connect(self.set_class_operation)
        # layout.addWidget(set_class_btn)
        label_replc_layout.addRow(set_class_btn)
        label_replc_group.setLayout(label_replc_layout)
        layout.addWidget(label_replc_group)


        label_split_group = QGroupBox("Label Split")
        label_split_group.setToolTip("Split a selected label class into connected components.")
        label_split_layout = QFormLayout()
        
        
        self.split_class_spin = QSpinBox()
        self.split_class_spin.setMinimum(0)
        self.split_class_spin.setMaximum(10000)
        self.split_class_spin.setToolTip("Enter the label class ID to split into components.")

        label_split_layout.addRow(QLabel("Target Class to Split:"), self.split_class_spin)
        
        self.sort_checkbox = QCheckBox("Sort by size (reassign IDs by size)")
        self.sort_checkbox.setChecked(True)
        
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
        split_btn.setStyleSheet("""QPushButton { font-weight: bold; background-color: #45a049; color: white;}""")
        split_btn.setToolTip("Split selected label into connected components.")
        split_btn.clicked.connect(self.split_class_operation)
        label_split_layout.addRow(split_btn)
        label_split_group.setLayout(label_split_layout)
        layout.addWidget(label_split_group)

        self.setLayout(layout)

        # Track layer selection
        self.viewer.layers.selection.events.active.connect(self.update_active_label_layer_binding)
        


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

        data = label_layer.data
        if src not in data:
            QMessageBox.information(self, "Info", f"Label {src} not found in the current layer.")
            return


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

        self.original_data_backup = target_layer.data.copy()
        data[data == src] = dst
        target_layer.data = data
        QMessageBox.information(self, "Success", f"Class {src} changed to {dst}.")

        self.clear_selection()

    def split_class_operation(self):
        label_layer = self.last_bound_layer
        if label_layer is None:
            QMessageBox.warning(self, "Error", "No label layer selected.")
            return

        target_class = self.split_class_spin.value()
        keep_top_n = self.top_n_spin.value()
        sort_by_size = self.sort_checkbox.isChecked()




        data = label_layer.data.copy()
        mask = (data == target_class)
        if not np.any(mask):
            QMessageBox.information(self, "Info", f"Class {target_class} not found.")
            return

        # Connected component labeling
        cc = cc_label(mask, connectivity=2)  
        
        num_components = cc.max()

        if num_components <= 1:
            QMessageBox.information(self, "Info", "No split needed.")
            return


        # Decide whether to duplicate the layer
        target_layer = label_layer
        if self.duplicate_checkbox.isChecked():
            duplicated_data = label_layer.data.copy()
            new_name = f"{label_layer.name}_split"
            target_layer = self.viewer.add_labels(duplicated_data, name=new_name)
            print(f"Operating on duplicated layer: {new_name}")
            if self.on_click not in target_layer.mouse_drag_callbacks:
                target_layer.mouse_drag_callbacks.append(self.on_click)
            self.last_bound_layer = target_layer

        # Optional: keep only top-N components
        if keep_top_n > 0:
            sizes = [(cc == i).sum() for i in range(1, num_components + 1)]
            sorted_indices = np.argsort(sizes)[::-1]
            keep_ids = set(sorted_indices[:keep_top_n] + 1)
            cc = np.where(np.isin(cc, list(keep_ids)), cc, 0)

        # update the num_components after filtering
        num_components = cc.max()

        data[mask] = 0 
        for i in range(1, num_components + 1):
            if np.sum(cc == i) == 0:
                continue
            data[cc == i] = i + data.max()  # Reassign new labels
            
        if sort_by_size:
            data,_ = sort_labels_by_size(data)


        # Update layer
        self.original_data_backup = target_layer.data.copy()
        target_layer.data = data
        QMessageBox.information(self, "Success", f"Split class {target_class} into {cc.max()} components.")
        
        self.clear_selection()


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
            target_layer.mouse_drag_callbacks.append(self.on_click)
            self.last_bound_layer = target_layer

        self.original_data_backup = deepcopy(target_layer.data)

        data = target_layer.data.copy()
        mask = np.isin(data, list(self.selected_labels))

        if self.mode_combo.currentText() == "Keep selected labels":
            data = data * mask
        elif self.mode_combo.currentText() == "Remove selected labels":
            data = data * (~mask)

        target_layer.data = data
        self.clear_selection()
        
        

    
        

    
