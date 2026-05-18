import numpy as np
import json
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QListWidget, QLineEdit, QMessageBox, QFileDialog, QRadioButton, QButtonGroup
)
from qtpy.QtCore import Qt
from napari.layers import Labels

class LabelMapperWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.label_layer = None

        self.scheme = []  # e.g., ["a", "b", "c"]
        self.label2class = {}  # {label: class_name}
        self.undo_stack = []
        self.current_class_index = 0

        self.setWindowTitle("Semantic Label Mapper")
        self.setLayout(QVBoxLayout())

        # --- Layer Info ---
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
        self.layer_label.setToolTip("Currently active label layer in the viewer. Labels will be assigned based on this layer.")
        self.layout().addWidget(self.layer_label)

        # --- Scheme Controls (more compact layout) ---
        # Top row: Load / Save
        scheme_controls_top = QHBoxLayout()
        self.load_scheme_btn = QPushButton("Load Scheme")
        self.save_scheme_btn = QPushButton("Save Scheme")
        self.load_scheme_btn.setToolTip("Load a class scheme from a JSON file. The scheme defines available semantic classes.")
        self.save_scheme_btn.setToolTip("Save the current class scheme to a JSON file.")
        scheme_controls_top.addWidget(self.load_scheme_btn)
        scheme_controls_top.addWidget(self.save_scheme_btn)
        self.layout().addLayout(scheme_controls_top)

        # Second row: class input + add/remove/clear
        scheme_controls = QHBoxLayout()
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("Class name")
        self.add_class_btn = QPushButton("Add Class")
        self.remove_class_btn = QPushButton("Remove Class")
        # Clear all classes button
        self.clear_classes_btn = QPushButton("Remove All Classes")

        # Set tooltips for scheme controls
        self.add_class_btn.setToolTip("Add a new semantic class to the scheme.")
        self.remove_class_btn.setToolTip("Remove the specified class name from the scheme.")
        self.class_input.setToolTip("Type the name of the class to add or remove.")
        self.clear_classes_btn.setToolTip("Remove all classes from the scheme and clear any associated mappings.")

        scheme_controls.addWidget(self.class_input)
        scheme_controls.addWidget(self.add_class_btn)
        scheme_controls.addWidget(self.remove_class_btn)
        scheme_controls.addWidget(self.clear_classes_btn)
        self.layout().addLayout(scheme_controls)

        # Scheme list
        self.scheme_list = QListWidget()
        self.scheme_list.setToolTip("Shows all classes in the current scheme and the labels assigned to each.")
        self.layout().addWidget(self.scheme_list)

        # --- Operation Mode ---
        mode_layout = QHBoxLayout()
        self.radio_auto = QRadioButton("Auto")
        self.radio_manual = QRadioButton("Manual")
        self.radio_manual.setChecked(True)
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.radio_auto)
        self.mode_group.addButton(self.radio_manual)
        
        self.radio_auto.setToolTip("Automatically switch to the next class after assigning a label.")
        self.radio_manual.setToolTip("Manually select the class to assign labels to.")
        
        mode_layout.addWidget(QLabel("Operation Mode:"))
        mode_layout.addWidget(self.radio_auto)
        mode_layout.addWidget(self.radio_manual)
        self.layout().addLayout(mode_layout)

        # --- Current Class ---
        current_class_layout = QHBoxLayout()
        self.class_dropdown = QComboBox()
        self.undo_btn = QPushButton("Undo")
        
        self.clear_current_btn = QPushButton("Reset This Mapping")
        self.clear_all_btn = QPushButton("Reset All")

        self.class_dropdown.setToolTip("Select the class to which labels will be assigned.")
        self.undo_btn.setToolTip("Undo the last label-to-class assignment operation.")
        self.clear_current_btn.setToolTip("Remove all label assignments associated with the currently selected class.")
        self.clear_all_btn.setToolTip("Remove all label-to-class mappings across all classes.")

        current_class_layout.addWidget(QLabel("Current Class:"))
        current_class_layout.addWidget(self.class_dropdown)
        current_class_layout.addWidget(self.undo_btn)
        
        current_class_layout.addWidget(self.clear_current_btn)
        current_class_layout.addWidget(self.clear_all_btn)
        
        self.layout().addLayout(current_class_layout)

        # --- Shortcut hint ---
        # shortcut_hint = QLabel(
        #     "Shortcuts: Ctrl+Click = assign  |  Alt+D = delete mapping  |  "
        #     "Alt+N = next class  |  Alt+B = prev class  "
        #     "(Alt+N / Alt+B disabled in Auto mode)"
        # )
        # shortcut_hint.setStyleSheet("color: gray; font-size: 10px;")
        # shortcut_hint.setWordWrap(True)
        # self.layout().addWidget(shortcut_hint)

        # --- Label Assign Summary ---
        self.mapping_summary = QListWidget()
        self.mapping_summary.setToolTip("Shows all label IDs and their assigned semantic classes.")
        self.layout().addWidget(QLabel("Current Mapping:"))
        self.layout().addWidget(self.mapping_summary)

        self.new_seg_button = QPushButton("Apply Mapping to Segmentation")
        self.new_seg_button.setStyleSheet("""QPushButton { font-weight: bold; background-color: #45a049;}""")
        self.new_seg_button.setToolTip("Create a new label image based on current label-to-class mapping.")
        self.new_seg_button.clicked.connect(self.apply_mapping_to_segmentation)
        self.layout().addWidget(self.new_seg_button)

        self.save_mapping_button = QPushButton("Export Mapping to JSON")
        self.save_mapping_button.setStyleSheet("""QPushButton { font-weight: bold; background-color: #45a049;}""")
        self.save_mapping_button.setToolTip("Save current label-to-class mapping to a JSON file.")
        self.save_mapping_button.clicked.connect(self.export_mapping_to_json)
        self.layout().addWidget(self.save_mapping_button)

        # --- Connect callbacks ---
        # For scheme controls
        self.load_scheme_btn.clicked.connect(self.load_scheme)
        self.save_scheme_btn.clicked.connect(self.save_scheme)
        self.add_class_btn.clicked.connect(self.add_class)
        self.remove_class_btn.clicked.connect(self.remove_class)
        self.clear_classes_btn.clicked.connect(self.clear_all_classes)
        
        # For class mapping
        self.undo_btn.clicked.connect(self.undo)
        self.clear_current_btn.clicked.connect(self.clear_current_class_mappings)
        self.clear_all_btn.clicked.connect(self.clear_all_mappings)

        self.viewer.layers.selection.events.active.connect(self.update_active_label_layer_binding)
        self.viewer.window.qt_viewer.canvas.events.key_press.connect(self.on_keypress)
        self.class_dropdown.currentIndexChanged.connect(self.sync_class_index)

    def sync_class_index(self, idx):
        self.current_class_index = idx

    def update_active_label_layer_binding(self, event):
        layer = event.value

        layer_changed = (self.label_layer != layer)
        if layer_changed and self.label2class:
            self.clear_all_mappings()
        # unbind from the last layer if it exists
        if self.label_layer and self.on_click in self.label_layer.mouse_drag_callbacks:
            self.label_layer.mouse_drag_callbacks.remove(self.on_click)
        self.label_layer = None
        if isinstance(layer, Labels):
            if self.on_click not in layer.mouse_drag_callbacks:
                layer.mouse_drag_callbacks.append(self.on_click)
                
            self.label_layer = layer
            self.layer_label.setText(f"Active Label Layer: {layer.name}")
            print(f"Bound to active label layer: {layer.name}")
        else:
            self.label_layer = None
            self.layer_label.setText("Active Label Layer: (none)")

    def load_scheme(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Scheme", filter="JSON files (*.json)")
        if not path:
            return
        with open(path, 'r') as f:
            self.scheme = json.load(f)
        self.update_scheme_ui()

    def save_scheme(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Scheme", filter="JSON files (*.json)")
        if not path:
            return
        with open(path, 'w') as f:
            json.dump(self.scheme, f, indent=4)

    def add_class(self):
        name = self.class_input.text().strip()
        if name and name not in self.scheme:
            self.scheme.append(name)
            self.update_scheme_ui()
            self.class_input.clear()

    def remove_class(self):
        name = self.class_input.text().strip()
        if name in self.scheme:
            # remove class and also drop any mappings for that class
            self.scheme.remove(name)
            labels_to_remove = [k for k, v in self.label2class.items() if v == name]
            for lbl in labels_to_remove:
                del self.label2class[lbl]
            self.update_scheme_ui()
            self.update_mapping_summary()
            self.class_input.clear()
        else:
            QMessageBox.information(self, "Info", f"Class '{name}' not found in the scheme.")

    def update_scheme_ui(self, update_dropdown=True, preserve_index=False):
        # Remember current index before rebuilding if needed
        saved_index = self.current_class_index if preserve_index else None

        self.scheme_list.clear()
        if update_dropdown:
            self.class_dropdown.blockSignals(True)
            self.class_dropdown.clear()
        for cls in self.scheme:
            self.scheme_list.addItem(f"{cls}: {self.get_labels_for_class(cls)}")
            if update_dropdown:
                self.class_dropdown.addItem(cls)
        if update_dropdown:
            if preserve_index and saved_index is not None:
                # Clamp to valid range in case scheme shrank
                clamped = min(saved_index, self.class_dropdown.count() - 1)
                clamped = max(clamped, 0)
                self.class_dropdown.setCurrentIndex(clamped)
                self.current_class_index = clamped
            self.class_dropdown.blockSignals(False)

    def get_labels_for_class(self, cls):
        return [k for k, v in self.label2class.items() if v == cls]

    def next_class(self):
        if not self.scheme:
            return
        self.current_class_index = (self.current_class_index + 1) % len(self.scheme)
        self.class_dropdown.blockSignals(True)
        self.class_dropdown.setCurrentIndex(self.current_class_index)
        self.class_dropdown.blockSignals(False)
        print(f"Switched to next class: {self.scheme[self.current_class_index]}")

    def prev_class(self):
        if not self.scheme:
            return
        self.current_class_index = (self.current_class_index - 1) % len(self.scheme)
        self.class_dropdown.blockSignals(True)
        self.class_dropdown.setCurrentIndex(self.current_class_index)
        self.class_dropdown.blockSignals(False)
        print(f"Switched to prev class: {self.scheme[self.current_class_index]}")

    def assign_label(self, label):
        cls = self.class_dropdown.currentText()
        if label == 0:
            return
        prev = self.label2class.get(label, None)
        self.undo_stack.append((label, prev))
        self.label2class[label] = cls
        self.update_scheme_ui(update_dropdown=False)
        self.update_mapping_summary()

        # Automatically advance to the next class if in auto mode
        if self.radio_auto.isChecked():
            self.next_class()

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

        # Not add mapping if no current class is selected / current class is empty
        current_cls = self.class_dropdown.currentText()
        if not current_cls:
            print("No current class selected; Ctrl+click ignored.")
            return

        self.assign_label(label)

    def on_keypress(self, event):
        # For now not hot key , as they can conflict with napari's built in shortcuts. 
        return
        # alt = "Alt" in event.modifiers

        # # Ctrl+D: delete mapping under cursor
        # if alt and event.key == 'd':
        #     if self.label_layer is None:
        #         return
        #     cursor = self.viewer.cursor.position
        #     coords = tuple(np.round(self.label_layer.world_to_data(cursor)).astype(int))
        #     if any(c < 0 or c >= s for c, s in zip(coords, self.label_layer.data.shape)):
        #         return
        #     label = int(self.label_layer.data[coords])
        #     prev = self.label2class.get(label, None)
        #     self.undo_stack.append((label, prev))
        #     if label in self.label2class:
        #         del self.label2class[label]
        #     self.update_mapping_summary()
        #     self.update_scheme_ui(update_dropdown=False)
        #     if self.radio_auto.isChecked():
        #         self.next_class()

        # # Ctrl+N: next class (Manual mode only)
        # elif alt and event.key == 'n':
        #     if self.radio_auto.isChecked():
        #         print("Alt+N disabled in Auto mode.")
        #         return
        #     self.next_class()

        # # Ctrl+B: previous class (Manual mode only)
        # elif alt and event.key == 'b':
        #     if self.radio_auto.isChecked():
        #         print("Alt+B disabled in Auto mode.")
        #         return
        #     self.prev_class()

    def update_mapping_summary(self):
        self.mapping_summary.clear()
        for label, cls in self.label2class.items():
            self.mapping_summary.addItem(f"Label {label} → {cls}")

    def undo(self):
        if not self.undo_stack:
            return
        label, prev = self.undo_stack.pop()
        if prev is None:
            self.label2class.pop(label, None)
        else:
            self.label2class[label] = prev
        self.update_mapping_summary()
        self.update_scheme_ui(preserve_index=True)
        
    def clear_current_class_mappings(self):
        current_cls = self.class_dropdown.currentText()
        labels_to_remove = [k for k, v in self.label2class.items() if v == current_cls]
        for lbl in labels_to_remove:
            del self.label2class[lbl]
        self.update_mapping_summary()
        self.update_scheme_ui(preserve_index=True)    
        print(f"Cleared mappings for class '{current_cls}'.")

    def clear_all_mappings(self):
        self.label2class.clear()
        self.update_mapping_summary()
        self.update_scheme_ui(preserve_index=True)
        print("Cleared all mappings.")

    def clear_all_classes(self):
        """Clear the entire scheme (all classes) and any associated mappings."""
        if not self.scheme:
            QMessageBox.information(self, "Info", "No classes to clear.")
            return

        resp = QMessageBox.question(
            self,
            "Confirm Clear All Classes",
            "This will remove all classes from the scheme and clear any associated mappings. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if resp != QMessageBox.Yes:
            return

        self.scheme.clear()
        self.label2class.clear()
        self.undo_stack.clear()
        self.update_mapping_summary()
        self.update_scheme_ui()
        print("Cleared all classes and mappings.")

    def apply_mapping_to_segmentation(self):
        if self.label_layer is None:
            QMessageBox.warning(self, "Error", "No label layer selected.")
            return

        if not self.label2class:
            QMessageBox.information(self, "Info", "No mappings available.")
            return

        # Generate mapping: class_name → new integer ID (based on scheme order)
        class_to_id = {cls: idx + 1 for idx, cls in enumerate(self.scheme) if any(v == cls for v in self.label2class.values())}
        label_map = {lbl: class_to_id[cls] for lbl, cls in self.label2class.items() if cls in class_to_id}

        # Create new data array
        old_data = self.label_layer.data
        new_data = np.zeros_like(old_data)

        for old_lbl, new_lbl in label_map.items():
            new_data[old_data == old_lbl] = new_lbl

        # Add new layer
        self.viewer.add_labels(new_data, name=f"{self.label_layer.name}_mapped")
        print("New mapped segmentation layer created.")

    def export_mapping_to_json(self):
        if not self.label2class:
            QMessageBox.information(self, "Info", "No mappings to export.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Mapping", filter="JSON files (*.json)")
        if not path:
            return

        export_data = {
            "scheme": self.scheme,
            "label2class": {str(k): v for k, v in self.label2class.items()}
        }

        with open(path, 'w') as f:
            json.dump(export_data, f, indent=4)

        print(f"Mapping exported to {path}.")
