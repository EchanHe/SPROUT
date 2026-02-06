import os
import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QSpinBox,
    QHBoxLayout, QFormLayout, QMessageBox, QLineEdit, QFileDialog
)
from napari.layers import Image, Labels, Points
from napari.utils.notifications import show_info
from sprout_core.sprout_prompt_core import extract_slices_and_prompts, load_prompts_as_points_layers

def create_output_folder_row(default_folder=None):
    layout = QHBoxLayout()
    label = QLabel("Output Folder:")
    folder_edit = QLineEdit()
    browse_btn = QPushButton("Browse")

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

class SproutPromptWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setWindowTitle("SPROUTPROMPT")
        main_layout = QVBoxLayout()
        main_layout.addWidget(QLabel("<b>SPROUTPROMPT</b>"))

        # --- Layer selection UI ---
        form = QFormLayout()
        self.img_layer_combo = QComboBox()
        self.seg_layer_combo = QComboBox()
        form.addRow("Input image layer:", self.img_layer_combo)
        form.addRow("Segmentation (label) layer:", self.seg_layer_combo)
        self.refresh_layers()
        refresh_btn = QPushButton("Refresh layer list")
        refresh_btn.clicked.connect(self.refresh_layers)
        main_layout.addLayout(form)
        main_layout.addWidget(refresh_btn)

        # --- Output folder selection ---
        output_row, self.output_folder_edit = create_output_folder_row()
        main_layout.addLayout(output_row)

        # --- Parameter UI ---
        self.axis_combo = QComboBox()
        self.axis_combo.addItems(["Z", "Y", "X"])
        self.n_points_spin = QSpinBox()
        self.n_points_spin.setRange(1, 100)
        self.n_points_spin.setValue(3)
        self.prompt_type_combo = QComboBox()
        self.prompt_type_combo.addItems(["point", "bbox"])
        self.sample_method_combo = QComboBox()
        self.sample_method_combo.addItems(["random", "kmeans", "center_edge", "skeleton"])
        param_form = QFormLayout()
        param_form.addRow("Slicing axis:", self.axis_combo)
        param_form.addRow("Points per class:", self.n_points_spin)
        param_form.addRow("Prompt type:", self.prompt_type_combo)
        param_form.addRow("Sample method:", self.sample_method_combo)
        main_layout.addLayout(param_form)

        # --- Action button ---
        self.run_btn = QPushButton("Generate prompts")
        self.run_btn.clicked.connect(self.on_run)
        main_layout.addWidget(self.run_btn)

        # --- Print Points Info Button ---
        self.points_layer_combo = QComboBox()
        main_layout.addWidget(QLabel("Select points layer to print:"))
        main_layout.addWidget(self.points_layer_combo)
        self.refresh_points_layers()
        refresh_points_btn = QPushButton("Refresh points layer list")
        refresh_points_btn.clicked.connect(self.refresh_points_layers)
        main_layout.addWidget(refresh_points_btn)
        self.print_points_btn = QPushButton("Print point info to console")
        self.print_points_btn.clicked.connect(self.on_print_points)
        main_layout.addWidget(self.print_points_btn)
        self.setLayout(main_layout)

    def refresh_layers(self):
        self.img_layer_combo.clear()
        self.seg_layer_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                self.img_layer_combo.addItem(layer.name)
            if isinstance(layer, Labels):
                self.seg_layer_combo.addItem(layer.name)
        if self.img_layer_combo.count() == 0:
            self.img_layer_combo.addItem("No image layer")
        if self.seg_layer_combo.count() == 0:
            self.seg_layer_combo.addItem("No label layer")
        

    def refresh_points_layers(self):
        self.points_layer_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, Points):
                self.points_layer_combo.addItem(layer.name)
        if self.points_layer_combo.count() == 0:
            self.points_layer_combo.addItem("No points layer")

    def on_run(self):
        img_name = self.img_layer_combo.currentText()
        seg_name = self.seg_layer_combo.currentText()
        if "No image layer" in img_name or "No label layer" in seg_name:
            QMessageBox.warning(self, "Missing layers", "Please select valid image and label layers.")
            return
        img_layer = self.viewer.layers[img_name]
        seg_layer = self.viewer.layers[seg_name]
        axis = self.axis_combo.currentText()
        n_points = self.n_points_spin.value()
        prompt_type = self.prompt_type_combo.currentText()
        sample_method = self.sample_method_combo.currentText()

        # --- Collect output folder and define subfolders
        out_base = self.output_folder_edit.text()
        out_prompts = os.path.join(out_base, "prompts")
        out_imgs = os.path.join(out_base, "imgs")
        
        # if folders exist remove all files inside
        if os.path.exists(out_prompts):
            for f in os.listdir(out_prompts):
                os.remove(os.path.join(out_prompts, f))
        if os.path.exists(out_imgs):
            for f in os.listdir(out_imgs):
                os.remove(os.path.join(out_imgs, f))
        # create folders
        os.makedirs(out_prompts, exist_ok=True)
        os.makedirs(out_imgs, exist_ok=True)
        
        

        try:
            result = extract_slices_and_prompts(
                img=img_layer.data,
                seg=seg_layer.data,
                axis=axis,
                n_points_per_class=n_points,
                prompt_type=prompt_type,
                sample_method=sample_method,
                per_slice_2d_input=False,
                output_prompt_dir=out_prompts,
                output_img_dir=out_imgs,
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prompt extraction failed:\n{str(e)}")
            return

        layers_dict = load_prompts_as_points_layers(result['prompts_dir'], axis=axis)
        for clsname, d in layers_dict.items():
            self.viewer.add_points(
                d["coords"],  # (N, 3) 
                properties={"label": d["labels"], "name": d["names"]},
                face_color="label",
                face_color_cycle={1: "red", 0: "blue"},
                name=f"points_{clsname}",
                size=2
            )

    def on_print_points(self):
        layer_name = self.points_layer_combo.currentText()
        if "No points layer" in layer_name:
            QMessageBox.information(self, "No layer", "No points layer selected.")
            return
        pts_layer = self.viewer.layers[layer_name]
        coords = pts_layer.data
        props = pts_layer.properties
        print(f"Points in layer '{layer_name}':")
        for i, pt in enumerate(coords):
            info = f"Point {i}: coord={pt}"
            for key in ("label", "name"):
                if key in props:
                    info += f", {key}={props[key][i]}"
            print(info)

def napari_experimental_provide_dock_widget():
    return SproutPromptWidget