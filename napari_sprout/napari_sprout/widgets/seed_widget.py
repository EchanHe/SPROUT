"""Widget for seed generation in SPROUT workflow."""

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox, QCheckBox,
    QSlider, QFormLayout, QMessageBox, QLineEdit, QFileDialog
)
from qtpy.QtCore import Qt, Signal
import numpy as np
from typing import Optional
from napari.layers import Image, Labels
from napari.utils.notifications import show_info, show_error

from ..utils.sprout_bridge import SPROUTBridge


import sys
import os
# get the current file's absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
    
from make_seeds import make_seeds
from make_adaptive_seed import make_adaptive_seed_thre, make_adaptive_seed_ero


class SeedGenerationWidget(QWidget):
    """Widget for interactive seed generation."""
    
    seeds_generated = Signal(np.ndarray, list)  # seeds array, sizes
    
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.bridge = SPROUTBridge()
        self.current_image = None
        self.current_boundary = None
        self.preview_layer = None
        self.seeds_layer = None
        
        self._init_ui()
        self._connect_signals()
        self._on_seed_method_changed("Original") # Set initial state

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        
        # Image selection
        image_group = QGroupBox("Image Selection")
        image_layout = QFormLayout()
        
        self.image_combo = QComboBox()
        self.boundary_combo = QComboBox()
        self.boundary_combo.addItem("None")
        
        image_layout.addRow("Input Image:", self.image_combo)
        image_layout.addRow("Boundary Mask (optional):", self.boundary_combo)
        
        self.refresh_btn = QPushButton("Refresh Layers")
        image_layout.addRow(self.refresh_btn)
        
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)

        # Seed method selection
        seed_method_group = QGroupBox("Seed Method")
        seed_method_layout = QFormLayout()
        self.seed_method_combo = QComboBox()
        self.seed_method_combo.addItems(["Original", "Adaptive (Erosion)", "Adaptive (Thresholds)"])
        seed_method_layout.addRow("Method:", self.seed_method_combo)
        seed_method_group.setLayout(seed_method_layout)
        layout.addWidget(seed_method_group)
        
        # --- Parameters for Original and Adaptive (Thresholds) methods ---
        self.list_params_group = QGroupBox("Threshold List Parameters")
        self.list_params_group.setFocusPolicy(Qt.NoFocus) # Allow children to get focus
        list_params_layout = QFormLayout()
        self.lower_thresholds_list = QLineEdit("130, 140, 150")
        self.upper_thresholds_list = QLineEdit("255, 255, 255")

        # Style to make QLineEdit look editable, text visible, and ensure cursor blinks.
        style = """
            QLineEdit {
                background-color: white;
                border: 1px solid #999;
                color: black;
                padding: 2px;
            }
            QLineEdit:focus {
                border: 1px solid #0078d7; /* Blue border on focus */
            }
        """
        self.lower_thresholds_list.setStyleSheet(style)
        self.upper_thresholds_list.setStyleSheet(style)

        # Add tooltips for clarity
        self.lower_thresholds_list.setToolTip("Enter comma-separated integer values for lower thresholds.")
        self.upper_thresholds_list.setToolTip("Enter comma-separated integer values for upper thresholds.")

        self.use_upper_list = QCheckBox("Use upper thresholds list")
        self.use_upper_list.setChecked(True)
        list_params_layout.addRow("Lower Thresholds (comma-separated):", self.lower_thresholds_list)
        list_params_layout.addRow(self.use_upper_list)
        list_params_layout.addRow("Upper Thresholds (comma-separated):", self.upper_thresholds_list)
        self.list_params_group.setLayout(list_params_layout)
        layout.addWidget(self.list_params_group)

        # --- Parameters for Adaptive (Erosion) method ---
        self.single_param_group = QGroupBox("Threshold Parameters")
        threshold_layout = QFormLayout()
        
        # Lower threshold with slider
        self.lower_threshold = QDoubleSpinBox()
        self.lower_threshold.setRange(0, 65535)
        self.lower_threshold.setDecimals(1)
        self.lower_threshold.setValue(150)
        
        self.lower_slider = QSlider(Qt.Horizontal)
        self.lower_slider.setRange(0, 65535)
        self.lower_slider.setValue(150)
        
        lower_layout = QHBoxLayout()
        lower_layout.addWidget(self.lower_threshold)
        lower_layout.addWidget(self.lower_slider)
        
        # Upper threshold
        self.use_upper = QCheckBox("Use upper threshold")
        self.upper_threshold = QDoubleSpinBox()
        self.upper_threshold.setRange(0, 65535)
        self.upper_threshold.setDecimals(1)
        self.upper_threshold.setValue(255)
        self.upper_threshold.setEnabled(False)
        
        threshold_layout.addRow("Lower Threshold:", lower_layout)
        threshold_layout.addRow(self.use_upper)
        threshold_layout.addRow("Upper Threshold:", self.upper_threshold)
        
        self.preview_threshold_btn = QPushButton("Preview Threshold")
        threshold_layout.addRow(self.preview_threshold_btn)
        
        self.single_param_group.setLayout(threshold_layout)
        layout.addWidget(self.single_param_group)
        
        # Morphological parameters
        morph_group = QGroupBox("Morphological Parameters")
        morph_layout = QFormLayout()
        
        self.erosion_iter = QSpinBox()
        self.erosion_iter.setRange(0, 50)
        self.erosion_iter.setValue(3)
        
        self.footprint_combo = QComboBox()
        self.footprint_combo.addItems(self.bridge.get_footprint_options())
        
        self.segments_spin = QSpinBox()
        self.segments_spin.setRange(1, 1000)
        self.segments_spin.setValue(10)
        
        morph_layout.addRow("Erosion Iterations:", self.erosion_iter)
        morph_layout.addRow("Footprint Shape:", self.footprint_combo)
        morph_layout.addRow("Max Segments:", self.segments_spin)
        
        morph_group.setLayout(morph_layout)
        layout.addWidget(morph_group)

        # Output folder selection
        output_group = QGroupBox("Output Settings")
        output_layout = QHBoxLayout()
        self.output_folder_edit = QLineEdit("napari_temp")
        self.output_folder_edit.setToolTip("Select the folder to save seed files.")
        self.browse_folder_btn = QPushButton("Browse...")
        output_layout.addWidget(QLabel("Output Folder:"))
        output_layout.addWidget(self.output_folder_edit)
        output_layout.addWidget(self.browse_folder_btn)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Generate button
        self.generate_btn = QPushButton("Generate Seeds")
        self.generate_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        layout.addWidget(self.generate_btn)
        
        # Results info
        self.results_label = QLabel("No seeds generated yet")
        layout.addWidget(self.results_label)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def _connect_signals(self):
        """Connect widget signals."""
        self.refresh_btn.clicked.connect(self.refresh_layers)
        self.use_upper.toggled.connect(self.upper_threshold.setEnabled)
        self.lower_threshold.valueChanged.connect(lambda v: self.lower_slider.setValue(int(v)))
        self.lower_slider.valueChanged.connect(lambda v: self.lower_threshold.setValue(float(v)))
        self.preview_threshold_btn.clicked.connect(self.preview_threshold)
        self.generate_btn.clicked.connect(self.generate_seeds)
        self.image_combo.currentIndexChanged.connect(self._on_image_changed)
        self.seed_method_combo.currentTextChanged.connect(self._on_seed_method_changed)
        self.use_upper_list.toggled.connect(self.upper_thresholds_list.setEnabled)
        self.browse_folder_btn.clicked.connect(self._browse_output_folder)

    def _browse_output_folder(self):
        """Open a dialog to select an output folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder_edit.setText(folder)

    def _on_seed_method_changed(self, method: str):
        """Show/hide parameter sections based on the selected method."""
        if method in ["Original", "Adaptive (Thresholds)"]:
            self.list_params_group.setVisible(True)
            self.single_param_group.setVisible(False)
        elif method == "Adaptive (Erosion)":
            self.list_params_group.setVisible(False)
            self.single_param_group.setVisible(True)

    def refresh_layers(self):
        """Refresh the list of available layers."""
        self.image_combo.clear()
        self.boundary_combo.clear()
        self.boundary_combo.addItem("None")
        
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                self.image_combo.addItem(layer.name)
            elif isinstance(layer, Labels):
                self.boundary_combo.addItem(layer.name)
    
    def _on_image_changed(self):
        """Handle image selection change."""
        if self.image_combo.currentText():
            layer = self.viewer.layers[self.image_combo.currentText()]
            if isinstance(layer, Image):
                self.current_image = layer.data
                # Update threshold range based on image
                img_min, img_max = np.min(self.current_image), np.max(self.current_image)
                self.lower_threshold.setRange(img_min, img_max)
                self.upper_threshold.setRange(img_min, img_max)
                self.lower_slider.setRange(int(img_min), int(img_max))
                self.lower_threshold.setValue(img_min + (img_max - img_min) * 0.1)
                self.upper_threshold.setValue(img_min + (img_max - img_min) * 0.5)
    
    def preview_threshold(self):
        """Preview the threshold result."""
        if self.current_image is None:
            show_error("Please select an image first")
            return
        
        try:
            # Get threshold values
            lower = self.lower_threshold.value()
            upper = self.upper_threshold.value() if self.use_upper.isChecked() else None
            
            # Apply threshold
            binary = self.bridge.apply_threshold_preview(self.current_image, lower, upper)
            
            # Update or create preview layer
            if self.preview_layer is not None and self.preview_layer in self.viewer.layers:
                self.preview_layer.data = binary
            else:
                self.preview_layer = self.viewer.add_labels(
                    binary.astype(np.uint8),
                    name="Threshold Preview",
                    opacity=0.5
                )
            
            show_info(f"Threshold preview updated: {np.sum(binary)} pixels selected")
            
        except Exception as e:
            show_error(f"Error in threshold preview: {str(e)}")
    
    def add_labels_layer(self, seeds_dict):
        for seed_name, seed in seeds_dict.items():
            print(f"Adding seed layer: {seed_name}")
            self.seeds_layer = self.viewer.add_labels(
                seed,
                name=seed_name
            )
    
    def generate_seeds(self):
        """Generate seeds with current parameters."""
        if self.current_image is None:
            show_error("Please select an image first")
            return
        
        try:
            # Get parameters based on selected method
            selected_method = self.seed_method_combo.currentText()
            
            lower = None
            upper = None
            thresholds = []
            upper_thresholds = None

            if selected_method == "Adaptive (Erosion)":
                lower = int(self.lower_threshold.value())
                if self.use_upper.isChecked():
                    upper = int(self.upper_threshold.value())
            else: # Original or Adaptive (Thresholds)
                try:
                    thresholds = [int(x.strip()) for x in self.lower_thresholds_list.text().split(',') if x.strip()]
                    if self.use_upper_list.isChecked():
                        upper_thresholds = [int(x.strip()) for x in self.upper_thresholds_list.text().split(',') if x.strip()]
                        if len(thresholds) != len(upper_thresholds):
                            show_error("Lower and upper threshold lists must have the same number of elements.")
                            return
                except ValueError:
                    show_error("Thresholds must be comma-separated integers.")
                    return

            erosion = self.erosion_iter.value()
            footprint = self.footprint_combo.currentText()
            segments = self.segments_spin.value()
            
            # Get boundary if selected
            boundary = None
            if self.boundary_combo.currentText() != "None":
                boundary_layer = self.viewer.layers[self.boundary_combo.currentText()]
                if isinstance(boundary_layer, Labels):
                    boundary = boundary_layer.data

            output_folder = self.output_folder_edit.text()
            if not output_folder:
                show_error("Please specify an output folder. Default is 'napari_temp'")
                output_folder = 'napari_temp'
            
            # TODO: @ioannouE
            # I have added interfaces for seed making they are
            # make_seeds that does orignal seed making
            # make_adaptive_seed_thre that does adaptive seed on a list of thresholds/ upper thresholds
            # make_adaptive_seed_ero that does adaptive seed making with erosions
            # So probably make a selection widget that allows user to select which method to use
            
            # They return seeds_dict for {name: seed}, and will be added to labels
            # NOTE: maybe add UI for selecting the output folder
            # NOTE: need to add list of thresholds for adaptive seed making
            
            seeds_dict = {}
            
            print(f"Generating seeds with method: {selected_method}")

            if selected_method == "Original":
                seeds_dict, _ = make_seeds(
                    img=self.current_image,
                    boundary=boundary,
                    thresholds=thresholds,
                    upper_thresholds=upper_thresholds,
                    erosion_steps=erosion,
                    segments=segments,
                    output_folder=output_folder,
                    num_threads=4,
                    footprints=None,
                    base_name="to_get_name",
                    return_for_napari=True
                )
            elif selected_method == "Adaptive (Erosion)":
                seeds_dict, _, _ = make_adaptive_seed_ero(
                    img=self.current_image,
                    boundary=boundary,
                    threshold=lower,
                    upper_threshold=upper,
                    erosion_steps=erosion,
                    segments=segments,
                    output_folder=output_folder,
                    num_threads=4,
                    footprints=None,
                    sort=True,
                    no_split_max_iter=3,
                    min_size=5,
                    min_split_ratio=0.01,
                    min_split_total_ratio=0,
                    save_every_iter=True,
                    init_segments=None,
                    split_size_limit=(None, None),
                    split_convex_hull_limit=(None, None),
                    return_for_napari=True
                )
            elif selected_method == "Adaptive (Thresholds)":
                seeds_dict, _, _ = make_adaptive_seed_thre(
                    img=self.current_image,
                    boundary=boundary,
                    thresholds=thresholds,
                    upper_thresholds=upper_thresholds,
                    erosion_steps=erosion,
                    segments=segments,
                    output_folder=output_folder,
                    num_threads=4,
                    footprints=None,
                    sort=True,
                    no_split_max_iter=3,
                    min_size=5,
                    min_split_ratio=0.01,
                    min_split_total_ratio=0,
                    save_every_iter=True,
                    init_segments=None,
                    split_size_limit=(None, None),
                    split_convex_hull_limit=(None, None),
                    return_for_napari=True
                )

            if not seeds_dict:
                show_error("Seed generation returned no seeds.")
                return

            self.add_labels_layer(seeds_dict)
            
            self.results_label.setText(f"Generated seeds using {selected_method}")
            
            if seeds_dict:
                first_seed_name = list(seeds_dict.keys())[0]
                seeds = seeds_dict[first_seed_name]
                sizes = [np.sum(seeds == i) for i in np.unique(seeds) if i != 0]
                self.seeds_generated.emit(seeds, sizes)
                show_info(f"Successfully generated {len(sizes)} seeds using {selected_method}")

        except Exception as e:
            show_error(f"Error generating seeds: {str(e)}")
