"""Widget for seed generation in SPROUT workflow."""

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox, QCheckBox,
    QSlider, QFormLayout, QMessageBox
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
        
        # Threshold parameters
        threshold_group = QGroupBox("Threshold Parameters")
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
        
        threshold_group.setLayout(threshold_layout)
        layout.addWidget(threshold_group)
        
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
            # Get parameters
            lower = self.lower_threshold.value()
            upper = self.upper_threshold.value() if self.use_upper.isChecked() else None

            # Cast to int if not None
            if lower is not None:
                lower = int(lower)
            if upper is not None:
                upper = int(upper)
            erosion = self.erosion_iter.value()
            footprint = self.footprint_combo.currentText()
            segments = self.segments_spin.value()
            
            # Get boundary if selected
            boundary = None
            if self.boundary_combo.currentText() != "None":
                boundary_layer = self.viewer.layers[self.boundary_combo.currentText()]
                if isinstance(boundary_layer, Labels):
                    boundary = boundary_layer.data

            
            # TODO: @ioannouE
            # I have added interfaces for seed making they are
            # make_seeds that does orignal seed making
            # make_adaptive_seed_thre that does adaptive seed on a list of thresholds/ upper thresholds
            # make_adaptive_seed_ero that does adaptive seed making with erosions
            # So probably make a selection widget that allows user to select which method to use
            
            # They return seeds_dict for {name: seed}, and will be added to labels
            
            print(f"Generating seeds with parameters")
            # seeds_dict , _ = make_seeds(
            #     img = self.current_image,
            #     boundary = boundary,
                
            #     thresholds = lower,
            #     upper_thresholds = upper,
            #     ero_iters = erosion,
            #     segments = segments,
                
            #     # TODO add widget that has a default output folder, also allow user to select output folder
            #     output_folder = 'napari_temp',
            #      # TODO add default, also allow user change
            #     num_threads = 4,
            #     # TODO add a field for None as default, but also allow user input
            #     # it can be either a string or a list of string that matches the number of erosions
            #     footprints = None,
            #     # TODO get the name of current image too
            #     base_name = "to_get_name", 

            #     # fixed parameters
            #     is_napari=True
            #     )

            # self.add_labels_layer(seeds_dict)
            
            # @ioannouE, using make_adaptive_seed_ero or make_adaptive_seed_thre
            # is based on the threshold, if threshold is a single value, use make_adaptive_seed_ero
            # if threshold is a list, use make_adaptive_seed_thre, in make_adaptive_seed_thre is thresholds and upper_thresholds
            # See line 844-857 in make_adaptive_seed.py
            seeds_dict ,_ , _ = make_adaptive_seed_ero(
                img = self.current_image,
                boundary = boundary,  
                   
                threshold =lower,
                upper_threshold= upper,
                
                ero_iters = erosion, 
                segments= segments,        
                
                
                # TODO add qt fields for following parameters
                # path, field that has a default output folder, also allow user to select output folder
                output_folder = 'napari_temp',             
                # int, has default value, say 1, use can change
                num_threads = 4,
                
                footprints = None,
                
                # TODO fields for adadptive seed, 
                # maybe put them into a group, and only reveal when adaptive seed is selected
                sort = True,
                no_split_limit =3,
                min_size=5,
                min_split_prop = 0.01,
                min_split_sum_prop = 0,
                save_every_iter = True,
                save_merged_every_iter = True,
                # list of int, default is none
                init_segments = None,
                # tuple of int/None, default is (None,None)                
                split_size_limit = (None,None),
                # tuple of int/None, default is (None,None)   
                split_convex_hull_limit = (None, None),    
                
                # fixed parameters
                is_napari=True
                       
                )

            seeds_dict ,_ , _ = make_adaptive_seed_thre(
                img = self.current_image,
                boundary = boundary,  
                   
                # TODO 
                # Currently using default values, as this function require a list than just an int
                thresholds =[130,140,150],
                # Similar for upper_thresholds
                upper_thresholds= None,
                
                ero_iters = erosion, 
                segments= segments, 
            
            
                # TODO add qt fields for following parameters
                # path, field that has a default output folder, also allow user to select output folder
                output_folder = 'napari_temp',             
                # int, has default value, say 1, use can change
                num_threads = 4,
                footprints = None,
                
                
                # TODO fields for adadptive seed, 
                # maybe put them into a group, and only reveal when adaptive seed is selected

                sort = True,
                no_split_limit =3,
                min_size=5,
                min_split_prop = 0.01,
                min_split_sum_prop = 0,
                save_every_iter = True,
                save_merged_every_iter = True,
                init_segments = None,
                split_size_limit = (None,None),
                split_convex_hull_limit = (None, None),      
                
                # fixed parameters
                is_napari=True,       
                )

            
            self.add_labels_layer(seeds_dict)

            
            # @ioannouE I kept the orignal interface for comparing for now.
            
            # Generate seeds
            seeds, sizes = self.bridge.generate_seeds(
                self.current_image,
                lower,
                segments,
                erosion,
                footprint,
                upper,
                boundary
            )
            
            # Add seeds to viewer
            if self.seeds_layer is not None and self.seeds_layer in self.viewer.layers:
            #     self.seeds_layer.data = seeds
            # else:
                self.seeds_layer = self.viewer.add_labels(
                    seeds,
                    name="Generated Seeds"
                )
            
            # Update results
            self.results_label.setText(
                f"Generated {len(sizes)} seeds\n"
                f"Sizes: {sizes[:5]}{'...' if len(sizes) > 5 else ''}"
            )
            
            # Emit signal
            self.seeds_generated.emit(seeds, sizes)
            
            show_info(f"Successfully generated {len(sizes)} seeds")
            
        except Exception as e:
            show_error(f"Error generating seeds: {str(e)}")
