"""Widget for seed generation in SPROUT workflow."""

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox, QCheckBox,
    QSlider, QFormLayout, QMessageBox, QLineEdit, QFileDialog,
    QScrollArea
)
from qtpy.QtCore import Qt, Signal, QThread
import numpy as np
from typing import Optional
from napari.layers import Image, Labels
from napari.utils.notifications import show_info, show_error

from ..utils.util_widget import (create_output_folder_row, SeedOptionalParamGroupBox,
                                 MainSeedParamWidget, ThresholdWidget,apply_threshold_preview)

import yaml
import sys
import os
# get the current file's absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
    
from make_seeds import make_seeds
from make_adaptive_seed import make_adaptive_seed_thre, make_adaptive_seed_ero

class SeedWorker(QThread):
  
    progress = Signal(int)
    finished = Signal(dict) 
    error = Signal(str)
    
    def __init__(self, seed_mode, img, boundary, thresholds, upper_thresholds,
                 erosion_steps, segments, output_folder, num_threads,
                    footprints=None, base_name="seeds",
                    sort=True, no_split_max_iter=3,
                    min_size=5, min_split_ratio=0.01, min_split_total_ratio=0,
                    save_every_iter=True, init_segments=None,last_segments=None,
                    split_size_limit=(None, None), split_convex_hull_limit=(None, None)
                    
                    ):
        super().__init__()
        ## Parameters for both original and adaptive seed making
        self.seed_mode = seed_mode  # "Original", "Adaptive (Erosion)", "Adaptive (Thresholds)"
        self.img = img
        self.boundary = boundary
        self.thresholds = thresholds
        self.upper_thresholds = upper_thresholds
        self.erosion_steps = erosion_steps
        self.segments = segments
        self.output_folder = output_folder
        self.num_threads = num_threads
        self.footprints = footprints
        self.base_name = base_name
        
        ## parameters for adaptive seed making
        self.sort = sort
        self.no_split_max_iter = no_split_max_iter
        self.min_split_ratio = min_split_ratio
        self.min_split_total_ratio = min_split_total_ratio
        
        self.min_size = min_size
        self.save_every_iter = save_every_iter
        self.init_segments = init_segments
        self.last_segments = last_segments
        self.split_size_limit = split_size_limit
        self.split_convex_hull_limit = split_convex_hull_limit
        
        
    def run(self):
        try:
            if self.seed_mode not in ["Original", "Adaptive (Erosion)", "Adaptive (Thresholds)"]:
                raise ValueError("Invalid seed mode selected.")
            if self.seed_mode == "Original":
                # Call the original seed making function
                seeds_dict, _ = make_seeds(
                    img=self.img,
                    boundary=self.boundary,
                    thresholds=self.thresholds,
                    upper_thresholds=self.upper_thresholds,
                    erosion_steps=self.erosion_steps,
                    segments=self.segments,
                    output_folder=self.output_folder,
                    num_threads=self.num_threads,
                    footprints=self.footprints,
                    base_name=self.base_name,
                    
                    return_for_napari=True
                )
            elif self.seed_mode == "Adaptive (Erosion)":

                seeds_dict, _, _ = make_adaptive_seed_ero(
                    img=self.img,
                    boundary=self.boundary,
                    threshold=self.thresholds,
                    upper_threshold=self.upper_thresholds,
                    erosion_steps=self.erosion_steps,
                    segments=self.segments,
                    output_folder=self.output_folder,
                    num_threads=self.num_threads,
                    footprints=self.footprints,
                    
                    base_name=self.base_name,
                                        
                    sort=self.sort,
                    no_split_max_iter=self.no_split_max_iter,
                    min_split_ratio=self.min_split_ratio,
                    min_split_total_ratio=self.min_split_total_ratio,
                    min_size=self.min_size,
                    save_every_iter=self.save_every_iter,
                    init_segments=self.init_segments,
                    last_segments=self.last_segments,
                    split_size_limit=self.split_size_limit,
                    split_convex_hull_limit=self.split_convex_hull_limit,
                    
                    
                    return_for_napari=True
                )
            elif self.seed_mode == "Adaptive (Thresholds)":
                seeds_dict, _, _ = make_adaptive_seed_thre(
                    img=self.img,
                    boundary=self.boundary,
                    thresholds=self.thresholds,
                    upper_thresholds=self.upper_thresholds,
                    erosion_steps=self.erosion_steps,
                    segments=self.segments,
                    output_folder=self.output_folder,
                    num_threads=self.num_threads,
                    footprints=self.footprints,
                    
                    base_name=self.base_name,
                                        
                    sort=self.sort,
                    no_split_max_iter=self.no_split_max_iter,
                    min_split_ratio=self.min_split_ratio,
                    min_split_total_ratio=self.min_split_total_ratio,
                    min_size=self.min_size,
                    save_every_iter=self.save_every_iter,
                    init_segments=self.init_segments,
                    last_segments=self.last_segments,
                    split_size_limit=self.split_size_limit,
                    split_convex_hull_limit=self.split_convex_hull_limit,
                    
                    return_for_napari=True
                )
            # Emit finished signal with results
            self.finished.emit(seeds_dict)

        except Exception as e:
            self.error.emit(str(e))

class SeedGenerationWidget(QWidget):
    """Widget for interactive seed generation."""
    
    seeds_generated = Signal(np.ndarray, list)  # seeds array, sizes
    
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        
        self.worker = None
        
        self.previous_image_name = None
        
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
        image_group = QGroupBox("Input Options")
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
        # layout.addWidget(self.list_params_group)

        
        self.threshold_widget = ThresholdWidget(add_connected_components=True)
        self.threshold_widget.preview_requested.connect(self.preview_threshold)
        layout.addWidget(self.threshold_widget)
        
        # --- Main parameters widget ---
        self.main_param_widget = MainSeedParamWidget(title="Parameters",
                                                 image_combo=self.image_combo,viewer=self.viewer)
        layout.addWidget(self.main_param_widget)


        # checkbox for advanced adaptive seed options        
        self.show_advanced_checkbox = QCheckBox("Show Advanced Adaptive Seed Options")
        self.show_advanced_checkbox.stateChanged.connect(self._toggle_seed_params_visibility)

        # Advanced parameters group box
        self.advanced_params_box = SeedOptionalParamGroupBox()
        self.advanced_params_box.setVisible(False)

        # scroll area for advanced parameters
        # scroll = QScrollArea()
        # scroll.setWidgetResizable(True)
        # scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # scroll.setFrameShape(QScrollArea.NoFrame)

        # # container 
        # container = QWidget()
        # container_layout = QVBoxLayout()
        # container_layout.setSpacing(2)
        # container_layout.setContentsMargins(0, 0, 0, 0)
        # container_layout.addWidget(self.advanced_params_box)
        # container.setLayout(container_layout)
        # scroll.setWidget(container)

        advanced_block = QWidget()
        advanced_layout = QVBoxLayout()
        advanced_layout.setSpacing(4)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.addWidget(self.show_advanced_checkbox)
        advanced_layout.addWidget(self.advanced_params_box)
        advanced_block.setLayout(advanced_layout)

        layout.addWidget(advanced_block)

        # Import/Export yaml buttons
        button_layout = QHBoxLayout()
        self.import_btn = QPushButton("Import YAML")
        self.export_btn = QPushButton("Export YAML")
        button_layout.addWidget(self.import_btn)
        button_layout.addWidget(self.export_btn)
        layout.addLayout(button_layout)   

        # Output folder selection
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout()
        
        output_dir_layout, self.output_folder_line = create_output_folder_row()
        
        # output_dir_widget = QWidget()
        # output_dir_widget.setLayout(output_dir_layout)
        
        
        # save_every_iter (bool)
        self.save_every_iter_checkbox = QCheckBox("Save mid results")
        self.save_every_iter_checkbox.setChecked(False)
    
        
        output_layout.addLayout(output_dir_layout)
        # output_layout.addWidget(output_dir_widget)
        output_layout.addWidget(self.save_every_iter_checkbox)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Execute buttons
        btn_layout = QHBoxLayout()
        # Generate button
        self.generate_btn = QPushButton("Generate Seeds")
        self.generate_btn.setStyleSheet("""QPushButton { font-weight: bold; background-color: #45a049;}""")
        # layout.addWidget(self.generate_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                background-color: #d9534f;  
                color: white;
            }
            QPushButton:disabled {
                background-color: #aaa;
                color: #eee;
            }
        """)
        self.stop_btn.setEnabled(False)        
        

        btn_layout.addWidget(self.generate_btn)
        btn_layout.addWidget(self.stop_btn)
        # btn_layout.addWidget(self.save_btn)
        layout.addLayout(btn_layout)
        
        # Results info
        self.results_label = QLabel("No seeds generated yet")
        layout.addWidget(self.results_label)
        
        
        main_scroll_container = QWidget()
        main_scroll_container.setLayout(layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(main_scroll_container)
        
        final_layout = QVBoxLayout()
        final_layout.addWidget(scroll)
        self.setLayout(final_layout)
        
        # layout.addStretch()
        # self.setLayout(layout)

    def _toggle_seed_params_visibility(self, state: int):
        """Toggle visibility of advanced seed parameters."""
        self.advanced_params_box.setVisible(state == Qt.Checked)  

    def _connect_signals(self):
        """Connect widget signals."""
        self.refresh_btn.clicked.connect(self.refresh_layers)
        # TODO to be delete if thresholdwidget works
        # self.use_upper.toggled.connect(self.upper_threshold.setEnabled)
        # self.lower_threshold.valueChanged.connect(lambda v: self.lower_slider.setValue(int(v)))
        # self.lower_slider.valueChanged.connect(lambda v: self.lower_threshold.setValue(float(v)))
        # self.preview_threshold_btn.clicked.connect(self.preview_threshold)
        self.generate_btn.clicked.connect(self.generate_seeds)
        self.stop_btn.clicked.connect(self.stop_generation)
        
        self.image_combo.currentIndexChanged.connect(self._on_image_changed)

        self.use_upper_list.toggled.connect(self.upper_thresholds_list.setEnabled)
        
        self.import_btn.clicked.connect(self.import_yaml)
        self.export_btn.clicked.connect(self.export_yaml)
        
        # self.browse_folder_btn.clicked.connect(self._browse_output_folder)

    def _browse_output_folder(self):
        """Open a dialog to select an output folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder_edit.setText(folder)


    def refresh_layers(self):
        """Refresh the list of available layers."""
        current_image_name = self.image_combo.currentText()
        self.image_combo.blockSignals(True)  # prevent triggering img_changed temporarily
        
        self.image_combo.clear()
        self.boundary_combo.clear()
        self.boundary_combo.addItem("None")
        
        image_names = [layer.name for layer in self.viewer.layers if isinstance(layer, Image)]
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                self.image_combo.addItem(layer.name)
            elif isinstance(layer, Labels):
                self.boundary_combo.addItem(layer.name)
    
        # Restore image selection if it still exists
        if current_image_name in image_names:
            self.image_combo.setCurrentText(current_image_name)
        self.image_combo.blockSignals(False)

        # If current_text changed (e.g., setCurrentText fails), manually trigger img_changed
        if self.image_combo.currentText() != self.previous_image_name:
            self._on_image_changed(self.image_combo.currentText())
    
    
    def _on_image_changed(self , _):
        """Handle image selection change."""
        text = self.image_combo.currentText()
        if text and text != self.previous_image_name:
            self.previous_image_name = text 
            self.main_param_widget.clean_table()
            
            layer = self.viewer.layers[self.image_combo.currentText()]
            if isinstance(layer, Image):
                self.current_image = layer.data
                self.threshold_widget.set_range_by_dtype(self.current_image.dtype)
       

    
    def preview_threshold(self, lower=None, upper=None):
        if self.current_image is None:
            show_error("Please select an image first")
            return

        try:
            if lower is None:
                lower, upper = self.threshold_widget.get_thresholds()

            self.threshold_widget.run_preview_in_thread(self.current_image, callback=self._on_preview_result)


            # binary = self.threshold_widget.apply_preview(self.current_image)


            # # delete existing preview layer if it exists
            # if self.preview_layer and self.preview_layer in self.viewer.layers:
            #     self.viewer.layers.remove(self.preview_layer)
            #     self.preview_layer = None

            # # delete existing seeds layer if it exists
            # self.preview_layer = self.viewer.add_labels(
            #     binary.astype(np.uint8),
            #     name="Threshold Preview",
            #     opacity=0.5,
            # )


            # show_info(f"Preview updated: {np.sum(binary)} pixels selected")
        except Exception as e:
            show_error(f"Error in preview: {e}")
    
    def _on_preview_result(self, binary):
        """Handle the result of the threshold preview."""
        # delete existing preview layer if it exists
        if self.preview_layer and self.preview_layer in self.viewer.layers:
            self.viewer.layers.remove(self.preview_layer)
            self.preview_layer = None

        # delete existing seeds layer if it exists
        self.preview_layer = self.viewer.add_labels(
            binary.astype(np.uint8),
            name="Threshold Preview",
            opacity=0.5,
        )
        show_info(f"Preview updated: {np.sum(binary)} pixels selected")

        self.threshold_widget.preview_btn.setEnabled(True)  # Re-enable button after processing
    
        
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
        current_name = self.image_combo.currentText()
        
        try:
        

            
            # Get boundary if selected
            boundary = None
            if self.boundary_combo.currentText() != "None":
                boundary_layer = self.viewer.layers[self.boundary_combo.currentText()]
                if isinstance(boundary_layer, Labels):
                    boundary = boundary_layer.data

            output_folder = self.output_folder_line.text()
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
            
            self._set_ui_for_generation()
            
            main_params = self.main_param_widget.get_params()
            selected_method = main_params['seed_method']
            print(f"Generating seeds with method: {selected_method}")
            
            
            advance_params  = self.advanced_params_box.get_params()
            
            
            self.worker = SeedWorker(
                seed_mode=selected_method,
                img=self.current_image,
                boundary=boundary,
                
                thresholds=main_params['thresholds'],
                upper_thresholds=main_params['upper_thresholds'],
                erosion_steps=main_params['erosion_steps'],
                num_threads= main_params['num_threads'],
                
                segments=main_params['segments'],
                output_folder=output_folder,
                
                footprints=main_params['footprints'],
                base_name= current_name,
                
                sort=advance_params['sort'],
                no_split_max_iter=advance_params['no_split_max_iter'],
                min_size=advance_params['min_size'],
                min_split_ratio=advance_params['min_split_ratio'],
                min_split_total_ratio=advance_params['min_split_total_ratio'],
                save_every_iter=self.save_every_iter_checkbox.isChecked(),
                init_segments=None,  # Add if needed
                last_segments=None,  # Add if needed
                split_size_limit = advance_params['split_size_limit'],
                split_convex_hull_limit = advance_params['split_convex_hull_limit']
                
            )
            self.worker.finished.connect(self._on_seed_finished)
            self.worker.error.connect(self._on_seed_error)
            
            self.worker.start()
            # self.viewer.window.statusBar().showMessage(
            #     f"Generating seeds using {selected_method} method...")
            
            ##TODO, as we got worker ready, we comment below code
            ## To be delete
            
            # seeds_dict = {}
            # if selected_method == "Original_ignore":
            #     seeds_dict, _ = make_seeds(
            #         img=self.current_image,
            #         boundary=boundary,
            #         thresholds=thresholds,
            #         upper_thresholds=upper_thresholds,
            #         erosion_steps=erosion,
            #         segments=segments,
            #         output_folder=output_folder,
            #         num_threads=4,
            #         footprints=None,
            #         base_name="to_get_name",
            #         return_for_napari=True
            #     )
            # elif selected_method == "Adaptive (Erosion)_ignore":
            #     seeds_dict, _, _ = make_adaptive_seed_ero(
            #         img=self.current_image,
            #         boundary=boundary,
            #         threshold=lower,
            #         upper_threshold=upper,
            #         erosion_steps=erosion,
            #         segments=segments,
            #         output_folder=output_folder,
            #         num_threads=4,
            #         footprints=None,
            #         sort=True,
            #         no_split_max_iter=3,
            #         min_size=5,
            #         min_split_ratio=0.01,
            #         min_split_total_ratio=0,
            #         save_every_iter=True,
            #         init_segments=None,
            #         split_size_limit=(None, None),
            #         split_convex_hull_limit=(None, None),
            #         return_for_napari=True
            #     )
            # elif selected_method == "Adaptive (Thresholds)_ignore":
            #     seeds_dict, _, _ = make_adaptive_seed_thre(
            #         img=self.current_image,
            #         boundary=boundary,
            #         thresholds=thresholds,
            #         upper_thresholds=upper_thresholds,
            #         erosion_steps=erosion,
            #         segments=segments,
            #         output_folder=output_folder,
            #         num_threads=4,
            #         footprints=None,
            #         sort=True,
            #         no_split_max_iter=3,
            #         min_size=5,
            #         min_split_ratio=0.01,
            #         min_split_total_ratio=0,
            #         save_every_iter=True,
            #         init_segments=None,
            #         split_size_limit=(None, None),
            #         split_convex_hull_limit=(None, None),
            #         return_for_napari=True
            #     )

            # if not seeds_dict:
            #     show_error("Seed generation returned no seeds.")
            #     return

            # self.add_labels_layer(seeds_dict)
            
            # self.results_label.setText(f"Generated seeds using {selected_method}")
            
            # if seeds_dict:
            #     first_seed_name = list(seeds_dict.keys())[0]
            #     seeds = seeds_dict[first_seed_name]
            #     sizes = [np.sum(seeds == i) for i in np.unique(seeds) if i != 0]
            #     self.seeds_generated.emit(seeds, sizes)
            #     show_info(f"Successfully generated {len(sizes)} seeds using {selected_method}")

        except Exception as e:
            show_error(f"Error generating seeds: {str(e)}")
            self._reset_ui_after_generation()

    def _on_seed_finished(self, seeds_dict):
        """Handle seed generation completion."""
        # Remove any existing preview layer
        # if self.preview_layer is not None:
        #     self.viewer.layers.remove(self.preview_layer)
        #     self.preview_layer = None
        
        # Add the generated seeds to the viewer
        self.add_labels_layer(seeds_dict)
        
        # Update results label
        self.results_label.setText(f"Seeds generated: {len(seeds_dict)} layers")
        
        show_info("Seeds generated successfully!")

        
        self._reset_ui_after_generation()
    
    def _on_seed_error(self, error_message):
        """Handle errors during seed generation."""
        show_error(f"Error generating seeds: {error_message}")
        
        # Reset UI state
        self._reset_ui_after_generation()
        
    def _reset_ui_after_generation(self):
        """Reset UI after growth."""
        self.generate_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        # self.progress_bar.setVisible(False)
    
    def _set_ui_for_generation(self):
        """Set UI state for ongoing seed generation."""
        self.generate_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        # self.progress_bar.setVisible(True)
        # self.progress_bar.setValue(0)  # Reset progress bar
        # self.results_label.setText("Generating seeds...")  # Update status message
    
    def stop_generation(self):
        """Stop the current seed generation process."""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self._reset_ui_after_generation()
            # show_info("Seed generation stopped.")
            # self.viewer.window.statusBar().showMessage("Seed generation stopped.", 1000)
            

    # TODO to implement import/export yaml for seed parameters
    # This is a placeholder for the import/export functionality.
    def import_yaml(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open YAML File", "", "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, "r") as f:
                    data = yaml.safe_load(f)
                print("[Imported YAML]:", data)
            except Exception as e:
                QMessageBox.critical(self, "Import Error", str(e))

    def export_yaml(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save YAML File", "", "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if file_path:
            try:
                main_params = self.main_param_widget.get_params()
                advanced_params = self.advanced_params_box.get_params()
                print("TODO export yaml with main and advanced params")
                print("[Exported YAML]:", main_params, advanced_params)
                with open(file_path, "w") as f:
                    yaml.dump({"a":"test"}, f, sort_keys=False)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))