"""Main widget for napari-SPROUT plugin."""

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QPushButton,
    QLabel, QHBoxLayout, QFileDialog
)
# from qtpy.QtCore import Qt
import napari
from napari.utils.notifications import show_info, show_error
import numpy as np
import tifffile

from .widgets.seed_widget import SeedGenerationWidget
from .widgets.grow_widget import SeedGrowthWidget
from .widgets.boundary_widget import BoundaryWidget
from .widgets.prompt_tab import PromptTab
from .utils.roi_widget import RoiTools

from .widgets.info_widget import LayerInfoWidget
from .widgets.edit_widget import QtLabelSelector
from .widgets.map_widget import LabelMapperWidget
from .widgets.tools_widget import SPROUTToolWidget

class SPROUTWidget(QWidget):
    """Main SPROUT widget containing all workflow steps."""
    
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        
        self._init_ui()
        self._connect_signals()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        
        # Set minimum size to ensure visibility
        self.setMinimumSize(500, 500)
        
        # Header
        header_layout = QHBoxLayout()
        title = QLabel("<h4>SPROUT Segmentation</h2>")
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        # Load image button
        # self.load_btn = QPushButton("Load Image")
        # header_layout.addWidget(self.load_btn)
        
        layout.addLayout(header_layout)
        
        # Info label
        # info_text = (
        #     "Semi-automated Parcellation of Region Outputs Using "
        #     "Thresholding and Transformation"
        # )
        # info_label = QLabel(info_text)
        # info_label.setWordWrap(True)
        # info_label.setStyleSheet("QLabel { color: #666; }")
        # layout.addWidget(info_label)
        
        # Tab widget for workflow steps
        self.tabs = QTabWidget()
        
        # Create widgets for each step
        self.seed_widget = SeedGenerationWidget(self.viewer)
        self.grow_widget = SeedGrowthWidget(self.viewer)
        self.prompt_tab = PromptTab(self.viewer)
        self.boundary_widget = BoundaryWidget(self.viewer)

        # ROI tools and the Boundary tab share the last tab (two sub-tabs)
        self.roi_tools = RoiTools(self.viewer)
        self.roi_boundary_tabs = QTabWidget()
        self.roi_boundary_tabs.addTab(self.roi_tools, "ROI")
        self.roi_boundary_tabs.addTab(self.boundary_widget, "Boundary")

        # Add tabs
        self.tabs.addTab(self.seed_widget, "1. Seeds")
        self.tabs.addTab(self.grow_widget, "2. Grow")
        self.tabs.addTab(self.prompt_tab, "3. Prompt (AI)")
        self.tabs.addTab(self.roi_boundary_tabs, "ROI && Boundary")   # "&&": a single "&" is a Qt mnemonic
        
        layout.addWidget(self.tabs)
        
        
        self.setLayout(layout)
    
    def _connect_signals(self):
        """Connect widget signals."""
        # self.load_btn.clicked.connect(self.load_image)
        self.seed_widget.seeds_generated.connect(self._on_seeds_generated)
        self.grow_widget.growth_completed.connect(self._on_growth_completed)
        self.boundary_widget.boundary_created.connect(self._on_boundary_created)

    def _on_boundary_created(self, name):
        """Select a newly created boundary layer in the Seeds and Grow tabs."""
        for w in (self.seed_widget, self.grow_widget):
            w.boundary_combo.setCurrentText(name)

    def load_image(self):
        """Load an image file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Image",
            "",
            "Image Files (*.tif *.tiff);;All Files (*)"
        )
        
        if filename:
            try:
                # Load image
                image = tifffile.imread(filename)
                
                # Add to viewer
                self.viewer.add_image(
                    image,
                    name=filename.split('/')[-1]
                )
                
                # Refresh layer lists
                self.seed_widget.refresh_layers()
                self.grow_widget.refresh_layers()
                
                show_info(f"Loaded image: {filename}")
                
            except Exception as e:
                show_error(f"Error loading image: {str(e)}")
    
    def _on_seeds_generated(self, seeds, sizes):
        """Handle seed generation completion."""
        # Switch to grow tab
        self.tabs.setCurrentIndex(1)
        
        # Refresh grow widget layers
        self.grow_widget.refresh_layers()
        
        show_info("Seeds generated! You can now edit them or proceed to growth.")
    
    def _on_growth_completed(self, result):
        """Handle growth completion."""
        show_info("Growth completed! You can save the result or continue editing.")


def make_sprout_widget(viewer: "napari.viewer.Viewer" = None):
    """Create the SPROUT widget.
    
    Parameters
    ----------
    viewer : napari.viewer.Viewer, optional
        The napari viewer instance. If not provided, will get current viewer.
        
    Returns
    -------
    widget : SPROUTWidget
        The SPROUT widget instance.
    """
    if viewer is None:
        # Get the current viewer from napari
        import napari
        viewer = napari.current_viewer()
        if viewer is None:
            raise RuntimeError("No napari viewer found")
    
    widget = SPROUTWidget(viewer)
    # no widget.show() in these factories: napari shows the widget when it docks it
    # (a show() before that flashes it as a separate window)
    return widget

def make_sprout_widget_edit(viewer: "napari.viewer.Viewer" = None):
    """Create the SPROUT widget for editing.
    
    Parameters
    ----------
    viewer : napari.viewer.Viewer, optional
        The napari viewer instance. If not provided, will get current viewer.
        
    Returns
    -------
    widget : SPROUTWidget
        The SPROUT widget instance.
    """
    if viewer is None:
        # Get the current viewer from napari
        import napari
        viewer = napari.current_viewer()
        if viewer is None:
            raise RuntimeError("No napari viewer found")
    
    widget = QtLabelSelector(viewer)
    return widget

def make_sprout_widget_info(viewer: "napari.viewer.Viewer" = None):
    """Create the SPROUT widget for information.
    
    Parameters
    ----------
    viewer : napari.viewer.Viewer, optional
        The napari viewer instance. If not provided, will get current viewer.
        
    Returns
    -------
    widget : SPROUTWidget
        The SPROUT widget instance.
    """
    if viewer is None:
        # Get the current viewer from napari
        import napari
        viewer = napari.current_viewer()
        if viewer is None:
            raise RuntimeError("No napari viewer found")
    
    widget = LayerInfoWidget(viewer)
    return widget


def make_sprout_widget_map(viewer: "napari.viewer.Viewer" = None):
    """Create the SPROUT widget for mapping.

    Parameters
    ----------
    viewer : napari.viewer.Viewer, optional
        The napari viewer instance. If not provided, will get current viewer.

    Returns
    -------
    widget : SPROUTWidget
        The SPROUT widget instance.
    """
    if viewer is None:
        # Get the current viewer from napari
        import napari
        viewer = napari.current_viewer()
        if viewer is None:
            raise RuntimeError("No napari viewer found")

    widget = LabelMapperWidget(viewer)
    return widget


def make_sprout_widget_tools(viewer: "napari.viewer.Viewer" = None):
    """Create the SPROUT widget for tools.

    Parameters
    ----------
    viewer : napari.viewer.Viewer, optional
        The napari viewer instance. If not provided, will get current viewer.

    Returns
    -------
    widget : SPROUTWidget
        The SPROUT widget instance.
    """
    if viewer is None:
        # Get the current viewer from napari
        import napari
        viewer = napari.current_viewer()
        if viewer is None:
            raise RuntimeError("No napari viewer found")

    widget = SPROUTToolWidget(viewer)
    return widget
