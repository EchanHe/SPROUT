"""Main widget for napari-SPROUT plugin."""

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QPushButton,
    QLabel, QHBoxLayout, QFileDialog
)
# from qtpy.QtCore import Qt
import napari
from napari.utils.notifications import show_info, show_error
import numpy as np

from .widgets.seed_widget import SeedGenerationWidget
from .widgets.grow_widget import SeedGrowthWidget
from .utils.sprout_bridge import SPROUTBridge

from .widgets.info_widget import LayerInfoWidget
from .widgets.edit_widget import QtLabelSelector
from .widgets.map_widget import LabelMapperWidget
from .widgets.tools_widget import SPROUTToolWidget
from .widgets.prompt_widget import SproutPromptWidget

class SPROUTWidget(QWidget):
    """Main SPROUT widget containing all workflow steps."""
    
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.bridge = SPROUTBridge()
        
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
        
        # Add tabs
        self.tabs.addTab(self.seed_widget, "1. Generate Seeds")
        self.tabs.addTab(self.grow_widget, "2. Grow Seeds")
        
        layout.addWidget(self.tabs)
        
        # Help text
        help_text = QLabel(
            "<b>Workflow:</b> 1) Generate seeds → 2) Review/edit seeds → 3) Grow regions"
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        
        self.setLayout(layout)
    
    def _connect_signals(self):
        """Connect widget signals."""
        # self.load_btn.clicked.connect(self.load_image)
        self.seed_widget.seeds_generated.connect(self._on_seeds_generated)
        self.grow_widget.growth_completed.connect(self._on_growth_completed)
    
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
                image = self.bridge.load_image(filename)
                
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
    # Ensure widget is visible
    widget.show()
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
    # Ensure widget is visible
    widget.show()
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
    # Ensure widget is visible
    widget.show()
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
    # Ensure widget is visible
    widget.show()
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
    # Ensure widget is visible
    widget.show()
    return widget

def make_sprout_widget_prompt(viewer: "napari.viewer.Viewer" = None):
    """Create the SPROUT PROMPT extraction widget.

    Parameters
    ----------
    viewer : napari.viewer.Viewer, optional
        The napari viewer instance. If not provided, will get current viewer.
    Returns
    -------
    widget : SproutPromptWidget
        The SPROUT PROMPT extraction widget instance.
    """
    if viewer is None:
        # Get the current viewer from napari
        import napari
        viewer = napari.current_viewer()
        if viewer is None:
            raise RuntimeError("No napari viewer found")

    widget = SproutPromptWidget(viewer)
    # Ensure widget is visible
    widget.show()
    return widget

## Codes for load widget
import os
from skimage.io import imread
from magicgui import magic_factory

@magic_factory(
    load_mode={
        "choices": ["multiple_files", "folder"],
        "label": "Load mode"
    },
    convert_to_labels={"widget_type": "CheckBox", "label": "Convert to Labels"},
    call_button="Load"
)
def load_images_widget(viewer: "napari.viewer.Viewer",
                       load_mode: str, convert_to_labels: bool  = True):
    file_paths = []

    if load_mode == "multiple_files":
        file_paths, _ = QFileDialog.getOpenFileNames(
            caption="Select image files",
            filter="Images (*.png *.jpg *.tif *.tiff *.bmp *.npy)"
        )
    elif load_mode == "folder":
        folder = QFileDialog.getExistingDirectory(caption="Select folder")
        if folder:
            image_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy")
            file_paths = [
                os.path.join(folder, f)
                for f in sorted(os.listdir(folder))
                if f.lower().endswith(image_exts)
            ]

    if not file_paths:
        print("[INFO] No files selected or folder empty.")
        return

    for path in file_paths:
        name = os.path.basename(path)

        if path.endswith(".npy"):
            img = np.load(path)
        else:
            img = imread(path)

        if convert_to_labels:
            if not np.issubdtype(img.dtype, np.integer):
                img = (img > 0).astype(np.uint8)
            viewer.add_labels(img, name=f"{name}_label")
        else:
            viewer.add_image(img, name=name)

    print(f"[✅] Loaded {len(file_paths)} images (convert_to_labels={convert_to_labels})")