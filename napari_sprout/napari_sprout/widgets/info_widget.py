from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit
from napari.layers import Labels
import numpy as np

class LabelLayerInfoWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.current_layer = None

        layout = QVBoxLayout()

        # Show the active label layer name
        self.layer_label = QLabel("Active Label Layer: (none)")
        self.layer_label.setToolTip("Currently active label layer.")
        layout.addWidget(self.layer_label)

        # QTextEdit for displaying label layer properties
        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setToolTip("Shows properties of the currently active label layer.")
        layout.addWidget(self.info_box)

        self.setLayout(layout)

        # listen to event
        self.viewer.layers.selection.events.active.connect(self.on_active_layer_changed)

    def on_active_layer_changed(self, event=None):
        layer = self.viewer.layers.selection.active
        if isinstance(layer, Labels):
            self.current_layer = layer
            self.layer_label.setText(f"Active Label Layer: {layer.name}")
        else:
            self.current_layer = None
            self.layer_label.setText("Active Label Layer: (none)")
        self.update_info()

    def update_info(self):
        if self.current_layer is None:
            self.info_box.setPlainText("No valid label layer selected.")
            return

        data = self.current_layer.data
        unique = np.unique(data)
        label_count = len(unique)

        
        sizes = {label: np.sum(data == label) for label in unique}
        nonzero_labels = [l for l in sizes if l != 0]

        max_size = max(sizes.values()) if sizes else 0
        min_size = min(sizes[l] for l in nonzero_labels) if nonzero_labels else 0
        total_size = sum(sizes[l] for l in nonzero_labels) if nonzero_labels else 0

        summary = [
            f"Name: {self.current_layer.name}",
            f"Shape: {data.shape}",
            f"Data type: {data.dtype}",
            f"Label count: {label_count}",
            f"Min label: {unique.min()}",
            f"Max label: {unique.max()}",
            f"Labels: {unique[:20]}{' ...' if label_count > 20 else ''}",
            f"",
            f"Voxel count per label:",
            f"  Max label size: {max_size}",
            f"  Min label size (nonzero): {min_size}",
            f"  Total nonzero size: {total_size}",
        ]

        self.info_box.setPlainText("\n".join(summary))