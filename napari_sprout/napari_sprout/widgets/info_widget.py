from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton
from napari.layers import Labels, Image
import numpy as np

class LayerInfoWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.current_layer = None

        layout = QVBoxLayout()

        # Show active layer name
        self.layer_label = QLabel("Active Layer: (none)")
        self.layer_label.setToolTip("Currently active layer.")
        layout.addWidget(self.layer_label)

        self.refresh_btn = QPushButton("🔄 Refresh Info")
        self.refresh_btn.setToolTip("Manually refresh the layer information display.")
        self.refresh_btn.clicked.connect(self.update_info)
        layout.addWidget(self.refresh_btn)

        # Info display
        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setToolTip("Shows properties of the currently active layer.")
        layout.addWidget(self.info_box)

        self.setLayout(layout)

        # Connect active layer change
        self.viewer.layers.selection.events.active.connect(self.on_active_layer_changed)

    def on_active_layer_changed(self, event=None):
        layer = self.viewer.layers.selection.active
        self.current_layer = layer

        if layer is not None:
            self.layer_label.setText(f"Active Layer: {layer.name}")
        else:
            self.layer_label.setText("Active Layer: (none)")

        self.update_info()

    def update_info(self):
        layer = self.current_layer
        if layer is None:
            self.info_box.setPlainText("No valid layer selected.")
            return

        summary = [
            f"Name: {layer.name}",
            f"Layer type: {type(layer).__name__}",
            f"Shape: {layer.data.shape}",
            f"Data type: {layer.data.dtype}",
        ]

        # Handle Labels
        if isinstance(layer, Labels):
            data = layer.data
            unique = np.unique(data)
            label_count = len(unique)

            sizes = {label: np.sum(data == label) for label in unique}
            nonzero_labels = [l for l in sizes if l != 0]
            max_size = max(sizes.values()) if sizes else 0
            min_size = min(sizes[l] for l in nonzero_labels) if nonzero_labels else 0
            total_size = sum(sizes[l] for l in nonzero_labels) if nonzero_labels else 0

            summary += [
                f"Label count: {label_count}",
                f"Min label: {unique.min()}",
                f"Max label: {unique.max()}",
                f"Labels: {unique[:20]}{' ...' if label_count > 20 else ''}",
                "",
                f"Voxel count per label:",
                f"  Max label size: {max_size}",
                f"  Min label size (nonzero): {min_size}",
                f"  Total nonzero size: {total_size}",
            ]

        # Handle Image
        elif isinstance(layer, Image):
            data = layer.data
            summary += [
                f"Min value: {np.min(data):.3f}",
                f"Max value: {np.max(data):.3f}",
                f"Mean: {np.mean(data):.3f}",
                f"Std dev: {np.std(data):.3f}",
            ]


        else:
            summary += ["Unsupported layer type."]

        self.info_box.setPlainText("\n".join(str(line) for line in summary))
