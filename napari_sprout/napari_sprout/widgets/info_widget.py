from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton
from napari.layers import Labels, Image
import numpy as np

try:
    from napari.qt import thread_worker
except ImportError:
    from napari.utils import thread_worker


class LayerInfoWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.current_layer = None
        self.worker = None  # Save worker to prevent GC

        layout = QVBoxLayout()

        # Show active layer name
        self.layer_label = QLabel("Active Layer: (none)")
        self.layer_label.setToolTip("Currently active layer.")
        layout.addWidget(self.layer_label)

        self.refresh_btn = QPushButton("🔄 Refresh Info")
        self.refresh_btn.setToolTip("Manually refresh the layer information display.")
        self.refresh_btn.clicked.connect(self.update_info)
        layout.addWidget(self.refresh_btn)

        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setToolTip("Shows properties of the currently active layer.")
        layout.addWidget(self.info_box)

        self.setLayout(layout)

        # Connect layer selection event
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
        print(f"Updating info for layer: {layer.name}")
        self.info_box.setPlainText("Calculating layer info...")

        @thread_worker
        def compute_info():
            summary = [
                f"Name: {layer.name}",
                f"Layer type: {type(layer).__name__}",
                f"Shape: {layer.data.shape}",
                f"Data type: {layer.data.dtype}",
            ]

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

            return summary

        def on_return(summary):
            self.info_box.setPlainText("\n".join(summary))

        def on_error(e):
            self.info_box.setPlainText(f"[Error] Failed to calculate info:\n{str(e)}")

        self.worker = compute_info()
        self.worker.returned.connect(on_return)
        self.worker.errored.connect(on_error)
        self.worker.start()

   
    def closeEvent(self, event):
        # print("closeEvent called")
        try:
            self.viewer.layers.selection.events.active.disconnect(self.on_active_layer_changed)
        except Exception:
            print("Warning: Could not disconnect active layer change event.")
            pass
        if self.worker:
            self.worker.quit()

    def showEvent(self, event):
        # print("showEvent called")
        try:
            self.viewer.layers.selection.events.active.connect(self.on_active_layer_changed)
        except Exception:
            print("Warning: Could not connect active layer change event.")
            pass
        self.on_active_layer_changed()
        
        
    def hideEvent(self, event):
        # print("hideEvent called")
        try:
            self.viewer.layers.selection.events.active.disconnect(self.on_active_layer_changed)
        except Exception:
            print("Warning: Could not disconnect active layer change event.")
            pass
        if self.worker:
            self.worker.quit()