from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton
from napari.layers import Labels, Image, Shapes
import numpy as np

from sprout_core.roi import RoiError, format_percent

from ..utils.roi_layers import DASH, get_roi_hub, is_roi_layer, layer_meta, roi_from_layer

try:
    from napari.qt import thread_worker
except ImportError:
    from napari.utils import thread_worker


TIMES = " × "


def roi_layer_summary(layer, viewer=None):
    """Lines describing an ROI layer (one box; z range in its metadata)."""
    md = layer_meta(layer)
    shape = tuple(int(s) for s in md.get("shape", ()))
    lines = [f"Name: {layer.name}", "Layer type: ROI (Shapes)"]
    try:
        roi = roi_from_layer(layer)
    except RoiError as e:
        roi = None
        lines.append(str(e))
    if roi is not None:
        axes = roi.axis_names
        lines += [
            "Range " + " · ".join(f"{a} {l}{DASH}{h - 1}" for a, l, h in zip(axes, roi.lo, roi.hi)),
            f"Size {TIMES.join(str(v) for v in roi.sizes)} voxels ({TIMES.join(axes)})",
            f"Voxels {roi.n_voxels:,} ({format_percent(roi.fraction)} of the reference shape)",
        ]
    ref = md.get("ref")
    lines.append(f"Reference image: {ref if ref else '(unknown)'}"
                 + (f" ({TIMES.join(str(v) for v in shape)})" if shape else ""))
    if viewer is not None:
        users = get_roi_hub(viewer).used_by(layer)
        lines.append(f"Used by: {', '.join(users) if users else 'no tab'}")
    if len(shape) == 3 and md.get("z_all"):
        lines.append("z: all slices (not set)")
    return lines


def shapes_summary(layer):
    """Number of shapes and their overall bounding box (data coordinates of the layer)."""
    n = len(layer.data)
    lines = [f"Name: {layer.name}", "Layer type: Shapes", f"Shapes: {n}"]
    if n:
        pts = np.concatenate([np.asarray(d, float) for d in layer.data], axis=0)
        lo, hi = pts.min(0), pts.max(0)
        nd = pts.shape[1]
        names = ["z", "y", "x"][-nd:] if nd <= 3 else [f"axis {i}" for i in range(nd)]
        lines.append("Bounding box " + " · ".join(f"{a} {l:.1f}{DASH}{h:.1f}"
                                                        for a, l, h in zip(names, lo, hi)) + " (data coordinates)")
    return lines


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
        # a result of an older request (another layer) must not overwrite this one
        self._job = getattr(self, "_job", 0) + 1
        job = self._job
        if layer is None:
            self.info_box.setPlainText("No valid layer selected.")
            return
        if is_roi_layer(layer):
            self.info_box.setPlainText(chr(10).join(roi_layer_summary(layer, self.viewer)))
            return
        if isinstance(layer, Shapes):
            self.info_box.setPlainText(chr(10).join(shapes_summary(layer)))
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
                # Single pass: bincount yields the size of every label at once,
                # and its nonzero positions are exactly the unique labels (sorted).
                counts = np.bincount(np.asarray(data).ravel())
                unique = np.nonzero(counts)[0]
                label_count = len(unique)

                sizes = {int(label): int(counts[label]) for label in unique}
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
            if job == self._job:
                self.info_box.setPlainText("\n".join(summary))

        def on_error(e):
            if job == self._job:
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