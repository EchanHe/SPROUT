import weakref

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton
from napari.layers import Labels, Image, Shapes
import numpy as np

from sprout_core.prompt_points import class_stats
from sprout_core.roi import RoiError, format_percent

from ..utils.roi_layers import DASH, get_roi_hub, is_roi_layer, layer_meta, roi_from_layer

try:
    from napari.qt import thread_worker
except ImportError:
    from napari.utils import thread_worker


TIMES = " × "
CHUNK_VOXELS = 1 << 22      # voxels read (and, for dask, computed) at a time
CACHE_SIZE = 8              # summaries kept, newest last


class _Stale(Exception):
    """A newer request replaced this one."""


def _chunks(shape):
    """Slices along the first axis, about CHUNK_VOXELS voxels each."""
    if len(shape) == 0:
        return [()]
    per_row = max(1, int(np.prod(shape[1:], dtype=np.int64)))
    step = max(1, CHUNK_VOXELS // per_row)
    return [slice(i, min(i + step, shape[0])) for i in range(0, shape[0], step)]


def _blocks(data, alive):
    """numpy blocks of `data` (a numpy or dask array), one chunk at a time."""
    for sl in _chunks(data.shape):
        if not alive():
            raise _Stale
        yield np.asarray(data[sl])


def labels_summary(data, alive=lambda: True):
    """The Labels lines of the info box, counted per chunk with class_stats."""
    counts = {}
    background = 0
    for block in _blocks(data, alive):
        stats = class_stats(block)
        for i, n in zip(stats["class_id"].tolist(), stats["voxels"].tolist()):
            counts[i] = counts.get(i, 0) + n
        background += block.size - int(stats["voxels"].sum())
    if background:
        counts[0] = background
    unique = np.array(sorted(counts), dtype=np.intp)
    label_count = len(unique)

    sizes = {int(label): int(counts[label]) for label in unique}
    nonzero_labels = [l for l in sizes if l != 0]
    max_size = max(sizes.values()) if sizes else 0
    min_size = min(sizes[l] for l in nonzero_labels) if nonzero_labels else 0
    total_size = sum(sizes[l] for l in nonzero_labels) if nonzero_labels else 0

    return [
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


def image_summary(data, alive=lambda: True):
    """The Image lines of the info box: min / max / mean / std, two passes over chunks
    (float64 running sums; the std uses the mean of the first pass, as np.std does)."""
    if data.size == 0:
        np.min(np.asarray(data))        # the same error as before for an empty image
    lo = hi = None
    total = 0.0
    for block in _blocks(data, alive):
        bmin, bmax = np.min(block), np.max(block)
        lo = bmin if lo is None else np.minimum(lo, bmin)       # NaN propagates like np.min
        hi = bmax if hi is None else np.maximum(hi, bmax)
        total += np.sum(block, dtype=np.float64)
    n = data.size
    mean = total / n
    squares = 0.0
    for block in _blocks(data, alive):
        squares += np.sum((block.astype(np.float64) - mean) ** 2)
    std = np.sqrt(squares / n)
    return [
        f"Min value: {lo:.3f}",
        f"Max value: {hi:.3f}",
        f"Mean: {mean:.3f}",
        f"Std dev: {std:.3f}",
    ]


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
        self._job = 0               # token of the newest request; older jobs stop at their next chunk
        self._scheduled = False
        self._running_key = None    # cache key of the job in flight
        self._cache = []            # [(key, refs, summary)] newest last
        self._versions = weakref.WeakKeyDictionary()    # layer -> data / paint events seen

        layout = QVBoxLayout()

        # Show active layer name
        self.layer_label = QLabel("Active Layer: (none)")
        self.layer_label.setToolTip("Currently active layer.")
        layout.addWidget(self.layer_label)

        self.refresh_btn = QPushButton("Refresh Info")
        self.refresh_btn.setToolTip("Recompute the information of the active layer.")
        self.refresh_btn.clicked.connect(self.refresh)
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
            self._watch(layer)
        else:
            self.layer_label.setText("Active Layer: (none)")
        # napari sends several selection events for one change: compute once, after them
        if not self._scheduled:
            self._scheduled = True
            QTimer.singleShot(0, self._run_scheduled)

    def _run_scheduled(self):
        self._scheduled = False
        self.update_info(force=False)

    def refresh(self):
        """The Refresh button: recompute even if the layer and its data did not change."""
        self.update_info(force=True)

    def _watch(self, layer):
        """Count data / paint events of `layer` (part of the cache key)."""
        if layer in self._versions:
            return
        versions = self._versions       # not self: the layer's events must not keep the dock alive
        versions[layer] = 0
        ref = weakref.ref(layer)

        def bump(event=None):
            lay = ref()
            if lay is not None and lay in versions:
                versions[lay] += 1
        for name in ("data", "paint"):
            emitter = getattr(layer.events, name, None)
            if emitter is not None:
                emitter.connect(bump)

    def _key(self, layer):
        return (id(layer), id(layer.data), layer.name, self._versions.get(layer, 0))

    def _cached(self, key, layer):
        for k, refs, summary in self._cache:
            if k == key and refs[0]() is layer and refs[1]() is layer.data:
                return summary
        return None

    def _store(self, key, layer, summary):
        try:
            refs = (weakref.ref(layer), weakref.ref(layer.data))
        except TypeError:
            return
        self._cache = [c for c in self._cache if c[0] != key][-(CACHE_SIZE - 1):] + [(key, refs, summary)]

    def update_info(self, force=True):
        layer = self.current_layer
        # a result of an older request (another layer) must not overwrite this one
        if layer is None:
            self._job += 1
            self._running_key = None
            self.info_box.setPlainText("No valid layer selected.")
            return
        if is_roi_layer(layer):
            self._job += 1
            self._running_key = None
            self.info_box.setPlainText(chr(10).join(roi_layer_summary(layer, self.viewer)))
            return
        if isinstance(layer, Shapes):
            self._job += 1
            self._running_key = None
            self.info_box.setPlainText(chr(10).join(shapes_summary(layer)))
            return
        key = self._key(layer)
        if not force:
            if key == self._running_key:
                return                          # the same layer and data are being computed
            summary = self._cached(key, layer)
            if summary is not None:
                self._job += 1
                self._running_key = None
                self.info_box.setPlainText("\n".join(summary))
                return
        self._job += 1
        job = self._job
        self._running_key = key
        self.info_box.setPlainText("Calculating layer info...")
        data = layer.data

        def alive():
            return job == self._job

        @thread_worker
        def compute_info():
            summary = [
                f"Name: {layer.name}",
                f"Layer type: {type(layer).__name__}",
                f"Shape: {data.shape}",
                f"Data type: {data.dtype}",
            ]
            try:
                if isinstance(layer, Labels):
                    summary += labels_summary(data, alive)
                elif isinstance(layer, Image):
                    summary += image_summary(data, alive)
                else:
                    summary += ["Unsupported layer type."]
            except _Stale:
                return None
            return summary

        def on_return(summary):
            if job == self._job and summary is not None:
                self._running_key = None
                self._store(key, layer, summary)
                self.info_box.setPlainText("\n".join(summary))

        def on_error(e):
            if job == self._job:
                self._running_key = None
                self.info_box.setPlainText(f"[Error] Failed to calculate info:\n{str(e)}")

        self.worker = compute_info()
        self.worker.returned.connect(on_return)
        self.worker.errored.connect(on_error)
        self.worker.start()

   
    def closeEvent(self, event):
        try:
            self.viewer.layers.selection.events.active.disconnect(self.on_active_layer_changed)
        except Exception:
            pass
        self._job += 1                  # a running job stops at its next chunk
        self._running_key = None
        if self.worker:
            self.worker.quit()

    def showEvent(self, event):
        try:
            self.viewer.layers.selection.events.active.connect(self.on_active_layer_changed)
        except Exception:
            pass
        self.on_active_layer_changed()


    def hideEvent(self, event):
        try:
            self.viewer.layers.selection.events.active.disconnect(self.on_active_layer_changed)
        except Exception:
            pass
        self._job += 1                  # a running job stops at its next chunk
        self._running_key = None
        if self.worker:
            self.worker.quit()