"""ROI layers: one region of interest = one napari Shapes layer.

An ROI layer is a 2D Shapes layer (y, x) holding exactly one axis-aligned rectangle,
with the reference image's scale / translate for its last two axes, so napari shows it
on every z slice without any per-slice updates. The z range (3D images) and the voxel
grid it belongs to live in the layer metadata:

    layer.metadata["sprout_roi_layer"] = {"version": 1, "shape": [Z, Y, X],
        "z": [z0, z1] (half-open; not for 2D images), "ref": "<image name>",
        "z_scale": s, "z_translate": t}

The metadata dict is never changed in place (a duplicated layer shares it): a change
assigns a new dict. The rectangle's corners are voxel edges (k - 0.5); a voxel is in the
ROI when its centre is inside (lo = ceil(min), hi = floor(max) + 1).

RoiHub (one per viewer, get_roi_hub) owns every napari connection the ROI widgets need
and re-emits them as Qt signals, so a closed SPROUT panel leaves no callbacks behind:

    rois_changed        ROI layers were added / removed / renamed
    roi_edited(layer)   an ROI was changed and the change is final (mouse released,
                        a number typed, z range set): the moment to recompute
    roi_moving(layer)   the rectangle is being dragged (throttled): update texts only
    layer_inserted / layer_removed / dims_changed

The hub also keeps each ROI layer's canvas label and colour ("ROI_1 201×71×72" yellow
inside the z range, "ROI_1 outside (z 250–450)" orange-red outside; recoloured only when
that state flips), snaps edited shapes back to one rectangle, and hides ROI layers in the
3D view and in transposed views (restoring them after).
"""
import math
import warnings
import weakref

import numpy as np
from napari.layers import Shapes
from qtpy.QtCore import QObject, QTimer, Signal

from sprout_core.roi import Roi, RoiError

MARKER = "sprout_roi_layer"
YELLOW = "yellow"
OUTSIDE = "#ff5a36"
DASH = "–"
TEXT = {"string": "{label}", "anchor": "upper_left", "size": 10, "color": {"feature": "tcolor"}}
HIDDEN_NOTE = "ROI boxes are shown in the default z-slice 2D view only"

_HUBS = weakref.WeakKeyDictionary()       # viewer -> RoiHub


# --------------------------------------------------------------------------
# layers
# --------------------------------------------------------------------------
def is_roi_layer(layer):
    return isinstance(layer, Shapes) and isinstance(layer.metadata.get(MARKER), dict)


def roi_layers(viewer):
    return [l for l in viewer.layers if is_roi_layer(l)]


def next_roi_name(viewer):
    n = 1
    while f"ROI_{n}" in viewer.layers:
        n += 1
    return f"ROI_{n}"


def layer_meta(layer):
    return dict(layer.metadata.get(MARKER) or {})


def _set_meta(layer, **changes):
    """Never mutate the metadata in place (a duplicated layer shares the dict)."""
    md = {**layer_meta(layer), **changes}
    layer.metadata = {**layer.metadata, MARKER: md}


def rect_corners(lo_yx, hi_yx):
    (y0, x0), (y1, x1) = lo_yx, hi_yx
    return np.array([[y0 - 0.5, x0 - 0.5], [y0 - 0.5, x1 - 0.5], [y1 - 0.5, x1 - 0.5], [y1 - 0.5, x0 - 0.5]], float)


def yx_box(verts, shape_yx):
    """(lo, hi) of the voxels whose centres lie inside the vertices' bbox, clipped."""
    v = np.asarray(verts, float)[:, -2:]
    lo = [min(max(int(math.ceil(a)), 0), n) for a, n in zip(v.min(0), shape_yx)]
    hi = [min(max(int(math.floor(b)) + 1, l), n) for b, l, n in zip(v.max(0), lo, shape_yx)]
    return lo, hi


def roi_from_layer(layer):
    """The layer's ROI (a box Roi named after the layer). RoiError if it has none yet."""
    md = layer.metadata.get(MARKER)
    if not isinstance(md, dict) or "shape" not in md:
        raise RoiError(f"'{layer.name}' is not an ROI layer")
    shape = tuple(int(s) for s in md["shape"])
    if len(layer.data) == 0:
        raise RoiError(f"{layer.name}: no box drawn yet")
    lo, hi = yx_box(layer.data[-1], shape[-2:])
    if len(shape) == 3:
        z0, z1 = md.get("z") or (0, shape[0])
        lo, hi = [int(z0)] + lo, [int(z1)] + hi
    roi = Roi.box(shape, lo, hi, name=layer.name)
    if roi.empty:
        raise RoiError(f"{layer.name} is empty")
    return roi


def sizes_text(sizes, sep="×"):
    return sep.join(str(int(s)) for s in sizes)


def add_roi_layer(viewer, ref=None, lo=None, hi=None, name=None, z=None, *, shape=None, scale=None,
                  translate=None, ref_name=None, z_all=False):
    """A new ROI layer on the voxel grid of `ref` (an Image or Labels layer), or of
    shape / scale / translate / ref_name given directly (e.g. loaded from a file).

    lo / hi: the box (full ndim), or None for an empty layer to draw in.
    z: (z0, z1) for an empty 3D layer (default: all slices). z_all: the z range was not
    chosen by the user yet (the ROI tools say so). Returns the layer.
    """
    if ref is not None:
        shape, scale, translate, ref_name = np.shape(ref.data), ref.scale, ref.translate, ref.name
    shape = tuple(int(s) for s in shape)
    nd = len(shape)
    if nd not in (2, 3):
        raise RoiError("ROIs need a 2D or 3D image")
    scale = np.ones(nd) if scale is None else np.asarray(scale, float)[-nd:]
    translate = np.zeros(nd) if translate is None else np.asarray(translate, float)[-nd:]
    md = {"version": 1, "shape": list(shape), "ref": ref_name}
    if nd == 3:
        md["z"] = [int(lo[0]), int(hi[0])] if lo is not None else list(z or (0, shape[0]))
        md["z_scale"], md["z_translate"] = float(scale[0]), float(translate[0])
        if z_all:
            md["z_all"] = True
    name = name or next_roi_name(viewer)
    width = max(1.0, 0.004 * float(max(shape[-2:])))
    data = [] if lo is None else [rect_corners(lo[-2:], hi[-2:])]
    active = viewer.layers.selection.active
    hub = get_roi_hub(viewer)
    with warnings.catch_warnings():          # the text encoding has no rows yet
        warnings.simplefilter("ignore", RuntimeWarning)
        layer = viewer.add_shapes(
            data, ndim=2, shape_type="rectangle", name=name, edge_color=YELLOW, face_color="transparent",
            edge_width=width, scale=scale[-2:], translate=translate[-2:],
            text=TEXT, features={"label": [name] * len(data), "tcolor": [YELLOW] * len(data)},
            metadata={MARKER: md})
    if active is not None and active in viewer.layers and lo is not None:
        viewer.layers.selection.active = active          # creating a box doesn't steal the selection
    hub.refresh_label(layer, force=True)
    return layer


def layer_extra(layer):
    """{'ref_image', 'scale', 'translate'} of an ROI layer (for save_rois)."""
    md = layer_meta(layer)
    sc, tr = list(map(float, layer.scale)), list(map(float, layer.translate))
    if len(md.get("shape", ())) == 3:
        sc, tr = [md.get("z_scale", 1.0)] + sc, [md.get("z_translate", 0.0)] + tr
    return {"ref_image": md.get("ref"), "scale": sc, "translate": tr}


# --------------------------------------------------------------------------
# hub
# --------------------------------------------------------------------------
def _alive(qobject):
    """False once the Qt object behind a Python wrapper was deleted (a closed panel)."""
    try:
        qobject.objectName()
        return True
    except RuntimeError:
        return False


def get_roi_hub(viewer):
    hub = _HUBS.get(viewer)
    if hub is None:
        hub = _HUBS[viewer] = RoiHub(viewer)
    return hub


class RoiHub(QObject):
    rois_changed = Signal()
    roi_edited = Signal(object)
    roi_moving = Signal(object)
    layer_inserted = Signal(object)
    layer_removed = Signal(object)
    dims_changed = Signal()
    users_changed = Signal()          # a tab picked another ROI ("used by" in the ROI tools)

    MOVING_MS = 30

    def __init__(self, viewer):
        super().__init__()
        self._viewer = weakref.ref(viewer)
        self._attached = weakref.WeakSet()
        self._rois = weakref.WeakKeyDictionary()      # layer -> Roi | RoiError (rebuilt on commit)
        self._inside = weakref.WeakKeyDictionary()    # layer -> (inside?, label) as drawn
        self._hidden = weakref.WeakKeyDictionary()    # layer -> visibility before we hid it
        self._users = weakref.WeakKeyDictionary()     # RoiSelect -> tab name
        self._syncing = False
        self.view_hidden = False                      # ROI layers hidden (3D / transposed view)
        self._moving = None
        self._moving_timer = QTimer(self)
        self._moving_timer.setSingleShot(True)
        self._moving_timer.setInterval(self.MOVING_MS)
        self._moving_timer.timeout.connect(self._emit_moving)
        self._dims_timer = QTimer(self)
        self._dims_timer.setSingleShot(True)
        self._dims_timer.timeout.connect(self._update_view)
        viewer.layers.events.inserted.connect(self._on_inserted)
        viewer.layers.events.removed.connect(self._on_removed)
        for name in ("current_step", "order", "ndisplay"):
            getattr(viewer.dims.events, name).connect(self._on_dims)
        for layer in list(viewer.layers):
            if is_roi_layer(layer):
                self._attach(layer)

    @property
    def viewer(self):
        return self._viewer()

    # ---- ROI layers
    def layers(self):
        v = self.viewer
        return [] if v is None else roi_layers(v)

    def roi_of(self, layer):
        """The layer's Roi (cached between committed edits); RoiError if it has none."""
        r = self._rois.get(layer)
        if r is None:
            try:
                r = roi_from_layer(layer)
            except RoiError as e:
                r = e
            self._rois[layer] = r
        if isinstance(r, RoiError):
            raise r
        if r.name != layer.name:
            r = self._rois[layer] = r.with_name(layer.name)
        return r

    def register_user(self, select, name):
        self._users[select] = name

    def used_by(self, layer):
        """Names of the tabs whose ROI combo points at this layer."""
        return [name for sel, name in list(self._users.items()) if _alive(sel) and sel.layer is layer]

    def live_roi(self, layer):
        """The layer's Roi right now (also while dragging); None if it has none."""
        try:
            return roi_from_layer(layer)
        except RoiError:
            return None

    def set_z_range(self, layer, z0, z1):
        md = layer_meta(layer)
        shape = md["shape"]
        z0, z1 = sorted((int(z0), int(z1)))
        z0, z1 = max(0, min(z0, shape[0])), max(0, min(z1, shape[0]))
        if md.get("z") == [z0, z1] and not md.get("z_all"):
            return
        _set_meta(layer, z=[z0, z1], z_all=False)
        self._commit(layer)

    def set_yx(self, layer, lo_yx, hi_yx):
        self._syncing = True
        try:
            layer.data = [(rect_corners(lo_yx, hi_yx), "rectangle")]
        finally:
            self._syncing = False
        self._commit(layer)

    # ---- napari events
    def _attach(self, layer):
        if layer in self._attached:
            return
        self._attached.add(layer)
        layer.events.name.connect(self._on_name)
        layer.events.data.connect(self._on_data)
        if hasattr(layer.events, "set_data"):
            layer.events.set_data.connect(self._on_set_data)

    def _detach(self, layer):
        self._attached.discard(layer)
        for ev, cb in (("name", self._on_name), ("data", self._on_data), ("set_data", self._on_set_data)):
            em = getattr(layer.events, ev, None)
            if em is not None:
                try:
                    em.disconnect(cb)
                except Exception:
                    pass

    def _on_inserted(self, event):
        layer = event.value
        if is_roi_layer(layer):
            self._attach(layer)
            self._rois.pop(layer, None)
            self.refresh_label(layer, force=True)
            if self.view_hidden:
                self._hide(layer)
            self.rois_changed.emit()
        self.layer_inserted.emit(layer)

    def _on_removed(self, event):
        layer = event.value
        if layer in self._attached:
            self._detach(layer)
            self.rois_changed.emit()
        self.layer_removed.emit(layer)

    def _on_name(self, event=None):
        layer = getattr(event, "source", None)
        if layer is not None and is_roi_layer(layer):
            self.refresh_label(layer, force=True)
        self.rois_changed.emit()

    def _on_data(self, event):
        if self._syncing:
            return
        layer = getattr(event, "source", None)
        action = str(getattr(event, "action", ""))
        if layer is None or action not in ("added", "changed", "removed"):
            return
        # ADDED fires inside napari's mouse-release handler: change the layer later
        QTimer.singleShot(0, lambda: self._commit(layer))

    def _on_set_data(self, event):
        if self._syncing:
            return
        layer = getattr(event, "source", None)
        if layer is not None and is_roi_layer(layer):
            self._moving = weakref.ref(layer)
            if not self._moving_timer.isActive():
                self._moving_timer.start()

    def _emit_moving(self):
        layer = self._moving() if self._moving is not None else None
        if layer is not None and layer in self.viewer.layers:
            self.roi_moving.emit(layer)

    def _on_dims(self, event=None):
        self._dims_timer.start(0)

    # ---- committed edits
    def _commit(self, layer):
        """Snap the layer back to one axis-aligned rectangle on voxel edges, then announce it."""
        v = self.viewer
        if v is None or layer not in v.layers or not is_roi_layer(layer):
            return
        shape = layer_meta(layer)["shape"]
        if len(layer.data):
            lo, hi = yx_box(layer.data[-1], shape[-2:])      # draw again = replace: keep the newest
            rect = rect_corners(lo, hi)
            if len(layer.data) != 1 or str(layer.shape_type[0]) != "rectangle" \
                    or not np.allclose(np.asarray(layer.data[0], float), rect):
                self._syncing = True
                try:
                    layer.data = [(rect, "rectangle")]
                finally:
                    self._syncing = False
        self._rois.pop(layer, None)
        self.refresh_label(layer, force=True)
        self.roi_edited.emit(layer)

    # ---- canvas label / colour / visibility
    def _z_index(self, layer):
        """Index on the layer's z axis of the slice the viewer shows, or None (2D image)."""
        md = layer_meta(layer)
        v = self.viewer
        if len(md.get("shape", ())) != 3 or v is None or v.dims.ndim < 3:
            return None
        world = v.dims.point[v.dims.ndim - 3]
        return int(round((world - md.get("z_translate", 0.0)) / (md.get("z_scale", 1.0) or 1.0)))

    def refresh_label(self, layer, force=False):
        """Text label + colour; only touched when the inside / outside state (or text) changes."""
        if len(layer.data) == 0:
            self._inside.pop(layer, None)
            return
        try:
            roi = roi_from_layer(layer)
        except RoiError:
            return
        z = self._z_index(layer)
        inside = z is None or roi.lo[0] <= z < roi.hi[0]
        if inside:
            label = f"{layer.name} {sizes_text(roi.sizes)}"
        else:
            label = f"{layer.name} outside (z {roi.lo[0]}{DASH}{roi.hi[0] - 1})"
        if not force and self._inside.get(layer) == (inside, label):
            return
        colour = YELLOW if inside else OUTSIDE
        self._syncing = True
        try:
            layer.features = {"label": [label] * len(layer.data), "tcolor": [colour] * len(layer.data)}
            if self._inside.get(layer, (None,))[0] != inside or force:
                layer.edge_color = [colour] * len(layer.data)
        finally:
            self._syncing = False
        self._inside[layer] = (inside, label)

    def label_of(self, layer):
        st = self._inside.get(layer)
        return None if st is None else st[1]

    def _view_ok(self):
        v = self.viewer
        if v is None:
            return True
        n = v.dims.ndim
        return v.dims.ndisplay == 2 and tuple(v.dims.displayed) == (n - 2, n - 1)

    def _hide(self, layer):
        if layer not in self._hidden:
            self._hidden[layer] = layer.visible
        layer.visible = False

    def _update_view(self):
        ok = self._view_ok()
        if not ok and not self.view_hidden:
            self.view_hidden = True
            for layer in self.layers():
                self._hide(layer)
        elif ok and self.view_hidden:
            self.view_hidden = False
            for layer in self.layers():
                if layer in self._hidden:
                    layer.visible = self._hidden.pop(layer)
        if ok:
            for layer in self.layers():
                self.refresh_label(layer)
        self.dims_changed.emit()
