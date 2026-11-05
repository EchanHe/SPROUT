"""
Headless check of ROI layers (napari_sprout/utils/roi_layers.py): one ROI = one 2D Shapes
layer with one rectangle and its z range in the metadata, and the per-viewer RoiHub.

    create / draw / edit, value of the Roi, drag storm (live texts only, one commit),
    snap-back (rotation, polygon, a second rectangle), anisotropic placement,
    transposed / 3D view hide + restore, rename, duplicate does not share metadata,
    canvas labels and colours inside / outside the z range (recoloured only on a flip),
    (the tabs' ROI combos, per-tab recomputes and the dock-close lifecycle: check_roi_widget.py).

Run in the `sprout` env:
    python tests/check_roi_layers.py
Prints PASS/FAIL per check; exits non-zero if anything failed.
"""
import gc
import os
import sys
import time
import traceback
import weakref
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "napari_sprout")]

import napari  # noqa: E402
import _headless_gl  # noqa: E402,F401  (napari >= 0.7 offscreen: no GL context)
import numpy as np  # noqa: E402
from napari.layers import Layer  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from napari_sprout.utils import roi_layers as RL  # noqa: E402
from sprout_core.roi import Roi  # noqa: E402

FAILED = []
UNCAUGHT = []
sys.excepthook = lambda *a: UNCAUGHT.append("".join(traceback.format_exception(*a)))
YELLOW = np.array([1, 1, 0, 1.0])
OUT = np.array([1.0, 0x5a / 255, 0x36 / 255, 1.0])


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILED.append(msg)


def pump(n=20):
    for _ in range(n):
        QApplication.processEvents()
        time.sleep(0.005)


def pump_for(sec):
    t = time.time()
    while time.time() - t < sec:
        QApplication.processEvents()
        time.sleep(0.005)


def edge(layer):
    return np.asarray(layer.edge_color)[0]


class Spy:
    def __init__(self, hub):
        self.edited, self.moving, self.changed = [], [], 0
        hub.roi_edited.connect(lambda l: self.edited.append(l.name))
        hub.roi_moving.connect(lambda l: self.moving.append(l.name))
        hub.rois_changed.connect(self._ch)

    def _ch(self):
        self.changed += 1

    def reset(self):
        self.edited.clear(), self.moving.clear()
        self.changed = 0


# ======================================================================
print("== create, value, label")
v = napari.Viewer(show=False)
img = np.zeros((40, 60, 80), np.uint8)
il = v.add_image(img, name="img")
hub = RL.get_roi_hub(v)
ok(RL.get_roi_hub(v) is hub and hub.layers() == [], "one hub per viewer, no ROI layers")
spy = Spy(hub)
v.dims.set_point(0, 15)
pump()
l1 = RL.add_roi_layer(v, il, (10, 6, 8), (21, 31, 51))
pump()
r1 = hub.roi_of(l1)
md = l1.metadata[RL.MARKER]
ok(l1.name == "ROI_1" and l1.ndim == 2 and len(l1.data) == 1 and RL.is_roi_layer(l1)
   and md["shape"] == [40, 60, 80] and md["z"] == [10, 21] and md["ref"] == "img",
   f"add_roi_layer: '{l1.name}', 2D Shapes, one rectangle, metadata {md}")
ok(r1.name == "ROI_1" and (r1.lo, r1.hi) == ((10, 6, 8), (21, 31, 51)) and r1.key == Roi.box(img.shape, (10, 6, 8),
                                                                                             (21, 31, 51)).key,
   "roi_of(): the box, named after the layer")
ok(v.layers.selection.active is il, "creating an ROI layer does not steal the selection")
ok(spy.changed == 1 and hub.label_of(l1) == "ROI_1 11×25×43" and np.allclose(edge(l1), YELLOW)
   and list(l1.features["label"]) == ["ROI_1 11×25×43"], f"label {hub.label_of(l1)!r}, yellow; rois_changed")
ok(RL.next_roi_name(v) == "ROI_2", "next name ROI_2")
_probe = v.add_shapes([RL.rect_corners((0, 0), (2, 2))], ndim=2, shape_type="rectangle", name="probe")
_pe = []
_probe.events.edge_color.connect(lambda e: _pe.append(1))
_probe.edge_color = ["red"]
PER = len(_pe)                              # napari emits this many edge_color events per assignment
v.layers.remove(_probe)
v.layers.selection.active = il
ev = []
l1.events.edge_color.connect(lambda e: ev.append(1))


class _EC:
    def __len__(self):
        return len(ev) // max(PER, 1)


ec = _EC()
for z in (12, 14, 16, 18, 20):
    v.dims.set_point(0, z)
    pump(3)
pump()
ok(not ec and hub.label_of(l1) == "ROI_1 11×25×43", "paging inside the z range: no recolour at all")
v.dims.set_point(0, 25)
pump()
ok(len(ec) == 1 and np.allclose(edge(l1), OUT) and hub.label_of(l1) == "ROI_1 outside (z 10–20)",
   f"crossing out of the z range: one recolour, orange-red, {hub.label_of(l1)!r} ({len(ec)} recolours)")
for z in (27, 30, 35):
    v.dims.set_point(0, z)
    pump(3)
pump()
ok(len(ec) == 1, "paging outside: no further recolour")
v.dims.set_point(0, 15)
pump()
ok(len(ec) == 2 and np.allclose(edge(l1), YELLOW), "back inside: yellow (one recolour)")

# ======================================================================
print("\n== edits: z range, y / x, drag storm, snap-back")
spy.reset()
hub.set_z_range(l1, 5, 30)
pump()
ok(hub.roi_of(l1).lo[0] == 5 and hub.roi_of(l1).hi[0] == 30 and spy.edited == ["ROI_1"], "set_z_range: one roi_edited")
hub.set_yx(l1, (2, 3), (20, 30))
pump()
r = hub.roi_of(l1)
ok((r.lo, r.hi) == ((5, 2, 3), (30, 20, 30)) and spy.edited == ["ROI_1"] * 2, "set_yx moves the rectangle")
spy.reset()
for i in range(25):                      # a drag: napari shifts the shape and emits set_data each move
    l1._data_view.shift(0, np.array([0.25, 0.5]))
    l1.events.set_data()
    QApplication.processEvents()
pump_for(0.2)
ok(spy.edited == [] and 1 <= len(spy.moving) <= 10, f"drag storm: 25 moves -> {len(spy.moving)} throttled "
                                                     f"'moving' signals, 0 commits")
l1.events.data(value=l1.data, action="changed", data_indices=(0,), vertex_indices=((),))
pump()
r = hub.roi_of(l1)
v0 = np.asarray(l1.data[0])
ok(spy.edited == ["ROI_1"] and (r.lo[1:], r.hi[1:]) == ((8, 15), (26, 43))
   and np.allclose(np.sort(np.unique(v0[:, 0])), [7.5, 25.5]), f"release: one commit, snapped to voxel edges {r.lo}..{r.hi}")
spy.reset()
c = np.array([20.0, 20.0])
ang = np.deg2rad(20)
rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
corners = (np.array([[-5, -10], [-5, 10], [5, 10], [5, -10]], float) @ rot.T) + c
l1.data = [(corners, "rectangle")]
pump()
r = hub.roi_of(l1)
lo_e, hi_e = RL.yx_box(corners, (60, 80))
ok(len(l1.data) == 1 and (list(r.lo[1:]), list(r.hi[1:])) == (lo_e, hi_e) and spy.edited == ["ROI_1"]
   and np.allclose(np.asarray(l1.data[0]), RL.rect_corners(lo_e, hi_e)), "rotated rectangle -> axis-aligned bbox")
l1.data = [(np.array([[5, 5], [5, 25], [15, 30], [25, 25], [25, 5]], float), "polygon")]
pump()
ok(str(l1.shape_type[0]) == "rectangle" and (hub.roi_of(l1).lo[1:], hub.roi_of(l1).hi[1:]) == ((5, 5), (26, 31)),
   "polygon (vertex inserted) -> back to one rectangle")
l1.add_rectangles(RL.rect_corners((30, 40), (50, 70)))
pump()
r = hub.roi_of(l1)
ok(len(l1.data) == 1 and (r.lo[1:], r.hi[1:]) == ((30, 40), (50, 70)), "a second rectangle drawn: the newest replaces the box")
ok(r.lo[0] == 5 and r.hi[0] == 30, "y / x edits keep the z range")

# ======================================================================
print("\n== anisotropic image, 2D image, empty layer")
sc, tr = (2.0, 0.5, 0.25), (10.0, 3.0, 4.0)
ia = v.add_image(np.zeros((30, 50, 70), np.uint8), name="aniso", scale=sc, translate=tr)
la = RL.add_roi_layer(v, ia, (4, 10, 20), (9, 30, 50))
pump()
ra = hub.roi_of(la)
w_roi = la.data_to_world(np.array([10 - 0.5, 20 - 0.5]))
w_img = ia.data_to_world(np.array([4, 10 - 0.5, 20 - 0.5]))
ok(np.allclose(la.scale, sc[1:]) and np.allclose(la.translate, tr[1:]) and np.allclose(w_roi, w_img[1:])
   and la.metadata[RL.MARKER]["z_scale"] == 2.0 and (ra.lo, ra.hi) == ((4, 10, 20), (9, 30, 50)),
   "anisotropic: the layer has the image's y / x scale + translate; corners land on the image's voxel edges")
v.dims.set_point(0, tr[0] + sc[0] * 6)                    # data z 6 of 'aniso'
pump()
ok(hub.label_of(la) == f"{la.name} 5×20×30" and np.allclose(edge(la), YELLOW),
   f"aniso: z index from the image's z scale / translate ({hub.label_of(la)!r})")
i2 = v.add_image(np.zeros((50, 70), np.uint8), name="flat")
l2 = RL.add_roi_layer(v, i2, (5, 6), (20, 30))
pump()
ok("z" not in l2.metadata[RL.MARKER] and hub.roi_of(l2).shape == (50, 70) and hub.label_of(l2) == f"{l2.name} 15×24"
   and np.allclose(edge(l2), YELLOW), "2D image: no z, always 'inside'")
le = RL.add_roi_layer(v, il)
pump()
try:
    hub.roi_of(le)
    ok(False, "empty layer raises")
except Exception as e:
    ok("no box drawn yet" in str(e) and le.metadata[RL.MARKER]["z"] == [0, 40], f"empty layer: '{e}', z: all")
v.layers.remove(le)
pump()

# ======================================================================
print("\n== views, rename, duplicate")
v.dims.set_point(0, 15)
l2.visible = False                         # the user hid this one
v.dims.order = (2, 0, 1)
pump()
ok(hub.view_hidden and not any(l.visible for l in hub.layers()), "transposed view: ROI layers hidden")
v.dims.order = (0, 1, 2)
pump()
ok(not hub.view_hidden and l1.visible and la.visible and not l2.visible,
   "default view again: visibility restored (a layer the user hid stays hidden)")
v.dims.ndisplay = 3
pump()
ok(hub.view_hidden and not l1.visible, "3D view: hidden")
v.dims.ndisplay = 2
pump()
ok(l1.visible and not l2.visible, "2D again: restored")
spy.reset()
l1.name = "lungs"
pump()
ok(spy.changed >= 1 and hub.label_of(l1).startswith("lungs ") and hub.roi_of(l1).name == "lungs",
   f"rename: rois_changed, label {hub.label_of(l1)!r}")
data, state, kind = l1.as_layer_data_tuple()
dup = v.add_layer(Layer.create(data, state, kind))
pump()
ok(RL.is_roi_layer(dup) and dup.metadata[RL.MARKER] is l1.metadata[RL.MARKER], "a duplicate starts with the same metadata dict")
hub.set_z_range(dup, 1, 3)
pump()
ok(l1.metadata[RL.MARKER]["z"] == [5, 30] and dup.metadata[RL.MARKER]["z"] == [1, 3],
   "editing the duplicate's z range leaves the original alone (new dict, never in place)")
v.layers.remove(dup)
pump()
ok(dup not in hub._attached, "removed ROI layer: its callbacks are disconnected")
v.close()

print("\n== Draw box: the temporary helper layer (hub)")
w = napari.Viewer(show=False)
flat = w.add_image(np.zeros((50, 60), np.uint8), name="flat", scale=(0.5, 2.0), translate=(3.0, 4.0))
h2 = RL.get_roi_hub(w)
got, ended = [], []
if hasattr(h2, "draw_finished"):
    h2.draw_finished.connect(got.append)
    h2.draw_ended.connect(lambda: ended.append(1))
start = getattr(h2, "start_draw", None)
d = start(flat) if start else None
pump()
ok(d is not None and d.name == "SPROUT draw" and RL.is_draw_layer(d) and not RL.is_roi_layer(d)
   and not RL.roi_layers(w) and np.allclose(d.scale, (0.5, 2.0)) and np.allclose(d.translate, (3.0, 4.0))
   and w.layers.selection.active is d and str(d.mode) == "add_rectangle",
   "start_draw: a helper layer in the image's y / x frame, add_rectangle, no ROI layer")
if d is not None:
    d.add_rectangles(RL.rect_corners((5, 6), (20, 30)))
    pump(40)
rois = RL.roi_layers(w)
ok(len(rois) == 1 and rois[0].name == "ROI_1" and (h2.roi_of(rois[0]).lo, h2.roi_of(rois[0]).hi) == ((5, 6), (20, 30))
   and d not in w.layers and got == rois and ended == [1] and w.layers.selection.active is flat,
   "2D image: the first rectangle -> ROI_1 (y / x box), helper removed, draw_finished, image active again")
d = start(flat) if start else None
if d is not None:
    h2.cancel_draw()
ok(d is not None and d in w.layers, "cancel_draw is deferred (never removes inside the caller's napari callback)")
pump()
ok(d is not None and d not in w.layers and len(RL.roi_layers(w)) == 1 and h2.drawing_layer() is None,
   "... then the helper is gone and nothing was made")
d = start(flat) if start else None
d2 = start(flat) if start else None
ok(d is not None and d2 is d and [l.name for l in w.layers].count("SPROUT draw") == 1, "start_draw twice: one helper (reused)")
other = w.add_image(np.zeros((20, 30), np.uint8), name="other")
w.layers.selection.active = d
d3 = start(other) if start else None
pump()
ok(d3 is not None and d3 is not d and d not in w.layers and [l.name for l in w.layers].count("SPROUT draw") == 1
   and np.allclose(d3.scale, (1, 1)), "start_draw on another image: the helper is replaced, still one")
w.layers.remove(other)
pump()
if d3 is not None:
    d3.add_rectangles(RL.rect_corners((1, 1), (5, 5)))
    pump(40)
ok(len(RL.roi_layers(w)) == 1 and not any(l.name == "SPROUT draw" for l in w.layers),
   "its image removed while drawing: the rectangle makes nothing, helper removed")
cbs = [str(c) for em in (w.layers.selection.events.active,) for c in em.callbacks]
ok(not any("_on_draw" in c for c in cbs) and getattr(h2, "_draw", None) is None,
   "no Draw box callbacks left once the drawing is over")
ok(not UNCAUGHT, f"no uncaught errors ({len(UNCAUGHT)})" + (f": {UNCAUGHT[0][-400:]}" if UNCAUGHT else ""))
w.close()

print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
sys.exit(1 if FAILED else 0)
