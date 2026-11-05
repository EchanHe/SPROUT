"""
Result Labels layers are at least uint16 (option c): napari writes a brushed id that does not fit the
layer dtype modulo 2**bits (300 -> 44 in uint8) without any warning, so the layers SPROUT creates for
results - Seeds (already), Grow, Label Mapping Apply, Prompt - are at least uint16, wider only when the
ids need it (> 65535 -> uint32). Values unchanged; files on disk keep their smallest dtype.

    python tests/check_result_layer_dtype.py [--root <tree>]
Prints PASS/FAIL per check; exits non-zero if anything failed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _docks_common as C  # noqa: E402

ROOT = C.pick_root(sys.argv)

import napari  # noqa: E402
import numpy as np  # noqa: E402
from napari.layers import Labels  # noqa: E402

from napari_sprout._widget import SPROUTWidget  # noqa: E402
from napari_sprout.widgets.map_widget import LabelMapperWidget  # noqa: E402
import napari_sprout.widgets.prompt_tab as PT  # noqa: E402

print("source tree:", Path(PT.__file__).resolve().parents[3], flush=True)
PT.SETTINGS = ("SPROUT-checks", "napari-sprout-checks")
FAILED = []
DLG = C.Dialogs().install()


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILED.append(msg)


# why: napari wraps a brushed id in a uint8 layer (the reason for the rule)
lab = Labels(np.zeros((5, 5), np.uint8))
lab.paint((2, 2), 300)
lab16 = Labels(np.zeros((5, 5), np.uint16))
lab16.paint((2, 2), 300)
ok(int(lab.data[2, 2]) == 44 and int(lab16.data[2, 2]) == 300,
   f"napari: brushing id 300 writes {int(lab.data[2, 2])} into uint8, {int(lab16.data[2, 2])} into uint16")

viewer = napari.Viewer(show=False)
main = SPROUTWidget(viewer)
gw = main.grow_widget

# Grow: the finished handler adds the layers
small = np.zeros((4, 6, 6), np.uint8)
small[1:3, 1:3, 1:3] = 7
big = np.zeros((4, 6, 6), np.uint32)
big[1:3, 3:5, 3:5] = 70000
gw._run_roi, gw._run_layer_kwargs, gw._run_note = None, {}, ""
gw._on_growth_finished(({"FINAL_GROW_small": small, "FINAL_GROW_big": big}, {}), 1.0)
gs, gb = viewer.layers["FINAL_GROW_small"].data, viewer.layers["FINAL_GROW_big"].data
ok(gs.dtype == np.uint16 and np.array_equal(gs, small),
   f"Grow result with ids <= 255: layer {gs.dtype} (core result uint8), same values")
ok(gb.dtype == np.uint32 and np.array_equal(gb, big), f"Grow result with id 70000: layer {gb.dtype}, same values")
ok(small.dtype == np.uint8, "Grow: the core result itself is untouched (uint8; files keep it)")

# Label Mapping Apply
src = viewer.add_labels(small.copy(), name="S")
mw = LabelMapperWidget(viewer)
viewer.layers.selection.active = src
C.pump()
mw.class_input.setText("head")
mw.add_class()
mw.radio_manual.setChecked(True)
pos = tuple(int(v) for v in np.argwhere(small == 7)[0])
viewer.dims.set_point(0, pos[0])
mw.on_click(src, C.FakeClick(pos, ndim=3))
n0 = len(viewer.layers)
mw.apply_mapping_to_segmentation()
C.wait(lambda: len(viewer.layers) > n0, timeout=60, what="apply")
C.pump(5)
m = viewer.layers["S_mapped"].data
ok(m.dtype == np.uint16 and np.array_equal(m, (small == 7).astype(np.uint16)),
   f"Label Mapping Apply on a uint8 layer: '{'S_mapped'}' is {m.dtype}, region 7 -> class 1")

# Prompt: the dtype of its result / per-class layers
ok(PT.layer_dtype(6) == np.uint16 and PT.layer_dtype(70000) == np.uint32 if hasattr(PT, "layer_dtype") else False,
   "Prompt result layers: uint16 for class ids <= 65535, uint32 above (prompt_tab.layer_dtype)")

viewer.close()
print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
sys.exit(1 if FAILED else 0)
