"""
The known bugs of the plugin, one per run, for test_known_bugs.py.

    python tests/probe_known_bugs.py <name>

Exit code 0: it works. 3: the bug is still there. Anything else: the probe itself broke.
"""
import os
import sys
import tempfile
from pathlib import Path

import _docks_common as C

ROOT = C.pick_root(sys.argv)

import napari  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
import yaml  # noqa: E402

import napari_sprout.widgets.prompt_tab as PT  # noqa: E402

PT.SETTINGS = ("SPROUT-checks", "napari-sprout-checks")      # not the user's
DIALOGS = C.Dialogs().install()
TMP = Path(tempfile.mkdtemp(prefix="sprout_known_bugs_"))
SCALE, TRANSLATE = (2.0, 1.0, 1.0), (5.0, 0.0, 0.0)


def result(ok, why):
    print(("works: " if ok else "BUG: ") + why, flush=True)
    os._exit(0 if ok else 3)                       # the answer is known: no Qt teardown


def labels():
    seg = np.zeros((8, 8, 8), np.uint8)
    seg[1:3] = 1
    seg[4:6] = 2
    return seg


def seed_tab(viewer):
    _, sprout = viewer.window.add_plugin_dock_widget("napari-sprout", "SPROUT")
    C.pump(10)
    return next(w for w in sprout.findChildren(object) if hasattr(w, "import_yaml") and hasattr(w, "params_status"))


def keeps_scale(layer):
    return (tuple(layer.scale) == SCALE and tuple(layer.translate) == TRANSLATE,
            f"{layer.name}: scale {tuple(layer.scale)}, translate {tuple(layer.translate)}")


def edit_keeps_scale(viewer):
    layer = viewer.add_labels(labels(), name="L", scale=SCALE, translate=TRANSLATE)
    viewer.layers.selection.active = layer
    _, edit = viewer.window.add_plugin_dock_widget("napari-sprout", "Edit")
    C.pump(10)
    edit.duplicate_checkbox.setChecked(True)
    edit.src_spin.setValue(1)
    edit.dst_spin.setValue(3)
    edit.set_class_operation()
    C.pump(10)
    result(*keeps_scale(viewer.layers["L_set"]))


def mapping_keeps_scale(viewer):
    layer = viewer.add_labels(labels(), name="L", scale=SCALE, translate=TRANSLATE)
    viewer.layers.selection.active = layer
    _, mapping = viewer.window.add_plugin_dock_widget("napari-sprout", "Label Mapping")
    C.pump(10)
    mapping.class_input.setText("head")
    mapping.add_class()
    mapping.label2class[1] = "head"
    mapping.apply_mapping_to_segmentation()
    C.wait(lambda: "L_mapped" in viewer.layers, 60, "mapped layer")
    result(*keeps_scale(viewer.layers["L_mapped"]))


def float_image_threshold_row(viewer):
    tab = seed_tab(viewer)
    viewer.add_image(np.random.default_rng(0).random((8, 8, 8), dtype=np.float32), name="F")
    C.pump(10)
    tab.image_combo.setCurrentText("F")
    C.pump(10)
    params = tab.main_param_widget
    try:
        params.populate_thresholds_from_list([0.5], [1.0])
    except ValueError as e:
        result(False, str(e))
    result(params.threshold_table.rowCount() == 1, "the table has its row")


def import_adaptive_template(viewer):
    tab = seed_tab(viewer)
    viewer.add_image(np.zeros((8, 8, 8), np.uint8), name="I")
    C.pump(10)
    tab.import_yaml(str(ROOT / "template" / "make_adaptive_seed.yaml"))
    C.pump(10)
    status = tab.params_status.text()
    result("Failed" not in status, status)


def load_scheme_checks_the_file(viewer):
    viewer.add_labels(labels(), name="L")
    _, mapping = viewer.window.add_plugin_dock_widget("napari-sprout", "Label Mapping")
    C.pump(10)
    path = TMP / "other.json"
    path.write_text('{"a": 1}')
    DIALOGS.open_path = str(path)
    before = list(mapping.scheme)
    try:
        mapping.load_scheme()
    except Exception as e:
        result(False, f"{type(e).__name__}: {e}")
    result(mapping.scheme == before, f"scheme after loading any JSON: {mapping.scheme!r}")


def export_writes_the_image_path(viewer):
    path = TMP / "img.tif"
    tifffile.imwrite(path, np.zeros((8, 8, 8), np.uint8))
    tab = seed_tab(viewer)
    viewer.open(str(path), plugin="napari")
    C.pump(10)
    out = TMP / "export.yaml"
    tab.export_yaml(str(out))
    written = yaml.safe_load(open(out, encoding="utf-8"))["img_path"]
    result(Path(written) == path, f"img_path: {written}")


if __name__ == "__main__":
    globals()[sys.argv[1]](napari.Viewer(show=False))
