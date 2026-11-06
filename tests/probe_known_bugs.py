"""
The known bugs of the plugin, one per run, for test_known_bugs.py.

    python tests/probe_known_bugs.py <name>

Exit code 0: it works. 42: the bug is still there. Anything else: the probe itself broke.
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
BUG = 42                                           # not 3: that is how abort() ends on Windows


def result(ok, why):
    print(("works: " if ok else "BUG: ") + why, flush=True)
    os._exit(0 if ok else BUG)                     # the answer is known: no Qt teardown


def labels():
    seg = np.zeros((8, 8, 8), np.uint8)
    seg[1:3] = 1
    seg[4:6] = 2
    return seg


def sprout_tab(viewer, kind):
    _, sprout = viewer.window.add_plugin_dock_widget("napari-sprout", "SPROUT")
    C.pump(10)
    return next(w for w in sprout.findChildren(object) if type(w).__name__ == kind)


def seed_tab(viewer):
    return sprout_tab(viewer, "SeedGenerationWidget")


def table(params):
    t = params.threshold_table
    return [[t.cellWidget(r, c).value() for c in range(t.columnCount())] for r in range(t.rowCount())]


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
    def uncaught(kind, error, trace):              # napari 0.7: through a Qt slot, else PyQt aborts
        if issubclass(kind, ValueError):
            result(False, str(error))
        sys.__excepthook__(kind, error, trace)

    sys.excepthook = uncaught
    tab = seed_tab(viewer)
    try:
        viewer.add_image(np.random.default_rng(0).random((8, 8, 8), dtype=np.float32), name="F")
    except ValueError as e:
        result(False, str(e))
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
    rows = table(tab.main_param_widget)
    status = tab.params_status.text()
    result("Failed" not in status and len(rows) == 1 and rows[0][0] == 1, f"{status} {rows}")


def import_single_values_into_grow(viewer):
    tab = sprout_tab(viewer, "SeedGrowthWidget")
    viewer.add_image(np.zeros((8, 8, 8), np.uint8), name="I")
    viewer.add_labels(labels(), name="L")
    C.pump(10)
    path = TMP / "grow.yaml"
    path.write_text("thresholds: 100\nupper_thresholds: 200\ndilation_steps: 3\n")
    tab.import_yaml(str(path))
    C.pump(10)
    rows = table(tab.main_param_widget)
    result(rows == [[100, 200, 3]], f"{tab.params_status.text()} {rows}")


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
    if mapping.scheme != before:
        result(False, f"scheme after loading any JSON: {mapping.scheme!r}")
    path.write_text("not json")
    mapping.load_scheme()
    result(mapping.scheme == before and mapping.status.text().endswith("list of class names."), mapping.status.text())


def load_scheme_reads_good_files(viewer):
    viewer.add_labels(labels(), name="L")
    _, mapping = viewer.window.add_plugin_dock_widget("napari-sprout", "Label Mapping")
    C.pump(10)
    path = TMP / "scheme.json"
    DIALOGS.open_path = str(path)
    path.write_text('["head", "body"]')
    mapping.load_scheme()
    if mapping.scheme != ["head", "body"]:
        result(False, f"a saved scheme: {mapping.scheme!r}")
    path.write_text('{"scheme": ["leg"], "label2class": {"1": "leg"}}')
    mapping.load_scheme()
    result(mapping.scheme == ["leg"] and "not loaded" in mapping.status.text(),
           f"an exported mapping: {mapping.scheme!r}, {mapping.status.text()}")


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


def grow_export_writes_the_paths(viewer):
    image, seeds = TMP / "img.tif", TMP / "seeds.tif"
    tifffile.imwrite(image, np.zeros((8, 8, 8), np.uint8))
    tifffile.imwrite(seeds, labels())
    tab = sprout_tab(viewer, "SeedGrowthWidget")
    viewer.open(str(image), plugin="napari")
    viewer.open(str(seeds), plugin="napari", layer_type="labels")
    C.pump(10)
    tab.image_combo.setCurrentText("img")
    tab.seeds_combo.setCurrentText("seeds")
    out = TMP / "export.yaml"
    tab.export_yaml(str(out))
    written = yaml.safe_load(open(out, encoding="utf-8"))
    result(Path(written["img_path"]) == image and Path(written["seg_path"]) == seeds, str(written))


def export_then_import_finds_the_layer(viewer):
    path = TMP / "img.tif"
    tifffile.imwrite(path, np.zeros((8, 8, 8), np.uint8))
    tab = seed_tab(viewer)
    viewer.add_image(np.zeros((8, 8, 8), np.uint8), name="other")
    viewer.open(str(path), plugin="napari")
    C.pump(10)
    tab.image_combo.setCurrentText("img")
    out = TMP / "export.yaml"
    tab.export_yaml(str(out))
    tab.image_combo.setCurrentText("other")
    tab.import_yaml(str(out))
    C.pump(10)
    result(tab.image_combo.currentText() == "img", f"{tab.params_status.text()} [{tab.image_combo.currentText()}]")


def export_of_an_unsaved_layer_warns(viewer):
    tab = seed_tab(viewer)
    viewer.add_image(np.zeros((8, 8, 8), np.uint8), name="in_memory")
    C.pump(10)
    out = TMP / "export.yaml"
    tab.export_yaml(str(out))
    written = yaml.safe_load(open(out, encoding="utf-8"))["img_path"]
    level, text = tab.params_status.level, tab.params_status.text()
    tab.import_yaml(str(out))
    C.pump(10)
    result(written == "in_memory" and level == "warn" and "in_memory" in text
           and tab.image_combo.currentText() == "in_memory", f"{written}: {level}: {text}")


if __name__ == "__main__":
    globals()[sys.argv[1]](napari.Viewer(show=False))
