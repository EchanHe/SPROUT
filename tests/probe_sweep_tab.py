"""
Sweep in the Seeds tab, one case per run, for test_sweep_tab.py.

    python tests/probe_sweep_tab.py <name>

Exit code 0: it works. 1: it does not.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

import _docks_common as C

ROOT = C.pick_root(sys.argv)

import napari  # noqa: E402
import numpy as np  # noqa: E402

import napari_sprout.widgets.prompt_tab as PT  # noqa: E402
import napari_sprout.widgets.seed_widget as SWM  # noqa: E402
from napari_sprout.utils import status as S  # noqa: E402

PT.SETTINGS = ("SPROUT-checks", "napari-sprout-checks")      # not the user's
DIALOGS = C.Dialogs().install()
TMP = Path(tempfile.mkdtemp(prefix="sprout_sweep_tab_"))
SCALE, TRANSLATE = (2.0, 1.0, 1.0), (5.0, 0.0, 0.0)
LAYER = "I_FINAL_sweep_seed"
FINAL = TMP / "I" / "FINAL_sweep_seed.tif"


def result(ok, why):
    print(("works: " if ok else "FAILS: ") + why, flush=True)
    os._exit(0 if ok else 1)                       # the answer is known: no Qt teardown


def two_slabs():
    """A slab at 200 and a slab at 150: one region above 180, two above 100."""
    img = np.zeros((8, 8, 8), np.uint8)
    img[1:3] = 200
    img[5:7] = 150
    return img


def two_cubes_bridged():
    """Two cubes joined by a bridge one voxel thick: one region, two after one erosion."""
    img = np.zeros((12, 12, 24), np.uint8)
    img[3:9, 3:9, 2:10] = 200
    img[3:9, 3:9, 14:22] = 200
    img[6, 6, 10:14] = 200
    return img


def sweep_tab(viewer, img, thresholds, erosion_steps=0):
    _, sprout = viewer.window.add_plugin_dock_widget("napari-sprout", "SPROUT")
    C.pump(10)
    tab = next(w for w in sprout.findChildren(object) if type(w).__name__ == "SeedGenerationWidget")
    viewer.add_image(img, name="I", scale=SCALE, translate=TRANSLATE)
    C.pump(10)
    tab.image_combo.setCurrentText("I")
    params = tab.main_param_widget
    params.seed_method_combo.setCurrentText("Sweep (Thresholds)")
    params.populate_thresholds_from_list(thresholds, [255] * len(thresholds))
    params.erosion_spin.setValue(erosion_steps)
    params.thread_spin.setValue(1)
    tab.output_folder_line.setText(str(TMP))
    return tab


def run(tab):
    tab.generate_seeds()
    C.wait(lambda: tab.generate_btn.isEnabled() and not tab.run_control.running, 120, "sweep run")
    C.pump(10)


def one_id_per_slab(seeds):
    """Each slab is one seed, the two seeds differ, nothing else is a seed."""
    top, bottom = np.unique(seeds[1:3]), np.unique(seeds[5:7])
    rest = np.concatenate([seeds[0:1], seeds[3:5], seeds[7:]]).ravel()
    return len(top) == 1 and len(bottom) == 1 and 0 not in (top[0], bottom[0]) and top[0] != bottom[0] \
        and not rest.any()


def strict_to_loose_thresholds(viewer):
    tab = sweep_tab(viewer, two_slabs(), [180, 100])
    greyed = not tab.advanced_params_box.no_split_spin.isEnabled() and tab.advanced_params_box.min_size_spin.isEnabled()
    run(tab)
    layer = viewer.layers[LAYER] if LAYER in viewer.layers else None
    if layer is None:
        result(False, f"no {LAYER}: {tab.results_label.text()!r}, layers {[l.name for l in viewer.layers]}")
    same_place = tuple(layer.scale) == SCALE and tuple(layer.translate) == TRANSLATE
    result(greyed and same_place and FINAL.is_file() and one_id_per_slab(layer.data) and len(viewer.layers) == 2
           and S.level_of(tab.results_label) == "ok",
           f"greyed {greyed}, scale {tuple(layer.scale)}, translate {tuple(layer.translate)}, final {FINAL.is_file()}, "
           f"ids {np.unique(layer.data)}, layers {[l.name for l in viewer.layers]}, {tab.results_label.text()!r}")


def loose_to_strict_thresholds(viewer):
    tab = sweep_tab(viewer, two_slabs(), [100, 180])
    run(tab)
    seeds = viewer.layers[LAYER].data if LAYER in viewer.layers else None
    result(seeds is not None and one_id_per_slab(seeds) and FINAL.is_file(),
           f"ids {None if seeds is None else np.unique(seeds)}, {tab.results_label.text()!r}")


def erosion_sweep_from_one_row(viewer):
    tab = sweep_tab(viewer, two_cubes_bridged(), [100], erosion_steps=2)
    run(tab)
    seeds = viewer.layers[LAYER].data if LAYER in viewer.layers else None
    steps = sorted(p.name for p in (TMP / "I" / "steps").glob("step_*.tif"))
    ids = [] if seeds is None else np.unique(seeds)
    left, right = ([], []) if seeds is None else (np.setdiff1d(seeds[3:9, 3:9, 2:10], [0]),
                                                  np.setdiff1d(seeds[3:9, 3:9, 14:22], [0]))
    # the seeds meet at erosion 0, so both keep their eroded shape inside their cube
    split = seeds is not None and len(ids) == 3 and len(left) == 1 and len(right) == 1 and left[0] != right[0] \
        and np.count_nonzero(seeds) == np.count_nonzero(seeds[3:9, 3:9, 2:10]) + np.count_nonzero(seeds[3:9, 3:9, 14:22])
    result(split and len(steps) == 3 and FINAL.is_file(),
           f"ids {list(ids)}, left {left}, right {right}, steps {steps}, {tab.results_label.text()!r}")


def stop_leaves_no_final(viewer):
    real = SWM.make_sweep_adaptive_seed

    def slow(**kw):                                # time to press Stop between the steps
        progress = kw["progress"]
        kw["progress"] = lambda *a: (time.sleep(0.3), progress(*a))[1]
        return real(**kw)

    SWM.make_sweep_adaptive_seed = slow
    tab = sweep_tab(viewer, two_slabs(), [200, 180, 160, 140, 120, 100])
    bar = tab.progress_bar
    tab.generate_seeds()
    C.wait(lambda: bar.maximum() > 0 and bar.value() >= 1, 60, "first step")
    tab.stop_btn.click()
    C.wait(lambda: tab.generate_btn.isEnabled() and not tab.run_control.running, 60, "stop")
    C.pump(10)
    text, level = tab.results_label.text(), S.level_of(tab.results_label)
    result(text.startswith("Stopped after ") and level == "warn" and LAYER not in viewer.layers
           and not FINAL.is_file() and len(viewer.layers) == 1,
           f"{level}: {text!r}, layers {[l.name for l in viewer.layers]}, final {FINAL.is_file()}")


def thresholds_out_of_order_are_refused(viewer):
    tab = sweep_tab(viewer, two_slabs(), [150, 100, 200])
    run(tab)
    text, level = tab.results_label.text(), S.level_of(tab.results_label)
    result(level == "error" and "threshold" in text.lower() and LAYER not in viewer.layers and not FINAL.is_file()
           and not DIALOGS.take(),
           f"{level}: {text!r}, layers {[l.name for l in viewer.layers]}, final {FINAL.is_file()}")


def other_methods_get_their_options_back(viewer):
    tab = sweep_tab(viewer, two_slabs(), [180, 100])
    box = tab.advanced_params_box
    off = not box.no_split_spin.isEnabled()
    tab.main_param_widget.seed_method_combo.setCurrentText("Adaptive (Thresholds)")
    C.pump(5)
    result(off and box.no_split_spin.isEnabled() and box.min_size_spin.isEnabled(),
           f"sweep greys no_split {off}; back on Adaptive (Thresholds): {box.no_split_spin.isEnabled()}")


if __name__ == "__main__":
    globals()[sys.argv[1]](napari.Viewer(show=False))
