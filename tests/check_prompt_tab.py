"""
Headless check of the Prompt (AI) tab (napari_sprout/widgets/prompt_tab.py).

Drives a real napari viewer (offscreen) and the real SPROUT main widget; the model
is a fake nnInteractive session (tests/_prompt_check_utils.py) injected via
engine_factory / status_fn, so no nnInteractive is needed.
Run in the `sprout` env:
    python tests/check_prompt_tab.py
Prints PASS/FAIL per check; exits non-zero if anything failed.
"""
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.path.isdir(r"C:/Windows/Fonts"):
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:/Windows/Fonts")  # real font metrics for the width check
ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")         # status texts contain '✓'
sys.path[:0] = [str(ROOT), str(ROOT / "napari-sprout"), str(Path(__file__).resolve().parent)]

import napari  # noqa: E402
import _headless_gl  # noqa: E402,F401  (napari >= 0.7 offscreen: no GL context)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tifffile  # noqa: E402
from qtpy.QtCore import QEventLoop, Qt, QTimer  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from napari_sprout._widget import SPROUTWidget  # noqa: E402
from napari_sprout.utils import roi_layers as RL  # noqa: E402
from napari_sprout.widgets.prompt_tab import PromptTab  # noqa: E402
from napari_sprout.utils import status as S  # noqa: E402
import napari_sprout.widgets.prompt_tab as _prompt_tab  # noqa: E402
from qtpy.QtCore import QSettings  # noqa: E402

# Keep the tab's remembered settings (model folder) out of the user's real QSettings
_prompt_tab.SETTINGS = ("SPROUT-checks", "napari-sprout-checks")
QSettings(*_prompt_tab.SETTINGS).clear()
from sprout_core import prompt_points as PP  # noqa: E402
from sprout_core import prompt_seg as PS  # noqa: E402
from sprout_core import roi as R  # noqa: E402
from sprout_core.nni_engine import INSTALL_HINT, NniEngine  # noqa: E402
from _prompt_check_utils import FakeSession26  # noqa: E402

FAILED = []
TMP = Path(tempfile.mkdtemp(prefix="sprout_prompt_tab_"))


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILED.append(msg)


def wait(cond, timeout=300, what=""):
    t = time.time()
    while not cond():
        QApplication.processEvents()
        time.sleep(0.005)
        if time.time() - t > timeout:
            raise TimeoutError(what)


def run_event_loop(sec):
    """A real Qt event loop for `sec` seconds: Qt6 runs deleteLater only inside one, not in processEvents."""
    loop = QEventLoop()
    QTimer.singleShot(int(sec * 1000), loop.quit)
    loop.exec()

def pump(n=30):
    for _ in range(n):
        QApplication.processEvents()
        time.sleep(0.005)


def fake_engine(cls=NniEngine, session=FakeSession26):
    return lambda: cls(session_factory=lambda dev: session(device=dev), buffer_factory=lambda s: np.zeros(s, np.uint8))


def idle(tab):
    return not any(tab._busy.values())


img = tifffile.imread(ROOT / "demo_data/dog_img.tif")
seeds = tifffile.imread(ROOT / "demo_data/dog_seed.tif")
SCALE, TRANS = (1.5, 1.0, 1.0), (10.0, 0.0, 0.0)

viewer = napari.Viewer(show=False)
main = SPROUTWidget(viewer)
viewer.window.add_dock_widget(main)
tab = main.prompt_tab

# ---------------------------------------------------------------- 1. placement, layout, no nnInteractive
print("== 1. tab placement, layout (pipeline prompt types); without nnInteractive")
from qtpy.QtWidgets import QCheckBox, QGroupBox, QLabel, QPushButton, QRadioButton  # noqa: E402
from sprout_core import config_core  # noqa: E402

RULES = config_core.input_val_nninteractive_run
ti = main.tabs.indexOf(tab)
ok(ti == main.tabs.indexOf(main.grow_widget) + 1 and main.tabs.tabText(ti) == "3. Prompt (AI)"
   and main.tabs.indexOf(main.roi_boundary_tabs) == ti + 1, "'3. Prompt (AI)' right after Grow, 'ROI & Boundary' after it")
ok("torch" not in sys.modules and "nnInteractive" not in sys.modules, "building the panel did not import torch / nnInteractive")
ok(not tab.available and not tab.banner.isHidden() and tab.banner.text() == INSTALL_HINT,
   "nnInteractive missing (env sprout): install banner shown")
ok(not tab.load_btn.isEnabled() and not tab.run_btn.isEnabled(), "Load / Run disabled without nnInteractive")
titles = [g.title() for g in tab.findChildren(QGroupBox)]
ok(titles == ["1. Input", "2. Classes", "3. Model", "4. Prompts", "Manual points", "5. Run"], f"sections in order: {titles}")
REMOVED_ATTRS = ["min_spin", "min_btn", "ids_edit", "ids_btn", "new_btn", "only_cb", "clear_pts_btn", "update_btn",
                 "revert_btn", "overlap_combo", "margin_spin", "seed_spin", "zero_cb", "type_hint", "neg_seed_cb",
                 "output_label", "flip_btn", "type_combo", "scrib_btn", "scrib_neg_btn", "scrib_row",
                 "create_scribble_layer"]
REMOVED_TEXTS = {"Check >= N voxels", "Check ids", "+ New class", "Import CSV", "Export CSV", "Clear points of checked",
                 "Update current class", "Revert class", "Show only the current class", "Flip selected +/-",
                 "+ Scribble layer", "+ Negative scribble layer", "Drawn scribbles (+points)", "Points",
                 "Generate for checked classes"}
present = [a for a in REMOVED_ATTRS if hasattr(tab, a)]
texts = ({w.text() for w in tab.findChildren(QPushButton)} | {w.text() for w in tab.findChildren(QCheckBox)}
         | {w.text() for w in tab.findChildren(QRadioButton)})
ok(not present and not (texts & REMOVED_TEXTS), f"removed controls are gone, incl. the drawn-scribble UI "
                                                f"(attributes {present}, texts {sorted(texts & REMOVED_TEXTS)})")
ok(tab.scribble_rb.text() == "Seeds as scribbles" and tab.points_rb.text() == "Points sampled from seeds"
   and tab.scribble_rb.isChecked() and tab.prompt_mode() == "scribble", "default prompt type: (•) Seeds as scribbles")
ok(not tab.scribble_box.isHidden() and tab.points_box.isHidden()
   and tab.neg_scribble_cb.text() == "Other classes' seeds as negative scribbles"
   and tab.scribble_hint.text() == "The seeds are the scribbles: paint or erase in the seeds Labels layer to change "
                                   "them, then Run." and S.level_of(tab.scribble_hint) == "info",
   "scribble options: negative-scribble checkbox + grey hint; point options hidden")
yaml_ui = {"default_n_pos": tab.npos_spin.value(), "default_n_neg": tab.nneg_spin.value(),
           "default_method": tab.method_combo.currentText(), "negative_from_bg": tab.neg_bg_cb.isChecked(),
           "negative_from_other_classes": tab.neg_other_cb.isChecked(),
           "negative_per_other_class": tab.neg_each_cb.isChecked(),
           "use_negative_scribble": tab.neg_scribble_cb.isChecked(),
           "return_per_class_masks": tab.per_class_cb.isChecked()}
yaml_ui["seed"] = tab.base_seed
ok(yaml_ui == {k: RULES[k]["default"] for k in yaml_ui}, f"UI defaults == the YAML schema defaults: {yaml_ui}")
ok(tab.import_btn.text() == "Import YAML" and tab.export_btn.text() == "Export YAML", "Import YAML / Export YAML in 4. Prompts")
ok(tab.manual_hint.text() == "Add points to fix a class, check only that class and Run again. To change a scribble, "
                             "edit the seeds Labels layer directly." and S.level_of(tab.manual_hint) == "info",
   "Manual points: grey hint")
ok([tab.init_combo.itemText(i) for i in range(tab.init_combo.count())] == ["None"], "Initial mask: None by default")
ok(tab.per_class_cb.text() == "Also add each class as a separate layer" and not tab.per_class_cb.isChecked()
   and tab.save_cb.text() == "Save results to folder" and not tab.save_cb.isChecked(),
   "5. Run: per-class layers and Save results to folder, both off")
labels_txt = " ".join(w.text() for w in tab.findChildren(QLabel))
ok("under-segments" not in labels_txt and "combine with points" not in labels_txt,
   "no 'usually combine with points / under-segments badly' hint")
models = [tab.model_combo.itemText(i) for i in range(tab.model_combo.count())]
ok(models == ["nnInteractive"] and _prompt_tab.MODELS[0][1] is NniEngine, f"Model combo from the registry: {models}")
ok(tab.supports_label.text() == "Supports: points · scribbles · initial mask"
   and tab.supports_label.text() == "Supports: " + _prompt_tab.capabilities_text(NniEngine.capabilities)
   and not tab.engine.loaded and "torch" not in sys.modules,
   f"'{tab.supports_label.text()}' from the engine class, nothing loaded, no torch")
ok(_prompt_tab.OVERLAP_RULE == "min_id", "overlap rule: 'min_id' (smaller id wins), as the scripts")
ok(tab.output_edit.text() == "", "output name empty without seeds")

il = viewer.add_image(img, name="dog_img", scale=SCALE, translate=TRANS)
sl = viewer.add_labels(seeds, name="dog_seed", scale=SCALE, translate=TRANS)
wait(lambda: tab.table.rowCount() == 4 and tab.image_layer is il and tab.seeds_layer is sl, what="table")
st = PP.class_stats(seeds)
ok(tab.class_ids() == [3, 4, 5, 6] and all(tab._stats[c] == v for c, v in zip(st.class_id, st.voxels)),
   f"class table: ids {tab.class_ids()} with seed voxel counts")
ok(tab.output_edit.text() == "dog_seed_nni" and [tab.init_combo.itemText(i) for i in range(tab.init_combo.count())]
   == ["None", "dog_seed"], "output 'dog_seed_nni'; Initial mask lists the Labels layers")

# width: the main panel at 500 px needs no horizontal scrolling in this tab (both modes)
main.tabs.setCurrentWidget(tab)
main.resize(500, 900)
needs = []
for mode in ("scribble", "point"):
    tab.set_prompt_mode(mode)
    pump(60)
    needs.append(tab.scroll.widget().minimumSizeHint().width())
tab.set_prompt_mode("scribble")
# a 500 px panel leaves ~460 px for the tab's content (panel margins + the vertical scrollbar)
if os.path.isdir(r"C:/Windows/Fonts"):               # widths only mean something with the real font
    ok(max(needs) <= 460 and tab.scroll.horizontalScrollBar().maximum() == 0,
       f"fits a 500 px panel: content needs {needs} px (scribbles / points, <= 460), no horizontal scrollbar")
else:
    print(f"SKIP fits a 500 px panel: needs the Windows fonts (content needs {needs} px)")

# filters
tab.none_btn.click()
ok(tab.checked_classes() == [], "None")
tab.all_btn.click()
ok(tab.checked_classes() == [3, 4, 5, 6], "All")
r = tab._row_of(5)
tab.table.item(r, 0).setCheckState(Qt.Unchecked)
ok(tab.checked_classes() == [3, 4, 6], "unticking a row unchecks the class")
tab.all_btn.click()

# points mode
tab.set_prompt_mode("point")
ok(tab.scribble_box.isHidden() and not tab.points_box.isHidden() and tab.gen_btn.text() == "Generate points"
   and [tab.neg_bg_cb.text(), tab.neg_other_cb.text(), tab.neg_each_cb.text()] == ["background", "other classes",
                                                                                   "each other class"],
   "Points mode: Positive / method / Negative, 'Negatives from' background / other classes / each other class, "
   "[Generate points]")
ok(tab.neg_slow_label.isHidden(), "no 'slow' note with 0 negatives")
tab.nneg_spin.setValue(2)
ok(not tab.neg_slow_label.isHidden() and tab.neg_slow_label.text() == "(slow on big images)"
   and S.level_of(tab.neg_slow_label) == "info", "'(slow on big images)' in grey when negatives > 0")
tab.nneg_spin.setValue(0)

# Generate
tab.npos_spin.setValue(3)
tab.generate()
wait(lambda: idle(tab) and tab.points_layer() is not None, what="generate")
pts = tab.points_layer()
ref = PP.seeds_to_point_prompts(seeds, img, rng=np.random.RandomState(0), default_n_pos=3, image_name="dog_img",
                                log=lambda m: None)
got = tab.prompts_df()
ok(pts.name == "dog_seed_prompts" and got.equals(ref), f"Generate -> 'dog_seed_prompts' == seeds_to_point_prompts ({len(got)} points)")
ok(np.allclose(pts.scale, SCALE) and np.allclose(pts.translate, TRANS), "points layer has the image's scale / translate")
ok(list(pts.features.columns) == ["class_id", "label", "point_type", "sample_method"] and
   pts.features.class_id.dtype == np.int64 and pts.features.label.dtype == bool, "features class_id / label / point_type / sample_method")
ok(np.allclose(pts.face_color[0], [0, 1, 0, 1]), "positive points are green")
row = tab._row_of(4)
ok(tab.table.item(row, 3).text() == "3 / 0", f"table shows points + / - ({tab.table.item(row, 3).text()})")
ok(np.asarray(pts.shown).all(), "every point is shown after Generate")
tab.neg_each_cb.setChecked(True)
tab.nneg_spin.setValue(1)
tab.set_checked([4])
tab.generate()
wait(lambda: idle(tab), what="generate each")
g4 = tab.prompts_df()
g4 = g4[g4.class_id == 4].reset_index(drop=True)
ref4 = PP.seeds_to_point_prompts(seeds, img, classes=[4], rng=np.random.RandomState(0), default_n_pos=3,
                                 default_n_neg=1, negative_per_other_class=True, image_name="dog_img", log=lambda m: None)
ok(g4.equals(ref4) and set(g4.point_type) >= {"negative_class3", "negative_class5", "negative_class6"},
   "the 'each other class' option is passed to the sampling (negative_per_other_class)")
tab.neg_each_cb.setChecked(False)
tab.nneg_spin.setValue(0)
tab.npos_spin.setValue(3)
tab.set_checked([4])
tab.generate()
wait(lambda: idle(tab), what="generate back")
tab.all_btn.click()

# highlighted class -> feature defaults; new points
tab.select_class(5)
pump()
ok(tab._current == 5 and pts.feature_defaults["class_id"][0] == 5 and bool(pts.feature_defaults["label"][0]),
   "highlighting class 5 sets feature_defaults (class 5, positive)")
tab.neg_rb.setChecked(True)
n0 = len(pts.data)
pts.selected_data = {0}                       # napari copies the selected point's features into the defaults
pump()                                        # ... the tab puts its own back
pts.add([[300, 80, 60]])                      # data coordinates of the layer
wait(lambda: tab.table.item(tab._row_of(5), 3).text() == "3 / 1", what="new point counted")
f = pts.features
ok(len(pts.data) == n0 + 1 and f.class_id.iloc[-1] == 5 and not f.label.iloc[-1] and f.sample_method.iloc[-1] == "manual"
   and f.class_id.iloc[0] == ref.class_id.iloc[0], "New points: negative -> a new point gets class 5 / negative / manual; "
                                                   "the selected point is unchanged")
ok(np.asarray(pts.shown)[-1] and np.allclose(pts.face_color[-1], [1, 0, 1, 1]), "new point shown and magenta")
df_now = tab.prompts_df()
ok(df_now.iloc[-1][["z", "y", "x"]].tolist() == [300, 80, 60], "prompts_df gives voxel coordinates (data coords of the layer)")
tab.pos_rb.setChecked(True)

# Generate again for class 5 keeps the manual point
tab.set_checked([5])
tab.npos_spin.setValue(2)
tab.generate()
wait(lambda: idle(tab), what="generate 2")
d2 = tab.prompts_df()
c5 = d2[d2.class_id == 5]
ok(len(c5) == 3 and (c5.sample_method == "manual").sum() == 1 and c5.iloc[-1].sample_method == "manual"
   and len(d2[d2.class_id == 4]) == 3, "re-Generate for class 5: generated points replaced, the manual one kept (last)")

# CSV: kept as methods (no buttons)
p = TMP / "p.csv"
n = tab.export_csv(p)
tab.clear_points_of_checked()
ok(len(tab.prompts_df()[lambda d: d.class_id == 5]) == 0, "clear_points_of_checked (method kept)")
tab.import_csv(p)
back = tab.prompts_df()
ok(n == len(d2) and back.drop(columns="image_name").equals(d2.drop(columns="image_name")), "CSV export -> import round trip (methods kept)")
tab.all_btn.click()

tab.select_class(6)
tab.start_adding_points()
ok(viewer.layers.selection.active is pts and pts.mode == "add" and pts.feature_defaults["class_id"][0] == 6,
   "Add points: prompts layer active in add mode for the highlighted class 6")
pts.mode = "pan_zoom"
ok(tab.run_checked() is None and tab.output_name() not in viewer.layers, "Run does nothing without nnInteractive")
tab.set_prompt_mode("scribble")

# ---------------------------------------------------------------- 2. fake engine: the pipeline's logic
print("== 2. with a (fake) model: scribbles, negative scribbles, initial mask, points, outputs")
main.tabs.setCurrentWidget(main.grow_widget)
ft = PromptTab(viewer, engine_factory=fake_engine(), status_fn=lambda: (True, "fake"))
wait(lambda: ft.table.rowCount() >= 4, what="ft table")
ok(ft.banner.isHidden() and ft.load_btn.isEnabled() and ft.run_btn.isEnabled() and not ft.engine.loaded,
   "model available: no banner, Load enabled, Run clickable before loading")


def orange_labels(t):
    """Visible labels with text in the 'act now' colours (orange / red)."""
    out = []
    for w in t.findChildren(QLabel):
        css = w.styleSheet().lower()
        if w.text() and not w.isHidden() and any(c in css for c in ("#e08a00", "#d08000", "#d9534f", "orange", "red",
                                                                    S.WARN, S.ERROR)):
            out.append(w.text())
    return out


ft.device_combo.setCurrentIndex(ft.device_combo.findData("cuda"))
pump()
ok(orange_labels(ft) == [], f"default state with GPU: no orange / red text ({orange_labels(ft)})")
ft.run_btn.click()
pump()
ok(ft.run_status.text() == "Load the model first (3. Model)" and S.level_of(ft.run_status) == "error"
   and "dog_seed_nni" not in viewer.layers, "Run before Load: red 'Load the model first (3. Model)' (it blocks), nothing runs")
ft.model_edit.setText(str(TMP))
ft.device_combo.setCurrentIndex(ft.device_combo.findData("cpu"))
ok(ft.cpu_label.text() == "very slow: minutes per class" and S.level_of(ft.cpu_label) == "warn",
   "CPU device: orange 'very slow: minutes per class'")
ft.load_model()
wait(lambda: idle(ft) and ft.engine.loaded, what="load")
ok(ft.engine.loaded and "on cpu" in ft.model_status.text() and ft.run_btn.isEnabled() and ft.run_status.text() == "",
   f"Load: {ft.model_status.text()!r}; the 'load first' note is cleared")
ft.device_combo.setCurrentIndex(ft.device_combo.findData("cuda"))
ok(orange_labels(ft) == [], f"loaded, GPU selected: no orange / red text ({orange_labels(ft)})")


def core_run(df, classes, rule="min_id", **kw):
    eng = fake_engine()()
    eng.load("x", "cpu")
    store = PS.ResultStore()
    roi = kw.pop("roi", None)
    crop = PS.inference_crop(seeds.shape, roi, kw.pop("margin", 32))
    for r in PS.run_classes(eng, img, seeds, df, classes, crop=crop, image_key="k", roi=roi, **kw):
        if not isinstance(r, tuple):
            store.set(r)
    return store, store.compose(seeds.shape, rule=rule, seeds=seeds)


def descending_overwrite(store):
    """nninter_predict's total: classes painted in descending id (smaller id wins)."""
    tot = np.zeros(seeds.shape, PS._label_dtype(max(store.classes())))
    full = PS.inference_crop(seeds.shape)
    for c in sorted(store.classes(), reverse=True):
        tot[store.get(c).full_mask_in(full)] = c
    return tot


ft.set_prompts(PP.empty_points_df())          # no manual points: exactly the YAML's scribble run
ft.set_checked([3, 4, 5, 6])
ok(ft.prompt_mode() == "scribble", "the fake-model tab starts in Seeds as scribbles")
ft.run_checked()
wait(lambda: idle(ft), what="scribble run")
out = viewer.layers["dog_seed_nni"]
store_ref, comp_ref = core_run(PP.empty_points_df(), [3, 4, 5, 6], prompt_type="scribble")
# the layer is at least uint16 (a brushed id > 255 would wrap in uint8); the saved file keeps the smallest dtype
ok(np.array_equal(out.data, comp_ref) and out.data.dtype == np.uint16 and comp_ref.dtype == np.uint8
   and np.array_equal(out.data, descending_overwrite(store_ref)),
   f"Seeds as scribbles -> 'dog_seed_nni' == run_classes(prompt_type='scribble', the seeds) == descending overwrite "
   f"(min_id); layer {out.data.dtype} (core {comp_ref.dtype})")
meta = out.metadata.get("sprout_prompt", {})
ok(sorted(meta.get("classes", {})) == [3, 4, 5, 6] and meta.get("overlap_rule") == "min_id" and meta.get("roi_margin") == 32
   and meta.get("prompt_type") == "scribble" and meta.get("config", {}).get("prompt_type") == "scribble"
   and "sprout_roi" not in out.metadata, f"metadata['sprout_prompt']: scribble, rule 'min_id', margin 32, config")
ok(np.allclose(out.scale, SCALE) and np.allclose(out.translate, TRANS), "output has the image's scale / translate")
ok(all("done" in ft._class_status[c] for c in (3, 4, 5, 6)) and ft.table.item(ft._row_of(4), 5).text() != "",
   "class status / result voxels in the table")

ft.neg_scribble_cb.setChecked(True)
n_calls = len(ft.engine.session.calls)
ft.run_checked()
wait(lambda: idle(ft), what="neg scribble run")
_, comp_neg = core_run(PP.empty_points_df(), [3, 4, 5, 6], prompt_type="scribble", neg_scribble=True)
neg_calls = [c for c in ft.engine.session.calls[n_calls:] if c[0] == "scribble" and c[2] is False]
ok(np.array_equal(out.data, comp_neg) and len(neg_calls) == 4,
   "Other classes' seeds as negative scribbles -> == run_classes(neg_scribble=True); one negative scribble per "
   "class (use_negative_scribble)")
ft.neg_scribble_cb.setChecked(False)

from scipy.ndimage import grey_dilation  # noqa: E402

grown = grey_dilation(seeds, size=(3, 7, 7))
gl = viewer.add_labels(grown, name="dog_grown", scale=SCALE, translate=TRANS)
wait(lambda: ft.init_combo.findData("dog_grown") >= 0, what="init combo")
ft.init_combo.setCurrentIndex(ft.init_combo.findData("dog_grown"))
ft.run_checked()
wait(lambda: idle(ft), what="init run")
_, comp_init = core_run(PP.empty_points_df(), [3, 4, 5, 6], prompt_type="scribble", init_masks=grown)
ok(np.array_equal(out.data, comp_init) and not np.array_equal(comp_init, comp_ref)
   and out.metadata["sprout_prompt"]["initial_mask"] == "dog_grown",
   "Initial mask 'dog_grown' -> == run_classes(init_masks=grown) (init_seg_path)")
ft.init_combo.setCurrentIndex(0)

# points mode
df_run = PP.seeds_to_point_prompts(seeds, img, rng=np.random.RandomState(1), default_n_pos=3, default_n_neg=1,
                                   negative_from_bg=False, image_name="dog_img", log=lambda m: None)
ft.set_prompts(df_run)
ok(ft.prompts_df().equals(df_run), "set_prompts / prompts_df round trip (shared prompts layer)")
ft.set_prompt_mode("point")
ft.run_checked()
wait(lambda: idle(ft), what="points run")
store_pts, comp_pts = core_run(df_run, [3, 4, 5, 6])
ok(np.array_equal(out.data, comp_pts) and np.array_equal(out.data, descending_overwrite(store_pts)),
   "Points sampled from seeds -> == run_classes(points) == descending overwrite (min_id)")

# points generated in the points mode stay in the layer but are not sent in scribble mode
ft.set_prompt_mode("scribble")
ft.set_checked([4])
ft.run_checked()
wait(lambda: idle(ft), what="scribble, generated points left")
full = PS.inference_crop(seeds.shape)
st_s, _ = core_run(PP.empty_points_df(), [4], prompt_type="scribble")
got = ft._store_for_current_output(create=False).get(4)
ok(np.array_equal(got.full_mask_in(full), st_s.get(4).full_mask_in(full)) and got.n_points == 0
   and len(ft.points_layer().data) == len(df_run)
   and f"{len(df_run)} generated points not used (scribble mode uses manual points only)" in ft.run_status.text(),
   f"Seeds as scribbles ignores generated points: == the scribble alone, layer unchanged ({ft.run_status.text()})")
ok(ft.run_points("scribble")[1] == len(df_run) and ft.run_points("points")[0].equals(df_run),
   "run_points: scribble mode leaves out generated points; points mode sends all")

# manual points in scribble mode: added after the scribble
df_man = df_run.copy()
df_man["sample_method"] = "manual"
ft.set_prompts(df_man)
ft.run_checked()
wait(lambda: idle(ft), what="scribble + manual points")
st_sp, _ = core_run(df_man, [4], prompt_type="scribble")
got = ft._store_for_current_output(create=False).get(4)
ok(np.array_equal(got.full_mask_in(full), st_sp.get(4).full_mask_in(full)) and got.n_points == 4
   and "not used" not in ft.run_status.text(),
   "Seeds as scribbles + manual points of the class: the points follow the scribble (as nninter_predict)")
ft.set_prompts(df_run)
ft.set_checked([3, 4, 5, 6])

# per-class layers + saving
ft.set_prompt_mode("point")
ft.per_class_cb.setChecked(True)
ft.save_cb.setChecked(True)
save_dir = TMP / "saved"
ft.folder_edit.setText(str(save_dir))
ft.run_checked()
wait(lambda: idle(ft), what="per class + save")
cls_layers = {c: viewer.layers[f"dog_seed_nni_class_{c}"] for c in (3, 4, 5, 6) if f"dog_seed_nni_class_{c}" in viewer.layers}
ok(len(cls_layers) == 4 and all(np.array_equal(lay.data > 0, store_pts.get(c).full_mask_in(full))
                                and set(np.unique(lay.data)) == {0, c} and np.allclose(lay.scale, SCALE)
                                and np.allclose(lay.translate, TRANS) for c, lay in cls_layers.items()),
   "Also add each class as a separate layer: 'dog_seed_nni_class_<id>' = the class's own mask (value id), "
   "image scale / translate")
files = sorted(p.name for p in save_dir.iterdir()) if save_dir.exists() else []
exp_files = sorted(["dog_img_segmentation.tif", "used_config.yaml", "dog_img_point_prompts.csv"]
                   + [f"dog_img_class_{c}.tif" for c in (3, 4, 5, 6)])
ok(files == exp_files, f"Save results to folder: the pipeline's files {files}")
if files == exp_files:
    seg = tifffile.imread(save_dir / "dog_img_segmentation.tif")
    c4 = tifffile.imread(save_dir / "dog_img_class_4.tif")
    import yaml as _yaml  # noqa: E402

    ucfg = _yaml.safe_load(open(save_dir / "used_config.yaml"))
    csv = pd.read_csv(save_dir / "dog_img_point_prompts.csv")
    ok(np.array_equal(seg, out.data) and seg.dtype == np.uint8 and out.data.dtype == np.uint16 and c4.dtype == np.uint8
       and set(np.unique(c4)) == {0, 255} and np.array_equal(c4 > 0, store_pts.get(4).full_mask_in(full))
       and ucfg["prompt_type"] == "point" and ucfg["return_per_class_masks"] is True and len(csv) == len(df_run),
       "files: total (smallest label dtype, the layer uint16) == the layer, class tif uint8 x255, used_config.yaml (prompt_type point), "
       "point CSV")
ft.per_class_cb.setChecked(False)
ft.save_cb.setChecked(False)

# Update / Revert: kept as methods (no buttons)
before = np.array(out.data)
old4 = ft._store_for_current_output(create=False).get(4)
others = {c: ft._store_for_current_output(create=False).get(c) for c in (3, 5, 6)}
d_up = pd.concat([ft.prompts_df(), pd.DataFrame([dict(image_name="dog_img", class_id=4, x=60, y=120, z=430, label=False,
                                                      point_type="negative", sample_method="manual")])],
                 ignore_index=True)
ft.set_prompts(d_up)
ft.select_class(4)
ft.update_current()
wait(lambda: idle(ft), what="update")
st4 = ft._store_for_current_output(create=False)
new4 = st4.get(4)
_, comp_scratch = core_run(d_up, [3, 4, 5, 6])
ok(new4.mode in ("refine_cold", "refine_warm") and all(st4.get(c) is others[c] for c in others)
   and np.array_equal(out.data, comp_scratch), f"update_current (method kept): class 4 {new4.mode}, others untouched")
ft.revert_current()
pump()
ok(np.array_equal(out.data, before) and st4.get(4) is old4, "revert_current (method kept): back to before")

# custom output name
ft.output_edit.setText("my_output")
ft.output_edit.textEdited.emit("my_output")
ft.set_checked([6])
ft.run_checked()
wait(lambda: idle(ft), what="custom output")
ok("my_output" in viewer.layers and np.array_equal(viewer.layers["my_output"].data > 0,
                                                     store_pts.get(6).full_mask_in(full)),
   "Output field: a typed name is used for the run")
ft.output_edit.setText("")
ft.output_edit.textEdited.emit("")
ok(ft.output_edit.text() == "dog_seed_nni", "clearing the field goes back to the automatic name")
ft.set_checked([3, 4, 5, 6])


# Stop
class SlowSession(FakeSession26):
    def _predict(self):
        # long enough for the Stop (sent when the first result reaches the tab) to land
        # before the second class finishes; 0.05 s was flaky
        time.sleep(0.5)
        super()._predict()


st_tab = PromptTab(viewer, engine_factory=fake_engine(session=SlowSession), status_fn=lambda: (True, "fake"))
wait(lambda: st_tab.table.rowCount() >= 4, what="stop tab")
st_tab.model_edit.setText("x")
st_tab.load_model()
wait(lambda: idle(st_tab) and st_tab.engine.loaded, what="load slow")
st_tab.set_checked([3, 4, 5, 6])
st_tab.output_edit.setText("stop_test_out")              # its own layer (another tab on this viewer)
st_tab.output_edit.textEdited.emit("stop_test_out")
st_tab.output_updated.connect(lambda name: st_tab.stop())
st_tab.run_checked()
wait(lambda: idle(st_tab), what="stopped run")
sstore = st_tab._store_for_current_output(create=False)
ok(len(sstore) == 1 and st_tab.run_status.text().startswith("Stopped after 1 of 4") and
   sum(v == "stopped" for v in st_tab._class_status.values()) == 3,
   f"Stop after the first class: 1 result kept, 3 stopped ({st_tab.run_status.text()})")
ok(st_tab.run_btn.isEnabled() and not st_tab.stop_btn.isEnabled(), "buttons back to idle after Stop")

# ---------------------------------------------------------------- 3. capability gating
print("== 3. capabilities")


class SamLike(NniEngine):
    name = "SAM-like"
    capabilities = frozenset({"points", "bbox"})


sam_tab = PromptTab(viewer, engine_factory=fake_engine(cls=SamLike), status_fn=lambda: (True, "fake"))
ok(not sam_tab.scribble_rb.isEnabled() and sam_tab.points_rb.isChecked() and sam_tab.prompt_mode() == "point"
   and sam_tab.scribble_rb.toolTip() == "SAM-like does not support scribbles",
   f"engine without scribble: 'Seeds as scribbles' disabled ({sam_tab.scribble_rb.toolTip()!r}); Points is the fallback")
ok(not sam_tab.init_combo.isEnabled() and sam_tab.init_combo.toolTip() == "SAM-like does not support an initial mask",
   "...and the Initial mask combo is disabled with the reason")
ok(sam_tab.supports_label.text() == "Supports: points · boxes", f"its Supports line: {sam_tab.supports_label.text()!r}")
reg = PromptTab(viewer, models=[("SAM-like", SamLike, lambda: (True, "x"))])
ok(reg.supports_label.text() == "Supports: points · boxes" and not reg.engine.loaded
   and [reg.model_combo.itemText(i) for i in range(reg.model_combo.count())] == ["SAM-like"],
   "a registry entry (class) gives the combo label and the Supports line without loading")
reg._finalizer()
ok(ft.type_enabled("seed_scribble") and ft.type_enabled("points") and ft.type_enabled("init_mask"),
   "nnInteractive engine: scribbles, points and the initial mask enabled")

# ---------------------------------------------------------------- 4. YAML (the CLI config)
print("== 4. Import / Export YAML (sprout.py --prompt --nninteractive)")
import yaml  # noqa: E402

D = ROOT / "demo_data"
v4 = napari.Viewer(show=False)
v4.open(str(D / "dog_img.tif"))
v4.open(str(D / "dog_seed.tif"), layer_type="labels")
init_path = TMP / "dog_grown.tif"
tifffile.imwrite(init_path, grown)
v4.open(str(init_path), layer_type="labels")
yt = PromptTab(v4, engine_factory=fake_engine(), status_fn=lambda: (True, "fake"))
wait(lambda: yt.table.rowCount() == 4, what="yaml tab")
yt.model_edit.setText(str(TMP))
yt.device_combo.setCurrentIndex(yt.device_combo.findData("cuda"))
yt.set_prompt_mode("point")
yt.npos_spin.setValue(4)
yt.nneg_spin.setValue(2)
yt.method_combo.setCurrentText("grid")
yt.neg_bg_cb.setChecked(False)
yt.neg_each_cb.setChecked(True)
yt.neg_scribble_cb.setChecked(True)
yt.per_class_cb.setChecked(True)
yt.init_combo.setCurrentIndex(yt.init_combo.findData("dog_grown"))
yt.folder_edit.setText(str(TMP / "yaml_out"))
yfile = TMP / "exported.yaml"
cfg = yt.export_yaml(str(yfile))
exp_cfg = {"model_path": str(TMP), "device": "cuda:0", "img_path": str(D / "dog_img.tif").replace("\\", "/"),
           "seg_path": str(D / "dog_seed.tif").replace("\\", "/"), "output_folder": str(TMP / "yaml_out"),
           "init_seg_path": str(init_path).replace("\\", "/"), "prompt_type": "point", "default_n_pos": 4,
           "default_n_neg": 2, "default_method": "grid", "negative_from_bg": False,
           "negative_from_other_classes": True, "negative_per_other_class": True, "use_negative_scribble": True,
           "return_per_class_masks": True, "seed": 0}
loaded = yaml.safe_load(open(yfile))
norm = {k: (str(v).replace("\\", "/") if isinstance(v, str) else v) for k, v in loaded.items()}
ok(norm == {k: (str(v).replace("\\", "/") if isinstance(v, str) else v) for k, v in exp_cfg.items()},
   f"Export YAML: the pipeline's keys and values ({sorted(loaded)})")
ok(S.level_of(yt.yaml_status) == "ok", f"export with real file paths: no note ({yt.yaml_status.text()!r})")
try:
    with __import__("contextlib").redirect_stdout(__import__("io").StringIO()):
        opt = config_core.validate_input_yaml(dict(loaded), RULES)
    ok(opt["default_n_pos"] == 4 and opt["init_seg_path"] == loaded["init_seg_path"],
       "the exported YAML passes config_core.validate_input_yaml (the CLI's loader)")
except ValueError as e:
    ok(False, f"the exported YAML passes config_core.validate_input_yaml: {e}")
# round trip into a fresh tab
yt2 = PromptTab(v4, engine_factory=fake_engine(), status_fn=lambda: (True, "fake"))
wait(lambda: yt2.table.rowCount() == 4, what="yaml tab 2")
yt2.import_yaml(str(yfile))
pump()
back = yt2.yaml_config()[0]
ok({k: (str(v).replace("\\", "/") if isinstance(v, str) else v) for k, v in back.items()} == norm
   and yt2.init_layer is not None and yt2.init_layer.name == "dog_grown" and yt2.save_cb.isChecked()
   and S.level_of(yt2.yaml_status) == "ok",
   "Export -> Import round trip: every option, the initial mask layer (by its file), output folder + save on")
# the YAML seed is the tab's base seed (sampling + the model's seed + class id)
with open(TMP / "seed7.yaml", "w") as fh:
    yaml.safe_dump(dict(loaded, seed=7), fh)
yt2.import_yaml(str(TMP / "seed7.yaml"))
ok(yt2.base_seed == 7 and yt2.yaml_config()[0]["seed"] == 7, "YAML 'seed: 7' -> base seed 7, exported again")
yt2.set_checked([4])
yt2.generate()
wait(lambda: idle(yt2), what="seed 7 generate")
g7 = yt2.prompts_df()
r7 = PP.seeds_to_point_prompts(seeds, img, classes=[4], rng=np.random.RandomState(7), default_n_pos=4, default_n_neg=2,
                               default_method="grid", negative_from_bg=False, negative_per_other_class=True,
                               image_name=yt2.image_layer.name, log=lambda m: None)
ok(g7[g7.class_id == 4].reset_index(drop=True).equals(r7), "Generate samples with RandomState(seed)")
yt2.set_checked([3, 4, 5, 6])
# a YAML whose files are not open: selection unchanged, orange note
other = dict(loaded, img_path=str(TMP / "nope.tif"), prompt_type="scribble")
with open(TMP / "other.yaml", "w") as fh:
    yaml.safe_dump(other, fh)
cur_img = yt2.image_combo.currentText()
yt2.import_yaml(str(TMP / "other.yaml"))
ok(yt2.image_combo.currentText() == cur_img and yt2.prompt_mode() == "scribble" and S.level_of(yt2.yaml_status) == "warn"
   and "img_path nope.tif is not open" in yt2.yaml_status.text(),
   f"Import with a file that is not open: layer unchanged, orange note ({yt2.yaml_status.text()!r})")
# layers without files: placeholders + orange note
ft_cfg, missing = ft.yaml_config()
ok(missing == ["img_path", "seg_path"] and ft_cfg["img_path"].startswith("<path to"), f"layers without a file -> "
                                                                                      f"placeholders {missing}")
ft.export_yaml(str(TMP / "placeholder.yaml"))
ok(S.level_of(ft.yaml_status) == "warn" and "fill in img_path, seg_path" in ft.yaml_status.text(),
   f"...and an orange note ({ft.yaml_status.text()!r})")
for t_ in (yt, yt2):
    t_._finalizer()
v4.close()

# ---------------------------------------------------------------- 5. ROI
print("== 5. ROI")
hub = RL.get_roi_hub(viewer)
lo, hi = (250, 60, 30), (480, 160, 110)
ok(ft.roi_select.state() == "none" and [ft.roi_select.combo.itemText(i) for i in range(ft.roi_select.combo.count())]
   == ["None"], "1. Input: an ROI combo, None")
lroi = RL.add_roi_layer(viewer, il, lo, hi)
pump()
roi = hub.roi_of(lroi)
ok(ft.output_edit.text() == "dog_seed_nni" and ft.run_btn.text() == "Run checked"
   and ft.gen_btn.text() == "Generate points" and ft.roi_select.state() == "none",
   "a new ROI layer changes nothing in the tab (None stays picked)")
ft.roi_select.select(lroi)
pump()
ok(ft.output_name(roi) == "dog_seed_nni_ROI_1" and ft.output_edit.text() == "dog_seed_nni_ROI_1"
   and ft.roi_select.error.isHidden(), "ROI_1 picked: output name 'dog_seed_nni_ROI_1'")
ok(ft.run_btn.text() == "Run checked in ROI_1" and ft.gen_btn.text() == "Generate points",
   f"Run names the ROI, Generate does not: {ft.gen_btn.text()!r} / {ft.run_btn.text()!r}")
st_r = PP.class_stats(roi.crop(seeds), allowed=roi.mask_crop())
exp_stats = dict(zip(st_r.class_id.tolist(), st_r.voxels.tolist()))
wait(lambda: ft._stats == exp_stats, what="roi stats")
ok(ft.table.horizontalHeaderItem(2).text() == "seed voxels in ROI", "seed voxel column counts inside the ROI")
ft.set_checked([3, 4, 5, 6])
ft.set_prompt_mode("point")
ft.npos_spin.setValue(3)
ft.nneg_spin.setValue(1)
ft.generate()
wait(lambda: idle(ft), what="roi generate")
dr = ft.prompts_df()
ok(len(dr) > 0 and roi.contains(dr[["z", "y", "x"]].to_numpy()).all(), f"Generate with ROI: all {len(dr)} points inside the ROI")
ref_r = PP.seeds_to_point_prompts_in_roi(seeds, img, roi, classes=[3, 4, 5, 6], rng=np.random.RandomState(0),
                                        default_n_pos=3, default_n_neg=1, image_name="dog_img", log=lambda m: None)
gen_rows = dr[dr.sample_method != "manual"].reset_index(drop=True)
ok(gen_rows.equals(ref_r) and (dr.sample_method == "manual").sum() > 0,
   "generated rows == seeds_to_point_prompts_in_roi (negatives from other classes + background; hand-placed kept)")
ft.per_class_cb.setChecked(True)
ft.run_checked()
wait(lambda: idle(ft), what="roi run")
outr = viewer.layers["dog_seed_nni_ROI_1"]
_, comp_r = core_run(dr, [3, 4, 5, 6], roi=roi, margin=32)
inside = roi.mask_full()
ok(np.array_equal(outr.data, comp_r) and not outr.data[~inside].any() and outr.data[inside].any(),
   "ROI run: 'dog_seed_nni_ROI_1' == run_classes with the ROI (margin 32); 0 outside the ROI")
roi_cls = [n for n in (f"dog_seed_nni_class_{c}_ROI_1" for c in (3, 4, 5, 6)) if n in viewer.layers]
ok(len(roi_cls) == len([c for c in (3, 4, 5, 6) if c in exp_stats]) and len(roi_cls) > 0,
   f"per-class layers keep the ROI suffix: {roi_cls}")
ok(outr.metadata.get(R.META_KEY, {}).get("bbox_start") == list(lo) and outr.metadata[R.META_KEY].get("name") == "ROI_1"
   and outr.metadata["sprout_prompt"]["roi_margin"] == 32, "metadata: sprout_roi (+ name) + roi_margin 32")
ok(np.array_equal(viewer.layers["dog_seed_nni"].data, st4.compose(seeds.shape, rule="min_id", seeds=seeds)),
   "the whole-image output layer is left alone")
ft.per_class_cb.setChecked(False)
lother = RL.add_roi_layer(viewer, None, (0, 0, 0), (5, 5, 5), shape=(10, 20, 30))   # on another image's grid
ft.roi_select.select(lother)
pump()
shape_txt = "×".join(str(v) for v in seeds.shape)
mism = f"{lother.name} is 10×20×30; image is {shape_txt}"
ok(ft.roi_select.error.text() == mism and not ft.roi_select.error.isHidden(),
   f"ROI / image shape mismatch shown in red: {ft.roi_select.error.text()!r}")
ft.roi_select.select(None)
pump()
ok(ft.output_edit.text() == "dog_seed_nni" and ft.run_btn.text() == "Run checked"
   and ft.gen_btn.text() == "Generate points", "None again: plain texts, 'dog_seed_nni'")

# ---------------------------------------------------------------- 6. teardown
ft.engine.release()
for t in (ft, st_tab, sam_tab):
    t._finalizer()
ok(not ft.engine.loaded, "engines released (weakref.finalize)")

# ---------------------------------------------------------------- 8. plugin manifest: no 'Prompts' dock
print("== 8. plugin manifest")
from npe2 import PluginManifest, PluginManager  # noqa: E402

mf = PluginManifest.from_file(ROOT / "napari-sprout" / "napari_sprout" / "napari.yaml")
wnames = [w.display_name for w in mf.contributions.widgets]
cmds = [c.id for c in mf.contributions.commands]
ok(wnames == ["SPROUT", "Edit", "Label Mapping", "Batch Builder", "Data Info"]
   and not {"napari-sprout.prompt_widget", "napari-sprout.load_widget"} & set(cmds),
   f"napari.yaml validates (npe2); widgets {wnames}")
pm = PluginManager.instance()
pm.discover()
inst = [w.display_name for w in pm.get_manifest("napari-sprout").contributions.widgets]
ok(inst == wnames, f"the installed manifest is this file: {inst}")


def menu_texts(menu, out=None):
    out = [] if out is None else out
    for a in menu.actions():
        out.append(a.text())
        if a.menu():
            menu_texts(a.menu(), out)
    return out


plug = [a for a in viewer.window._qt_window.menuBar().actions() if "Plugin" in a.text()]
mtexts = menu_texts(plug[0].menu()) if plug else []
sub = [t for t in mtexts if t and t not in ("Install/Uninstall Plugins...", "Plugin Errors...")
       and "napari-animation" not in t and "napari-metadata" not in t   # other plugins (napari[all] 0.7)
       and not t.endswith("(napari builtins)")]      # napari 0.7 lists its own widgets here too
ok(sub == ["SPROUT", "SPROUT", "Edit", "Label Mapping", "Batch Builder", "Data Info"],
   f"Plugins -> SPROUT lists SPROUT, Edit, Label Mapping, Batch Builder, Data Info (no Prompts, no Load): {sub}")
import napari_sprout  # noqa: E402
_w = __import__("napari_sprout._widget", fromlist=["x"])
ok(all(not hasattr(napari_sprout, f) and f not in napari_sprout.__all__ and not hasattr(_w, f)
       for f in ("make_sprout_widget_prompt", "load_images_widget")),
   "make_sprout_widget_prompt / load_images_widget are no longer exported")
viewer.close()

# ---------------------------------------------------------------- 9. Classes table height follows its rows
print("== 9. Classes table height")
v9 = napari.Viewer(show=False)
img9 = (np.random.default_rng(3).random((20, 60, 80)) * 200 + 20).astype(np.uint8)
s3 = np.zeros(img9.shape, np.uint8)
for k, c in enumerate((2, 5, 9)):
    s3[2 + 5 * k: 5 + 5 * k, 10:20, 10:20] = c
s25 = np.zeros(img9.shape, np.uint8)
for c in range(1, 26):
    zz, yy = divmod(c - 1, 5)
    s25[1 + 3 * zz: 3 + 3 * zz, 2 + 11 * yy: 8 + 11 * yy, 5:15] = c
v9.add_image(img9, name="img9")
v9.add_labels(s25, name="seeds25")
v9.add_labels(s3, name="seeds3")
mw9 = SPROUTWidget(v9)
mw9.tabs.setCurrentWidget(mw9.prompt_tab)
mw9.resize(500, 1400)
mw9.show()
pt9 = mw9.prompt_tab
tbl = pt9.table


def table_fit(n_expected):
    """(height, expected height = header + rows shown + frame (+ h-scrollbar), v-scrollbar max, gap to All/None)."""
    pump(40)
    n = tbl.rowCount()
    shown = min(n, 10)
    exp = tbl.horizontalHeader().sizeHint().height() + sum(tbl.rowHeight(i) for i in range(shown)) + 2 * tbl.frameWidth()
    if tbl.horizontalScrollBar().maximum() > 0:
        exp += tbl.horizontalScrollBar().sizeHint().height()
    gap = pt9.all_btn.geometry().top() - tbl.geometry().bottom()
    return n, tbl.height(), exp, tbl.verticalScrollBar().maximum(), gap


pt9.image_combo.setCurrentText("img9")
pt9.seeds_combo.setCurrentText("seeds3")
wait(lambda: tbl.rowCount() == 3, what="3 classes")
n, h, exp, vmax, gap = table_fit(3)
ok(n == 3 and abs(h - exp) <= 2 and vmax == 0 and 0 <= gap <= 16,
   f"3 classes: table {h} px = header + 3 rows ({exp} px), no vertical scrollbar (max {vmax}), All/None {gap} px below")
pt9.seeds_combo.setCurrentText("seeds25")
wait(lambda: tbl.rowCount() == 25, what="25 classes")
n, h25, exp, vmax, gap = table_fit(25)
row_h = tbl.rowHeight(0)
ok(n == 25 and abs(h25 - exp) <= 2 and vmax > 0 and 9 * row_h < h25 - tbl.horizontalHeader().sizeHint().height() <= 11 * row_h,
   f"25 classes: height capped at ~10 rows ({h25} px, expected {exp}), scrolls inside (max {vmax}), All/None {gap} px below")
pt9.seeds_combo.setCurrentText("seeds3")
wait(lambda: tbl.rowCount() == 3, what="back to 3 classes")
n, h3b, exp, vmax, gap = table_fit(3)
ok(n == 3 and abs(h3b - exp) <= 2 and h3b < h25 and vmax == 0, f"back to 3 classes: shrinks to {h3b} px, no scrollbar")
# ROI change: fewer classes inside the ROI -> shorter table
lr9 = RL.add_roi_layer(v9, v9.layers["img9"], (0, 0, 0), (8, 60, 80))         # z 0-7: classes 2 and 5 only
pt9.roi_select.select(lr9)
wait(lambda: tbl.rowCount() == 2, what="roi 2 classes")
n, h2, exp, vmax, gap = table_fit(2)
ok(n == 2 and abs(h2 - exp) <= 2 and h2 < h3b, f"ROI with 2 classes inside: {h2} px (header + 2 rows)")
pt9.roi_select.select(None)
wait(lambda: tbl.rowCount() == 3, what="roi off")
v9.close()

# ---------------------------------------------------------------- 7. lifecycle
print("== 7. lifecycle (dock closed, ROI changes, viewer closed)")
import gc  # noqa: E402
import traceback  # noqa: E402
import napari_sprout.widgets.prompt_tab as PTM  # noqa: E402

UNCAUGHT = []
_old_hook = sys.excepthook
def _hook(*a):
    UNCAUGHT.append("".join(traceback.format_exception(*a)))
    sys.stderr.write(UNCAUGHT[-1])


sys.excepthook = _hook


def pump_for(sec):
    t = time.time()
    while time.time() - t < sec:
        QApplication.processEvents()
        time.sleep(0.005)


small_img = (np.random.default_rng(0).random((40, 60, 80)) * 255).astype(np.uint8)
small_seeds = np.zeros(small_img.shape, np.uint8)
small_seeds[5:15, 10:30, 10:30] = 2
small_seeds[20:30, 30:50, 40:70] = 3

# --- 7a. the SPROUT dock is closed, the viewer stays open
v = napari.Viewer(show=False)
v.add_image(small_img, name="img")
seeds_l = v.add_labels(small_seeds.copy(), name="seeds")
mw = SPROUTWidget(v)
v.window.add_dock_widget(mw)
pump_for(1.0)
pt = mw.prompt_tab
pt.set_prompts(PP.seeds_to_point_prompts(small_seeds, small_img, rng=np.random.RandomState(0), log=lambda m: None))
pts_l = pt.points_layer()
pts_l.selected_data = {0}
pump_for(0.3)
n0 = len(UNCAUGHT)
dw = [d for d in v.window._dock_widgets.values() if d.widget() is mw][0]
v.window.remove_dock_widget(dw)
del mw, dw, pt
gc.collect()
run_event_loop(1.0)
gc.collect()                    # Qt deleted the dock in the loop; this frees its Python objects
m7 = RL.get_roi_hub(v)


def tab_callbacks(emitter):
    """Callbacks of `emitter` that belong to a PromptTab (bound methods or closures over one)."""
    found = []
    if hasattr(emitter, "callbacks"):             # napari EventEmitter
        cbs = list(emitter.callbacks)
    else:                                            # psygnal SignalInstance
        cbs = []
        for slot in list(getattr(emitter, "_slots", [])):
            try:
                cbs.append(slot.dereference())
            except Exception:
                pass
    for cb in cbs:
        objs = []
        if isinstance(cb, tuple):                       # napari: (weakref to obj, method name)
            objs.append(cb[0]() if callable(cb[0]) else cb[0])
        else:
            objs.append(getattr(cb, "__self__", None))
            for cell in (getattr(cb, "__closure__", None) or ()):
                try:
                    objs.append(cell.cell_contents)
                except ValueError:          # empty cell
                    pass
        if any(type(o).__name__ == "PromptTab" for o in objs):
            found.append(cb)
    return found


left = {name: len(tab_callbacks(em)) for name, em in (("layers.inserted", v.layers.events.inserted),
                                                      ("layers.removed", v.layers.events.removed),
                                                      ("prompts.data", pts_l.events.data),
                                                      ("prompts.selection", pts_l.selected_data.events.items_changed))}
ok(not any(left.values()), f"after closing the dock no napari event still calls into the tab: {left}")
steps = [("page z", lambda: v.dims.set_current_step(0, 5)),
         ("add layer", lambda: v.add_image(small_img[::2], name="img2")),
         ("remove layer", lambda: v.layers.remove("img2")),
         ("add a prompt point", lambda: pts_l.add([[3, 3, 3]])),
         ("select prompt points", lambda: setattr(pts_l, "selected_data", {1})),
         ("move prompt points", lambda: setattr(pts_l, "data", np.asarray(pts_l.data) + 1)),
         ("paint the seeds", lambda: seeds_l.paint((10, 20, 20), 2, refresh=True)),
         ("new ROI layer", lambda: RL.add_roi_layer(v, v.layers["img"], (0, 0, 0), (10, 10, 10))),
         ("edit the ROI", lambda: m7.set_yx(RL.roi_layers(v)[0], (1, 1), (9, 9))),
         ("remove the ROI", lambda: v.layers.remove(RL.roi_layers(v)[0])),
         ("remove the seeds", lambda: v.layers.remove("seeds"))]
raised = []
for what, fn in steps:
    try:
        fn()
        pump_for(0.3)
    except Exception as e:
        raised.append(f"{what}: {type(e).__name__}: {e}")
errs = UNCAUGHT[n0:]
ok(not raised and not errs, f"after closing the SPROUT dock: paging / layers / prompt edits / painting seeds / ROI raise "
                            f"nothing ({len(raised)} raised, {len(errs)} uncaught)"
   + (f": {(raised or errs)[0][-400:]}" if raised or errs else ""))
v.close()
pump_for(0.5)

# --- 7b. an ROI layer the tab does not pick: edits do not make the tab recount
v = napari.Viewer(show=False)
v.add_image(small_img, name="img")
v.add_labels(small_seeds.copy(), name="seeds")
mw = SPROUTWidget(v)
v.window.add_dock_widget(mw)
pump_for(1.0)
pt = mw.prompt_tab
pt.seeds_combo.setCurrentText("seeds")
pump_for(0.5)
m7 = RL.get_roi_hub(v)
lr7 = RL.add_roi_layer(v, v.layers["img"], (2, 5, 5), (30, 40, 60))
pump_for(1.0)
calls = []
_cs = PTM.PP.class_stats


def counting(*a, **k):
    calls.append(1)
    return _cs(*a, **k)


PTM.PP.class_stats = counting
try:
    m7.set_yx(lr7, (3, 6), (35, 50))
    pump_for(0.8)
    m7.set_z_range(lr7, 4, 25)
    pump_for(0.8)
    n_paint = len(calls)
    RL.add_roi_layer(v, v.layers["img"], (0, 0, 0), (5, 5, 5))
    pump_for(0.5)
    n_kind = len(calls) - n_paint
finally:
    PTM.PP.class_stats = _cs
ok(n_paint == 0 and n_kind == 0, f"ROI not picked: 2 edits of the ROI layer -> {n_paint} recounts, "
                                 f"a new ROI layer -> {n_kind} recounts (both must be 0)")
pt.roi_select.select(lr7)
pump_for(1.0)
roi7 = m7.roi_of(lr7)
st7 = PP.class_stats(roi7.crop(small_seeds), allowed=roi7.mask_crop())
ok(pt._stats == dict(zip(st7.class_id.tolist(), st7.voxels.tolist())), "picking it recounts inside the ROI (on the next turn)")
v.close()
pump_for(0.5)


# --- 7c. closing napari with a mask ROI: no worker starts during close, no errors after
def close_with_mask(enabled):
    vv = napari.Viewer(show=False)
    vv.add_image(small_img, name="img")
    vv.add_labels(small_seeds.copy(), name="seeds")
    mw_ = SPROUTWidget(vv)
    vv.window.add_dock_widget(mw_)
    pump_for(1.0)
    mw_.prompt_tab.seeds_combo.setCurrentText("seeds")
    lay = RL.add_roi_layer(vv, vv.layers["img"], (2, 5, 5), (30, 40, 60))
    if enabled:
        mw_.prompt_tab.roi_select.select(lay)
    pump_for(1.0)
    started = []
    orig = PTM.PromptTab._run_bg

    def spy(self, fn, *a, **k):
        started.append(getattr(fn, "__qualname__", str(fn)))
        return orig(self, fn, *a, **k)

    PTM.PromptTab._run_bg = spy
    n0_ = len(UNCAUGHT)
    try:
        vv.close()
        during = list(started)
        pump_for(1.5)
        gc.collect()
        pump_for(0.5)
    finally:
        PTM.PromptTab._run_bg = orig
    return during, UNCAUGHT[n0_:]


for en in (False, True):
    during, errs = close_with_mask(en)
    ok(not during and not errs, f"close napari, an ROI layer {'picked' if en else 'not picked'}: Prompt workers started "
                                f"during close {during}, uncaught errors {len(errs)}"
       + (f": {errs[0][-400:]}" if errs else ""))
sys.excepthook = _old_hook
if UNCAUGHT:
    print(f"{len(UNCAUGHT)} uncaught errors in section 7; the first:", UNCAUGHT[0][-1500:])
print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
sys.exit(1 if FAILED else 0)
