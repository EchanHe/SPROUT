"""
The default output folder (<cwd>/sprout_temp) is shown in Seeds / Grow / Prompt (AI) but only
created when a run writes files there.

Run in the `sprout` env:
    python tests/check_output_folder.py
Prints PASS/FAIL per check; exits non-zero if anything failed.
"""
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "napari_sprout"), str(Path(__file__).resolve().parent)]
sys.stdout.reconfigure(encoding="utf-8")

import napari  # noqa: E402
import _headless_gl  # noqa: E402,F401  (napari >= 0.7 offscreen: no GL context)
import numpy as np  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

import napari_sprout.widgets.prompt_tab as PTM  # noqa: E402

PTM.SETTINGS = ("SPROUT-checks", "napari-sprout-checks")
from napari_sprout._widget import SPROUTWidget  # noqa: E402
from sprout_core.nni_engine import NniEngine  # noqa: E402
from _prompt_check_utils import FakeSession26  # noqa: E402

FAILED = []


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


def pump(n=40):
    for _ in range(n):
        QApplication.processEvents()
        time.sleep(0.005)


cwd0 = os.getcwd()
tmp = Path(tempfile.mkdtemp(prefix="sprout_cwd_"))
os.chdir(tmp)                                   # an empty working directory
out = tmp / "sprout_temp"
try:
    rng = np.random.default_rng(0)
    img = (rng.random((24, 60, 70)) * 60).astype(np.uint8)
    img[4:12, 10:30, 10:30] = 200
    img[14:22, 30:50, 35:60] = 180
    seeds = np.zeros(img.shape, np.uint8)
    seeds[6:10, 15:25, 15:25] = 1
    seeds[16:20, 35:45, 40:55] = 2
    v = napari.Viewer(show=False)
    v.add_image(img, name="img")
    v.add_labels(seeds, name="seeds")
    main = SPROUTWidget(v)
    v.window.add_dock_widget(main)
    pump(100)
    sw, gw, pt = main.seed_widget, main.grow_widget, main.prompt_tab
    shown = {sw.output_folder_line.text(), gw.output_folder_line.text(), pt.folder_edit.text()}
    ok(shown == {str(out)}, f"Seeds / Grow / Prompt show the default folder {out.name} in the cwd ({shown})")
    ok(not out.exists() and not any(tmp.iterdir()), f"building the panel creates nothing in the cwd "
                                                    f"({sorted(p.name for p in tmp.iterdir())})")

    # Seeds
    st = sw.threshold_widget
    wait(lambda: st.hist._ready(), what="seeds histogram")
    sw.image_combo.setCurrentText("img")
    mp = sw.main_param_widget
    mp.seed_method_combo.setCurrentText("Original")
    mp.populate_thresholds_from_list([150], [255])
    mp.erosion_spin.setValue(1)
    mp.segments_spin.setValue(3)
    sw.generate_seeds()
    wait(lambda: sw.generate_btn.isEnabled() and sw.worker is not None and not sw.worker.isRunning(), what="seeds")
    pump()
    files = [p for p in out.rglob("*") if p.is_file()] if out.exists() else []
    ok(out.exists() and any(p.suffix == ".tif" for p in files), f"running Seeds creates {out.name} and writes there "
                                                                 f"({len(files)} files)")

    # Grow
    shutil.rmtree(out)
    gw.refresh_layers()
    gw.image_combo.setCurrentText("img")
    gw.seeds_combo.setCurrentText("seeds")
    gw.main_param_widget.populate_thresholds_from_list([150], [255], [2])
    gw.start_growth()
    wait(lambda: gw.grow_btn.isEnabled(), what="grow")
    pump()
    files = [p for p in out.rglob("*") if p.is_file()] if out.exists() else []
    ok(out.exists() and any(p.suffix == ".tif" for p in files), f"running Grow creates {out.name} and writes there "
                                                                 f"({len(files)} files)")

    # Prompt (AI): Save results to folder (fake model)
    shutil.rmtree(out)
    ft = PTM.PromptTab(v, engine_factory=lambda: NniEngine(session_factory=lambda d: FakeSession26(device=d),
                                                          buffer_factory=lambda s: np.zeros(s, np.uint8)),
                       status_fn=lambda: (True, "fake"))
    wait(lambda: ft.table.rowCount() == 2, what="prompt table")
    ok(ft.folder_edit.text() == str(out) and not out.exists(), "a new Prompt tab does not create it either")
    ft.model_edit.setText(str(tmp))
    ft.load_model()
    wait(lambda: ft.engine.loaded and not any(ft._busy.values()), what="load")
    ft.run_checked()
    wait(lambda: not any(ft._busy.values()), what="prompt run, no save")
    ok(not out.exists(), "a Prompt run without 'Save results to folder' writes nothing")
    ft.save_cb.setChecked(True)
    ft.run_checked()
    wait(lambda: not any(ft._busy.values()), what="prompt run + save")
    names = sorted(p.name for p in out.iterdir()) if out.exists() else []
    ok(names == ["img_segmentation.tif", "used_config.yaml"], f"Prompt 'Save results to folder' creates it and "
                                                              f"writes there: {names}")
    ft._finalizer()
    v.close()
finally:
    os.chdir(cwd0)
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
sys.exit(1 if FAILED else 0)
