"""
One test per known bug: each says what should happen and is marked xfail until the bug is fixed.

strict=True turns the run red when a bug stops failing, so the mark gets removed with the fix.
`raises` is the failure of the bug itself: any other exception is a real failure, so a broken
test cannot pass as "still the known bug". The numbers are those of the bug list in the PRs.

The bugs of the plugin run in their own process, see probe_known_bugs.py.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
import yaml
from scipy.spatial import QhullError

from test_checks import environment

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
sys.path.insert(0, str(ROOT))

import make_adaptive_seed  # noqa: E402
import make_grow  # noqa: E402
import sprout_core.config_core as config_core  # noqa: E402
import sprout_core.sprout_core as sprout_core  # noqa: E402


def bug(number, why, raises=AssertionError):
    return pytest.mark.xfail(strict=True, raises=raises, reason=f"#{number}: {why}")


def blob_2d():
    mask = np.zeros((16, 16), bool)
    mask[3:12, 3:12] = True
    return mask


def blob_3d():
    mask = np.zeros((12, 16, 16), bool)
    mask[2:10, 3:12, 3:12] = True
    return mask


def grow_template():
    return yaml.safe_load(open(ROOT / "template" / "make_grow.yaml", encoding="utf-8"))


def small_grow_files(folder):
    img = np.zeros((12, 12, 12), np.uint8)
    img[2:10, 2:10, 2:10] = 200
    seg = np.zeros_like(img)
    seg[5, 5, 5] = 1
    tifffile.imwrite(folder / "img.tif", img)
    tifffile.imwrite(folder / "seg.tif", seg)
    return dict(img_path=str(folder / "img.tif"), seg_path=str(folder / "seg.tif"))


@pytest.mark.core
@bug(1, "touch_rule='overwrite' grows into voxels below the threshold")
def test_1_overwrite_stays_in_the_threshold(tmp_path):
    img = np.zeros((20, 20, 20), np.uint8)
    img[:, :, :10] = 200
    seed = np.zeros_like(img)
    seed[8:12, 8:12, 4:8] = 1
    grows, _ = make_grow.grow_mp(img=img, seg=seed, thresholds=[150], upper_thresholds=[255], dilation_steps=[8],
                                 touch_rule="overwrite", num_threads=1, output_folder=str(tmp_path), base_name="t",
                                 save_every_n_iters=None, is_sort=False, no_growth_max_iter=None,
                                 return_for_napari=True)
    assert np.count_nonzero(grows["FINAL_GROW_t"][img < 150]) == 0


@pytest.mark.core
@bug(3, "float thresholds are cut to int, so [0.5, 0.9] becomes [0, 0]")
def test_3_float_thresholds_are_kept():
    thresholds, _ = config_core.check_and_assign_thresholds([0.5, 0.9], None)
    assert thresholds == [0.5, 0.9]


@pytest.mark.core
@bug(4, "a footprint for the other dimension does nothing and says nothing")
@pytest.mark.parametrize("make, footprint", [(blob_2d, "ball_XY"), (blob_2d, "Z"), (blob_2d, "2XZ_1Y"),
                                             (blob_3d, "disk"), (blob_3d, "square")])
def test_4_wrong_footprint_erodes_or_is_refused(make, footprint):
    mask = make()
    before = mask.sum()
    try:
        after = sprout_core.erosion_binary_img_on_sub(mask, footprint=footprint).sum()
    except ValueError:
        return
    assert after < before


@pytest.mark.core
@bug(5, "a negative id wraps around: -1 becomes 255")
def test_5_negative_ids_do_not_wrap():
    try:
        out = sprout_core.cast_labels(np.array([-1, 0, 1], np.int32))
    except ValueError:
        return
    assert out.min() == -1


@pytest.mark.core
@bug(6, "2D erosion of an empty mask crashes", raises=ValueError)
def test_6_erosion_of_an_empty_2d_mask():
    assert not sprout_core.erosion_binary_img_on_sub(np.zeros((16, 16), bool)).any()


@pytest.mark.core
@bug(7, "a list written as a string stays a string in the optional values")
def test_7_string_list_is_parsed(monkeypatch):
    monkeypatch.chdir(ROOT)
    config = grow_template()
    config.update(thresholds=[100, 120], upper_thresholds="[255, 255]", dilation_steps=[1, 1], num_threads=1)
    assert config_core.validate_input_yaml(config, config_core.input_val_make_grow)["upper_thresholds"] == [255, 255]


@pytest.mark.core
@bug(8, "sprout.py without --config crashes before its error message")
def test_8_missing_config_gives_a_message(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "sprout.py"), "--seeds"], cwd=tmp_path, env=environment(tmp_path),
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode != 0
    assert "AttributeError" not in r.stderr
    assert "--config" in r.stdout + r.stderr


@pytest.mark.core
@bug(9, "a missing final_grow_output_folder fails after Grow has finished", raises=FileNotFoundError)
def test_9_final_output_folder_is_created(tmp_path):
    folder = tmp_path / "does" / "not" / "exist"
    make_grow.grow_mp(**small_grow_files(tmp_path), output_folder=str(tmp_path / "out"), thresholds=[100],
                      dilation_steps=[2], num_threads=1, save_every_n_iters=None, touch_rule="stop",
                      base_name="t", final_grow_output_folder=str(folder))
    assert (folder / "FINAL_GROW_t.tif").is_file()


@pytest.mark.core
@bug(10, "the convex hull of a flat seed crashes", raises=QhullError)
def test_10_convex_hull_of_a_flat_seed():
    mask = np.zeros((10, 10, 10), bool)
    mask[5, 2:8, 2:8] = True
    assert make_adaptive_seed.split_convex_hull_check(mask, [1, None]) is False


@pytest.mark.core
@bug(11, "a missing file gives 'No active exception to reraise'", raises=RuntimeError)
def test_11_missing_tiff_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="no_such.tif"):
        sprout_core.check_tiff_files([str(tmp_path / "no_such.tif")])


@pytest.mark.core
@bug(11, "an unreadable file gives an UnboundLocalError instead of the reason", raises=UnboundLocalError)
def test_11_unreadable_tiff_gives_the_reason(tmp_path):
    path = tmp_path / "bad.tif"
    path.write_text("not a tif")
    with pytest.raises(tifffile.TiffFileError):
        sprout_core.check_tiff_files([str(path)])


@pytest.mark.core
@bug(12, "a blank cell makes pandas read the column as float, so 5 becomes 5.0 and is refused", raises=ValueError)
def test_12_csv_column_with_a_blank(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    path = tmp_path / "grow.csv"
    path.write_text("name,min_growth_size\nx,5\ny,\n")
    row = {"min_growth_size": pd.read_csv(path).iloc[0]["min_growth_size"]}
    config = grow_template()
    config.pop("min_growth_size", None)
    config["num_threads"] = 1
    config_core.validate_input_yaml(config_core.merge_row_and_yaml_no_conflict(row, config),
                                    config_core.input_val_make_grow)


@pytest.mark.core
@bug(14, "sam_predict.yaml points to ./data/demo, the demo data is in ./demo_data")
def test_14_paths_of_the_sam_template_exist():
    config = yaml.safe_load(open(ROOT / "template" / "sam_predict.yaml", encoding="utf-8"))
    assert (ROOT / config["img_path"]).is_file() and (ROOT / config["seg_path"]).is_file()


@pytest.mark.core
@bug(15, "num_threads above the CPU cores is an error, so the templates fail on small machines")
def test_15_more_threads_than_cores(tmp_path):
    code = ("import psutil, yaml\n"
            "psutil.cpu_count = lambda *a, **k: 2\n"
            "import sprout_core.config_core as c\n"
            "config = yaml.safe_load(open('template/make_grow.yaml', encoding='utf-8'))\n"
            "assert config['num_threads'] > 2\n"
            "c.validate_input_yaml(config, c.input_val_make_grow)\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=environment(tmp_path), capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if r.returncode and "num_threads" not in r.stdout + r.stderr:
        pytest.fail(r.stderr[-1000:], pytrace=False)
    assert r.returncode == 0


PLUGIN_BUGS = [
    ("edit_keeps_scale", 2, "Edit results drop the scale and translate of their layer"),
    ("mapping_keeps_scale", 2, "Label Mapping results drop the scale and translate of their layer"),
    ("float_image_threshold_row", 13, "a float image crashes the threshold tables"),
    ("import_adaptive_template", 16, "Import of make_adaptive_seed.yaml fails on its single threshold"),
    ("load_scheme_checks_the_file", 17, "Load Scheme accepts any JSON file"),
    ("export_writes_the_image_path", 18, "Export writes the layer name as img_path"),
]


@pytest.mark.napari
@pytest.mark.parametrize("probe", [pytest.param(p, id=f"{n}_{p}", marks=bug(n, why)) for p, n, why in PLUGIN_BUGS])
def test_plugin(probe, tmp_path):
    if importlib.util.find_spec("napari") is None:
        pytest.skip("needs napari")
    r = subprocess.run([sys.executable, str(TESTS / "probe_known_bugs.py"), probe], cwd=tmp_path,
                       env=environment(tmp_path), capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=300)
    if r.returncode not in (0, 3):                 # 3: the bug is still there; anything else: the probe broke
        pytest.fail(f"exit code {r.returncode}\n{(r.stdout + r.stderr)[-2000:]}", pytrace=False)
    assert r.returncode == 0, r.stdout[-500:]
