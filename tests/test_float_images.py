"""
A float image, such as a probability map of nnInteractive: decimal thresholds all the way through.

The image has two cores above 0.8 that are one region above 0.4. An int() anywhere on the way
turns the thresholds into 0 and the whole image is one region.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
import yaml
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import batch_seeds  # noqa: E402
import make_adaptive_seed  # noqa: E402
import make_grow  # noqa: E402
import make_seeds  # noqa: E402
import make_sweep_adaptive_seed as sweep  # noqa: E402
import sprout_core.config_core as config_core  # noqa: E402
from sprout_core import adaptive_seed_combine as combine  # noqa: E402

pytestmark = pytest.mark.core


def probability_map():
    img = np.zeros((16, 32, 32), np.float32)
    img[4:12, 4:28, 4:14] = 0.5
    img[4:12, 4:28, 14:18] = 0.5
    img[4:12, 4:28, 18:28] = 0.5
    img[5:11, 6:12, 6:12] = 0.9
    img[5:11, 20:26, 20:26] = 0.9
    return img


def test_whole_numbers_stay_ints():
    assert config_core.check_and_assign_thresholds([100.0, 120], [255.0, 255]) == ([100, 120], [255, 255])
    assert all(type(t) is int for t in config_core.check_and_assign_thresholds([100.0], None)[0])
    assert config_core.check_and_assign_threshold(100.0, 255.0) == (100, 255)


def test_a_single_decimal_threshold():
    assert config_core.check_and_assign_threshold(0.5, 0.9) == (0.5, 0.9)
    assert config_core.check_and_assign_thresholds(0.5, 0.9) == ([0.5], [0.9])


def test_a_decimal_threshold_passes_the_config(monkeypatch):
    monkeypatch.chdir(ROOT)
    config = yaml.safe_load(open(ROOT / "template" / "sweep_adaptive_seed.yaml", encoding="utf-8"))
    config.update(thresholds=0.5, upper_thresholds=0.9, num_threads=1)
    config_core.validate_input_yaml(config, config_core.input_val_sweep_adaptive_seed)


def test_seeds(tmp_path):
    seeds, _ = make_seeds.make_seeds(img=probability_map(), thresholds=[0.8], upper_thresholds=[1.0], erosion_steps=0,
                                     segments=5, num_threads=1, output_folder=str(tmp_path), return_for_napari=True,
                                     base_name="p")
    final = next(iter(seeds.values()))
    assert len(np.unique(final)) == 3


def test_adaptive_seed(tmp_path):
    img = probability_map()
    make_adaptive_seed.make_adaptive_seed_thre(img=img, thresholds=[0.4, 0.8], upper_thresholds=[1.0, 1.0],
                                               erosion_steps=0, segments=5, num_threads=1, base_name="p",
                                               output_folder=str(tmp_path), min_size=1, min_split_ratio=0)
    final = tifffile.imread(next(tmp_path.rglob("FINAL_adaptive_seed.tif")))
    assert len(np.unique(final)) == 3
    assert np.array_equal(final > 0, img >= 0.8)      # a region over two seeds locks them


def adaptive_yaml(tmp_path, thresholds, **changes):
    path = tmp_path / "p.tif"
    tifffile.imwrite(path, probability_map())
    config = yaml.safe_load(open(ROOT / "template" / "make_adaptive_seed.yaml", encoding="utf-8"))
    config.update(img_path=str(path), output_folder=str(tmp_path / "out"), thresholds=thresholds, erosion_steps=1,
                  segments=5, num_threads=1, min_size=1, save_every_iter=False, **changes)
    yaml.safe_dump(config, open(tmp_path / "config.yaml", "w"))
    return tmp_path / "config.yaml"


@pytest.mark.parametrize("thresholds", [[0.4, 0.8], 0.4, [0.4]], ids=["list", "one", "list_of_one"])
def test_adaptive_seed_from_a_yaml(tmp_path, thresholds):
    make_adaptive_seed.run_make_adaptive_seed(str(adaptive_yaml(tmp_path, thresholds)))
    final = tifffile.imread(next(tmp_path.rglob("FINAL_adaptive_seed.tif")))
    assert final.any() and not final[probability_map() < 0.4].any()


def test_batch_adaptive_seed(tmp_path):
    config_path = adaptive_yaml(tmp_path, [0.4, 0.8])
    config = yaml.safe_load(open(config_path, encoding="utf-8"))
    pd.DataFrame({"img_path": [config.pop("img_path")]}).to_csv(tmp_path / "input.csv", index=False)
    config["csv_path"] = str(tmp_path / "input.csv")
    yaml.safe_dump(config, open(config_path, "w"))
    batch_seeds.run_batch_adaptive_seed(str(config_path))
    assert len(np.unique(tifffile.imread(next(tmp_path.rglob("FINAL_adaptive_seed.tif"))))) == 3


def test_grow(tmp_path):
    img = probability_map()
    seed = ndimage.label(img >= 0.8)[0].astype(np.uint8)
    grown, _ = make_grow.grow_mp(img=img, seg=seed, thresholds=[0.4], upper_thresholds=[1.0], dilation_steps=[30],
                                 touch_rule="stop", num_threads=1, output_folder=str(tmp_path), base_name="p",
                                 save_every_n_iters=None, is_sort=False, no_growth_max_iter=None,
                                 return_for_napari=True)
    assert np.array_equal(grown["FINAL_GROW_p"] > 0, img >= 0.4)


@pytest.mark.parametrize("thresholds", [[0.8, 0.4], [0.4, 0.8]], ids=["down", "up"])
def test_sweep(tmp_path, thresholds):
    path = tmp_path / "p.tif"
    tifffile.imwrite(path, probability_map())
    config = yaml.safe_load(open(ROOT / "template" / "sweep_adaptive_seed.yaml", encoding="utf-8"))
    config.update(img_path=str(path), output_folder=str(tmp_path / "out"), thresholds=thresholds, segments=5,
                  num_threads=1, min_size=1, top_n=None, verbose=False)
    sweep.run_sweep_adaptive_seed_from_dict(config)
    final = tifffile.imread(tmp_path / "out" / "p" / "FINAL_sweep_seed.tif")
    assert len(np.unique(final)) == 3
    assert np.array_equal(final > 0, probability_map() >= 0.8)      # a region over two seeds locks them


def test_seeds_folder_with_decimal_names(tmp_path):
    path = tmp_path / "p.tif"
    tifffile.imwrite(path, probability_map())
    config = yaml.safe_load(open(ROOT / "template" / "sweep_adaptive_seed.yaml", encoding="utf-8"))
    config.update(img_path=str(path), output_folder=str(tmp_path / "a"), thresholds=[0.4, 0.8], segments=5,
                  num_threads=1, min_size=1, top_n=None, verbose=False)
    sweep.run_sweep_adaptive_seed_from_dict(config)
    fresh = tifffile.imread(tmp_path / "a" / "p" / "FINAL_sweep_seed.tif")
    sweep.run_sweep_adaptive_seed_from_dict({**config, "output_folder": str(tmp_path / "b"),
                                             "seeds_folder": str(tmp_path / "a" / "p" / "steps")})
    assert np.array_equal(fresh, tifffile.imread(tmp_path / "b" / "p" / "FINAL_sweep_seed.tif"))


def test_files_sort_by_a_decimal_threshold(tmp_path):
    names = ["step_000_thre_0.9_None_ero_0.tif", "step_001_thre_0.5_None_ero_0.tif", "step_002_thre_0.7_None_ero_0.tif"]
    for value, name in enumerate(names):
        tifffile.imwrite(tmp_path / name, np.full((2, 2), value, np.uint8))
    _, paths = combine.load_threshold_sweep_seeds([tmp_path / name for name in names])
    assert [path.name for path in paths] == [names[0], names[2], names[1]]
