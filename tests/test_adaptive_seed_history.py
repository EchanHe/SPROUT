"""
adaptive_seed_regions.csv and adaptive_seed_steps.csv: what classic adaptive seed did, step by step.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sprout_core import make_adaptive_seed  # noqa: E402

pytestmark = pytest.mark.core


def block(core_value=200, speck=False):
    """A block at 100 holding two cores at core_value (8.3 % of the block each), and a speck."""
    img = np.zeros((14, 14, 30), np.uint8)
    img[2:12, 2:12, 2:28] = 100
    img[4:10, 4:10, 4:10] = core_value
    img[4:10, 4:10, 20:26] = core_value
    if speck:
        img[11, 11, 14:16] = core_value
    return img


def run_thre(tmp_path, img, thresholds, **changes):
    values = dict(img=img, thresholds=thresholds, erosion_steps=0, segments=10, num_threads=1,
                  output_folder=str(tmp_path), base_name="b", min_size=1, min_split_ratio=0)
    values.update(changes)
    make_adaptive_seed.make_adaptive_seed_thre(**values)
    return read(tmp_path)


def read(folder):
    regions = pd.read_csv(next(Path(folder).rglob("adaptive_seed_regions.csv")))
    steps = pd.read_csv(next(Path(folder).rglob("adaptive_seed_steps.csv")), keep_default_na=False)
    final = tifffile.imread(next(Path(folder).rglob("FINAL_adaptive_seed.tif")))
    return regions, steps, final


def rows_are_the_image(regions, final):
    return sorted(regions["id"]) == sorted(set(np.unique(final)) - {0}) and \
        all(row["area"] == np.count_nonzero(final == row["id"]) for _, row in regions.iterrows())


def test_a_split(tmp_path):
    regions, steps, final = run_thre(tmp_path, block(), [100, 150])
    assert rows_are_the_image(regions, final)
    assert sorted(regions["began"]) == ["separated (1 of 2)", "separated (2 of 2)"]
    assert (regions["first_step"] == 1).all() and (regions["threshold"] == 150).all()
    assert steps[["step", "threshold", "regions", "split", "new_seeds", "seeds_after"]].values.tolist() == \
        [[0, 100, 1, 0, 1, 1], [1, 150, 2, 1, 2, 2]]


def test_a_split_by_erosion(tmp_path):
    img = np.zeros((12, 12, 24), np.uint8)
    img[3:9, 3:9, 2:10] = img[3:9, 3:9, 14:22] = img[6, 6, 10:14] = 200      # two cubes and a thin bridge
    make_adaptive_seed.make_adaptive_seed_ero(img=img, threshold=100, erosion_steps=2, segments=10, num_threads=1,
                                              output_folder=str(tmp_path), base_name="c", min_size=1,
                                              min_split_ratio=0)
    regions, steps, final = read(tmp_path)
    assert rows_are_the_image(regions, final)
    assert sorted(regions["began"]) == ["separated (1 of 2)", "separated (2 of 2)"]
    assert (regions["erosion"] == 1).all()
    assert steps["erosion"].tolist() == [0, 1, 2] and steps["split"].tolist() == [0, 1, 0]


def test_a_part_too_small_to_count_leaves_a_shrunk_seed(tmp_path):
    img = np.zeros((12, 12, 12), np.uint8)
    img[1:11, 1:11, 1:11] = 100
    img[3:9, 3:9, 3:9] = 200                     # 21.6 % of the block
    img[10, 10, 3:5] = 200                       # 0.2 %
    regions, steps, final = run_thre(tmp_path, img, [100, 150], min_split_ratio=0.05)
    assert rows_are_the_image(regions, final)
    assert regions["began"].tolist() == ["shrank (rest disappeared)"]
    assert steps["shrunk"].tolist() == [0, 1]


def test_parts_all_too_small_drop_the_region(tmp_path):
    regions, steps, final = run_thre(tmp_path, block(), [100, 150], min_split_ratio=0.5)
    assert not final.any() and regions.empty
    assert steps["dropped"].tolist() == [0, 1] and steps["seeds_after"].tolist() == [1, 0]


def test_an_early_stop_is_noted(tmp_path):
    _, steps, _ = run_thre(tmp_path, block(), [100, 110, 120, 150], no_split_max_iter=1)
    assert steps["note"].tolist() == ["", "", "stopped: no split for 1 steps"]      # 110 splits, 120 does not


def test_pruning_keeps_the_rows_on_the_seeds(tmp_path):
    regions, steps, final = run_thre(tmp_path, block(speck=True), [100, 150, 180], min_size=5, prune_every_iter=True)
    assert rows_are_the_image(regions, final)
    assert steps["pruned"].sum() >= 1                   # the speck
    assert set(regions["began"]) == {"separated (1 of 3)", "separated (2 of 3)"}      # the speck was 3 of 3
