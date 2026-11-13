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
    assert (regions["first_step"] == 1).all() and (regions["first_threshold"] == 150).all()
    assert (regions["ended"] == "reached the end").all() and regions["at_step"].isna().all()
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
    assert (regions["first_erosion"] == 1).all() and (regions["last_erosion"] == 2).all()
    assert steps["erosion"].tolist() == [0, 1, 2] and steps["split"].tolist() == [0, 1, 0]


def core_and_speck():
    """A block at 100 (1000 voxels) holding a core at 200 (216 voxels) and a speck at 200 (2 voxels)."""
    img = np.zeros((12, 12, 12), np.uint8)
    img[1:11, 1:11, 1:11] = 100
    img[3:9, 3:9, 3:9] = 200                     # 21.6 % of the block
    img[10, 10, 3:5] = 200                       # 0.2 %
    return img


# A region splits only into parts that count (min_size and min_split_ratio); else it stays as it is.
@pytest.mark.parametrize("small", [dict(min_split_ratio=0.05), dict(min_size=5)], ids=["by ratio", "by size"])
def test_a_core_and_a_speck_are_no_split(tmp_path, small):
    regions, steps, final = run_thre(tmp_path, core_and_speck(), [100, 150], **small)
    assert rows_are_the_image(regions, final)
    assert regions["began"].tolist() == ["first step"] and regions["area"].tolist() == [1000]   # the loose region
    assert steps["split"].tolist() == [0, 0]


def test_parts_all_too_small_keep_the_region(tmp_path):
    regions, steps, final = run_thre(tmp_path, block(), [100, 150], min_split_ratio=0.5)      # cores: 8.3 % each
    assert rows_are_the_image(regions, final) and regions["area"].tolist() == [10 * 10 * 26]
    assert steps["seeds_after"].tolist() == [1, 1]


def test_a_thin_part_is_not_lost_by_erosion(tmp_path):
    """Two 3x3x3 knobs on a thin bar: one erosion leaves two single voxels, below min_size."""
    img = np.zeros((7, 7, 15), np.uint8)
    img[2:5, 2:5, 2:5] = img[2:5, 2:5, 10:13] = img[3, 3, 5:10] = 200
    make_adaptive_seed.make_adaptive_seed_ero(img=img, threshold=100, erosion_steps=1, segments=10, num_threads=1,
                                              output_folder=str(tmp_path), base_name="t", min_size=5,
                                              min_split_ratio=0)
    regions, _, final = read(tmp_path)
    assert rows_are_the_image(regions, final) and np.array_equal(final > 0, img > 0)


def test_the_columns_are_those_of_sweep(tmp_path):
    regions, _, _ = run_thre(tmp_path, block(), [100, 150])          # the upper threshold does not change
    assert regions.columns.tolist() == ["id", "area", "first_step", "last_step", "first_threshold",
                                        "last_threshold", "began", "ended", "at_step"]


def test_a_region_without_a_core_has_disappeared(tmp_path):
    """Two blocks: one at 200, one at 120, which has nothing above 150. Its seed keeps its shape."""
    img = np.zeros((12, 12, 24), np.uint8)
    img[2:10, 2:10, 2:10] = 200
    img[2:10, 2:10, 14:20] = 120
    regions, _, final = run_thre(tmp_path, img, [100, 150, 180])
    assert rows_are_the_image(regions, final) and np.array_equal(final > 0, img > 0)
    ended = regions.set_index("area").loc[8 * 8 * 6]
    assert ended[["ended", "at_step", "last_step", "last_threshold"]].tolist() == ["disappeared", 1, 0, 100]
    assert regions["ended"].tolist().count("reached the end") == 1


def test_a_core_left_out_by_segments_that_comes_back_has_not_disappeared(tmp_path):
    """segments 2 (3 at the last step, so all three are kept). Block A holds cores at 220 and 170; block B a small core at 220.
    At 150, B's core is the third largest and is left out; at 200, the 170 core is gone and B's is back."""
    img = np.zeros((14, 14, 40), np.uint8)
    img[2:12, 2:12, 2:26] = 100
    img[3:11, 3:11, 3:12] = 220
    img[3:11, 3:11, 15:24] = 170
    img[4:9, 4:9, 30:35] = 100
    img[5:8, 5:8, 31:34] = 220
    regions, _, final = run_thre(tmp_path, img, [100, 150, 200], segments=2, last_segments=3)
    assert rows_are_the_image(regions, final)
    b = regions[regions["began"] == "first step"].iloc[0]
    assert b["ended"] == "reached the end" and b["last_step"] == 2
    gone = regions[regions["ended"] == "disappeared"]
    assert gone[["first_step", "last_step", "at_step"]].values.tolist() == [[1, 1, 2]]    # the 170 core


def test_an_early_stop_is_noted(tmp_path):
    _, steps, _ = run_thre(tmp_path, block(), [100, 110, 120, 150], no_split_max_iter=1)
    assert steps["note"].tolist() == ["", "", "stopped: no split for 1 steps"]      # 110 splits, 120 does not


def test_pruning_keeps_the_rows_on_the_seeds(tmp_path):
    regions, steps, final = run_thre(tmp_path, block(speck=True), [100, 150, 180], min_size=1, last_segments=2,
                                     prune_every_iter=True)        # the speck counts as a part, then top 2 are kept
    assert rows_are_the_image(regions, final)
    assert steps["pruned"].sum() >= 1                   # the speck
    assert set(regions["began"]) == {"separated (1 of 3)", "separated (2 of 3)"}      # the speck was 3 of 3
