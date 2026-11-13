"""
The sweep route: what a run has to keep true, checked on a small image with a known answer.

The expected values are computed here in the plainest way, not with the code under test.
"""
import builtins
import inspect
import json
import subprocess
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
import yaml
from scipy import ndimage
from skimage import measure

from test_checks import environment

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
sys.path.insert(0, str(ROOT))

from sprout_core import batch_seeds  # noqa: E402
from sprout_core import make_sweep_adaptive_seed as sweep  # noqa: E402
import sprout_core.config_core as config_core  # noqa: E402
import sprout_core.sprout_core as sprout_core  # noqa: E402
from sprout_core import adaptive_seed_combine as combine  # noqa: E402
from sprout_core import seed_sweep  # noqa: E402

THRESHOLDS = [200, 150, 100]
pytestmark = pytest.mark.core


def image():
    """Two bright cores in one dim body, and a separate dim block."""
    img = np.zeros((20, 40, 40), np.uint8)
    img[4:16, 4:24, 4:24] = 120
    img[6:14, 6:12, 6:12] = 220
    img[6:14, 16:22, 16:22] = 220
    img[6:14, 28:36, 28:36] = 170
    return img


def template(name="sweep_adaptive_seed.yaml"):
    return yaml.safe_load(open(ROOT / "template" / name, encoding="utf-8"))


def config(tmp_path, **changes):
    path = tmp_path / "img.tif"
    if not path.exists():
        tifffile.imwrite(path, image())
    values = template()
    values.update(img_path=str(path), output_folder=str(tmp_path / "out"), thresholds=list(THRESHOLDS), segments=10,
                  num_threads=1, min_size=1, top_n=None, verbose=False)
    values.update(changes)
    return values


def run(values):
    sweep.run_sweep_adaptive_seed_from_dict(values)
    return Path(values["output_folder"]) / "img"


def result(folder, suffix=""):
    return tifffile.imread(folder / f"FINAL_sweep_seed{suffix}.tif")


def test_min_size_drops_small_regions(tmp_path):
    every = result(run(config(tmp_path, output_folder=str(tmp_path / "a"))))
    large = result(run(config(tmp_path, output_folder=str(tmp_path / "b"), min_size=400)))  # each core: 288 voxels
    assert len(np.unique(large)) < len(np.unique(every))


@pytest.mark.parametrize("old, new", [("min_area", "min_size"), ("save_intermediate", "save_every_iter")])
def test_an_old_key_name_gives_the_new_one(tmp_path, old, new):
    values = config(tmp_path)
    values[old] = values.pop(new)
    with pytest.raises(ValueError, match=f"'{old}' is now '{new}'"):
        run(values)


def test_connectivity_is_no_longer_a_key(tmp_path):
    with pytest.raises(ValueError, match="'connectivity' is no longer used"):
        run(config(tmp_path, connectivity=1))


def corner_image(ndim):
    """Two blocks that touch only at their corners: one region, as everywhere in SPROUT."""
    img = np.zeros((6, 12, 12), np.uint8)
    img[1:5, 2:6, 2:6] = 200
    img[1:5, 6:10, 6:10] = 200
    return img if ndim == 3 else img[2]


@pytest.mark.parametrize("ndim", [3, 2])
def test_blocks_touching_at_a_corner_are_one_seed_from_yaml_and_napari(tmp_path, ndim):
    tifffile.imwrite(tmp_path / "img.tif", corner_image(ndim))
    values = config(tmp_path, thresholds=[200, 150], segments=1)       # one region per step: id 1 only
    for key in ("input_mode", "connectivity"):                         # not in the YAML: the defaults
        values.pop(key, None)
    from_yaml = result(run(values))
    from_napari = next(iter(sweep.make_sweep_adaptive_seed(
        img=corner_image(ndim), thresholds=[200, 150], erosion_steps=0, segments=1,
        output_folder=str(tmp_path / "napari")).values()))
    assert len(np.unique(from_yaml)) == len(np.unique(from_napari)) == 2       # background and one seed


def boundary_file(folder):
    boundary = np.zeros(image().shape, np.uint8)
    boundary[:, 14, :] = 1
    tifffile.imwrite(folder / "boundary.tif", boundary)
    return boundary > 0


# ---- what the steps hold

def test_steps_are_the_components_of_each_threshold(tmp_path):
    folder = run(config(tmp_path))
    steps = sorted((folder / "steps").glob("step_*.tif"))
    assert len(steps) == len(THRESHOLDS)
    for path, threshold in zip(steps, THRESHOLDS):
        expected = ndimage.label(image() >= threshold)[0]
        got = tifffile.imread(path)
        assert np.array_equal(got > 0, expected > 0), path.name
        assert len(np.unique(got)) == len(np.unique(expected)), path.name


def test_the_order_of_the_thresholds_does_not_matter(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    down = run(config(tmp_path / "a"))
    up = run(config(tmp_path / "b", thresholds=THRESHOLDS[::-1]))
    assert np.array_equal(result(down), result(up))


def test_threads_do_not_change_the_result(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    one = run(config(tmp_path / "a", num_threads=1))
    two = run(config(tmp_path / "b", num_threads=2))
    assert np.array_equal(result(one), result(two))


def test_strict_to_loose_keeps_the_cores_apart(tmp_path):
    seeds = result(run(config(tmp_path)))
    img = image()
    assert seeds[10, 8, 8] != seeds[10, 18, 18] and seeds[10, 8, 8] and seeds[10, 18, 18]
    assert seeds[10, 30, 30] not in (0, seeds[10, 8, 8], seeds[10, 18, 18])
    assert not seeds[img < min(THRESHOLDS)].any()


def test_both_gives_what_each_direction_gives(tmp_path):
    for name in "abc":
        (tmp_path / name).mkdir()
    both = run(config(tmp_path / "a", direction="both"))
    s2l = run(config(tmp_path / "b", direction="strict_to_loose"))
    l2s = run(config(tmp_path / "c", direction="loose_to_strict"))
    assert np.array_equal(result(both, "_strict_to_loose"), result(s2l))
    assert np.array_equal(result(both, "_loose_to_strict"), result(l2s))


# ---- what a run leaves

def test_the_output_folder(tmp_path):
    folder = run(config(tmp_path))
    names = {path.name for path in folder.iterdir()}
    assert names == {"FINAL_sweep_seed.tif", "sweep_regions.csv", "sweep_steps.csv", "sweep_metadata.json", "steps",
                     "used_config.yaml"}


def test_regions_say_what_the_image_holds(tmp_path):
    folder = run(config(tmp_path))
    seeds = result(folder)
    regions = pd.read_csv(folder / "sweep_regions.csv")
    assert sorted(regions["id"]) == sorted(set(np.unique(seeds)) - {0})
    for _, row in regions.iterrows():
        assert row["area"] == np.count_nonzero(seeds == row["id"])
    core = regions[regions["id"] == seeds[10, 8, 8]].iloc[0]
    assert core["first_threshold"] == 200 and core["first_step"] == 0 and core["began"] == "first step"
    side = regions[regions["id"] == seeds[10, 30, 30]].iloc[0]
    assert side["first_threshold"] == 150 and side["first_step"] == 1 and side["began"] == "emerged"
    assert set(regions["ended"]) <= {"locked", "never locked"}
    assert regions["at_step"].notna().tolist() == (regions["ended"] == "locked").tolist()


def test_ids_are_by_size(tmp_path):
    regions = pd.read_csv(run(config(tmp_path)) / "sweep_regions.csv")
    assert regions["id"].tolist() == list(range(1, len(regions) + 1))
    assert regions["area"].is_monotonic_decreasing


def test_regions_show_the_setting_that_changes(tmp_path):
    erosion = pd.read_csv(run(config(tmp_path, thresholds=[100], erosion_steps=2)) / "sweep_regions.csv")
    assert {"first_erosion", "last_erosion"} <= set(erosion.columns)
    assert not {"first_threshold", "first_upper"} & set(erosion.columns)


def test_loose_to_strict_says_which_seed_a_seed_split_from(tmp_path):
    regions = pd.read_csv(run(config(tmp_path, direction="loose_to_strict")) / "sweep_regions.csv")
    split = regions[regions["began"].str.startswith("split from")]
    assert len(split) > 0
    for _, row in split.iterrows():
        parent = int(row["began"].split()[-1])
        assert parent in set(regions["id"]) and parent != row["id"]
    assert set(regions["ended"]) <= {"disappeared", "reached the end"}


def test_the_metadata_says_how_it_was_made(tmp_path):
    folder = run(config(tmp_path))
    metadata = json.loads((folder / "sweep_metadata.json").read_text(encoding="utf-8"))
    assert metadata["direction"] == "strict_to_loose"
    assert metadata["steps"] == sorted(path.name for path in (folder / "steps").iterdir())
    assert [item["seed_id"] for item in metadata["seeds"]] == list(range(1, len(metadata["seeds"]) + 1))


@pytest.mark.parametrize("direction", ["strict_to_loose", "loose_to_strict"])
def test_save_every_iter_writes_the_seeds_after_each_step(tmp_path, direction):
    folder = run(config(tmp_path, direction=direction, save_every_iter=True))
    names = sorted(path.name for path in (folder / "intermediate" / direction).iterdir())
    assert names == ["after_step_000.tif", "after_step_001.tif", "after_step_002.tif"]
    last = 2 if direction == "strict_to_loose" else 0              # the last step of the combine
    assert np.array_equal(tifffile.imread(folder / "intermediate" / direction / f"after_step_00{last}.tif") > 0,
                          result(folder) > 0)


@pytest.mark.parametrize("direction, order", [("strict_to_loose", [200, 150, 100]), ("loose_to_strict", [100, 150, 200])])
def test_steps_are_in_the_order_of_the_combine(tmp_path, direction, order):
    steps = pd.read_csv(run(config(tmp_path, direction=direction)) / "sweep_steps.csv")
    assert steps["lower"].tolist() == order
    assert steps["step"].tolist() == [THRESHOLDS.index(t) for t in order]       # the number of its file
    by_threshold = dict(zip(steps["lower"], steps["regions"]))
    assert by_threshold == {t: ndimage.label(image() >= t)[0].max() for t in THRESHOLDS}


@pytest.mark.parametrize("direction", ["strict_to_loose", "loose_to_strict"])
def test_steps_count_the_seeds(tmp_path, direction):
    folder = run(config(tmp_path, direction=direction, top_n=2))
    steps, regions = pd.read_csv(folder / "sweep_steps.csv"), pd.read_csv(folder / "sweep_regions.csv")
    assert steps["new_seeds"].iloc[0] == steps["regions"].iloc[0] == steps["seeds_after"].iloc[0]
    assert steps["seeds_after"].iloc[-1] >= len(regions) == 2 == steps["in_final"].sum()    # top_n comes after
    assert (steps["in_final"] <= steps["new_seeds"]).all()


def test_regions_of_each_direction(tmp_path):
    folder = run(config(tmp_path, direction="both"))
    for suffix in ("_strict_to_loose", "_loose_to_strict"):
        for name in (f"FINAL_sweep_seed{suffix}.tif", f"sweep_regions{suffix}.csv", f"sweep_steps{suffix}.csv"):
            assert (folder / name).is_file()


def test_a_run_says_where_its_seeds_are(tmp_path):
    assert sweep.run_sweep_adaptive_seed_from_dict(config(tmp_path)) == [tmp_path / "out" / "img" / "FINAL_sweep_seed.tif"]


# ---- from an array, as napari runs it

def from_array(tmp_path, **changes):
    values = dict(img=image(), thresholds=list(THRESHOLDS), erosion_steps=0, segments=10,
                  output_folder=str(tmp_path / "array"), base_name="img")
    values.update(changes)
    return sweep.make_sweep_adaptive_seed(**values)


def test_an_array_gives_what_a_yaml_gives(tmp_path):
    seeds = from_array(tmp_path)["img_FINAL_sweep_seed"]
    assert np.array_equal(seeds, result(run(config(tmp_path))))
    assert seeds.dtype == np.uint16
    assert (tmp_path / "array" / "img" / "FINAL_sweep_seed.tif").is_file()
    assert (tmp_path / "array" / "img" / "sweep_regions.csv").is_file()


@pytest.mark.parametrize("changes", [dict(thresholds=THRESHOLDS[::-1]), dict(thresholds=[100], erosion_steps=2),
                                     dict(upper_thresholds=[255, 255, 255])],
                         ids=["ascending", "erosion", "upper"])
def test_an_array_in_each_mode(tmp_path, changes):
    values = {k: v for k, v in changes.items()}
    fresh = run(config(tmp_path, **values))
    assert np.array_equal(from_array(tmp_path, **changes)["img_FINAL_sweep_seed"], result(fresh))


def test_progress_ends_at_the_total(tmp_path):
    seen = []
    from_array(tmp_path, progress=lambda done, total, text: seen.append((done, total)))
    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]


@pytest.mark.parametrize("at", [0, 2, 3], ids=["start", "steps", "combine"])
def test_stop_leaves_no_final(tmp_path, at):
    seen = []
    with pytest.raises(sprout_core.Cancelled):
        from_array(tmp_path, cancel=lambda: len(seen) >= at, progress=lambda *a: seen.append(a))
    assert not (tmp_path / "array" / "img" / "FINAL_sweep_seed.tif").exists()


# ---- the combine alone

def nested_steps():
    img = ndimage.gaussian_filter(np.random.default_rng(0).random((24, 32, 32)), 1.5)
    return [measure.label(img >= np.quantile(img, q), connectivity=1) for q in (0.95, 0.85, 0.7, 0.5)]


@pytest.mark.parametrize("keep", [True, False])
def test_every_seed_of_the_metadata_is_in_the_image(keep):
    steps = [np.zeros((6, 6, 6), np.uint16) for _ in range(2)]
    steps[0][1:3, 1:3, 1:3] = 1
    steps[0][4:6, 4:6, 4:6] = 2
    steps[1][1:3, 1:3, 1:3] = 1                  # the second region disappears at the stricter step
    seeds, metadata = combine.adaptive_seed_l2s_from_seeds(steps, input_mode="label", keep_disappeared=keep)
    assert {item["seed_id"] for item in metadata} == set(np.unique(seeds)) - {0}
    assert all(item["area"] == np.count_nonzero(seeds == item["seed_id"]) for item in metadata)


@pytest.mark.parametrize("run_combine", [combine.adaptive_seed_s2l_from_seeds,
                                         lambda s, **k: combine.adaptive_seed_l2s_from_seeds(s[::-1], **k)],
                         ids=["s2l", "l2s"])
def test_metadata_on_a_nested_sweep(run_combine):
    seeds, metadata = run_combine(nested_steps(), input_mode="label", min_area=3)
    assert {item["seed_id"] for item in metadata} == set(np.unique(seeds)) - {0}
    assert all(item["area"] == np.count_nonzero(seeds == item["seed_id"]) for item in metadata)


def test_a_seed_is_only_cleared_where_it_still_is():
    """The voxels a seed has lost to another seed stay with that seed. Smallest case found by fuzzing."""
    steps = [np.array([[1, 2, 2], [1, 0, 1], [2, 0, 0]], np.uint16), np.array([[2, 0, 2], [0, 2, 2], [3, 2, 0]], np.uint16)]
    seeds, _ = combine.adaptive_seed_l2s_from_seeds(steps, input_mode="label", min_new_coverage=None, max_new_coverage=None)
    assert seeds.tolist() == [[1, 0, 1], [0, 1, 1], [2, 1, 0]]


def test_a_seed_that_comes_back_still_splits():
    """A region left out of one step (as by segments) and back at a stricter step updates its seed."""
    steps = [np.zeros((4, 12), np.uint16) for _ in range(3)]
    steps[0][:, 0:5] = 1
    steps[0][:, 7:12] = 2
    steps[1][:, 0:5] = 1                         # the second region is left out
    steps[2][1:3, 7:9] = 1                       # back, in two parts
    steps[2][1:3, 10:12] = 2
    seeds, metadata = combine.adaptive_seed_l2s_from_seeds(steps, input_mode="label")
    back = seeds[:, 7:12]
    assert len(set(np.unique(back)) - {0}) == 2 and np.count_nonzero(back) == 8
    status = {item["seed_id"]: item["status"] for item in metadata}
    assert status[seeds[1, 7]] == status[seeds[1, 10]] == "active"


def test_memory_does_not_grow_with_the_regions():
    shape = (64, 64, 64)
    step = np.zeros(shape, np.uint16)
    step[::4, ::4, ::4] = np.arange(1, 16 ** 3 + 1).reshape(16, 16, 16)    # 4096 regions
    tracemalloc.start()
    combine.adaptive_seed_s2l_from_seeds([step, step.copy()], input_mode="label")
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 200 * step.size                # a mask per region would be 4096 * size


# ---- steps on disk (memory_limit_gb)

@pytest.mark.parametrize("direction", ["strict_to_loose", "loose_to_strict", "both"])
def test_steps_from_disk_give_what_steps_in_memory_give(tmp_path, direction):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    memory = run(config(tmp_path / "a", direction=direction, num_threads=2))
    disk = run(config(tmp_path / "b", direction=direction, num_threads=2, memory_limit_gb=0))
    names = sorted(p.name for p in memory.iterdir() if p.is_file())        # used_config.yaml: each run's own
    assert names == sorted(p.name for p in disk.iterdir() if p.is_file())
    names.remove("used_config.yaml")
    for name in names:
        assert (memory / name).read_bytes() == (disk / name).read_bytes(), name


def test_the_limit_decides_where_the_steps_are(tmp_path, capsys):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    run(config(tmp_path / "a", verbose=True, memory_limit_gb=0))
    assert "steps read from disk one at a time" in capsys.readouterr().out
    run(config(tmp_path / "b", verbose=True, memory_limit_gb=1))
    assert "steps kept in memory" in capsys.readouterr().out


def test_steps_on_disk_take_less_memory(tmp_path):
    img = np.tile(image(), (2, 3, 3))
    tifffile.imwrite(tmp_path / "img.tif", img)
    thresholds = [220, 210, 200, 190, 180, 170, 160, 150, 140, 130]
    peaks = {}
    for limit in (None, 0):
        values = config(tmp_path, output_folder=str(tmp_path / f"out_{limit}"), thresholds=thresholds,
                        num_threads=1, memory_limit_gb=limit)
        tracemalloc.start()
        run(values)
        peaks[limit] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    steps = len(thresholds) * img.size                          # all steps in memory, in uint8
    assert peaks[0] < peaks[None] - steps / 2, (peaks, steps)


def test_the_plan_stays_under_the_limit():
    voxels, n_steps = 10 ** 6, 10
    steps, work = n_steps * voxels * sweep.STEP_BYTES, voxels * sweep.WORK_BYTES
    gb = 2 ** 30
    assert sweep._memory_plan(voxels, n_steps, 8, (steps + 8 * work) / gb, False) == (True, 8)
    assert sweep._memory_plan(voxels, n_steps, 8, (steps + 8 * work - 1) / gb, False) == (False, 8)
    assert sweep._memory_plan(voxels, n_steps, 8, 3.5 * work / gb, False) == (False, 3)
    assert sweep._memory_plan(voxels, n_steps, 8, 0, False) == (False, 1)        # always at least one
    assert sweep._memory_plan(voxels, n_steps, 8, None, False) == (True, 8)     # the default: as before


def test_by_default_the_steps_stay_in_memory(tmp_path, capsys):
    run(config(tmp_path, verbose=True, num_threads=2))
    out = capsys.readouterr().out
    assert "memory limit" not in out and "num_threads" not in out and "[WARNING]" not in out


def test_a_warning_when_the_steps_may_not_fit(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sweep.psutil, "virtual_memory", lambda: type("Memory", (), {"available": 1000})())
    run(config(tmp_path, verbose=True))
    assert "[WARNING]" in capsys.readouterr().out and (tmp_path / "out" / "img" / "FINAL_sweep_seed.tif").exists()


def test_out_of_memory_says_what_to_set(tmp_path, monkeypatch):
    def no_memory(*args, **kwargs):
        raise MemoryError("Unable to allocate 6.5 GiB")
    monkeypatch.setattr(sweep, "seed_sweep_threshold", no_memory)
    with pytest.raises(MemoryError, match="memory_limit_gb"):
        run(config(tmp_path))
    with pytest.raises(MemoryError, match="memory_limit_gb"):
        sweep.make_sweep_adaptive_seed(image(), list(THRESHOLDS), 0, 10, str(tmp_path / "napari"))


def test_fewer_threads_are_said(tmp_path, capsys):
    run(config(tmp_path, verbose=True, num_threads=2, memory_limit_gb=0))      # 2: every CI machine has the cores
    assert "num_threads 2 -> 1" in capsys.readouterr().out


# ---- configuration

def test_workspace_reaches_the_boundary(tmp_path, monkeypatch):
    boundary = boundary_file(tmp_path)
    values = config(tmp_path, workspace=str(tmp_path), img_path="img.tif", output_folder="out",
                    boundary_path="boundary.tif")
    monkeypatch.chdir(tmp_path.parent)
    sweep.run_sweep_adaptive_seed_from_dict(values)
    assert not result(tmp_path / "out" / "img")[boundary].any()


def test_workspace_in_seed_sweep_alone(tmp_path, monkeypatch):
    boundary = boundary_file(tmp_path)
    values = config(tmp_path, workspace=str(tmp_path), img_path="img.tif", output_folder="out",
                    boundary_path="boundary.tif")
    path = tmp_path / "sweep.yaml"
    keys = config_core.input_val_seed_sweep
    path.write_text(yaml.safe_dump({k: v for k, v in values.items() if k in keys}), encoding="utf-8")
    monkeypatch.chdir(tmp_path.parent)
    seed_sweep.run_seed_sweep(str(path))
    steps = sorted((tmp_path / "out" / "img").glob("step_*.tif"))
    assert steps and not tifffile.imread(steps[-1])[boundary].any()


def test_a_wrong_seeds_folder_is_an_error(tmp_path):
    values = config(tmp_path, seeds_folder=str(tmp_path / "no_such_folder"))
    with pytest.raises(FileNotFoundError, match="no_such_folder"):
        sweep.run_sweep_adaptive_seed_from_dict(values)


@pytest.mark.parametrize("changes", [dict(thresholds=THRESHOLDS), dict(thresholds=THRESHOLDS[::-1]),
                                     dict(thresholds=100, erosion_steps=3)],
                         ids=["descending", "ascending", "erosion"])
def test_seeds_folder_gives_what_a_fresh_run_gives(tmp_path, changes):
    fresh = run(config(tmp_path, **changes))
    again = config(tmp_path, output_folder=str(tmp_path / "again"), seeds_folder=str(fresh / "steps"), **changes)
    assert np.array_equal(result(fresh), result(run(again)))


def test_step_files_of_two_runs_are_refused(tmp_path):
    folder = run(config(tmp_path, output_folder=str(tmp_path / "a")))
    steps = folder / "steps"
    (steps / "step_003_thre_180_None_ero_0.tif").write_bytes((steps / "step_000_thre_200_None_ero_0.tif").read_bytes())
    values = config(tmp_path, output_folder=str(tmp_path / "b"), seeds_folder=str(steps))
    with pytest.raises(ValueError, match="monotonic"):
        sweep.run_sweep_adaptive_seed_from_dict(values)


def test_seeds_sweep_direction_still_decides(tmp_path):
    folder = run(config(tmp_path, output_folder=str(tmp_path / "a")))
    steps = folder / "steps"
    named = steps.parent / "own_steps"
    named.mkdir()
    for index, path in enumerate(sorted(steps.glob("step_*.tif"))):
        (named / f"step_{index}.tif").write_bytes(path.read_bytes())
    values = config(tmp_path, output_folder=str(tmp_path / "b"), seeds_folder=str(named))
    with pytest.raises(ValueError, match="seeds_sweep_direction"):
        sweep.run_sweep_adaptive_seed_from_dict(values)
    assert np.array_equal(result(folder), result(run({**values, "seeds_sweep_direction": "s2l"})))


def test_an_erosion_sweep_takes_one_upper_threshold(tmp_path):
    values = config(tmp_path, thresholds=[100], erosion_steps=2, upper_thresholds=[200, 255])
    with pytest.raises(ValueError, match="upper"):
        sweep.run_sweep_adaptive_seed_from_dict(values)


def test_combine_keys_exist_in_the_functions():
    known = {*inspect.signature(combine.adaptive_seed_s2l_from_seeds).parameters,
             *inspect.signature(combine.adaptive_seed_l2s_from_seeds).parameters}
    assert combine.COMBINE_SHARED_KEYS | combine.COMBINE_S2L_KEYS | combine.COMBINE_L2S_KEYS <= known


# ---- templates

def test_single_and_batch_templates_agree():
    single, batch = template(), template("batch_sweep_adaptive_seed.yaml")
    for key in single.keys() & batch.keys() - {"output_folder"}:
        assert single[key] == batch[key], key


def test_template_image_has_something_to_sweep():
    path = ROOT / template()["img_path"]
    assert len(np.unique(tifffile.imread(path)[::8, ::8, ::8])) > 2


def test_template_loads_whatever_the_locale(monkeypatch):
    real = builtins.open

    def locale_open(file, mode="r", *args, **kwargs):
        if "b" not in mode and not args and "encoding" not in kwargs:
            kwargs["encoding"] = "cp932"
        return real(file, mode, *args, **kwargs)

    seen = []
    monkeypatch.setattr(builtins, "open", locale_open)
    monkeypatch.setattr(sweep, "run_sweep_adaptive_seed_from_dict", seen.append)
    sweep.run_sweep_adaptive_seed(str(ROOT / "template" / "sweep_adaptive_seed.yaml"))
    assert seen


def test_a_console_without_arrows(tmp_path):
    values = config(tmp_path, verbose=True)
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    env = environment(tmp_path)
    env.update(PYTHONUTF8="0", PYTHONIOENCODING="cp1252")
    r = subprocess.run([sys.executable, str(ROOT / "sprout.py"), "--sweep_adaptive", "--config", str(path)],
                       cwd=ROOT, env=env, capture_output=True)
    assert r.returncode == 0, r.stderr[-1000:]


# ---- batch

def batch_files(tmp_path, csv, **changes):
    tifffile.imwrite(tmp_path / "img.tif", image())
    tifffile.imwrite(tmp_path / "other.tif", image())
    (tmp_path / "input.csv").write_text(csv.format(folder=tmp_path), encoding="utf-8")
    values = template("batch_sweep_adaptive_seed.yaml")
    values.update(csv_path=str(tmp_path / "input.csv"), output_folder=str(tmp_path / "out"), thresholds=list(THRESHOLDS),
                  segments=10, num_threads=1, min_size=1, top_n=None, verbose=False)
    values.update(changes)
    path = tmp_path / "batch.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return str(path)


def test_a_csv_column_overrides_the_config(tmp_path):
    path = batch_files(tmp_path, '''img_path,thresholds
{folder}/img.tif,"[200, 150]"
{folder}/other.tif,
''')
    batch_seeds.run_batch_sweep_adaptive_seed(path)
    assert len(list((tmp_path / "out" / "img" / "steps").glob("step_*.tif"))) == 2
    assert len(list((tmp_path / "out" / "other" / "steps").glob("step_*.tif"))) == 3     # a blank cell: the config


def test_a_failed_row_does_not_stop_the_batch(tmp_path):
    path = batch_files(tmp_path, '''img_path,thresholds
{folder}/img.tif,"[200, 100, 150]"
{folder}/other.tif,
''')
    batch_seeds.run_batch_sweep_adaptive_seed(path)
    final = tmp_path / "out" / "other" / "FINAL_sweep_seed.tif"
    assert final.is_file()
    report = pd.read_csv(next((tmp_path / "out").glob("*_results.csv")))
    assert report["error"].notna().tolist() == [True, False]
    assert report["final_output_path"].iloc[1] == str(final)


@pytest.mark.parametrize("route", ["array", "yaml"])
def test_a_boundary_of_another_shape_is_refused(tmp_path, route):
    boundary = np.zeros((20, 40, 41), np.uint8)
    boundary[:, :, 0] = 1
    if route == "array":
        call = lambda: from_array(tmp_path, boundary=boundary)  # noqa: E731
    else:
        tifffile.imwrite(tmp_path / "boundary.tif", boundary)
        call = lambda: run(config(tmp_path, boundary_path=str(tmp_path / "boundary.tif")))  # noqa: E731
    with pytest.raises(ValueError, match="same shape"):
        call()


def test_an_array_with_a_label_boundary_gives_what_a_yaml_gives(tmp_path):
    """napari passes the boundary layer as it is (0/1 integers), the YAML route casts it to bool."""
    boundary = np.zeros((20, 40, 40), np.uint8)
    boundary[:, 14, :] = 1
    tifffile.imwrite(tmp_path / "boundary.tif", boundary)
    seeds = from_array(tmp_path, boundary=boundary)["img_FINAL_sweep_seed"]
    assert np.array_equal(seeds, result(run(config(tmp_path, boundary_path=str(tmp_path / "boundary.tif")))))


@pytest.mark.parametrize("mode", ["threshold", "erosion"])
def test_the_step_functions_take_a_label_boundary_as_a_mask(tmp_path, mode):
    """Called directly, as from a script: 0/1 integers give what True/False gives."""
    boundary = np.zeros((20, 40, 40), np.uint8)
    boundary[:, 14, :] = 1

    def steps(b, name):
        if mode == "threshold":
            paths, _ = seed_sweep.seed_sweep_threshold(image(), str(tmp_path / name), list(THRESHOLDS), segments=10,
                                                       boundary=b, verbose=False)
        else:
            paths, _ = seed_sweep.seed_sweep_erosion(image(), str(tmp_path / name), 100, erosion_steps=2, segments=10,
                                                     boundary=b, verbose=False)
        return [tifffile.imread(p) for p in paths]

    as_labels, as_mask = steps(boundary, "labels"), steps(boundary.astype(bool), "mask")
    assert all(np.array_equal(a, b) for a, b in zip(as_labels, as_mask))
    assert not any((s[boundary > 0] > 0).any() for s in as_labels)


def test_the_step_functions_refuse_a_boundary_of_another_shape(tmp_path):
    with pytest.raises(ValueError, match="same shape"):
        seed_sweep.seed_sweep_threshold(image(), str(tmp_path / "s"), list(THRESHOLDS), segments=10,
                                        boundary=np.zeros((20, 40, 41), bool), verbose=False)
