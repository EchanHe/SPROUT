"""
What a run leaves in its result folder: used_config.yaml (every parameter, defaults filled in, a config
that makes the same result again), fixed file names (a run again in the same folder overwrites, it does
not pile up files with times in their names), and batch tables in output_folder.
"""
import contextlib
import fnmatch
import hashlib
import io
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
import yaml

from test_checks import environment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sprout_core  # noqa: E402
import sprout_core.config_core as config_core  # noqa: E402
from sprout_core import batch_seeds, make_adaptive_seed, make_grow, make_seeds, make_sweep_adaptive_seed  # noqa: E402

pytestmark = pytest.mark.core
TIME_IN_NAME = re.compile(r"\d{8}_\d{4}")


def image():
    img = np.zeros((10, 20, 30), np.uint8)
    img[2:8, 3:11, 3:11] = img[2:8, 3:11, 17:25] = 200
    img[4:6, 6:8, 11:17] = 150                                       # a bridge: one region at 100, two at 180
    return img


def seed():
    s = np.zeros(image().shape, np.uint8)
    s[3:7, 5:9, 5:9], s[3:7, 5:9, 19:23] = 1, 2
    return s


# step: (YAML runner, rules, YAML, result folder)
STEPS = {
    "seeds": (make_seeds.run_make_seeds, config_core.input_val_make_seeds,
              dict(img_path="img.tif", thresholds=[100], erosion_steps=1, segments=5, num_threads=1,
                   output_folder="out"), "out/img"),
    "adaptive_ero": (make_adaptive_seed.run_make_adaptive_seed, config_core.input_val_make_adaptive_seed,
                     dict(img_path="img.tif", thresholds=[100], erosion_steps=2, segments=5, num_threads=1,
                          output_folder="out"), "out/img"),
    "adaptive_thre": (make_adaptive_seed.run_make_adaptive_seed, config_core.input_val_make_adaptive_seed,
                      dict(img_path="img.tif", thresholds=[100, 180], erosion_steps=0, segments=5, num_threads=1,
                           output_folder="out"), "out/img"),
    "sweep": (make_sweep_adaptive_seed.run_sweep_adaptive_seed, config_core.input_val_sweep_adaptive_seed,
              dict(img_path="img.tif", thresholds=[180, 100], erosion_steps=0, segments=5, num_threads=1,
                   output_folder="out", verbose=False), "out/img"),
    "grow": (make_grow.run_make_grow, config_core.input_val_make_grow,
             dict(img_path="img.tif", seg_path="seed.tif", thresholds=[100], dilation_steps=[3], touch_rule="stop",
                  num_threads=1, output_folder="out"), "out/img"),
}


@pytest.fixture
def here(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tifffile.imwrite(tmp_path / "img.tif", image())
    tifffile.imwrite(tmp_path / "seed.tif", seed())
    return tmp_path


def run(runner, config_path):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        runner(str(config_path))
    return out.getvalue()


def files(folder):
    """The results; not the logs, which hold the times and full paths of the run."""
    return {p.relative_to(folder).as_posix(): hashlib.sha1(p.read_bytes()).hexdigest()
            for p in sorted(Path(folder).rglob("*")) if p.is_file() and p.suffix in (".tif", ".csv", ".json")
            and p.name != make_seeds.SEED_LOG and not p.name.startswith("grow_log_")}


def load(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


@pytest.mark.parametrize("step", STEPS)
def test_used_config_is_a_config_of_every_parameter(here, step):
    runner, rules, values, result = STEPS[step]
    (here / "run.yaml").write_text(yaml.safe_dump(values), encoding="utf-8")
    run(runner, here / "run.yaml")
    used = here / result / "used_config.yaml"
    assert used.read_text(encoding="utf-8").startswith(f"# SPROUT {sprout_core.__version__}, ")
    config = load(used)
    assert list(config) == list(rules)
    assert all(config[k] == v for k, v in values.items() if k != "thresholds") and \
        np.ravel(config["thresholds"]).tolist() == values["thresholds"]
    config_core.validate_input_yaml(dict(config), rules)


@pytest.mark.parametrize("step", STEPS)
def test_used_config_makes_the_same_result_again(here, step):
    runner, rules, values, result = STEPS[step]
    (here / "run.yaml").write_text(yaml.safe_dump(values), encoding="utf-8")
    run(runner, here / "run.yaml")
    first = files(here / result)
    printed = run(runner, here / result / "used_config.yaml")         # as it is: into the same folder
    assert files(here / result) == first
    assert "has results of an earlier run" in printed


@pytest.mark.parametrize("step", STEPS)
def test_a_run_again_leaves_no_files_with_times_in_their_names(here, step):
    runner, rules, values, result = STEPS[step]
    (here / "run.yaml").write_text(yaml.safe_dump(values), encoding="utf-8")
    run(runner, here / "run.yaml")
    run(runner, here / "run.yaml")
    names = [p.name for p in (here / "out").rglob("*") if p.is_file()]
    assert not [n for n in names if TIME_IN_NAME.search(n) or n.startswith(("output_dict", "config_"))
                or n.endswith(".csv.png")], names


DOCS = {"seeds": "config_seed.md", "adaptive_ero": "config_adaptive_seed.md", "adaptive_thre": "config_adaptive_seed.md",
        "sweep": "config_sweep_adaptive_seed.md", "grow": "config_grow.md"}


def documented(page):
    """The files in the Output table of a docs page, as patterns: `<name>` and NNN match anything."""
    section = (ROOT / "docs" / page).read_text(encoding="utf-8").split("\n## Output", 1)[1].split("\n## ", 1)[0]
    table = re.search(r"(?:^\|.*\n?)+", section, re.M).group(0)       # the first table
    names = [re.search(r"`([^`]+)`", row.split("|")[1]).group(1) for row in table.splitlines()
             if row.startswith("| `")]
    patterns = [re.sub(r"<[^>]+>|NNN", "*", n) for n in names]
    return [p + "*" if p.endswith("/") else "*" + p if p.startswith(".") else p for p in patterns]


@pytest.mark.parametrize("step", STEPS)
def test_every_file_of_a_run_is_in_its_docs(here, step):
    runner, _, values, result = STEPS[step]
    (here / "run.yaml").write_text(yaml.safe_dump(values), encoding="utf-8")
    run(runner, here / "run.yaml")
    patterns = documented(DOCS[step])
    written = [p.relative_to(here / result).as_posix() for p in (here / result).rglob("*") if p.is_file()]
    assert written and not [w for w in written if not any(fnmatch.fnmatch(w, p) for p in patterns)], patterns


def test_the_seed_log_holds_one_run(here):
    runner, _, values, result = STEPS["seeds"]
    (here / "run.yaml").write_text(yaml.safe_dump(values), encoding="utf-8")
    run(runner, here / "run.yaml")
    once = [len(load(p)) for p in sorted((here / result).rglob("seed_log.json"))]
    run(runner, here / "run.yaml")
    assert once and [len(load(p)) for p in sorted((here / result).rglob("seed_log.json"))] == once


# napari gives arrays; its used_config, with the image as a file, makes the same result from a YAML
NAPARI = {
    "seeds": lambda out: make_seeds.make_seeds(img=image(), thresholds=[100], erosion_steps=1, segments=5,
                                               num_threads=1, output_folder=out, base_name="img",
                                               footprints="ball", return_for_napari=True),
    "adaptive_ero": lambda out: make_adaptive_seed.make_adaptive_seed_ero(
        img=image(), threshold=100, erosion_steps=2, segments=5, num_threads=1, output_folder=out, base_name="img",
        footprints="ball", min_size=1, min_split_ratio=0, save_every_iter=False, return_for_napari=True),
    "sweep": lambda out: make_sweep_adaptive_seed.make_sweep_adaptive_seed(
        img=image(), thresholds=[180, 100], erosion_steps=0, segments=5, output_folder=out, base_name="img"),
    "grow": lambda out: make_grow.grow_mp(img=image(), seg=seed(), thresholds=[100], dilation_steps=[3],
                                          touch_rule="stop", num_threads=1, output_folder=out, base_name="img",
                                          return_for_napari=True),
}


@pytest.mark.parametrize("step", NAPARI)
def test_napari_used_config_tells_how_the_result_was_made(here, step):
    with contextlib.redirect_stdout(io.StringIO()):
        NAPARI[step](str(here / "napari"))
    config = load(here / "napari" / "img" / "used_config.yaml")
    assert config["img_path"] is None                                   # an array: no path
    config.update(img_path="img.tif", output_folder="yaml", **({"seg_path": "seed.tif"} if step == "grow" else {}))
    (here / "again.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    run(STEPS[step][0], here / "again.yaml")
    napari, again = files(here / "napari" / "img"), files(here / "yaml" / "img")
    assert napari == again


def test_grow_finishes_without_matplotlib(here):
    """pip install sprout-seg has no matplotlib: the grow plot is left out, the run still ends well."""
    (here / "run.yaml").write_text(yaml.safe_dump(STEPS["grow"][2]), encoding="utf-8")
    code = ("import importlib.abc, sys\n"
            "class Block(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] == 'matplotlib':\n"
            "            raise ModuleNotFoundError(name)\n"
            "sys.meta_path.insert(0, Block())\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from sprout_core.make_grow import run_make_grow\n"
            "run_make_grow('run.yaml')\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=here, env=environment(here), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "pip install matplotlib" in r.stdout
    assert (here / "out" / "img" / "FINAL_GROW_img.tif").is_file() and not list((here / "out").rglob("*.png"))


def test_a_batch_table_is_in_output_folder(here):
    (here / "rows.csv").write_text("img_path\nimg.tif\n", encoding="utf-8")
    (here / "batch.yaml").write_text(yaml.safe_dump(dict(csv_path="rows.csv", thresholds=[100], erosion_steps=1,
                                                         segments=5, num_threads=1, output_folder="out")),
                                     encoding="utf-8")
    run(batch_seeds.run_batch_seeds, here / "batch.yaml")
    assert pd.read_csv(here / "out" / "rows_results.csv")["img_path"].tolist() == ["img.tif"]
    assert sorted(p.name for p in here.iterdir() if p.is_file()) == ["batch.yaml", "img.tif", "rows.csv", "seed.tif"]
