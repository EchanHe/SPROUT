"""
The classic adaptive seed runs on one thread since 1.1.0 (faster than it was on many): a config needs no
num_threads, an older config that has it still runs (with a note) and makes the same result, and
used_config.yaml leaves it out.
"""
import contextlib
import io
import sys
from pathlib import Path

import pytest
import yaml

from test_batch import batch_yaml
from test_outputs import STEPS, files, here, load, run  # noqa: F401  (here is a fixture)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sprout_core import batch_seeds  # noqa: E402

pytestmark = pytest.mark.core
NOTE = "'num_threads' is not used"


def write(path, values):
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


@pytest.mark.parametrize("step", ["adaptive_ero", "adaptive_thre"])
def test_a_config_without_num_threads_runs(here, step):
    runner, _, values, result = STEPS[step]
    values = {k: v for k, v in values.items() if k != "num_threads"}
    printed = run(runner, write(here / "run.yaml", values))
    assert files(here / result) and NOTE not in printed


@pytest.mark.parametrize("step", ["adaptive_ero", "adaptive_thre"])
def test_an_older_config_with_num_threads_gives_a_note_and_the_same_result(here, step):
    runner, _, values, result = STEPS[step]
    run(runner, write(here / "run.yaml", {k: v for k, v in values.items() if k != "num_threads"}))
    without = files(here / result)
    printed = run(runner, write(here / "run.yaml", dict(values, num_threads=8)))
    assert NOTE in printed and files(here / result) == without
    assert "num_threads" not in load(here / result / "used_config.yaml")


def test_a_batch_without_num_threads_runs(tmp_path):
    path = Path(batch_yaml(tmp_path, "make_adaptive_seed.yaml", thresholds=[150, 100]))
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config.pop("num_threads")
    write(path, config)
    with contextlib.redirect_stdout(io.StringIO()):
        batch_seeds.run_batch_adaptive_seed(str(path))
    assert (tmp_path / "out" / "input_results.csv").is_file()
    assert "error" not in (tmp_path / "out" / "input_results.csv").read_text(encoding="utf-8").splitlines()[1]


@pytest.mark.parametrize("template", ["make_adaptive_seed.yaml", "batch_adaptive_seed.yaml"])
def test_the_adaptive_templates_have_no_num_threads(template):
    assert "num_threads" not in (ROOT / "template" / template).read_text(encoding="utf-8")
