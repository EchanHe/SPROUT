"""
Batch runs: a row that fails is written to the results CSV, also when no row created the output folder.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sprout_core import batch_grow  # noqa: E402
from sprout_core import batch_seeds  # noqa: E402
from sprout_core import make_adaptive_seed  # noqa: E402
from sprout_core import make_grow  # noqa: E402
from sprout_core import make_seeds  # noqa: E402

pytestmark = pytest.mark.core


def fail(**kwargs):
    raise RuntimeError("this row fails")


def batch_yaml(tmp_path, template, **changes):
    img = np.zeros((8, 8, 8), np.uint8)
    img[2:6, 2:6, 2:6] = 200
    tifffile.imwrite(tmp_path / "img.tif", img)
    tifffile.imwrite(tmp_path / "seg.tif", (img > 0).astype(np.uint8))
    rows = {"img_path": [str(tmp_path / "img.tif")]}
    if template == "make_grow.yaml":
        rows["seg_path"] = [str(tmp_path / "seg.tif")]
    pd.DataFrame(rows).to_csv(tmp_path / "input.csv", index=False)
    config = yaml.safe_load(open(ROOT / "template" / template, encoding="utf-8"))
    config.pop("img_path", None)
    config.pop("seg_path", None)
    config.update(csv_path=str(tmp_path / "input.csv"), output_folder=str(tmp_path / "out"), num_threads=1, **changes)
    yaml.safe_dump(config, open(tmp_path / "config.yaml", "w"))
    return str(tmp_path / "config.yaml")


@pytest.mark.parametrize("run, module, function, template, changes", [
    (batch_seeds.run_batch_seeds, make_seeds, "make_seeds", "make_seeds.yaml", {}),
    (batch_seeds.run_batch_adaptive_seed, make_adaptive_seed, "make_adaptive_seed_thre", "make_adaptive_seed.yaml", {}),
    (batch_grow.run_batch_grow, make_grow, "grow_mp", "make_grow.yaml",
     dict(dilation_steps=[2, 2], save_every_n_iters=5)),
], ids=["seeds", "adaptive_seed", "grow"])
def test_the_error_of_a_row_is_written(tmp_path, monkeypatch, run, module, function, template, changes):
    monkeypatch.setattr(module, function, fail)
    run(batch_yaml(tmp_path, template, thresholds=[150, 100], **changes))
    results = list((tmp_path / "out").glob("input.csv_running_results_*.csv"))
    assert len(results) == 1
    assert pd.read_csv(results[0])["error"].tolist() == ["this row fails"]


def test_grow_without_save_every_n_iters(tmp_path):
    config = batch_yaml(tmp_path, "make_grow.yaml", thresholds=[150, 100], dilation_steps=[2, 2])
    assert "save_every_n_iters" not in yaml.safe_load(open(config, encoding="utf-8"))   # optional
    batch_grow.run_batch_grow(config)
    results = pd.read_csv(next((tmp_path / "out").glob("input.csv_running_results_*.csv")))
    assert "error" not in results and Path(results["final_output_path"][0]).is_file()
