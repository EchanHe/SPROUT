"""
The YAML templates (template/, also what sprout --template writes) agree with the config rules and with
each other: a batch template runs the step like the single one, and its CSV is there.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sprout_core.config_core as config_core  # noqa: E402

TEMPLATES = ROOT / "template"
# single template, batch template, rules
STEPS = [("make_seeds", "batch_seeds", config_core.input_val_make_seeds),
         ("make_adaptive_seed", "batch_adaptive_seed", config_core.input_val_make_adaptive_seed),
         ("sweep_adaptive_seed", "batch_sweep_adaptive_seed", config_core.input_val_sweep_adaptive_seed),
         ("make_grow", "batch_grow", config_core.input_val_make_grow),
         ("nninteractive_predict", "batch_nninteractive", config_core.input_val_nninteractive_run),
         ("sam_predict", "batch_sam", config_core.input_val_sam_run)]
DEMO_ONLY = {"output_folder", "num_threads", "segments"}     # settings of each demo, not of the step


def load(name):
    return yaml.safe_load((TEMPLATES / f"{name}.yaml").read_text(encoding="utf-8")) or {}


@pytest.mark.core
@pytest.mark.parametrize("single, batch, rules", STEPS, ids=[s[1] for s in STEPS])
def test_every_key_is_a_parameter(single, batch, rules):
    unknown = [k for k in load(single) if k not in rules] + [k for k in load(batch) if k not in rules | {"csv_path": 0}]
    assert not unknown


@pytest.mark.core
@pytest.mark.parametrize("single, batch, rules", STEPS, ids=[s[1] for s in STEPS])
def test_a_batch_template_runs_the_step_like_the_single_one(single, batch, rules):
    one, many = load(single), load(batch)
    differ = {k: (one[k], many[k]) for k in one.keys() & many.keys() - DEMO_ONLY if one[k] != many[k]}
    assert not differ


@pytest.mark.core
@pytest.mark.parametrize("single, batch, rules", STEPS, ids=[s[1] for s in STEPS])
def test_the_csv_of_a_batch_template_is_there(single, batch, rules):
    csv = TEMPLATES / Path(load(batch)["csv_path"]).name
    assert load(batch)["csv_path"] == f"./template/{csv.name}" and csv.is_file()
    assert "img_path" in pd.read_csv(csv).columns
