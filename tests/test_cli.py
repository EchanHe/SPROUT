"""
The sprout command: --template writes the YAML of a step here (no clone needed), and a mistake gets
a short message that says where to look, not a traceback.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from test_checks import environment

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "template"
DOCS = "https://github.com/EchanHe/SPROUT"


def sprout(tmp_path, *args):
    return subprocess.run([sys.executable, str(ROOT / "sprout.py"), *args], cwd=tmp_path, env=environment(tmp_path),
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)


@pytest.mark.core
@pytest.mark.parametrize("step, name", [("seeds", "make_seeds.yaml"), ("adaptive_seed", "make_adaptive_seed.yaml"),
                                        ("sweep_adaptive", "sweep_adaptive_seed.yaml"), ("grow", "make_grow.yaml"),
                                        ("nninteractive", "nninteractive_predict.yaml"), ("sam", "sam_predict.yaml")])
def test_template_writes_the_yaml_of_the_step(tmp_path, step, name):
    r = sprout(tmp_path, "--template", step)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / name).read_bytes() == (TEMPLATES / name).read_bytes()
    assert name in r.stdout


@pytest.mark.core
@pytest.mark.parametrize("step, name, csv", [("seeds", "batch_seeds.yaml", "seeds_input.csv"),
                                             ("adaptive_seed", "batch_adaptive_seed.yaml", "seeds_input.csv"),
                                             ("sweep_adaptive", "batch_sweep_adaptive_seed.yaml",
                                              "sweep_adaptive_seeds_input.csv"),
                                             ("grow", "batch_grow.yaml", "grow_input.csv"),
                                             ("nninteractive", "batch_nninteractive.yaml", "nninteractive_input.csv"),
                                             ("sam", "batch_sam.yaml", "sam_input.csv")])
def test_batch_template_writes_the_yaml_and_its_csv_side_by_side(tmp_path, step, name, csv):
    r = sprout(tmp_path, "--template", step, "--batch")
    assert r.returncode == 0, r.stdout + r.stderr
    assert f'csv_path: "./{csv}"' in (tmp_path / name).read_text(encoding="utf-8")
    assert (tmp_path / csv).read_bytes() == (TEMPLATES / csv).read_bytes()


@pytest.mark.core
def test_template_does_not_overwrite_a_file(tmp_path):
    (tmp_path / "make_seeds.yaml").write_text("mine", encoding="utf-8")
    r = sprout(tmp_path, "--template", "seeds")
    assert r.returncode == 1 and "Traceback" not in r.stderr
    assert (tmp_path / "make_seeds.yaml").read_text(encoding="utf-8") == "mine"


@pytest.mark.core
@pytest.mark.parametrize("args, says", [(["--seeds", "--config", "missing.yaml"], "sprout --template seeds"),
                                        (["--grow", "--config", "grow.txt"], ".yaml"),
                                        (["--seeds"], "--config"),
                                        ([], DOCS),
                                        (["--template", "segment"], "invalid choice")],
                         ids=["missing", "not yaml", "no config", "no step", "unknown step"])
def test_a_mistake_gets_a_short_message(tmp_path, args, says):
    r = sprout(tmp_path, *args)
    out = (r.stdout + r.stderr).strip()
    assert r.returncode != 0 and "Traceback" not in out and says in out, out
    assert len(out.splitlines()) <= 2, out


@pytest.mark.core
def test_help_says_where_the_templates_and_docs_are(tmp_path):
    r = sprout(tmp_path, "--help")
    assert r.returncode == 0 and "sprout --template" in r.stdout and DOCS in r.stdout
