"""
Guards for the code that came from nn_prob_map: what a merge with main could silently undo.

nninteractive_predict.py needs torch and nnInteractive, which CI does not install, so its
guards read the source. The real runs are in the `local` checks.
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sprout_core import seed_sweep  # noqa: E402


def function(name):
    tree = ast.parse((ROOT / "sprout_core" / "nninteractive_predict.py").read_text(encoding="utf-8"))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


@pytest.mark.core
@pytest.mark.parametrize("name", ["nninter_main", "nninter_predict"])
def test_random_seed_is_not_overwritten(name):
    """`seed` is the random seed of main: an image stored under that name would be passed on as it."""
    assigned = [n.id for n in ast.walk(function(name)) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]
    assert "seed" not in assigned


@pytest.mark.core
def test_session_comes_from_make_session():
    """make_session filters the arguments for nnInteractive 1.x and 2.x and sets deterministic cuDNN."""
    calls = [n for n in ast.walk(function("nninter_main")) if isinstance(n, ast.Call)]
    names = [c.func.id for c in calls if isinstance(c.func, ast.Name)]
    assert "make_session" in names and "SessionClass" not in names
    session = next(c for c in calls if isinstance(c.func, ast.Name) and c.func.id == "make_session")
    assert "session_cls" in [k.arg for k in session.keywords]


@pytest.mark.core
def test_sweep_keeps_ids_above_uint16(tmp_path):
    img = np.zeros((84, 84, 84), np.uint8)
    img[::2, ::2, ::2] = 200                  # 42 ** 3 = 74088 single voxels
    _, path, _ = seed_sweep._generate_one_threshold_step(img, 100, 255, [], 0, None, 0, tmp_path)
    seed = tifffile.imread(path)
    assert seed.max() == 42 ** 3 and np.unique(seed).size == 42 ** 3 + 1
