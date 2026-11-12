"""
A parameter left out gives the same value from Python as from a YAML (and napari): the defaults of
the pipeline functions are those of the config rules in config_core.
"""
import ast
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sprout_core.config_core as config_core  # noqa: E402
from sprout_core import make_adaptive_seed, make_grow, make_seeds, make_sweep_adaptive_seed, seed_sweep  # noqa: E402

pytestmark = pytest.mark.core

FUNCTIONS = [(make_seeds.make_seeds, config_core.input_val_make_seeds),
             (make_grow.grow_mp, config_core.input_val_make_grow),
             (make_adaptive_seed.make_adaptive_seed_ero, config_core.input_val_make_adaptive_seed),
             (make_adaptive_seed.make_adaptive_seed_thre, config_core.input_val_make_adaptive_seed),
             (make_sweep_adaptive_seed.make_sweep_adaptive_seed, config_core.input_val_sweep_adaptive_seed),
             (seed_sweep.seed_sweep_threshold, config_core.input_val_seed_sweep),
             (seed_sweep.seed_sweep_erosion, config_core.input_val_seed_sweep)]
SAME = {"workspace": (None, "")}            # both: no folder in front of the paths
NOT_FROM_A_YAML = {("make_sweep_adaptive_seed", "base_name")}   # napari gives an array: no file name to use


def defaults(function):
    """The defaults of a function: its signature, or kwargs.get(name, default) for a **kwargs function."""
    found = {name: p.default for name, p in inspect.signature(function).parameters.items()
             if p.default is not inspect.Parameter.empty}
    for node in ast.walk(ast.parse(inspect.getsource(function).lstrip())):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "kwargs" and len(node.args) == 2
                and isinstance(node.args[0], ast.Constant)):
            try:
                found[node.args[0].value] = ast.literal_eval(node.args[1])
            except ValueError:                   # a default computed from other values
                pass
    return found


def same(name, a, b):
    a, b = (list(v) if isinstance(v, tuple) else v for v in (a, b))
    return a == b or {a, b} <= set(SAME.get(name, ()))


@pytest.mark.parametrize("function, rules", FUNCTIONS, ids=[f.__name__ for f, _ in FUNCTIONS])
def test_a_function_has_the_defaults_of_the_yaml(function, rules):
    differ = {name: (value, rules[name]["default"]) for name, value in defaults(function).items()
              if name in rules and "default" in rules[name] and (function.__name__, name) not in NOT_FROM_A_YAML
              and not same(name, value, rules[name]["default"])}
    assert not differ, f"{function.__name__}: (function, yaml) {differ}"
