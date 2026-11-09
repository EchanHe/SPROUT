"""
Sweep in the Seeds tab of napari. Each case runs in its own process, see probe_sweep_tab.py.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from test_checks import environment

TESTS = Path(__file__).resolve().parent

PROBES = [
    "strict_to_loose_thresholds",
    "loose_to_strict_thresholds",
    "erosion_sweep",
    "stop_leaves_no_final",
    "thresholds_out_of_order_are_refused",
    "one_row_is_refused_by_the_threshold_sweep",
    "two_rows_are_refused_by_the_erosion_sweep",
    "the_erosion_sweep_needs_an_erosion_step",
    "each_method_greys_what_it_does_not_use",
]


@pytest.mark.napari
@pytest.mark.parametrize("probe", PROBES)
def test_sweep_tab(probe, tmp_path):
    if importlib.util.find_spec("napari") is None:
        pytest.skip("needs napari")
    r = subprocess.run([sys.executable, str(TESTS / "probe_sweep_tab.py"), probe], cwd=tmp_path,
                       env=environment(tmp_path), capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=300)
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
