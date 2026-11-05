"""
Runs every tests/check_*.py as one pytest test.

The checks are plain scripts: they print PASS/FAIL lines and exit non-zero if anything failed.
Each one runs here in its own process, in a temp folder, and passes only if

    - it exits with 0 and prints ALL PASS...,
    - no line starts with FAIL,
    - it printed at least `min_pass` PASS lines (a script that quietly skipped most of its
      checks also ends with ALL PASSED),
    - `git status` is the same as before (it wrote nothing into the repo).

    pytest                run everything this machine can run
    pytest -m core        the checks that need no napari (what the CLI users have)
    pytest -m napari      the checks of the plugin
    pytest -m golden      outputs compared with tests/data/golden_*.json
    pytest --full         without the shortcuts (--quick) of the slow checks

The full output of each check goes to test-logs/.
"""
import importlib.util
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
LOGS = ROOT / "test-logs"
QT_NOISE = ("WARNING:", "Note that Qt no longer ships fonts")
QT_PLATFORM = "cocoa" if sys.platform == "darwin" else "offscreen"   # no GL context offscreen: a macOS crash


@dataclass
class Check:
    script: str
    mark: str                 # core, napari, golden or local
    min_pass: int
    args: tuple = ()
    quick: tuple = ()         # left out with --full
    timeout: int = 600        # seconds
    repeat: int = 1           # for failures that only show now and then
    needs: tuple = ()         # modules; without them the check is skipped


NAPARI = ("napari",)
CHECKS = [
    # the core alone
    Check("check_adaptive_seed.py", "core", 12, timeout=1800),
    Check("check_footprints.py", "core", 10),
    Check("check_grow_determinism.py", "core", 10),
    Check("check_label_dtype.py", "core", 14),
    # the plugin
    Check("check_boundary_tab.py", "napari", 26, needs=NAPARI),
    Check("check_boundary_warning.py", "napari", 13, needs=NAPARI),
    Check("check_core_fixes.py", "napari", 12, needs=NAPARI),
    Check("check_docks.py", "napari", 67, needs=NAPARI),
    Check("check_edit_round2.py", "napari", 133, quick=("--quick",), timeout=1800, needs=NAPARI),
    Check("check_info_widget.py", "napari", 9, needs=NAPARI),
    Check("check_jobs.py", "napari", 58, needs=NAPARI),
    Check("check_layer_lock.py", "napari", 14, needs=NAPARI),
    Check("check_light_exit.py", "napari", 6, repeat=5, needs=NAPARI),
    Check("check_output_folder.py", "napari", 7, needs=NAPARI),
    Check("check_panel_fixes.py", "napari", 57, needs=NAPARI),
    Check("check_polish_docks.py", "napari", 33, needs=NAPARI),
    Check("check_polish_panel.py", "napari", 37, needs=NAPARI),
    Check("check_prompt_core.py", "napari", 44, needs=NAPARI),
    Check("check_prompt_tab.py", "napari", 111, needs=NAPARI),
    Check("check_result_layer_dtype.py", "napari", 6, needs=NAPARI),
    Check("check_roi_core.py", "napari", 105, quick=("--quick",), needs=NAPARI),
    Check("check_roi_layers.py", "napari", 39, needs=NAPARI),
    Check("check_roi_widget.py", "napari", 85, needs=NAPARI),
    Check("check_seedgrow_ui.py", "napari", 53, needs=NAPARI),
    Check("check_threshold_preview.py", "napari", 28, needs=NAPARI),
    # outputs against the stored hashes
    Check("check_core_flow.py", "golden", 2),
    Check("check_roi_off_golden.py", "golden", 3, args=("--check",), needs=NAPARI),
    # this machine only: a GPU with the model weights / speed against an older commit
    Check("check_prompt_nni.py", "local", 28, needs=("nnInteractive",)),
    Check("check_core_speed.py", "local", 11),
]


def missing(check):
    return [m for m in check.needs if importlib.util.find_spec(m) is None]


def git_status():
    try:
        return subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, capture_output=True,
                              text=True, encoding="utf-8").stdout
    except OSError:
        return None


def environment(tmp):
    env = dict(os.environ)
    env.update(QT_QPA_PLATFORM=QT_PLATFORM, MPLBACKEND="Agg", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               CUDA_VISIBLE_DEVICES="", TMP=str(tmp), TEMP=str(tmp), TMPDIR=str(tmp),
               NAPARI_CONFIG=str(tmp / "napari.yaml"), XDG_CONFIG_HOME=str(tmp / "config"))
    if os.environ.get("CI"):
        env["SPROUT_NO_TIMING"] = "1"
    return env


def verdict(returncode, out, min_pass):
    """Why a run of a check failed; an empty list if it passed."""
    lines = out.splitlines()
    problems = [line for line in lines if line.startswith("FAIL")]
    passed = sum(line.startswith("PASS") for line in lines)
    if returncode != 0:
        problems.append(f"exit code {returncode}")
    if not any(re.match(r"ALL PASS", line) for line in lines):
        problems.append("no ALL PASS line")
    if passed < min_pass:
        problems.append(f"only {passed} PASS lines, expected at least {min_pass}")
    return problems


def run_once(check, args, tmp):
    command = [sys.executable, str(TESTS / check.script), *args]
    try:
        r = subprocess.run(command, cwd=tmp, env=environment(tmp), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=check.timeout)
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return out, [f"no result after {check.timeout} s"]
    return r.stdout + r.stderr, verdict(r.returncode, r.stdout, check.min_pass)


@pytest.mark.parametrize(
    "check", [pytest.param(c, id=c.script[len("check_"):-len(".py")], marks=getattr(pytest.mark, c.mark))
              for c in CHECKS])
def test_check(check, tmp_path, request):
    if missing(check):
        pytest.skip(f"needs {', '.join(missing(check))}")
    args = list(check.args)
    if not request.config.getoption("--full"):
        args += check.quick
    before = git_status()
    LOGS.mkdir(exist_ok=True)
    for i in range(check.repeat):
        out, problems = run_once(check, args, tmp_path)
        (LOGS / f"{check.script[:-3]}.log").write_text(out, encoding="utf-8")
        if problems:
            tail = "\n".join([line for line in out.splitlines() if not line.startswith(QT_NOISE)][-60:])
            pytest.fail(f"{check.script} (run {i + 1} of {check.repeat}):\n  " + "\n  ".join(problems)
                        + f"\n--- last lines of the output ---\n{tail}", pytrace=False)
    after = git_status()
    assert after == before, f"{check.script} changed the repo:\n{after}"


@pytest.mark.core
def test_every_check_is_listed():
    on_disk = {p.name for p in TESTS.glob("check_*.py")}
    listed = [c.script for c in CHECKS]
    assert len(listed) == len(set(listed)), "a check is listed twice"
    assert on_disk == set(listed), f"not in CHECKS: {on_disk - set(listed)}; no such file: {set(listed) - on_disk}"


@pytest.mark.core
@pytest.mark.parametrize("returncode, out, why", [
    (0, "PASS a\nPASS b\n\nALL PASSED\n", []),
    (1, "PASS a\nPASS b\nFAIL c\n\n1 FAILED\n", ["FAIL c", "exit code 1", "no ALL PASS line"]),
    (0, "PASS a\nPASS b\n", ["no ALL PASS line"]),
    (0, "PASS a\n\nALL PASSED\n", ["only 1 PASS lines, expected at least 2"]),
    (0, "PASS a\nPASS b\nFAIL c\n\nALL PASSED\n", ["FAIL c"]),
    (139, "PASS a\nPASS b\n\nALL PASSED\n", ["exit code 139"]),
])
def test_verdict(returncode, out, why):
    assert verdict(returncode, out, 2) == why
