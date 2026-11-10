"""
Packaging: sprout-seg (sprout_core) and napari-sprout find each other by installation, not by folders.
The wheels themselves are checked in a clean environment by clean_install.py (CI job "package").
"""
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

from test_checks import environment

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "napari-sprout"
AI_LIBRARIES = ("torch", "monai", "segment_anything", "sam2", "nnInteractive", "sklearn")
AI_ONLY = {"batch_nninteractive", "batch_sam", "nni_prob_session", "nninteractive_predict", "sam_predict",
           "sprout_prompt_core"}   # they need the AI libraries ("sprout-seg[ai]", SAM)

# A Python without the AI libraries, also where they are installed
BLOCK = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in %r:
            raise ModuleNotFoundError(name)
sys.meta_path.insert(0, Block())
""" % (AI_LIBRARIES,)


def python(code, cwd, tmp_path):
    return subprocess.run([sys.executable, "-c", code], cwd=cwd, env=environment(tmp_path), capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=600)


def version_of(path):
    return re.search(r'^__version__ = "(.+)"', path.read_text(encoding="utf-8"), re.M).group(1)


@pytest.mark.core
def test_the_plugin_has_no_path_tricks():
    found = [f"{p.relative_to(ROOT)}:{i}" for p in (PLUGIN / "napari_sprout").rglob("*.py")
             for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if "sys.path" in line]
    assert not found


@pytest.mark.core
def test_one_version():
    core = version_of(ROOT / "sprout_core" / "__init__.py")
    assert version_of(PLUGIN / "napari_sprout" / "__init__.py") == core
    minor = ".".join(core.split(".")[:2])
    needs = re.findall(r'"sprout-seg(?:\[\w+\])?~=([\d.]+)"', (PLUGIN / "pyproject.toml").read_text(encoding="utf-8"))
    assert needs and all(n.startswith(minor + ".") for n in needs), needs


@pytest.mark.core
def test_python_sprout_py_still_works(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "sprout.py"), "--version"], cwd=tmp_path, env=environment(tmp_path),
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0 and version_of(ROOT / "sprout_core" / "__init__.py") in r.stdout, r.stdout + r.stderr


@pytest.mark.core
def test_modules_import_without_the_ai_libraries(tmp_path):
    names = sorted(p.stem for p in (ROOT / "sprout_core").glob("*.py") if p.stem not in AI_ONLY | {"__init__"})
    code = BLOCK + f"sys.path.insert(0, {str(ROOT)!r})\nimport importlib\n" + \
        "".join(f"importlib.import_module('sprout_core.{n}')\n" for n in names)
    r = python(code, tmp_path, tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]


@pytest.mark.core
def test_nninteractive_without_the_ai_libraries_says_what_to_install(tmp_path):
    (tmp_path / "config.yaml").write_text("img_path: x.tif\n", encoding="utf-8")
    code = BLOCK + f"sys.path.insert(0, {str(ROOT)!r})\n" \
        "sys.argv = ['sprout', '--prompt', '--nninteractive', '--config', 'config.yaml']\n" \
        "from sprout_core.cli import main\nmain()\n"
    r = python(code, tmp_path, tmp_path)
    assert r.returncode == 1 and 'pip install "sprout-seg[ai]"' in r.stdout, r.stdout + r.stderr[-2000:]


@pytest.mark.napari
def test_every_panel_from_the_repository_folder(tmp_path):
    """Started in the clone (python -m napari, a notebook there), napari finds the installed plugin."""
    if importlib.util.find_spec("napari") is None:
        pytest.skip("needs napari")
    code = ("import napari_sprout, npe2\n"
            "pm = npe2.PluginManager.instance(); pm.discover()\n"
            "print(napari_sprout.__file__)\n"
            "print(len(pm.get_manifest('napari-sprout').contributions.widgets))\n")
    r = python(code, ROOT, tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    file, widgets = r.stdout.split()[-2:]
    assert file.endswith("__init__.py") and int(widgets) == 5, r.stdout
