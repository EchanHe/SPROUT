"""
The two wheels in a clean environment, away from the repository (CI job "package").

    python -m build --outdir dist . && python -m build --outdir dist napari-sprout
    python tests/clean_install.py dist

Checks what each wheel holds, installs both in a new venv (napari-sprout[all]: with napari and Qt) and,
from an empty folder, runs the sprout command and opens every SPROUT panel. Exit code 0: all good.
"""
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import tifffile

TESTS = Path(__file__).resolve().parent
CORE_MODULES = ("make_seeds", "make_grow", "make_adaptive_seed", "make_sweep_adaptive_seed", "batch_seeds",
                "batch_grow", "cli", "sprout_core", "config_core")
FAILED = []

PANELS = r"""
import os, sys
import _headless_gl
import napari, npe2, sprout_core, napari_sprout
venv = os.path.realpath(sys.argv[1])            # macOS: /var is a link to /private/var
for m in (sprout_core, napari_sprout):
    assert os.path.commonpath([venv, os.path.realpath(m.__file__)]) == venv, m.__file__   # the installed copy
pm = npe2.PluginManager.instance()
pm.discover()
names = [w.display_name for w in pm.get_manifest("napari-sprout").contributions.widgets]
viewer = napari.Viewer(show=False)
opened = [n for n in names if viewer.window.add_plugin_dock_widget("napari-sprout", n)[1] is not None]
print("panels:", opened, flush=True)
os._exit(0 if opened == names and len(names) == 5 else 1)
"""


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILED.append(msg)


def tops(names):
    return sorted({n.split("/")[0] for n in names if ".dist-info/" not in n})


def check_wheels(dist):
    core, plugin = next(dist.glob("sprout_seg-*.whl")), next(dist.glob("napari_sprout-*.whl"))
    with zipfile.ZipFile(core) as z:
        names = z.namelist()
        entry = next(z.read(n).decode() for n in names if n.endswith("entry_points.txt"))
    ok(tops(names) == ["sprout_core"], f"{core.name} holds only sprout_core: {tops(names)}")
    missing = [m for m in CORE_MODULES if f"sprout_core/{m}.py" not in names]
    ok(not missing, f"{core.name} holds the pipelines, missing {missing}")
    ok("sprout = sprout_core.cli:main" in entry, "the sprout command is declared")
    ok("sprout_core/templates/make_seeds.yaml" in names, "the YAML templates are in the core wheel")
    with zipfile.ZipFile(plugin) as z:
        names = z.namelist()
        entry = next(z.read(n).decode() for n in names if n.endswith("entry_points.txt"))
    ok(tops(names) == ["napari_sprout"], f"{plugin.name} holds only napari_sprout: {tops(names)}")
    ok("napari_sprout/napari.yaml" in names, "the manifest is in the plugin wheel")
    ok("napari.manifest" in entry, "napari finds the plugin by its entry point")
    return core, plugin


def run(cmd, cwd, **kw):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", **kw)
    if r.returncode:
        print((r.stdout + r.stderr)[-3000:], flush=True)
    return r


def main(dist):
    core, plugin = check_wheels(dist)
    tmp = Path(tempfile.mkdtemp(prefix="sprout_package_"))
    venv = tmp / "venv"
    run([sys.executable, "-m", "venv", str(venv)], tmp, check=True)
    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    py, sprout = bin_dir / "python", bin_dir / "sprout"
    r = run([str(py), "-m", "pip", "install", "-q", str(core), f"{plugin}[all]"], tmp)
    ok(r.returncode == 0, "both wheels install, napari-sprout[all] with napari and Qt")
    if r.returncode:
        return
    print(run([str(py), "-m", "pip", "list"], tmp).stdout, flush=True)        # the versions this run tested

    work = tmp / "work"                        # no repository here
    work.mkdir()
    r = run([str(sprout), "--version"], work)
    ok(r.returncode == 0 and r.stdout.startswith("sprout"), f"sprout --version: {r.stdout.strip()!r}")
    r = run([str(sprout), "--template", "seeds"], work)
    ok(r.returncode == 0 and (work / "make_seeds.yaml").is_file(), f"sprout --template seeds: {r.stdout.strip()!r}")

    img = np.zeros((12, 12, 24), np.uint8)
    img[3:9, 3:9, 2:10] = 200
    img[3:9, 3:9, 14:22] = 200
    img[6, 6, 10:14] = 200                     # two cubes, a thin bridge: one region, two after an erosion
    tifffile.imwrite(work / "img.tif", img)
    (work / "seeds.yaml").write_text("img_path: img.tif\nthresholds: [100]\nerosion_steps: 1\nsegments: 5\n"
                                     "num_threads: 1\noutput_folder: out\n", encoding="utf-8")
    r = run([str(sprout), "--seeds", "--config", "seeds.yaml"], work, timeout=600)
    seeds = sorted((work / "out").rglob("*.tif"))
    ok(r.returncode == 0 and seeds, f"sprout --seeds writes seeds: {[p.name for p in seeds]}")

    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(TESTS), NAPARI_CONFIG=str(tmp / "napari.yaml"))
    r = run([str(py), "-c", PANELS, str(venv)], work, env=env, timeout=600)
    ok(r.returncode == 0, f"every SPROUT panel opens: {r.stdout.strip()[-300:]}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
    print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
    sys.exit(1 if FAILED else 0)
