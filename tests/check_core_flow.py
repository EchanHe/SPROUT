"""
End-to-end check of the core: a dog (3D) and 2d_cell (2D) seeds -> grow -> adaptive-seed
flow gives the outputs recorded in tests/data/golden_core_flow.json. Compared, by sha1:
every returned array (values + dtype + shape) and every .tif written (decoded), and the CSV
logs (temp paths and time stamps in names replaced). The golden also keeps the number of
labels and foreground voxels of each array, to show how much a difference is.

A golden only says "unchanged", not "correct": after a change that is meant to alter results,
record it again (--record) and say why in the commit. The hashes hold for the versions in
requirements-ci.txt; another numpy / scikit-image may change them without any bug.

--base <commit> compares with an older commit instead: its sprout_core + make_*.py are
extracted with `git archive` into a temp folder and the flow runs with both trees. Returned
seeds may then differ in dtype only (B11: seeds are handed to napari with the label dtype
instead of int32); their values are compared after casting to the base dtype.

Run in the `sprout` env:
    python tests/check_core_flow.py [--record | --base 4cec0bc]
"""
import argparse
import json
import subprocess
import sys
import tarfile
import io
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).resolve().parent / "data" / "golden_core_flow.json"

FLOW = r'''
import contextlib, hashlib, io, json, os, re, sys
from pathlib import Path
tree, out, demo = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, tree)
import numpy as np, tifffile
try:
    from sprout_core import make_seeds, make_grow, make_adaptive_seed
except ImportError:                # trees before 1.1.0 keep them at the root
    import make_seeds, make_grow, make_adaptive_seed
import sprout_core.sprout_core as sc
assert Path(sc.__file__).resolve().is_relative_to(Path(tree).resolve()), sc.__file__

def h(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha1(a.tobytes()).hexdigest()

def size(a):
    return {"labels": int(np.unique(a[a != 0]).size), "voxels": int(np.count_nonzero(a))}

ret = {}
def keep(prefix, d):
    for k, a in d.items():
        a = np.asarray(a)
        ret[f"{prefix}/{k}"] = {"sha1": h(a), "dtype": str(a.dtype), "shape": list(a.shape),
                                "sha1_as": {dt: h(a.astype(dt)) for dt in ("int32", "uint8", "uint16")}, **size(a)}

def q(fn, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(**kw)

dog = tifffile.imread(demo / "dog_img.tif")
dseed = tifffile.imread(demo / "dog_seed.tif")
bnd = ((dog < 25) * 255).astype(np.uint8)                    # a 0/255 boundary (B5 path)
cell = tifffile.imread(demo / "2d_cell.tif")
if cell.ndim == 3:
    cell = cell[..., 0] if cell.shape[-1] in (3, 4) else cell[0]

s, _ = q(make_seeds.make_seeds, img=dog, boundary=bnd, thresholds=[140, 160], upper_thresholds=[255, 255],
         erosion_steps=2, segments=6, footprints=["ball", "cube"], num_threads=2, output_folder=str(out / "s1"),
         base_name="dog", return_for_napari=True)
keep("seeds_custom", s)
s, _ = q(make_seeds.make_seeds, img=dog, thresholds=[150], erosion_steps=2, segments=20, num_threads=4,
         output_folder=str(out / "s2"), base_name="dog", return_for_napari=True)
keep("seeds_preset", s)
def first(folder, pattern):      # the same file on every system (Windows sorts paths ignoring case)
    return sorted(folder.rglob(pattern), key=lambda p: p.as_posix().lower())[0]

seed = tifffile.imread(first(out / "s2", "seed_ero_2_*.tif"))
g, _ = q(make_grow.grow_mp, img=dog, seg=seed, boundary=bnd, thresholds=[150, 120], upper_thresholds=[255, 255],
         dilation_steps=[3, 3], save_every_n_iters=1, num_threads=4, is_sort=True, output_folder=str(out / "g1"),
         base_name="dog", return_for_napari=True)
keep("grow_sorted", g)
g, _ = q(make_grow.grow_mp, img=dog, seg=dseed, thresholds=[130], dilation_steps=[10], min_growth_size=2000,
         no_growth_max_iter=2, num_threads=3, is_sort=False, output_folder=str(out / "g2"), base_name="dogseed",
         return_for_napari=True)
keep("grow_unsorted", g)
a, _, _ = q(make_adaptive_seed.make_adaptive_seed_ero, img=dog, boundary=bnd, threshold=140, upper_threshold=255,
            erosion_steps=3, segments=10, num_threads=3, footprints="ball", save_every_iter=True,
            output_folder=str(out / "a1"), base_name="ero", return_for_napari=True)
keep("adaptive_ero", a)
a, _, _ = q(make_adaptive_seed.make_adaptive_seed_thre, img=dog, thresholds=[130, 150, 170],
            upper_thresholds=[255, 255, 255], erosion_steps=1, segments=10, num_threads=3, footprints="ball",
            sort=False, min_size=20, output_folder=str(out / "a2"), base_name="thre", return_for_napari=True)
keep("adaptive_thre", a)
thr = int(np.percentile(cell, 70))
s, _ = q(make_seeds.make_seeds, img=cell, thresholds=[thr], erosion_steps=3, segments=30, num_threads=2,
         output_folder=str(out / "c1"), base_name="cell", return_for_napari=True)
keep("cell_seeds", s)
cseed = tifffile.imread(first(out / "c1", "seed_ero_3_*.tif"))
g, _ = q(make_grow.grow_mp, img=cell, seg=cseed, thresholds=[int(np.percentile(cell, 50))], dilation_steps=[5],
         num_threads=2, is_sort=True, output_folder=str(out / "c2"), base_name="cell", return_for_napari=True)
keep("cell_grow", g)

disk = {}
for p in sorted(out.rglob("*")):
    rel = re.sub(r"_\d{8}_\d{4}", "_<TIME>", p.relative_to(out).as_posix())   # log names carry the minute
    if p.suffix == ".tif":
        a = tifffile.imread(p)
        disk[rel] = f"{h(a)} {a.dtype} {a.shape} {size(a)}"
    elif p.suffix == ".csv":
        disk[rel] = hashlib.sha1(p.read_text().replace(str(out), "<OUT>").replace(str(out).replace(os.sep, "/"),
                                 "<OUT>").encode()).hexdigest()
print(json.dumps({"returned": ret, "disk": disk}))
'''


def run(tree, out):
    r = subprocess.run([sys.executable, "-c", FLOW, str(tree), str(out), str(ROOT / "demo_data")],
                       capture_output=True, text=True, cwd=str(Path.home()))
    if r.returncode:
        raise RuntimeError(f"flow failed in {tree}:\n{r.stderr[-3000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def compare_base(new, base, ok):
    ok(base["disk"] == new["disk"], f"{len(new['disk'])} files written (.tif decoded, .csv), all sha1 identical"
       + "".join(f"\n     differs: {k}" for k in sorted(set(base["disk"]) | set(new["disk"]))
                 if base["disk"].get(k) != new["disk"].get(k)))
    same, dtype_only, bad = [], [], []
    for k in sorted(set(base["returned"]) | set(new["returned"])):
        a, b = new["returned"].get(k), base["returned"].get(k)
        if a == b:
            same.append(k)
        elif a and b and a["shape"] == b["shape"] and a["dtype"] != b["dtype"] \
                and a["sha1_as"][b["dtype"]] == b["sha1"] and k.startswith(("seeds_", "cell_seeds")):
            dtype_only.append(f"{k}: {b['dtype']} -> {a['dtype']}")
        else:
            bad.append(k)
    ok(not bad, f"{len(same) + len(dtype_only)} returned arrays: {len(same)} identical, {len(dtype_only)} "
                f"identical values with a new dtype (B11 seeds only) {bad}")
    for d in dtype_only:
        print(f"     dtype only: {d}")


def compare_golden(new, gold, ok):
    def differs(name):
        return sorted(k for k in set(gold[name]) | set(new[name]) if gold[name].get(k) != new[name].get(k))

    bad = differs("disk")
    ok(not bad, f"{len(new['disk'])} files written (.tif decoded, .csv) == golden"
       + "".join(f"\n     differs: {k}\n       golden {gold['disk'].get(k)}\n       now    {new['disk'].get(k)}"
                 for k in bad))
    bad = differs("returned")
    ok(not bad, f"{len(new['returned'])} returned arrays == golden" + "".join(
        f"\n     differs: {k}: labels {gold['returned'].get(k, {}).get('labels')} -> "
        f"{new['returned'].get(k, {}).get('labels')}, voxels {gold['returned'].get(k, {}).get('voxels')} -> "
        f"{new['returned'].get(k, {}).get('voxels')}" for k in bad))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--record", action="store_true", help="write the golden from this tree")
    g.add_argument("--base", help="compare with this commit instead of the golden")
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="sprout_core_flow_"))
    new = run(ROOT, tmp / "out_new")
    if args.record:
        GOLDEN.write_text(json.dumps(new, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"recorded {len(new['returned'])} arrays, {len(new['disk'])} files -> {GOLDEN}")
        return
    failed = []

    def ok(cond, msg):
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            failed.append(msg)

    if args.base:
        listed = subprocess.run(["git", "-C", str(ROOT), "ls-tree", "--name-only", args.base], capture_output=True,
                                text=True, check=True).stdout.split()
        paths = ["sprout_core"] + [p for p in ("make_seeds.py", "make_grow.py", "make_adaptive_seed.py") if p in listed]
        tar = subprocess.run(["git", "-C", str(ROOT), "archive", args.base, *paths], capture_output=True,
                             check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(tar)) as t:
            t.extractall(tmp / "base_tree")
        compare_base(new, run(tmp / "base_tree", tmp / "out_base"), ok)
    else:
        compare_golden(new, json.loads(GOLDEN.read_text(encoding="utf-8")), ok)
    print(f"\n{'ALL PASSED' if not failed else f'{len(failed)} FAILED'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
