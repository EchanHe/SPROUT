"""
Checks the footprint size convention: sizes are radii, cube/square edge = 2r + 1.

'cube' erosion used to call skimage cube(1), a single voxel, so it did nothing.

Run in the `sprout` env:
    python tests/check_footprints.py
"""
import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from skimage.morphology import ball, binary_closing, binary_dilation, binary_erosion, cube, disk, square

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from make_seeds import make_seeds  # noqa: E402
from sprout_core import sprout_core as sc  # noqa: E402

FAILED = []


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILED.append(msg)


# a sphere-ish blob away from the edges, so cube and ball erosion differ
zz, yy, xx = np.mgrid[:30, :30, :30]
blob = (zz - 15) ** 2 + (yy - 15) ** 2 + (xx - 15) ** 2 <= 9 ** 2
blob2d = blob[15]

ok(sc.cube_fp(1).shape == (3, 3, 3) and sc.cube_fp(2).shape == (5, 5, 5) and sc.square_fp(1).shape == (3, 3),
   "cube_fp / square_fp: radius r -> edge 2r + 1")

e_cube = sc.erosion_binary_img_on_sub(blob.copy(), footprint="cube")
e_ball = sc.erosion_binary_img_on_sub(blob.copy(), footprint="ball")
ok(np.array_equal(e_cube, binary_erosion(blob, cube(3))) and e_cube.sum() < blob.sum(),
   f"'cube' erosion = full 3x3x3 cube: {int(blob.sum())} -> {int(e_cube.sum())} voxels (used to stay {int(blob.sum())})")
ok(np.array_equal(e_ball, binary_erosion(blob, ball(1))), "'ball' erosion unchanged: 3x3x3 cross")
ok(e_cube.sum() < e_ball.sum(), f"cube erodes more than ball on a round blob ({int(e_cube.sum())} < {int(e_ball.sum())})")

e_sq = sc.erosion_binary_img_on_sub(blob2d.copy(), footprint="cube")
ok(np.array_equal(e_sq, binary_erosion(blob2d, square(3))) and e_sq.sum() < blob2d.sum(), "2D 'cube'/'square' erosion = 3x3 square")
ok(np.array_equal(sc.erosion_binary_img_on_sub(blob2d.copy(), footprint="disk"), binary_erosion(blob2d, disk(1))),
   "2D 'disk' erosion unchanged")

point = np.zeros((9, 9, 9), bool)
point[4, 4, 4] = True
d = sc.dilation_binary_img_on_sub(point.copy(), margin=2, kernal_size=1, is_round=False)
ok(int(d.sum()) == 27, f"dilation with is_round=False, radius 1 -> 3x3x3 = 27 voxels (got {int(d.sum())})")
d = sc.dilation_binary_img_on_sub(point.copy(), margin=2, kernal_size=1, is_round=True)
ok(np.array_equal(d, binary_dilation(point, ball(1))), "round dilation unchanged")

c = sc.closing_binary_img_on_sub_one_step_iter(blob.copy(), 1, footprint="cube")
ok(np.array_equal(c, binary_closing(blob, cube(3))), "closing helper with 'cube' still a 3x3x3 closing (it used edge 3 before)")

# end to end: Seeds with footprint 'cube' now actually erode
img = np.where(blob, 200, 0).astype(np.uint8)
with contextlib.redirect_stdout(io.StringIO()):
    seeds, _ = make_seeds(img=img, thresholds=[100], upper_thresholds=[255], erosion_steps=2, segments=5,
                          footprints="cube", output_folder=tempfile.mkdtemp(), num_threads=1, return_for_napari=True)
sizes = {k.split("seed_ero_")[1].split("_")[0]: int((v > 0).sum()) for k, v in seeds.items()}
ok(sizes["0"] > sizes["1"] > sizes["2"], f"make_seeds footprint 'cube': seed size per erosion step {sizes}")

print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
sys.exit(1 if FAILED else 0)
