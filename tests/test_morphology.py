"""
Erosion, dilation and closing of SPROUT without skimage's binary_* (deprecated in 0.26, removed in
0.28) and cube / square (deprecated in 0.25, removed in 0.27): scipy.ndimage does the same.
"""
import ast
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sprout_core.sprout_core as sc  # noqa: E402

GONE = {"binary_erosion", "binary_dilation", "binary_closing", "binary_opening", "cube", "square"}


@pytest.mark.core
def test_no_skimage_function_that_is_going_away():
    used = []
    for f in sorted((ROOT / "sprout_core").glob("*.py")):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("skimage"):
                used += [f"{f.name}: {a.name}" for a in node.names if a.name in GONE]
    assert not used, used


@pytest.mark.core
def test_the_same_result_as_skimage():
    """The expected values are scipy's calls inside skimage 0.22-0.26 (border_value=True for erosion)."""
    rng = np.random.default_rng(0)
    vol = ndi.binary_opening(rng.random((14, 15, 16)) > 0.4)
    vol[0] = True                                     # foreground on the border: border_value matters
    img = vol[7]
    fps3 = [sc.ball_fp_XY, sc.fp_X, sc.cube_fp(1), sc.cube_fp(2), np.ones((3, 3, 3), np.uint8)]
    fps2 = [sc.fp_X_2d, sc.square_fp(1), sc.square_fp(2)]
    for a, fps in ((vol, fps3), (img, fps2)):
        for fp in fps:
            assert np.array_equal(sc.binary_erosion(a, fp), ndi.binary_erosion(a, structure=fp, border_value=True))
            assert np.array_equal(sc.binary_dilation(a, fp), ndi.binary_dilation(a, structure=fp))
            assert np.array_equal(sc.binary_closing(a, fp),
                                  ndi.binary_erosion(ndi.binary_dilation(a, structure=fp), structure=fp,
                                                     border_value=True))
    assert np.array_equal(sc.cube_fp(2), np.ones((5, 5, 5), np.uint8)) and sc.cube_fp(2).dtype == np.uint8
    assert np.array_equal(sc.square_fp(1), np.ones((3, 3), np.uint8)) and sc.square_fp(1).dtype == np.uint8


@pytest.mark.core
def test_skimage_gives_the_same_while_it_still_has_them():
    morph = pytest.importorskip("skimage.morphology")
    if not all(hasattr(morph, n) for n in ("binary_erosion", "binary_dilation", "binary_closing")):
        pytest.skip("this scikit-image has no binary_* any more")
    rng = np.random.default_rng(1)
    vol = ndi.binary_opening(rng.random((12, 13, 14)) > 0.4)
    vol[:, 0] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for fp in (sc.ball_fp_XY, sc.cube_fp(1), morph.ball(1), morph.ball(2)):
            assert np.array_equal(sc.binary_erosion(vol, fp), morph.binary_erosion(vol, fp))
            assert np.array_equal(sc.binary_dilation(vol, fp), morph.binary_dilation(vol, fp))
            assert np.array_equal(sc.binary_closing(vol, fp), morph.binary_closing(vol, fp))


@pytest.mark.core
def test_erosion_and_dilation_without_future_warnings():
    vol = np.zeros((9, 9, 9), bool)
    vol[2:7, 2:7, 2:7] = True
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        for fp in ("ball", "cube"):
            sc.erosion_binary_img_on_sub(vol.copy(), kernal_size=1, footprint=fp)
            sc.dilation_binary_img_on_sub(vol.copy(), margin=2, kernal_size=1, is_round=fp == "ball")
