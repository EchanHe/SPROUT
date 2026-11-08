"""
The label helpers of sprout_core skip zero voxels when most of an array is 0 (seeds are often a
small part of a scan). Their results must stay those of plain numpy.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sprout_core.sprout_core as sc  # noqa: E402

pytestmark = pytest.mark.core


def arrays(n):
    """Random label arrays: all dtypes, mostly zero to no zero, empty, not contiguous."""
    for case in range(n):
        rng = np.random.default_rng(case)
        shape = tuple(int(x) for x in rng.integers(0, 30, rng.integers(1, 4)))
        dtype = rng.choice([np.uint8, np.uint16, np.uint32, np.int32, np.int64])
        a = rng.integers(0, int(rng.integers(1, 300)), shape).astype(dtype)
        a[rng.random(shape) < rng.choice([0.0, 0.3, 0.5, 0.9, 0.999, 1.0])] = 0
        if rng.random() < 0.3 and a.ndim and a.shape[-1] > 1:
            a = a[..., ::2]
        yield rng, a


@pytest.fixture(params=[7, 1 << 22], ids=["many chunks", "one chunk"])
def chunk(request, monkeypatch):
    monkeypatch.setattr(sc, "_CHUNK", request.param)


def test_bincount_is_numpy_bincount(chunk):
    for rng, a in arrays(1500):
        length = int(a.max()) + 1 + int(rng.integers(0, 3)) if a.size else 1
        got = sc._bincount(a, length)
        assert got.dtype == np.intp and np.array_equal(got, np.bincount(a.ravel().astype(np.intp), minlength=length))


def test_apply_lut_is_fancy_indexing(chunk):
    for rng, a in arrays(1500):
        lut = rng.integers(0, 70000, int(a.max()) + 1 if a.size else 1).astype(rng.choice([np.uint16, np.int64]))
        if rng.random() < 0.7:
            lut[0] = 0
        want = lut[a]
        assert np.array_equal(sc._apply_lut(lut, a), want)
        if a.flags.c_contiguous:
            stale = np.full(a.shape, 7, lut.dtype)                 # every voxel must be written
            assert np.array_equal(sc._apply_lut(lut, a, out=stale), want)
            if lut.dtype == a.dtype:
                b = a.copy()
                assert np.array_equal(sc._apply_lut(lut, b, out=b), want)


def test_unique_labels_is_numpy_unique():
    for rng, a in arrays(1500):
        for arr in (a, a.astype(np.int64) - int(rng.integers(0, 5))):   # also negative values
            got, want = sc.unique_labels(arr), np.unique(arr)
            assert got.dtype == want.dtype and np.array_equal(got, want)
