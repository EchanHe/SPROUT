"""
Boundary generation for SPROUT (no Qt here; usable from napari, CLI or batch).

Top-hat
-------
    black top-hat = closing(img) - img    small DARK structures (sutures, gaps)
    white top-hat = img - opening(img)    small BRIGHT structures

The response has the same dtype as the image; a boundary mask is obtained by
thresholding it, optionally only among voxels whose ORIGINAL intensity lies in
a given range (e.g. to exclude a zeroed background).

Footprints (all are unions of 1D rods along the last axis, which is what makes
the exact ball fast on both CPU and GPU):

    ball        exact Euclidean ball       dz^2 + dy^2 + dx^2 <= s^2
    cube        (2s+1)^3 cube              == Avizo: Cube, neighbourhood 26, size s
    octahedron  |dz| + |dy| + |dx| <= s    == Avizo: Cube, neighbourhood 6,  size s

2D images are handled as a single-slice volume, so the 3D shapes reduce to
their 2D counterparts (disk, square, diamond).

CPU (numpy/scipy, multi-threaded over chunks) and GPU (torch + CUDA) give
voxel-identical results.
"""
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

SHAPES = ("ball", "cube", "octahedron")
KINDS = ("black", "white")


class Cancelled(Exception):
    """Raised inside a computation when the caller asked it to stop."""


def _is_cancelled(cancel):
    if cancel is None:
        return False
    if isinstance(cancel, threading.Event):
        return cancel.is_set()
    return bool(cancel())


# --------------------------------------------------------------------------
# Footprints
# --------------------------------------------------------------------------
def footprint_rods(shape, s):
    """List of (dz, dy, w): a rod of half-width w along x at offset (dz, dy)."""
    rods = []
    for dz in range(-s, s + 1):
        for dy in range(-s, s + 1):
            if shape == "cube":
                w = s
            elif shape == "octahedron":
                w = s - abs(dz) - abs(dy)
            elif shape == "ball":
                rem = s * s - dz * dz - dy * dy
                w = math.isqrt(rem) if rem >= 0 else -1
            else:
                raise ValueError(f"unknown shape {shape!r}, choose from {SHAPES}")
            if w >= 0:
                rods.append((dz, dy, w))
    return rods


def footprint_array(shape, s):
    """Dense boolean 3D footprint (for checks and display)."""
    fp = np.zeros((2 * s + 1,) * 3, bool)
    for dz, dy, w in footprint_rods(shape, s):
        fp[s + dz, s + dy, s - w: s + w + 1] = True
    return fp


# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------
def gpu_status():
    """(available, description) for display in a UI."""
    try:
        import torch
    except ImportError:
        return False, "torch not installed"
    try:
        if not torch.cuda.is_available():
            return False, "no CUDA GPU"
        return True, torch.cuda.get_device_name(0)
    except Exception as e:  # broken driver etc.
        return False, f"CUDA error: {type(e).__name__}"


def gpu_exact_dtype(dtype):
    """float32 holds integers exactly only up to 2**24."""
    dtype = np.dtype(dtype)
    return (dtype.kind in "ui" and dtype.itemsize <= 2) or dtype == np.float32


def pick_device(requested, dtype):
    """Return ('cuda' | 'cpu', reason)."""
    if requested == "cpu":
        return "cpu", "CPU selected"
    ok, desc = gpu_status()
    if not ok:
        reason = desc
    elif not gpu_exact_dtype(dtype):
        reason = f"{np.dtype(dtype)} is not exact on GPU (float32)"
    else:
        return "cuda", desc
    if requested == "gpu":
        raise RuntimeError(f"GPU requested but unavailable: {reason}")
    return "cpu", reason


# --------------------------------------------------------------------------
# CPU
# --------------------------------------------------------------------------
def _limits(dtype):
    dtype = np.dtype(dtype)
    info = np.iinfo(dtype) if dtype.kind in "ui" else np.finfo(dtype)
    return info.min, info.max


def _morph_np(a, shape, s, op):
    """Grey dilation ('max') / erosion ('min') of a 3D chunk; outside never wins."""
    from scipy import ndimage as ndi
    lo, hi = _limits(a.dtype)
    fill = lo if op == "max" else hi
    reduce_ = np.maximum if op == "max" else np.minimum
    filt1d = ndi.maximum_filter1d if op == "max" else ndi.minimum_filter1d

    if shape == "cube":  # separable
        out = a
        for ax in range(3):
            out = filt1d(out, 2 * s + 1, axis=ax, mode="constant", cval=fill)
        return out

    D, H, _ = a.shape
    ap = np.pad(a, ((s, s), (s, s), (0, 0)), constant_values=fill)
    rods_by_w = {}
    out = np.full_like(a, fill)
    for dz, dy, w in footprint_rods(shape, s):
        if w not in rods_by_w:
            rods_by_w[w] = filt1d(ap, 2 * w + 1, axis=2, mode="constant", cval=fill) if w else ap
        reduce_(out, rods_by_w[w][s + dz: s + dz + D, s + dy: s + dy + H], out=out)
    return out


def _tophat_chunk_np(a, kind, shape, s):
    if kind == "black":
        return _morph_np(_morph_np(a, shape, s, "max"), shape, s, "min") - a
    return a - _morph_np(_morph_np(a, shape, s, "min"), shape, s, "max")


def _chunks(Z, cz, halo):
    for z0 in range(0, Z, cz):
        z1 = min(Z, z0 + cz)
        yield z0, z1, max(0, z0 - halo), min(Z, z1 + halo)


def _tophat_cpu(img, kind, shape, s, progress=None, cancel=None, threads=None, cz=64):
    threads = threads or min(os.cpu_count() or 1, 16)
    out = np.empty_like(img)
    chunks = list(_chunks(img.shape[0], cz, 2 * s))

    def work(c):
        if _is_cancelled(cancel):
            raise Cancelled()
        z0, z1, a, b = c
        res = _tophat_chunk_np(np.ascontiguousarray(img[a:b]), kind, shape, s)
        out[z0:z1] = res[z0 - a: z0 - a + (z1 - z0)]

    with ThreadPoolExecutor(threads) as ex:
        futs = [ex.submit(work, c) for c in chunks]
        try:
            for i, f in enumerate(as_completed(futs), 1):
                f.result()
                if progress:
                    progress(i, len(chunks))
        except BaseException:
            for f in futs:
                f.cancel()
            raise
    return out


# --------------------------------------------------------------------------
# GPU
# --------------------------------------------------------------------------
def _dilate_torch(x, shape, s):
    import torch
    import torch.nn.functional as F
    if shape == "cube":
        return F.max_pool3d(x, 2 * s + 1, stride=1, padding=s)
    _, _, D, H, _ = x.shape
    neg = torch.finfo(x.dtype).min
    xp = F.pad(x, (0, 0, s, s, s, s), value=neg)
    rods_by_w = {}
    out = torch.full_like(x, neg)
    for dz, dy, w in footprint_rods(shape, s):
        if w not in rods_by_w:
            rods_by_w[w] = F.max_pool3d(xp, (1, 1, 2 * w + 1), stride=1, padding=(0, 0, w)) if w else xp
        out = torch.maximum(out, rods_by_w[w][:, :, s + dz: s + dz + D, s + dy: s + dy + H, :])
    return out


def _tophat_chunk_torch(x, kind, shape, s):
    if kind == "black":
        dil = _dilate_torch(x, shape, s)
        return -_dilate_torch(-dil, shape, s) - x
    ero = -_dilate_torch(-x, shape, s)
    return x - _dilate_torch(ero, shape, s)


def _auto_chunk(img, s):
    import torch
    free, _ = torch.cuda.mem_get_info(0)
    _, H, W = img.shape
    per_slice = (H + 2 * s) * W * 4 * ((s + 1) + 6)   # cached rods + temps, float32
    # Cap at 128 slices: more chunks = finer progress and a quicker Stop.
    return int(max(8, min(128, free * 0.5 / per_slice - 4 * s)))


def _tophat_gpu(img, kind, shape, s, progress=None, cancel=None, cz=None):
    import torch
    cz = cz or _auto_chunk(img, s)
    while True:
        try:
            out = np.empty_like(img)
            chunks = list(_chunks(img.shape[0], cz, 2 * s))
            for i, (z0, z1, a, b) in enumerate(chunks, 1):
                if _is_cancelled(cancel):
                    raise Cancelled()
                x = torch.from_numpy(np.ascontiguousarray(img[a:b])).to("cuda:0", torch.float32)[None, None]
                res = _tophat_chunk_torch(x, kind, shape, s)[0, 0]
                out[z0:z1] = res[z0 - a: z0 - a + (z1 - z0)].round().cpu().numpy().astype(img.dtype)
                del x, res
                if progress:
                    progress(i, len(chunks))
            return out, cz
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if cz <= 8:
                raise
            cz //= 2


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def tophat(img, kind="black", shape="ball", size=3, device="auto", progress=None, cancel=None):
    """Grey top-hat of a 2D or 3D image.

    Parameters
    ----------
    img : ndarray, 2D or 3D
    kind : 'black' | 'white'
    shape : 'ball' | 'cube' | 'octahedron'
    size : int >= 1, footprint radius
    device : 'auto' | 'gpu' | 'cpu'
    progress : callable(done, total), optional, called from the worker thread
    cancel : threading.Event or callable returning bool, optional

    Returns
    -------
    response : ndarray, same shape and dtype as img
    info : dict with 'device' and 'device_reason'
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {SHAPES}")
    size = int(size)
    if size < 1:
        raise ValueError("size must be >= 1")
    img = np.asarray(img)
    if img.ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D image, got {img.ndim}D")
    vol = img[None] if img.ndim == 2 else img

    dev, why = pick_device(device, vol.dtype)
    info = {"device": dev, "device_reason": why}
    out = None
    if dev == "cuda":
        try:
            out, cz = _tophat_gpu(vol, kind, shape, size, progress, cancel)
            info["chunk"] = cz
        except Cancelled:
            raise
        except Exception as e:
            if device == "gpu":
                raise
            info["device"], info["device_reason"] = "cpu", f"GPU failed ({type(e).__name__}), used CPU"
    if out is None:
        out = _tophat_cpu(vol, kind, shape, size, progress, cancel)
    if _is_cancelled(cancel):  # stopped while the last chunk was running
        raise Cancelled()
    return (out[0] if img.ndim == 2 else out), info


def subsample(arr, target=15_000_000):
    """Strided view with roughly `target` elements, for fast statistics."""
    step = max(1, int(round((arr.size / target) ** (1 / arr.ndim))))
    return arr[(slice(None, None, step),) * arr.ndim]


def intensity_mask(img, lower=None, upper=None):
    """Boolean mask of voxels whose ORIGINAL intensity is in [lower, upper]."""
    m = np.ones(img.shape, bool)
    if lower is not None:
        m &= img >= lower
    if upper is not None:
        m &= img <= upper
    return m


def histogram(arr, bins=256, img=None, lower=None, upper=None):
    """Coarse histogram from a subsample: ~0.1-0.4 s on 1e9 voxels.

    If img and an intensity range are given, only voxels whose original
    intensity is in range are counted. Integer data with a small range get
    one bin per value.

    Returns (counts, edges).
    """
    sub = subsample(arr)
    if img is not None and (lower is not None or upper is not None):
        sub = sub[intensity_mask(subsample(img), lower, upper)]
    else:
        sub = sub.ravel()
    if sub.size == 0:
        return np.zeros(1, np.int64), np.array([0.0, 1.0])
    lo, hi = float(sub.min()), float(sub.max())
    if np.dtype(arr.dtype).kind in "ui" and hi - lo + 1 <= bins:
        return np.histogram(sub, bins=np.arange(lo, hi + 2) - 0.5)
    return np.histogram(sub, bins=bins, range=(lo, hi if hi > lo else lo + 1))


def default_intensity_range(img):
    """Full range, except that a dominant zero background is excluded.

    Returns (lower, upper, data_min, data_max).
    """
    sub = subsample(img)
    dmin, dmax = float(sub.min()), float(sub.max())
    lower = dmin
    nz = sub[sub != 0]
    if dmin == 0 and nz.size and (sub == 0).mean() > 0.2:
        lower = float(nz.min())
    return lower, dmax, dmin, dmax


def suggest_threshold(response, img=None, lower=None, upper=None, top_fraction=0.01):
    """Starting point only: the value above which `top_fraction` of the
    eligible voxels lie."""
    sub = subsample(response)
    if img is not None and (lower is not None or upper is not None):
        sub = sub[intensity_mask(subsample(img), lower, upper)]
    if sub.size == 0:
        return float(response.max())
    return float(np.quantile(sub, 1 - top_fraction))


def threshold_mask(response, lower=None, upper=None, img=None, img_lower=None, img_upper=None,
                   cancel=None, cz=64):
    """Boundary mask (uint8 0/1), computed in chunks to keep temporaries small.

    A bound of None means unbounded on that side.
    """
    vol_r = response[None] if response.ndim == 2 else response
    vol_i = None if img is None else (img[None] if img.ndim == 2 else img)
    out = np.zeros(vol_r.shape, np.uint8)
    for z0 in range(0, vol_r.shape[0], cz):
        if _is_cancelled(cancel):
            raise Cancelled()
        sl = slice(z0, z0 + cz)
        m = np.ones(vol_r[sl].shape, bool) if lower is None else vol_r[sl] >= lower
        if upper is not None:
            m &= vol_r[sl] <= upper
        if vol_i is not None and (img_lower is not None or img_upper is not None):
            m &= intensity_mask(vol_i[sl], img_lower, img_upper)
        out[sl] = m
    return out[0] if response.ndim == 2 else out
