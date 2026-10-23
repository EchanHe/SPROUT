"""
Point and scribble prompts from seeds, for promptable segmentation models
(nnInteractive now, SAM later). No Qt, no torch here; sklearn only on demand.

Coordinates are voxel indices of the image array. A prompt table (DataFrame)
has the columns POINT_COLUMNS, one row per point, in the order the points are
given to the model (the order matters: nnInteractive centres each prediction on
the newest interaction).

`_grid_sampling` and `sample_points_3d` were moved here unchanged from
sprout_prompt_core.py (which re-exports them); the only additions are the
optional `rng` (default: the global np.random stream, as before) and the lazy
KMeans import.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

POINT_COLUMNS = ['image_name', 'class_id', 'x', 'y', 'z', 'label', 'point_type', 'sample_method']
FEATURE_COLUMNS = ['class_id', 'label', 'point_type', 'sample_method']

# A crop of bbox(roi) + 1 keeps the distance transform of `center_edge` equal to a
# whole-image run (same value as sprout_core.roi.WALL_MARGIN).
_ROI_MARGIN = 1


### Sample for nninteractive input ###

def _grid_sampling(coords: np.ndarray, n: int, rng=None) -> np.ndarray:
    """
    Grid-based sampling: divide 3D space into grid cells, sample from each.

    Advantages:
    - Very uniform
    - Fast
    - No extra dependencies

    Steps:
    1. Compute the bounding box
    2. Divide into n grid cells
    3. Randomly pick one point from each cell
    """
    r = np.random if rng is None else rng
    # Calculate grid dimensions (try to make cubic cells)
    n_cells_per_dim = int(np.ceil(n ** (1/3)))

    # Get bounding box
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    ranges = maxs - mins + 1

    # Create grid
    grid_size = ranges / n_cells_per_dim

    # Assign each coordinate to a grid cell
    grid_indices = ((coords - mins) / grid_size).astype(int)
    grid_indices = np.clip(grid_indices, 0, n_cells_per_dim - 1)

    # Create unique cell identifiers
    cell_ids = (grid_indices[:, 0] * n_cells_per_dim**2 +
                grid_indices[:, 1] * n_cells_per_dim +
                grid_indices[:, 2])

    # Sample one point from each non-empty cell
    selected_points = []
    unique_cells = np.unique(cell_ids)

    for cell_id in unique_cells:
        cell_coords = coords[cell_ids == cell_id]
        # Randomly pick one point from this cell
        selected_points.append(cell_coords[r.randint(len(cell_coords))])

        if len(selected_points) >= n:
            break

    # If not enough cells, add random points
    while len(selected_points) < n:
        selected_points.append(coords[r.randint(len(coords))])

    return np.array(selected_points[:n])

def sample_points_3d(mask_3d: np.ndarray, n: int = 5, method: str = 'random', rng=None) -> np.ndarray:
    """
    Sample n points from a 3D binary mask.

    Parameters:
    -----------
    mask_3d : np.ndarray
        3D binary mask
    n : int
        Number of points to sample
    method : str
        'random', 'grid', 'kmeans', 'center_edge'
    rng : np.random.RandomState, optional
        Random stream; None = the global np.random stream.

    Returns:
    --------
    np.ndarray
        Array of shape (n, 3) with [z, y, x] coordinates
    """
    r = np.random if rng is None else rng
    coords = np.column_stack(np.where(mask_3d > 0))  # [z, y, x]

    if len(coords) == 0:
        return np.array([]).reshape(0, 3)

    if len(coords) <= n:
        return coords

    if n==1:
        # Just return one random point
        idx = r.randint(len(coords))
        return coords[idx:idx+1]

    if method == 'random':
        indices = r.choice(len(coords), n, replace=False)
        return coords[indices]

    elif method == 'grid':
        return _grid_sampling(coords, n, rng=rng)

    elif method == 'kmeans':
        try:
            from sklearn.cluster import KMeans
        except Exception:
            KMeans = None
        if KMeans is None:
            raise ImportError("sklearn required. Install: pip install scikit-learn")
        kmeans = KMeans(n_clusters=n, random_state=0, n_init=10).fit(coords)
        return np.round(kmeans.cluster_centers_).astype(int)

    elif method == 'center_edge':
        if distance_transform_edt is None:
            raise ImportError("scipy required. Install: pip install scipy")
        dist = distance_transform_edt(mask_3d)
        center = np.unravel_index(np.argmax(dist), dist.shape)

        edge_coords = np.column_stack(np.where((dist > 0) & (dist < 3)))
        if len(edge_coords) >= (n - 1):
            indices = r.choice(len(edge_coords), n - 1, replace=False)
            edge_samples = edge_coords[indices]
        else:
            edge_samples = edge_coords

        if len(edge_samples) > 0:
            return np.vstack(([center], edge_samples))
        else:
            return np.array([center])

    else:
        raise ValueError(f"Unknown method: {method}")


# --------------------------------------------------------------------------
# Seeds -> point prompts
# --------------------------------------------------------------------------
def empty_points_df():
    """An empty prompt table with the standard columns and dtypes."""
    return pd.DataFrame({
        'image_name': pd.Series([], dtype=object),
        'class_id': pd.Series([], dtype=np.int64),
        'x': pd.Series([], dtype=np.int64),
        'y': pd.Series([], dtype=np.int64),
        'z': pd.Series([], dtype=np.int64),
        'label': pd.Series([], dtype=bool),
        'point_type': pd.Series([], dtype=object),
        'sample_method': pd.Series([], dtype=object),
    })


_STATS_CHUNK = 1 << 21          # voxels per chunk in class_stats (bounds the temporaries)


def class_stats(seg, allowed=None):
    """Voxels per non-zero class id (optionally only where `allowed`).

    Returns DataFrame[class_id, voxels] sorted by id, classes with 0 voxels left out.
    Counted in chunks along the first axis, so no full-size temporary is made.
    """
    seg = np.asarray(seg)
    if allowed is not None:
        allowed = np.asarray(allowed, dtype=bool)
        if allowed.shape != seg.shape:
            raise ValueError(f"allowed shape {allowed.shape} must match seeds shape {seg.shape}")
    if seg.ndim == 0 or seg.size == 0:
        return pd.DataFrame({'class_id': np.zeros(0, np.int64), 'voxels': np.zeros(0, np.int64)})
    if seg.ndim == 1:
        seg = seg[None]
        allowed = None if allowed is None else allowed[None]
    per_row = max(1, int(np.prod(seg.shape[1:])))
    step = max(1, _STATS_CHUNK // per_row)
    chunks = [slice(i, min(i + step, seg.shape[0])) for i in range(0, seg.shape[0], step)]

    def vals(sl):
        v = seg[sl]
        return v[allowed[sl]] if allowed is not None else v.ravel()

    lo, hi = None, None
    for sl in chunks:
        v = vals(sl)
        if v.size:
            lo = int(v.min()) if lo is None else min(lo, int(v.min()))
            hi = int(v.max()) if hi is None else max(hi, int(v.max()))
    if hi is None:
        return pd.DataFrame({'class_id': np.zeros(0, np.int64), 'voxels': np.zeros(0, np.int64)})
    if lo < 0:
        raise ValueError("label ids must be non-negative")
    if hi < 2 ** 20:
        counts = np.zeros(hi + 1, np.int64)
        for sl in chunks:
            v = vals(sl)
            if v.size:
                counts += np.bincount(v.astype(np.intp, copy=False), minlength=hi + 1)
        ids = np.nonzero(counts)[0]
        counts = counts[ids]
    else:
        acc = {}
        for sl in chunks:
            u, n = np.unique(vals(sl), return_counts=True)
            for a, b in zip(u.tolist(), n.tolist()):
                acc[a] = acc.get(a, 0) + b
        ids = np.array(sorted(acc), dtype=np.int64)
        counts = np.array([acc[i] for i in ids.tolist()], dtype=np.int64)
    keep = ids != 0
    return pd.DataFrame({'class_id': ids[keep].astype(np.int64),
                         'voxels': counts[keep].astype(np.int64)})


def _class_bboxes(seg, ids):
    """{id: tuple of slices} tight bounding boxes of the given ids (absent ids left out)."""
    out = {}
    if not len(ids):
        return out
    max_id = int(max(ids))
    if max_id < 2 ** 20 and seg.dtype.kind in 'iu':
        from scipy.ndimage import find_objects
        try:
            objs = find_objects(seg, max_label=max_id)
        except Exception:
            objs = None
        if objs is not None:
            for i in ids:
                sl = objs[int(i) - 1]
                if sl is not None:
                    out[int(i)] = sl
            return out
    for i in ids:
        m = seg == i
        sl = []
        for ax in range(seg.ndim):
            proj = np.any(m, axis=tuple(a for a in range(seg.ndim) if a != ax))
            nz = np.flatnonzero(proj)
            if not len(nz):
                break
            sl.append(slice(int(nz[0]), int(nz[-1]) + 1))
        else:
            out[int(i)] = tuple(sl)
    return out


def _grow_slices(sl, shape, margin):
    return tuple(slice(max(s.start - margin, 0), min(s.stop + margin, n)) for s, n in zip(sl, shape))


def _union_slices(boxes):
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return tuple(slice(min(b[a].start for b in boxes), max(b[a].stop for b in boxes)) for a in range(len(boxes[0])))


def _window_mask(seg, select, bbox, masks, method):
    """(mask, window start) of select(seg) & masks..., cut to bbox + 1 voxel (clamped).

    The coordinates keep their raster order under the shift and the 1-voxel ring of
    zeros (or the unchanged image border) keeps the distance transform of
    'center_edge' equal, so 'random', 'grid' and 'center_edge' give exactly the
    whole-image points. 'kmeans' uses the whole image (its float result is not
    shift-exact). bbox must contain every voxel select() can be True on.
    """
    if method == 'kmeans' or bbox is None:
        win = tuple(slice(0, n) for n in seg.shape)
    else:
        win = _grow_slices(bbox, seg.shape, 1)
    m = select(seg[win])
    for extra in masks:
        if extra is not None:
            m &= extra[win]
    return m.astype(np.uint8), np.array([w.start for w in win], dtype=np.int64)


def _shifted(pts, start):
    return pts + start if len(pts) else pts


def seeds_to_point_prompts(seg, img=None, *, classes=None, allowed=None, exclude_zero_intensity=True,
                           class_config=None, default_n_pos=1, default_n_neg=0, default_method='random',
                           negative_from_bg=True, negative_from_other_classes=True,
                           negative_per_other_class=False, image_name='', rng=None, cancel=None,
                           progress=None, log=None):
    """Array version of sprout_prompt_core.seed_to_point_prompts_nninteractive.

    Same loop order, random calls and quirks as the file version (so the same seed
    gives the same table): positives come from `seg == class` (zero-intensity voxels
    are NOT excluded there), negatives from the background / other classes with
    zero-intensity voxels excluded, except the "all other classes combined" mask;
    negative rows always say sample_method 'random'.

    seg : 3D (or 2D, then z = 0) label array.  img : same shape (or 4D, first channel
    used); needed when exclude_zero_intensity.
    classes : ids that get points (None = all); negatives "from other classes" still
    use every class.  allowed : bool array ANDed into every sampling mask (ROI).
    rng : np.random.RandomState (same stream as np.random.seed(s)); None = global.
    cancel : threading.Event or callable, checked before each class (raises
    boundary.Cancelled).  progress(i, n) after each class.  log(str) for warnings
    (default print).

    Returns a DataFrame with POINT_COLUMNS (empty table with those columns if no point).
    """
    from sprout_core.boundary import Cancelled, _is_cancelled

    say = print if log is None else log
    if class_config is None:
        class_config = {}
    seg = np.asarray(seg)
    if seg.ndim == 2:
        seg = seg[None]
        if img is not None:
            img = np.asarray(img)[None]
        if allowed is not None:
            allowed = np.asarray(allowed)[None]
    if seg.ndim != 3:
        raise ValueError(f"seeds must be 2D or 3D, got {seg.ndim}D")
    if allowed is not None:
        allowed = np.asarray(allowed, dtype=bool)
        if allowed.shape != seg.shape:
            raise ValueError(f"allowed shape {allowed.shape} must match seeds shape {seg.shape}")

    if exclude_zero_intensity:
        if img is None:
            raise ValueError("exclude_zero_intensity=True requires img")
        img = np.asarray(img)
        if img.ndim == 4:
            img_vol = img[0]
        elif img.ndim == 3:
            img_vol = img
        else:
            raise ValueError(f"Unsupported img ndim={img.ndim}, expected 3D or 4D")
        if img_vol.shape != seg.shape:
            raise ValueError(f"img shape {img_vol.shape} must match seg shape {seg.shape}")
        nonzero_mask = img_vol != 0
    else:
        nonzero_mask = None

    class_ids = np.unique(seg)
    class_ids = class_ids[class_ids != 0]
    todo = class_ids if classes is None else class_ids[np.isin(class_ids, np.asarray(list(classes)))]
    if len(todo) == 0:
        return empty_points_df()
    bboxes = _class_bboxes(seg, [int(c) for c in class_ids])

    # Same for every class: build once (the file version rebuilds it per class).
    bg_mask = None

    rows = []

    def add(pts, cls, label, point_type, method):
        for pt in pts:
            rows.append({
                'image_name': image_name,
                'class_id': int(cls),
                'x': int(pt[2]),
                'y': int(pt[1]),
                'z': int(pt[0]),
                'label': label,
                'point_type': point_type,
                'sample_method': method,
            })

    for i, cls in enumerate(todo):
        if _is_cancelled(cancel):
            raise Cancelled()
        cfg = class_config.get(int(cls), {})
        n_pos = cfg.get('n_pos', default_n_pos)
        n_neg = cfg.get('n_neg', default_n_neg)
        method = cfg.get('method', default_method)

        try:
            mask, start = _window_mask(seg, lambda a: a == cls, bboxes.get(int(cls)), (allowed,), method)
            pos_points = _shifted(sample_points_3d(mask, n=n_pos, method=method, rng=rng), start)
        except Exception as e:
            say(f"  [WARNING] Failed to sample positive for class {cls}: {e}")
            if progress:
                progress(i + 1, len(todo))
            continue
        if len(pos_points) == 0:
            say(f"  [WARNING] No positive points for class {cls}")
            if progress:
                progress(i + 1, len(todo))
            continue
        add(pos_points, cls, True, 'positive', method)

        if n_neg > 0:
            if negative_from_bg:
                if bg_mask is None:
                    bg_mask = (seg == 0)
                    if nonzero_mask is not None:
                        bg_mask &= nonzero_mask
                    if allowed is not None:
                        bg_mask &= allowed
                    bg_mask = bg_mask.astype(np.uint8)
                if np.sum(bg_mask) > 0:
                    try:
                        add(sample_points_3d(bg_mask, n=n_neg, method=method, rng=rng),
                            cls, False, 'negative_bg', 'random')
                    except Exception as e:
                        say(f"    [WARNING] Failed to sample negative bg: {e}")
            if negative_from_other_classes:
                if negative_per_other_class:
                    for other_cls in class_ids:
                        if other_cls == cls:
                            continue
                        other_mask, start = _window_mask(seg, lambda a: a == other_cls, bboxes.get(int(other_cls)),
                                                         (nonzero_mask, allowed), method)
                        if np.sum(other_mask) > 0:
                            try:
                                add(_shifted(sample_points_3d(other_mask, n=n_neg, method=method, rng=rng), start),
                                    cls, False, f'negative_class{int(other_cls)}', 'random')
                            except Exception as e:
                                say(f"    [WARNING] Failed to sample from class {other_cls}: {e}")
                else:
                    # quirk kept from the file version: zero-intensity voxels are not excluded here
                    others = _union_slices([b for c, b in bboxes.items() if c != int(cls)])
                    other_mask, start = _window_mask(seg, lambda a: (a != cls) & (a > 0), others, (allowed,),
                                                     method)
                    if np.sum(other_mask) > 0:
                        try:
                            add(_shifted(sample_points_3d(other_mask, n=n_neg, method=method, rng=rng), start),
                                cls, False, 'negative_other', 'random')
                        except Exception as e:
                            say(f"    [WARNING] Failed to sample negative other: {e}")
        if progress:
            progress(i + 1, len(todo))

    if not rows:
        return empty_points_df()
    return pd.DataFrame(rows)[POINT_COLUMNS]


def seeds_to_point_prompts_in_roi(seg, img, roi, *, margin=_ROI_MARGIN, **kw):
    """seeds_to_point_prompts restricted to an ROI without a full-size mask.

    Samples on roi.crop(..., margin) with allowed = roi.mask_crop(margin) and shifts the
    points back by roi.offset(margin). Equal to the whole-image run with
    allowed=roi.mask_full() for 'random', 'grid' and 'center_edge' (margin 1 keeps the
    distance transform equal), not for 'kmeans'.
    """
    seg_c = roi.crop(seg, margin)
    img_c = roi.crop(img, margin) if img is not None else None
    allowed = roi.mask_crop(margin)
    df = seeds_to_point_prompts(seg_c, img_c, allowed=allowed, **kw)
    off = roi.offset(margin)
    if len(df):
        if len(off) == 3:
            df['z'] += int(off[0])
        df['y'] += int(off[-2])
        df['x'] += int(off[-1])
    return df


# --------------------------------------------------------------------------
# Prompt table <-> napari Points layer data
# --------------------------------------------------------------------------
def _point_type(label):
    return 'positive' if label else 'negative'


def points_to_df(data, features=None, image_name=''):
    """Points layer data (N, ndim) in voxel coordinates + features -> prompt table.

    Row order is kept. Coordinates are rounded with np.rint. 2D data are (y, x), z = 0.
    Missing feature columns get defaults: class_id 1, label True, point_type from the
    label ('positive' / 'negative'), sample_method 'manual'.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1 and data.size == 0:
        data = data.reshape(0, 3)
    if data.ndim != 2:
        raise ValueError(f"points data must be (N, ndim), got shape {data.shape}")
    n, ndim = data.shape
    if ndim not in (2, 3):
        raise ValueError(f"points must be 2D or 3D, got {ndim}D")
    feats = pd.DataFrame(index=range(n)) if features is None else pd.DataFrame(features).reset_index(drop=True)
    if len(feats) != n:
        raise ValueError(f"{len(feats)} feature rows for {n} points")
    idx = np.rint(data).astype(np.int64)
    if ndim == 3:
        z, y, x = idx[:, 0], idx[:, 1], idx[:, 2]
    else:
        z, y, x = np.zeros(n, np.int64), idx[:, 0], idx[:, 1]
    label = (feats['label'].to_numpy().astype(bool) if 'label' in feats else np.ones(n, bool))
    cls = (feats['class_id'].to_numpy().astype(np.int64) if 'class_id' in feats else np.ones(n, np.int64))
    if 'point_type' in feats:
        ptype = [(_point_type(lb) if (v is None or (isinstance(v, float) and np.isnan(v)) or v == '') else str(v))
                 for v, lb in zip(feats['point_type'].tolist(), label)]
    else:
        ptype = [_point_type(lb) for lb in label]
    if 'sample_method' in feats:
        meth = [('manual' if (v is None or (isinstance(v, float) and np.isnan(v)) or v == '') else str(v))
                for v in feats['sample_method'].tolist()]
    else:
        meth = ['manual'] * n
    return pd.DataFrame({
        'image_name': pd.Series([image_name] * n, dtype=object),
        'class_id': cls,
        'x': x,
        'y': y,
        'z': z,
        'label': label,
        'point_type': pd.Series(ptype, dtype=object),
        'sample_method': pd.Series(meth, dtype=object),
    })[POINT_COLUMNS]


def df_to_points(df, ndim=3):
    """Prompt table -> (data float (N, ndim), features DataFrame[class_id, label, point_type, sample_method]).

    Row order is kept; coordinates rounded with np.rint. ndim 2 -> (y, x).
    """
    if ndim not in (2, 3):
        raise ValueError(f"ndim must be 2 or 3, got {ndim}")
    n = len(df)
    cols = ['z', 'y', 'x'] if ndim == 3 else ['y', 'x']
    if n == 0:
        data = np.zeros((0, ndim), float)
    else:
        data = np.rint(df[cols].to_numpy(dtype=float))
    label = df['label'].to_numpy().astype(bool) if 'label' in df else np.ones(n, bool)
    feats = pd.DataFrame({
        'class_id': (df['class_id'].to_numpy().astype(np.int64) if 'class_id' in df else np.ones(n, np.int64)),
        'label': label,
        'point_type': pd.Series([str(v) for v in df['point_type']] if 'point_type' in df
                                else [_point_type(lb) for lb in label], dtype=object),
        'sample_method': pd.Series([str(v) for v in df['sample_method']] if 'sample_method' in df
                                   else ['manual'] * n, dtype=object),
    })
    return data, feats


def points_data_to_voxels(data, scale_p, translate_p, scale_i, translate_i, *, rotate=None, shear=None):
    """Points layer data coordinates -> image voxel coordinates (float).

    world = data * scale_p + translate_p;  voxel = (world - translate_i) / scale_i.
    Rotation / shear are ignored (a warning is emitted when they are not identity).
    """
    for name, m in (('rotate', rotate), ('shear', shear)):
        if m is None:
            continue
        m = np.asarray(m, dtype=float)
        ident = np.eye(m.shape[0]) if (m.ndim == 2 and m.shape[0] == m.shape[1]) else np.zeros_like(m)
        if not np.allclose(m, ident):
            warnings.warn(f"Points/Image {name} is ignored when converting point coordinates")
    data = np.asarray(data, dtype=float)
    sp, tp = np.asarray(scale_p, float), np.asarray(translate_p, float)
    si, ti = np.asarray(scale_i, float), np.asarray(translate_i, float)
    return ((data * sp + tp) - ti) / si


# --------------------------------------------------------------------------
# Scribbles
# --------------------------------------------------------------------------
def seed_scribble(seeds, class_id, negative=False, allowed=None):
    """Seeds as a scribble prompt (the YAML `prompt_type: scribble` path of nninter_predict).

    Returns (positive uint8 mask = seeds == class_id, negative uint8 mask of the other
    classes or None). With `allowed`, both are restricted to it.
    """
    pos = seeds == class_id
    if allowed is not None:
        pos &= allowed
    neg = None
    if negative:
        neg = (seeds != class_id) & (seeds != 0)
        if allowed is not None:
            neg &= allowed
        neg = neg.astype(np.uint8)
    return pos.astype(np.uint8), neg


def scribbles_from_layer(labels, class_id):
    """User-drawn scribbles: voxels painted with `class_id` (bool mask)."""
    return np.asarray(labels) == class_id
