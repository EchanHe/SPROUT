"""
Plain-script check for sprout_core/seed_roi.py and the seed-ROI config keys (no GPU, no pytest).

    python check_seed_roi.py

Prints OK per section. Expectations that seed_roi.py does not meet are collected as
FAIL lines and the script exits with code 1 at the end (all other sections still run).
"""
import contextlib
import io
import sys

import numpy as np
import yaml

from sprout_core import config_core
from sprout_core.seed_roi import (expand_to_min_size, inner_face_fraction, margin_voxels,
                                  roi_log, seed_bboxes, seed_rois)

FAILS = []


def expect(cond, msg):
    """Soft assert: record a failure but keep going."""
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL: {msg}")
    return cond


def box(roi):
    return tuple(int(v) for v in roi.lo), tuple(int(v) for v in roi.hi)


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


SHAPE = (40, 50, 60)


def make_seed(dtype=np.uint16):
    seed = np.zeros(SHAPE, dtype)
    seed[10:15, 20:30, 5:8] = 1       # bbox size (5, 10, 3)
    seed[30:35, 40:45, 50:58] = 2     # bbox size (5, 5, 8)
    return seed


# ---------------------------------------------------------------- class vs union
def check_class_union():
    seed = make_seed()
    rois = seed_rois(seed, mode="class", margin=0)
    assert set(rois) == {1, 2}, rois.keys()
    assert box(rois[1]) == ((10, 20, 5), (15, 30, 8)), box(rois[1])
    assert box(rois[2]) == ((30, 40, 50), (35, 45, 58)), box(rois[2])

    u = seed_rois(seed, mode="union", margin=0)
    assert set(u) == {1, 2}
    assert box(u[1]) == box(u[2]) == ((10, 20, 5), (35, 45, 58)), box(u[1])
    assert u[1] is u[2]

    # crop of the image is exactly the seed bbox
    assert rois[1].crop(seed).shape == (5, 10, 3)
    assert rois[1].offset() == (10, 20, 5)

    # 2 dtypes of the seed give the same boxes
    for dt in (np.int32, np.uint8, np.int64):
        r = seed_rois(make_seed(dt), margin=0)
        assert {k: box(v) for k, v in r.items()} == {k: box(v) for k, v in rois.items()}, dt

    assert raises(lambda: seed_rois(seed, mode="foo"))
    print("OK class vs union")


# ---------------------------------------------------------------- voxel margin
def check_voxel_margin():
    seed = make_seed()
    r = seed_rois(seed, margin=3)
    assert box(r[1]) == ((7, 17, 2), (18, 33, 11)), box(r[1])
    r = seed_rois(seed, margin=[1, 2, 3])
    assert box(r[1]) == ((9, 18, 2), (16, 32, 11)), box(r[1])
    r = seed_rois(seed, margin=(0, 0, 0))
    assert box(r[1]) == ((10, 20, 5), (15, 30, 8))
    assert margin_voxels((0, 0, 0), (5, 5, 5), [1, 2, 3], "voxel") == (1, 2, 3)

    assert raises(lambda: seed_rois(seed, margin=-1))
    assert raises(lambda: seed_rois(seed, margin=[1, 2]))
    assert raises(lambda: seed_rois(seed, unit="mm"))
    print("OK voxel margin (scalar, per-axis list, errors)")


# ---------------------------------------------------------------- percent margin
def check_percent_margin():
    seed = make_seed()
    # label 1 bbox size (5, 10, 3); 20% -> (1.0, 2.0, 0.6) -> ceil -> (1, 2, 1)
    r = seed_rois(seed, margin=20, unit="percent")
    assert r[1].source["margin"] == [1, 2, 1], r[1].source["margin"]
    assert box(r[1]) == ((9, 18, 4), (16, 32, 9)), box(r[1])
    # 10% of 3 = 0.3 -> 1 (rounded up, never 0 for a positive percentage)
    assert margin_voxels((0, 0, 0), (3, 3, 3), 10, "percent") == (1, 1, 1)
    # per-axis percent
    assert margin_voxels((0, 0, 0), (10, 10, 10), [0, 50, 100], "percent") == (0, 5, 10)
    # exact multiples must not be rounded up by float error: k% of size 100 is exactly k voxels
    bad = []
    for k in range(1, 101):
        got = margin_voxels((0, 0, 0), (100, 100, 100), k, "percent")[0]
        if got != k:
            bad.append((k, got))
    expect(not bad, f"percent margin: k% of a 100-voxel bbox should be k voxels, got (k, got) = {bad[:6]}"
                    f"{' ...' if len(bad) > 6 else ''} ({len(bad)} of 100 values wrong; float error before ceil)")
    print("OK percent margin (rounding up)" if not bad else "-- percent margin: see FAIL above")


# ---------------------------------------------------------------- clipping
def check_clipping():
    n_fails = len(FAILS)
    seed = make_seed()
    r = seed_rois(seed, margin=3)
    assert roi_log(r[1])["clipped"] is False

    r = seed_rois(seed, margin=20)
    # label 1: z 10-20 < 0, y 20-20 = 0, x 5-20 < 0; z 15+20 = 35, y 30+20 = 50 (= border), x 8+20 = 28
    assert box(r[1]) == ((0, 0, 0), (35, 50, 28)), box(r[1])
    # label 2: hi side clipped
    assert box(r[2]) == ((10, 20, 30), (40, 50, 60)), box(r[2])
    for lab in (1, 2):
        roi = r[lab]
        assert all(0 <= l < h <= n for l, h, n in zip(roi.lo, roi.hi, roi.shape))
        expect(roi_log(roi)["clipped"] is True,
               f"clipped flag: label {lab} margin 20 was cut at the image border but roi_log(...)['clipped'] "
               f"= {roi_log(roi)['clipped']} (seed_roi_from_bbox clips before Roi.box, so Roi.box never sees it)")
    print("OK clipping at image borders" if len(FAILS) == n_fails else "-- clipping: box OK, clipped flag see FAIL above")


# ---------------------------------------------------------------- min size
def check_min_size():
    seed = make_seed()
    # label 1 size (5, 10, 3), min 16: z 10-5..15+6, y 20-3..30+3, x 5-6..8+7
    r = seed_rois(seed, margin=0, min_size=16)
    assert box(r[1]) == ((5, 17, 0), (21, 33, 16)), box(r[1])     # x: 5-6 < 0 -> shifted to [0, 16)
    assert r[1].sizes == (16, 16, 16)
    # larger than the box already: untouched
    r = seed_rois(seed, margin=0, min_size=3)
    assert box(r[1]) == ((10, 20, 5), (15, 30, 8))
    # per-axis list
    r = seed_rois(seed, margin=0, min_size=[0, 0, 30])
    assert box(r[1]) == ((10, 20, 0), (15, 30, 30)), box(r[1])

    # shift at the low border and at the high border
    assert expand_to_min_size((0, 0, 0), (2, 2, 2), SHAPE, 10) == ((0, 0, 0), (10, 10, 10))
    assert expand_to_min_size((38, 48, 58), (40, 50, 60), SHAPE, 10) == ((30, 40, 50), (40, 50, 60))
    # min_size > image size -> the whole axis
    assert expand_to_min_size((10, 10, 10), (12, 12, 12), SHAPE, 100) == ((0, 0, 0), SHAPE)
    r = seed_rois(seed, margin=0, min_size=192)
    assert box(r[1]) == ((0, 0, 0), SHAPE), box(r[1])
    r = seed_rois(seed, margin=0, min_size=[192, 0, 0])
    assert box(r[2]) == ((0, 40, 50), (40, 45, 58)), box(r[2])

    # margin then min size: label 2 size (5, 5, 8) + margin 2 -> (9, 9, 12); min 16 at the high border
    r = seed_rois(seed, margin=2, min_size=16)
    lo, hi = box(r[2])
    assert tuple(h - l for l, h in zip(lo, hi)) == (16, 16, 16), (lo, hi)
    assert hi == (40, 50, 60), (lo, hi)                    # 37 + 4 > 40 etc. -> shifted inwards
    assert all(l <= s for l, s in zip(lo, (30, 40, 50)))  # still contains the seed
    assert raises(lambda: seed_rois(seed, min_size=-1))
    print("OK min_size growth (shift at borders, larger than image, per-axis)")


# ---------------------------------------------------------------- missing labels
def check_missing_labels():
    seed = make_seed()
    assert set(seed_rois(seed, labels=[1, 2, 5])) == {1, 2}
    assert set(seed_rois(seed, labels=[2])) == {2}
    assert seed_rois(seed, labels=[5]) == {}
    assert seed_rois(seed, labels=[0]) == {}
    assert seed_rois(np.zeros(SHAPE, np.uint8)) == {}
    assert set(seed_rois(seed, labels=[1, 2, 5], mode="union")) == {1, 2}
    # labels given as numpy ints
    assert set(seed_rois(seed, labels=np.array([1, 7], np.int64))) == {1}
    print("OK labels missing from the seed are skipped")


# ---------------------------------------------------------------- huge ids
def check_huge_ids():
    seed = np.zeros(SHAPE, np.int32)
    seed[1:4, 2:6, 3:9] = 3
    seed[20:25, 30:31, 40:41] = 20000
    seed[35:40, 0:2, 59:60] = 65000
    b = seed_bboxes(seed)
    assert b == {3: ((1, 2, 3), (4, 6, 9)), 20000: ((20, 30, 40), (25, 31, 41)),
                 65000: ((35, 0, 59), (40, 2, 60))}, b
    # subset, and a huge id that is not in the seed
    b = seed_bboxes(seed, labels=[20000, 30000, 3])
    assert b == {3: ((1, 2, 3), (4, 6, 9)), 20000: ((20, 30, 40), (25, 31, 41))}, b
    b = seed_bboxes(seed, labels=[3, 99999])
    assert b == {3: ((1, 2, 3), (4, 6, 9))}, b
    # same result as the small-id path for the small label
    assert seed_bboxes(seed, labels=[3]) == {3: ((1, 2, 3), (4, 6, 9))}
    # uint16 seed with a huge id
    s16 = seed.astype(np.uint16)
    assert seed_bboxes(s16)[65000] == ((35, 0, 59), (40, 2, 60))
    r = seed_rois(seed, margin=1)
    assert box(r[20000]) == ((19, 29, 39), (26, 32, 42)), box(r[20000])
    print("OK huge label ids (> 10000)")


# ---------------------------------------------------------------- crop + paste
def check_crop_paste():
    rng = np.random.default_rng(0)
    img = rng.random(SHAPE).astype(np.float32)
    seed = make_seed()
    for roi in list(seed_rois(seed, margin=4).values()) + list(seed_rois(seed, margin=20).values()):
        c = roi.crop(img)
        assert c.shape == tuple(h - l for l, h in zip(roi.lo, roi.hi))
        full = np.zeros_like(img)
        out = roi.paste(c, region="bbox", base=full, inplace=True)
        assert out is full
        bb = roi.bbox()
        assert np.array_equal(full[bb], img[bb])
        outside = np.ones(SHAPE, bool)
        outside[bb] = False
        assert not full[outside].any()
        # without base: zeros outside, dtype of the values
        out2 = roi.paste(c, region="bbox")
        assert out2.dtype == img.dtype and np.array_equal(out2, full)
        # point shift: a seed voxel at full-image p is at p - offset in the crop
        p = np.argwhere(seed != 0)[0]
        q = p - np.asarray(roi.offset())
        if all(0 <= v < s for v, s in zip(q, c.shape)):
            assert c[tuple(q)] == img[tuple(p)]
    # uint8 mask pasted into a bool / uint8 full array
    roi = seed_rois(seed, margin=2)[1]
    m = (roi.crop(seed) == 1).astype(np.uint8)
    full = roi.paste(m, region="bbox", base=np.zeros(SHAPE, np.uint8), inplace=True)
    assert np.array_equal(full, (seed == 1).astype(np.uint8))
    print("OK crop + paste round trip")


# ---------------------------------------------------------------- inner face fraction
def check_inner_face():
    seed = make_seed()
    roi = seed_rois(seed, margin=20)[1]           # ((0, 0, 0), (35, 50, 28)): z-lo, y-lo/hi, x-lo on the border
    cs = roi.sizes
    m = np.zeros(cs, np.uint8)
    # border faces only (the z-high and x-high faces are inner: leave out their voxels)
    m[0, :, :-1] = 1                              # z low face = image border
    m[:-1, 0, :-1] = 1                            # y low face = image border
    m[:-1, -1, :-1] = 1                           # y high face = image border (hi = 50)
    m[:-1, :, 0] = 1                              # x low face = image border
    assert inner_face_fraction(m, roi) == 0.0
    assert inner_face_fraction(np.zeros(cs, bool), roi) == 0.0

    m = np.zeros(cs, bool)
    m[5:10, 5:10, 5:10] = True                    # 125 voxels inside
    assert inner_face_fraction(m, roi) == 0.0
    m[-1, 20, 20] = True                          # z high face (35 < 40): inner
    m[20, 20, -1] = True                          # x high face (28 < 60): inner
    f = inner_face_fraction(m, roi)
    assert f > 0 and abs(f - 2 / 127) < 1e-12, f

    # ROI that is the whole image: nothing is an inner face
    whole = seed_rois(seed, margin=0, min_size=1000)[1]
    assert box(whole) == ((0, 0, 0), SHAPE)
    assert inner_face_fraction(np.ones(SHAPE, bool), whole) == 0.0

    # ROI strictly inside: every face is inner; a full mask touches all six faces
    inner = seed_rois(seed, margin=1)[1]          # ((9, 19, 4), (16, 31, 9))
    full = np.ones(inner.sizes, bool)
    n_face = full.size - np.ones(tuple(s - 2 for s in inner.sizes), bool).size
    assert abs(inner_face_fraction(full, inner) - n_face / full.size) < 1e-12
    print("OK inner_face_fraction")


# ---------------------------------------------------------------- roi_log
def check_roi_log():
    seed = make_seed()
    roi = seed_rois(seed, margin=[1, 2, 3], min_size=[0, 0, 12])[1]
    d = roi_log(roi)
    keys = {"bbox_start", "bbox_stop", "crop_shape", "seed_bbox_start", "seed_bbox_stop",
            "margin_voxels", "margin_unit", "min_size", "clipped"}
    assert set(d) == keys, set(d) ^ keys
    assert d["bbox_start"] == list(roi.lo) and d["bbox_stop"] == list(roi.hi)
    assert d["crop_shape"] == [h - l for l, h in zip(roi.lo, roi.hi)] == list(roi.crop(seed).shape)
    assert d["seed_bbox_start"] == [10, 20, 5] and d["seed_bbox_stop"] == [15, 30, 8]
    assert d["margin_voxels"] == [1, 2, 3] and d["margin_unit"] == "voxel"
    assert d["min_size"] == [0, 0, 12]
    assert d["clipped"] is False
    assert d["crop_shape"][2] == 12                # x: 3 + 2 * 3 = 9 -> grown to 12
    d2 = roi_log(seed_rois(seed, margin=50, unit="percent")[2])
    assert d2["margin_unit"] == "percent" and d2["margin_voxels"] == [3, 3, 4], d2["margin_voxels"]
    # YAML-friendly: plain python types only (safe_dump refuses numpy scalars)
    for dd in (d, d2):
        txt = yaml.safe_dump(dd)
        assert yaml.safe_load(txt) == dd
    print("OK roi_log fields")


# ---------------------------------------------------------------- config
ROI_KEYS = ["roi_from_seed", "roi_mode", "roi_margin", "roi_margin_unit", "roi_min_size",
            "do_autozoom", "save_prob_maps"]
RULES = {k: config_core.input_val_nninteractive_run[k] for k in ROI_KEYS}


def validate(cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        return config_core.validate_input_yaml(dict(cfg), RULES)


def invalid(cfg, word):
    try:
        validate(cfg)
    except ValueError as e:
        assert word in str(e), (cfg, str(e))
        return True
    return False


def check_config():
    defaults = validate({})
    assert defaults == {"roi_from_seed": False, "roi_mode": "class", "roi_margin": 32,
                        "roi_margin_unit": "voxel", "roi_min_size": 0, "do_autozoom": True,
                        "save_prob_maps": False}, defaults

    ok = [
        {"roi_from_seed": True, "roi_mode": "union", "roi_margin": 16, "roi_margin_unit": "percent",
         "roi_min_size": 192, "do_autozoom": False, "save_prob_maps": True},
        {"roi_margin": 12.5},
        {"roi_margin": [8, 16, 16.5]},
        {"roi_margin": 0, "roi_min_size": [192, 192, 96]},
        {"roi_min_size": 0.0},
    ]
    for cfg in ok:
        out = validate(cfg)
        for k, v in cfg.items():
            assert out[k] == v, (k, out[k], v)
    # a list written as a string (e.g. a CSV cell) is parsed
    cfg = {"roi_margin": "[4, 8, 8]"}
    validate(cfg)

    bad = [
        ({"roi_mode": "box"}, "roi_mode"),
        ({"roi_margin_unit": "mm"}, "roi_margin_unit"),
        ({"roi_margin": -1}, "roi_margin"),
        ({"roi_margin": [1, -2, 3]}, "roi_margin"),
        ({"roi_margin": [1, 2]}, "roi_margin"),
        ({"roi_margin": [1, 2, 3, 4]}, "roi_margin"),
        ({"roi_margin": ["a", 2, 3]}, "roi_margin"),
        ({"roi_margin": "big"}, "roi_margin"),
        ({"roi_min_size": -192}, "roi_min_size"),
        ({"roi_min_size": [192, 192]}, "roi_min_size"),
        ({"roi_from_seed": "yes"}, "roi_from_seed"),
        ({"do_autozoom": 1}, "do_autozoom"),
        ({"roi_margins": 3}, "roi_margins"),       # unknown key
    ]
    for cfg, word in bad:
        assert invalid(cfg, word), f"should be rejected: {cfg}"

    # the template parses and its roi keys validate
    with open("template/nninteractive_predict.yaml") as f:
        tpl = yaml.safe_load(f)
    out = validate({k: tpl[k] for k in ROI_KEYS if k in tpl})
    assert all(k in tpl for k in ROI_KEYS), [k for k in ROI_KEYS if k not in tpl]
    assert out["roi_from_seed"] is False and out["do_autozoom"] is True
    print("OK config validation (roi_* keys, number-or-list)")


if __name__ == "__main__":
    print(f"python {sys.version.split()[0]}, numpy {np.__version__}")
    check_class_union()
    check_voxel_margin()
    check_percent_margin()
    check_clipping()
    check_min_size()
    check_missing_labels()
    check_huge_ids()
    check_crop_paste()
    check_inner_face()
    check_roi_log()
    check_config()
    if FAILS:
        print(f"\n{len(FAILS)} FAIL(s) in seed_roi.py:")
        for m in FAILS:
            print(f"  - {m}")
        sys.exit(1)
    print("\nALL OK")
