"""
Check nnInteractiveSessionWithProb (sprout_core/nni_prob_session.py) with the REAL model,
on nnInteractive 1.1.3 or 2.6.0 (whichever the running env has).

    python check_prob_session.py [--device cuda:1] [--weights ...] [--old path/to/old_nni_prob_session.py]

Cases: a (160,200,200) foram crop and a (250,161,122) dog crop (both images contain zeros, so the
1.x nonzero-crop offset is exercised). Per case: positive point -> predict; whole-seed scribble
-> predict (autozoom + refinement); negative point -> predict; then reset + initial seg with
run_prediction=True (force_full_refine).
After every _predict: prob_buffer.shape == target.shape; agreement of (prob > 0.5) with target_buffer
(Dice, % disagreeing voxels) overall and inside the boxes pasted by that predict.
--old (1.x only): runs the same sequence with the old class and requires identical prob/target.
Prints PASS/FAIL/INFO; exits non-zero on failure.
"""
import argparse
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda:1")
ap.add_argument("--weights", default=r"C:/Users/yichen/work/weights/nnInteractive_v1.0")
ap.add_argument("--old", default=None, help="file with the pre-change nnInteractiveSessionWithProb (1.x compare)")
ap.add_argument("--max-disagree", type=float, default=1e-4, help="max fraction of disagreeing predicted voxels")
args = ap.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from sprout_core import nni_prob_session as NPS  # noqa: E402

FAILED = []


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILED.append(msg)


def info(msg):
    print("INFO " + msg, flush=True)


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


info(f"nnInteractive {NPS.NNINTERACTIVE_VERSION}, capture mode {NPS.nnInteractiveSessionWithProb.prob_capture_mode}, "
     f"torch {torch.__version__}, device {args.device}")


def make_session(cls):
    with quiet():
        s = cls(device=torch.device(args.device), verbose=False, use_torch_compile=False, do_autozoom=True,
                use_pinned_memory=False)
        s.initialize_from_trained_model_folder(args.weights)
    return s


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def box_mask(shape, boxes):
    m = np.zeros(shape, bool)
    for b in boxes:
        m[tuple(slice(max(lo, 0), min(hi, n)) for (lo, hi), n in zip(b, shape))] = True
    return m


def load_cases():
    cases = []
    img = tifffile.imread(ROOT / "demo_data/foram_img.tif")[30:190, 360:560, 360:560]
    seed = tifffile.imread(ROOT / "demo_data/foram_seed.tif")[30:190, 360:560, 360:560]
    cases.append(("foram(160,200,200) class 2", img, seed == 2, seed == 3))
    img = tifffile.imread(ROOT / "demo_data/dog_img.tif")[200:450]
    seed = tifffile.imread(ROOT / "demo_data/dog_seed.tif")[200:450]
    cases.append(("dog(250,161,122) class 6", img, seed == 6, seed == 5))
    return cases


def run_sequence(session, img, pos, neg, record):
    """Run the prompt sequence; returns list of (step, target, prob, pasted_boxes)."""
    out = []
    np.random.seed(0)  # nnInteractive's refinement bbox planner has a np.random fallback
    torch.manual_seed(0)
    pos_idx = np.argwhere(pos)
    neg_idx = np.argwhere(neg)
    p_pt = tuple(int(i) for i in pos_idx[len(pos_idx) // 2])
    n_pt = tuple(int(i) for i in neg_idx[len(neg_idx) // 2])
    with quiet():
        session.set_image(img[None])
    target = torch.zeros(img.shape, dtype=torch.uint8)
    session.reset_interactions()
    session.set_target_buffer(target)

    def step(name, fn):
        record.clear()
        with quiet():
            fn()
            session._predict()
        out.append((name, session.target_buffer.cpu().numpy().copy(), np.asarray(session.prob_buffer).copy(),
                    list(record)))
        if getattr(record, "refined", None) is not None and NPS._PROB_CAPTURE_MODE == "v2":
            info(f"  [{name}] v2 pastes: {sum(record.refined)} from refinement forwards, "
                 f"{len(record.refined) - sum(record.refined)} from the no-zoom path")

    step("pos point", lambda: session.add_point_interaction(p_pt, include_interaction=True, run_prediction=False))
    step("scribble", lambda: session.add_scribble_interaction(pos.astype(np.uint8), include_interaction=True,
                                                               run_prediction=False))
    step("neg point", lambda: session.add_point_interaction(n_pt, include_interaction=False, run_prediction=False))

    # initial seg (+ force_full_refine on 2.x) in a fresh object
    session.reset_interactions()
    session.set_target_buffer(torch.zeros(img.shape, dtype=torch.uint8))
    init = np.zeros(img.shape, np.uint8)
    init[tuple(slice(i.min(), i.max() + 1) for i in np.nonzero(pos))] = 1   # seed bbox as a crude init mask
    record.clear()
    with quiet():
        session.add_initial_seg_interaction(init, run_prediction=True)
    out.append(("init seg+predict", session.target_buffer.cpu().numpy().copy(),
                np.asarray(session.prob_buffer).copy(), list(record), init.astype(bool)))
    return out


class _Rec(list):
    def __init__(self):
        super().__init__()
        self.refined = []

    def clear(self):
        super().clear()
        self.refined.clear()


def instrument(session):
    """Record the bboxes the session pastes probabilities into."""
    record = _Rec()
    refined = record.refined
    orig = session._paste_prob

    def wrapped(fg_prob, bbox, shape=None):
        record.append([list(b) for b in bbox])
        refined.append(getattr(session, "_refine_bboxes", None) is not None)
        return orig(fg_prob, bbox, shape)

    session._paste_prob = wrapped
    return record


new = make_session(NPS.nnInteractiveSessionWithProb)
old = None
if args.old and NPS.nnInteractiveSessionWithProb.prob_capture_mode == "legacy":
    spec = importlib.util.spec_from_file_location("old_nni_prob_session", args.old)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    old = make_session(mod.nnInteractiveSessionWithProb)

for name, img, pos, neg in load_cases():
    print(f"---- {name}: image {img.shape}, nonzero bbox "
          f"{[(int(i.min()), int(i.max()) + 1) for i in np.nonzero(img)]}", flush=True)
    rec = instrument(new)
    try:
        res = run_sequence(new, img, pos, neg, rec)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        ok(False, f"{name}: sequence raised {type(e).__name__}: {e}")
        continue
    ok(True, f"{name}: sequence ran without exception")
    for r in res:
        step, tgt, prob, boxes = r[:4]
        t = tgt > 0
        p = prob > 0.5
        ok(prob.shape == tgt.shape and prob.dtype == np.float32, f"{name} [{step}]: prob shape/dtype {prob.shape}")
        dis = np.logical_xor(t, p)
        written = box_mask(t.shape, boxes)
        dis_in = dis[written].mean() if written.any() else 0.0
        info(f"{name} [{step}]: fg {int(t.sum())}, pasted boxes {len(boxes)} "
             f"({100 * written.mean():.1f}% of volume), Dice(prob>.5, target) {dice(t, p):.6f}, "
             f"disagree overall {100 * dis.mean():.5f}% ({int(dis.sum())} vox), inside boxes {100 * dis_in:.5f}%, "
             f"prob range [{prob.min():.3f}, {prob.max():.3f}]")
        if len(r) == 5:
            # initial seg: voxels never re-predicted keep target=init, prob=0 (documented limitation)
            stale = r[4] & ~written
            ok(dis_in <= args.max_disagree, f"{name} [{step}]: inside predicted boxes disagreement "
                                            f"{100 * dis_in:.5f}% <= {100 * args.max_disagree:.3f}%")
            info(f"{name} [{step}]: init-mask voxels outside predicted boxes {int(stale.sum())} "
                 f"(prob stays 0 there)")
        else:
            ok(dis.mean() <= args.max_disagree, f"{name} [{step}]: disagreement {100 * dis.mean():.5f}% "
                                                f"<= {100 * args.max_disagree:.3f}%")
    if old is not None:
        rec_old = []
        try:
            res_old = run_sequence(old, img, pos, neg, rec_old)
        except Exception as e:  # noqa: BLE001
            ok(False, f"{name}: OLD class raised {type(e).__name__}: {e}")
            continue
        for a, b in zip(res, res_old):
            same_t = np.array_equal(a[1], b[1])
            same_p = np.array_equal(a[2], b[2])
            ok(same_t and same_p, f"{name} [{a[0]}]: new == old implementation (target {same_t}, prob {same_p}; "
                                  f"max |dprob| {np.abs(a[2] - b[2]).max():.3g})")

print("\nALL PASS" if not FAILED else f"\n{len(FAILED)} FAILED:\n  " + "\n  ".join(FAILED))
sys.exit(1 if FAILED else 0)
