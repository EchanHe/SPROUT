"""
nnInteractiveSessionWithProb: an nnInteractive session that also keeps a float32
foreground-probability map (prob_buffer) next to the binary target_buffer.

Works with nnInteractive 1.x (tested 1.1.3) and 2.x (tested 2.6.0). The two versions
write predictions into the target buffer differently, so the probability capture uses a
different seam per version (chosen by feature detection, see _PROB_CAPTURE_MODE):

* "legacy" (1.x): every target_buffer write is ``paste_tensor(target_buffer, pred, bbox)``
  directly after the network forward that produced ``pred`` (initial pass and each
  refinement patch). We temporarily replace the module-level ``paste_tensor`` of
  ``nnInteractive.inference.inference_session`` and paste softmax(last logits)[1] into
  prob_buffer at the same bbox.

* "v2" (2.x): ``paste_tensor`` takes a ``channel_idx`` kwarg, and refinement runs all patch
  forwards first (writing into a local cache, later patches overwriting earlier ones) and
  pastes the refined boxes into target_buffer only after the loop — so "last forward" no
  longer belongs to the box being pasted. Instead:
    - refinement (``_refine_coarse_with_local_cache``): the k-th network forward inside it is
      the patch of ``bboxes_ordered[k]``; its fg probability is pasted into prob_buffer at
      that bbox right away. Same boxes, same order, last-writer-wins => identical
      composition to what ends up in target_buffer.
    - no-zoom path (``_paste_prediction_to_target_buffer`` outside refinement): the single
      forward's probability is pasted at the same bbox.
  2.x keeps the image uncropped, so all bboxes are already target-buffer coordinates.

Limitations: voxels that are written into target_buffer without a network prediction
(add_initial_seg_interaction and never re-predicted) keep probability 0; 2.x undo() does
not roll back prob_buffer.
"""
import importlib.metadata
import inspect
from typing import Optional

import numpy as np
import torch
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

try:
    NNINTERACTIVE_VERSION = importlib.metadata.version("nnInteractive")
except Exception:  # pragma: no cover - editable / unusual installs
    NNINTERACTIVE_VERSION = getattr(__import__("nnInteractive"), "__version__", "unknown")

_V2_SEAMS = ("_refine_coarse_with_local_cache", "_paste_prediction_to_target_buffer")
_PROB_CAPTURE_MODE = (
    "v2" if all(hasattr(nnInteractiveInferenceSession, m) for m in _V2_SEAMS) else "legacy"
)


def _fg_prob(logits: torch.Tensor) -> torch.Tensor:
    """(2, *patch) logits -> (*patch) float32 CPU foreground probability."""
    return torch.softmax(logits.float(), dim=0)[1].detach().cpu()


class nnInteractiveSessionWithProb(nnInteractiveInferenceSession):
    """
    Extension of nnInteractiveInferenceSession that also maintains a float32
    foreground probability map (prob_buffer) in addition to the binary target_buffer.

    A forward hook keeps the logits of the latest network forward pass; the fg
    probability softmax(logits)[1] is written into prob_buffer wherever the matching
    binary prediction is written into target_buffer (see module docstring).
    """

    prob_capture_mode = _PROB_CAPTURE_MODE
    nninteractive_version = NNINTERACTIVE_VERSION

    def __init__(self, *args, **kwargs):
        # Drop constructor kwargs this nnInteractive version does not know
        # (e.g. use_pinned_memory exists in 1.x but not in 2.x).
        params = inspect.signature(nnInteractiveInferenceSession.__init__).parameters
        if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
            kwargs = {k: v for k, v in kwargs.items() if k in params}
        super().__init__(*args, **kwargs)
        self.prob_buffer: Optional[torch.Tensor] = None
        self._last_logits: Optional[torch.Tensor] = None  # (2, *patch) of the latest forward, on device
        self._hooked_network = None
        self._hook_handle = None
        # v2 refinement bookkeeping: list of bboxes being refined and index of the next forward
        self._refine_bboxes = None
        self._refine_idx = 0

    # ------------------------------------------------------------------ hook
    def initialize_from_trained_model_folder(self, *args, **kwargs):
        super().initialize_from_trained_model_folder(*args, **kwargs)
        self._ensure_hook()

    def _ensure_hook(self):
        net = getattr(self, "network", None)
        if net is None or net is self._hooked_network:
            return
        if self._hook_handle is not None:
            self._hook_handle.remove()
        self._hook_handle = net.register_forward_hook(self._capture_fg_prob_hook)
        self._hooked_network = net

    def _capture_fg_prob_hook(self, module, input, output):
        """
        Fires after every network forward pass. output: (1, 2, *patch_size) logits (bg, fg).
        """
        self._last_logits = output[0].detach()
        if self._refine_bboxes is not None:
            # v2 refinement: forward k belongs to bboxes_ordered[k]
            k = self._refine_idx
            self._refine_idx += 1
            if k < len(self._refine_bboxes):
                self._paste_prob(self._take_fg_prob(), self._refine_bboxes[k])

    def _take_fg_prob(self) -> Optional[torch.Tensor]:
        logits, self._last_logits = self._last_logits, None
        return None if logits is None else _fg_prob(logits)

    def _paste_prob(self, fg_prob: Optional[torch.Tensor], bbox, shape=None):
        """Paste fg_prob into prob_buffer at bbox (target-buffer coordinates; may exceed bounds)."""
        if fg_prob is None or self.prob_buffer is None:
            return
        from nnInteractive.utils.crop import paste_tensor
        # Resize if the pasted prediction was spatially resampled
        if shape is not None and list(fg_prob.shape) != list(shape):
            fg_prob = torch.nn.functional.interpolate(
                fg_prob[None, None].float(), list(shape), mode='trilinear', align_corners=False,
            )[0, 0]
        dst = self.prob_buffer if isinstance(self.prob_buffer, torch.Tensor) else torch.from_numpy(self.prob_buffer)
        paste_tensor(dst, fg_prob.cpu(), bbox)  # numpy prob_buffer is written through the shared-memory view

    # ------------------------------------------------------------------ buffers
    def set_target_buffer(self, target_buffer):
        super().set_target_buffer(target_buffer)
        # Initialise a matching float32 prob_buffer
        if isinstance(target_buffer, torch.Tensor):
            self.prob_buffer = torch.zeros(target_buffer.shape, dtype=torch.float32)
        elif isinstance(target_buffer, np.ndarray):
            self.prob_buffer = np.zeros(target_buffer.shape, dtype=np.float32)

    def reset_interactions(self, *args, **kwargs):
        super().reset_interactions(*args, **kwargs)
        if self.prob_buffer is not None:
            if isinstance(self.prob_buffer, torch.Tensor):
                self.prob_buffer.zero_()
            else:
                self.prob_buffer.fill(0.0)
        self._last_logits = None

    # ------------------------------------------------------------------ v2 seams
    def _paste_prediction_to_target_buffer(self, prediction, bbox, *args, **kwargs):
        out = super()._paste_prediction_to_target_buffer(prediction, bbox, *args, **kwargs)
        if self._refine_bboxes is None:
            # no-zoom path: the prediction comes from the latest (single) forward
            self._paste_prob(self._take_fg_prob(), bbox, shape=prediction.shape)
        # during refinement the probs were already pasted per forward (hook)
        return out

    def _refine_coarse_with_local_cache(self, bboxes_ordered, *args, **kwargs):
        self._last_logits = None
        self._refine_bboxes = [list(map(list, b)) for b in bboxes_ordered]
        self._refine_idx = 0
        try:
            return super()._refine_coarse_with_local_cache(bboxes_ordered, *args, **kwargs)
        finally:
            n_fwd, n_box = self._refine_idx, len(self._refine_bboxes)
            self._refine_bboxes = None
            self._refine_idx = 0
            if n_fwd != n_box:
                print(f"WARNING nnInteractiveSessionWithProb: {n_fwd} refinement forwards for {n_box} boxes; "
                      f"prob_buffer may be misaligned (nnInteractive {NNINTERACTIVE_VERSION})")

    # ------------------------------------------------------------------ predict
    @torch.inference_mode()
    def _predict(self, force_full_refine: bool = False):
        """
        Run the base prediction and also write foreground probabilities into prob_buffer.
        """
        self._ensure_hook()
        self._last_logits = None
        if self.prob_capture_mode == "v2":
            return super()._predict(force_full_refine=force_full_refine)
        return self._predict_legacy()

    def _predict_legacy(self):
        """
        nnInteractive 1.x: temporarily replace paste_tensor in the inference_session module
        with a wrapper that, whenever it writes into target_buffer, also writes the fg
        probability of the latest forward into prob_buffer at the same bbox. The original
        function is always restored in the finally block.
        """
        import nnInteractive.inference.inference_session as _sess_mod
        _orig_paste = _sess_mod.paste_tensor
        session = self

        def _prob_paste(dst, src, bbox, *args, **kwargs):
            out = _orig_paste(dst, src, bbox, *args, **kwargs)
            if dst is session.target_buffer and not args and not kwargs:
                session._paste_prob(session._take_fg_prob(), bbox, shape=src.shape)
            return out

        _sess_mod.paste_tensor = _prob_paste
        try:
            return super()._predict()
        finally:
            _sess_mod.paste_tensor = _orig_paste  # always restore original
