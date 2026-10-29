import numpy as np
import torch
from typing import Optional
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

class nnInteractiveSessionWithProb(nnInteractiveInferenceSession):
    """
    Extension of nnInteractiveInferenceSession that also maintains a float32
    foreground probability map (prob_buffer) in addition to the binary target_buffer.

    A forward hook captures softmax(logits)[1] (foreground probability) after each
    network forward pass. A patched paste_tensor writes those probabilities into
    prob_buffer whenever the binary prediction is written into target_buffer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prob_buffer: Optional[torch.Tensor] = None
        self._last_fg_prob: Optional[torch.Tensor] = None  # CPU float32 tensor, updated by hook

    def initialize_from_trained_model_folder(self, *args, **kwargs):
        super().initialize_from_trained_model_folder(*args, **kwargs)
        # Register hook to capture foreground probability after every forward pass
        self.network.register_forward_hook(self._capture_fg_prob_hook)

    def _capture_fg_prob_hook(self, module, input, output):
        """
        Fires after every network forward pass.
        output shape: (1, 2, *patch_size) -- two logit channels (bg, fg).
        Stores softmax fg probability as a CPU float32 tensor.
        """
        logits = output[0].float()  # (2, *patch_size) on device
        self._last_fg_prob = torch.softmax(logits, dim=0)[1].detach().cpu()

    def set_target_buffer(self, target_buffer):
        super().set_target_buffer(target_buffer)
        # Initialise a matching float32 prob_buffer
        if isinstance(target_buffer, torch.Tensor):
            self.prob_buffer = torch.zeros(target_buffer.shape, dtype=torch.float32)
        elif isinstance(target_buffer, np.ndarray):
            self.prob_buffer = np.zeros(target_buffer.shape, dtype=np.float32)

    def reset_interactions(self):
        super().reset_interactions()
        if self.prob_buffer is not None:
            if isinstance(self.prob_buffer, torch.Tensor):
                self.prob_buffer.zero_()
            else:
                self.prob_buffer.fill(0.0)
        self._last_fg_prob = None

    @torch.inference_mode()
    def _predict(self, force_full_refine: bool = False):
        """
        Override _predict to also write foreground probabilities into prob_buffer.

        We temporarily replace paste_tensor in the inference_session module's namespace
        with a wrapper. Whenever paste_tensor writes into target_buffer, the wrapper
        also writes the corresponding fg probability into prob_buffer.
        The original function is always restored in the finally block.
        """
        import nnInteractive.inference.inference_session as _sess_mod
        from nnInteractive.utils.crop import paste_tensor as _orig_paste

        session = self

        def _prob_paste(dst, src, bbox):
            _orig_paste(dst, src, bbox)
            if (
                dst is session.target_buffer
                and session._last_fg_prob is not None
                and session.prob_buffer is not None
            ):
                fg_prob = session._last_fg_prob
                # Resize fg_prob if the prediction patch was spatially resampled (autozoom)
                if list(fg_prob.shape) != list(src.shape):
                    fg_prob = torch.nn.functional.interpolate(
                        fg_prob[None, None].float(),
                        list(src.shape),
                        mode='trilinear',
                        align_corners=False,
                    )[0, 0]
                prob_dst = (
                    session.prob_buffer
                    if isinstance(session.prob_buffer, torch.Tensor)
                    else torch.from_numpy(session.prob_buffer)
                )
                _orig_paste(prob_dst, fg_prob.cpu(), bbox)
                if isinstance(session.prob_buffer, np.ndarray):
                    session.prob_buffer[:] = prob_dst.numpy()

        _sess_mod.paste_tensor = _prob_paste
        try:
            super()._predict()
        finally:
            _sess_mod.paste_tensor = _orig_paste  # always restore original
