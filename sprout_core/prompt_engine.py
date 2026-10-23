"""
Promptable segmentation engines (no Qt, no torch at import time).

An engine wraps one interactive model (nnInteractive now, SAM later) behind a
small synchronous API. What prompt INPUT TYPES the user may choose is driven by
the engine's `capabilities`: the UI offers only the prompt types whose required
capabilities the engine declares, and prompt_seg.run_classes rejects the rest.

Capabilities used so far:
    "points"    positive / negative 3D points
    "scribble"  positive / negative scribble masks
    "init_mask" an initial segmentation to refine
    "bbox"      boxes (a future SAM engine; not used by nnInteractive v1.0 weights)
    "lasso"     lasso masks (not in v1)
"""

CAPABILITIES = ("points", "scribble", "init_mask", "bbox", "lasso")

# prompt_type -> (required capabilities, label for the UI)
PROMPT_TYPES = {
    "points": (frozenset({"points"}), "Points"),
    "points+init": (frozenset({"points", "init_mask"}), "Points + seed as initial mask"),
    "scribble": (frozenset({"scribble"}), "Scribbles (+ points)"),
}


class PromptEngineError(RuntimeError):
    """User-facing engine problem (not installed, not loaded, out of memory, ...)."""


class EngineOutOfMemory(PromptEngineError):
    """The model ran out of (GPU) memory."""


def is_oom(exc):
    """True for torch's OutOfMemoryError (recognised by name: no torch import needed)."""
    return type(exc).__name__ == "OutOfMemoryError" or isinstance(exc, EngineOutOfMemory)


def required_capabilities(prompt_type):
    try:
        return PROMPT_TYPES[prompt_type][0]
    except KeyError:
        raise ValueError(f"unknown prompt type {prompt_type!r}; known: {', '.join(PROMPT_TYPES)}") from None


class PromptEngine:
    """Abstract interactive segmentation engine.

    Coordinates are voxel indices (z, y, x) of the image given to set_image (usually a
    crop of the full image). All methods are synchronous; an implementation may run
    them on its own worker thread.

    Attributes
    ----------
    name : str
    capabilities : frozenset of str (see CAPABILITIES)
    loaded : bool
    info : dict (at least 'version', 'device')
    active : (image_key, class_id) the model's interaction state currently belongs to,
             or None. Lets callers add prompts to the last class without a reset.
    """

    name = "abstract"
    capabilities = frozenset()

    def __init__(self):
        self.active = None

    # -- capabilities ------------------------------------------------------
    def supports(self, *caps):
        return all(c in self.capabilities for c in caps)

    def supports_prompt_type(self, prompt_type):
        return required_capabilities(prompt_type) <= self.capabilities

    def available_prompt_types(self):
        """[(prompt_type, label)] this engine can run, in PROMPT_TYPES order."""
        return [(k, lab) for k, (req, lab) in PROMPT_TYPES.items() if req <= self.capabilities]

    def require_prompt_type(self, prompt_type):
        req = required_capabilities(prompt_type)
        missing = sorted(req - self.capabilities)
        if missing:
            raise ValueError(f"{self.name} does not support prompt type {prompt_type!r} "
                             f"(missing: {', '.join(missing)}); supported: "
                             f"{', '.join(k for k, _ in self.available_prompt_types()) or 'none'}")

    # -- interface ---------------------------------------------------------
    @property
    def loaded(self):
        raise NotImplementedError

    @property
    def info(self):
        raise NotImplementedError

    def load(self, model_dir, device=None):
        """Load (or reuse) the model; returns info."""
        raise NotImplementedError

    def set_image(self, img3d, key):
        """Give the model a 3D image. No-op (returns False) if `key` is unchanged."""
        raise NotImplementedError

    def valid_point(self, z, y, x):
        """Whether a point (image coordinates) can be used by the model."""
        raise NotImplementedError

    def predict_class(self, points, *, init_mask=None, scribble=None, neg_scribble=None, warm=False,
                      cancel=None, on_point=None, seed=None, class_id=None):
        """Segment one object; returns a bool mask of the image shape.

        points: [(z, y, x, positive)] in image coordinates, in the order given.
        warm: add the prompts to the current interaction state instead of starting over.
        """
        raise NotImplementedError

    def undo(self):
        """Undo the last interaction if the model supports it. Returns True if undone."""
        return False

    def release(self):
        """Free the model and its (GPU) memory."""
        raise NotImplementedError
