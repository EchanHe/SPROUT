"""
nnInteractive engine (no Qt; torch / nnInteractive are imported only when a model
is loaded).

Works with nnInteractive 1.1.x and 2.x: the session is built with the keyword
arguments its __init__ accepts (2.x removed use_pinned_memory and added
interactions_storage / enable_undo). Differences handled here:
  - 1.x crops the image to its nonzero bounding box in set_image, so points outside
    that box cannot be used (valid_point); 2.x does not crop.
  - 1.x gets a float32 image (as sprout_core.nninteractive_predict does); 2.x
    converts the raw dtype itself.
  - undo() exists only in 2.x (and only with enable_undo=True).

Determinism (default ON): cudnn.benchmark=False and cudnn.deterministic=True are
set right after the session is constructed (2.x turns benchmark on in __init__ and
runs a warm-up forward pass while loading the weights, so the flags must be set
before initialize_from_trained_model_folder). These are PROCESS-GLOBAL torch
settings. np.random is reseeded before each class (seed=... in predict_class):
nnInteractive's refinement fallback draws from the global np.random stream.
"""
import gc
import inspect
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from sprout_core.prompt_engine import EngineOutOfMemory, PromptEngine, PromptEngineError, is_oom

INSTALL_HINT = ("nnInteractive is not installed in this Python environment. Install: "
                "`pip install nninteractive` (needs PyTorch; for a GPU install the CUDA build of torch "
                "first, see pytorch.org). Prompts can still be generated and exported.")

OOM_HINT = ("Out of GPU memory. Use an ROI (the model then runs on a smaller crop), close other "
            "programs that use the GPU, or switch to the CPU (very slow).")

_DROPPED_LOGGED = set()


def nninteractive_status():
    """(True, version) if nnInteractive can be imported, else (False, INSTALL_HINT).

    Never imports nnInteractive or torch (find_spec of the top package first: a
    submodule lookup raises when the package is absent).
    """
    try:
        import importlib.util
        if importlib.util.find_spec("nnInteractive") is None:
            return False, INSTALL_HINT
        if importlib.util.find_spec("nnInteractive.inference.inference_session") is None:
            return False, INSTALL_HINT
    except Exception:
        return False, INSTALL_HINT
    try:
        import importlib.metadata as md
        version = md.version("nnInteractive")
    except Exception:
        version = "unknown"
    return True, version


def filter_kwargs(fn, kwargs, log=print):
    """Keep the keyword arguments `fn` accepts (all of them if it takes **kwargs).

    Dropped names are logged once per (function, names).
    """
    params = inspect.signature(fn).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    kept = {k: v for k, v in kwargs.items() if k in params}
    dropped = tuple(sorted(set(kwargs) - set(kept)))
    if dropped:
        key = (getattr(fn, "__qualname__", repr(fn)), getattr(fn, "__module__", ""), dropped)
        if key not in _DROPPED_LOGGED:
            _DROPPED_LOGGED.add(key)
            if log is not None:
                log(f"[nnInteractive] {key[0]} does not accept {', '.join(dropped)}; not passed")
    return kept


def resolve_device(device=None):
    """'cuda' if available else 'cpu' for None; otherwise str(device). Imports torch only for None."""
    if device is None:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def set_deterministic():
    """cudnn.benchmark=False, cudnn.deterministic=True (process-global torch settings)."""
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_session(device=None, *, session_cls=None, verbose=False, do_autozoom=True, use_pinned_memory=False,
                 interactions_storage="auto", enable_undo=False, use_torch_compile=False, deterministic=True):
    """Construct an nnInteractiveInferenceSession for the installed version (weights not loaded).

    device None -> cuda if available else cpu. With deterministic=True the cuDNN flags
    are set right after construction (before the weights are loaded / warmed up).
    """
    import torch
    if session_cls is None:
        from nnInteractive.inference.inference_session import nnInteractiveInferenceSession as session_cls
    dev = torch.device(resolve_device(device))
    kwargs = filter_kwargs(session_cls.__init__, dict(
        device=dev, use_torch_compile=use_torch_compile, verbose=verbose, do_autozoom=do_autozoom,
        use_pinned_memory=use_pinned_memory, interactions_storage=interactions_storage,
        enable_undo=enable_undo))
    session = session_cls(**kwargs)
    if deterministic:
        set_deterministic()
    return session


def session_major(session):
    """Major nnInteractive version of a session (from its __init__ signature)."""
    params = inspect.signature(type(session).__init__).parameters
    if "interactions_storage" in params or "enable_undo" in params:
        return 2
    if "use_pinned_memory" in params:
        return 1
    try:
        import importlib.metadata as md
        return int(md.version("nnInteractive").split(".")[0])
    except Exception:
        return 2


def _session_version(session):
    mod = type(session).__module__ or ""
    if mod.startswith("nnInteractive"):
        try:
            import importlib.metadata as md
            return md.version("nnInteractive")
        except Exception:
            pass
    return str(getattr(session, "__version__", "unknown"))


def _mark_thread(holder):
    holder["ident"] = threading.get_ident()


def _empty_cuda_cache():
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class NniEngine(PromptEngine):
    """nnInteractive behind the PromptEngine API.

    Every session call runs on one dedicated worker thread ("sprout-nni"): the session
    is not thread-safe and cuDNN's algorithm cache is per thread (the 2.x warm-up runs
    there too). Public methods block until done, so call them from a background
    thread in a GUI.
    """

    name = "nnInteractive"
    capabilities = frozenset({"points", "scribble", "init_mask"})

    def __init__(self, session_factory=make_session, buffer_factory=None):
        super().__init__()
        self._factory = session_factory
        self._buffer_factory = buffer_factory
        self._thread = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sprout-nni",
                                            initializer=_mark_thread, initargs=(self._thread,))
        self._session = None
        self._load_key = None
        self._info = {}
        self._major = None
        self._image_key = None
        self._shape = None
        self._buf = None
        self._valid_lo = self._valid_hi = None

    # -- threading ---------------------------------------------------------
    def _run(self, fn, *args, **kwargs):
        if threading.get_ident() == self._thread.get("ident"):
            return fn(*args, **kwargs)
        return self._executor.submit(fn, *args, **kwargs).result()

    # -- state -------------------------------------------------------------
    @property
    def loaded(self):
        return self._session is not None

    @property
    def info(self):
        return dict(self._info)

    @property
    def major(self):
        return self._major

    @property
    def image_key(self):
        return self._image_key

    @property
    def session(self):
        return self._session

    def supports_undo(self):
        s = self._session
        return s is not None and hasattr(s, "undo") and bool(getattr(s, "supports_undo", False))

    # -- model -------------------------------------------------------------
    def load(self, model_dir, device=None):
        """Load the weights once per (model_dir, device); a different key releases first."""
        dev = resolve_device(device)
        key = (str(model_dir), dev)
        if self._session is not None and self._load_key == key:
            return self.info
        self.release()

        def work():
            t0 = time.perf_counter()
            s = self._factory(dev)
            s.initialize_from_trained_model_folder(str(model_dir))
            return s, time.perf_counter() - t0

        try:
            session, seconds = self._run(work)
        except Exception as e:
            if is_oom(e):
                self._run(_empty_cuda_cache)
                raise EngineOutOfMemory(OOM_HINT) from e
            raise
        self._set_session(session, dev, seconds)
        self._load_key = key
        return self.info

    def adopt(self, session, info=None):
        """Use an existing, initialised session (scripts / tests)."""
        self.release()
        dev = str(getattr(session, "device", "unknown"))
        self._set_session(session, dev, 0.0)
        if info:
            self._info.update(info)
        return self.info

    def _set_session(self, session, device, seconds):
        self._session = session
        self._major = session_major(session)
        self._info = {"version": _session_version(session), "major": self._major, "device": str(device),
                      "license": getattr(session, "license", None), "seconds": round(float(seconds), 2)}
        self._image_key = None
        self._shape = None
        self._buf = None
        self.active = None

    def release(self):
        """Drop the session and free GPU memory."""
        def work():
            s, self._session, self._buf = self._session, None, None
            ex = getattr(s, "executor", None)        # 1.x/2.x sessions own a small thread pool
            if ex is not None:
                try:
                    ex.shutdown(wait=False)
                except Exception:
                    pass
            del s
            gc.collect()
            _empty_cuda_cache()

        if self._session is not None or self._buf is not None:
            self._run(work)
        self._load_key = None
        self._info = {}
        self._major = None
        self._image_key = None
        self._shape = None
        self._valid_lo = self._valid_hi = None
        self.active = None

    def close(self):
        """release() and stop the worker thread (the engine cannot be used afterwards)."""
        try:
            self.release()
        finally:
            self._executor.shutdown(wait=False)

    def _require_session(self):
        if self._session is None:
            raise PromptEngineError("nnInteractive model is not loaded")
        return self._session

    # -- image -------------------------------------------------------------
    def _new_buffer(self, shape):
        if self._buffer_factory is not None:
            return self._buffer_factory(shape)
        import torch
        return torch.zeros(shape, dtype=torch.uint8)

    def set_image(self, img3d, key):
        """Give the model a 3D image (usually the crop). No-op if `key` is unchanged."""
        s = self._require_session()
        img3d = np.asarray(img3d)
        if img3d.ndim != 3:
            raise ValueError(f"nnInteractive needs a 3D image, got {img3d.ndim}D")
        if key is not None and key == self._image_key and self._shape == img3d.shape:
            return False
        lo, hi = [0, 0, 0], list(img3d.shape)
        if self._major is not None and self._major < 2:
            # 1.x crops the image to its nonzero box: points outside it cannot be used
            nonzero = img3d != 0
            for ax in range(3):
                nz = np.flatnonzero(np.any(nonzero, axis=tuple(a for a in range(3) if a != ax)))
                if not len(nz):
                    raise PromptEngineError("the image (crop) is entirely zero")
                lo[ax], hi[ax] = int(nz[0]), int(nz[-1]) + 1
            del nonzero
        elif not img3d.any():
            raise PromptEngineError("the image (crop) is entirely zero")

        def work():
            arr = img3d if (self._major or 2) >= 2 else img3d.astype(np.float32)
            s.set_image(arr[None])
            if self._buf is None or tuple(self._buf.shape) != tuple(img3d.shape):
                self._buf = None
                self._buf = self._new_buffer(img3d.shape)
            s.set_target_buffer(self._buf)       # after set_image: set_image drops the buffer

        self.active = None
        self._image_key = None
        self._run(work)
        self._image_key = key
        self._shape = tuple(img3d.shape)
        self._valid_lo, self._valid_hi = tuple(lo), tuple(hi)
        return True

    def valid_point(self, z, y, x):
        if self._valid_lo is None:
            return False
        return all(lo <= int(v) < hi for v, lo, hi in zip((z, y, x), self._valid_lo, self._valid_hi))

    def current_mask(self):
        """Copy of the model's current result (bool, image shape)."""
        return self._run(self._read_buffer)

    def _read_buffer(self):
        b = self._buf
        if b is None:
            raise PromptEngineError("no image set")
        arr = b.cpu().numpy() if hasattr(b, "cpu") else np.asarray(b)
        return arr > 0

    # -- prediction --------------------------------------------------------
    def predict_class(self, points, *, init_mask=None, scribble=None, neg_scribble=None, warm=False,
                      cancel=None, on_point=None, seed=None, class_id=None):
        """Segment one object; returns a bool mask (image shape).

        points: [(z, y, x, positive)] image coordinates, in order (one forward pass per
        point: nnInteractive centres a prediction on the newest interaction only).
        Unless warm: reset, np.random.seed(seed), initial mask (if any voxel), then
        scribbles, then points. warm adds the prompts to the current state.
        cancel (Event / callable) is checked before each interaction: raises
        boundary.Cancelled.
        """
        from sprout_core.boundary import Cancelled, _is_cancelled

        s = self._require_session()
        if self._shape is None:
            raise PromptEngineError("no image set")
        for name, m in (("init_mask", init_mask), ("scribble", scribble), ("neg_scribble", neg_scribble)):
            if m is not None and tuple(np.shape(m)) != self._shape:
                raise ValueError(f"{name} shape {np.shape(m)} does not match the image {self._shape}")
        points = [(int(p[0]), int(p[1]), int(p[2]), bool(p[3])) for p in points]
        init = None if init_mask is None or not np.any(init_mask) else np.asarray(init_mask).astype(np.uint8)
        pos_s = None if scribble is None or not np.any(scribble) else np.asarray(scribble).astype(np.uint8)
        neg_s = None if neg_scribble is None or not np.any(neg_scribble) else np.asarray(neg_scribble).astype(np.uint8)
        if warm and self.active is None:
            raise PromptEngineError("warm refinement needs an active interaction state")

        def check():
            if _is_cancelled(cancel):
                raise Cancelled()

        def work():
            if not warm:
                check()
                s.reset_interactions()
            if seed is not None:
                np.random.seed(int(seed) % (2 ** 32))
            others = bool(points) or pos_s is not None or neg_s is not None
            if not warm and init is not None:
                check()
                s.add_initial_seg_interaction(init, run_prediction=not others)
            if pos_s is not None:
                check()
                s.add_scribble_interaction(pos_s, include_interaction=True, run_prediction=True)
            if neg_s is not None:
                check()
                s.add_scribble_interaction(neg_s, include_interaction=False, run_prediction=True)
            for i, (z, y, x, positive) in enumerate(points):
                check()
                s.add_point_interaction((z, y, x), include_interaction=positive, run_prediction=True)
                if on_point is not None:
                    on_point(i + 1, len(points))
            return self._read_buffer()

        self.active = None
        self.state_token += 1
        try:
            mask = self._run(work)
        except Exception as e:
            if is_oom(e):
                self._run(_empty_cuda_cache)
                raise EngineOutOfMemory(OOM_HINT) from e
            raise
        self.active = (self._image_key, class_id)
        return mask

    def undo(self):
        """Single-level undo (nnInteractive 2.x with enable_undo=True); False otherwise."""
        if not self.supports_undo():
            return False
        return bool(self._run(self._session.undo))
