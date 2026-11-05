"""
Shared setup for the dock checks (Edit / Label Mapping / Batch Builder / Data Info):
choose the source tree (--root, default: this repo), headless Qt, dialog and
notification recorders, data hashing.

Used by tests/check_docks.py and benchmarks/char_docks.py.
"""
import hashlib
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.path.isdir(r"C:/Windows/Fonts"):
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:/Windows/Fonts")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]


def pick_root(argv):
    """--root <tree> runs the checks against another source tree (e.g. the base commit)."""
    root = REPO
    if "--root" in argv:
        root = Path(argv[argv.index("--root") + 1]).resolve()
    sys.path[:0] = [str(root), str(root / "napari_sprout"), str(HERE)]
    try:
        import _headless_gl  # noqa: F401  (napari 0.7 offscreen GL shim, if present)
    except ImportError:
        pass
    return root


def sha(a):
    import numpy as np
    a = np.ascontiguousarray(np.asarray(a))
    return f"{a.dtype}{a.shape}:" + hashlib.sha1(a.tobytes()).hexdigest()[:16]


def pump(n=20):
    from qtpy.QtWidgets import QApplication
    for _ in range(n):
        QApplication.processEvents()
        time.sleep(0.002)


def wait(cond, timeout=300, what=""):
    from qtpy.QtWidgets import QApplication
    t = time.time()
    while not cond():
        QApplication.processEvents()
        time.sleep(0.003)
        if time.time() - t > timeout:
            raise TimeoutError(what)


class Dialogs:
    """Replace QMessageBox / QFileDialog statics: record calls, answer from a queue."""

    def __init__(self):
        self.calls = []
        self.answer_question = True
        self.save_path = ""
        self.open_path = ""
        self.dir_path = ""

    def install(self):
        from qtpy.QtWidgets import QDialog, QFileDialog, QMessageBox
        rec = self

        def mk(kind):
            def f(parent, title, text, *a, **k):
                rec.calls.append((kind, str(title), str(text)))
                return QMessageBox.Ok
            return staticmethod(f)

        for kind in ("warning", "information", "critical"):
            setattr(QMessageBox, kind, mk(kind))

        def question(parent, title, text, *a, **k):
            rec.calls.append(("question", str(title), str(text)))
            return QMessageBox.Yes if rec.answer_question else QMessageBox.No
        QMessageBox.question = staticmethod(question)
        QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (rec.save_path, ""))
        QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (rec.open_path, ""))
        QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: rec.dir_path)

        def modal(dialog, *a, **k):
            rec.calls.append(("dialog", dialog.windowTitle(), ""))
            return QDialog.Rejected
        QDialog.exec_ = QDialog.exec = modal
        return self

    def take(self):
        out, self.calls = self.calls, []
        return out


class Notes:
    """Record napari show_info / show_error called through a module's own names."""

    def __init__(self):
        self.items = []

    def patch(self, *modules):
        for mod in modules:
            for kind in ("show_info", "show_error", "show_warning"):
                if hasattr(mod, kind):
                    setattr(mod, kind, self._mk(kind))
        return self

    def _mk(self, kind):
        def f(msg, *a, **k):
            self.items.append((kind, str(msg)))
        return f

    def take(self):
        out, self.items = self.items, []
        return out


class FakeClick:
    """A mouse event as napari hands it to mouse_drag_callbacks (enough for on_click)."""

    def __init__(self, pos, ctrl=True, ndim=3):
        self.position = tuple(float(p) for p in pos)
        self.modifiers = ("Control",) if ctrl else ()
        self.view_direction = None
        self.dims_displayed = list(range(ndim))[-2:]
        self.type = "mouse_press"


def synthetic_3d():
    """Label map with multi-component labels, a cavity, a satellite and thin parts."""
    import numpy as np
    a = np.zeros((24, 40, 48), np.uint16)
    a[2:8, 2:12, 2:12] = 1          # label 1: two components (big + small)
    a[14:17, 30:34, 36:40] = 1
    a[4:16, 16:30, 16:30] = 2       # label 2: block with a cavity + satellite
    a[9:11, 21:24, 21:24] = 0
    a[20:22, 2:4, 40:42] = 2
    a[2:10, 32:38, 2:10] = 5        # label 5: one block
    a[18:21, 10:14, 10:14] = 9      # label 9: three components of different sizes
    a[18:22, 20:26, 2:6] = 9
    a[19:20, 36:37, 20:21] = 9
    a[12, 2:12, 30:46] = 12         # label 12: a thin plane
    return a


def synthetic_2d():
    import numpy as np
    a = np.zeros((60, 80), np.uint8)
    a[5:20, 5:20] = 3
    a[10:13, 10:13] = 0             # hole
    a[40:44, 60:70] = 3             # second component
    a[25:50, 25:45] = 7
    a[2:5, 70:78] = 7
    a[30:33, 5:8] = 4
    return a
