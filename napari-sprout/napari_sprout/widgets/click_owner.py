"""Which dock answers a Ctrl+click in the viewer.

Edit and Label Mapping both act on a Ctrl+click on the active Labels layer. Each dock calls
``register(self)`` once and its click handler returns early unless ``handles_click(self)``:

- a dock that was shown and is now hidden, closed or a tab behind another tab never handles it;
- of the open docks, only the one used last handles it (a mouse press or keyboard focus anywhere
  inside it; opening a dock counts as using it);
- with one dock open, that one handles it. A dock that was never shown (built in code) takes
  part as an open one.
"""
import atexit
import weakref

from qtpy.QtCore import QEvent, QObject
from qtpy.QtWidgets import QApplication, QWidget

_docks = []                  # weakrefs to the registered docks, least recently used first
_shown = weakref.WeakSet()   # docks that got a Show event (from then on visibility decides)
_watch = None


def _live():
    """The registered docks that still exist (drops the deleted ones)."""
    out = []
    for r in list(_docks):
        d = r()
        try:
            if d is not None:
                d.isVisible()          # RuntimeError once the C++ widget is deleted
                out.append(d)
                continue
        except RuntimeError:
            pass
        _docks.remove(r)
    return out


def _use(dock):
    for r in [r for r in _docks if r() is dock]:
        _docks.remove(r)
    _docks.append(weakref.ref(dock))


class _Watch(QObject):
    def eventFilter(self, obj, ev):
        t = ev.type()
        if t in (QEvent.MouseButtonPress, QEvent.FocusIn, QEvent.Show) and isinstance(obj, QWidget):
            try:
                for d in _live():
                    if obj is d:
                        if t == QEvent.Show:
                            _shown.add(d)
                        else:
                            _use(d)
                        break
                    if t != QEvent.Show and d.isAncestorOf(obj):
                        _use(d)
                        break
            except RuntimeError:     # a widget deleted meanwhile
                pass
        return False


def _unwatch():
    """Remove the app filter before Python shuts down (a Python filter called while the
    interpreter is torn down crashes the process at exit)."""
    global _watch
    app = QApplication.instance()
    if _watch is not None and app is not None:
        try:
            app.removeEventFilter(_watch)
        except RuntimeError:
            pass
    _watch = None


def register(dock):
    """Take part in Ctrl+click ownership; the newest dock is the one used last."""
    global _watch
    app = QApplication.instance()
    if _watch is None and app is not None:
        _watch = _Watch()
        app.installEventFilter(_watch)
        app.aboutToQuit.connect(_unwatch)
        atexit.register(_unwatch)
    _use(dock)


def _on_screen(dock):
    """Shown and not covered: a dock tab behind another tab stays isVisible(), with nothing of
    it on screen."""
    return dock.isVisible() and not dock.visibleRegion().isEmpty()


def handles_click(dock):
    """True if `dock` is the one that should answer a Ctrl+click now."""
    open_docks = [d for d in _live() if d not in _shown or _on_screen(d)]
    if not any(d is dock for d in open_docks):
        return not any(r() is dock for r in _docks)   # unregistered: as before
    return open_docks[-1] is dock
