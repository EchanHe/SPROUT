"""One message style for the whole plugin: status lines and inline notes.

Four levels, one colour each (readable on napari's dark and light themes; the info grey
follows the theme, see follow_theme):

    info   grey    in progress or neutral       "Generating seeds…"
    ok     green   done, with a short summary    "✓ Seeds generated: 3 layers (2.5 s)"
    warn   orange  needs attention, can run      "z: all — set the z range …"
                   or stopped by the user        "Stopped after 3 of 8 seeds; 3 files …"
    error  red     cannot run / failed, and why  "The boundary mask 'x' cannot be used: …"
                                                 "Growth failed: not enough memory …"

Texts: one ellipsis character "…", "Select …" (not "Please select"), failures as
"<What> failed: <reason>". Long runs (Seeds, Grow, Boundary Compute) go through
utils/jobs.RunControl, which writes these lines for them.

Dialogs are only for decisions (file pickers, overwrite / lose-data confirmations); napari
notifications only for "run finished" and "run failed", so they are seen from another tab.
Everything else is a StatusLine (a line of a tab) or an inline note under a control.
"""
import re
import weakref

from qtpy.QtCore import QTimer
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QLabel, QWidget

# the info grey per theme: >= 4.5:1 on napari's panel background ("gray" was 3.7:1 on dark)
GREY_DARK = "#a9a9a9"    # 6.2:1 on the dark theme (#262930)
GREY_LIGHT = "#666666"   # 4.9:1 on the light theme (#efebe9)
INFO = GREY_DARK         # the grey of the current theme (follow_theme sets it)
OK = "#3fa34d"
WARN = "#cc7a00"
ERROR = "#d9534f"
COLORS = {"info": INFO, "ok": OK, "warn": WARN, "error": ERROR}
CHECK = "✓ "

# run / stop buttons of the tabs; only while enabled, so a disabled one looks disabled
RUN_BUTTON = "QPushButton:enabled { font-weight: bold; background-color: #45a049; }"
STOP_BUTTON = "QPushButton:enabled { font-weight: bold; background-color: #d9534f; color: white; }"

# the target layer header of the Edit and Label Mapping docks (white on blue)
LAYER_HEADER = ("QLabel { font-weight: bold; color: white; background-color: #007ACC; padding: 4px; "
                "border-radius: 4px; }")
NO_LAYER = "Active Label Layer: (none) - select a Labels layer in the layer list"


def tab_hidden(widget):
    """True while `widget` is in a shown window but not visible itself (another tab is
    shown): its work can wait for its showEvent. Never True before the window is shown."""
    return widget.window().isVisible() and not widget.isVisible()


def style(level):
    """The label style sheet of a level ('info' / 'ok' / 'warn' / 'error')."""
    return f"color: {COLORS[level]}"


def level_of(widget):
    """The level a label's style sheet shows (None if it has no level colour)."""
    css = widget.styleSheet().replace(" ", "").lower()
    for level, colour in COLORS.items():
        if f"color:{colour.lower()}" in css:
            return level
    if _GREY_RE.search(widget.styleSheet()):      # the info grey of the other theme
        return "info"
    return None


# ---------------------------------------------------------------- theme
_GREY_RE = re.compile(r"(?<![\w-])color:\s*(darkgray|#a9a9a9|#666666)\b", re.I)
_LINES = weakref.WeakSet()       # every StatusLine: recoloured on a theme change
_ROOTS = weakref.WeakSet()       # panels whose grey hints (plain QLabels) are recoloured
_VIEWERS = weakref.WeakSet()     # viewers whose theme is followed


def grey_for(theme):
    """The info grey readable on napari theme `theme` (a name; 'system' included)."""
    try:
        from napari.utils.theme import get_theme
        bg = QColor(get_theme(theme).background.as_hex())
    except Exception:            # an unknown theme: napari's default is dark
        return GREY_DARK
    return GREY_LIGHT if bg.lightnessF() > 0.5 else GREY_DARK


def follow_theme(viewer, root):
    """Use the info grey of the viewer's theme, now and after every theme change.

    Call at the start of a panel's __init__ (labels made after it get the grey of the
    theme); on a light theme the grey hints of `root`'s top SPROUT panel that were styled
    from a module constant (the dark grey) are recoloured once it is built (next event-loop
    turn); on every theme change all of them.
    """
    _ROOTS.add(root)
    try:
        if viewer not in _VIEWERS:
            viewer.events.theme.connect(lambda e: set_theme(e.value))
            _VIEWERS.add(viewer)
    except (AttributeError, TypeError):      # not a napari viewer (a test double)
        return
    set_theme(viewer.theme)
    if INFO == GREY_DARK:                    # the dark theme: nothing to recolour
        return
    # a timer of the panel (not QTimer.singleShot with a function): goes with the panel
    ref = weakref.ref(root)
    timer = QTimer(root)
    timer.setSingleShot(True)
    timer.timeout.connect(lambda: ref() is not None and _recolour_root(ref()))
    timer.timeout.connect(timer.deleteLater)
    timer.start(0)


def set_theme(theme):
    """Switch the info grey to the one of napari theme `theme`; recolour the shown greys."""
    global INFO
    grey = grey_for(theme)
    if grey == INFO:
        return
    INFO = COLORS["info"] = grey
    for line in list(_LINES):
        try:
            line.setStyleSheet(style(line.level or "info"))
        except RuntimeError:     # deleted with its panel
            pass
    for root in list(_ROOTS):
        _recolour_root(root)


def _top_panel(w):
    """The outermost SPROUT widget holding `w` (the whole panel, ROI tools included)."""
    top = w
    p = w.parentWidget()
    while p is not None:
        if type(p).__module__.startswith("napari_sprout"):
            top = p
        p = p.parentWidget()
    return top


def _recolour_root(root):
    try:
        top = _top_panel(root)
        widgets = [top] + top.findChildren(QWidget)
    except RuntimeError:         # deleted with its panel
        return
    for w in widgets:
        css = w.styleSheet()
        if css and _GREY_RE.search(css):
            # only another grey is replaced ("darkgray" is the dark grey: left as it is)
            new = _GREY_RE.sub(lambda m: m.group(0) if QColor(m.group(1)) == QColor(INFO)
                               else f"color: {INFO}", css)
            if new != css:
                w.setStyleSheet(new)


class StatusLine(QLabel):
    """A word-wrapped label showing one message at one level.

    set(text, level): 'ok' texts start with '✓ '. With hide_empty=True the line takes no
    space while it has no text (inline notes under a control).
    """

    def __init__(self, text="", level="info", hide_empty=False, parent=None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.hide_empty = hide_empty
        self.level = None
        _LINES.add(self)
        self.set(text, level)

    def set(self, text, level="info"):
        text = str(text or "")
        if level == "ok" and text and not text.startswith(CHECK):
            text = CHECK + text
        self.level = level
        self.setStyleSheet(style(level))
        self.setText(text)
        if self.hide_empty:
            self.setVisible(bool(text))

    def clear(self):
        self.set("", "info")


def set_note(label, text, level="info"):
    """Show `text` in a plain QLabel at `level` (hidden while empty): an inline note."""
    if isinstance(label, StatusLine):
        label.set(text, level)
        return
    label.setText(text)
    label.setStyleSheet(style(level))
    label.setVisible(bool(text))
