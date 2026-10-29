"""One message style for the whole plugin: status lines and inline notes.

Four levels, one colour each (readable on napari's dark and light themes):

    info   grey    in progress or neutral       "Generating seeds…"
    ok     green   done, with a short summary    "✓ Seeds generated: 3 layers (2.5 s)"
    warn   orange  needs attention, can run      "z: all — set the z range …"
    error  red     cannot run, and why           "The boundary mask 'x' cannot be used: …"

Dialogs are only for decisions (file pickers, overwrite / lose-data confirmations); napari
notifications only for "run finished" and "run failed", so they are seen from another tab.
Everything else is a StatusLine (a line of a tab) or an inline note under a control.
"""
from qtpy.QtWidgets import QLabel

INFO = "gray"
OK = "#3fa34d"
WARN = "#cc7a00"
ERROR = "#d9534f"
COLORS = {"info": INFO, "ok": OK, "warn": WARN, "error": ERROR}
CHECK = "✓ "


def style(level):
    """The label style sheet of a level ('info' / 'ok' / 'warn' / 'error')."""
    return f"color: {COLORS[level]}"


def level_of(widget):
    """The level a label's style sheet shows (None if it has no level colour)."""
    css = widget.styleSheet().replace(" ", "").lower()
    for level, colour in COLORS.items():
        if f"color:{colour.lower()}" in css:
            return level
    return None


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
