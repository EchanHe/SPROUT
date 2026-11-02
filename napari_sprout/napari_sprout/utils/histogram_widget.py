"""Histogram with a draggable [lower, upper] range, for napari-sprout."""
import numpy as np
from qtpy.QtCore import QPointF, QRectF, Qt, Signal
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .status import style

_LOWER_COLOR = QColor(80, 160, 255)
_UPPER_COLOR = QColor(255, 150, 60)


class _HistCanvas(QWidget):
    """Paints the histogram (log y) and the two range lines; lines are draggable."""

    PAD = 6
    GRAB_PX = 8
    HEIGHT = 50          # px; the range handles span the full height

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.setFixedHeight(self.HEIGHT)
        self.setMouseTracking(True)
        self._drag = None  # 'lower' | 'upper' | None

    # value <-> pixel
    def _x_of(self, v):
        lo, hi = self.owner._xmin, self.owner._xmax
        w = self.width() - 2 * self.PAD
        return self.PAD + (v - lo) / (hi - lo) * w if hi > lo else self.PAD

    def _v_of(self, x):
        lo, hi = self.owner._xmin, self.owner._xmax
        w = self.width() - 2 * self.PAD
        return lo + (x - self.PAD) / max(w, 1) * (hi - lo)

    def paintEvent(self, _):
        o = self.owner
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        p.fillRect(rect, self.palette().base())

        if o._message or o._counts is None:
            p.setPen(self.palette().text().color())
            p.drawText(rect, Qt.AlignCenter, o._message or "")
            return

        # bars, log scale: height ~ log10(count + 1)
        counts, edges = o._counts, o._edges
        logc = np.log10(counts.astype(float) + 1)
        top = logc.max() or 1.0
        h = self.height() - 2 * self.PAD
        base_y = self.height() - self.PAD
        path = QPainterPath(QPointF(self._x_of(edges[0]), base_y))
        for c, e0, e1 in zip(logc, edges[:-1], edges[1:]):
            y = base_y - c / top * h
            path.lineTo(self._x_of(e0), y)
            path.lineTo(self._x_of(e1), y)
        path.lineTo(self._x_of(edges[-1]), base_y)
        path.closeSubpath()
        p.fillPath(path, QColor(150, 150, 150))

        # selected range
        xl, xu = self._x_of(o._lower), self._x_of(o._upper)
        p.fillRect(QRectF(xl, self.PAD, max(xu - xl, 1), h), QColor(255, 80, 80, 45))
        for x, col in ((xl, _LOWER_COLOR), (xu, _UPPER_COLOR)):
            p.setPen(QPen(col, 2))
            p.drawLine(QPointF(x, self.PAD), QPointF(x, base_y))

    def mousePressEvent(self, ev):
        if not self.owner._ready():
            return
        x = ev.pos().x()
        dl = abs(x - self._x_of(self.owner._lower))
        du = abs(x - self._x_of(self.owner._upper))
        self._drag = "lower" if dl <= du else "upper"
        self._move_to(x)

    def mouseMoveEvent(self, ev):
        o = self.owner
        if self._drag:
            self._move_to(ev.pos().x())
        elif o._ready():
            x = ev.pos().x()
            near = min(abs(x - self._x_of(o._lower)), abs(x - self._x_of(o._upper))) <= self.GRAB_PX
            self.setCursor(Qt.SizeHorCursor if near else Qt.ArrowCursor)

    def mouseReleaseEvent(self, _):
        if self._drag:
            self._drag = None
            self.owner._emit_committed()

    def _move_to(self, x):
        o = self.owner
        v = float(np.clip(self._v_of(x), o._xmin, o._xmax))
        if o._integer:
            v = round(v)
        if self._drag == "lower":
            o._set(v, max(v, o._upper))
        else:
            o._set(min(v, o._lower), v)


class HistogramRangeWidget(QWidget):
    """Histogram (log y) with a draggable [lower, upper] range and spin boxes.

    Signals
    -------
    range_changed(lower, upper)    continuously while dragging / editing
    range_committed(lower, upper)  on mouse release / editing finished
    """

    range_changed = Signal(float, float)
    range_committed = Signal(float, float)

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self._counts = self._edges = None
        self._xmin, self._xmax = 0.0, 1.0
        self._lower, self._upper = 0.0, 1.0
        self._integer = True
        self._message = ""

        self.title = QLabel(title)
        self.canvas = _HistCanvas(self)
        self.lower_spin = QDoubleSpinBox()
        self.upper_spin = QDoubleSpinBox()
        for sp in (self.lower_spin, self.upper_spin):
            sp.setKeyboardTracking(False)
            # keyboard tracking off: valueChanged fires on Enter / focus-out / arrows,
            # each of which is a discrete edit, so it also counts as "committed"
            sp.valueChanged.connect(self._on_spin)

        # The axis comes from a subsample, so the true min/max can lie beyond it.
        # A handle parked at an end therefore means "no limit on that side".
        self.lower_note = QLabel("")
        self.upper_note = QLabel("")
        for n in (self.lower_note, self.upper_note):
            n.setStyleSheet(style("info"))

        row = QHBoxLayout()
        lbl_l = QLabel("lower")
        lbl_l.setStyleSheet(f"color: {_LOWER_COLOR.name()}")
        lbl_u = QLabel("upper")
        lbl_u.setStyleSheet(f"color: {_UPPER_COLOR.name()}")
        row.addWidget(lbl_l)
        row.addWidget(self.lower_spin)
        row.addWidget(self.lower_note)
        row.addStretch()
        row.addWidget(lbl_u)
        row.addWidget(self.upper_spin)
        row.addWidget(self.upper_note)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        if title:
            lay.addWidget(self.title)
        lay.addWidget(self.canvas)
        lay.addLayout(row)
        self.show_message("")

    # ---- public API -------------------------------------------------------
    def show_message(self, text):
        """Replace the histogram by a message (e.g. 'Computing...'); disables input."""
        self._message = text
        self._counts = None if text else self._counts
        self._set_controls_enabled(False)
        self.canvas.update()

    def set_data(self, counts, edges, integer=True, lower=None, upper=None):
        """Show a histogram; the selectable range is [edges[0], edges[-1]]."""
        self._counts = np.asarray(counts)
        self._edges = np.asarray(edges, float)
        self._integer = bool(integer)
        self._xmin, self._xmax = float(self._edges[0]), float(self._edges[-1])
        if self._integer:
            # one-bin-per-value edges sit at k +- 0.5; the selectable values are integers inside
            self._xmin, self._xmax = float(np.ceil(self._xmin)) + 0.0, float(np.floor(self._xmax)) + 0.0
        self._message = ""
        decimals = 0 if self._integer else 4
        for sp in (self.lower_spin, self.upper_spin):
            sp.blockSignals(True)
            sp.setDecimals(decimals)
            sp.setRange(self._xmin, self._xmax)
            sp.blockSignals(False)
        self._set_controls_enabled(True)
        lo = self._xmin if lower is None else lower
        hi = self._xmax if upper is None else upper
        self.set_range(lo, hi, emit=False)

    def set_range(self, lower, upper, emit=True):
        lower = float(np.clip(lower, self._xmin, self._xmax))
        upper = float(np.clip(upper, self._xmin, self._xmax))
        self._set(min(lower, upper), max(lower, upper), emit=emit)
        if emit:
            self._emit_committed()

    def get_range(self):
        """Raw handle positions."""
        return self._lower, self._upper

    def get_bounds(self):
        """(lower, upper) for thresholding; None where a handle sits at the axis end."""
        lo = None if self._lower <= self._xmin else self._lower
        hi = None if self._upper >= self._xmax else self._upper
        return lo, hi

    def is_full_range(self):
        return self._lower <= self._xmin and self._upper >= self._xmax

    # ---- internals --------------------------------------------------------
    def _ready(self):
        return self._counts is not None and not self._message

    def _set_controls_enabled(self, on):
        self.lower_spin.setEnabled(on)
        self.upper_spin.setEnabled(on)

    def _set(self, lower, upper, emit=True):
        self._lower, self._upper = lower, upper
        for sp, v in ((self.lower_spin, lower), (self.upper_spin, upper)):
            sp.blockSignals(True)
            sp.setValue(v)
            sp.blockSignals(False)
        self.lower_note.setText("(no limit)" if lower <= self._xmin else "")
        self.upper_note.setText("(no limit)" if upper >= self._xmax else "")
        self.canvas.update()
        if emit:
            self.range_changed.emit(lower, upper)

    def _on_spin(self, _):
        lo, hi = self.lower_spin.value(), self.upper_spin.value()
        if self.sender() is self.lower_spin and lo > hi:
            hi = lo
        elif self.sender() is self.upper_spin and hi < lo:
            lo = hi
        self._set(lo, hi)
        self._emit_committed()

    def _emit_committed(self):
        if self._ready():
            self.range_committed.emit(self._lower, self._upper)
