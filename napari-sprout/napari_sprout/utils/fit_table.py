"""A table as high as its rows (no empty filler), capped; it scrolls inside beyond that."""
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QSizePolicy, QTableWidget


class FitTable(QTableWidget):
    """Header + rows high, up to MAX_ROWS rows; an inner scrollbar beyond that.

    The height follows row inserts / removals by itself (call fit() after anything else
    that changes row heights). Full width: the columns stretch over it.
    """

    MAX_ROWS = 10

    def __init__(self, *args, max_rows=None, auto_fit=True, **kwargs):
        super().__init__(*args, **kwargs)
        if max_rows is not None:
            self.MAX_ROWS = max_rows
        if auto_fit:
            m = self.model()
            for sig in (m.rowsInserted, m.rowsRemoved, m.modelReset, m.layoutChanged):
                sig.connect(self._fit_later)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.fit()

    def _fit_later(self, *_):
        self.fit()
        QTimer.singleShot(0, self._fit_if_alive)      # row heights settle after the insert

    def _fit_if_alive(self):
        try:
            self.fit()
        except RuntimeError:                          # deleted meanwhile
            pass

    def fit(self):
        n = min(self.rowCount(), self.MAX_ROWS)
        h = 2 * self.frameWidth() + self.horizontalHeader().sizeHint().height()
        h += sum(self.rowHeight(i) for i in range(n))
        if self.horizontalHeader().length() > self.viewport().width():      # columns wider than the table
            h += self.horizontalScrollBar().sizeHint().height()
        if self.minimumHeight() != h or self.maximumHeight() != h:
            self.setFixedHeight(h)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit()                      # a narrower table may need its horizontal scrollbar
