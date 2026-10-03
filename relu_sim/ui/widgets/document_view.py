"""A scrollable view of a document (the worked solution, a step explanation, ...).

The scroll bar floats over the right margin instead of taking layout space,
so showing or hiding it never changes the text width (no relayout loops).
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QScrollBar, QSizePolicy, QWidget

from ...content.blocks import Block
from ..document_painter import DocLayout, DocumentPainter, FigurePainter
from ..theme import Theme

SCROLLBAR = 10


class DocumentView(QWidget):
    def __init__(self, theme: Theme, kind: str = "paper", parent: QWidget | None = None,
                 figure_painter: FigurePainter | None = None, max_paper_width: int = 900):
        super().__init__(parent)
        self._kind = kind
        self._theme = theme
        self._figure_painter = figure_painter
        self._max_paper = max_paper_width
        self._blocks: tuple[Block, ...] = ()
        self._painter = DocumentPainter(theme, "paper" if kind == "paper" else "card", figure_painter)
        self._layout: DocLayout | None = None
        self._layout_width = -1.0
        self._zoom = 1.0
        self._scroll = QScrollBar(Qt.Orientation.Vertical, self)
        self._scroll.valueChanged.connect(self.update)
        self._scroll.hide()
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, kind == "paper")

    # ------------------------------------------------------------------ public

    def apply_theme(self, theme: Theme, figure_painter: FigurePainter | None = None) -> None:
        self._theme = theme
        if figure_painter is not None:
            self._figure_painter = figure_painter
        self._painter = DocumentPainter(theme, "paper" if self._kind == "paper" else "card", self._figure_painter)
        self._invalidate()

    def set_blocks(self, blocks: tuple[Block, ...], keep_scroll: bool = False) -> None:
        value = self._scroll.value()
        self._blocks = tuple(blocks)
        self._invalidate()
        self._ensure_layout()
        self._scroll.setValue(value if keep_scroll else 0)

    def blocks(self) -> tuple[Block, ...]:
        return self._blocks

    def set_zoom(self, zoom: float) -> None:
        self._zoom = max(0.7, min(zoom, 1.6))
        self._invalidate()

    @property
    def zoom(self) -> float:
        return self._zoom

    def content_height(self) -> float:
        self._ensure_layout()
        return self._total_height()

    # ------------------------------------------------------------------ geometry

    def _paper_rect_width(self) -> float:
        width = self.width()
        if self._kind == "paper":
            return min(width - 48.0, self._max_paper * self._zoom)
        return width - SCROLLBAR - 4.0

    def _padding(self) -> tuple[float, float]:
        """(horizontal, vertical) padding inside the paper."""
        if self._kind == "paper":
            paper = self._paper_rect_width()
            horizontal = 56.0 if paper > 620 else 28.0
            return horizontal, 44.0
        return 2.0, 2.0

    def _content_width(self) -> float:
        pad_x, _ = self._padding()
        return max(120.0, (self._paper_rect_width() - 2 * pad_x) / self._zoom)

    def _total_height(self) -> float:
        if self._layout is None:
            return 0.0
        _, pad_y = self._padding()
        margin = 24.0 if self._kind == "paper" else 0.0
        return self._layout.height * self._zoom + 2 * pad_y + 2 * margin

    def _invalidate(self) -> None:
        self._layout = None
        self._layout_width = -1.0
        self._ensure_layout()
        self.update()

    def _ensure_layout(self) -> None:
        width = self._content_width()
        if self._layout is None or abs(width - self._layout_width) > 0.5:
            self._layout = self._painter.layout(self._blocks, width)
            self._layout_width = width
            self._update_scrollbar()

    def _update_scrollbar(self) -> None:
        total = self._total_height()
        overflow = max(0, int(total - self.height()))
        self._scroll.setRange(0, overflow)
        self._scroll.setPageStep(max(1, self.height()))
        self._scroll.setSingleStep(48)
        self._scroll.setVisible(overflow > 0)
        self._scroll.setGeometry(self.width() - SCROLLBAR - 1, 2, SCROLLBAR, self.height() - 4)

    def sizeHint(self) -> QSize:
        return QSize(600, 400)

    # ------------------------------------------------------------------ events

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._ensure_layout()
        self._update_scrollbar()

    def wheelEvent(self, event) -> None:
        if self._scroll.maximum() <= 0:
            event.ignore()
            return
        delta = event.pixelDelta().y() or int(event.angleDelta().y() * 0.6)
        before = self._scroll.value()
        self._scroll.setValue(before - delta)
        if self._scroll.value() == before:
            event.ignore()
        else:
            event.accept()

    def keyPressEvent(self, event) -> None:
        keys = {Qt.Key.Key_PageDown: self.height() - 40, Qt.Key.Key_PageUp: -(self.height() - 40),
                Qt.Key.Key_Down: 48, Qt.Key.Key_Up: -48}
        if event.key() in keys:
            self._scroll.setValue(self._scroll.value() + keys[event.key()])
        elif event.key() == Qt.Key.Key_Home:
            self._scroll.setValue(0)
        elif event.key() == Qt.Key.Key_End:
            self._scroll.setValue(self._scroll.maximum())
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event) -> None:
        self._ensure_layout()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        t = self._theme
        offset = self._scroll.value()
        pad_x, pad_y = self._padding()
        if self._kind == "paper":
            painter.fillRect(self.rect(), QColor(t.window))
            paper_w = self._paper_rect_width()
            paper = QRectF((self.width() - paper_w) / 2, 24 - offset, paper_w, self._total_height() - 48)
            for i, alpha in enumerate((0.05, 0.035, 0.02) if not t.dark else (0.25, 0.15, 0.08)):
                shadow = QColor(0, 0, 0)
                shadow.setAlphaF(alpha)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(shadow)
                painter.drawRoundedRect(paper.adjusted(-i, 1 + i, i, 2 + 2 * i), 8 + i, 8 + i)
            painter.setBrush(QColor(t.surface))
            painter.setPen(QPen(QColor(t.border), 1))
            painter.drawRoundedRect(paper, 8, 8)
            x, y = paper.x() + pad_x, paper.y() + pad_y
        else:
            x, y = pad_x, pad_y - offset
        if self._layout is None:
            return
        painter.save()
        painter.translate(x, y)
        painter.scale(self._zoom, self._zoom)
        clip = QRectF(0, (-y + event.rect().top()) / self._zoom, self._layout.width,
                      event.rect().height() / self._zoom)
        DocumentPainter.paint(painter, self._layout, 0.0, 0.0, clip)
        painter.restore()
