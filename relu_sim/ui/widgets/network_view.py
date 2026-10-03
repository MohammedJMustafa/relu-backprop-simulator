"""The interactive network diagram widget."""

from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

from ...content.diagram import DiagramFrame
from ..network_painter import VH, VW, NetworkPainter
from ..theme import Theme


class NetworkView(QWidget):
    def __init__(self, theme: Theme, mode: str = "simulation", parent: QWidget | None = None):
        super().__init__(parent)
        self._mode = mode
        self._painter = NetworkPainter(theme)
        self._frame: DiagramFrame | None = None
        self._progress = 1.0
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(360 if mode == "mini" else 560, 200 if mode == "mini" else 320)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

    def apply_theme(self, theme: Theme) -> None:
        self._painter = NetworkPainter(theme)
        self.update()

    def set_frame(self, frame: DiagramFrame | None, progress: float = 1.0) -> None:
        self._frame = frame
        self._progress = progress
        self.update()

    def set_progress(self, progress: float) -> None:
        if progress != self._progress:
            self._progress = progress
            self.update()

    def frame(self) -> DiagramFrame | None:
        return self._frame

    def sizeHint(self) -> QSize:
        return QSize(900, int(900 * VH / VW))

    def hasHeightForWidth(self) -> bool:
        return False

    def paintEvent(self, event) -> None:
        if self._frame is None:
            return
        painter = QPainter(self)
        margin = 6.0
        self._painter.paint(painter, QRectF(self.rect()).adjusted(margin, margin, -margin, -margin), self._frame,
                            self._progress, mode=self._mode)

    def render_image(self, width: int = 2000) -> QImage:
        """A high-resolution image of the current frame (PNG export)."""
        height = int(width * VH / VW)
        image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        self._painter.paint(painter, QRectF(0, 0, width, height), self._frame, self._progress, mode=self._mode)
        painter.end()
        return image
