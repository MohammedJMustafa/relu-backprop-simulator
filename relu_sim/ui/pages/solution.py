"""The worked solution (sections 1-6 of the PDF), recomputed live."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Signal
from PySide6.QtGui import QGuiApplication, QPainter
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from ...content.blocks import plain_text
from ..icons import icon
from ..network_painter import NetworkPainter
from ..state import AppState
from ..theme import Theme
from ..widgets.common import IconButton, PageHeader
from ..widgets.document_view import DocumentView


def figure_painter_for(theme: Theme):
    painter_obj = NetworkPainter(theme)

    def paint(painter: QPainter, rect: QRectF, kind: str, payload) -> None:
        if kind == "network" and payload is not None:
            painter_obj.paint(painter, rect, payload, 1.0, mode="figure")

    return paint


class SolutionPage(QWidget):
    exportPdfRequested = Signal()
    copied = Signal(str)

    def __init__(self, state: AppState, parent: QWidget | None = None):
        super().__init__(parent)
        self.state = state
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 0)
        layout.setSpacing(10)

        self.header = PageHeader("Worked solution",
                                 "Sections 1–6 of the homework, typeset and recomputed live for the current values.")
        self.zoom_out = IconButton("zoom_out", "Smaller text (Ctrl+−)")
        self.zoom_in = IconButton("zoom_in", "Larger text (Ctrl++)")
        self.copy = QPushButton(" Copy as text")
        self.export = QPushButton(" Export PDF")
        self.export.setObjectName("Primary")
        for widget in (self.zoom_out, self.zoom_in, self.copy, self.export):
            self.header.add_action(widget)
        layout.addWidget(self.header)

        self.document = DocumentView(state.theme, "paper", figure_painter=figure_painter_for(state.theme))
        layout.addWidget(self.document, 1)

        self.zoom_in.clicked.connect(lambda: self.document.set_zoom(self.document.zoom + 0.1))
        self.zoom_out.clicked.connect(lambda: self.document.set_zoom(self.document.zoom - 0.1))
        self.copy.clicked.connect(self.copy_text)
        self.export.clicked.connect(self.exportPdfRequested)
        state.configChanged.connect(self.reload)
        self.apply_theme(state.theme)
        self.reload()

    def reload(self) -> None:
        self.document.set_blocks(self.state.solution, keep_scroll=True)

    def copy_text(self) -> None:
        QGuiApplication.clipboard().setText(plain_text(self.state.solution))
        self.copied.emit("The worked solution was copied to the clipboard as plain text.")

    def apply_theme(self, theme: Theme) -> None:
        self.document.apply_theme(theme, figure_painter_for(theme))
        for button in (self.zoom_in, self.zoom_out):
            button.apply_theme(theme)
        self.copy.setIcon(icon("copy", theme.text2, 15))
        self.export.setIcon(icon("pdf", "#FFFFFF", 15))
