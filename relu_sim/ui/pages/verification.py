"""Verification: the PDF's printed answers and a numerical gradient check."""

from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from ...content.report import verification_document
from ..state import AppState
from ..theme import Theme
from ..widgets.common import PageHeader
from ..widgets.document_view import DocumentView


class VerificationPage(QWidget):
    def __init__(self, state: AppState, parent: QWidget | None = None):
        super().__init__(parent)
        self.state = state
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 0)
        layout.setSpacing(10)
        layout.addWidget(PageHeader("Verification",
                                    "Proof that the simulator reproduces the homework, and that backpropagation "
                                    "agrees with numerical differentiation."))
        self.document = DocumentView(state.theme, "paper")
        layout.addWidget(self.document, 1)
        state.configChanged.connect(self.reload)
        self.reload()

    def reload(self) -> None:
        self.document.set_blocks(verification_document(self.state.answer_checks, self.state.gradient_check,
                                                       self.state.config), keep_scroll=True)

    def apply_theme(self, theme: Theme) -> None:
        self.document.apply_theme(theme)
