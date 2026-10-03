"""Small building blocks shared by the pages: cards, segmented controls, icon
buttons, statistic tiles and a label that typesets a formula."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (QButtonGroup, QFrame, QHBoxLayout, QLabel, QSizePolicy, QToolButton, QVBoxLayout,
                               QWidget)

from ...content import mathexpr as mx
from ..icons import icon
from ..math_painter import MathRenderer, math_style
from ..theme import Theme


class Card(QFrame):
    """A rounded surface with an optional title row."""

    def __init__(self, title: str | None = None, parent: QWidget | None = None, padding: int = 16,
                 spacing: int = 10):
        super().__init__(parent)
        self.setObjectName("Card")
        self.outer = QVBoxLayout(self)
        self.outer.setContentsMargins(padding, padding - 2, padding, padding)
        self.outer.setSpacing(spacing)
        self.header = None
        if title is not None:
            self.header = QHBoxLayout()
            self.header.setSpacing(8)
            self.title_label = QLabel(title)
            self.title_label.setObjectName("CardTitle")
            self.header.addWidget(self.title_label)
            self.header.addStretch(1)
            self.outer.addLayout(self.header)

    def add_header_widget(self, widget: QWidget) -> None:
        if self.header is not None:
            self.header.addWidget(widget)


def caption(text: str, parent: QWidget | None = None) -> QLabel:
    label = QLabel(text.upper(), parent)
    label.setObjectName("Caption")
    return label


def muted(text: str, parent: QWidget | None = None, wrap: bool = False) -> QLabel:
    label = QLabel(text, parent)
    label.setObjectName("Muted")
    label.setWordWrap(wrap)
    return label


class Segmented(QWidget):
    """Mutually exclusive options shown as a compact pill control."""

    changed = Signal(str)

    def __init__(self, options: list[tuple[str, str]], parent: QWidget | None = None, tooltips: dict | None = None):
        super().__init__(parent)
        self.setObjectName("Segmented")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(2)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QToolButton] = {}
        for key, label in options:
            button = QToolButton(self)
            button.setObjectName("Segment")
            button.setText(label)
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            if tooltips and key in tooltips:
                button.setToolTip(tooltips[key])
            self._group.addButton(button)
            self._buttons[key] = button
            layout.addWidget(button)
            button.clicked.connect(lambda _=False, k=key: self.changed.emit(k))
        if options:
            self._buttons[options[0][0]].setChecked(True)

    def value(self) -> str:
        for key, button in self._buttons.items():
            if button.isChecked():
                return key
        return ""

    def set_value(self, key: str) -> None:
        if key in self._buttons:
            self._buttons[key].setChecked(True)


class IconButton(QToolButton):
    def __init__(self, glyph: str, tooltip: str, parent: QWidget | None = None, size: int = 18):
        super().__init__(parent)
        self.setObjectName("IconButton")
        self._glyph = glyph
        self._size = size
        self.setToolTip(tooltip)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setIconSize(QSize(size, size))
        self.setAutoRaise(True)

    def set_glyph(self, glyph: str, theme: Theme) -> None:
        self._glyph = glyph
        self.apply_theme(theme)

    def apply_theme(self, theme: Theme, color: str | None = None) -> None:
        self.setIcon(icon(self._glyph, color or theme.text2, self._size, disabled=theme.border2))


class StatTile(QFrame):
    """A small statistic: caption, big value and a note."""

    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(2)
        self.caption = caption(title)
        self.value = QLabel("—")
        self.value.setObjectName("StatValue")
        self.note = QLabel("")
        self.note.setObjectName("StatNote")
        self.note.setWordWrap(True)
        layout.addWidget(self.caption)
        layout.addWidget(self.value)
        layout.addWidget(self.note)

    def set(self, value: str, note: str = "", color: QColor | None = None) -> None:
        self.value.setText(value)
        self.note.setText(note)
        self.value.setStyleSheet(f"color: {color.name()};" if color is not None else "")


class MathLabel(QWidget):
    """Typesets a math expression (e.g. the symbol w₁ next to its input box)."""

    def __init__(self, node: mx.Node, theme: Theme, px: float = 16.0, parent: QWidget | None = None):
        super().__init__(parent)
        self._node = node
        self._px = px
        self._renderer = MathRenderer(math_style(theme))
        self._box = self._renderer.layout(node, px, display=False)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

    def apply_theme(self, theme: Theme) -> None:
        self._renderer = MathRenderer(math_style(theme))
        self._box = self._renderer.layout(self._node, self._px, display=False)
        self.updateGeometry()
        self.update()

    def set_node(self, node: mx.Node) -> None:
        self._node = node
        self._box = self._renderer.layout(node, self._px, display=False)
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(int(self._box.width + self._box.italic + 4), int(max(self._box.height, self._px * 1.2) + 6))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        box = self._box
        baseline = (self.height() + box.ascent - box.descent) / 2
        box.paint(painter, 2, baseline)


class PageHeader(QWidget):
    """Title + subtitle of a page, with room for actions on the right."""

    def __init__(self, title: str, subtitle: str, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(10)
        texts = QVBoxLayout()
        texts.setSpacing(1)
        self.title = QLabel(title)
        self.title.setObjectName("PageTitle")
        self.subtitle = QLabel(subtitle)
        self.subtitle.setObjectName("PageSubtitle")
        self.subtitle.setWordWrap(True)
        texts.addWidget(self.title)
        texts.addWidget(self.subtitle)
        layout.addLayout(texts, 1)
        self.actions = QHBoxLayout()
        self.actions.setSpacing(8)
        layout.addLayout(self.actions)

    def add_action(self, widget: QWidget) -> None:
        self.actions.addWidget(widget)


def hline(parent: QWidget | None = None) -> QFrame:
    line = QFrame(parent)
    line.setObjectName("Divider")
    line.setFrameShape(QFrame.Shape.NoFrame)
    return line
