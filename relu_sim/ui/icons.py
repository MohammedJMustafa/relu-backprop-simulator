"""Icons drawn from the Windows icon font (Segoe Fluent Icons, or Segoe MDL2 Assets on Windows 10)."""

from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QFontMetricsF, QIcon, QPainter, QPixmap

from .theme import ICON_FALLBACK, ICON_FAMILY

GLYPHS: dict[str, str] = {
    "play": "", "pause": "", "previous": "", "next": "",
    "chevron_left": "", "chevron_right": "", "restart": "", "undo": "",
    "download": "", "sun": "", "moon": "", "info": "", "help": "",
    "shuffle": "", "check": "", "check_circle": "", "error": "", "warning": "",
    "document": "", "picture": "", "chart": "", "pulse": "", "bolt": "",
    "repeat": "", "table": "", "keyboard": "", "pdf": "", "lightbulb": "",
    "education": "", "copy": "", "zoom_in": "", "zoom_out": "",
    "sidebar": "",
}

FALLBACK_TEXT: dict[str, str] = {
    "play": "▶", "pause": "⏸", "previous": "⏮", "next": "⏭", "chevron_left": "‹",
    "chevron_right": "›", "restart": "↻", "undo": "↶", "download": "↓", "sun": "☀",
    "moon": "☾", "info": "i", "help": "?", "shuffle": "⇄", "check": "✓", "check_circle": "✓",
    "error": "✕", "warning": "!", "document": "☰", "picture": "▣", "chart": "↗",
    "pulse": "∿", "bolt": "ϟ", "repeat": "↻", "table": "☷", "keyboard": "⌨",
    "pdf": "☰", "lightbulb": "☀", "education": "⌂", "copy": "⎘", "zoom_in": "+",
    "zoom_out": "−", "sidebar": "☰",
}


@lru_cache(maxsize=1)
def icon_family() -> str | None:
    families = set(QFontDatabase.families())
    for family in (ICON_FAMILY, ICON_FALLBACK):
        if family in families:
            return family
    return None


def glyph_font(pixel_size: float) -> tuple[QFont, str | None]:
    family = icon_family()
    font = QFont(family or "Segoe UI Symbol")
    font.setPixelSize(max(1, round(pixel_size)))
    return font, family


def draw_glyph(painter: QPainter, rect: QRectF, name: str, color: QColor, pixel_size: float) -> None:
    font, family = glyph_font(pixel_size)
    text = GLYPHS[name] if family else FALLBACK_TEXT.get(name, "?")
    painter.save()
    painter.setFont(font)
    painter.setPen(color)
    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
    painter.restore()


@lru_cache(maxsize=512)
def _pixmap(name: str, color_rgba: int, size: int, dpr: float) -> QPixmap:
    pixmap = QPixmap(round(size * dpr), round(size * dpr))
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    draw_glyph(painter, QRectF(0, 0, size, size), name, QColor.fromRgba(color_rgba), size * 0.78)
    painter.end()
    return pixmap


def icon(name: str, color: QColor | str, size: int = 18, disabled: QColor | str | None = None) -> QIcon:
    """A crisp, theme-coloured icon (rendered at 1x and 2x)."""
    color = QColor(color)
    result = QIcon()
    for dpr in (1.0, 2.0):
        result.addPixmap(_pixmap(name, color.rgba(), size, dpr), QIcon.Mode.Normal)
        result.addPixmap(_pixmap(name, color.rgba(), size, dpr), QIcon.Mode.Active)
        if disabled is not None:
            result.addPixmap(_pixmap(name, QColor(disabled).rgba(), size, dpr), QIcon.Mode.Disabled)
    return result


def glyph_width(name: str, pixel_size: float) -> float:
    font, family = glyph_font(pixel_size)
    return QFontMetricsF(font).horizontalAdvance(GLYPHS[name] if family else FALLBACK_TEXT.get(name, "?"))
