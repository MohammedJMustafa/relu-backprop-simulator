"""The app logo: the ReLU curve on a blue tile, drawn with QPainter (no image files needed)."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap


def paint_logo(painter: QPainter, rect: QRectF) -> None:
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = min(rect.width(), rect.height())
    x0 = rect.x() + (rect.width() - s) / 2
    y0 = rect.y() + (rect.height() - s) / 2
    tile = QRectF(x0, y0, s, s)
    gradient = QLinearGradient(tile.topLeft(), tile.bottomRight())
    gradient.setColorAt(0.0, QColor("#4F7DFF"))
    gradient.setColorAt(1.0, QColor("#7A4FF0"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(gradient)
    painter.drawRoundedRect(tile, s * 0.24, s * 0.24)

    # faint axis
    axis = QPen(QColor(255, 255, 255, 70), max(1.0, s * 0.035))
    axis.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(axis)
    painter.drawLine(QPointF(x0 + s * 0.18, y0 + s * 0.70), QPointF(x0 + s * 0.84, y0 + s * 0.70))

    # the ReLU curve: flat, then a straight rise
    curve = QPainterPath(QPointF(x0 + s * 0.17, y0 + s * 0.70))
    curve.lineTo(QPointF(x0 + s * 0.47, y0 + s * 0.70))
    curve.lineTo(QPointF(x0 + s * 0.80, y0 + s * 0.27))
    pen = QPen(QColor("#FFFFFF"), max(1.6, s * 0.105))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPath(curve)

    # a "signal" dot travelling on the curve
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#7CF5C4"))
    painter.drawEllipse(QPointF(x0 + s * 0.64, y0 + s * 0.485), s * 0.085, s * 0.085)
    painter.restore()


def logo_pixmap(size: int, dpr: float = 2.0) -> QPixmap:
    pixmap = QPixmap(round(size * dpr), round(size * dpr))
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    paint_logo(painter, QRectF(0, 0, size, size))
    painter.end()
    return pixmap


def app_icon() -> QIcon:
    icon = QIcon()
    for size in (16, 20, 24, 32, 40, 48, 64, 128, 256):
        icon.addPixmap(logo_pixmap(size, 1.0))
    return icon
