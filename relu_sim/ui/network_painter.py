"""The network diagram (Figure 1 of the homework), animated.

``NetworkPainter.paint`` draws a DiagramFrame at animation progress 0..1:
values computed in the step fade in, pulses travel along the edges
(forwards in the forward pass, backwards during backpropagation), ReLU gates
show their derivative, and edge widths and labels morph from the old to the
new weights during the update.  The same painter draws the interactive
view, the figure in the worked solution, the PDF and the PNG export.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen, QPixmap,
                           QRadialGradient, QTransform)

from ..content import mathexpr as mx
from ..content import numfmt
from ..content.diagram import EDGES, DiagramFrame, Pulse, Snapshot
from ..core.network import BIAS_NAMES, Params
from .math_painter import MathRenderer, math_style
from .theme import UI_FAMILIES, UI_FAMILY, Theme

VW, VH = 1000.0, 560.0
MAX_STRETCH = 1.3         # how much taller than the design the diagram may grow to fill a tall view
R = 36.0                  # node radius
E_HALF = 30.0             # half size of the error node (a rounded square)

POS: dict[str, tuple[float, float]] = {
    "x1": (100, 170), "x2": (100, 390),
    "f1": (392, 170), "h1": (552, 170),
    "f2": (392, 390), "h2": (552, 390),
    "f_out": (752, 280), "h_out": (872, 280),
    "E": (962, 280),
    "bias1": (392, 54), "bias2": (392, 486), "bias3": (752, 152),
}

# how each bias tag hangs off its point: "center" = first row centred on it, "top" = the tag starts there
BIAS_PLACE = {"b1": "center", "b2": "top", "b3": "center"}

KIND = {"x1": "input", "x2": "input", "f1": "hidden", "h1": "hidden", "f2": "hidden", "h2": "hidden",
        "f_out": "output", "h_out": "output", "E": "error"}

SYMBOL = {
    "x1": ("x", "1"), "x2": ("x", "2"), "f1": ("f", "1"), "h1": ("h", "1"), "f2": ("f", "2"),
    "h2": ("h", "2"), "f_out": ("f", "out"), "h_out": ("h", "out"), "E": ("E", None),
}

# where each weight label sits: (fraction along the edge, anchor) - anchor "above"/"below"/"center"
LABEL_AT = {"w1": (0.5, "above"), "w4": (0.5, "below"), "w2": (0.72, "center"), "w3": (0.72, "center"),
            "w5": (0.46, "center"), "w6": (0.46, "center")}

# where the gradient chip of a node goes (offset from the node centre)
CHIP_AT = {"h_out": (0, 1), "f_out": (0, 1), "h1": (0, -1), "f1": (0, 1), "h2": (0, 1), "f2": (0, -1)}

GATES = {"f1": "a1", "f2": "a2", "f_out": "a_out"}


def ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def fade(progress: float, start: float = 0.55, end: float = 1.0) -> float:
    if progress >= end:
        return 1.0
    if progress <= start:
        return 0.0
    x = (progress - start) / (end - start)
    return x * x * (3 - 2 * x)


class NetworkPainter:
    def __init__(self, theme: Theme):
        self.theme = theme
        self.math = MathRenderer(math_style(theme))
        self._fonts: dict[tuple, tuple[QFont, QFontMetricsF]] = {}
        self._height = VH
        self._pos = dict(POS)
        self._grid_tile: QPixmap | None = None

    # ------------------------------------------------------------------ helpers

    def font(self, px: float, weight: QFont.Weight = QFont.Weight.Normal) -> tuple[QFont, QFontMetricsF]:
        key = (round(px * 4) / 4, weight)
        cached = self._fonts.get(key)
        if cached is None:
            font = QFont(UI_FAMILY)
            font.setFamilies(UI_FAMILIES)
            font.setPointSizeF(key[0] * 0.75)
            font.setWeight(weight)
            font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
            cached = (font, QFontMetricsF(font))
            self._fonts[key] = cached
        return cached

    def _point(self, name: str) -> QPointF:
        x, y = self._pos[name]
        return QPointF(x, y)

    def _arrange(self, stretch: float) -> None:
        """Spread the rows vertically when the view is taller than the design (sizes stay the same)."""
        self._height = VH * stretch
        self._pos = {name: (x, self._height / 2 + (y - VH / 2) * stretch) for name, (x, y) in POS.items()}

    def _edge_ends(self, edge: str) -> tuple[QPointF, QPointF]:
        source, target = EDGES[edge]
        a, b = self._point(source), self._point(target)
        dx, dy = b.x() - a.x(), b.y() - a.y()
        length = math.hypot(dx, dy) or 1.0
        ux, uy = dx / length, dy / length
        start_gap = 15.0 if source.startswith("bias") else (E_HALF if source == "E" else R) + 2.0
        end_gap = (E_HALF if target == "E" else R) + 3.0
        return (QPointF(a.x() + ux * start_gap, a.y() + uy * start_gap),
                QPointF(b.x() - ux * end_gap, b.y() - uy * end_gap))

    @staticmethod
    def _lerp(a: QPointF, b: QPointF, t: float) -> QPointF:
        return QPointF(a.x() + (b.x() - a.x()) * t, a.y() + (b.y() - a.y()) * t)

    def _text(self, painter: QPainter, text: str, center: QPointF, px: float, color: QColor,
              weight: QFont.Weight = QFont.Weight.Normal) -> float:
        font, fm = self.font(px, weight)
        width = fm.horizontalAdvance(text)
        painter.setFont(font)
        painter.setPen(color)
        painter.drawText(QPointF(center.x() - width / 2, center.y() + (fm.ascent() - fm.descent()) / 2), text)
        return width

    def _math(self, painter: QPainter, node: mx.Node, center: QPointF, px: float, color: QColor | None = None):
        box = self.math.layout(node, px, display=False, color=color)
        box.paint(painter, center.x() - (box.width + box.italic) / 2, center.y() + (box.ascent - box.descent) / 2)
        return box

    @staticmethod
    def _symbol(name: str) -> mx.Node:
        letter, sub = SYMBOL[name]
        return mx.var(letter, sub)

    # ------------------------------------------------------------------ main

    def paint(self, painter: QPainter, rect: QRectF, frame: DiagramFrame, progress: float = 1.0, *,
              mode: str = "simulation", background: bool = True) -> None:
        """mode: 'simulation' (interactive), 'figure' (static Figure 1) or 'mini' (small, no legend)."""
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        stretch = 1.0
        if mode == "simulation":
            wanted = VW * rect.height() / max(rect.width(), 1.0)
            stretch = min(max(wanted / VH, 1.0), MAX_STRETCH)
        self._arrange(stretch)
        scale = min(rect.width() / VW, rect.height() / self._height)
        painter.translate(rect.x() + (rect.width() - VW * scale) / 2,
                          rect.y() + (rect.height() - self._height * scale) / 2)
        painter.scale(scale, scale)

        if background and mode != "figure":
            self._grid(painter)

        before, after = frame.before, frame.after
        p = progress if before is not after else 1.0
        params = self._interpolated_params(before.params, after.params, ease(p))
        phase_color = self._phase_color(frame.phase)

        for edge in EDGES:
            self._edge(painter, edge, params, edge in frame.active_edges, phase_color, mode)
        self._gate_labels(painter, frame, p, mode)
        for name in ("w1", "w2", "w3", "w4", "w5", "w6"):
            self._weight_label(painter, name, params, before, after, p, name in frame.active_edges, mode)
        for name in BIAS_NAMES:
            self._bias_tag(painter, name, params, before, after, p, name in frame.active_edges, mode)
        for node in KIND:
            self._node(painter, node, before, after, p, node in frame.active_nodes, phase_color, mode)
        self._target(painter, after)
        if mode != "mini" and frame.phase not in ("update", "verify"):
            self._node_gradients(painter, before, after, p)
        for pulse in frame.pulses:
            self._pulse(painter, pulse, progress)
        if mode == "simulation":
            self._legend(painter)
        painter.restore()

    # ------------------------------------------------------------------ pieces

    def _grid(self, painter: QPainter) -> None:
        """The dotted background, filled with one cached tile instead of a thousand dots."""
        tile = self._grid_tile
        if tile is None:
            tile = QPixmap(48, 48)
            tile.fill(Qt.GlobalColor.transparent)
            tile_painter = QPainter(tile)
            tile_painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            tile_painter.setPen(Qt.PenStyle.NoPen)
            tile_painter.setBrush(QColor(self.theme.grid_dot))
            tile_painter.drawEllipse(QPointF(24, 24), 2.3, 2.3)
            tile_painter.end()
            self._grid_tile = tile
        painter.save()
        brush = QBrush(tile)
        brush.setTransform(QTransform.fromScale(0.5, 0.5))     # 24 design units between dots
        painter.fillRect(QRectF(0, 0, VW, self._height), brush)
        painter.restore()

    def _phase_color(self, phase: str) -> QColor:
        t = self.theme
        if phase == "backward":
            return QColor(t.gradient)
        if phase == "update":
            return t.role("good")
        if phase == "error":
            return t.role("error")
        return QColor(t.signal)

    @staticmethod
    def _interpolated_params(a: Params, b: Params, t: float) -> Params:
        if a is b or t >= 1.0:
            return b
        return Params.from_values(x + (y - x) * t for x, y in zip(a.values(), b.values()))

    def _edge(self, painter: QPainter, edge: str, params: Params, active: bool, phase_color: QColor,
              mode: str) -> None:
        t = self.theme
        start, end = self._edge_ends(edge)
        if edge.startswith("w"):
            w = params[edge]
            color = QColor(t.edge_pos if w >= 0 else t.edge_neg)
            width = 1.3 + 3.2 * min(abs(w), 1.5) / 1.5
        else:
            color = QColor(t.edge_neutral)
            width = 1.6
        if active and mode == "simulation":
            glow = QColor(phase_color)
            glow.setAlphaF(0.22)
            painter.setPen(QPen(glow, width + 9, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(start, end)
            color = QColor(phase_color)
            width = max(width, 2.6)
        else:
            color.setAlphaF(0.9 if mode != "figure" else 1.0)
        pen = QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        dx, dy = end.x() - start.x(), end.y() - start.y()
        length = math.hypot(dx, dy) or 1.0
        ux, uy = dx / length, dy / length
        head = 9.0 + width * 0.8
        shaft_end = QPointF(end.x() - ux * head * 0.6, end.y() - uy * head * 0.6)
        painter.drawLine(start, shaft_end)
        arrow = QPainterPath(end)
        arrow.lineTo(end.x() - ux * head - uy * head * 0.45, end.y() - uy * head + ux * head * 0.45)
        arrow.lineTo(end.x() - ux * head + uy * head * 0.45, end.y() - uy * head - ux * head * 0.45)
        arrow.closeSubpath()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPath(arrow)

    def _pill(self, painter: QPainter, rect: QRectF, border: QColor | None = None, fill: QColor | None = None,
              radius: float | None = None) -> None:
        t = self.theme
        painter.save()
        painter.setPen(QPen(border or QColor(t.border2), 1.2))
        painter.setBrush(fill or QColor(t.surface))
        r = rect.height() / 2 if radius is None else radius
        painter.drawRoundedRect(rect, r, r)
        painter.restore()

    def _label_rows(self, name: str, params: Params, before: Snapshot, after: Snapshot,
                    progress: float) -> list[tuple]:
        """The rows of a weight/bias label: (kind, payload, alpha)."""
        rows: list[tuple] = []
        old = after.old_params
        value = params[name]
        if old is not None and abs(old[name] - after.params[name]) > 1e-15:
            rows.append(("update", (old[name], value, after.params[name]), 1.0))
        else:
            rows.append(("value", value, 1.0))
        if name in after.param_grads:
            alpha = 1.0 if name in before.param_grads else fade(progress, 0.45, 1.0)
            rows.append(("grad", after.param_grads[name], alpha))
        return rows

    def _draw_label(self, painter: QPainter, name: str, rows: list[tuple], anchor: QPointF, place: str,
                    active: bool) -> None:
        t = self.theme
        symbol = mx.var(name[0], name[1:])
        laid = []
        for kind, payload, alpha in rows:
            if kind == "value":
                box = self.math.layout(mx.row(symbol, mx.EQ, mx.num(payload)), 15.5, display=False)
                laid.append((kind, box, alpha, payload))
            elif kind == "update":
                old, current, new = payload
                box = self.math.layout(mx.row(symbol, mx.EQ, mx.num(old), mx.ARROW,
                                              mx.Num(current, "good", None, False) if abs(current - new) < 1e-12
                                              else mx.Text(numfmt.fmt_compact(current, 5), "upright", "good")),
                                       15.5, display=False)
                laid.append((kind, box, alpha, payload))
            else:
                text = "∇ " + numfmt.fmt_compact(payload, 4)
                font, fm = self.font(13.0, QFont.Weight.DemiBold)
                laid.append((kind, (text, fm.horizontalAdvance(text)), alpha, payload))
        widths = [(item[1].width + item[1].italic) if item[0] != "grad" else item[1][1] for item in laid]
        heights = [25.0 if item[0] != "grad" else 19.0 for item in laid]
        w = max(widths) + 18
        h = sum(heights) + 4
        if place == "above":
            top = anchor.y() - 8 - h
        elif place == "below":
            top = anchor.y() + 8
        elif place == "top":
            top = anchor.y() - 12
        else:
            top = anchor.y() - heights[0] / 2 - 2
        rect = QRectF(anchor.x() - w / 2, top, w, h)
        border = QColor(t.signal) if active else None
        self._pill(painter, rect, border, radius=9)
        y = top + 2
        for (kind, item, alpha, payload), row_h in zip(laid, heights):
            painter.save()
            painter.setOpacity(painter.opacity() * alpha)
            if kind == "grad":
                text, width = item
                font, fm = self.font(13.0, QFont.Weight.DemiBold)
                painter.setFont(font)
                painter.setPen(QColor(t.gradient))
                painter.drawText(QPointF(anchor.x() - width / 2, y + row_h / 2 + (fm.ascent() - fm.descent()) / 2 - 1),
                                 text)
            else:
                item.paint(painter, anchor.x() - (item.width + item.italic) / 2,
                           y + row_h / 2 + (item.ascent - item.descent) / 2)
            painter.restore()
            y += row_h

    def _weight_label(self, painter: QPainter, name: str, params: Params, before: Snapshot, after: Snapshot,
                      progress: float, active: bool, mode: str) -> None:
        start, end = self._edge_ends(name)
        fraction, place = LABEL_AT[name]
        anchor = self._lerp(start, end, fraction)
        rows = self._label_rows(name, params, before, after, progress)
        if mode == "mini":
            rows = rows[:1]
        self._draw_label(painter, name, rows, anchor, place, active and mode == "simulation")

    def _bias_tag(self, painter: QPainter, name: str, params: Params, before: Snapshot, after: Snapshot,
                  progress: float, active: bool, mode: str) -> None:
        anchor = self._point(f"bias{name[1]}")
        rows = self._label_rows(name, params, before, after, progress)
        if mode == "mini":
            rows = rows[:1]
        self._draw_label(painter, name, rows, anchor, BIAS_PLACE[name], active and mode == "simulation")

    def _gate_labels(self, painter: QPainter, frame: DiagramFrame, progress: float, mode: str) -> None:
        t = self.theme
        after, before = frame.after, frame.before
        relu = after.activation == "relu"
        for node, edge in GATES.items():
            start, end = self._edge_ends(edge)
            mid = self._lerp(start, end, 0.5)
            anchor = QPointF(mid.x(), mid.y() - 21)
            if node in after.gates and mode != "mini":
                if node == "f_out":          # the output gate is short: lift its badge above the nodes
                    anchor = QPointF(mid.x(), mid.y() - R - 13)
                gate = after.gates[node]
                alpha = 1.0 if node in before.gates else fade(progress, 0.4, 1.0)
                value = numfmt.fmt_compact(gate, 4)
                dead = gate == 0
                color = t.role("bad") if dead else (t.role("good") if relu else t.role("warn"))
                label = ("ReLU′ = " if relu else "σ′ = ") + value
                font, fm = self.font(12.5, QFont.Weight.DemiBold)
                w = fm.horizontalAdvance(label) + 16
                rect = QRectF(anchor.x() - w / 2, anchor.y() - 10, w, 20)
                painter.save()
                painter.setOpacity(alpha)
                fill = QColor(color)
                fill.setAlphaF(0.14)
                self._pill(painter, rect, color, fill)
                self._text(painter, label, rect.center(), 12.5, color, QFont.Weight.DemiBold)
                painter.restore()
            else:
                name = mx.Text("ReLU") if relu else mx.Text("σ", "italic")
                self._math(painter, name, anchor, 13.0, QColor(t.muted))

    def _node(self, painter: QPainter, node: str, before: Snapshot, after: Snapshot, progress: float,
              active: bool, phase_color: QColor, mode: str) -> None:
        t = self.theme
        kind = KIND[node]
        center = self._point(node)
        known = node in after.values or node in ("x1", "x2")
        appearing = node in after.values and node not in before.values
        changed = (node in after.values and node in before.values
                   and abs(after.values[node] - before.values[node]) > 1e-15)
        filled = known or mode == "figure"

        if active and mode == "simulation":
            glow = QRadialGradient(center, R + 22)
            c0 = QColor(phase_color)
            c0.setAlphaF(0.30)
            c1 = QColor(phase_color)
            c1.setAlphaF(0.0)
            glow.setColorAt(0.55, c0)
            glow.setColorAt(1.0, c1)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(glow))
            painter.drawEllipse(center, R + 22, R + 22)

        fill = t.fill(kind) if filled else QColor(t.surface2)
        stroke = t.stroke(kind) if filled else QColor(t.border2)
        if appearing and mode == "simulation":
            a = fade(progress, 0.5, 1.0)
            base = QColor(t.surface2)
            fill = QColor.fromRgbF(base.redF() + (fill.redF() - base.redF()) * a,
                                   base.greenF() + (fill.greenF() - base.greenF()) * a,
                                   base.blueF() + (fill.blueF() - base.blueF()) * a)
        pen = QPen(stroke, 1.8)
        if not filled:
            pen.setStyle(Qt.PenStyle.DashLine)
        if active and mode == "simulation":
            pen = QPen(phase_color, 2.6)
        painter.setPen(pen)
        painter.setBrush(fill)
        if node == "E":
            painter.drawRoundedRect(QRectF(center.x() - E_HALF, center.y() - E_HALF, 2 * E_HALF, 2 * E_HALF), 11, 11)
        else:
            painter.drawEllipse(center, R, R)

        symbol_color = t.role(kind) if filled else QColor(t.muted)
        has_value = node in after.values and mode != "figure" or node in ("x1", "x2")
        value_alpha = 1.0
        if appearing:
            value_alpha = fade(progress, 0.55, 1.0)
        if has_value:
            symbol_y = center.y() - 11.5
            self._math(painter, self._symbol(node), QPointF(center.x(), symbol_y), 16.5, symbol_color)
            value = after.values[node] if node in after.values else 0.0
            value_text = numfmt.fmt_compact(value, 4)
            px = 14.5 if len(value_text) <= 7 else 12.5
            if changed and mode == "simulation" and progress < 1.0:
                a = fade(progress, 0.35, 1.0)
                old_text = numfmt.fmt_compact(before.values[node], 4)
                painter.save()
                painter.setOpacity(1 - a)
                self._text(painter, old_text, QPointF(center.x(), center.y() + 13 - 8 * a), px, QColor(t.text),
                           QFont.Weight.DemiBold)
                painter.setOpacity(a)
                self._text(painter, value_text, QPointF(center.x(), center.y() + 13 + 8 * (1 - a)), px,
                           QColor(t.text), QFont.Weight.DemiBold)
                painter.restore()
            else:
                painter.save()
                painter.setOpacity(value_alpha)
                self._text(painter, value_text, QPointF(center.x(), center.y() + 13), px, QColor(t.text),
                           QFont.Weight.DemiBold)
                painter.restore()
        else:
            self._math(painter, self._symbol(node), center, 20.0, symbol_color)

        if node == "E" and after.previous_error is not None and mode == "simulation":
            self._error_change(painter, center, after.previous_error, after.values.get("E", 0.0),
                               1.0 if before.previous_error is not None else fade(progress, 0.6, 1.0))

    def _error_change(self, painter: QPainter, center: QPointF, before: float, after: float, alpha: float) -> None:
        t = self.theme
        if before == 0:
            return
        change = (after - before) / before
        good = after < before
        text = ("↓ " if good else "↑ " if after > before else "") + numfmt.fmt_percent(abs(change), 1)
        color = t.role("good") if good else t.role("bad") if after > before else QColor(t.muted)
        painter.save()
        painter.setOpacity(alpha)
        font, fm = self.font(12.0, QFont.Weight.Bold)
        w = fm.horizontalAdvance(text) + 14
        rect = QRectF(center.x() - w / 2, center.y() - E_HALF - 30, w, 20)
        fill = QColor(color)
        fill.setAlphaF(0.15)
        self._pill(painter, rect, color, fill)
        self._text(painter, text, rect.center(), 12.0, color, QFont.Weight.Bold)
        was = "was " + numfmt.fmt_compact(before, 4)
        self._text(painter, was, QPointF(center.x(), center.y() - E_HALF - 42), 11.0, QColor(t.muted))
        painter.restore()

    def _target(self, painter: QPainter, after: Snapshot) -> None:
        t = self.theme
        center = self._point("E")
        node = mx.row(mx.var("y", "true", "target"), mx.EQ, mx.num(after.sample.y_true, "target"))
        box = self.math.layout(node, 14.0, display=False)
        w = box.width + 16
        rect = QRectF(center.x() - w / 2, center.y() + E_HALF + 12, w, 24)
        self._pill(painter, rect, t.role("target", 0.6))
        box.paint(painter, rect.x() + 8, rect.center().y() + (box.ascent - box.descent) / 2)

    def _node_gradients(self, painter: QPainter, before: Snapshot, after: Snapshot, progress: float) -> None:
        t = self.theme
        for node, (dx, dy) in CHIP_AT.items():
            if node not in after.node_grads:
                continue
            alpha = 1.0 if node in before.node_grads else fade(progress, 0.5, 1.0)
            center = self._point(node)
            text = "∇ " + numfmt.fmt_compact(after.node_grads[node], 4)
            font, fm = self.font(12.5, QFont.Weight.DemiBold)
            w = fm.horizontalAdvance(text) + 14
            cy = center.y() + dy * (R + 15)
            rect = QRectF(center.x() - w / 2, cy - 10, w, 20)
            painter.save()
            painter.setOpacity(alpha)
            fill = QColor(t.gradient)
            fill.setAlphaF(0.13)
            self._pill(painter, rect, QColor(t.gradient), fill)
            self._text(painter, text, rect.center(), 12.5, QColor(t.gradient), QFont.Weight.DemiBold)
            painter.restore()

    def _pulse(self, painter: QPainter, pulse: Pulse, progress: float) -> None:
        span = max(pulse.end - pulse.start, 1e-6)
        local = (progress - pulse.start) / span
        if local <= 0.0 or local >= 1.0:
            return
        t = self.theme
        color = QColor(t.signal if pulse.kind == "signal" else t.gradient)
        start, end = self._edge_ends(pulse.edge)
        if pulse.direction < 0:
            start, end = end, start
        alpha = 1.0
        position = ease(local)
        if pulse.blocked:
            if local > 0.55:
                alpha = max(0.0, 1.0 - (local - 0.55) / 0.45)
            position = min(position, 0.5)
        elif local > 0.85:
            alpha = 1.0 - (local - 0.85) / 0.15
        painter.save()
        painter.setOpacity(alpha)
        for k in range(4, 0, -1):
            tail = max(0.0, position - 0.035 * k)
            p = self._lerp(start, end, tail)
            c = QColor(color)
            c.setAlphaF(0.16 * (5 - k) / 4)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(c)
            painter.drawEllipse(p, 4.5 - 0.6 * k, 4.5 - 0.6 * k)
        p = self._lerp(start, end, position)
        glow = QRadialGradient(p, 17)
        g0 = QColor(color)
        g0.setAlphaF(0.45)
        g1 = QColor(color)
        g1.setAlphaF(0.0)
        glow.setColorAt(0.0, g0)
        glow.setColorAt(1.0, g1)
        painter.setBrush(QBrush(glow))
        painter.drawEllipse(p, 17, 17)
        painter.setBrush(color)
        painter.drawEllipse(p, 6, 6)
        painter.setBrush(QColor(255, 255, 255, 230))
        painter.drawEllipse(p, 2.3, 2.3)
        if pulse.blocked and local > 0.45:
            pen = QPen(t.role("bad"), 2.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.setOpacity(min(1.0, (local - 0.45) / 0.15))
            s = 7.0
            painter.drawLine(QPointF(p.x() - s, p.y() - s), QPointF(p.x() + s, p.y() + s))
            painter.drawLine(QPointF(p.x() - s, p.y() + s), QPointF(p.x() + s, p.y() - s))
        painter.restore()

    def _legend(self, painter: QPainter) -> None:
        t = self.theme
        font, fm = self.font(12.5)
        painter.setFont(font)
        items = [("dot", QColor(t.signal), "signal (forward)"), ("dot", QColor(t.gradient), "gradient ∇ (backward)"),
                 ("line", QColor(t.edge_pos), "w > 0"), ("line", QColor(t.edge_neg), "w < 0")]
        total = sum(17 + fm.horizontalAdvance(label) + 20 for _, _, label in items) - 20
        x, y = VW - 16.0 - total, self._height - 14.0
        for kind, color, label in items:
            if kind == "dot":
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
                painter.drawEllipse(QPointF(x + 5, y - 4), 4.5, 4.5)
            else:
                painter.setPen(QPen(color, 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawLine(QPointF(x, y - 4), QPointF(x + 12, y - 4))
            painter.setPen(QColor(t.muted))
            painter.drawText(QPointF(x + 17, y), label)
            x += 17 + fm.horizontalAdvance(label) + 20
