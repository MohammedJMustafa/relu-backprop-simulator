"""Typesetting of relu_sim.content.mathexpr trees with QPainter.

The metrics follow TeX's rules for display math, measured for Cambria:
math axis 0.286 em, rule thickness 0.066 em, display fractions shifted
0.677 em up / 0.686 em down, scripts at 70 % size, 5/18 em around relations
and 4/18 em around binary operators.  Vertical extents come from the glyph
outlines and widths from unhinted font metrics, so a formula looks the same
on screen and in the exported PDF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen

from ..content import mathexpr as mx
from .theme import MATH_FAMILIES, Theme

AXIS = 0.286
RULE = 0.066
SCRIPT_SCALE = 0.70
SPACING = {"rel": 5 / 18, "bin": 4 / 18, "punct": 3 / 18}

PaintFn = Callable[[QPainter, float, float], None]


def _nothing(painter: QPainter, x: float, y: float) -> None:
    pass


@dataclass
class Box:
    """A laid-out formula: ``paint(painter, x, baseline_y)`` draws it."""

    width: float
    ascent: float
    descent: float
    paint: PaintFn = _nothing
    italic: float = 0.0          # italic correction (overhang of a slanted last letter)
    single_char: bool = False    # TeX treats scripts on a single character differently

    @property
    def height(self) -> float:
        return self.ascent + self.descent


@dataclass
class MathStyle:
    text: QColor
    roles: dict[str, QColor]
    box_border: QColor
    box_fill: QColor | None
    rule_scale: float = 1.0

    def color(self, role: str | None, inherited: QColor) -> QColor:
        if role and role in self.roles:
            return self.roles[role]
        return inherited


def math_style(theme: Theme, print_mode: bool = False) -> MathStyle:
    roles = {name: QColor(value) for name, value in theme.roles.items()}
    if print_mode:
        return MathStyle(QColor(theme.text), roles, QColor(theme.text), None)
    return MathStyle(QColor(theme.text), roles, theme.c("accent", 0.75), theme.c("accent", 0.08))


class FontCache:
    """Fonts, metrics and glyph-outline bounds, cached by size and style."""

    def __init__(self, families: list[str] | None = None):
        self.families = list(families or MATH_FAMILIES)
        self._fonts: dict[tuple[float, str], tuple[QFont, QFontMetricsF]] = {}
        self._ink: dict[tuple[str, float, str], QRectF] = {}

    def get(self, px: float, style: str) -> tuple[QFont, QFontMetricsF]:
        key = (round(px * 8) / 8, style)
        cached = self._fonts.get(key)
        if cached is None:
            font = QFont(self.families[0])
            font.setFamilies(self.families)
            font.setPointSizeF(key[0] * 0.75)          # 96 dpi: 1 px = 0.75 pt
            font.setItalic(style == "italic")
            font.setBold(style == "bold")
            font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
            cached = (font, QFontMetricsF(font))
            self._fonts[key] = cached
        return cached

    def ink(self, text: str, px: float, style: str) -> QRectF:
        key = (text, round(px * 8) / 8, style)
        rect = self._ink.get(key)
        if rect is None:
            font, _ = self.get(px, style)
            path = QPainterPath()
            path.addText(QPointF(0, 0), font, text)
            rect = path.boundingRect()
            self._ink[key] = rect
        return rect


_SHARED_FONTS = FontCache()


class MathRenderer:
    def __init__(self, style: MathStyle, fonts: FontCache | None = None):
        self.style = style
        self.fonts = fonts or _SHARED_FONTS

    # ------------------------------------------------------------------ public

    def layout(self, node: mx.Node, px: float, *, display: bool = True, color: QColor | None = None) -> Box:
        return self._layout(node, px, 0, display, color or self.style.text)

    def paint(self, painter: QPainter, node: mx.Node, x: float, y: float, px: float, **kwargs) -> Box:
        box = self.layout(node, px, **kwargs)
        box.paint(painter, x, y)
        return box

    # ------------------------------------------------------------------ dispatch

    def _layout(self, node: mx.Node, px: float, level: int, display: bool, color: QColor) -> Box:
        if isinstance(node, mx.Text):
            return self._text(node.text, px, node.style, self.style.color(node.role, color))
        if isinstance(node, mx.Num):
            return self._text(node.text, px, "upright", self.style.color(node.role, color))
        if isinstance(node, mx.Op):
            return self._text(node.symbol, px, "upright", color)
        if isinstance(node, mx.Row):
            return self._row(node.items, px, level, display, color)
        if isinstance(node, mx.Frac):
            return self._frac(node, px, level, display, color)
        if isinstance(node, mx.Script):
            return self._script(node, px, level, display, color)
        if isinstance(node, mx.Fenced):
            return self._fenced(node, px, level, display, color)
        if isinstance(node, mx.Cases):
            return self._cases(node, px, level, display, color)
        if isinstance(node, mx.Matrix):
            return self._matrix(node, px, level, display, color)
        if isinstance(node, mx.Boxed):
            return self._boxed(node, px, level, display, color)
        if isinstance(node, mx.Space):
            return Box(node.em * px, 0.0, 0.0)
        if isinstance(node, mx.Styled):
            return self._layout(node.body, px, level, display, self.style.color(node.role, color))
        raise TypeError(f"cannot lay out {node!r}")

    # ------------------------------------------------------------------ atoms

    def _text(self, text: str, px: float, style: str, color: QColor) -> Box:
        if not text:
            return Box(0.0, 0.0, 0.0)
        font, fm = self.fonts.get(px, style)
        advance = fm.horizontalAdvance(text)
        ink = self.fonts.ink(text, px, style)
        ascent = max(0.0, -ink.top())
        descent = max(0.0, ink.bottom())
        left = italic = 0.0
        if style == "italic":
            left = max(0.0, -ink.left())
            italic = max(0.0, ink.right() - advance)
        pen = QColor(color)

        def paint(painter: QPainter, x: float, y: float) -> None:
            painter.setFont(font)
            painter.setPen(pen)
            painter.drawText(QPointF(x + left, y), text)

        return Box(advance + left, ascent, descent, paint, italic, len(text) == 1)

    @staticmethod
    def _shift(box: Box, left: float, right: float) -> Box:
        inner = box

        def paint(painter: QPainter, x: float, y: float) -> None:
            inner.paint(painter, x + left, y)

        return Box(left + box.width + right, box.ascent, box.descent, paint)

    @staticmethod
    def hbox(boxes: list[Box]) -> Box:
        placed: list[tuple[float, Box]] = []
        width = ascent = descent = 0.0
        for box in boxes:
            placed.append((width, box))
            width += box.width + box.italic
            ascent = max(ascent, box.ascent)
            descent = max(descent, box.descent)

        def paint(painter: QPainter, x: float, y: float) -> None:
            for dx, box in placed:
                box.paint(painter, x + dx, y)

        return Box(width, ascent, descent, paint)

    def _row(self, items: tuple[mx.Node, ...], px: float, level: int, display: bool, color: QColor) -> Box:
        boxes: list[Box] = []
        previous: mx.Node | None = None
        for item in items:
            if isinstance(item, mx.Op):
                kind = item.kind
                if kind == "bin" and (previous is None or isinstance(previous, mx.Op)):
                    kind = "ord"                     # a leading minus is a sign, not an operator
                glyph = self._text(item.symbol, px, "upright", color)
                if level > 0 or kind == "ord":
                    left = right = 0.0
                elif kind == "punct":
                    left, right = 0.0, SPACING["punct"] * px
                else:
                    left = right = SPACING[kind] * px
                boxes.append(self._shift(glyph, left, right))
            else:
                boxes.append(self._layout(item, px, level, display, color))
            previous = item
        return self.hbox(boxes)

    # ------------------------------------------------------------------ fractions and scripts

    def _frac(self, node: mx.Frac, px: float, level: int, display: bool, color: QColor) -> Box:
        display_style = display and level == 0
        child_px = px if display_style else px * (0.82 if level == 0 else 1.0)
        num = self._layout(node.num, child_px, level, display, color)
        den = self._layout(node.den, child_px, level, display, color)
        em = px
        axis = AXIS * em
        rule = max(RULE * em * self.style.rule_scale, 0.9)
        if display_style:
            up = max(0.677 * em, axis + 3.5 * rule + num.descent)
            down = max(0.686 * em, den.ascent + 3.5 * rule - axis)
        else:
            up = max(0.394 * em, axis + 1.5 * rule + num.descent)
            down = max(0.345 * em, den.ascent + 1.5 * rule - axis)
        pad = 0.12 * em
        width = max(num.width + num.italic, den.width + den.italic) + 2 * pad
        pen_color = QColor(color)

        def paint(painter: QPainter, x: float, y: float) -> None:
            num.paint(painter, x + (width - num.width - num.italic) / 2, y - up)
            den.paint(painter, x + (width - den.width - den.italic) / 2, y + down)
            painter.fillRect(QRectF(x + pad * 0.6, y - axis - rule / 2, width - 1.2 * pad, rule), pen_color)

        return Box(width, up + num.ascent, down + den.descent, paint)

    def _script(self, node: mx.Script, px: float, level: int, display: bool, color: QColor) -> Box:
        base = self._layout(node.base, px, level, display, color)
        script_px = max(px * (SCRIPT_SCALE if level == 0 else 0.75), 6.0)
        sub = self._layout(node.sub, script_px, level + 1, False, color) if node.sub is not None else None
        sup = self._layout(node.sup, script_px, level + 1, False, color) if node.sup is not None else None
        em = px
        _, fm = self.fonts.get(px, "upright")
        x_height = fm.xHeight()
        rule = RULE * em
        if base.single_char:
            u0 = v0 = 0.0
        else:
            u0 = base.ascent - 0.386 * script_px
            v0 = base.descent + 0.05 * script_px
        up = down = 0.0
        if sup is not None:
            up = max(u0, 0.36 * em, sup.descent + x_height / 4)
        if sub is not None and sup is None:
            down = max(v0, 0.15 * em, sub.ascent - 0.8 * x_height)
        elif sub is not None and sup is not None:
            down = max(v0, 0.24 * em)
            gap = (up - sup.descent) - (sub.ascent - down)
            if gap < 4 * rule:
                down += 4 * rule - gap
                psi = 0.8 * x_height - (up - sup.descent)
                if psi > 0:
                    up += psi
                    down -= psi

        script_space = 0.05 * em
        width = base.width + max(base.italic + sup.width + sup.italic if sup else 0.0,
                                 sub.width + sub.italic if sub else 0.0) + script_space
        ascent = max(base.ascent, up + sup.ascent if sup else 0.0)
        descent = max(base.descent, down + sub.descent if sub else 0.0)

        def paint(painter: QPainter, x: float, y: float) -> None:
            base.paint(painter, x, y)
            if sup is not None:
                sup.paint(painter, x + base.width + base.italic, y - up)
            if sub is not None:
                sub.paint(painter, x + base.width, y + down)

        italic = base.italic if sub is None and sup is None else 0.0
        return Box(width, ascent, descent, paint, italic)

    # ------------------------------------------------------------------ delimiters

    def _delimiter(self, symbol: str, px: float, half_height: float, color: QColor) -> Box:
        """A bracket around content reaching ``half_height`` above/below the math axis."""
        glyph = self._text(symbol, px, "upright", color)
        em = px
        axis = AXIS * em
        natural = glyph.ascent + glyph.descent
        if 2 * half_height <= natural * 1.3:     # ordinary content (even with subscripts): the font's glyph
            return glyph
        height = 2 * half_height + 0.14 * em
        width = min(0.30 * em + 0.035 * height, 0.55 * em)
        top_above_baseline = axis + height / 2
        pen_color = QColor(color)

        def paint(painter: QPainter, x: float, y: float) -> None:
            top = y - top_above_baseline
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            if symbol in "()":
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(pen_color)
                painter.drawPath(_paren_path(x, top, width, height, em, symbol == "("))
            else:
                pen = QPen(pen_color, max(0.055 * em, 1.0))
                pen.setCapStyle(Qt.PenCapStyle.FlatCap if symbol in "[]" else Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(_bracket_path(x, top, width, height, symbol))
            painter.restore()

        return Box(width, top_above_baseline, height - top_above_baseline, paint)

    def _fenced(self, node: mx.Fenced, px: float, level: int, display: bool, color: QColor) -> Box:
        body = self._layout(node.body, px, level, display, color)
        axis = AXIS * px
        half_height = max(body.ascent - axis, body.descent + axis)
        parts = []
        if node.left:
            parts.append(self._delimiter(node.left, px, half_height, color))
        parts.append(body)
        if node.right:
            parts.append(self._delimiter(node.right, px, half_height, color))
        return self.hbox(parts)

    def _stack(self, rows: list[list[Box]], px: float, align: str, col_gap: float) -> tuple[Box, float]:
        """Rows of cells centred on the math axis; returns the box and its half height."""
        em = px
        axis = AXIS * em
        columns = max(len(cells) for cells in rows)
        col_width = [max((cells[c].width + cells[c].italic) for cells in rows if c < len(cells))
                     for c in range(columns)]
        row_ascent = [max([b.ascent for b in cells] + [0.70 * em]) for cells in rows]
        row_descent = [max([b.descent for b in cells] + [0.22 * em]) for cells in rows]
        pitch = max(1.25 * em, max(a + d for a, d in zip(row_ascent, row_descent)) + 0.28 * em)
        baselines = [0.0]
        for i in range(1, len(rows)):
            baselines.append(baselines[-1] + max(pitch, row_descent[i - 1] + row_ascent[i] + 0.25 * em))
        top = -row_ascent[0]
        bottom = baselines[-1] + row_descent[-1]
        shift = -axis - (top + bottom) / 2          # centre the stack on the axis
        width = sum(col_width) + col_gap * (columns - 1)

        def paint(painter: QPainter, x: float, y: float) -> None:
            for cells, base in zip(rows, baselines):
                cx = x
                for c, box in enumerate(cells):
                    w = box.width + box.italic
                    if align == "r":
                        offset = col_width[c] - w
                    elif align == "c":
                        offset = (col_width[c] - w) / 2
                    else:
                        offset = 0.0
                    box.paint(painter, cx + offset, y + base + shift)
                    cx += col_width[c] + col_gap

        ascent = -(top + shift)
        descent = bottom + shift
        return Box(width, ascent, descent, paint), max(ascent - axis, descent + axis)

    def _matrix(self, node: mx.Matrix, px: float, level: int, display: bool, color: QColor) -> Box:
        rows = [[self._layout(c, px, level, display, color) for c in cells] for cells in node.rows]
        body, half_height = self._stack(rows, px, "r", 0.9 * px)
        gap = 0.16 * px
        body = self._shift(body, gap, gap)
        return self.hbox([self._delimiter(node.left, px, half_height, color), body,
                          self._delimiter(node.right, px, half_height, color)])

    def _cases(self, node: mx.Cases, px: float, level: int, display: bool, color: QColor) -> Box:
        rows = [[self._layout(v, px, level, display, color), self._layout(c, px, level, display, color)]
                for v, c in node.rows]
        body, half_height = self._stack(rows, px, "l", 0.9 * px)
        brace = self._delimiter("{", px, half_height, color)
        return self.hbox([brace, self._shift(body, 0.22 * px, 0.0)])

    def _boxed(self, node: mx.Boxed, px: float, level: int, display: bool, color: QColor) -> Box:
        body = self._layout(node.body, px, level, display, color)
        em = px
        pad_x, pad_y = 0.34 * em, 0.20 * em
        inner_ascent = max(body.ascent, 0.70 * em)
        inner_descent = max(body.descent, 0.20 * em)
        width = body.width + body.italic + 2 * pad_x
        ascent, descent = inner_ascent + pad_y, inner_descent + pad_y
        border = QColor(self.style.box_border)
        fill = QColor(self.style.box_fill) if self.style.box_fill is not None else None
        line = max(0.06 * em, 1.0)

        def paint(painter: QPainter, x: float, y: float) -> None:
            rect = QRectF(x + line / 2, y - ascent + line / 2, width - line, ascent + descent - line)
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(border, line))
            painter.setBrush(fill if fill is not None else Qt.BrushStyle.NoBrush)
            radius = 0.14 * em if fill is not None else 0.0
            painter.drawRoundedRect(rect, radius, radius)
            painter.restore()
            body.paint(painter, x + pad_x, y)

        return Box(width, ascent, descent, paint)


def _paren_path(x: float, top: float, width: float, height: float, em: float, left: bool) -> QPainterPath:
    """A filled crescent-shaped parenthesis."""
    thick = min(0.075 * em * (1 + 0.12 * (height / em - 1)), 0.13 * em)
    tip = 0.022 * em
    if left:
        end_x, deep_x = x + width * 0.86, x + width * 0.12
        inner_end_x, inner_deep_x = end_x + tip, deep_x + thick
    else:
        end_x, deep_x = x + width * 0.14, x + width * 0.88
        inner_end_x, inner_deep_x = end_x - tip, deep_x - thick
    # control points so that the curve reaches deep_x at mid height
    c_out = (deep_x - 0.25 * end_x) / 0.75
    c_in = (inner_deep_x - 0.25 * inner_end_x) / 0.75
    path = QPainterPath(QPointF(end_x, top))
    path.cubicTo(QPointF(c_out, top + height * 0.24), QPointF(c_out, top + height * 0.76), QPointF(end_x, top + height))
    path.lineTo(QPointF(inner_end_x, top + height))
    path.cubicTo(QPointF(c_in, top + height * 0.76), QPointF(c_in, top + height * 0.24), QPointF(inner_end_x, top))
    path.closeSubpath()
    return path


def _bracket_path(x: float, top: float, width: float, height: float, symbol: str) -> QPainterPath:
    bottom = top + height
    path = QPainterPath()
    if symbol == "[":
        path.moveTo(x + width * 0.85, top)
        path.lineTo(x + width * 0.28, top)
        path.lineTo(x + width * 0.28, bottom)
        path.lineTo(x + width * 0.85, bottom)
    elif symbol == "]":
        path.moveTo(x + width * 0.15, top)
        path.lineTo(x + width * 0.72, top)
        path.lineTo(x + width * 0.72, bottom)
        path.lineTo(x + width * 0.15, bottom)
    elif symbol == "{":
        right, stem, tip = x + width * 0.92, x + width * 0.52, x + width * 0.08
        mid = top + height / 2
        hook = min(height * 0.12, width * 0.9)
        path.moveTo(right, top)
        path.cubicTo(QPointF(stem, top), QPointF(stem, top + hook * 0.4), QPointF(stem, top + hook))
        path.lineTo(stem, mid - hook)
        path.cubicTo(QPointF(stem, mid - hook * 0.35), QPointF(tip + (stem - tip) * 0.4, mid), QPointF(tip, mid))
        path.cubicTo(QPointF(tip + (stem - tip) * 0.4, mid), QPointF(stem, mid + hook * 0.35), QPointF(stem, mid + hook))
        path.lineTo(stem, top + height - hook)
        path.cubicTo(QPointF(stem, top + height - hook * 0.4), QPointF(stem, top + height), QPointF(right, top + height))
    else:  # "|" or anything else: a straight bar
        path.moveTo(x + width / 2, top)
        path.lineTo(x + width / 2, bottom)
    return path
