"""Layout and painting of document blocks (relu_sim.content.blocks) with QPainter.

One code path draws the worked solution on screen and into the PDF export,
so both look the same.  Paragraphs wrap their words and inline formulas;
display equations are aligned on their relation sign and, when the space is
too narrow, broken at a top-level "=" and finally scaled down.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter

from ..content import blocks as bk
from ..content import mathexpr as mx
from .icons import draw_glyph
from .math_painter import Box, MathRenderer, math_style
from .theme import LIGHT, MATH_FAMILIES, UI_FAMILIES, Theme

PaintAt = Callable[[QPainter, float, float], None]
FigurePainter = Callable[[QPainter, QRectF, str, object], None]


@dataclass
class DocStyle:
    kind: str
    body_family: list[str]
    heading_family: list[str]
    body_px: float
    math_px: float
    line_height: float
    text: QColor
    muted: QColor
    heading: QColor
    rule: QColor
    group_fill: QColor
    zebra: QColor | None
    callouts: dict[str, tuple[QColor, QColor]]
    roles: dict[str, QColor]


def doc_style(theme: Theme, kind: str) -> DocStyle:
    """kind: 'paper' (the worked solution), 'card' (step explanations), 'print' (PDF export)."""
    if kind == "print":
        theme = LIGHT
    serif = list(MATH_FAMILIES)
    sans = list(UI_FAMILIES)
    sizes = {"paper": (16.0, 17.5, 1.55), "card": (13.5, 16.5, 1.5), "print": (13.0, 14.0, 1.5)}[kind]
    callouts = {
        "info": (theme.c("accent", 0.10), theme.c("accent")),
        "note": (theme.c("accent", 0.08), theme.c("accent")),
        "success": (theme.role("good", 0.12), theme.role("good")),
        "warning": (theme.role("warn", 0.14), theme.role("warn")),
    }
    return DocStyle(
        kind=kind,
        body_family=serif if kind in ("paper", "print") else sans,
        heading_family=serif if kind in ("paper", "print") else sans,
        body_px=sizes[0], math_px=sizes[1], line_height=sizes[2],
        text=theme.c("text"), muted=theme.c("muted"),
        heading=theme.c("heading") if kind != "card" else theme.c("text"),
        rule=theme.c("text", 0.85) if kind != "card" else theme.c("border2"),
        group_fill=theme.c("surface3"), zebra=theme.c("surface2") if kind != "print" else QColor("#F6F7FA"),
        callouts=callouts,
        roles={name: QColor(value) for name, value in theme.roles.items()},
    )


@dataclass
class Placed:
    top: float
    height: float
    paint: PaintAt
    keep_with_next: bool = False


@dataclass
class DocLayout:
    width: float
    height: float
    blocks: list[Placed]


@dataclass
class InlineBox:
    """A laid-out paragraph: its size and how to paint it at (x, top)."""

    width: float
    height: float
    natural_width: float
    paint: PaintAt


class DocumentPainter:
    def __init__(self, theme: Theme, kind: str = "paper", figure_painter: FigurePainter | None = None):
        self.kind = kind
        self.style = doc_style(theme, kind)
        self.math = MathRenderer(math_style(LIGHT if kind == "print" else theme, print_mode=kind == "print"))
        self.figure_painter = figure_painter
        self._fonts: dict[tuple, tuple[QFont, QFontMetricsF]] = {}

    # ------------------------------------------------------------------ public

    def layout(self, blocks: tuple[bk.Block, ...], width: float) -> DocLayout:
        placed: list[Placed] = []
        y = 0.0
        previous_after = 0.0
        for block in blocks:
            before, after, keep = self._spacing(block)
            if placed:
                y += max(previous_after, before)
            height, paint = self._layout_block(block, width)
            placed.append(Placed(y, height, paint, keep))
            y += height
            previous_after = after
        return DocLayout(width, y, placed)

    @staticmethod
    def paint(painter: QPainter, layout: DocLayout, x: float, y: float, clip: QRectF | None = None) -> None:
        for block in layout.blocks:
            top = y + block.top
            if clip is not None and (top > clip.bottom() or top + block.height < clip.top()):
                continue
            block.paint(painter, x, top)

    # ------------------------------------------------------------------ fonts

    def font(self, px: float, bold: bool = False, italic: bool = False, family: list[str] | None = None,
             weight: QFont.Weight | None = None) -> tuple[QFont, QFontMetricsF]:
        families = tuple(family or self.style.body_family)
        key = (families, round(px * 8) / 8, bold, italic, weight)
        cached = self._fonts.get(key)
        if cached is None:
            font = QFont(families[0])
            font.setFamilies(list(families))
            font.setPointSizeF(px * 0.75)
            font.setItalic(italic)
            if weight is not None:
                font.setWeight(weight)
            elif bold:
                font.setWeight(QFont.Weight.Bold)
            font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
            cached = (font, QFontMetricsF(font))
            self._fonts[key] = cached
        return cached

    # ------------------------------------------------------------------ spacing

    def _spacing(self, block: bk.Block) -> tuple[float, float, bool]:
        em = self.style.body_px
        if isinstance(block, bk.TitleBlock):
            return 0.0, 1.3 * em, False
        if isinstance(block, bk.Heading):
            return ((1.7 * em, 0.55 * em, True) if block.level <= 2 else (1.05 * em, 0.4 * em, True))
        if isinstance(block, bk.Paragraph):
            return 0.55 * em, 0.55 * em, False
        if isinstance(block, (bk.Equation, bk.EquationGrid)):
            return 0.7 * em, 0.7 * em, False
        if isinstance(block, bk.Table):
            return 0.95 * em, 0.95 * em, False
        if isinstance(block, bk.Figure):
            return 1.0 * em, 1.0 * em, False
        if isinstance(block, bk.Callout):
            return 0.95 * em, 0.95 * em, False
        if isinstance(block, bk.Spacer):
            return block.em * em, 0.0, False
        return 0.5 * em, 0.5 * em, False

    def _layout_block(self, block: bk.Block, width: float) -> tuple[float, PaintAt]:
        if isinstance(block, bk.TitleBlock):
            return self._title(block, width)
        if isinstance(block, bk.Heading):
            return self._heading(block, width)
        if isinstance(block, bk.Paragraph):
            return self._paragraph(block, width)
        if isinstance(block, bk.Equation):
            return self._equation(list(block.lines), width)
        if isinstance(block, bk.EquationGrid):
            return self._grid(block, width)
        if isinstance(block, bk.Table):
            return self._table(block, width)
        if isinstance(block, bk.Figure):
            return self._figure(block, width)
        if isinstance(block, bk.Callout):
            return self._callout(block, width)
        return 0.0, lambda painter, x, y: None

    # ------------------------------------------------------------------ inline text

    def inline(self, items: tuple[bk.Inline, ...], width: float, px: float, color: QColor, *,
               bold: bool = False, align: str = "left", family: list[str] | None = None,
               weight: QFont.Weight | None = None, math_scale: float = 1.06) -> InlineBox:
        """Wrap text runs and inline math to ``width``."""
        tokens: list[tuple[str, object]] = []
        _, base_fm = self.font(px, bold, False, family, weight)
        for item in items:
            if isinstance(item, bk.Run):
                font, fm = self.font(px, bold or item.bold, item.italic, family, weight)
                run_color = self.style.roles.get(item.role, color) if item.role else color
                for piece in re.findall(r"\s+|\S+", item.text):
                    if piece.isspace():
                        tokens.append(("space", fm.horizontalAdvance(" ")))
                    else:
                        tokens.append(("word", self._text_box(piece, font, fm, run_color)))
            else:
                tokens.append(("word", self.math.layout(item, px * math_scale, display=False, color=color)))

        units: list[tuple[float, list[Box]]] = []
        current: list[Box] | None = None
        space_before = pending = 0.0
        for kind, value in tokens:
            if kind == "space":
                if current is not None:
                    units.append((space_before, current))
                    current = None
                pending = value
            else:
                if current is None:
                    current, space_before, pending = [value], pending, 0.0
                else:
                    current.append(value)
        if current is not None:
            units.append((space_before, current))

        lines: list[list[tuple[float, list[Box]]]] = []
        line: list[tuple[float, list[Box]]] = []
        line_width = 0.0
        for space, boxes in units:
            w = sum(b.width + b.italic for b in boxes)
            if line and line_width + space + w > width:
                lines.append(line)
                line, line_width = [(0.0, boxes)], w
            else:
                gap = space if line else 0.0
                line.append((gap, boxes))
                line_width += gap + w
        if line:
            lines.append(line)

        strut_a, strut_d = base_fm.ascent(), base_fm.descent()
        min_height = self.style.line_height * px
        laid = []
        y = 0.0
        natural = 0.0
        for parts in lines:
            ascent = max([strut_a] + [b.ascent for _, boxes in parts for b in boxes])
            descent = max([strut_d] + [b.descent for _, boxes in parts for b in boxes])
            height = max(ascent + descent, min_height)
            baseline = y + (height - ascent - descent) / 2 + ascent
            w = sum(gap + sum(b.width + b.italic for b in boxes) for gap, boxes in parts)
            natural = max(natural, w)
            laid.append((baseline, w, parts))
            y += height

        def paint(painter: QPainter, x: float, top: float) -> None:
            for baseline, w, parts in laid:
                cx = x + ((width - w) / 2 if align == "center" else (width - w) if align == "right" else 0.0)
                for gap, boxes in parts:
                    cx += gap
                    for box in boxes:
                        box.paint(painter, cx, top + baseline)
                        cx += box.width + box.italic

        return InlineBox(width, y, natural, paint)

    @staticmethod
    def _text_box(text: str, font: QFont, fm: QFontMetricsF, color: QColor) -> Box:
        pen = QColor(color)

        def paint(painter: QPainter, x: float, y: float) -> None:
            painter.setFont(font)
            painter.setPen(pen)
            painter.drawText(QPointF(x, y), text)

        return Box(fm.horizontalAdvance(text), fm.ascent(), fm.descent(), paint)

    # ------------------------------------------------------------------ blocks

    def _title(self, block: bk.TitleBlock, width: float) -> tuple[float, PaintAt]:
        s = self.style
        px = s.body_px
        title = self.inline(bk.inline(block.title), width, px * 1.75, s.text, bold=True, align="center",
                            family=s.heading_family)
        subtitle = self.inline(bk.inline(block.subtitle), width, px * 1.08, s.text, align="center")
        name = self.inline((bk.Run("Name: ", bold=True), bk.Run(block.name)), width, px, s.text)
        group = self.inline((bk.Run("Group: ", bold=True), bk.Run(block.group)), width, px, s.text, align="right")
        gap1, gap2 = 0.35 * px, 1.4 * px
        name_top = title.height + gap1 + subtitle.height + gap2
        height = name_top + name.height + 0.35 * px + 1.2

        def paint(painter: QPainter, x: float, top: float) -> None:
            title.paint(painter, x, top)
            subtitle.paint(painter, x, top + title.height + gap1)
            name.paint(painter, x, top + name_top)
            group.paint(painter, x, top + name_top)
            painter.fillRect(QRectF(x, top + height - 1.2, width, 1.2), s.rule)

        return height, paint

    def _heading(self, block: bk.Heading, width: float) -> tuple[float, PaintAt]:
        s = self.style
        if block.level <= 2:
            box = self.inline(bk.inline(block.text), width, s.body_px * 1.28, s.heading, bold=True,
                              family=s.heading_family)
        else:
            box = self.inline(bk.inline(block.text), width, s.body_px * 1.02, s.text, bold=True,
                              family=s.heading_family)
        return box.height, box.paint

    def _paragraph(self, block: bk.Paragraph, width: float) -> tuple[float, PaintAt]:
        s = self.style
        px = s.body_px * (0.86 if block.style == "small" else 1.0)
        color = s.muted if block.style in ("muted", "small") else s.text
        box = self.inline(block.items, width, px, color, align="center" if block.style == "center" else "left")
        return box.height, box.paint

    def _equation(self, lines: list[bk.EqLine], width: float) -> tuple[float, PaintAt]:
        px = self.style.math_px

        def lay(eq_lines):
            return [(self.math.layout(l.lhs, px) if l.lhs is not None else None, self.math.layout(l.rhs, px))
                    for l in eq_lines]

        def widths(laid_lines):
            lw = max((lb.width + lb.italic for lb, _ in laid_lines if lb is not None), default=0.0)
            rw = max((rb.width + rb.italic for _, rb in laid_lines), default=0.0)
            return lw, rw

        laid = lay(lines)
        lw, rw = widths(laid)
        if lw + rw > width:
            broken: list[bk.EqLine] = []
            for line, (lb, rb) in zip(lines, laid):
                if line.lhs is None or lw + rb.width <= width:
                    broken.append(line)
                    continue
                available = width - lw
                current: list[mx.Node] = []
                current_width = 0.0
                first = True
                for chunk in split_relations(line.rhs):
                    chunk_width = self.math.layout(chunk, px).width
                    if current and current_width + chunk_width > available:
                        broken.append(bk.EqLine(line.lhs if first else None, mx.row(*current)))
                        first, current, current_width = False, [chunk], chunk_width
                    else:
                        current.append(chunk)
                        current_width += chunk_width
                if current:
                    broken.append(bk.EqLine(line.lhs if first else None, mx.row(*current)))
            lines = broken
            laid = lay(lines)
            lw, rw = widths(laid)

        natural = lw + rw
        scale = 1.0 if natural <= width else max(width / natural, 0.5)
        strut_a, strut_d, gap = 0.82 * px, 0.34 * px, 0.42 * px
        rows = []
        y = 0.0
        for lb, rb in laid:
            ascent = max(strut_a, rb.ascent, lb.ascent if lb else 0.0)
            descent = max(strut_d, rb.descent, lb.descent if lb else 0.0)
            rows.append((y + ascent, lb, rb))
            y += ascent + descent + gap
        height = max(0.0, y - gap) * scale

        def paint(painter: QPainter, x: float, top: float) -> None:
            painter.save()
            painter.translate(x + (width - natural * scale) / 2, top)
            painter.scale(scale, scale)
            for baseline, lb, rb in rows:
                if lb is not None:
                    lb.paint(painter, lw - lb.width - lb.italic, baseline)
                rb.paint(painter, lw, baseline)
            painter.restore()

        return height, paint

    def _grid(self, block: bk.EquationGrid, width: float) -> tuple[float, PaintAt]:
        px = self.style.math_px
        columns = max(len(cells) for cells in block.rows)
        laid = [[(self.math.layout(c.lhs, px) if c.lhs is not None else None, self.math.layout(c.rhs, px))
                 if c is not None else None for c in cells] for cells in block.rows]
        col_l = [max((cells[c][0].width + cells[c][0].italic for cells in laid
                      if c < len(cells) and cells[c] is not None and cells[c][0] is not None), default=0.0)
                 for c in range(columns)]
        col_r = [max((cells[c][1].width + cells[c][1].italic for cells in laid
                      if c < len(cells) and cells[c] is not None), default=0.0) for c in range(columns)]
        gap = 2.0 * px
        total = sum(col_l) + sum(col_r) + gap * (columns - 1)
        if total > width:
            return self._equation([c for cells in block.rows for c in cells if c is not None], width)

        strut_a, strut_d, row_gap = 0.82 * px, 0.34 * px, 0.5 * px
        rows = []
        y = 0.0
        for cells in laid:
            present = [c for c in cells if c is not None]
            ascent = max([strut_a] + [max(rb.ascent, lb.ascent if lb else 0.0) for lb, rb in present])
            descent = max([strut_d] + [max(rb.descent, lb.descent if lb else 0.0) for lb, rb in present])
            rows.append((y + ascent, cells))
            y += ascent + descent + row_gap
        height = max(0.0, y - row_gap)

        def paint(painter: QPainter, x: float, top: float) -> None:
            x0 = x + (width - total) / 2
            for baseline, cells in rows:
                cx = x0
                for c in range(columns):
                    if c < len(cells) and cells[c] is not None:
                        lb, rb = cells[c]
                        if lb is not None:
                            lb.paint(painter, cx + col_l[c] - lb.width - lb.italic, top + baseline)
                        rb.paint(painter, cx + col_l[c], top + baseline)
                    cx += col_l[c] + col_r[c] + gap

        return height, paint

    def _table(self, table: bk.Table, width: float) -> tuple[float, PaintAt]:
        s = self.style
        px = s.body_px * 0.95
        columns = len(table.align)
        pad_x, pad_y = 0.75 * px, 0.42 * px

        def cell_box(cell: bk.Cell, header: bool = False) -> InlineBox:
            color = s.roles.get(cell.role, s.text) if cell.role else s.text
            return self.inline(cell.items, 10_000, px, color, bold=cell.bold or header)

        header = [cell_box(c, True) for c in table.header]
        body = [[cell_box(c) for c in cells] for cells in table.rows]
        measured = [header] + [cells for i, cells in enumerate(body) if i not in table.group_rows]
        col_w = [max((cells[c].natural_width for cells in measured if c < len(cells)), default=0.0)
                 for c in range(columns)]
        natural = sum(col_w) + 2 * pad_x * columns
        scale = min(1.0, width / natural) if natural > 0 else 1.0
        row_h = max(s.line_height * px, px * 1.35) + 2 * pad_y

        def row_height(cells: list[InlineBox]) -> float:
            return max([row_h] + [c.height + 2 * pad_y for c in cells])

        caption = self.inline(table.caption, width / scale, px * 0.92, s.text, align="center") if table.caption else None
        header_h = row_height(header) if header else 0.0
        body_h = [row_height(cells) for cells in body]
        thick, thin = 1.4, 0.9
        table_h = thick + header_h + (thin if header else 0.0) + sum(body_h) + thick
        caption_gap = 0.55 * px
        height = (table_h + (caption_gap + caption.height if caption else 0.0)) * scale
        zebra = s.zebra if len(body) > 6 else None

        def draw_row(painter: QPainter, cells: list[InlineBox], x: float, y: float, h: float) -> None:
            cx = x
            for c, box in enumerate(cells):
                align = table.align[c] if c < len(table.align) else "l"
                inner = col_w[c]
                if align == "r":
                    offset = inner - box.natural_width
                elif align == "c":
                    offset = (inner - box.natural_width) / 2
                else:
                    offset = 0.0
                box.paint(painter, cx + pad_x + offset, y + (h - box.height) / 2)
                cx += inner + 2 * pad_x

        def paint(painter: QPainter, x: float, top: float) -> None:
            painter.save()
            painter.translate(x + (width - natural * scale) / 2, top)
            painter.scale(scale, scale)
            y = 0.0
            painter.fillRect(QRectF(0, y, natural, thick), s.rule)
            y += thick
            if header:
                draw_row(painter, header, 0.0, y, header_h)
                y += header_h
                painter.fillRect(QRectF(0, y, natural, thin), s.rule)
                y += thin
            stripe = 0
            for i, cells in enumerate(body):
                h = body_h[i]
                if i in table.group_rows:
                    painter.fillRect(QRectF(0, y, natural, h), s.group_fill)
                    cells[0].paint(painter, pad_x, y + (h - cells[0].height) / 2)
                    stripe = 0
                else:
                    if zebra is not None and stripe % 2 == 1:
                        painter.fillRect(QRectF(0, y, natural, h), zebra)
                    draw_row(painter, cells, 0.0, y, h)
                    stripe += 1
                y += h
            painter.fillRect(QRectF(0, y, natural, thick), s.rule)
            y += thick
            if caption is not None:
                caption.paint(painter, (natural - caption.width) / 2, y + caption_gap)
            painter.restore()

        return height, paint

    def _figure(self, figure: bk.Figure, width: float) -> tuple[float, PaintAt]:
        s = self.style
        w = min(width, 780.0)
        h = w * figure.aspect
        caption = (self.inline(figure.caption, width, s.body_px * 0.9, s.text, align="center")
                   if figure.caption else None)
        gap = 0.4 * s.body_px
        height = h + (gap + caption.height if caption else 0.0)

        def paint(painter: QPainter, x: float, top: float) -> None:
            if self.figure_painter is not None:
                painter.save()
                self.figure_painter(painter, QRectF(x + (width - w) / 2, top, w, h), figure.kind, figure.payload)
                painter.restore()
            if caption is not None:
                caption.paint(painter, x, top + h + gap)

        return height, paint

    def _callout(self, callout: bk.Callout, width: float) -> tuple[float, PaintAt]:
        s = self.style
        px = s.body_px * 0.95
        background, accent = s.callouts.get(callout.kind, s.callouts["info"])
        pad = 0.8 * px
        icon_w = 1.55 * px
        items = ((bk.Run(callout.title + "  ", bold=True, role=None),) if callout.title else ()) + callout.items
        text = self.inline(items, width - 2 * pad - icon_w - 6, px, s.text)
        height = text.height + 1.3 * pad
        glyph = {"info": "info", "note": "lightbulb", "success": "check_circle", "warning": "warning"}.get(
            callout.kind, "info")

        def paint(painter: QPainter, x: float, top: float) -> None:
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(background)
            painter.drawRoundedRect(QRectF(x, top, width, height), 8, 8)
            painter.setBrush(accent)
            painter.drawRoundedRect(QRectF(x, top, 3.5, height), 1.75, 1.75)
            draw_glyph(painter, QRectF(x + pad * 0.7, top + 0.65 * pad, icon_w, 1.4 * px), glyph, accent, px * 1.1)
            painter.restore()
            text.paint(painter, x + pad * 0.7 + icon_w + 6, top + 0.65 * pad)

        return height, paint


def split_relations(rhs: mx.Node) -> list[mx.Node]:
    """Cut '= a = b ≈ c' into ['= a', '= b', '≈ c'] at its top-level relations."""
    items = rhs.items if isinstance(rhs, mx.Row) else (rhs,)
    chunks: list[list[mx.Node]] = []
    for item in items:
        if isinstance(item, mx.Op) and item.kind == "rel" and chunks and chunks[-1]:
            chunks.append([item])
        else:
            if not chunks:
                chunks.append([])
            chunks[-1].append(item)
    return [mx.row(*chunk) for chunk in chunks if chunk]
