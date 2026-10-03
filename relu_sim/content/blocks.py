"""Document blocks: the pieces the worked solution and the step explanations are made of.

A document is a tuple of blocks.  relu_sim/ui/document_painter.py lays the
blocks out for a given width and paints them on screen or into a PDF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

from .mathexpr import Node, to_plain


@dataclass(frozen=True)
class Run:
    """A run of ordinary text inside a paragraph."""

    text: str
    bold: bool = False
    italic: bool = False
    role: str | None = None


Inline = Union[Run, Node]


def inline(*parts: Union[str, Inline]) -> tuple[Inline, ...]:
    """Paragraph content from strings (plain text) and math nodes (inline math)."""
    return tuple(Run(part) if isinstance(part, str) else part for part in parts)


class Block:
    """Base class of every document block."""


@dataclass(frozen=True)
class TitleBlock(Block):
    title: str
    subtitle: str
    name: str
    group: str


@dataclass(frozen=True)
class Heading(Block):
    text: str
    level: int = 2      # 2 = section ("1. Forward Pass"), 3 = subsection


@dataclass(frozen=True)
class Paragraph(Block):
    items: tuple[Inline, ...]
    style: str = "body"   # body, muted, small, center


@dataclass(frozen=True)
class EqLine:
    """One line of a display equation, aligned on the relation that starts ``rhs``."""

    lhs: Node | None
    rhs: Node


@dataclass(frozen=True)
class Equation(Block):
    lines: tuple[EqLine, ...]


@dataclass(frozen=True)
class EquationGrid(Block):
    """Equations side by side in aligned columns (stacked when the page is narrow)."""

    rows: tuple[tuple[EqLine | None, ...], ...]


@dataclass(frozen=True)
class Cell:
    items: tuple[Inline, ...]
    role: str | None = None
    bold: bool = False


@dataclass(frozen=True)
class Table(Block):
    header: tuple[Cell, ...]
    rows: tuple[tuple[Cell, ...], ...]
    align: tuple[str, ...]                       # "l", "c" or "r" per column
    caption: tuple[Inline, ...] = ()
    group_rows: frozenset[int] = frozenset()     # rows drawn as a full-width group heading


@dataclass(frozen=True)
class Figure(Block):
    kind: str                                    # "network"
    payload: Any = field(default=None, compare=False)
    caption: tuple[Inline, ...] = ()
    aspect: float = 0.52


@dataclass(frozen=True)
class Callout(Block):
    kind: str                                    # info, success, warning, note
    items: tuple[Inline, ...]
    title: str | None = None


@dataclass(frozen=True)
class Spacer(Block):
    em: float = 1.0


def cell(*parts: Union[str, Inline], role: str | None = None, bold: bool = False) -> Cell:
    return Cell(inline(*parts), role, bold)


def plain_inline(items: tuple[Inline, ...]) -> str:
    return "".join(item.text if isinstance(item, Run) else to_plain(item) for item in items)


def plain_text(blocks: tuple[Block, ...]) -> str:
    """The document as plain text (tests, clipboard)."""
    out: list[str] = []
    for block in blocks:
        if isinstance(block, TitleBlock):
            out += [block.title, block.subtitle, f"Name: {block.name}    Group: {block.group}"]
        elif isinstance(block, Heading):
            out.append(block.text)
        elif isinstance(block, (Paragraph, Callout)):
            text = plain_inline(block.items)
            out.append(f"{block.title}: {text}" if isinstance(block, Callout) and block.title else text)
        elif isinstance(block, Equation):
            out += [_plain_line(line) for line in block.lines]
        elif isinstance(block, EquationGrid):
            for cells in block.rows:
                out.append("    ".join(_plain_line(c) for c in cells if c is not None))
        elif isinstance(block, Table):
            out.append(" | ".join(plain_inline(c.items) for c in block.header))
            for cells in block.rows:
                out.append(" | ".join(plain_inline(c.items) for c in cells))
            if block.caption:
                out.append(plain_inline(block.caption))
        elif isinstance(block, Figure) and block.caption:
            out.append(plain_inline(block.caption))
    return "\n".join(out)


def _plain_line(line: EqLine) -> str:
    lhs = to_plain(line.lhs) if line.lhs is not None else ""
    return (lhs + " " + to_plain(line.rhs)).strip()
