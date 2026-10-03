"""A small math-expression tree for typesetting the homework's equations.

The content modules build these trees; relu_sim/ui/math_painter.py lays them
out and paints them like a textbook (Leibniz fractions, sub/superscripts,
stretchy brackets, column vectors, boxed answers).  ``to_plain`` gives a
readable one-line version used by the tests and by "copy as text".

Roles (``role=``) colour a symbol like its node in the network diagram:
input, hidden, output, error, target, grad, good, bad, muted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Union

from . import numfmt


class Node:
    """Base class of every math node."""


@dataclass(frozen=True)
class Text(Node):
    text: str
    style: str = "upright"          # "upright", "italic" or "bold"
    role: str | None = None


@dataclass(frozen=True)
class Num(Node):
    value: float
    role: str | None = None
    decimals: int | None = None     # fixed decimals (tables, vectors); None = automatic
    parens: bool = False            # written as "(value)": a factor in a product

    @property
    def text(self) -> str:
        if self.decimals is None:
            body = numfmt.fmt(self.value)
        else:
            body = numfmt.fmt_fixed(self.value, self.decimals)
        return f"({body})" if self.parens else body

    @property
    def rounded(self) -> bool:
        """True when the written value is not the exact value."""
        if self.decimals is None:
            return not numfmt.is_exact(self.value)
        return not numfmt.is_exact_fixed(self.value, self.decimals)


@dataclass(frozen=True)
class Op(Node):
    symbol: str
    kind: str = "bin"               # "rel" (=, ≈, <), "bin" (+, −, ·), "punct" (,), "ord" (no space)


@dataclass(frozen=True)
class Row(Node):
    items: tuple[Node, ...]


@dataclass(frozen=True)
class Frac(Node):
    num: Node
    den: Node


@dataclass(frozen=True)
class Script(Node):
    base: Node
    sub: Node | None = None
    sup: Node | None = None


@dataclass(frozen=True)
class Fenced(Node):
    body: Node
    left: str = "("
    right: str = ")"


@dataclass(frozen=True)
class Cases(Node):
    rows: tuple[tuple[Node, Node], ...]      # (value, condition)


@dataclass(frozen=True)
class Matrix(Node):
    rows: tuple[tuple[Node, ...], ...]
    left: str = "["
    right: str = "]"


@dataclass(frozen=True)
class Boxed(Node):
    body: Node


@dataclass(frozen=True)
class Space(Node):
    em: float


@dataclass(frozen=True)
class Styled(Node):
    body: Node
    role: str


MathLike = Union[Node, str, int, float]


# --------------------------------------------------------------------------- builders

def as_node(item: MathLike) -> Node:
    if isinstance(item, Node):
        return item
    if isinstance(item, bool):
        raise TypeError("booleans are not math")
    if isinstance(item, (int, float)):
        return Num(float(item))
    if isinstance(item, str):
        return Text(item)
    raise TypeError(f"cannot turn {item!r} into a math node")


def row(*items: MathLike | None) -> Row:
    """A horizontal sequence; nested rows are flattened and None is skipped."""
    out: list[Node] = []
    for item in items:
        if item is None:
            continue
        node = as_node(item)
        if isinstance(node, Row):
            out.extend(node.items)
        else:
            out.append(node)
    return Row(tuple(out))


def var(name: str, sub: str | None = None, role: str | None = None, sup: MathLike | None = None,
        style: str = "italic") -> Node:
    """A variable such as w₅ or h_out (italic letter, upright subscript)."""
    base = Text(name, style, role)
    if sub is None and sup is None:
        return base
    return Script(base, Text(sub, "upright", role) if sub is not None else None,
                  as_node(sup) if sup is not None else None)


def num(value: float, role: str | None = None, *, parens: bool = False, decimals: int | None = None) -> Num:
    return Num(float(value), role, decimals, parens)


def paren(*items: MathLike | None) -> Fenced:
    return Fenced(row(*items))


def frac(n: MathLike, d: MathLike) -> Frac:
    return Frac(as_node(n), as_node(d))


def sup(base: MathLike, exponent: MathLike) -> Script:
    return Script(as_node(base), None, as_node(exponent))


def boxed(item: MathLike) -> Boxed:
    return Boxed(as_node(item))


def styled(item: MathLike, role: str) -> Styled:
    return Styled(as_node(item), role)


EQ = Op("=", "rel")
APPROX = Op("≈", "rel")
LT = Op("<", "rel")
GT = Op(">", "rel")
LE = Op("≤", "rel")
ARROW = Op("→", "rel")
PLUS = Op("+", "bin")
MINUS = Op("−", "bin")
DOT = Op("·", "bin")
COMMA = Op(",", "punct")
THIN = Space(1 / 6)
QUAD = Space(1.0)
QQUAD = Space(2.0)
PARTIAL = Text("∂")


def partial(numerator: MathLike, denominator: MathLike) -> Frac:
    """The Leibniz derivative ∂a/∂b."""
    return Frac(row(PARTIAL, numerator), row(PARTIAL, denominator))


def half() -> Frac:
    return Frac(Num(1.0), Num(2.0))


# --------------------------------------------------------------------------- queries

def walk(node: Node) -> Iterator[Node]:
    yield node
    if isinstance(node, Row):
        for item in node.items:
            yield from walk(item)
    elif isinstance(node, Frac):
        yield from walk(node.num)
        yield from walk(node.den)
    elif isinstance(node, Script):
        yield from walk(node.base)
        if node.sub is not None:
            yield from walk(node.sub)
        if node.sup is not None:
            yield from walk(node.sup)
    elif isinstance(node, (Fenced, Boxed, Styled)):
        yield from walk(node.body)
    elif isinstance(node, Cases):
        for value, condition in node.rows:
            yield from walk(value)
            yield from walk(condition)
    elif isinstance(node, Matrix):
        for cells in node.rows:
            for cell in cells:
                yield from walk(cell)


def has_rounded_number(node: Node) -> bool:
    return any(isinstance(n, Num) and n.rounded for n in walk(node))


def relation_for(expression: Node) -> Op:
    """"≈" in front of an expression that shows rounded numbers, "=" otherwise."""
    return APPROX if has_rounded_number(expression) else EQ


def chain(*expressions: MathLike) -> Row:
    """'= e1 = e2 = ...' where each relation is '=' or '≈' as appropriate."""
    items: list[Node] = []
    for expression in expressions:
        node = as_node(expression)
        items.extend((relation_for(node), node))
    return row(*items)


# --------------------------------------------------------------------------- plain text

def to_plain(node: Node) -> str:
    text = _plain(node)
    return re.sub(r"\s+", " ", text).strip()


def _plain(node: Node) -> str:
    if isinstance(node, Text):
        return node.text
    if isinstance(node, Num):
        return node.text
    if isinstance(node, Op):
        if node.kind in ("rel", "bin"):
            return f" {node.symbol} "
        if node.kind == "punct":
            return node.symbol + " "
        return node.symbol
    if isinstance(node, Row):
        return "".join(_plain(item) for item in node.items)
    if isinstance(node, Frac):
        n, d = _plain(node.num).strip(), _plain(node.den).strip()
        n = f"({n})" if " " in n else n
        d = f"({d})" if " " in d else d
        return f"{n}/{d}"
    if isinstance(node, Script):
        text = _plain(node.base)
        if node.sub is not None:
            sub = _plain(node.sub).strip()
            text += sub if len(sub) == 1 else "_" + sub
        if node.sup is not None:
            text += "^" + _plain(node.sup).strip()
        return text
    if isinstance(node, Fenced):
        return node.left + _plain(node.body).strip() + node.right
    if isinstance(node, Cases):
        return "{" + "; ".join(f"{_plain(v).strip()} {_plain(c).strip()}" for v, c in node.rows) + "}"
    if isinstance(node, Matrix):
        return "[" + "; ".join(", ".join(_plain(c).strip() for c in cells) for cells in node.rows) + "]"
    if isinstance(node, (Boxed, Styled)):
        return _plain(node.body)
    if isinstance(node, Space):
        return " " if node.em >= 0.5 else ""
    raise TypeError(f"unknown math node {node!r}")
