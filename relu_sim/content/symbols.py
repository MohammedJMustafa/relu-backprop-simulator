"""The symbols of the homework (x₁, w₅, h_out, E, ...) and how activations are written.

Symbols carry a colour role so that an equation's x₁, f₁ or h_out matches the
colour of that node in the network diagram.
"""

from __future__ import annotations

from ..core.network import PARAM_NAMES
from .mathexpr import (COMMA, EQ, GT, LE, MINUS, PLUS, QQUAD, Cases, Node, Script, Text,
                       frac, num, paren, row, sup, var)

X1 = var("x", "1", "input")
X2 = var("x", "2", "input")
F1 = var("f", "1", "hidden")
H1 = var("h", "1", "hidden")
F2 = var("f", "2", "hidden")
H2 = var("h", "2", "hidden")
F_OUT = var("f", "out", "output")
H_OUT = var("h", "out", "output")
E = var("E", role="error")
Y = var("y", "true", "target")
ETA = Text("η", "italic")
LR = Text("lr", "italic")
Z = Text("z", "italic")

PARAM: dict[str, Node] = {name: var(name[0], name[1:]) for name in PARAM_NAMES}

NODE: dict[str, Node] = {
    "x1": X1, "x2": X2, "f1": F1, "h1": H1, "f2": F2, "h2": H2,
    "f_out": F_OUT, "h_out": H_OUT, "E": E, "y": Y,
}

NODE_ROLE: dict[str, str] = {
    "x1": "input", "x2": "input", "f1": "hidden", "h1": "hidden", "f2": "hidden", "h2": "hidden",
    "f_out": "output", "h_out": "output", "E": "error", "y": "target",
}


def param_new(name: str) -> Node:
    """w₅ with the superscript "new"."""
    return Script(Text(name[0], "italic"), Text(name[1:]), Text("new"))


def value(node_name: str, x: float, parens: bool = False) -> Node:
    """A number coloured like the node it belongs to."""
    return num(x, NODE_ROLE.get(node_name), parens=parens)


class Notation:
    """How one activation function is written in the equations."""

    def __init__(self, key: str):
        self.key = key
        self.is_relu = key == "relu"
        self.name = "ReLU" if self.is_relu else "Sigmoid"

    @property
    def g(self) -> Node:
        return Text("ReLU") if self.is_relu else Text("σ", "italic")

    @property
    def g_prime(self) -> Node:
        return Text("ReLU′") if self.is_relu else row(Text("σ", "italic"), Text("′"))

    def apply(self, argument) -> Node:
        return row(self.g, paren(argument))

    def prime(self, argument) -> Node:
        return row(self.g_prime, paren(argument))

    def evaluation(self, f_value: float, role: str | None = None) -> Node:
        """What g(f) becomes after substitution: max(0, 1.32) or 1/(1 + e^(-1.32))."""
        if self.is_relu:
            return row(Text("max"), paren(num(0), COMMA, num(f_value, role)))
        return frac(num(1), row(num(1), PLUS, sup(Text("e", "italic"), num(-f_value, role))))

    def derivative_symbolic(self, f_symbol: Node, h_symbol: Node) -> Node:
        """ReLU'(f) or, for the sigmoid, h(1 - h)."""
        if self.is_relu:
            return self.prime(f_symbol)
        return row(h_symbol, paren(num(1), MINUS, h_symbol))

    def derivative_substituted(self, f_value: float, h_value: float, role: str | None = None) -> Node:
        """ReLU'(0.76) or 0.6087(1 - 0.6087)."""
        if self.is_relu:
            return self.prime(num(f_value, role))
        return row(num(h_value, role), paren(num(1), MINUS, num(h_value, role)))

    def definition(self) -> Node:
        """ReLU(z) = max(0, z),  ReLU'(z) = {1, z > 0; 0, z <= 0}   (or the sigmoid version)."""
        if self.is_relu:
            cases = Cases(((row(num(1), COMMA), row(Z, GT, num(0))),
                           (row(num(0), COMMA), row(Z, LE, num(0)))))
            return row(self.apply(Z), EQ, Text("max"), paren(num(0), COMMA, Z), COMMA, QQUAD,
                       self.prime(Z), EQ, cases)
        sigma_z = self.apply(Z)
        return row(sigma_z, EQ, frac(num(1), row(num(1), PLUS, sup(Text("e", "italic"), row(Text("−"), Z)))),
                   COMMA, QQUAD, self.prime(Z), EQ, sigma_z, paren(num(1), MINUS, sigma_z))
