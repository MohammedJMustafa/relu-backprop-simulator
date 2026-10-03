"""Activation functions and their derivatives.

The homework uses ReLU; the lecture example it is compared with uses the
logistic sigmoid.  Each activation provides its value g(z) and its derivative
g'(z) with respect to the pre-activation z.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


def relu(z: float) -> float:
    """ReLU(z) = max(0, z)."""
    return z if z > 0.0 else 0.0


def relu_prime(z: float) -> float:
    """ReLU'(z) = 1 for z > 0 and 0 for z <= 0 (the homework's convention at z = 0)."""
    return 1.0 if z > 0.0 else 0.0


def sigmoid(z: float) -> float:
    """The logistic sigmoid 1 / (1 + e^-z), written so that exp() cannot overflow."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def sigmoid_prime(z: float) -> float:
    """sigma'(z) = sigma(z) (1 - sigma(z)), which is never larger than 0.25."""
    s = sigmoid(z)
    return s * (1.0 - s)


@dataclass(frozen=True)
class Activation:
    key: str
    name: str
    function: Callable[[float], float]
    derivative: Callable[[float], float]

    def __call__(self, z: float) -> float:
        return self.function(z)

    def prime(self, z: float) -> float:
        return self.derivative(z)


RELU = Activation("relu", "ReLU", relu, relu_prime)
SIGMOID = Activation("sigmoid", "Sigmoid", sigmoid, sigmoid_prime)

ACTIVATIONS: dict[str, Activation] = {a.key: a for a in (RELU, SIGMOID)}


def get_activation(key: str) -> Activation:
    try:
        return ACTIVATIONS[key]
    except KeyError:
        raise ValueError(f"unknown activation {key!r}; expected one of {sorted(ACTIVATIONS)}") from None
