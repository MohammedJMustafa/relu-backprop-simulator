"""Many gradient-descent iterations in a row (the "Error vs Iterations" curve).

Iterations are numbered like the homework: iteration 1 is the forward pass
with the given parameters, iteration 2 uses the parameters after one update,
and so on.  The loop works on plain floats, so even 100 000 iterations take
well under a second.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

from .network import Config, Params

DEFAULT_CAP = 200_000


@dataclass(frozen=True)
class TrainingRun:
    activation: str
    lr: float
    errors: tuple[float, ...]     # errors[k - 1] is E at iteration k
    outputs: tuple[float, ...]    # h_out at iteration k
    params: tuple[Params, ...]    # parameters used at iteration k
    diverged: bool                # stopped early because the values were no longer finite

    def __len__(self) -> int:
        return len(self.errors)

    @property
    def iterations(self) -> range:
        return range(1, len(self.errors) + 1)

    def first_below(self, threshold: float) -> int | None:
        for k, error in enumerate(self.errors, start=1):
            if error < threshold:
                return k
        return None


def iterate(config: Config) -> Iterator[tuple[float, float, tuple[float, ...], bool]]:
    """Yield (E, h_out, parameters, learning) for iteration 1, 2, 3, ...

    ``learning`` is False when every gradient is exactly zero (for example a
    dead ReLU output), because the parameters can then never change again.
    The generator stops by itself once the values are no longer finite.
    """
    act = config.act
    g, dg = act.function, act.derivative
    x1, x2, y = config.sample.x1, config.sample.x2, config.sample.y_true
    lr = config.lr
    w1, w2, w3, w4, w5, w6, b1, b2, b3 = config.params.values()
    while True:
        f1 = w1 * x1 + w2 * x2 + b1
        h1 = g(f1)
        f2 = w3 * x1 + w4 * x2 + b2
        h2 = g(f2)
        f_out = w5 * h1 + w6 * h2 + b3
        h_out = g(f_out)
        diff = h_out - y
        error = 0.5 * diff * diff
        if not math.isfinite(error):
            return

        delta_out = diff * dg(f_out)
        delta1 = delta_out * w5 * dg(f1)
        delta2 = delta_out * w6 * dg(f2)
        learning = delta_out != 0.0 or delta1 != 0.0 or delta2 != 0.0
        yield error, h_out, (w1, w2, w3, w4, w5, w6, b1, b2, b3), learning

        w1 -= lr * delta1 * x1
        w2 -= lr * delta1 * x2
        w3 -= lr * delta2 * x1
        w4 -= lr * delta2 * x2
        w5 -= lr * delta_out * h1
        w6 -= lr * delta_out * h2
        b1 -= lr * delta1
        b2 -= lr * delta2
        b3 -= lr * delta_out


def train(config: Config, iterations: int) -> TrainingRun:
    """Record ``iterations`` forward passes (that is, iterations - 1 updates)."""
    errors: list[float] = []
    outputs: list[float] = []
    params: list[Params] = []
    for error, h_out, values, _ in iterate(config):
        errors.append(error)
        outputs.append(h_out)
        params.append(Params(*values))
        if len(errors) >= iterations:
            break
    diverged = len(errors) < iterations
    return TrainingRun(config.activation, config.lr, tuple(errors), tuple(outputs), tuple(params), diverged)


def iterations_to_reach(config: Config, threshold: float, cap: int = DEFAULT_CAP) -> int | None:
    """First iteration whose error is below ``threshold`` (None if not reached within ``cap``)."""
    for k, (error, _, _, learning) in enumerate(iterate(config), start=1):
        if error < threshold:
            return k
        if not learning or k >= cap:
            return None
    return None
