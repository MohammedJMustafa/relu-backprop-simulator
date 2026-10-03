"""Numerical check of the backpropagation gradients.

Each dE/dp from backpropagation is compared with the central difference
(E(p + eps) - E(p - eps)) / (2 eps).  Close to the kink of ReLU at 0 the
function is not differentiable, so those cases are flagged instead of failed.
"""

from __future__ import annotations

from dataclasses import dataclass

from .network import PARAM_NAMES, Config, Params, forward, run_iteration

KINK_MARGIN = 1e-4


@dataclass(frozen=True)
class GradientCheck:
    name: str
    analytic: float
    numeric: float
    near_kink: bool

    @property
    def abs_error(self) -> float:
        return abs(self.analytic - self.numeric)

    @property
    def rel_error(self) -> float:
        scale = max(abs(self.analytic), abs(self.numeric))
        return 0.0 if scale == 0.0 else self.abs_error / scale

    @property
    def ok(self) -> bool:
        return self.near_kink or self.abs_error <= 1e-7 + 1e-5 * abs(self.analytic)


def numeric_gradients(config: Config, eps: float = 1e-6) -> Params:
    act = config.act
    values = []
    for name in PARAM_NAMES:
        p = config.params[name]
        plus = forward(config.params.with_values(**{name: p + eps}), config.sample, act).error
        minus = forward(config.params.with_values(**{name: p - eps}), config.sample, act).error
        values.append((plus - minus) / (2.0 * eps))
    return Params.from_values(values)


def gradient_check(config: Config, eps: float = 1e-6) -> list[GradientCheck]:
    iteration = run_iteration(config)
    numeric = numeric_gradients(config, eps)
    fwd = iteration.forward
    near_kink = config.activation == "relu" and min(abs(fwd.f1), abs(fwd.f2), abs(fwd.f_out)) < KINK_MARGIN
    return [GradientCheck(name, iteration.grads[name], numeric[name], near_kink) for name in PARAM_NAMES]
