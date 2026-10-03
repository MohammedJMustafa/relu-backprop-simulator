"""The homework's 2-2-1 network: forward pass, backpropagation and one update.

    f1    = w1 x1 + w2 x2 + b1        h1    = g(f1)
    f2    = w3 x1 + w4 x2 + b2        h2    = g(f2)
    f_out = w5 h1 + w6 h2 + b3        h_out = g(f_out)
    E     = 1/2 (h_out - y_true)^2

g is the activation function (ReLU in the homework).  One iteration is a
forward pass, the chain rule backwards to get dE/dp for every parameter p,
and a gradient-descent update p_new = p_old - lr * dE/dp.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from .activations import Activation, get_activation

PARAM_NAMES: tuple[str, ...] = ("w1", "w2", "w3", "w4", "w5", "w6", "b1", "b2", "b3")
WEIGHT_NAMES: tuple[str, ...] = PARAM_NAMES[:6]
BIAS_NAMES: tuple[str, ...] = PARAM_NAMES[6:]


@dataclass(frozen=True)
class Params:
    """The nine weights and biases, or a gradient (dE/d of each of them)."""

    w1: float
    w2: float
    w3: float
    w4: float
    w5: float
    w6: float
    b1: float
    b2: float
    b3: float

    def __getitem__(self, name: str) -> float:
        if name not in PARAM_NAMES:
            raise KeyError(name)
        return getattr(self, name)

    def values(self) -> tuple[float, ...]:
        return tuple(getattr(self, name) for name in PARAM_NAMES)

    def items(self) -> tuple[tuple[str, float], ...]:
        return tuple((name, getattr(self, name)) for name in PARAM_NAMES)

    def as_dict(self) -> dict[str, float]:
        return dict(self.items())

    @classmethod
    def from_values(cls, values: Iterable[float]) -> Params:
        values = tuple(float(v) for v in values)
        if len(values) != len(PARAM_NAMES):
            raise ValueError(f"expected {len(PARAM_NAMES)} values, got {len(values)}")
        return cls(*values)

    def with_values(self, **changes: float) -> Params:
        return replace(self, **changes)

    def is_close(self, other: Params, tol: float = 1e-12) -> bool:
        return all(abs(a - b) <= tol for a, b in zip(self.values(), other.values()))


@dataclass(frozen=True)
class Sample:
    """One training example: the two inputs and the target output."""

    x1: float
    x2: float
    y_true: float

    def is_close(self, other: Sample, tol: float = 1e-12) -> bool:
        return (abs(self.x1 - other.x1) <= tol and abs(self.x2 - other.x2) <= tol
                and abs(self.y_true - other.y_true) <= tol)


@dataclass(frozen=True)
class Config:
    """Everything that defines a run: parameters, sample, learning rate and activation."""

    params: Params
    sample: Sample
    lr: float
    activation: str = "relu"

    @property
    def act(self) -> Activation:
        return get_activation(self.activation)

    def is_close(self, other: Config, tol: float = 1e-12) -> bool:
        return (self.activation == other.activation and abs(self.lr - other.lr) <= tol
                and self.params.is_close(other.params, tol) and self.sample.is_close(other.sample, tol))


@dataclass(frozen=True)
class Forward:
    """The values computed by a forward pass."""

    f1: float
    h1: float
    f2: float
    h2: float
    f_out: float
    h_out: float
    error: float

    def value(self, node: str) -> float:
        """Value of a node by its diagram name ("f1", "h_out", "E", ...)."""
        return self.error if node == "E" else getattr(self, node)


@dataclass(frozen=True)
class Backward:
    """The local derivatives and gradients found by backpropagation."""

    dE_dhout: float      # dE/dh_out = h_out - y_true
    dhout_dfout: float   # g'(f_out)
    dh1_df1: float       # g'(f1)
    dh2_df2: float       # g'(f2)
    delta_out: float     # dE/df_out
    dE_dh1: float        # dE/dh1 = dE/df_out * w5
    dE_dh2: float        # dE/dh2 = dE/df_out * w6
    delta1: float        # dE/df1
    delta2: float        # dE/df2
    grads: Params        # dE/dw1 ... dE/db3


@dataclass(frozen=True)
class Iteration:
    """One full gradient-descent iteration, including the verification pass."""

    config: Config
    forward: Forward
    backward: Backward
    params_new: Params
    forward_new: Forward   # forward pass with the updated parameters

    @property
    def grads(self) -> Params:
        return self.backward.grads

    @property
    def config_new(self) -> Config:
        return replace(self.config, params=self.params_new)

    @property
    def error_change(self) -> float:
        """Relative change of the error after the update (-0.675 means 67.5 % lower)."""
        before = self.forward.error
        return 0.0 if before == 0.0 else (self.forward_new.error - before) / before


def forward(params: Params, sample: Sample, act: Activation) -> Forward:
    f1 = params.w1 * sample.x1 + params.w2 * sample.x2 + params.b1
    h1 = act(f1)
    f2 = params.w3 * sample.x1 + params.w4 * sample.x2 + params.b2
    h2 = act(f2)
    f_out = params.w5 * h1 + params.w6 * h2 + params.b3
    h_out = act(f_out)
    diff = h_out - sample.y_true
    return Forward(f1, h1, f2, h2, f_out, h_out, 0.5 * diff * diff)


def backward(params: Params, sample: Sample, act: Activation, fwd: Forward) -> Backward:
    dE_dhout = fwd.h_out - sample.y_true
    dhout_dfout = act.prime(fwd.f_out)
    dh1_df1 = act.prime(fwd.f1)
    dh2_df2 = act.prime(fwd.f2)

    delta_out = dE_dhout * dhout_dfout
    dE_dh1 = delta_out * params.w5
    dE_dh2 = delta_out * params.w6
    delta1 = dE_dh1 * dh1_df1
    delta2 = dE_dh2 * dh2_df2

    grads = Params(
        w1=delta1 * sample.x1, w2=delta1 * sample.x2,
        w3=delta2 * sample.x1, w4=delta2 * sample.x2,
        w5=delta_out * fwd.h1, w6=delta_out * fwd.h2,
        b1=delta1, b2=delta2, b3=delta_out,
    )
    return Backward(dE_dhout, dhout_dfout, dh1_df1, dh2_df2, delta_out,
                    dE_dh1, dE_dh2, delta1, delta2, grads)


def sgd_step(params: Params, grads: Params, lr: float) -> Params:
    """Gradient descent: p_new = p_old - lr * dE/dp for every parameter."""
    return Params.from_values(p - lr * g for p, g in zip(params.values(), grads.values()))


def run_iteration(config: Config) -> Iteration:
    act = config.act
    fwd = forward(config.params, config.sample, act)
    bwd = backward(config.params, config.sample, act, fwd)
    params_new = sgd_step(config.params, bwd.grads, config.lr)
    return Iteration(config, fwd, bwd, params_new, forward(params_new, config.sample, act))
