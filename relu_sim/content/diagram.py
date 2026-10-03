"""What the network diagram shows at one moment of the simulation (plain data).

A ``DiagramFrame`` holds the snapshot before and after a simulation step plus
what moves during the step (pulses along edges).  The painter in
relu_sim/ui/network_painter.py fades in what is new and moves the pulses as
the step's animation progresses from 0 to 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..core.network import Config, Forward, Params, Sample, forward

NODES: tuple[str, ...] = ("x1", "x2", "f1", "h1", "f2", "h2", "f_out", "h_out", "E")

# edge id -> (source, target); bias edges start at the small bias tags.
EDGES: dict[str, tuple[str, str]] = {
    "w1": ("x1", "f1"), "w2": ("x2", "f1"), "w3": ("x1", "f2"), "w4": ("x2", "f2"),
    "w5": ("h1", "f_out"), "w6": ("h2", "f_out"),
    "b1": ("bias1", "f1"), "b2": ("bias2", "f2"), "b3": ("bias3", "f_out"),
    "a1": ("f1", "h1"), "a2": ("f2", "h2"), "a_out": ("f_out", "h_out"),
    "loss": ("h_out", "E"),
}

GATE_EDGE: dict[str, str] = {"f1": "a1", "f2": "a2", "f_out": "a_out"}


@dataclass(frozen=True, eq=False)
class Snapshot:
    params: Params
    sample: Sample
    activation: str
    values: dict[str, float] = field(default_factory=dict)        # node values shown
    node_grads: dict[str, float] = field(default_factory=dict)    # dE/d(node) shown
    param_grads: dict[str, float] = field(default_factory=dict)   # dE/d(parameter) shown
    gates: dict[str, float] = field(default_factory=dict)         # g'(f) shown at f1, f2, f_out
    old_params: Params | None = None                              # during the update: old -> new labels
    previous_error: float | None = None                           # shown next to E after verification

    def evolve(self, **changes) -> Snapshot:
        """A copy with some fields replaced; dictionaries are merged, not replaced."""
        merged = {}
        for name in ("values", "node_grads", "param_grads", "gates"):
            if name in changes:
                merged[name] = {**getattr(self, name), **changes.pop(name)}
        return replace(self, **merged, **changes)


@dataclass(frozen=True)
class Pulse:
    edge: str
    direction: int = 1        # +1 along the edge (forward pass), -1 against it (backpropagation)
    kind: str = "signal"      # "signal" or "gradient"
    start: float = 0.0        # fraction of the step's animation at which the pulse sets off
    end: float = 1.0          # ... and arrives
    blocked: bool = False     # a gradient stopped by a dead ReLU (g' = 0)


@dataclass(frozen=True, eq=False)
class DiagramFrame:
    before: Snapshot
    after: Snapshot
    phase: str
    active_nodes: frozenset[str] = frozenset()
    active_edges: frozenset[str] = frozenset()
    pulses: tuple[Pulse, ...] = ()


def given_snapshot(config: Config) -> Snapshot:
    """Only the given values: inputs, target and parameters (Figure 1 of the homework)."""
    s = config.sample
    return Snapshot(config.params, s, config.activation, {"x1": s.x1, "x2": s.x2, "y": s.y_true})


def forward_values(fwd: Forward, sample: Sample) -> dict[str, float]:
    return {"x1": sample.x1, "x2": sample.x2, "y": sample.y_true, "f1": fwd.f1, "h1": fwd.h1,
            "f2": fwd.f2, "h2": fwd.h2, "f_out": fwd.f_out, "h_out": fwd.h_out, "E": fwd.error}


def evaluated_snapshot(params: Params, sample: Sample, activation: str) -> Snapshot:
    """Every node value of a forward pass with ``params`` (used by the training view)."""
    config = Config(params, sample, 0.0, activation)
    return Snapshot(params, sample, activation, forward_values(forward(params, sample, config.act), sample))


def static_frame(snapshot: Snapshot, phase: str = "given") -> DiagramFrame:
    return DiagramFrame(snapshot, snapshot, phase)
