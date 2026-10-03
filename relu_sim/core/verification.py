"""Check every answer printed in the homework PDF against the simulator."""

from __future__ import annotations

from dataclasses import dataclass

from .. import homework
from ..homework import Answer
from .network import run_iteration

EXACT = "exact"
ROUNDED = "rounded"
CLOSE = "close"
MISMATCH = "mismatch"


@dataclass(frozen=True)
class AnswerCheck:
    answer: Answer
    computed: float

    @property
    def difference(self) -> float:
        return abs(self.computed - self.answer.value)

    @property
    def status(self) -> str:
        """exact, rounded (correct rounding), close (within one unit of the last digit) or mismatch."""
        diff = self.difference
        if self.answer.exact:
            return EXACT if diff <= 1e-9 else MISMATCH
        if diff <= 0.5 * self.answer.unit + 1e-12:
            return ROUNDED
        if diff <= self.answer.unit + 1e-12:
            return CLOSE
        return MISMATCH

    @property
    def ok(self) -> bool:
        return self.status != MISMATCH


def computed_answers() -> dict[str, float]:
    """The simulator's value for every key in homework.ANSWERS."""
    relu = run_iteration(homework.HOMEWORK)
    sigmoid = run_iteration(homework.LECTURE)
    fwd, bwd, new = relu.forward, relu.backward, relu.forward_new
    values = {
        "relu.f1": fwd.f1, "relu.h1": fwd.h1, "relu.f2": fwd.f2, "relu.h2": fwd.h2,
        "relu.f_out": fwd.f_out, "relu.h_out": fwd.h_out, "relu.E": fwd.error,
        "relu.dE_dhout": bwd.dE_dhout, "relu.gate_out": bwd.dhout_dfout,
        "relu.gate1": bwd.dh1_df1, "relu.gate2": bwd.dh2_df2,
        "verify.f1": new.f1, "verify.h1": new.h1, "verify.f2": new.f2, "verify.h2": new.h2,
        "verify.f_out": new.f_out, "verify.h_out": new.h_out, "verify.E": new.error,
        "sigmoid.h_out": sigmoid.forward.h_out, "sigmoid.E": sigmoid.forward.error,
        "sigmoid.dE_dw1": sigmoid.grads.w1, "sigmoid.E_after": sigmoid.forward_new.error,
    }
    values.update({f"grad.{name}": value for name, value in relu.grads.items()})
    values.update({f"new.{name}": value for name, value in relu.params_new.items()})
    return values


def check_homework_answers() -> list[AnswerCheck]:
    computed = computed_answers()
    return [AnswerCheck(answer, computed[answer.key]) for answer in homework.ANSWERS]
