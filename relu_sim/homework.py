"""The homework itself: the given values, the student and the printed answers.

Everything here is copied from the Homework 3 solution (PDF) so that the
simulator can reproduce and check every number of the solution.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .core.network import Config, Params, Sample


@dataclass(frozen=True)
class Student:
    name: str
    group: str


STUDENT = Student(name="Mohammed Jalal Mustafa", group="A")

TITLE = "Homework 3"
SUBTITLE = "Gradient Descent with the ReLU Activation Function"

PARAMS = Params(w1=0.6, w2=0.8, w3=0.3, w4=0.4, w5=0.6, w6=-0.2, b1=0.2, b2=0.1, b3=0.1)
SAMPLE = Sample(x1=1.2, x2=0.5, y_true=1.0)
LEARNING_RATE = 0.1

HOMEWORK = Config(params=PARAMS, sample=SAMPLE, lr=LEARNING_RATE, activation="relu")
LECTURE = replace(HOMEWORK, activation="sigmoid")


@dataclass(frozen=True)
class Preset:
    key: str
    name: str
    description: str
    config: Config


PRESETS: tuple[Preset, ...] = (
    Preset("homework", "Homework 3 — ReLU",
           "The values given in the homework (default).", HOMEWORK),
    Preset("lecture", "Lecture example — Sigmoid",
           "The same network with the sigmoid activation, as in Lecture Example 1.", LECTURE),
    Preset("dead_hidden", "Dead hidden neuron (b₂ = −1)",
           "f₂ < 0, so ReLU′(f₂) = 0: w₃, w₄ and b₂ receive no gradient.",
           replace(HOMEWORK, params=PARAMS.with_values(b2=-1.0))),
    Preset("dead_output", "Dead output neuron (w₅ = −0.6)",
           "f_out < 0, so h_out = 0 and every gradient is 0: the network cannot learn.",
           replace(HOMEWORK, params=PARAMS.with_values(w5=-0.6))),
)

PRESETS_BY_KEY: dict[str, Preset] = {preset.key: preset for preset in PRESETS}


@dataclass(frozen=True)
class Answer:
    """A value printed in the PDF and where it appears."""

    key: str
    section: str
    printed: str
    exact: bool = True     # False when the PDF shows a rounded value
    note: str = ""

    @property
    def value(self) -> float:
        return float(self.printed.replace("−", "-"))

    @property
    def decimals(self) -> int:
        return len(self.printed.split(".")[1]) if "." in self.printed else 0

    @property
    def unit(self) -> float:
        """One unit in the last printed digit."""
        return 10.0 ** -self.decimals


S1 = "1. Forward pass"
S2 = "2. Error"
S31 = "3.1 Local derivatives"
S32 = "3.2 Output layer"
S33 = "3.3 Hidden neuron 1"
S34 = "3.4 Hidden neuron 2"
S4 = "4. Gradient descent"
S5 = "5. Verification"
S6 = "6. Comparison with sigmoid"

SIGMOID_OUTPUT_NOTE = ("The exact value is 0.608654, which rounds to 0.6087; the lecture "
                       "example rounded its intermediate values.")

ANSWERS: tuple[Answer, ...] = (
    Answer("relu.f1", S1, "1.32"),
    Answer("relu.h1", S1, "1.32"),
    Answer("relu.f2", S1, "0.66"),
    Answer("relu.h2", S1, "0.66"),
    Answer("relu.f_out", S1, "0.76"),
    Answer("relu.h_out", S1, "0.76"),
    Answer("relu.E", S2, "0.0288"),
    Answer("relu.dE_dhout", S31, "−0.24"),
    Answer("relu.gate_out", S31, "1"),
    Answer("relu.gate1", S31, "1"),
    Answer("relu.gate2", S31, "1"),
    Answer("grad.w5", S32, "−0.3168"),
    Answer("grad.w6", S32, "−0.1584"),
    Answer("grad.b3", S32, "−0.24"),
    Answer("grad.w1", S33, "−0.1728"),
    Answer("grad.w2", S33, "−0.072"),
    Answer("grad.b1", S33, "−0.144"),
    Answer("grad.w3", S34, "0.0576"),
    Answer("grad.w4", S34, "0.024"),
    Answer("grad.b2", S34, "0.048"),
    Answer("new.w1", S4, "0.61728"),
    Answer("new.w2", S4, "0.80720"),
    Answer("new.w3", S4, "0.29424"),
    Answer("new.w4", S4, "0.39760"),
    Answer("new.w5", S4, "0.63168"),
    Answer("new.w6", S4, "−0.18416"),
    Answer("new.b1", S4, "0.21440"),
    Answer("new.b2", S4, "0.09520"),
    Answer("new.b3", S4, "0.12400"),
    Answer("verify.f1", S5, "1.3587", exact=False),
    Answer("verify.h1", S5, "1.3587", exact=False),
    Answer("verify.f2", S5, "0.6471", exact=False),
    Answer("verify.h2", S5, "0.6471", exact=False),
    Answer("verify.f_out", S5, "0.8631", exact=False),
    Answer("verify.h_out", S5, "0.8631", exact=False),
    Answer("verify.E", S5, "0.0094", exact=False),
    Answer("sigmoid.h_out", S6, "0.6086", exact=False, note=SIGMOID_OUTPUT_NOTE),
    Answer("sigmoid.E", S6, "0.0766", exact=False),
    Answer("sigmoid.dE_dw1", S6, "−0.0112", exact=False),
    Answer("sigmoid.E_after", S6, "0.0748", exact=False),
)
