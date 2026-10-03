"""The step-by-step simulation of one iteration.

Each step is one line of the homework solution: a short explanation, its
equations, and what the network diagram shows - which values have been
computed so far and which pulses travel along the edges (forwards in the
forward pass, backwards during backpropagation).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.network import Config, Iteration, run_iteration
from .blocks import Block, Callout, EqLine, Equation, Paragraph, Run, inline
from .diagram import DiagramFrame, Pulse, Snapshot, given_snapshot
from .mathexpr import (EQ, GT, LE, MINUS, PLUS, THIN, Node, Script, Text, chain, half, num, paren,
                       partial, row, sup)
from .solution import (difference, error_change_sentence, gradient_lines, new_error_line, output_error_factor,
                       weighted_sum)
from .symbols import (E, ETA, F1, F2, F_OUT, H1, H2, H_OUT, PARAM, X1, X2, Y, Notation, param_new, value)

GIVEN = "given"
FORWARD = "forward"
ERROR = "error"
BACKWARD = "backward"
UPDATE = "update"
VERIFY = "verify"

PHASES: tuple[str, ...] = (GIVEN, FORWARD, ERROR, BACKWARD, UPDATE, VERIFY)
PHASE_LABELS: dict[str, str] = {
    GIVEN: "Given", FORWARD: "Forward pass", ERROR: "Error", BACKWARD: "Backpropagation",
    UPDATE: "Update", VERIFY: "Verification",
}


@dataclass(frozen=True, eq=False)
class SimStep:
    index: int
    key: str
    phase: str
    section: str
    title: str
    blocks: tuple[Block, ...]
    frame: DiagramFrame
    duration: float = 1.0       # relative length of the step's animation


def build_timeline(config: Config, iteration_no: int = 1) -> tuple[SimStep, ...]:
    builder = _TimelineBuilder(run_iteration(config), iteration_no)
    builder.given()
    builder.forward_pass()
    builder.error()
    builder.backpropagation()
    builder.update()
    builder.verification()
    return tuple(builder.steps)


def activation_sentence(nt: Notation, symbol: Node, z: float) -> tuple:
    if not nt.is_relu:
        return inline("The sigmoid squashes ", symbol, " into the interval (0, 1).")
    if z > 0:
        return inline(symbol, " is positive, so ReLU passes it through unchanged.")
    return inline(symbol, " is not positive, so ReLU outputs 0: this neuron is switched off.")


def gate_sentence(nt: Notation, symbol: Node, gate: float) -> tuple:
    if not nt.is_relu:
        return inline("The gradient is multiplied by the slope of the sigmoid at ", symbol,
                      ", which is never larger than 0.25.")
    if gate > 0:
        return inline(row(symbol, GT, num(0)), ", so ", row(nt.g_prime, EQ, num(1)),
                      ": the gradient passes through unchanged.")
    return inline(row(symbol, LE, num(0)), ", so ", row(nt.g_prime, EQ, num(0)),
                  ": the gradient is blocked here (a “dead” ReLU).")


def update_rule_single() -> Node:
    p = Text("p", "italic")
    return row(Script(p, None, Text("new")), EQ, Script(p, Text("old")), MINUS, ETA, THIN, partial(E, p))


class _TimelineBuilder:
    def __init__(self, iteration: Iteration, iteration_no: int):
        self.it = iteration
        self.nt = Notation(iteration.config.activation)
        self.iteration_no = iteration_no
        self.snap: Snapshot = given_snapshot(iteration.config)
        self.steps: list[SimStep] = []

    def add(self, key: str, phase: str, section: str, title: str, blocks, *, nodes=(), edges=(), pulses=(),
            duration: float = 1.0, after: Snapshot | None = None, **reveal) -> None:
        before = self.snap
        if after is None:
            after = before.evolve(**reveal) if reveal else before
        frame = DiagramFrame(before, after, phase, frozenset(nodes), frozenset(edges), tuple(pulses))
        self.steps.append(SimStep(len(self.steps), key, phase, section, title, tuple(blocks), frame, duration))
        self.snap = after

    # ------------------------------------------------------------------ given

    def given(self) -> None:
        cfg, nt = self.it.config, self.nt
        s = cfg.sample
        intro = inline("A 2-2-1 network with inputs ", row(X1, EQ, value("x1", s.x1)), " and ",
                       row(X2, EQ, value("x2", s.x2)), ", target ", row(Y, EQ, value("y", s.y_true)),
                       " and learning rate ", row(ETA, EQ, num(cfg.lr)), ". Every neuron uses ",
                       "ReLU:" if nt.is_relu else "the sigmoid:")
        if self.iteration_no > 1:
            title = f"Iteration {self.iteration_no}: the updated network"
            intro = inline(f"The weights and biases now come from the previous update. ", *intro)
        else:
            title = "The network and the given values"
        blocks = (
            Paragraph(intro),
            Equation((EqLine(None, nt.definition()),)),
            Callout("info", inline("Press ", Run("Play", bold=True), " (Space) to run the iteration, or step "
                                   "through it with the ← and → keys.")),
        )
        self.add("given", GIVEN, "Given", title, blocks, nodes=("x1", "x2"), duration=0.6)

    # ------------------------------------------------------------------ section 1

    def forward_pass(self) -> None:
        it, nt = self.it, self.nt
        p, s, f = it.config.params, it.config.sample, it.forward
        section = "§1 Forward pass"
        neurons = (
            ("1", F1, H1, "f1", "h1", ("w1", "w2", "b1"), "a1", f.f1, f.h1,
             row(PARAM["w1"], X1, PLUS, PARAM["w2"], X2, PLUS, PARAM["b1"]),
             weighted_sum(p.w1, s.x1, p.w2, s.x2, p.b1, "input")),
            ("2", F2, H2, "f2", "h2", ("w3", "w4", "b2"), "a2", f.f2, f.h2,
             row(PARAM["w3"], X1, PLUS, PARAM["w4"], X2, PLUS, PARAM["b2"]),
             weighted_sum(p.w3, s.x1, p.w4, s.x2, p.b2, "input")),
        )
        for label, fs, hs, f_name, h_name, edges, gate_edge, f_val, h_val, symbolic, numbers in neurons:
            self.add(f"f{label}", FORWARD, section, f"Hidden neuron {label}: weighted sum",
                     (Paragraph(inline(f"Neuron {label} multiplies each input by its weight and adds its bias.")),
                      Equation((EqLine(fs, chain(symbolic, numbers, value(f_name, f_val))),))),
                     nodes=("x1", "x2", f_name), edges=edges, pulses=tuple(Pulse(e) for e in edges),
                     values={f_name: f_val})
            self.add(f"h{label}", FORWARD, section, f"Hidden neuron {label}: activation",
                     (Paragraph(activation_sentence(nt, fs, f_val)),
                      Equation((EqLine(hs, chain(nt.apply(fs), nt.evaluation(f_val, "hidden"),
                                                 value(h_name, h_val))),))),
                     nodes=(f_name, h_name), edges=(gate_edge,), pulses=(Pulse(gate_edge),), duration=0.7,
                     values={h_name: h_val})

        self.add("f_out", FORWARD, section, "Output neuron: weighted sum",
                 (Paragraph(inline("The output neuron combines both hidden activations with ", PARAM["w5"], ", ",
                                   PARAM["w6"], " and the bias ", PARAM["b3"], ".")),
                  Equation((EqLine(F_OUT, chain(row(PARAM["w5"], H1, PLUS, PARAM["w6"], H2, PLUS, PARAM["b3"]),
                                                weighted_sum(p.w5, f.h1, p.w6, f.h2, p.b3, "hidden"),
                                                value("f_out", f.f_out))),))),
                 nodes=("h1", "h2", "f_out"), edges=("w5", "w6", "b3"),
                 pulses=(Pulse("w5"), Pulse("w6"), Pulse("b3")), values={"f_out": f.f_out})
        self.add("h_out", FORWARD, section, "Output neuron: activation",
                 (Paragraph(activation_sentence(nt, F_OUT, f.f_out)),
                  Equation((EqLine(H_OUT, chain(nt.apply(F_OUT), nt.evaluation(f.f_out, "output"),
                                                value("h_out", f.h_out))),))),
                 nodes=("f_out", "h_out"), edges=("a_out",), pulses=(Pulse("a_out"),), duration=0.7,
                 values={"h_out": f.h_out})

    # ------------------------------------------------------------------ section 2

    def error(self) -> None:
        f, y = self.it.forward, self.it.config.sample.y_true
        self.add("E", ERROR, "§2 Error", "Squared error",
                 (Paragraph(inline("The output ", row(H_OUT, EQ, value("h_out", f.h_out)),
                                   " is compared with the target ", row(Y, EQ, value("y", y)), ".")),
                  Equation((EqLine(E, chain(row(half(), sup(output_error_factor(), num(2))),
                                            row(half(), sup(paren(difference(f.h_out, y)), num(2))),
                                            value("E", f.error))),))),
                 nodes=("h_out", "E"), edges=("loss",), pulses=(Pulse("loss"),), values={"E": f.error})

    # ------------------------------------------------------------------ section 3

    def backpropagation(self) -> None:
        it, nt = self.it, self.nt
        p, s, f, b, g = it.config.params, it.config.sample, it.forward, it.backward, it.grads

        section = "§3.1 Local derivatives"
        self.add("dE_dhout", BACKWARD, section, "Gradient at the output",
                 (Paragraph(inline("Backpropagation starts at the error and works backwards. The derivative of ",
                                   row(half(), sup(output_error_factor(), num(2))), " with respect to ", H_OUT,
                                   " is:")),
                  Equation((EqLine(partial(E, H_OUT), chain(row(H_OUT, MINUS, Y), difference(f.h_out, s.y_true),
                                                            num(b.dE_dhout))),))),
                 nodes=("E", "h_out"), edges=("loss",),
                 pulses=(Pulse("loss", -1, "gradient", blocked=b.dE_dhout == 0),),
                 node_grads={"h_out": b.dE_dhout})

        self.add("gate_out", BACKWARD, section, "Through the output activation",
                 (Paragraph(gate_sentence(nt, F_OUT, b.dhout_dfout)),
                  Equation((
                      EqLine(partial(H_OUT, F_OUT), chain(nt.derivative_symbolic(F_OUT, H_OUT),
                                                          nt.derivative_substituted(f.f_out, f.h_out, "output"),
                                                          num(b.dhout_dfout))),
                      EqLine(partial(E, F_OUT), chain(row(partial(E, H_OUT), THIN, partial(H_OUT, F_OUT)),
                                                      row(num(b.dE_dhout, parens=True),
                                                          num(b.dhout_dfout, parens=True)),
                                                      num(b.delta_out))),
                  ))),
                 nodes=("h_out", "f_out"), edges=("a_out",),
                 pulses=(Pulse("a_out", -1, "gradient", blocked=b.delta_out == 0),),
                 gates={"f_out": b.dhout_dfout}, node_grads={"f_out": b.delta_out})

        section = "§3.2 Output layer"
        output_steps = (
            ("w5", "h1", inline(PARAM["w5"], " multiplies ", H1, ", so its gradient is the error signal at the "
                                "output neuron times ", H1, ":")),
            ("w6", "h2", inline(PARAM["w6"], " multiplies ", H2, ", so its gradient is the error signal at the "
                                "output neuron times ", H2, ":")),
            ("b3", None, inline("A bias always has the input 1, so the gradient of ", PARAM["b3"],
                                " is the error signal at the output neuron itself:")),
        )
        for name, source, sentence in output_steps:
            nodes = ("f_out",) + ((source,) if source else ())
            self.add(f"grad_{name}", BACKWARD, section, f"Gradient of {_unicode(name)}",
                     (Paragraph(sentence), Equation(gradient_lines(it, nt, name))),
                     nodes=nodes, edges=(name,),
                     pulses=(Pulse(name, -1, "gradient", blocked=g[name] == 0),),
                     param_grads={name: g[name]})

        for hidden in (1, 2):
            self._hidden_neuron(hidden)

    def _hidden_neuron(self, hidden: int) -> None:
        it, nt = self.it, self.nt
        p, f, b, g = it.config.params, it.forward, it.backward, it.grads
        if hidden == 1:
            hs, fs, h_name, f_name, w_name, gate_edge = H1, F1, "h1", "f1", "w5", "a1"
            dE_dh, gate, delta, names = b.dE_dh1, b.dh1_df1, b.delta1, ("w1", "w2", "b1")
            z, a = f.f1, f.h1
        else:
            hs, fs, h_name, f_name, w_name, gate_edge = H2, F2, "h2", "f2", "w6", "a2"
            dE_dh, gate, delta, names = b.dE_dh2, b.dh2_df2, b.delta2, ("w3", "w4", "b2")
            z, a = f.f2, f.h2
        section = f"§3.{2 + hidden} Hidden neuron {hidden}"

        blocks: list[Block] = [
            Paragraph(inline("The error signal at the output travels back through ", PARAM[w_name],
                             f" and the activation of neuron {hidden}.")),
            Equation((
                EqLine(partial(E, hs), chain(row(partial(E, F_OUT), THIN, partial(F_OUT, hs)),
                                             row(num(b.delta_out, parens=True), num(p[w_name], parens=True)),
                                             num(dE_dh))),
                EqLine(partial(hs, fs), chain(nt.derivative_symbolic(fs, hs),
                                              nt.derivative_substituted(z, a, "hidden"), num(gate))),
                EqLine(partial(E, fs), chain(row(partial(E, hs), THIN, partial(hs, fs)),
                                             row(num(dE_dh, parens=True), num(gate, parens=True)), num(delta))),
            )),
        ]
        if nt.is_relu and gate == 0:
            blocks.append(Callout("warning", inline(
                row(nt.g_prime, paren(fs), EQ, num(0)), ", so the gradients of ",
                ", ".join(_unicode(n) for n in names[:2]), " and ", _unicode(names[2]),
                " are all 0: this neuron does not learn in this iteration."), "Dead ReLU"))
        self.add(f"hidden{hidden}", BACKWARD, section, f"Error signal reaching hidden neuron {hidden}", blocks,
                 nodes=("f_out", h_name, f_name), edges=(w_name, gate_edge), duration=1.3,
                 pulses=(Pulse(w_name, -1, "gradient", 0.0, 0.55, blocked=dE_dh == 0),
                         Pulse(gate_edge, -1, "gradient", 0.55, 1.0, blocked=delta == 0)),
                 node_grads={h_name: dE_dh, f_name: delta}, gates={f_name: gate})

        sources = {"w1": "x1", "w2": "x2", "w3": "x1", "w4": "x2"}
        for name in names:
            source = sources.get(name)
            if source:
                sentence = inline("Full chain rule from the error back to ", PARAM[name], ", whose input is ",
                                  X1 if source == "x1" else X2, ":")
            else:
                sentence = inline("The bias ", PARAM[name], " has the input 1, so its gradient is the error "
                                  f"signal at neuron {hidden}:")
            nodes = (f_name,) + ((source,) if source else ())
            self.add(f"grad_{name}", BACKWARD, section, f"Gradient of {_unicode(name)}",
                     (Paragraph(sentence), Equation(gradient_lines(it, nt, name))),
                     nodes=nodes, edges=(name,),
                     pulses=(Pulse(name, -1, "gradient", blocked=g[name] == 0),),
                     param_grads={name: g[name]})

    # ------------------------------------------------------------------ section 4

    def update(self) -> None:
        it = self.it
        old, new, grads, lr = it.config.params, it.params_new, it.grads, it.config.lr
        section = "§4 Gradient descent"

        def lines(names):
            return tuple(EqLine(param_new(n), chain(row(num(old[n]), MINUS, num(lr), paren(num(grads[n]))),
                                                    num(new[n])))
                         for n in names)

        rule = Equation((EqLine(None, update_rule_single()),))
        output_names, hidden_names = ("w5", "w6", "b3"), ("w1", "w2", "w3", "w4", "b1", "b2")
        partly_new = old.with_values(**{n: new[n] for n in output_names})
        self.add("update_out", UPDATE, section, "Update the output layer",
                 (Paragraph(inline("Every parameter takes a small step against its gradient, scaled by the "
                                   "learning rate ", row(ETA, EQ, num(lr)), ":")),
                  rule, Equation(lines(output_names))),
                 nodes=("f_out",), edges=output_names, duration=1.4,
                 after=self.snap.evolve(params=partly_new, old_params=old))
        self.add("update_hidden", UPDATE, section, "Update the hidden layer",
                 (Paragraph(inline("The same rule for the hidden layer. A negative gradient makes a parameter "
                                   "grow, a positive one makes it shrink:")),
                  Equation(lines(hidden_names))),
                 nodes=("f1", "f2"), edges=hidden_names, duration=1.4,
                 after=self.snap.evolve(params=new, old_params=old))

    # ------------------------------------------------------------------ section 5

    def verification(self) -> None:
        it, nt = self.it, self.nt
        p, s, nf = it.params_new, it.config.sample, it.forward_new
        section = "§5 Verification"
        fresh = Snapshot(p, s, it.config.activation,
                         {**self.snap.values, "f1": nf.f1, "h1": nf.h1, "f2": nf.f2, "h2": nf.h2})
        self.add("verify_hidden", VERIFY, section, "Forward pass with the new parameters",
                 (Paragraph(inline("The updated network is run again on the same input:")),
                  Equation((
                      EqLine(F1, chain(weighted_sum(p.w1, s.x1, p.w2, s.x2, p.b1, "input"), value("f1", nf.f1))),
                      EqLine(H1, chain(nt.apply(F1), value("h1", nf.h1))),
                      EqLine(F2, chain(weighted_sum(p.w3, s.x1, p.w4, s.x2, p.b2, "input"), value("f2", nf.f2))),
                      EqLine(H2, chain(nt.apply(F2), value("h2", nf.h2))),
                  ))),
                 nodes=("x1", "x2", "f1", "h1", "f2", "h2"),
                 edges=("w1", "w2", "w3", "w4", "b1", "b2", "a1", "a2"), duration=1.4,
                 pulses=tuple(Pulse(e, 1, "signal", 0.0, 0.6) for e in ("w1", "w2", "w3", "w4", "b1", "b2"))
                 + (Pulse("a1", 1, "signal", 0.6, 1.0), Pulse("a2", 1, "signal", 0.6, 1.0)),
                 after=fresh)

        before, after = it.forward.error, nf.error
        if after < before:
            title, kind = "The error went down", "success"
        elif after > before:
            title, kind = "The error went up", "warning"
        else:
            title, kind = "The error did not change", "warning"
        self.add("verify_error", VERIFY, section, title,
                 (Equation((
                     EqLine(F_OUT, chain(weighted_sum(p.w5, nf.h1, p.w6, nf.h2, p.b3, "hidden"),
                                         value("f_out", nf.f_out))),
                     EqLine(H_OUT, chain(nt.apply(F_OUT), value("h_out", nf.h_out))),
                     new_error_line(it),
                 )),
                  Callout(kind, inline(error_change_sentence(it, False) + " Press “Next iteration” to "
                                       "keep training with the new parameters."))),
                 nodes=("f_out", "h_out", "E"), edges=("w5", "w6", "b3", "a_out", "loss"), duration=1.6,
                 pulses=(Pulse("w5", 1, "signal", 0.0, 0.45), Pulse("w6", 1, "signal", 0.0, 0.45),
                         Pulse("b3", 1, "signal", 0.0, 0.45), Pulse("a_out", 1, "signal", 0.45, 0.75),
                         Pulse("loss", 1, "signal", 0.75, 1.0)),
                 values={"f_out": nf.f_out, "h_out": nf.h_out, "E": after}, previous_error=before)


_SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _unicode(name: str) -> str:
    """w5 -> w₅ (for plain-text titles)."""
    return name[0] + name[1:].translate(_SUBSCRIPT)
