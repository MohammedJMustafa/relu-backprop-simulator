"""The worked solution of Homework 3 (sections 1-6 of the PDF) for any configuration.

With the homework's values the document reproduces the PDF; with other
values (or the sigmoid) every number and every sentence that depends on
them is recomputed, e.g. a "dead" ReLU is explained when a pre-activation
is not positive.
"""

from __future__ import annotations

from dataclasses import replace

from .. import homework
from ..core.network import BIAS_NAMES, PARAM_NAMES, WEIGHT_NAMES, Config, Iteration, run_iteration
from . import numfmt
from .blocks import (Block, Callout, EqLine, Equation, EquationGrid, Figure, Heading, Paragraph, Run,
                     Table, TitleBlock, cell, inline)
from .diagram import given_snapshot, static_frame
from .mathexpr import (COMMA, DOT, EQ, GT, LE, LT, MINUS, PLUS, QQUAD, QUAD, THIN, Matrix, Node, Script,
                       Text, boxed, chain, half, num, paren, partial, row, sup)
from .symbols import (E, ETA, F1, F2, F_OUT, H1, H2, H_OUT, LR, PARAM, X1, X2, Y, Notation, value)


def build_solution(config: Config, iteration_no: int = 1) -> tuple[Block, ...]:
    it = run_iteration(config)
    nt = Notation(config.activation)
    blocks: list[Block] = [
        TitleBlock(homework.TITLE, f"Gradient Descent with the {nt.name} Activation Function",
                   homework.STUDENT.name, homework.STUDENT.group),
    ]
    blocks += _modified_notice(config, iteration_no)
    blocks += _given(config, nt)
    blocks += _forward_pass(it, nt)
    blocks += _error(it)
    blocks += _backpropagation(it, nt)
    blocks += _gradient_descent(it)
    blocks += _verification(it, nt)
    blocks += _comparison(config)
    return tuple(blocks)


# --------------------------------------------------------------------------- helpers

def difference(a: float, b: float, a_role: str | None = "output", b_role: str | None = "target") -> Node:
    """a - b written with a single sign: '0.76 - 1' or '0.76 + 1' when b is negative."""
    if b < 0:
        return row(num(a, a_role), PLUS, num(-b, b_role))
    return row(num(a, a_role), MINUS, num(b, b_role))


def signed_term(x: float, role: str | None = None) -> tuple[Node, Node]:
    """'+ 0.2' or '- 0.2' for a term added at the end of a sum."""
    return (MINUS, num(-x, role)) if x < 0 else (PLUS, num(x, role))


def weighted_sum(w_a: float, x_a: float, w_b: float, x_b: float, bias: float, role: str) -> Node:
    """(0.6)(1.2) + (0.8)(0.5) + 0.2"""
    return row(num(w_a, parens=True), num(x_a, role, parens=True), PLUS,
               num(w_b, parens=True), num(x_b, role, parens=True), *signed_term(bias))


def assignments(pairs: list[tuple[Node, Node]]) -> Node:
    """x₁ = 1.2,  x₂ = 0.5"""
    items: list[Node] = []
    for i, (symbol, number) in enumerate(pairs):
        if i:
            items += [COMMA, QUAD]
        items += [symbol, EQ, number]
    return row(*items)


def with_comma(line: EqLine) -> EqLine:
    return EqLine(line.lhs, row(line.rhs, COMMA))


def grid_row(*lines: EqLine | None) -> tuple[EqLine | None, ...]:
    """A row of side-by-side equations separated by commas, as in the PDF."""
    real = [i for i, line in enumerate(lines) if line is not None]
    last = real[-1] if real else -1
    return tuple(line if line is None or i == last else with_comma(line) for i, line in enumerate(lines))


def output_error_factor() -> Node:
    return paren(H_OUT, MINUS, Y)


def output_error_number(it: Iteration) -> Node:
    return paren(difference(it.forward.h_out, it.config.sample.y_true))


def fmt4(x: float) -> str:
    return numfmt.fmt_compact(x, 4)


# --------------------------------------------------------------------------- sections

def _modified_notice(config: Config, iteration_no: int) -> list[Block]:
    if iteration_no > 1:
        text = (f"Iteration {iteration_no}: the parameters below are the result of {iteration_no - 1} "
                f"gradient-descent update{'s' if iteration_no > 2 else ''} of the homework network.")
        return [Callout("note", inline(text), "Simulation")]
    if config.is_close(homework.HOMEWORK):
        return []
    if config.is_close(homework.LECTURE):
        text = "The sigmoid version of the network (Lecture Example 1). Switch to ReLU to match the PDF."
    else:
        text = "These results use modified values. Reset to the homework values to reproduce the PDF exactly."
    return [Callout("note", inline(text), "Modified")]


def _given(config: Config, nt: Notation) -> list[Block]:
    p, s = config.params, config.sample
    rows = (
        (cell("Inputs"), cell(assignments([(X1, value("x1", s.x1)), (X2, value("x2", s.x2))]))),
        (cell("Hidden-layer weights"), cell(assignments([(PARAM[n], num(p[n])) for n in WEIGHT_NAMES[:4]]))),
        (cell("Output-layer weights"), cell(assignments([(PARAM[n], num(p[n])) for n in WEIGHT_NAMES[4:]]))),
        (cell("Biases"), cell(assignments([(PARAM[n], num(p[n])) for n in BIAS_NAMES]))),
        (cell("Target"), cell(row(Y, EQ, value("y", s.y_true)))),
        (cell("Learning rate"), cell(row(ETA, EQ, LR, EQ, num(config.lr)))),
        (cell("Error function"), cell(row(E, EQ, half(), sup(output_error_factor(), num(2))))),
        (cell("Activation"), cell(
            "ReLU instead of sigmoid, at " if nt.is_relu else "Sigmoid (lecture example), at ",
            H1, ", ", H2, " and ", H_OUT)),
    )
    caption = inline(Run("Figure 1:", bold=True), " Network with the given weights and biases (",
                     row(Y, EQ, num(s.y_true)), ", ", row(LR, EQ, num(config.lr)), ").")
    return [
        Heading("Given"),
        Table(header=(), rows=rows, align=("l", "l")),
        Equation((EqLine(None, nt.definition()),)),
        Figure("network", static_frame(given_snapshot(config)), caption),
    ]


def _forward_pass(it: Iteration, nt: Notation) -> list[Block]:
    p, s, f = it.config.params, it.config.sample, it.forward
    t5, t6 = p.w5 * f.h1, p.w6 * f.h2
    return [
        Heading(f"1. Forward Pass ({nt.name})"),
        Heading("Hidden neuron 1", 3),
        Equation((
            EqLine(F1, chain(row(PARAM["w1"], X1, PLUS, PARAM["w2"], X2, PLUS, PARAM["b1"]),
                             weighted_sum(p.w1, s.x1, p.w2, s.x2, p.b1, "input"), value("f1", f.f1))),
            EqLine(H1, chain(nt.apply(F1), nt.evaluation(f.f1, "hidden"), value("h1", f.h1))),
        )),
        Heading("Hidden neuron 2", 3),
        Equation((
            EqLine(F2, chain(row(PARAM["w3"], X1, PLUS, PARAM["w4"], X2, PLUS, PARAM["b2"]),
                             weighted_sum(p.w3, s.x1, p.w4, s.x2, p.b2, "input"), value("f2", f.f2))),
            EqLine(H2, chain(nt.apply(F2), nt.evaluation(f.f2, "hidden"), value("h2", f.h2))),
        )),
        Heading("Output neuron", 3),
        Equation((
            EqLine(F_OUT, chain(row(PARAM["w5"], H1, PLUS, PARAM["w6"], H2, PLUS, PARAM["b3"]),
                                weighted_sum(p.w5, f.h1, p.w6, f.h2, p.b3, "hidden"),
                                row(num(t5), *signed_term(t6), *signed_term(p.b3)),
                                value("f_out", f.f_out))),
            EqLine(H_OUT, chain(nt.apply(F_OUT), nt.evaluation(f.f_out, "output"), value("h_out", f.h_out))),
        )),
    ]


def _error(it: Iteration) -> list[Block]:
    f, y = it.forward, it.config.sample.y_true
    return [
        Heading("2. Error Calculation"),
        Equation((EqLine(E, chain(row(half(), sup(output_error_factor(), num(2))),
                                  row(half(), sup(paren(difference(f.h_out, y)), num(2))),
                                  row(half(), sup(paren(num(f.h_out - y)), num(2))),
                                  boxed(value("E", f.error)))),)),
    ]


def gate_sentence(it: Iteration, nt: Notation) -> list:
    """The sentence before the activation derivatives in 3.1 (adapted to dead ReLUs)."""
    if not nt.is_relu:
        return ["With the sigmoid, each activation derivative is ",
                row(nt.prime(Text("f", "italic")), EQ, Text("h", "italic"), paren(num(1), MINUS, Text("h", "italic"))),
                ", the slope of the sigmoid at the neuron's input:"]
    f = it.forward
    parts: list = ["With ReLU, the sigmoid terms ",
                   row(H_OUT, paren(num(1), MINUS, H_OUT)), ", ", row(H1, paren(num(1), MINUS, H1)), " and ",
                   row(H2, paren(num(1), MINUS, H2)), " are replaced by ReLU′. "]
    dead = [(name, symbol) for name, symbol, z in (("f_out", F_OUT, f.f_out), ("f1", F1, f.f1), ("f2", F2, f.f2))
            if z <= 0]
    if not dead:
        parts.append("All pre-activations are positive, so each of these derivatives equals 1:")
        return parts
    for name, symbol in dead:
        parts += [row(symbol, LE, num(0)), " so ", row(nt.prime(symbol), EQ, num(0)),
                  ": no gradient passes through this neuron (a “dead” ReLU). "]
    if len(dead) < 3:
        parts.append("The other pre-activations are positive, so their derivative equals 1:")
    return parts


def _gate_equations(it: Iteration, nt: Notation) -> Block:
    f, b = it.forward, it.backward
    items = ((H_OUT, F_OUT, f.f_out, f.h_out, b.dhout_dfout, "output"),
             (H1, F1, f.f1, f.h1, b.dh1_df1, "hidden"),
             (H2, F2, f.f2, f.h2, b.dh2_df2, "hidden"))
    if nt.is_relu:
        return EquationGrid((grid_row(*(EqLine(partial(h, fs), chain(nt.derivative_substituted(z, a, role), num(g)))
                                        for h, fs, z, a, g, role in items)),))
    return Equation(tuple(EqLine(partial(h, fs), chain(nt.derivative_symbolic(fs, h),
                                                       nt.derivative_substituted(z, a, role), num(g)))
                          for h, fs, z, a, g, role in items))


def _local_derivative_grid(it: Iteration) -> Block:
    p, s, f = it.config.params, it.config.sample, it.forward
    one = chain(num(1))
    return EquationGrid((
        grid_row(EqLine(partial(F_OUT, PARAM["w5"]), chain(H1, value("h1", f.h1))),
                 EqLine(partial(F_OUT, PARAM["w6"]), chain(H2, value("h2", f.h2))),
                 EqLine(partial(F_OUT, PARAM["b3"]), one)),
        grid_row(EqLine(partial(F_OUT, H1), chain(PARAM["w5"], num(p.w5))),
                 EqLine(partial(F_OUT, H2), chain(PARAM["w6"], num(p.w6))), None),
        grid_row(EqLine(partial(F1, PARAM["w1"]), chain(X1, value("x1", s.x1))),
                 EqLine(partial(F1, PARAM["w2"]), chain(X2, value("x2", s.x2))),
                 EqLine(partial(F1, PARAM["b1"]), one)),
        grid_row(EqLine(partial(F2, PARAM["w3"]), chain(X1, value("x1", s.x1))),
                 EqLine(partial(F2, PARAM["w4"]), chain(X2, value("x2", s.x2))),
                 EqLine(partial(F2, PARAM["b2"]), one)),
    ))


INPUT_OF = {  # parameter -> (symbol multiplying it, node name of that value or None for a bias)
    "w1": (X1, "x1"), "w2": (X2, "x2"), "w3": (X1, "x1"), "w4": (X2, "x2"),
    "w5": (H1, "h1"), "w6": (H2, "h2"), "b1": (None, None), "b2": (None, None), "b3": (None, None),
}
HIDDEN_OF = {"w1": 1, "w2": 1, "b1": 1, "w3": 2, "w4": 2, "b2": 2}


def gradient_lines(it: Iteration, nt: Notation, name: str, split: bool = True) -> tuple[EqLine, ...]:
    """The PDF's chain-rule derivation of dE/d(name), numbers substituted and the answer boxed."""
    p, s, f, b = it.config.params, it.config.sample, it.forward, it.backward
    symbol, source = INPUT_OF[name]
    if source is None:
        last_symbol, last_value = row(DOT, num(1)), num(1, parens=True)
    else:
        input_value = s.x1 if source == "x1" else s.x2 if source == "x2" else getattr(f, source)
        last_symbol, last_value = symbol, value(source, input_value, parens=True)

    lhs = partial(E, PARAM[name])
    hidden = HIDDEN_OF.get(name)
    if hidden is None:     # output layer: three factors
        fracs = row(partial(E, H_OUT), THIN, partial(H_OUT, F_OUT), THIN, partial(F_OUT, PARAM[name]))
        symbolic = row(output_error_factor(), THIN, nt.derivative_symbolic(F_OUT, H_OUT), THIN, last_symbol)
        numbers = row(output_error_number(it), num(b.dhout_dfout, parens=True), last_value)
    else:                  # hidden layer: five factors
        h, fs = (H1, F1) if hidden == 1 else (H2, F2)
        w_name = "w5" if hidden == 1 else "w6"
        gate = b.dh1_df1 if hidden == 1 else b.dh2_df2
        fracs = row(partial(E, H_OUT), THIN, partial(H_OUT, F_OUT), THIN, partial(F_OUT, h), THIN,
                    partial(h, fs), THIN, partial(fs, PARAM[name]))
        symbolic = row(output_error_factor(), THIN, nt.derivative_symbolic(F_OUT, H_OUT), THIN, PARAM[w_name],
                       THIN, nt.derivative_symbolic(fs, h), THIN, last_symbol)
        numbers = row(output_error_number(it), num(b.dhout_dfout, parens=True), num(p[w_name], parens=True),
                      num(gate, parens=True), last_value)
    answer = boxed(num(it.grads[name]))
    if hidden is None or not split:
        return (EqLine(lhs, chain(fracs, symbolic, numbers, answer)),)
    return (EqLine(lhs, chain(fracs, symbolic)), EqLine(None, chain(numbers, answer)))


def _backpropagation(it: Iteration, nt: Notation) -> list[Block]:
    f, y = it.forward, it.config.sample.y_true
    blocks: list[Block] = [
        Heading("3. Backpropagation"),
        Heading("3.1 Local derivatives", 3),
        Equation((EqLine(partial(E, H_OUT), chain(row(H_OUT, MINUS, Y), difference(f.h_out, y),
                                                  num(it.backward.dE_dhout))),)),
        Paragraph(inline(*gate_sentence(it, nt))),
        _gate_equations(it, nt),
        _local_derivative_grid(it),
        Heading("3.2 Output layer: w₅, w₆, b₃", 3),
    ]
    blocks += [Equation(gradient_lines(it, nt, name)) for name in ("w5", "w6", "b3")]
    blocks.append(Heading("3.3 Hidden neuron 1: w₁, w₂, b₁", 3))
    blocks += [Equation(gradient_lines(it, nt, name)) for name in ("w1", "w2", "b1")]
    blocks.append(Heading("3.4 Hidden neuron 2: w₃, w₄, b₂", 3))
    blocks += [Equation(gradient_lines(it, nt, name)) for name in ("w3", "w4", "b2")]
    return blocks


def update_rule() -> Node:
    w = Text("w", "italic")
    b = Text("b", "italic")
    return row(Script(w, None, Text("new")), EQ, Script(w, Text("old")), MINUS, ETA, THIN, partial(E, w), COMMA,
               QQUAD, Script(b, None, Text("new")), EQ, Script(b, Text("old")), MINUS, ETA, THIN, partial(E, b))


def _vector_equation(symbol: str, names: tuple[str, ...], it: Iteration) -> EqLine:
    old = [it.config.params[n] for n in names]
    grad = [it.grads[n] for n in names]
    new = [it.params_new[n] for n in names]

    def column(values: list[float]) -> Matrix:
        decimals = numfmt.common_decimals(values)
        return Matrix(tuple((num(v, decimals=decimals),) for v in values))

    new_column = column(new)
    lhs = Script(Text(symbol, "bold"), None, Text("new"))
    rhs = row(EQ, column(old), MINUS, num(it.config.lr), THIN, column(grad), *chain(new_column).items)
    return EqLine(lhs, rhs)


def _gradient_descent(it: Iteration) -> list[Block]:
    old, grads, new = it.config.params, it.grads, it.params_new
    d_old = numfmt.common_decimals(old.values())
    d_grad = numfmt.common_decimals(grads.values())
    d_new = numfmt.common_decimals(new.values())
    lr = it.config.lr
    rows = tuple(
        (cell(PARAM[n]), cell(num(old[n], decimals=d_old)), cell(num(grads[n], decimals=d_grad)),
         cell(row(num(old[n], decimals=d_old), MINUS, num(lr), paren(num(grads[n], decimals=d_grad)))),
         cell(num(new[n], decimals=d_new)))
        for n in PARAM_NAMES)
    return [
        Heading("4. Gradient Descent"),
        Equation((EqLine(None, row(update_rule(), COMMA, QQUAD, ETA, EQ, num(lr))),)),
        EquationGrid(((_vector_equation("w", WEIGHT_NAMES, it), _vector_equation("b", BIAS_NAMES, it)),)),
        Table(header=(cell("Parameter"), cell("Old value"), cell("Gradient"), cell("Update"), cell("New value")),
              rows=rows, align=("c", "r", "r", "l", "r"),
              caption=inline(Run("Table 1:", bold=True), " Gradients and updated parameters after one iteration.")),
    ]


def error_change_sentence(it: Iteration, homework_values: bool) -> str:
    before, after = it.forward.error, it.forward_new.error
    if after < before:
        text = f"The error drops from {fmt4(before)} to {fmt4(after)} after one iteration"
        text += f" ({numfmt.fmt_percent(-it.error_change)} lower)."
        if homework_values:
            text += " These are the first two points of the ReLU Error vs Iterations curve in the lecture."
        return text
    if after > before:
        return (f"The error rises from {fmt4(before)} to {fmt4(after)}: the learning rate is too large "
                "for this step, so the update overshoots.")
    return (f"The error stays at {fmt4(before)}: every gradient is zero, so the update changes nothing. "
            "With ReLU this happens when the output neuron is dead (f_out ≤ 0).")


def new_error_line(it: Iteration) -> EqLine:
    """E^new = 1/2 (h_out - y_true)^2 ≈ 0.0094 < 0.0288   (or "(unchanged)" for a dead network)."""
    after, before, f = it.forward_new.error, it.forward.error, it.forward_new
    e_new = Script(Text("E", "italic", "error"), None, Text("new"))
    rhs = chain(row(half(), sup(paren(difference(f.h_out, it.config.sample.y_true)), num(2))), value("E", after))
    if after == before:
        tail = (QUAD, Text("(unchanged)"))
    else:
        tail = (LT if after < before else GT, value("E", before))
    return EqLine(e_new, row(*rhs.items, *tail))


def _verification(it: Iteration, nt: Notation) -> list[Block]:
    p, s, f = it.params_new, it.config.sample, it.forward_new
    return [
        Heading("5. Verification"),
        Paragraph(inline(f"Forward pass ({nt.name}) with the updated parameters:")),
        EquationGrid((
            (EqLine(F1, chain(weighted_sum(p.w1, s.x1, p.w2, s.x2, p.b1, "input"), value("f1", f.f1))),
             EqLine(H1, chain(nt.apply(F1), value("h1", f.h1)))),
            (EqLine(F2, chain(weighted_sum(p.w3, s.x1, p.w4, s.x2, p.b2, "input"), value("f2", f.f2))),
             EqLine(H2, chain(nt.apply(F2), value("h2", f.h2)))),
            (EqLine(F_OUT, chain(weighted_sum(p.w5, f.h1, p.w6, f.h2, p.b3, "hidden"), value("f_out", f.f_out))),
             EqLine(H_OUT, chain(nt.apply(F_OUT), value("h_out", f.h_out)))),
        )),
        Equation((new_error_line(it),)),
        Paragraph(inline(error_change_sentence(it, it.config.is_close(homework.HOMEWORK)))),
    ]


def comparison_rows(config: Config) -> tuple[tuple, ...]:
    sig = run_iteration(replace(config, activation="sigmoid"))
    rel = run_iteration(replace(config, activation="relu"))
    return (
        (cell("Output ", H_OUT, " (iteration 1)"), cell(num(sig.forward.h_out)), cell(num(rel.forward.h_out))),
        (cell("Error ", E, " (iteration 1)"), cell(num(sig.forward.error)), cell(num(rel.forward.error))),
        (cell(partial(E, PARAM["w1"])), cell(num(sig.grads.w1)), cell(num(rel.grads.w1))),
        (cell("Error after one update"), cell(num(sig.forward_new.error)), cell(num(rel.forward_new.error))),
    )


def comparison_table(config: Config) -> Table:
    return Table(header=(cell(""), cell("Sigmoid"), cell("ReLU")), rows=comparison_rows(config),
                 align=("l", "r", "r"))


def sigmoid_explanation() -> Paragraph:
    sigma = Notation("sigmoid")
    z = Text("z", "italic")
    return Paragraph(inline(
        "The sigmoid derivative ", row(sigma.prime(z), EQ, sigma.apply(z), paren(num(1), MINUS, sigma.apply(z))),
        " is at most 0.25, so every sigmoid layer shrinks the gradient. ReLU has derivative 1 for positive "
        "inputs, so the gradients are larger and the error falls much faster for the same learning rate."))


def _comparison(config: Config) -> list[Block]:
    nt = Notation(config.activation)
    blocks: list[Block] = [
        Heading("6. Comparison with Sigmoid (Lecture Example 1)" if nt.is_relu else "6. Comparison with ReLU"),
        comparison_table(config),
    ]
    same_network = (config.params.is_close(homework.PARAMS) and config.sample.is_close(homework.SAMPLE)
                    and abs(config.lr - homework.LEARNING_RATE) < 1e-12)
    if same_network:
        blocks.append(Paragraph(inline(
            "Note: the PDF prints 0.6086 for the sigmoid output; the exact value is 0.608654, which rounds "
            "to 0.6087 (the lecture example rounded its intermediate values)."), "small"))
    blocks.append(sigmoid_explanation())
    if config.is_close(homework.HOMEWORK):
        blocks.append(Callout("success", inline(
            "All 40 values printed in the homework PDF are reproduced by this simulator "
            "(see the Verification page)."), "Checked"))
    return blocks
