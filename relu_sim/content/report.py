"""Documents for the Verification and Activations pages (plain data, like the solution)."""

from __future__ import annotations

from dataclasses import replace

from ..core.gradcheck import GradientCheck
from ..core.network import PARAM_NAMES, Config, run_iteration
from ..core.training import TrainingRun
from ..core.verification import CLOSE, EXACT, MISMATCH, ROUNDED, AnswerCheck
from . import numfmt
from .blocks import Block, Callout, Cell, Heading, Paragraph, Run, Table, cell, inline
from .mathexpr import EQ, MINUS, PLUS, Node, Script, Text, frac, num, paren, partial, row, var
from .solution import comparison_table, sigmoid_explanation
from .symbols import E, ETA, F1, F2, F_OUT, H1, H2, H_OUT, NODE, NODE_ROLE, PARAM, param_new

P = Text("p", "italic")
EPS = Text("ε", "italic")

STATUS_TEXT = {EXACT: ("✓ exact", "good"), ROUNDED: ("✓ rounded", "good"),
               CLOSE: ("✓ ±1 in last digit", "warn"), MISMATCH: ("✗ mismatch", "bad")}


def answer_label(key: str) -> Node:
    """The symbol of a checked quantity, e.g. 'grad.w5' -> dE/dw5."""
    kind, name = key.split(".", 1)
    if kind == "relu":
        if name == "E":
            return E
        if name == "dE_dhout":
            return partial(E, H_OUT)
        gates = {"gate_out": (H_OUT, F_OUT), "gate1": (H1, F1), "gate2": (H2, F2)}
        if name in gates:
            return partial(*gates[name])
        return NODE[name]
    if kind == "grad":
        return partial(E, PARAM[name])
    if kind == "new":
        return param_new(name)
    if kind == "verify":
        if name == "E":
            return Script(Text("E", "italic", "error"), None, Text("new"))
        sub = "out" if name.endswith("_out") else name[1:]
        return var(name[0], sub, NODE_ROLE[name], sup=Text("new"))
    if kind == "sigmoid":
        if name == "h_out":
            return H_OUT
        if name == "E":
            return E
        if name == "dE_dw1":
            return partial(E, PARAM["w1"])
        return Script(Text("E", "italic", "error"), None, Text("new"))
    raise KeyError(key)


def computed_text(value: float) -> str:
    if numfmt.is_exact(value):
        return numfmt.fmt(value)
    return numfmt.fmt_fixed(value, 6)


def verification_document(checks: list[AnswerCheck], gradients: list[GradientCheck], config: Config) -> tuple[Block, ...]:
    ok = sum(1 for c in checks if c.ok)
    close = [c for c in checks if c.status == CLOSE]
    blocks: list[Block] = [
        Heading("Homework answer check"),
        Paragraph(inline("Every value printed in the Homework 3 solution is recomputed by the simulator. Exact "
                         "values must agree to 10⁻⁹; rounded values must agree to the printed number of "
                         "decimals.")),
    ]
    if ok == len(checks):
        summary = f"All {len(checks)} printed values are reproduced."
        if close:
            names = ", ".join(c.answer.printed for c in close)
            summary += (f" {len(checks) - len(close)} match exactly or after rounding; {names} is one unit away in its "
                        "last digit because the lecture example rounded its intermediate values (exact: 0.608654).")
        blocks.append(Callout("success", inline(summary), "Verified"))
    else:
        blocks.append(Callout("warning", inline(f"{len(checks) - ok} of {len(checks)} values do not match."), "Check"))

    rows: list[tuple[Cell, ...]] = []
    groups: set[int] = set()
    section = None
    for check in checks:
        if check.answer.section != section:
            section = check.answer.section
            groups.add(len(rows))
            rows.append((cell(section, bold=True), cell(""), cell(""), cell(""), cell("")))
        text, role = STATUS_TEXT[check.status]
        rows.append((
            cell(answer_label(check.answer.key)),
            cell(check.answer.printed),
            cell(computed_text(check.computed)),
            cell(numfmt.fmt_sci(check.difference, 2) if check.difference > 1e-12 else "0"),
            cell(text, role=role, bold=True),
        ))
    blocks.append(Table(header=(cell("Quantity"), cell("PDF"), cell("Simulator"), cell("|Difference|"), cell("Result")),
                        rows=tuple(rows), align=("l", "r", "r", "r", "l"), group_rows=frozenset(groups)))
    if close:
        blocks.append(Paragraph(inline(close[0].answer.note), "small"))

    blocks += [
        Heading("Numerical gradient check"),
        Paragraph(inline("For the current values (", config.act.name, ", ", row(ETA, EQ, num(config.lr)),
                         "), every gradient found by backpropagation is compared with the central difference ",
                         frac(row(E, paren(P, PLUS, EPS), MINUS, E, paren(P, MINUS, EPS)), row(num(2), EPS)),
                         " with ", row(EPS, EQ, Text("10⁻⁶")), ".")),
    ]
    if any(g.near_kink for g in gradients):
        blocks.append(Callout("warning", inline("A pre-activation is within 10⁻⁴ of 0, where ReLU has a kink "
                                                "and no derivative: the numerical estimate is not reliable there."),
                              "Kink"))
    grad_rows = []
    for g in gradients:
        status = ("✓ agrees", "good") if g.ok and not g.near_kink else (
            ("~ near kink", "warn") if g.near_kink else ("✗ differs", "bad"))
        grad_rows.append((cell(partial(E, PARAM[g.name])), cell(numfmt.fmt_fixed(g.analytic, 8)),
                          cell(numfmt.fmt_fixed(g.numeric, 8)),
                          cell(numfmt.fmt_sci(g.abs_error, 2) if g.abs_error > 0 else "0"),
                          cell(status[0], role=status[1], bold=True)))
    blocks.append(Table(header=(cell("Gradient"), cell("Backpropagation"), cell("Numerical"), cell("|Difference|"),
                                cell("Result")),
                        rows=tuple(grad_rows), align=("l", "r", "r", "r", "l")))
    return tuple(blocks)


def signed(x: float, digits: int = 4) -> str:
    if x == 0:
        return "0"
    return ("+" if x > 0 else "−") + numfmt.fmt_compact(abs(x), digits)


def iteration_table(run: TrainingRun, k: int) -> tuple[Block, ...]:
    """The parameters used at iteration ``k`` of a training run and how far they moved since iteration 1."""
    params, first = run.params[k - 1], run.params[0]
    rows = tuple((cell(PARAM[name]), cell(numfmt.fmt_compact(params[name], 5)),
                  cell(signed(params[name] - first[name]), role="muted" if params[name] == first[name] else None))
                 for name in PARAM_NAMES)
    return (Table(header=(cell("Parameter"), cell("Value"), cell("Change since iteration 1")), rows=rows,
                  align=("l", "r", "r")),)


def activation_document(config: Config) -> tuple[Block, ...]:
    relu = run_iteration(replace(config, activation="relu"))
    sigmoid = run_iteration(replace(config, activation="sigmoid"))
    blocks: list[Block] = [comparison_table(config), sigmoid_explanation()]
    g_relu, g_sig = relu.grads.w1, sigmoid.grads.w1
    if g_relu != 0 and g_sig != 0:
        ratio = abs(g_relu) / abs(g_sig)
        blocks.append(Callout("info", inline(
            "With these values ", partial(E, PARAM["w1"]), f" is {ratio:.1f}× larger with ReLU ("
            f"{numfmt.fmt_compact(g_relu, 4)} against {numfmt.fmt_compact(g_sig, 4)}): each sigmoid layer multiplies "
            "the gradient by at most 0.25."), "Gradient size"))
    blocks.append(Paragraph(inline(
        Run("Dead ReLU. ", bold=True),
        "ReLU′(z) = 0 for z ≤ 0, so a neuron whose pre-activation is not positive passes no gradient "
        "back and its weights stop learning. Try the “Dead hidden neuron” and “Dead output "
        "neuron” presets.")))
    return tuple(blocks)
