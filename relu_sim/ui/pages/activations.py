"""ReLU against the sigmoid: the functions, their derivatives and the gradients they produce."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from PySide6.QtWidgets import QGridLayout, QVBoxLayout, QWidget

from ...content.report import activation_document
from ...core.activations import RELU, SIGMOID
from ...core.network import PARAM_NAMES, run_iteration
from ..charts import ChartCanvas
from ..state import AppState
from ..theme import Theme
from ..widgets.common import Card, PageHeader
from ..widgets.document_view import DocumentView

POINT_LABELS = {"f1": "$f_1$", "f2": "$f_2$", "f_out": "$f_{out}$"}


class ActivationsPage(QWidget):
    def __init__(self, state: AppState, parent: QWidget | None = None):
        super().__init__(parent)
        self.state = state
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)
        layout.addWidget(PageHeader("ReLU vs sigmoid",
                                    "Why the ReLU network of the homework learns so much faster than the sigmoid "
                                    "network of the lecture."))
        grid = QGridLayout()
        grid.setSpacing(14)
        self.function_chart = ChartCanvas(self._draw_functions, state.theme)
        self.derivative_chart = ChartCanvas(self._draw_derivatives, state.theme)
        self.gradient_chart = ChartCanvas(self._draw_gradients, state.theme)
        for chart, row, col in ((self.function_chart, 0, 0), (self.derivative_chart, 0, 1),
                                (self.gradient_chart, 1, 0)):
            card = Card(None, padding=12)
            card.outer.addWidget(chart)
            grid.addWidget(card, row, col)
        self.text_card = Card("Comparison (section 6)", padding=16)
        self.document = DocumentView(state.theme, "card")
        self.text_card.outer.addWidget(self.document, 1)
        grid.addWidget(self.text_card, 1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        layout.addLayout(grid, 1)
        state.configChanged.connect(self.reload)
        self.reload()

    def reload(self) -> None:
        self.document.set_blocks(activation_document(self.state.config), keep_scroll=True)
        for chart in (self.function_chart, self.derivative_chart, self.gradient_chart):
            chart.invalidate()

    def _points(self):
        """(name, z, activation) for every pre-activation of the current values, for both activations."""
        out = []
        for act in (RELU, SIGMOID):
            fwd = run_iteration(replace(self.state.config, activation=act.key)).forward
            for name in ("f1", "f2", "f_out"):
                out.append((name, getattr(fwd, name), act))
        return out

    def _z_range(self) -> np.ndarray:
        zs = [z for _, z, _ in self._points()]
        low = min(-4.0, min(zs) - 0.5)
        high = max(4.0, max(zs) + 0.5)
        return np.linspace(low, high, 600)

    def _colors(self, theme: Theme) -> dict[str, str]:
        return {"relu": theme.chart_relu, "sigmoid": theme.chart_sigmoid}

    def _info_box(self, ax, theme: Theme, value, kind: str) -> None:
        """List the marked points per activation (instead of labels that would overlap)."""
        colors = self._colors(theme)
        points = self._points()
        y = 0.05 if kind == "inputs" else 0.5      # an empty band of each chart
        for act in (SIGMOID, RELU):
            parts = [f"{POINT_LABELS[name]} {value(act, z):.3g}" for name, z, a in points if a is act]
            title = ("$\\sigma'$" if act is SIGMOID else "ReLU$'$") if kind == "slopes" else act.name
            ax.text(0.98, y, f"{title}:   " + "    ".join(parts), transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=9, color=colors[act.key],
                    bbox=dict(boxstyle="round,pad=0.45", fc=theme.surface2, ec=theme.border, lw=0.8))
            y += 0.11

    def _draw_functions(self, figure, theme: Theme) -> None:
        ax = figure.add_subplot(111)
        z = self._z_range()
        colors = self._colors(theme)
        ax.plot(z, np.maximum(z, 0), color=colors["relu"], lw=2.4, label="ReLU$(z) = \\max(0, z)$")
        ax.plot(z, 1 / (1 + np.exp(-z)), color=colors["sigmoid"], lw=2.4, label="$\\sigma(z) = 1/(1+e^{-z})$")
        for name, value, act in self._points():
            ax.plot([value], [act(value)], "o", color=colors[act.key], ms=6.5, mec=theme.surface, mew=1.5, zorder=5)
        self._info_box(ax, theme, lambda act, z: z, "inputs")
        ax.axhline(0, color=theme.border2, lw=0.9)
        ax.axvline(0, color=theme.border2, lw=0.9)
        ax.set_ylim(-0.4, max(2.2, float(max(act(v) for _, v, act in self._points())) + 0.4))
        ax.set_title("Activation $g(z)$ with the neurons' inputs marked")
        ax.set_xlabel("$z$")
        ax.legend(loc="upper left")

    def _draw_derivatives(self, figure, theme: Theme) -> None:
        ax = figure.add_subplot(111)
        z = self._z_range()
        colors = self._colors(theme)
        negative, positive = z[z <= 0], z[z > 0]
        ax.plot(negative, np.zeros_like(negative), color=colors["relu"], lw=2.4, label="ReLU$'(z)$")
        ax.plot(positive, np.ones_like(positive), color=colors["relu"], lw=2.4)
        ax.plot([0], [0], "o", color=colors["relu"], ms=5.5, zorder=5)
        ax.plot([0], [1], "o", mfc=theme.surface, mec=colors["relu"], ms=5.5, mew=1.6, zorder=5)
        s = 1 / (1 + np.exp(-z))
        ax.plot(z, s * (1 - s), color=colors["sigmoid"], lw=2.4, label="$\\sigma'(z) = \\sigma(z)(1-\\sigma(z))$")
        ax.axhline(0.25, color=theme.muted, lw=1.1, ls="--")
        ax.annotate("max $\\sigma' = 0.25$", (z[0], 0.25), xytext=(4, 5), textcoords="offset points", fontsize=9,
                    color=theme.muted)
        for name, value, act in self._points():
            ax.plot([value], [act.prime(value)], "o", color=colors[act.key], ms=6.5, mec=theme.surface, mew=1.5,
                    zorder=6)
        self._info_box(ax, theme, lambda act, z: act.prime(z), "slopes")
        ax.set_ylim(-0.12, 1.25)
        ax.set_title("Derivative $g'(z)$: the factor applied to the gradient")
        ax.set_xlabel("$z$")
        ax.legend(loc="center left")

    def _draw_gradients(self, figure, theme: Theme) -> None:
        ax = figure.add_subplot(111)
        colors = self._colors(theme)
        relu = run_iteration(replace(self.state.config, activation="relu")).grads
        sigmoid = run_iteration(replace(self.state.config, activation="sigmoid")).grads
        x = np.arange(len(PARAM_NAMES))
        width = 0.38
        floor = 1e-6
        r = [max(abs(relu[n]), floor) for n in PARAM_NAMES]
        s = [max(abs(sigmoid[n]), floor) for n in PARAM_NAMES]
        ax.bar(x - width / 2, r, width, color=colors["relu"], label="ReLU", zorder=3)
        ax.bar(x + width / 2, s, width, color=colors["sigmoid"], label="Sigmoid", zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(x, [f"${n[0]}_{n[1]}$" for n in PARAM_NAMES])
        ax.set_title("Gradient size $|\\partial E/\\partial p|$ at iteration 1 (log scale)")
        ax.legend(loc="upper left", ncol=2)
        ax.grid(axis="x", visible=False)

    def apply_theme(self, theme: Theme) -> None:
        for chart in (self.function_chart, self.derivative_chart, self.gradient_chart):
            chart.set_theme(theme)
        self.document.apply_theme(theme)
