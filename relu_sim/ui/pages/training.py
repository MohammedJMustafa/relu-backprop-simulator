"""Training: many iterations in a row - the "Error vs Iterations" curve for ReLU and sigmoid."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox, QVBoxLayout,
                               QWidget)

from ...content import numfmt
from ...content.report import iteration_table
from ...core.network import PARAM_NAMES
from ..charts import ChartCanvas
from ..icons import icon
from ..state import AppState
from ..theme import Theme
from ..widgets.common import Card, PageHeader, Segmented, StatTile, caption
from ..widgets.document_view import DocumentView

NAMES = {"relu": "ReLU", "sigmoid": "Sigmoid"}
THRESHOLD = 1e-4
PARAM_COLORS = ("#3E7BFA", "#22A6B3", "#8E5CF0", "#D64FA0", "#2EA05F", "#E07B39", "#7A8699", "#B0864C", "#E04F5F")


class TrainingPage(QWidget):
    exportCsvRequested = Signal()

    def __init__(self, state: AppState, parent: QWidget | None = None):
        super().__init__(parent)
        self.state = state
        self.view = "error"
        self.reveal: int | None = None
        self._cursor_line = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)
        header = PageHeader("Training", "Repeat the gradient-descent iteration and watch the error fall: "
                                        "the lecture's Error vs Iterations curve, ReLU against sigmoid.")
        self.export = QPushButton(" Export CSV")
        header.add_action(self.export)
        layout.addWidget(header)

        tiles = QHBoxLayout()
        tiles.setSpacing(14)
        self.tile_first = StatTile("Error at iteration 1")
        self.tile_last = StatTile("Error at the last iteration")
        self.tile_relu = StatTile("ReLU reaches E < 10⁻⁴")
        self.tile_sigmoid = StatTile("Sigmoid reaches E < 10⁻⁴")
        for tile in (self.tile_first, self.tile_last, self.tile_relu, self.tile_sigmoid):
            tiles.addWidget(tile)
        layout.addLayout(tiles)

        body = QHBoxLayout()
        body.setSpacing(14)
        self.chart_card = Card(None, padding=14)
        controls = QHBoxLayout()
        controls.setSpacing(10)
        self.view_switch = Segmented([("error", "Error"), ("output", "Output"), ("params", "Parameters")])
        controls.addWidget(self.view_switch)
        controls.addStretch(1)
        controls.addWidget(QLabel("Iterations"))
        self.iterations = QSpinBox()
        self.iterations.setRange(2, 5000)
        self.iterations.setValue(30)
        self.iterations.setKeyboardTracking(False)
        self.iterations.setMinimumWidth(80)
        controls.addWidget(self.iterations)
        self.log_scale = QCheckBox("Log scale")
        controls.addWidget(self.log_scale)
        self.animate = QPushButton(" Animate")
        self.animate.setObjectName("Primary")
        controls.addWidget(self.animate)
        self.chart_card.outer.addLayout(controls)
        self.chart = ChartCanvas(self._draw_chart, state.theme)
        self.chart_card.outer.addWidget(self.chart, 1)
        body.addWidget(self.chart_card, 1)

        self.inspector = Card("Inspect an iteration", padding=16)
        self.inspector.setFixedWidth(380)
        self.inspect_label = QLabel()
        self.inspect_label.setObjectName("Muted")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setCursor(Qt.CursorShape.PointingHandCursor)
        self.slider.setToolTip("Drag to inspect any iteration; the dotted line on the chart follows")
        values = QHBoxLayout()
        values.setSpacing(18)
        self.inspect_error = self._value_block(values, "Error E")
        self.inspect_output = self._value_block(values, "Output h_out")
        values.addStretch(1)
        self.table = DocumentView(state.theme, "card")
        self.inspector.outer.addWidget(self.inspect_label)
        self.inspector.outer.addWidget(self.slider)
        self.inspector.outer.addLayout(values)
        self.inspector.outer.addWidget(caption("Weights and biases used"))
        self.inspector.outer.addWidget(self.table, 1)
        body.addWidget(self.inspector)
        layout.addLayout(body, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)

        self.view_switch.changed.connect(self._set_view)
        self.iterations.valueChanged.connect(self.reload)
        self.log_scale.toggled.connect(lambda _: self.chart.invalidate())
        self.animate.clicked.connect(self.start_animation)
        self.slider.valueChanged.connect(self._inspect)
        self.export.clicked.connect(self.exportCsvRequested)
        state.configChanged.connect(self.reload)
        self.apply_theme(state.theme)
        self.reload()

    # ------------------------------------------------------------------ data

    @property
    def count(self) -> int:
        return self.iterations.value()

    def runs(self):
        return {key: self.state.training(key, self.count) for key in ("relu", "sigmoid")}

    def reload(self) -> None:
        self._timer.stop()
        self.reveal = None
        runs = self.runs()
        current = self.state.config.activation
        run = runs[current]
        name = NAMES[current]
        self.tile_first.set(numfmt.fmt_compact(run.errors[0], 4), f"{name}, before any update")
        drop = (1 - run.errors[-1] / run.errors[0]) if run.errors[0] else 0.0
        note = f"{name}, iteration {len(run)}"
        if run.diverged:
            note += " (stopped: values diverged)"
        elif run.errors[0]:
            note += f" · {numfmt.fmt_percent(drop)} lower"
        self.tile_last.set(numfmt.fmt_compact(run.errors[-1], 4), note)
        reach = {key: self.state.iterations_to(key, THRESHOLD) for key in ("relu", "sigmoid")}
        for key, tile in (("relu", self.tile_relu), ("sigmoid", self.tile_sigmoid)):
            k = reach[key]
            color = self.state.theme.chart_relu if key == "relu" else self.state.theme.chart_sigmoid
            if k is None:
                tile.set("never", "within 200,000 iterations (or no gradient at all)", QColor(color))
            else:
                text = f"{numfmt.fmt_int(k)} iteration{'s' if k != 1 else ''}"
                other = reach["sigmoid" if key == "relu" else "relu"]
                note = "same learning rate"
                if key == "sigmoid" and other and k > other:
                    note = f"{numfmt.fmt_int(round(k / other))}× more than ReLU"
                elif key == "relu" and other and other > k:
                    note = f"{numfmt.fmt_int(round(other / k))}× fewer than the sigmoid"
                tile.set(text, note, QColor(color))
        self.slider.blockSignals(True)
        self.slider.setRange(1, self.count)
        self.slider.setValue(min(max(self.slider.value(), 1), self.count))
        self.slider.blockSignals(False)
        self.chart.invalidate()
        self._inspect(self.slider.value())

    def _set_view(self, view: str) -> None:
        self.view = view
        self.chart.invalidate()

    def start_animation(self) -> None:
        self.reveal = 1
        self._timer.start()

    def _tick(self) -> None:
        if self.reveal is None:
            self._timer.stop()
            return
        step = max(1, self.count // 60)
        self.reveal = min(self.count, self.reveal + step)
        self.slider.setValue(self.reveal)
        self.chart.redraw()
        if self.reveal >= self.count:
            self.reveal = None
            self._timer.stop()

    @staticmethod
    def _value_block(row: QHBoxLayout, title: str) -> QLabel:
        column = QVBoxLayout()
        column.setSpacing(0)
        label = QLabel(title)
        label.setObjectName("StatNote")
        value = QLabel("—")
        value.setObjectName("StatValue")
        column.addWidget(label)
        column.addWidget(value)
        row.addLayout(column)
        return value

    def _inspect(self, k: int) -> None:
        runs = self.runs()
        current = self.state.config.activation
        run = runs[current]
        k = min(max(k, 1), len(run))
        self.inspect_label.setText(f"Iteration {k} of {len(run)} · {NAMES[current]} · "
                                   f"{k - 1} update{'s' if k != 2 else ''} applied")
        self.inspect_error.setText(numfmt.fmt_compact(run.errors[k - 1], 4))
        self.inspect_output.setText(numfmt.fmt_compact(run.outputs[k - 1], 4))
        self.table.set_blocks(iteration_table(run, k), keep_scroll=True)
        if self._cursor_line is not None and self.chart.isVisible() and self.reveal is None:
            self._cursor_line.set_xdata([k, k])
            self.chart.refresh()

    # ------------------------------------------------------------------ chart

    def _draw_chart(self, figure, theme: Theme) -> None:
        ax = figure.add_subplot(111)
        runs = self.runs()
        current = self.state.config.activation
        limit = self.reveal or self.count
        colors = {"relu": theme.chart_relu, "sigmoid": theme.chart_sigmoid}
        lr = numfmt.fmt(self.state.config.lr)
        if self.view in ("error", "output"):
            for key in ("sigmoid", "relu"):
                run = runs[key]
                xs = list(run.iterations)[:limit]
                ys = (run.errors if self.view == "error" else run.outputs)[:limit]
                if self.view == "error" and self.log_scale.isChecked():
                    ys = [max(y, 1e-30) for y in ys]
                emphasis = key == current
                ax.plot(xs, ys, color=colors[key], lw=2.6 if emphasis else 1.8, alpha=1.0 if emphasis else 0.9,
                        marker="o" if limit <= 40 else None, ms=4 if emphasis else 3,
                        label=f"{NAMES[key]}  (η = {lr})", zorder=3 if emphasis else 2)
            if self.view == "error":
                ax.set_ylabel("Error  $E$")
                ax.set_title("Error vs iterations")
                if self.log_scale.isChecked():
                    ax.set_yscale("log")
                else:
                    ax.set_ylim(bottom=0)
                relu = runs["relu"]
                if len(relu) >= 2 and limit >= 2:
                    ax.annotate(f"iteration 1: E = {numfmt.fmt_compact(relu.errors[0], 4)}",
                                (1, relu.errors[0]), xytext=(14, 10), textcoords="offset points", fontsize=8.5,
                                color=theme.text2,
                                arrowprops=dict(arrowstyle="-", color=theme.muted, lw=0.8))
                    ax.annotate(f"after one update: {numfmt.fmt_compact(relu.errors[1], 4)}",
                                (2, max(relu.errors[1], 1e-30)), xytext=(18, 18), textcoords="offset points",
                                fontsize=8.5, color=theme.text2,
                                arrowprops=dict(arrowstyle="-", color=theme.muted, lw=0.8))
            else:
                ax.axhline(self.state.config.sample.y_true, color=theme.muted, lw=1.2, ls="--", label="target $y_{true}$")
                ax.set_ylabel("Output  $h_{out}$")
                ax.set_title("Output vs iterations")
            ax.legend(loc="best")
        else:
            run = runs[current]
            xs = list(run.iterations)[:limit]
            for i, name in enumerate(PARAM_NAMES):
                ax.plot(xs, [p[name] for p in run.params[:limit]], color=PARAM_COLORS[i], lw=1.8,
                        ls="-" if name.startswith("w") else (0, (4, 2)), label=f"${name[0]}_{name[1]}$")
            ax.set_ylabel("Value")
            ax.set_title(f"Weights and biases vs iterations ({NAMES[current]})")
            ax.legend(ncol=5, loc="best", fontsize=8.5)
        ax.set_xlabel("Iteration")
        ax.set_xlim(left=0.5, right=max(self.count + 0.5, 2.5))
        k = self.slider.value()
        self._cursor_line = ax.axvline(k, color=theme.accent, lw=1.1, ls=":", alpha=0.9, zorder=1)

    # ------------------------------------------------------------------ theme

    def apply_theme(self, theme: Theme) -> None:
        self.chart.set_theme(theme)
        self.table.apply_theme(theme)
        self.animate.setIcon(icon("play", "#FFFFFF", 14))
        self.export.setIcon(icon("table", theme.text2, 15))
        self.reload()
