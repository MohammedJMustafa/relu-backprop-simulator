"""The sidebar with every value of the homework: presets, activation, inputs,
target, learning rate, weights and biases."""

from __future__ import annotations

from PySide6.QtCore import QLocale, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel, QPushButton,
                               QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from ... import homework
from ...content import mathexpr as mx
from ...content.symbols import ETA, PARAM, X1, X2, Y
from ..icons import icon
from ..state import AppState
from ..theme import Theme
from .common import MathLabel, Segmented, caption

TOLERANCE = 1e-12


class ValueSpin(QDoubleSpinBox):
    """A number box with 6-decimal precision that shows values without trailing zeros.

    The default QDoubleSpinBox keeps only 2 decimals, which would silently
    round weights such as 0.61728 after an update.
    """

    def __init__(self, minimum: float, maximum: float, step: float, parent: QWidget | None = None):
        super().__init__(parent)
        self.setLocale(QLocale.c())
        self.setDecimals(6)
        self.setRange(minimum, maximum)
        self.setSingleStep(step)
        self.setKeyboardTracking(False)
        self.setAccelerated(True)
        self.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumWidth(60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def textFromValue(self, value: float) -> str:
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return "0" if text in ("-0", "") else text

    def valueFromText(self, text: str) -> float:
        try:
            return float(text.replace("−", "-").replace(",", ".").strip())
        except ValueError:
            return self.value()

    def wheelEvent(self, event) -> None:
        # scrolling the sidebar must not change a value by accident
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class ChangedDot(QWidget):
    """A small dot shown next to values that differ from the homework."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedSize(8, 8)
        self._color = QColor("#2F6AF5")
        self._on = False

    def set_state(self, on: bool, color: QColor) -> None:
        self._on, self._color = on, color
        self.setToolTip("Changed from the homework value" if on else "")
        self.update()

    def paintEvent(self, event) -> None:
        if not self._on:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._color)
        painter.drawEllipse(1, 1, 6, 6)


FIELDS = {
    # key: (symbol, minimum, maximum, step, homework value getter, kind)
    "x1": (X1, -5.0, 5.0, 0.1, lambda: homework.SAMPLE.x1, "sample"),
    "x2": (X2, -5.0, 5.0, 0.1, lambda: homework.SAMPLE.x2, "sample"),
    "y_true": (Y, -3.0, 3.0, 0.1, lambda: homework.SAMPLE.y_true, "sample"),
    "lr": (ETA, 0.0001, 5.0, 0.01, lambda: homework.LEARNING_RATE, "lr"),
    **{name: (PARAM[name], -5.0, 5.0, 0.05, (lambda n=name: homework.PARAMS[n]), "param")
       for name in homework.PARAMS.as_dict()},
}


class ParameterPanel(QWidget):
    def __init__(self, state: AppState, parent: QWidget | None = None):
        super().__init__(parent)
        self.state = state
        self.setObjectName("Sidebar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedWidth(300)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll)
        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 20, 16)
        layout.setSpacing(8)

        title = QLabel("Parameters")
        title.setObjectName("CardTitle")
        layout.addWidget(title)
        self.status = QLabel()
        self.status.setObjectName("Muted")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addSpacing(6)

        layout.addWidget(caption("Preset"))
        self.preset = QComboBox()
        self.preset.setCursor(Qt.CursorShape.PointingHandCursor)
        for preset in homework.PRESETS:
            self.preset.addItem(preset.name, preset.key)
            self.preset.setItemData(self.preset.count() - 1, preset.description, Qt.ItemDataRole.ToolTipRole)
        self.preset.addItem("Custom values", "custom")
        layout.addWidget(self.preset)
        layout.addSpacing(6)

        layout.addWidget(caption("Activation function"))
        self.activation = Segmented([("relu", "ReLU"), ("sigmoid", "Sigmoid")],
                                    tooltips={"relu": "ReLU(z) = max(0, z) — the homework",
                                              "sigmoid": "σ(z) = 1/(1 + e⁻ᶻ) — the lecture example"})
        layout.addWidget(self.activation)
        layout.addSpacing(6)

        self._spins: dict[str, ValueSpin] = {}
        self._dots: dict[str, ChangedDot] = {}
        self._labels: list[MathLabel] = []
        groups = (
            ("Inputs and target", ("x1", "x2", "y_true"), 1),
            ("Learning rate", ("lr",), 1),
            ("Hidden-layer weights", ("w1", "w2", "w3", "w4"), 2),
            ("Output-layer weights", ("w5", "w6"), 2),
            ("Biases", ("b1", "b2", "b3"), 1),
        )
        for heading, keys, columns in groups:
            layout.addWidget(caption(heading))
            grid = QGridLayout()
            grid.setHorizontalSpacing(10)
            grid.setVerticalSpacing(7)
            for i, key in enumerate(keys):
                symbol, minimum, maximum, step, _, _ = FIELDS[key]
                label = MathLabel(mx.row(symbol), state.theme, 16.0)
                self._labels.append(label)
                spin = ValueSpin(minimum, maximum, step)
                spin.valueChanged.connect(lambda value, k=key: self._edited(k, value))
                dot = ChangedDot()
                self._spins[key], self._dots[key] = spin, dot
                row, col = divmod(i, columns)
                cell = QHBoxLayout()
                cell.setSpacing(4)
                label.setFixedWidth(28 if key != "y_true" else 40)
                cell.addWidget(label)
                cell.addWidget(spin, 1)
                cell.addWidget(dot)
                grid.addLayout(cell, row, col)
            layout.addLayout(grid)
            layout.addSpacing(6)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.reset = QPushButton(" Reset")
        self.reset.setToolTip("Back to the homework values (Ctrl+R)")
        self.randomize = QPushButton(" Randomize")
        self.randomize.setToolTip("Random weights and biases, to explore other networks")
        for button in (self.reset, self.randomize):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        layout.addSpacing(6)
        note = QLabel("Values from the Homework 3 solution. A dot marks a value that differs from it.")
        note.setObjectName("StatNote")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)

        self.preset.activated.connect(self._preset_chosen)
        self.activation.changed.connect(self.state.set_activation)
        self.reset.clicked.connect(self.state.reset)
        self.randomize.clicked.connect(lambda: self.state.randomize())
        self.state.configChanged.connect(self.refresh)
        self.apply_theme(state.theme)
        self.refresh()

    # ------------------------------------------------------------------ behaviour

    def _preset_chosen(self, index: int) -> None:
        key = self.preset.itemData(index)
        if key and key != "custom":
            self.state.apply_preset(key)
        else:
            self.refresh()

    def _edited(self, key: str, value: float) -> None:
        kind = FIELDS[key][5]
        config = self.state.config
        current = (config.params[key] if kind == "param" else config.lr if kind == "lr"
                   else getattr(config.sample, key))
        if abs(current - value) <= TOLERANCE:
            return
        if kind == "param":
            self.state.set_param(key, value)
        elif kind == "lr":
            self.state.set_lr(value)
        else:
            self.state.set_sample_value(key, value)

    def refresh(self) -> None:
        config = self.state.config
        values = {**config.params.as_dict(), "x1": config.sample.x1, "x2": config.sample.x2,
                  "y_true": config.sample.y_true, "lr": config.lr}
        accent = self.state.theme.c("accent")
        for key, spin in self._spins.items():
            spin.blockSignals(True)
            spin.setValue(values[key])
            spin.blockSignals(False)
            self._dots[key].set_state(abs(values[key] - FIELDS[key][4]()) > TOLERANCE, accent)
        self.activation.set_value(config.activation)
        preset = self.state.matching_preset
        index = self.preset.findData(preset or "custom")
        self.preset.setCurrentIndex(index)
        if self.state.iteration_no > 1:
            self.status.setText(f"Iteration {self.state.iteration_no}: weights after "
                                f"{self.state.iteration_no - 1} update{'s' if self.state.iteration_no > 2 else ''}.")
        elif preset == "homework":
            self.status.setText("The homework's values. Change any of them to explore.")
        elif preset:
            self.status.setText(homework.PRESETS_BY_KEY[preset].description)
        else:
            self.status.setText("Custom values: the solution and the simulation follow them live.")

    def apply_theme(self, theme: Theme) -> None:
        for label in self._labels:
            label.apply_theme(theme)
        self.reset.setIcon(icon("undo", theme.text2, 15))
        self.randomize.setIcon(icon("shuffle", theme.text2, 15))
        self.refresh()
