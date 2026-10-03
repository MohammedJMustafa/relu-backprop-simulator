"""Playback controls of the simulation and the clickable phase timeline."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QSizePolicy, QToolButton, QToolTip, QWidget

from ..icons import icon
from ..theme import UI_FAMILIES, UI_FAMILY, Theme
from .common import IconButton, Segmented

SHORT_LABELS = {"given": "Given", "forward": "Forward", "error": "Error", "backward": "Backprop",
                "update": "Update", "verify": "Verify"}


def phase_color(theme: Theme, phase: str) -> QColor:
    return {
        "given": theme.c("muted"), "forward": QColor(theme.signal), "error": theme.role("error"),
        "backward": QColor(theme.gradient), "update": theme.role("good"), "verify": theme.role("target"),
    }.get(phase, theme.c("accent"))


class PhaseTimeline(QWidget):
    """Segments for Given / Forward / Error / Backprop / Update / Verify, filled up to the current step."""

    seekRequested = Signal(int)

    def __init__(self, theme: Theme, parent: QWidget | None = None):
        super().__init__(parent)
        self._theme = theme
        self._phases: list[str] = []
        self._titles: list[str] = []
        self._index = 0
        self._progress = 1.0
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumWidth(320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        return QSize(560, 46)

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        self.update()

    def set_steps(self, phases: list[str], titles: list[str]) -> None:
        self._phases, self._titles = list(phases), list(titles)
        self.update()

    def set_position(self, index: int, progress: float) -> None:
        self._index, self._progress = index, progress
        self.update()

    def _segments(self) -> list[tuple[str, int, int, QRectF]]:
        """(phase, first step, last step, rect) for every phase present."""
        if not self._phases:
            return []
        groups: list[tuple[str, int, int]] = []
        for i, phase in enumerate(self._phases):
            if groups and groups[-1][0] == phase:
                groups[-1] = (phase, groups[-1][1], i)
            else:
                groups.append((phase, i, i))
        gap = 6.0
        available = self.width() - 4 - gap * (len(groups) - 1)
        minimum = 52.0                    # room for the shortest label
        extra = max(available - minimum * len(groups), 0.0)
        count = len(self._phases)
        x = 2.0
        out = []
        for phase, first, last in groups:
            w = minimum + extra * (last - first + 1) / count
            out.append((phase, first, last, QRectF(x, 8, w, 7)))
            x += w + gap
        return out

    def _step_at(self, x: float) -> int | None:
        for phase, first, last, rect in self._segments():
            if rect.left() - 3 <= x <= rect.right() + 3:
                fraction = min(max((x - rect.left()) / max(rect.width(), 1), 0.0), 0.999)
                return first + int(fraction * (last - first + 1))
        return None

    def mouseMoveEvent(self, event) -> None:
        step = self._step_at(event.position().x())
        if step is not None and step < len(self._titles):
            QToolTip.showText(event.globalPosition().toPoint(),
                              f"Step {step + 1}: {self._titles[step]}", self)

    def mousePressEvent(self, event) -> None:
        step = self._step_at(event.position().x())
        if step is not None:
            self.seekRequested.emit(step)

    def paintEvent(self, event) -> None:
        t = self._theme
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont(UI_FAMILY)
        font.setFamilies(UI_FAMILIES)
        font.setPointSizeF(8.2)
        bold = QFont(font)
        bold.setWeight(QFont.Weight.Bold)
        current_phase = self._phases[self._index] if self._phases else ""
        position = self._index + self._progress
        for phase, first, last, rect in self._segments():
            color = phase_color(t, phase)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(t.surface3))
            painter.drawRoundedRect(rect, 3.5, 3.5)
            steps = last - first + 1
            filled = min(max(position - first, 0.0), steps) / steps
            if filled > 0:
                painter.setBrush(color)
                painter.drawRoundedRect(QRectF(rect.x(), rect.y(), max(rect.width() * filled, 7), rect.height()), 3.5, 3.5)
            for k in range(1, steps):
                tick_x = rect.x() + rect.width() * k / steps
                tick = QColor(t.surface) if position - first > k else QColor(t.border2)
                painter.setBrush(tick)
                painter.drawRect(QRectF(tick_x - 0.6, rect.y(), 1.2, rect.height()))
            label = SHORT_LABELS.get(phase, phase)
            active = phase == current_phase
            painter.setFont(bold if active else font)
            painter.setPen(color if active else QColor(t.muted))
            fm = QFontMetricsF(bold if active else font)
            text = fm.elidedText(label, Qt.TextElideMode.ElideRight, rect.width() + 4)
            painter.drawText(QPointF(rect.x(), rect.bottom() + 6 + fm.ascent()), text)
        # marker on the current step
        for phase, first, last, rect in self._segments():
            if first <= self._index <= last:
                steps = last - first + 1
                x = rect.x() + rect.width() * min((self._index - first + self._progress) / steps, 1.0)
                color = phase_color(t, phase)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(t.surface))
                painter.drawEllipse(QPointF(x, rect.center().y()), 7, 7)
                painter.setBrush(color)
                painter.drawEllipse(QPointF(x, rect.center().y()), 4.6, 4.6)


class PlayerBar(QWidget):
    playToggled = Signal()
    restartRequested = Signal()
    previousStep = Signal()
    nextStep = Signal()
    previousPhase = Signal()
    nextPhase = Signal()
    speedChanged = Signal(float)
    nextIteration = Signal()

    SPEEDS = (("0.5", "0.5×"), ("1", "1×"), ("2", "2×"), ("4", "4×"))

    def __init__(self, theme: Theme, parent: QWidget | None = None):
        super().__init__(parent)
        self._theme = theme
        self._playing = False
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        self.restart = IconButton("restart", "Restart the iteration (Home)")
        self.prev_phase = IconButton("previous", "Previous phase (Page Up)")
        self.prev_step = IconButton("chevron_left", "Previous step (←)")
        self.play = QToolButton()
        self.play.setObjectName("PlayButton")
        self.play.setFixedSize(42, 42)
        self.play.setIconSize(QSize(18, 18))
        self.play.setCursor(Qt.CursorShape.PointingHandCursor)
        self.play.setToolTip("Play / pause (Space)")
        self.next_step = IconButton("chevron_right", "Next step (→)")
        self.next_phase = IconButton("next", "Next phase (Page Down)")
        for button in (self.restart, self.prev_phase, self.prev_step):
            layout.addWidget(button)
        layout.addWidget(self.play)
        for button in (self.next_step, self.next_phase):
            layout.addWidget(button)
        layout.addSpacing(10)

        self.timeline = PhaseTimeline(theme)
        layout.addWidget(self.timeline, 1)
        layout.addSpacing(10)

        self.speed = Segmented(list(self.SPEEDS), tooltips={k: f"Animation speed {v}" for k, v in self.SPEEDS})
        self.speed.set_value("1")
        layout.addWidget(self.speed)
        layout.addSpacing(6)
        self.iterate = QPushButton("  Next iteration")
        self.iterate.setObjectName("Primary")
        self.iterate.setCursor(Qt.CursorShape.PointingHandCursor)
        self.iterate.setToolTip("Apply the update and simulate the next iteration (Ctrl+N)")
        layout.addWidget(self.iterate)

        self.restart.clicked.connect(self.restartRequested)
        self.prev_phase.clicked.connect(self.previousPhase)
        self.prev_step.clicked.connect(self.previousStep)
        self.play.clicked.connect(self.playToggled)
        self.next_step.clicked.connect(self.nextStep)
        self.next_phase.clicked.connect(self.nextPhase)
        self.speed.changed.connect(lambda key: self.speedChanged.emit(float(key)))
        self.iterate.clicked.connect(self.nextIteration)
        self.apply_theme(theme)

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        for button in (self.restart, self.prev_phase, self.prev_step, self.next_step, self.next_phase):
            button.apply_theme(theme)
        self.iterate.setIcon(icon("repeat", "#FFFFFF", 16))
        self.timeline.apply_theme(theme)
        self.set_playing(self._playing)

    def set_playing(self, playing: bool) -> None:
        self._playing = playing
        self.play.setIcon(icon("pause" if playing else "play", "#FFFFFF", 18))
