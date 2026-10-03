"""The step-by-step simulation: animated network, synced explanation and player."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ...content import numfmt
from ...content.timeline import PHASE_LABELS, SimStep
from ..playback import Playback
from ..state import AppState
from ..theme import Theme
from ..widgets.common import Card, IconButton, caption, hline
from ..widgets.document_view import DocumentView
from ..widgets.network_view import NetworkView
from ..widgets.player_bar import PlayerBar, phase_color


def badge_style(color: QColor) -> str:
    fill = QColor(color)
    fill.setAlphaF(0.15)
    return (f"background: {fill.name(QColor.NameFormat.HexArgb)}; color: {color.name()}; border-radius: 9px;"
            "padding: 2px 10px; font-size: 8pt; font-weight: 700; letter-spacing: 0.6px;")


class StepCard(Card):
    """The explanation of the current step: phase, title, text and equations."""

    def __init__(self, theme: Theme, parent: QWidget | None = None):
        super().__init__(None, parent, padding=18, spacing=8)
        self._theme = theme
        header = QHBoxLayout()
        header.setSpacing(8)
        self.badge = QLabel()
        self.section = QLabel()
        self.section.setObjectName("Muted")
        self.counter = QLabel()
        self.counter.setObjectName("Muted")
        header.addWidget(self.badge)
        header.addWidget(self.section)
        header.addStretch(1)
        header.addWidget(self.counter)
        self.title = QLabel()
        self.title.setObjectName("PageTitle")
        self.title.setWordWrap(True)
        self.title.setStyleSheet("font-size: 14.5pt;")
        self.doc = DocumentView(theme, "card")
        self.outer.addLayout(header)
        self.outer.addWidget(self.title)
        self.outer.addWidget(hline())
        self.outer.addWidget(self.doc, 1)

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        self.doc.apply_theme(theme)

    def set_step(self, step: SimStep, count: int) -> None:
        color = phase_color(self._theme, step.phase)
        self.badge.setText(PHASE_LABELS[step.phase].upper())
        self.badge.setStyleSheet(badge_style(color))
        self.section.setText(step.section)
        self.counter.setText(f"Step {step.index + 1} / {count}")
        self.title.setText(step.title)
        self.doc.set_blocks(step.blocks)


class SummaryCard(Card):
    """Error before and after the update, revealed as the simulation reaches them."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(None, parent, padding=16, spacing=6)
        self.heading = caption("Iteration 1")
        self.outer.addWidget(self.heading)
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(0)
        self.values: dict[str, QLabel] = {}
        for column, (key, label) in enumerate((("before", "Error E"), ("after", "After update"), ("change", "Change"))):
            title = QLabel(label)
            title.setObjectName("StatNote")
            value = QLabel("—")
            value.setObjectName("StatValue")
            value.setStyleSheet("font-size: 15pt;")
            grid.addWidget(title, 0, column)
            grid.addWidget(value, 1, column)
            self.values[key] = value
        self.outer.addLayout(grid)
        self.note = QLabel()
        self.note.setObjectName("StatNote")
        self.note.setWordWrap(True)
        self.outer.addWidget(self.note)

    def update_values(self, state: AppState, step_key: str, step_index: int, keys: list[str]) -> None:
        it = state.iteration
        theme = state.theme
        self.heading.setText(f"ITERATION {state.iteration_no}  ·  {it.config.act.name.upper()}  ·  "
                             f"η = {numfmt.fmt(it.config.lr)}")
        reached = set(keys[: step_index + 1])
        before = "E" in reached
        after = "verify_error" in reached
        self.values["before"].setText(numfmt.fmt_compact(it.forward.error, 4) if before else "—")
        self.values["after"].setText(numfmt.fmt_compact(it.forward_new.error, 4) if after else "—")
        if after:
            change = it.error_change
            good = it.forward_new.error < it.forward.error
            arrow = "↓ " if good else ("↑ " if change > 0 else "")
            color = theme.role("good") if good else theme.role("bad") if change > 0 else theme.c("muted")
            self.values["change"].setText(arrow + numfmt.fmt_percent(abs(change)))
            self.values["change"].setStyleSheet(f"font-size: 15pt; color: {color.name()};")
            self.note.setText(f"Output h_out: {numfmt.fmt_compact(it.forward.h_out, 4)} → "
                              f"{numfmt.fmt_compact(it.forward_new.h_out, 4)}  (target {numfmt.fmt(it.config.sample.y_true)})")
        else:
            self.values["change"].setText("—")
            self.values["change"].setStyleSheet("font-size: 15pt;")
            self.note.setText("Play the simulation to compute the error before and after the update.")


class SimulationPage(QWidget):
    exportImageRequested = Signal()

    def __init__(self, state: AppState, parent: QWidget | None = None):
        super().__init__(parent)
        self.state = state
        self.playback = Playback(self)
        self._steps: tuple[SimStep, ...] = ()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        top = QHBoxLayout()
        top.setSpacing(14)
        self.network_card = Card("Network", padding=14)
        self.network_info = QLabel()
        self.network_info.setObjectName("Muted")
        self.network_card.header.insertWidget(1, self.network_info)
        self.save_image = IconButton("picture", "Save the diagram as a PNG image (Ctrl+Shift+S)")
        self.network_card.add_header_widget(self.save_image)
        self.network = NetworkView(state.theme)
        self.network.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.network_card.outer.addWidget(self.network, 1)
        top.addWidget(self.network_card, 1)

        right = QVBoxLayout()
        right.setSpacing(14)
        self.step_card = StepCard(state.theme)
        self.step_card.setMinimumWidth(380)
        self.step_card.setMaximumWidth(440)
        self.summary = SummaryCard()
        self.summary.setMaximumWidth(440)
        right.addWidget(self.step_card, 1)
        right.addWidget(self.summary)
        top.addLayout(right)
        layout.addLayout(top, 1)

        self.player_card = Card(None, padding=10)
        self.player = PlayerBar(state.theme)
        self.player_card.outer.addWidget(self.player)
        layout.addWidget(self.player_card)

        p = self.playback
        p.indexChanged.connect(self._show_step)
        p.progressChanged.connect(self._show_progress)
        p.playingChanged.connect(self.player.set_playing)
        self.player.playToggled.connect(p.toggle)
        self.player.nextStep.connect(p.step_forward)
        self.player.previousStep.connect(p.step_back)
        self.player.restartRequested.connect(lambda: p.seek(0))
        self.player.nextPhase.connect(self.next_phase)
        self.player.previousPhase.connect(self.previous_phase)
        self.player.speedChanged.connect(p.set_speed)
        self.player.nextIteration.connect(self.next_iteration)
        self.player.timeline.seekRequested.connect(lambda index: p.seek(index))
        self.save_image.clicked.connect(self.exportImageRequested)
        state.configChanged.connect(self.reload)

        self.apply_theme(state.theme)
        self.reload()

    # ------------------------------------------------------------------ data

    def reload(self) -> None:
        self._steps = self.state.timeline
        self.player.timeline.set_steps([s.phase for s in self._steps], [s.title for s in self._steps])
        cfg = self.state.config
        self.network_info.setText(f"·  iteration {self.state.iteration_no}  ·  {cfg.act.name}  ·  "
                                  f"η = {numfmt.fmt(cfg.lr)}")
        self.playback.set_steps([s.duration for s in self._steps])
        self._show_step(self.playback.index)

    def _show_step(self, index: int) -> None:
        if not self._steps:
            return
        step = self._steps[index]
        self.network.set_frame(step.frame, self.playback.progress)
        self.step_card.set_step(step, len(self._steps))
        self.summary.update_values(self.state, step.key, index, [s.key for s in self._steps])
        self.player.timeline.set_position(index, self.playback.progress)
        self.player.iterate.setEnabled(True)

    def _show_progress(self, progress: float) -> None:
        self.network.set_progress(progress)
        self.player.timeline.set_position(self.playback.index, progress)

    # ------------------------------------------------------------------ navigation

    def _phase_starts(self) -> list[int]:
        starts = []
        for i, step in enumerate(self._steps):
            if i == 0 or step.phase != self._steps[i - 1].phase:
                starts.append(i)
        return starts

    def next_phase(self) -> None:
        index = self.playback.index
        later = [s for s in self._phase_starts() if s > index]
        self.playback.seek(later[0] if later else len(self._steps) - 1)

    def previous_phase(self) -> None:
        index = self.playback.index
        starts = self._phase_starts()
        current = max(s for s in starts if s <= index)
        if index == current:      # already at the start of a phase: go to the previous phase
            earlier = [s for s in starts if s < current]
            self.playback.seek(earlier[-1] if earlier else 0)
        else:
            self.playback.seek(current)

    def next_iteration(self) -> None:
        self.playback.pause()
        self.state.next_iteration()
        self.playback.seek(0)
        self.playback.play()

    # ------------------------------------------------------------------ theme / visibility

    def apply_theme(self, theme: Theme) -> None:
        self.network.apply_theme(theme)
        self.step_card.apply_theme(theme)
        self.player.apply_theme(theme)
        self.save_image.apply_theme(theme)
        if self._steps:
            self._show_step(self.playback.index)

    def hideEvent(self, event) -> None:
        self.playback.pause()
        super().hideEvent(event)
