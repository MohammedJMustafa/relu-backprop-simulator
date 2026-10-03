"""The playback clock of the step-by-step simulation.

One QVariantAnimation drives a step's progress from 0 to 1; a single-shot
timer owned by this object waits a moment before auto-advancing.  Any manual
navigation stops both, so a paused simulation never moves by itself.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, QVariantAnimation, Signal


class Playback(QObject):
    indexChanged = Signal(int)
    progressChanged = Signal(float)
    playingChanged = Signal(bool)

    BASE_MS = 1150
    DWELL_MS = 650

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._durations: list[float] = []
        self._index = 0
        self._progress = 1.0
        self._playing = False
        self._speed = 1.0
        self._anim = QVariantAnimation(self)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.valueChanged.connect(self._on_value)
        self._anim.finished.connect(self._on_finished)
        self._dwell = QTimer(self)
        self._dwell.setSingleShot(True)
        self._dwell.timeout.connect(self._advance)

    # ------------------------------------------------------------------ state

    @property
    def count(self) -> int:
        return len(self._durations)

    @property
    def index(self) -> int:
        return self._index

    @property
    def progress(self) -> float:
        return self._progress

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def at_end(self) -> bool:
        return self._index >= self.count - 1 and self._progress >= 1.0

    # ------------------------------------------------------------------ control

    def set_steps(self, durations: list[float]) -> None:
        """New timeline (same position when possible), shown in its completed state."""
        self._stop_timers()
        self._durations = list(durations)
        self._index = min(self._index, max(self.count - 1, 0))
        self._progress = 1.0
        self.indexChanged.emit(self._index)
        self.progressChanged.emit(self._progress)
        if self._playing and not self.at_end:
            self._dwell.start(int(self.DWELL_MS / self._speed))
        elif self._playing:
            self._set_playing(False)

    def play(self) -> None:
        if not self.count:
            return
        if self.at_end:
            self._index = 0
            self.indexChanged.emit(0)
            self._set_playing(True)
            self._animate()
            return
        self._set_playing(True)
        if self._anim.state() == QVariantAnimation.State.Paused:
            self._anim.resume()
        elif self._progress < 1.0:
            self._animate(self._progress)
        else:
            self._dwell.start(120)

    def pause(self) -> None:
        self._set_playing(False)
        self._dwell.stop()
        if self._anim.state() == QVariantAnimation.State.Running:
            self._anim.pause()

    def toggle(self) -> None:
        self.pause() if self._playing else self.play()

    def step_forward(self) -> None:
        self._stop_timers()
        if self._index < self.count - 1:
            self._index += 1
            self.indexChanged.emit(self._index)
            self._animate()
        else:
            self._finish_static()
            self._set_playing(False)

    def step_back(self) -> None:
        self._stop_timers()
        self._set_playing(False)
        if self._progress < 1.0:
            self._finish_static()
            return
        if self._index > 0:
            self._index -= 1
            self.indexChanged.emit(self._index)
        self._finish_static()

    def seek(self, index: int, animate: bool = False) -> None:
        if not self.count:
            return
        self._stop_timers()
        index = max(0, min(index, self.count - 1))
        changed = index != self._index
        self._index = index
        if changed:
            self.indexChanged.emit(index)
        if animate:
            self._animate()
        else:
            if self._playing:
                self._set_playing(False)
            self._finish_static()

    def set_speed(self, speed: float) -> None:
        self._speed = max(0.1, speed)
        if self._anim.state() == QVariantAnimation.State.Running:
            self._animate(self._progress)

    # ------------------------------------------------------------------ internals

    def _set_playing(self, playing: bool) -> None:
        if playing != self._playing:
            self._playing = playing
            self.playingChanged.emit(playing)

    def _stop_timers(self) -> None:
        self._dwell.stop()
        self._anim.stop()

    def _finish_static(self) -> None:
        self._progress = 1.0
        self.progressChanged.emit(1.0)

    def _animate(self, start: float = 0.0) -> None:
        self._anim.stop()
        duration = self.BASE_MS * self._durations[self._index] * (1.0 - start) / self._speed
        self._progress = start
        self.progressChanged.emit(start)
        self._anim.setStartValue(float(start))
        self._anim.setEndValue(1.0)
        self._anim.setDuration(max(1, int(duration)))
        self._anim.start()

    def _on_value(self, value) -> None:
        self._progress = float(value)
        self.progressChanged.emit(self._progress)

    def _on_finished(self) -> None:
        self._progress = 1.0
        self.progressChanged.emit(1.0)
        if self._playing:
            if self._index < self.count - 1:
                self._dwell.start(int(self.DWELL_MS / self._speed))
            else:
                self._set_playing(False)

    def _advance(self) -> None:
        if self._playing and self._index < self.count - 1:
            self._index += 1
            self.indexChanged.emit(self._index)
            self._animate()
