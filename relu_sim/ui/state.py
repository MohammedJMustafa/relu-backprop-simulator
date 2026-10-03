"""The single source of truth of the application: the current configuration,
the simulation iteration and the theme, plus cached results derived from them."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Callable, TypeVar

from PySide6.QtCore import QObject, Signal

from .. import homework
from ..content.solution import build_solution
from ..content.timeline import build_timeline
from ..core.gradcheck import gradient_check
from ..core.network import PARAM_NAMES, Config, Iteration, Params, run_iteration
from ..core.training import TrainingRun, iterations_to_reach, train
from ..core.verification import check_homework_answers
from .theme import Theme

T = TypeVar("T")
CONVERGENCE_CAP = 200_000


class AppState(QObject):
    configChanged = Signal()
    themeChanged = Signal()

    def __init__(self, theme: Theme, parent: QObject | None = None):
        super().__init__(parent)
        self._config: Config = homework.HOMEWORK
        self._iteration_no = 1
        self._theme = theme
        self._cache: dict = {}
        self._static: dict = {}

    # ------------------------------------------------------------------ basics

    @property
    def config(self) -> Config:
        return self._config

    @property
    def iteration_no(self) -> int:
        return self._iteration_no

    @property
    def theme(self) -> Theme:
        return self._theme

    def set_theme(self, theme: Theme) -> None:
        if theme is not self._theme:
            self._theme = theme
            self.themeChanged.emit()

    def set_config(self, config: Config, iteration_no: int = 1) -> None:
        if config.is_close(self._config) and iteration_no == self._iteration_no:
            return
        self._config = config
        self._iteration_no = iteration_no
        self._cache.clear()
        self.configChanged.emit()

    # ------------------------------------------------------------------ edits

    def set_param(self, name: str, value: float) -> None:
        self.set_config(replace(self._config, params=self._config.params.with_values(**{name: float(value)})))

    def set_sample_value(self, name: str, value: float) -> None:
        self.set_config(replace(self._config, sample=replace(self._config.sample, **{name: float(value)})))

    def set_lr(self, value: float) -> None:
        self.set_config(replace(self._config, lr=float(value)))

    def set_activation(self, key: str) -> None:
        self.set_config(replace(self._config, activation=key), self._iteration_no)

    def apply_preset(self, key: str) -> None:
        self.set_config(homework.PRESETS_BY_KEY[key].config)

    def reset(self) -> None:
        self.apply_preset("homework")

    def randomize(self, rng: random.Random | None = None) -> None:
        rng = rng or random.Random()
        values = [round(rng.uniform(-1.0, 1.0), 2) if name.startswith("w") else round(rng.uniform(-0.5, 0.5), 2)
                  for name in PARAM_NAMES]
        self.set_config(replace(self._config, params=Params.from_values(values)))

    def next_iteration(self) -> None:
        """Continue training: the updated parameters become the given ones."""
        self.set_config(self.iteration.config_new, self._iteration_no + 1)

    # ------------------------------------------------------------------ derived results (cached)

    def _cached(self, key, factory: Callable[[], T]) -> T:
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]

    @property
    def iteration(self) -> Iteration:
        return self._cached("iteration", lambda: run_iteration(self._config))

    @property
    def timeline(self):
        return self._cached("timeline", lambda: build_timeline(self._config, self._iteration_no))

    @property
    def solution(self):
        return self._cached("solution", lambda: build_solution(self._config, self._iteration_no))

    @property
    def gradient_check(self):
        return self._cached("gradcheck", lambda: gradient_check(self._config))

    @property
    def answer_checks(self):
        if "answers" not in self._static:
            self._static["answers"] = check_homework_answers()
        return self._static["answers"]

    def training(self, activation: str, iterations: int) -> TrainingRun:
        config = replace(self._config, activation=activation)
        return self._cached(("train", activation, iterations), lambda: train(config, iterations))

    def iterations_to(self, activation: str, threshold: float) -> int | None:
        config = replace(self._config, activation=activation)
        return self._cached(("reach", activation, threshold),
                            lambda: iterations_to_reach(config, threshold, CONVERGENCE_CAP))

    @property
    def matching_preset(self) -> str | None:
        if self._iteration_no != 1:
            return None
        for preset in homework.PRESETS:
            if preset.config.is_close(self._config):
                return preset.key
        return None

    @property
    def is_homework(self) -> bool:
        return self._iteration_no == 1 and self._config.is_close(homework.HOMEWORK)
