"""Matplotlib charts embedded in the Qt window and themed like the app.

The Qt binding is pinned to PySide6 before matplotlib is imported (PyQt5 may
be installed too), and figures are drawn inside ``rc_context`` so that every
artist - tick labels included - picks up the theme.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyside6")

import matplotlib  # noqa: E402

matplotlib.use("QtAgg")

from functools import lru_cache  # noqa: E402
from typing import Callable  # noqa: E402

from matplotlib import rc_context  # noqa: E402
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtWidgets import QSizePolicy, QWidget  # noqa: E402

from .theme import Theme  # noqa: E402

DrawFn = Callable[[Figure, Theme], None]


@lru_cache(maxsize=1)
def _fonts() -> dict:
    """Segoe UI and Cambria on Windows; on other systems the closest installed fonts (no warnings)."""
    from matplotlib import font_manager

    names = {font.name for font in font_manager.fontManager.ttflist}
    sans = [n for n in ("Segoe UI", "Helvetica Neue", "Ubuntu", "Noto Sans", "DejaVu Sans") if n in names]
    if "Cambria" in names:
        math = {"mathtext.fontset": "custom", "mathtext.rm": "Cambria", "mathtext.it": "Cambria:italic",
                "mathtext.bf": "Cambria:bold"}
    else:
        math = {"mathtext.fontset": "stix"}
    return {"font.family": sans or ["DejaVu Sans"], **math}


def mpl_rc(theme: Theme) -> dict:
    return {
        **_fonts(),
        "font.size": 9.5,
        "figure.facecolor": theme.surface,
        "savefig.facecolor": theme.surface,
        "axes.facecolor": theme.surface,
        "axes.edgecolor": theme.border2,
        "axes.labelcolor": theme.text2,
        "axes.titlecolor": theme.text,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "grid.color": theme.border,
        "grid.linewidth": 0.8,
        "xtick.color": theme.border2,
        "ytick.color": theme.border2,
        "xtick.labelcolor": theme.muted,
        "ytick.labelcolor": theme.muted,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": theme.text2,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "text.color": theme.text,
        "patch.linewidth": 0,
    }


class ChartCanvas(FigureCanvasQTAgg):
    """A matplotlib figure that redraws itself (lazily) when its data or the theme changes."""

    def __init__(self, draw: DrawFn, theme: Theme, parent: QWidget | None = None):
        self._figure = Figure(layout="constrained")
        super().__init__(self._figure)
        if parent is not None:
            self.setParent(parent)
        self._draw = draw
        self._theme = theme
        self._dirty = True
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(240, 180)

    @property
    def theme(self) -> Theme:
        return self._theme

    def set_theme(self, theme: Theme) -> None:
        self._theme = theme
        self.invalidate()

    def invalidate(self) -> None:
        self._dirty = True
        if self.isVisible():
            self.redraw()

    def redraw(self) -> None:
        self.setStyleSheet(f"background: {self._theme.surface};")
        with rc_context(mpl_rc(self._theme)):
            self._figure.clear()
            self._figure.set_facecolor(self._theme.surface)
            self._draw(self._figure, self._theme)
            self.draw()
        self._dirty = False

    def refresh(self) -> None:
        """Repaint after artists were changed in place (e.g. set_data), keeping the theme."""
        with rc_context(mpl_rc(self._theme)):
            self.draw()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._dirty:
            self.redraw()
