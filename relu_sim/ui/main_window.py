"""The main window: header with navigation, the parameter sidebar and five pages."""

from __future__ import annotations

import os
import platform
from pathlib import Path

import matplotlib
from PySide6 import __version__ as pyside_version
from PySide6.QtCore import QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QGuiApplication, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (QApplication, QButtonGroup, QDialog, QFileDialog, QGridLayout, QHBoxLayout, QLabel,
                               QMainWindow, QMenu, QMessageBox, QPushButton, QStackedWidget, QToolButton,
                               QVBoxLayout, QWidget)

from .. import APP_NAME, __version__, homework
from ..core.verification import MISMATCH
from .branding import app_icon, logo_pixmap
from .export import export_network_png, export_solution_pdf, export_training_csv
from .icons import icon
from .pages.activations import ActivationsPage
from .pages.simulation import SimulationPage
from .pages.solution import SolutionPage
from .pages.training import TrainingPage
from .pages.verification import VerificationPage
from .state import AppState
from .theme import DARK, LIGHT, Theme, palette, style_sheet
from .widgets.common import IconButton
from .widgets.parameter_panel import ParameterPanel

PAGES = (("simulation", "Simulation", "pulse"), ("solution", "Solution", "document"), ("training", "Training", "chart"),
         ("activations", "Activations", "bolt"), ("verification", "Verification", "check_circle"))

SHORTCUTS = (
    ("Space", "Play or pause the simulation"),
    ("→  /  ←", "Next / previous step"),
    ("Page Down  /  Page Up", "Next / previous phase"),
    ("Home  /  End", "First / last step"),
    ("Ctrl + N", "Next iteration (apply the update)"),
    ("Ctrl + 1 … 5", "Switch page"),
    ("Ctrl + R", "Reset to the homework values"),
    ("Ctrl + B", "Show / hide the parameters"),
    ("Ctrl + T", "Light / dark theme"),
    ("Ctrl + E", "Export the worked solution as PDF"),
    ("Ctrl + Shift + S", "Save the network diagram as PNG"),
    ("F1", "This list"),
)


class Avatar(QWidget):
    """A round badge with the student's initials."""

    def __init__(self, initials: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._initials = initials
        self.setFixedSize(34, 34)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#5B6CF0"))
        painter.drawEllipse(QRectF(0, 0, 34, 34))
        font = QFont(self.font())
        font.setBold(True)
        font.setPointSizeF(9.5)
        painter.setFont(font)
        painter.setPen(QColor("#FFFFFF"))
        painter.drawText(QRectF(0, 0, 34, 34), Qt.AlignmentFlag.AlignCenter, self._initials)


class MainWindow(QMainWindow):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        # Qt appends the application name: "Homework 3 · Mohammed Jalal Mustafa - ReLU Backprop Simulator"
        self.setWindowTitle(f"{homework.TITLE} · {homework.STUDENT.name}")
        self.setWindowIcon(app_icon())
        self.setMinimumSize(1240, 760)

        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_header())
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self.sidebar = ParameterPanel(state)
        body.addWidget(self.sidebar)
        self.stack = QStackedWidget()
        self.simulation = SimulationPage(state)
        self.solution = SolutionPage(state)
        self.training = TrainingPage(state)
        self.activations = ActivationsPage(state)
        self.verification = VerificationPage(state)
        self.pages = {"simulation": self.simulation, "solution": self.solution, "training": self.training,
                      "activations": self.activations, "verification": self.verification}
        for key, _, _ in PAGES:
            self.stack.addWidget(self.pages[key])
        body.addWidget(self.stack, 1)
        layout.addLayout(body, 1)

        self.status_message = QLabel()
        self.status_check = QLabel()
        self.statusBar().addWidget(self.status_message, 1)
        self.statusBar().addPermanentWidget(self.status_check)
        self.statusBar().setSizeGripEnabled(False)

        self.simulation.exportImageRequested.connect(self.export_png)
        self.solution.exportPdfRequested.connect(self.export_pdf)
        self.solution.copied.connect(lambda text: self.show_message(text))
        self.training.exportCsvRequested.connect(self.export_csv)
        state.themeChanged.connect(self.apply_theme)
        state.configChanged.connect(self._update_status)
        self._build_shortcuts()
        self.apply_theme()
        self._update_status()
        self.show_page("simulation")
        self._initial_geometry()

    # ------------------------------------------------------------------ header

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("Header")
        header.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        header.setFixedHeight(66)
        row = QHBoxLayout(header)
        row.setContentsMargins(12, 0, 14, 0)
        row.setSpacing(12)

        self.sidebar_button = IconButton("sidebar", "Show or hide the parameters (Ctrl+B)")
        self.sidebar_button.clicked.connect(self.toggle_sidebar)
        row.addWidget(self.sidebar_button)
        logo = QLabel()
        logo.setPixmap(logo_pixmap(36))
        row.addWidget(logo)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addStretch(1)
        title = QLabel(APP_NAME)
        title.setObjectName("AppTitle")
        subtitle = QLabel(f"{homework.TITLE} · {homework.SUBTITLE}")
        subtitle.setObjectName("AppSubtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        titles.addStretch(1)
        row.addLayout(titles)
        row.addStretch(1)

        nav = QWidget()
        nav.setObjectName("NavBar")
        nav.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        nav.setFixedHeight(42)
        nav_row = QHBoxLayout(nav)
        nav_row.setContentsMargins(4, 4, 4, 4)
        nav_row.setSpacing(2)
        self.nav_group = QButtonGroup(self)
        self.nav_buttons: dict[str, QToolButton] = {}
        for index, (key, label, glyph) in enumerate(PAGES):
            button = QToolButton()
            button.setObjectName("NavButton")
            button.setText(label)
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            button.setIconSize(QSize(16, 16))
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolTip(f"{label} (Ctrl+{index + 1})")
            button.clicked.connect(lambda _=False, k=key: self.show_page(k))
            self.nav_group.addButton(button)
            self.nav_buttons[key] = button
            nav_row.addWidget(button)
        row.addWidget(nav, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addStretch(1)

        chip = QHBoxLayout()
        chip.setSpacing(9)
        chip.addWidget(Avatar("".join(part[0] for part in homework.STUDENT.name.split()[:2])))
        names = QVBoxLayout()
        names.setSpacing(0)
        names.addStretch(1)
        name = QLabel(homework.STUDENT.name)
        name.setObjectName("StudentName")
        meta = QLabel(f"Group {homework.STUDENT.group} · {homework.TITLE}")
        meta.setObjectName("StudentMeta")
        names.addWidget(name)
        names.addWidget(meta)
        names.addStretch(1)
        chip.addLayout(names)
        row.addLayout(chip)
        row.addSpacing(8)

        self.export_button = IconButton("download", "Export")
        self.export_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        export_menu = QMenu(self)
        self.action_pdf = export_menu.addAction("Export solution as PDF…", self.export_pdf)
        self.action_png = export_menu.addAction("Save network diagram as PNG…", self.export_png)
        self.action_csv = export_menu.addAction("Export training data as CSV…", self.export_csv)
        self.export_button.setMenu(export_menu)
        self.theme_button = IconButton("moon", "Dark theme (Ctrl+T)")
        self.theme_button.clicked.connect(self.toggle_theme)
        self.help_button = IconButton("help", "Help")
        self.help_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        help_menu = QMenu(self)
        self.action_keys = help_menu.addAction("Keyboard shortcuts", self.show_shortcuts)
        help_menu.addSeparator()
        self.action_about = help_menu.addAction(f"About {APP_NAME}", self.show_about)
        self.help_button.setMenu(help_menu)
        for button in (self.export_button, self.theme_button, self.help_button):
            row.addWidget(button)
        return header

    # ------------------------------------------------------------------ navigation and shortcuts

    def show_page(self, key: str) -> None:
        self.stack.setCurrentWidget(self.pages[key])
        self.nav_buttons[key].setChecked(True)
        for shortcut in self._simulation_shortcuts:
            shortcut.setEnabled(key == "simulation")

    def _build_shortcuts(self) -> None:
        sim = self.simulation
        playback = sim.playback

        def add(sequence, handler) -> QShortcut:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(handler)
            return shortcut

        self._simulation_shortcuts = [
            add(Qt.Key.Key_Space, playback.toggle),
            add(Qt.Key.Key_Right, playback.step_forward),
            add(Qt.Key.Key_Left, playback.step_back),
            add(Qt.Key.Key_PageDown, sim.next_phase),
            add(Qt.Key.Key_PageUp, sim.previous_phase),
            add(Qt.Key.Key_Home, lambda: playback.seek(0)),
            add(Qt.Key.Key_End, lambda: playback.seek(playback.count - 1)),
        ]
        add("Ctrl+N", sim.next_iteration)
        for index, (key, _, _) in enumerate(PAGES):
            add(f"Ctrl+{index + 1}", lambda k=key: self.show_page(k))
        add("Ctrl+R", self.state.reset)
        add("Ctrl+B", self.toggle_sidebar)
        add("Ctrl+T", self.toggle_theme)
        add("Ctrl+E", self.export_pdf)
        add("Ctrl+Shift+S", self.export_png)
        add("F1", self.show_shortcuts)
        add("Ctrl++", lambda: self.solution.document.set_zoom(self.solution.document.zoom + 0.1))
        add("Ctrl+=", lambda: self.solution.document.set_zoom(self.solution.document.zoom + 0.1))
        add("Ctrl+-", lambda: self.solution.document.set_zoom(self.solution.document.zoom - 0.1))

    def _initial_geometry(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(1560, 960)
            return
        area = screen.availableGeometry()
        width = min(1640, int(area.width() * 0.94))
        height = min(1000, int(area.height() * 0.94))
        self.resize(width, height)
        self.move(area.x() + (area.width() - width) // 2, area.y() + (area.height() - height) // 2)

    # ------------------------------------------------------------------ theme and status

    def toggle_theme(self) -> None:
        self.state.set_theme(LIGHT if self.state.theme.dark else DARK)

    def toggle_sidebar(self) -> None:
        self.sidebar.setVisible(not self.sidebar.isVisible())

    def apply_theme(self) -> None:
        theme: Theme = self.state.theme
        app = QApplication.instance()
        try:
            app.styleHints().setColorScheme(Qt.ColorScheme.Dark if theme.dark else Qt.ColorScheme.Light)
        except Exception:
            pass
        app.setPalette(palette(theme))
        app.setStyleSheet(style_sheet(theme))
        for key, label, glyph in PAGES:
            button = self.nav_buttons[key]
            normal = icon(glyph, theme.text2, 16)
            button.setIcon(normal)
        self.export_button.apply_theme(theme)
        self.help_button.apply_theme(theme)
        self.sidebar_button.apply_theme(theme)
        self.theme_button.set_glyph("sun" if theme.dark else "moon", theme)
        self.theme_button.setToolTip("Light theme (Ctrl+T)" if theme.dark else "Dark theme (Ctrl+T)")
        self.sidebar.apply_theme(theme)
        for page in self.pages.values():
            page.apply_theme(theme)
        self._update_status()

    def _update_status(self) -> None:
        checks = self.state.answer_checks
        ok = sum(1 for c in checks if c.status != MISMATCH)
        grads_ok = all(g.ok for g in self.state.gradient_check)
        theme = self.state.theme
        good = theme.role("good").name()
        self.status_check.setText(
            f"<span style='color:{good}'>✓</span> {ok}/{len(checks)} homework answers reproduced"
            f"  ·  gradient check {'passed' if grads_ok else 'see Verification'}"
            f"  ·  Python {platform.python_version()} · Qt {pyside_version}")
        if not self.status_message.text():
            self.status_message.setText("Ready. Press Space to play the simulation.")

    def show_message(self, text: str, timeout_ms: int = 6000) -> None:
        self.status_message.setText(text)
        QTimer.singleShot(timeout_ms, lambda: self.status_message.setText("Ready.")
                          if self.status_message.text() == text else None)

    # ------------------------------------------------------------------ exports

    def _save_path(self, title: str, name: str, pattern: str) -> str | None:
        folder = Path(os.path.expanduser("~")) / "Documents"
        if not folder.is_dir():
            folder = Path(os.path.expanduser("~"))
        path, _ = QFileDialog.getSaveFileName(self, title, str(folder / name), pattern)
        return path or None

    def export_pdf(self) -> None:
        path = self._save_path("Export the worked solution", "Homework3_ReLU_Solution.pdf", "PDF document (*.pdf)")
        if not path:
            return
        try:
            pages = export_solution_pdf(path, self.state.solution, f"{homework.TITLE} — {homework.SUBTITLE}")
        except OSError as error:
            QMessageBox.warning(self, APP_NAME, str(error))
            return
        self.show_message(f"Saved the solution ({pages} pages) to {path}")

    def export_png(self) -> None:
        frame = self.simulation.network.frame()
        if frame is None:
            return
        path = self._save_path("Save the network diagram", "Homework3_ReLU_Network.png", "PNG image (*.png)")
        if not path:
            return
        try:
            export_network_png(path, frame, self.state.theme, self.simulation.playback.progress)
        except OSError as error:
            QMessageBox.warning(self, APP_NAME, str(error))
            return
        self.show_message(f"Saved the diagram to {path}")

    def export_csv(self) -> None:
        path = self._save_path("Export the training data", "Homework3_ReLU_Training.csv", "CSV file (*.csv)")
        if not path:
            return
        try:
            export_training_csv(path, self.training.runs())
        except OSError as error:
            QMessageBox.warning(self, APP_NAME, str(error))
            return
        self.show_message(f"Saved {self.training.count} iterations per activation to {path}")

    # ------------------------------------------------------------------ dialogs

    def show_shortcuts(self) -> None:
        self.shortcuts_dialog().exec()

    def show_about(self) -> None:
        self.about_dialog().exec()

    def shortcuts_dialog(self) -> QDialog:
        dialog = QDialog(self)
        dialog.setWindowTitle("Keyboard shortcuts")
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(24, 20, 24, 20)
        title = QLabel("Keyboard shortcuts")
        title.setObjectName("CardTitle")
        layout.addWidget(title)
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(8)
        theme = self.state.theme
        for row, (keys, text) in enumerate(SHORTCUTS):
            key = QLabel(keys)
            key.setStyleSheet(f"background: {theme.surface3}; border: 1px solid {theme.border2}; border-radius: 6px;"
                              "padding: 3px 9px; font-weight: 600;")
            grid.addWidget(key, row, 0, Qt.AlignmentFlag.AlignLeft)
            grid.addWidget(QLabel(text), row, 1)
        layout.addLayout(grid)
        close = QPushButton("Close")
        close.clicked.connect(dialog.accept)
        layout.addWidget(close, 0, Qt.AlignmentFlag.AlignRight)
        return dialog

    def about_dialog(self) -> QDialog:
        dialog = QDialog(self)
        dialog.setWindowTitle(f"About {APP_NAME}")
        dialog.setMinimumWidth(460)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(28, 24, 28, 22)
        layout.setSpacing(6)
        logo = QLabel()
        logo.setPixmap(logo_pixmap(64))
        layout.addWidget(logo, 0, Qt.AlignmentFlag.AlignHCenter)
        name = QLabel(APP_NAME)
        name.setObjectName("PageTitle")
        layout.addWidget(name, 0, Qt.AlignmentFlag.AlignHCenter)
        version = QLabel(f"Version {__version__}")
        version.setObjectName("Muted")
        layout.addWidget(version, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(10)
        for text, object_name in (
            (f"{homework.TITLE}: {homework.SUBTITLE}", "CardTitle"),
            (f"{homework.STUDENT.name} · Group {homework.STUDENT.group}", ""),
            ("An interactive simulation of the forward pass, backpropagation and gradient-descent update of a "
             "2-2-1 ReLU network, checked against every number of the homework solution.", "Muted"),
            (f"Python {platform.python_version()} · PySide6 {pyside_version} · matplotlib "
             f"{matplotlib.__version__} · {platform.system()} {platform.release()}", "StatNote"),
        ):
            label = QLabel(text)
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            if object_name:
                label.setObjectName(object_name)
            layout.addWidget(label)
        layout.addSpacing(12)
        close = QPushButton("Close")
        close.setObjectName("Primary")
        close.clicked.connect(dialog.accept)
        layout.addWidget(close, 0, Qt.AlignmentFlag.AlignHCenter)
        return dialog

    def closeEvent(self, event) -> None:
        self.simulation.playback.pause()
        super().closeEvent(event)
