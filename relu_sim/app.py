"""Start the desktop application."""

from __future__ import annotations

import faulthandler
import logging
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from . import APP_ID, APP_NAME, __version__, homework  # noqa: E402
from .paths import data_dir  # noqa: E402

_crash_log = None       # kept open for faulthandler


def install_error_handling() -> Path:
    """Log every unhandled error and show it once in a dialog (pythonw has no console)."""
    global _crash_log
    log_dir = data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "app.log"
    logging.basicConfig(filename=str(log_path), encoding="utf-8", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    try:
        _crash_log = open(log_dir / "crash.log", "a", encoding="utf-8")
        faulthandler.enable(_crash_log)
    except OSError:
        pass

    shown: set[tuple[str, str]] = set()
    busy = [False]

    def hook(exc_type, exc, tb) -> None:
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        logging.error("Unhandled exception\n%s", text)
        if sys.__stderr__ is not None:
            try:
                sys.__stderr__.write(text)
            except Exception:
                pass
        key = (exc_type.__name__, str(exc))
        if busy[0] or key in shown or QApplication.instance() is None:
            return
        shown.add(key)

        def show() -> None:
            busy[0] = True
            try:
                QMessageBox.critical(None, APP_NAME, f"Something went wrong:\n\n{exc_type.__name__}: {exc}\n\n"
                                                     f"The details were saved to:\n{log_path}")
            finally:
                busy[0] = False

        QTimer.singleShot(0, show)

    sys.excepthook = hook
    return log_path


def set_windows_app_id() -> None:
    """Group the window under its own taskbar icon instead of Python's."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
        except Exception:
            pass


def run(argv: list[str] | None = None) -> int:
    set_windows_app_id()
    log_path = install_error_handling()
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setOrganizationName(homework.STUDENT.name)
    app.setStyle("Fusion")

    from .ui.branding import app_icon
    from .ui.main_window import MainWindow
    from .ui.state import AppState
    from .ui.theme import system_theme, ui_font

    app.setFont(ui_font(10))
    app.setWindowIcon(app_icon())
    logging.info("Starting %s %s (log: %s)", APP_NAME, __version__, log_path)
    state = AppState(system_theme())
    window = MainWindow(state)
    window.show()
    return app.exec()
