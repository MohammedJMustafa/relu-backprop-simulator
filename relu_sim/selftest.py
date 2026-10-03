"""`main.py --self-test [--gui]`: check the app without opening a window.

* always: recompute and compare all 40 values printed in the homework PDF,
  check the gradients numerically and the convergence speed;
* with --gui: also build the real window off-screen, show every page in both
  themes, paint every simulation step and export the solution as PDF (this is
  how the release builds are checked on every platform).

Exit code 0 means every check passed.  When there is no console (the
windowed .exe) the report goes to relu_sim_selftest.txt in the temp folder.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from typing import TextIO

from . import APP_NAME, __version__, homework
from .content import numfmt
from .core.gradcheck import gradient_check
from .core.training import iterations_to_reach
from .core.verification import MISMATCH, check_homework_answers


def _output() -> TextIO:
    stream = sys.stdout
    if stream is not None:
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
        return stream
    path = os.environ.get("RELU_SIM_SELFTEST_LOG") or os.path.join(tempfile.gettempdir(), "relu_sim_selftest.txt")
    return open(path, "w", encoding="utf-8")


def _answer_check(out: TextIO) -> bool:
    print(f"{APP_NAME} {__version__} - self-test", file=out)
    print(f"{homework.TITLE}: {homework.SUBTITLE}", file=out)
    print(f"{homework.STUDENT.name}, Group {homework.STUDENT.group}\n", file=out)

    checks = check_homework_answers()
    section = None
    for check in checks:
        if check.answer.section != section:
            section = check.answer.section
            print(f"  {section}", file=out)
        mark = "FAIL" if check.status == MISMATCH else "ok"
        print(f"    {check.answer.key:<18} PDF {check.answer.printed:>9}   computed {check.computed:>+.6f}   {mark}"
              f" ({check.status})", file=out)
    failures = [c for c in checks if c.status == MISMATCH]

    gradients = gradient_check(homework.HOMEWORK)
    worst = max(g.abs_error for g in gradients)
    gradients_ok = all(g.ok for g in gradients)
    print(f"\n  Numerical gradient check: largest difference {worst:.2e} ({'ok' if gradients_ok else 'FAIL'})",
          file=out)
    relu = iterations_to_reach(homework.HOMEWORK, 1e-4)
    sigmoid = iterations_to_reach(homework.LECTURE, 1e-4)
    print(f"  Iterations to E < 1e-4: ReLU {relu}, sigmoid {numfmt.fmt_int(sigmoid) if sigmoid else 'never'}",
          file=out)
    print(f"\n  {len(checks) - len(failures)}/{len(checks)} printed answers reproduced", file=out)
    return not failures and gradients_ok


def _gui_check(out: TextIO) -> bool:
    os.environ.setdefault("QT_API", "pyside6")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from .ui.export import export_solution_pdf
    from .ui.main_window import MainWindow
    from .ui.state import AppState
    from .ui.theme import DARK, LIGHT

    app = QApplication.instance() or QApplication([sys.argv[0] if sys.argv else APP_NAME])
    app.setStyle("Fusion")
    state = AppState(LIGHT)
    window = MainWindow(state)
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.resize(1400, 900)
    window.show()
    app.processEvents()
    for theme in (DARK, LIGHT):
        state.set_theme(theme)
        for key in window.pages:
            window.show_page(key)
            app.processEvents()
            if window.grab().isNull():
                raise RuntimeError(f"the {key} page could not be painted")
    simulation = window.simulation
    for index in range(simulation.playback.count):
        simulation.playback.seek(index)
        simulation.network.grab()
    with tempfile.TemporaryDirectory() as folder:
        pages = export_solution_pdf(os.path.join(folder, "solution.pdf"), state.solution, "self-test")
    window.close()
    print(f"  GUI check: {len(window.pages)} pages in 2 themes, {simulation.playback.count} simulation steps, "
          f"{pages}-page PDF export - ok", file=out)
    return True


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or [])
    out = _output()
    try:
        ok = _answer_check(out)
        if "--gui" in argv:
            ok = _gui_check(out) and ok
        print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}", file=out)
    except Exception:          # report instead of crashing (a windowed .exe would show a blocking dialog)
        traceback.print_exc(file=out)
        ok = False
    finally:
        out.flush()
    return 0 if ok else 1
