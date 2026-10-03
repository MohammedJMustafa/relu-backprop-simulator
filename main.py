"""ReLU Backpropagation Simulator - Homework 3 (Mohammed Jalal Mustafa, Group A).

Start it by double-clicking run_simulation.bat, or from a terminal:

    python main.py                       open the simulator
    python main.py --self-test           check the homework's answers without opening a window
    python main.py --self-test --gui     ... and check the whole user interface off-screen
"""

import os
import sys

os.environ.setdefault("QT_API", "pyside6")


def _show_error(text: str) -> None:
    """A message box even when started with pythonw (no console)."""
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, "ReLU Backprop Simulator", 0x10)
    except Exception:
        print(text, file=sys.stderr)


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        from relu_sim.selftest import main as self_test
        return self_test(sys.argv[1:])
    try:
        from relu_sim.app import run
    except ImportError as error:
        _show_error(f"A required package is missing: {error.name or error}.\n\n"
                    "Start the simulator with run_simulation.bat - it installs everything it needs.")
        return 1
    return run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
