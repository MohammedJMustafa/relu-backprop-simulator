"""Where the app keeps its files (logs, style images): the platform's per-user data folder."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

APP_FOLDER = "ReLUBackpropSim"


def data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir())
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    folder = base / APP_FOLDER
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        folder = Path(tempfile.gettempdir()) / APP_FOLDER
        folder.mkdir(parents=True, exist_ok=True)
    return folder
