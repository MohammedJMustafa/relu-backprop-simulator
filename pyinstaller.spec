# -*- mode: python -*-
# PyInstaller build of the ReLU Backprop Simulator, shared by build_exe.bat and the GitHub workflow:
#
#     python -m PyInstaller --noconfirm --clean pyinstaller.spec
#
# Windows and Linux: one self-contained executable in dist/.  macOS: dist/<name>.app
# The file name can be changed with the environment variable RELU_SIM_EXE_NAME.

import os
import re
import sys

NAME = os.environ.get("RELU_SIM_EXE_NAME", "ReLU Backprop Simulator")
ROOT = os.path.abspath(SPECPATH)
ASSETS = os.path.join(ROOT, "relu_sim", "assets")
with open(os.path.join(ROOT, "relu_sim", "__init__.py"), encoding="utf-8") as handle:
    VERSION = re.search(r'__version__ = "([^"]+)"', handle.read()).group(1)

# Parts of Qt the app never uses (software OpenGL, Qt Quick/QML, Qt PDF, Qt's own translations).
# Leaving them out makes the program about 45 MB smaller, so it unpacks and starts faster.
UNUSED = ("opengl32sw", "qt6quick", "qtquick", "qt6qml", "qtqml", "qt6pdf", "qtpdf", "qpdf",
          "qt6virtualkeyboard", "qtvirtualkeyboard")


def wanted(entry) -> bool:
    destination = entry[0].replace("\\", "/").lower()
    if "translations/" in destination:
        return False
    return not any(part in destination for part in UNUSED)


a = Analysis(
    [os.path.join(ROOT, "main.py")],
    pathex=[ROOT],
    excludes=["PyQt5", "PyQt6", "PySide2", "tkinter", "IPython", "pandas", "scipy", "torch"],
)
a.binaries = [entry for entry in a.binaries if wanted(entry)]
a.datas = [entry for entry in a.datas if wanted(entry)]
pyz = PYZ(a.pure)

if sys.platform == "darwin":
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=NAME, console=False,
              icon=os.path.join(ASSETS, "app.png"))
    collected = COLLECT(exe, a.binaries, a.datas, name=NAME)
    app = BUNDLE(collected, name=f"{NAME}.app", icon=os.path.join(ASSETS, "app.png"),
                 bundle_identifier="io.github.mohammedjmustafa.relubackpropsimulator",
                 info_plist={"CFBundleShortVersionString": VERSION, "NSHighResolutionCapable": True})
else:
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name=NAME, console=False, upx=False,
              icon=os.path.join(ASSETS, "app.ico") if sys.platform == "win32" else None)
