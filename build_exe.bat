@echo off
setlocal EnableExtensions DisableDelayedExpansion
title Build ReLU Backprop Simulator.exe
rem ===========================================================================
rem  Optional: build a self-contained Windows program - one .exe file that runs
rem  without Python. The result is  dist\ReLU Backprop Simulator.exe  next to this
rem  file. The build takes about two minutes.
rem ===========================================================================

set "PYTHONUTF8=1"
set "QT_API=pyside6"
set "WORK=%LOCALAPPDATA%\ReLUBackpropSim\build"
set "PYEXE="

where py >nul 2>&1
if errorlevel 1 goto no_python
for /f "delims=" %%P in ('py -3.13 -c "import sys; print(sys.executable)" 2^>nul') do set "PYEXE=%%P"
if not defined PYEXE for /f "delims=" %%P in ('py -3 -c "import sys; print(sys.executable)"') do set "PYEXE=%%P"
if not defined PYEXE goto no_python

"%PYEXE%" -c "import PySide6.QtWidgets, matplotlib, numpy" >nul 2>&1
if errorlevel 1 goto missing_packages
"%PYEXE%" -m PyInstaller --version >nul 2>&1
if errorlevel 1 "%PYEXE%" -m pip install pyinstaller
if errorlevel 1 goto failed

echo Building with %PYEXE% ...
"%PYEXE%" -m PyInstaller --noconfirm --clean --windowed --onefile ^
  --name "ReLU Backprop Simulator" ^
  --icon "%~dp0relu_sim\assets\app.ico" ^
  --workpath "%WORK%" --specpath "%WORK%" --distpath "%~dp0dist" ^
  --exclude-module PyQt5 --exclude-module PyQt6 --exclude-module PySide2 --exclude-module tkinter ^
  --exclude-module IPython --exclude-module pandas --exclude-module scipy --exclude-module torch ^
  "%~dp0main.py"
if errorlevel 1 goto failed

echo.
echo Done: "%~dp0dist\ReLU Backprop Simulator.exe"
pause
exit /b 0

:no_python
echo Python 3 was not found. Install it from https://www.python.org/downloads/ first.
pause
exit /b 1

:missing_packages
echo PySide6, matplotlib and numpy are needed. Run run_simulation.bat once first.
pause
exit /b 1

:failed
echo The build failed - see the messages above.
pause
exit /b 1
