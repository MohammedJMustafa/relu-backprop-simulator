@echo off
setlocal EnableExtensions DisableDelayedExpansion
title ReLU Backprop Simulator - Mohammed Jalal Mustafa
rem ===========================================================================
rem  ReLU Backpropagation Simulator - Homework 1
rem  Mohammed Jalal Mustafa - Group A
rem
rem  Double-click this file to start the simulator.
rem
rem  It uses a Python that already has PySide6, matplotlib and numpy. If none
rem  is found it creates a private environment once (outside OneDrive) and
rem  installs them there.
rem
rem    run_simulation.bat            start the simulator
rem    run_simulation.bat debug      start it with a console that shows errors
rem    run_simulation.bat selftest   check the homework answers in the console
rem ===========================================================================

set "PYTHONUTF8=1"
set "QT_API=pyside6"
set "MAIN=%~dp0main.py"
set "REQUIREMENTS=%~dp0requirements.txt"
set "ENV_DIR=%LOCALAPPDATA%\ReLUBackpropSim\venv"
set "CHECK=import PySide6.QtWidgets, matplotlib.backends.backend_qtagg, numpy"
set "MODE=%~1"
set "PYEXE="
set "PYWEXE="

if not exist "%MAIN%" goto missing_main

rem --- 1. a private environment created by an earlier run --------------------
if not exist "%ENV_DIR%\Scripts\python.exe" goto try_launcher
"%ENV_DIR%\Scripts\python.exe" -c "%CHECK%" >nul 2>&1
if errorlevel 1 goto try_launcher
set "PYEXE=%ENV_DIR%\Scripts\python.exe"
goto found

rem --- 2. the Windows py launcher: Python 3.13 first, then any Python 3 --------
:try_launcher
where py >nul 2>&1
if errorlevel 1 goto try_path
call :probe_py -3.13
if defined PYEXE goto found
call :probe_py -3
if defined PYEXE goto found

rem --- 3. python on PATH ---------------------------------------------------------
:try_path
where python >nul 2>&1
if errorlevel 1 goto install
python -c "%CHECK%" >nul 2>&1
if errorlevel 1 goto install
for /f "delims=" %%P in ('python -c "import sys; print(sys.executable)"') do set "PYEXE=%%P"
if defined PYEXE goto found
goto install

:found
for %%D in ("%PYEXE%") do set "PYWEXE=%%~dpDpythonw.exe"
if /i "%MODE%"=="selftest" goto run_selftest
if /i "%MODE%"=="debug" goto run_console
if not exist "%PYWEXE%" goto run_console
start "" "%PYWEXE%" "%MAIN%"
exit /b 0

:run_console
echo Starting the simulator with %PYEXE%
"%PYEXE%" "%MAIN%"
if errorlevel 1 pause
exit /b 0

:run_selftest
"%PYEXE%" "%MAIN%" --self-test
pause
exit /b 0

rem --- first run without the packages: create a private environment ---------
:install
echo.
echo  The simulator needs PySide6, matplotlib and numpy.
echo  Setting up a private Python environment - only the first time, 1 to 3 minutes:
echo    %ENV_DIR%
echo.
where py >nul 2>&1
if errorlevel 1 goto install_with_python
py -3 -m venv "%ENV_DIR%"
if errorlevel 1 goto install_failed
goto install_packages

:install_with_python
where python >nul 2>&1
if errorlevel 1 goto no_python
python -m venv "%ENV_DIR%"
if errorlevel 1 goto install_failed

:install_packages
"%ENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
"%ENV_DIR%\Scripts\python.exe" -m pip install -r "%REQUIREMENTS%"
if errorlevel 1 goto install_failed
set "PYEXE=%ENV_DIR%\Scripts\python.exe"
echo.
echo  Done. Starting the simulator...
goto found

:probe_py
py %1 -c "%CHECK%" >nul 2>&1
if errorlevel 1 exit /b 1
for /f "delims=" %%P in ('py %1 -c "import sys; print(sys.executable)"') do set "PYEXE=%%P"
exit /b 0

:no_python
echo.
echo  Python 3 was not found on this computer.
echo  Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH",
echo  then double-click run_simulation.bat again.
pause
exit /b 1

:install_failed
echo.
echo  Installing the packages failed. Check the internet connection and try again.
pause
exit /b 1

:missing_main
echo.
echo  main.py was not found next to run_simulation.bat. Keep the whole folder together.
pause
exit /b 1
