@echo off
setlocal EnableExtensions
chcp 65001 >nul
pushd "%~dp0"

set "APP=%~dp0icehalostack.py"
set "RUNTIME=%LOCALAPPDATA%\IceHaloStackRuntime0946"
set "VENV=%RUNTIME%\venv"
set "PY=%VENV%\Scripts\python.exe"
set "PYW=%VENV%\Scripts\pythonw.exe"
set "PKGS=numpy pillow tifffile rawpy opencv-python-headless imageio-ffmpeg"

rem ------------------------------------------------------------
rem Fast path: if IceHaloStack's private environment already works,
rem launch the program immediately. No Python-version whitelist.
rem ------------------------------------------------------------
if exist "%PYW%" (
    "%PY%" -c "import numpy,PIL,tifffile,rawpy,cv2,imageio_ffmpeg,tkinter" >nul 2>&1
    if not errorlevel 1 goto :launch
)

rem Also accept the old project-local .venv when it already exists.
if exist "%~dp0.venv\Scripts\pythonw.exe" (
    "%~dp0.venv\Scripts\python.exe" -c "import numpy,PIL,tifffile,rawpy,cv2,imageio_ffmpeg,tkinter" >nul 2>&1
    if not errorlevel 1 (
        start "" "%~dp0.venv\Scripts\pythonw.exe" "%APP%"
        popd
        exit /b 0
    )
)

rem ------------------------------------------------------------
rem First-run bootstrap only. Use whatever usable Python 3 the PC has;
rem do not reject it merely because it is not 3.12/3.13/3.14.
rem ------------------------------------------------------------
set "BASEPY="
where py >nul 2>&1
if not errorlevel 1 (
    py -3 -c "import sys; raise SystemExit(0 if sys.version_info.major==3 else 1)" >nul 2>&1
    if not errorlevel 1 set "BASEPY=py -3"
)
if not defined BASEPY (
    where python >nul 2>&1
    if not errorlevel 1 (
        python -c "import sys; raise SystemExit(0 if sys.version_info.major==3 else 1)" >nul 2>&1
        if not errorlevel 1 set "BASEPY=python"
    )
)
if not defined BASEPY goto :no_python

if not exist "%RUNTIME%" mkdir "%RUNTIME%" >nul 2>&1
if exist "%VENV%" rmdir /s /q "%VENV%" >nul 2>&1

echo IceHaloStack first-run setup: creating a private Python environment...
%BASEPY% -m venv "%VENV%"
if errorlevel 1 goto :setup_fail

"%PY%" -m pip --version >nul 2>&1
if errorlevel 1 "%PY%" -m ensurepip --upgrade >nul 2>&1

echo Installing IceHaloStack components. This is only needed on first setup...
"%PY%" -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
if errorlevel 1 goto :setup_fail
"%PY%" -m pip install --disable-pip-version-check %PKGS%
if errorlevel 1 goto :setup_fail

"%PY%" -c "import numpy,PIL,tifffile,rawpy,cv2,imageio_ffmpeg,tkinter" >nul 2>&1
if errorlevel 1 goto :setup_fail

:launch
start "" "%PYW%" "%APP%"
popd
exit /b 0

:no_python
echo.
echo IceHaloStack could not find Python 3 on this PC.
echo Install any normal 64-bit Python 3 distribution, then double-click this launcher again.
echo No specific 3.12 / 3.13 / 3.14 whitelist is enforced.
echo.
pause
popd
exit /b 2

:setup_fail
echo.
echo IceHaloStack could not finish its first-run Python environment setup.
echo The program itself and your system Python were not modified.
echo.
pause
popd
exit /b 1
