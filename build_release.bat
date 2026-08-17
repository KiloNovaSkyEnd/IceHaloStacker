@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
pushd "%~dp0"

set "BUILDROOT=%LOCALAPPDATA%\IceHaloStackBuild09415"
set "BUILDVENV=%BUILDROOT%\venv"
set "PY=%BUILDVENV%\Scripts\python.exe"
set "RUNTIMEPY=%LOCALAPPDATA%\IceHaloStackRuntime0946\venv\Scripts\python.exe"
set "BASEPY="
set "BASEMODE="
set "PKGS=numpy pillow tifffile rawpy opencv-python-headless imageio-ffmpeg pyinstaller"

echo ============================================================
echo IceHaloStack v0.9.4.17 - Windows Standalone EXE Builder
echo Build environment: %BUILDROOT%
echo Output: %~dp0dist\IceHaloStack\IceHaloStack.exe
echo ============================================================
echo.
echo Python policy: any usable 64-bit Python 3 is accepted.
echo No Python 3.12 / 3.13 / 3.14 whitelist is enforced.
echo.

rem ------------------------------------------------------------
rem 1) Reuse an existing build environment if it is a usable
rem    64-bit Python 3 installation.
rem ------------------------------------------------------------
if exist "%PY%" (
    "%PY%" -c "import sys,struct; raise SystemExit(0 if (sys.version_info.major==3 and struct.calcsize('P')*8==64) else 1)" >nul 2>&1
    if not errorlevel 1 goto :pip
    echo Existing build environment is invalid. Rebuilding it...
    rmdir /s /q "%BUILDVENV%" >nul 2>&1
)

rem ------------------------------------------------------------
rem 2) Prefer IceHaloStack's already working private runtime.
rem ------------------------------------------------------------
if exist "%RUNTIMEPY%" (
    "%RUNTIMEPY%" -c "import sys,struct; raise SystemExit(0 if (sys.version_info.major==3 and struct.calcsize('P')*8==64) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "BASEPY=%RUNTIMEPY%"
        set "BASEMODE=path"
        goto :create_venv
    )
)

rem ------------------------------------------------------------
rem 3) Otherwise accept whatever 64-bit Python 3 the PC provides.
rem ------------------------------------------------------------
where py >nul 2>&1
if not errorlevel 1 (
    py -3 -c "import sys,struct; raise SystemExit(0 if (sys.version_info.major==3 and struct.calcsize('P')*8==64) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "BASEPY=py"
        set "BASEMODE=launcher"
        goto :create_venv
    )
)

where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys,struct; raise SystemExit(0 if (sys.version_info.major==3 and struct.calcsize('P')*8==64) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "BASEPY=python"
        set "BASEMODE=command"
        goto :create_venv
    )
)

goto :nopython

:create_venv
if not exist "%BUILDROOT%" mkdir "%BUILDROOT%" >nul 2>&1
if exist "%BUILDVENV%" rmdir /s /q "%BUILDVENV%" >nul 2>&1

echo Creating isolated build environment...
if "%BASEMODE%"=="launcher" (
    py -3 -m venv "%BUILDVENV%"
) else if "%BASEMODE%"=="path" (
    "%BASEPY%" -m venv "%BUILDVENV%"
) else (
    python -m venv "%BUILDVENV%"
)
if errorlevel 1 goto :venv_fail

:pip
"%PY%" -m pip --version >nul 2>&1
if errorlevel 1 (
    echo pip is missing. Trying ensurepip...
    "%PY%" -m ensurepip --upgrade
)
if errorlevel 1 goto :fail

echo [1/4] Installing/updating build dependencies...
"%PY%" -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
if errorlevel 1 goto :fail
"%PY%" -m pip install --disable-pip-version-check %PKGS%
if errorlevel 1 goto :dependency_fail

echo.
"%PY%" -c "import sys,struct; print('Build Python:',sys.version.split()[0],str(struct.calcsize('P')*8)+'-bit')"
"%PY%" -m PyInstaller --version

echo [2/4] Cleaning previous output...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"

echo [3/4] Building standalone application with IceHaloStack icon...
"%PY%" -m PyInstaller --noconfirm --clean "IceHaloStack.spec"
if errorlevel 1 goto :fail

echo [4/4] Verifying output...
if not exist "dist\IceHaloStack\IceHaloStack.exe" goto :missing

echo.
echo ============================================================
echo BUILD SUCCESSFUL
echo ============================================================
echo EXE:
echo %~dp0dist\IceHaloStack\IceHaloStack.exe
echo.
echo IMPORTANT:
echo GitHub Release should contain the whole dist\IceHaloStack folder,
echo normally compressed as a ZIP. Do not upload IceHaloStack.exe alone,
echo because this build uses PyInstaller onedir mode.
echo.
explorer "dist\IceHaloStack"
pause
popd
exit /b 0

:nopython
echo.
echo ERROR: No usable 64-bit Python 3 was found.
echo.
echo If launch_IceHaloStack.bat already works, first run it once and then
echo retry this builder. The builder will reuse:
echo %LOCALAPPDATA%\IceHaloStackRuntime0946\venv\Scripts\python.exe
echo.
echo There is NO 3.12 / 3.13 / 3.14 version whitelist anymore.
pause
popd
exit /b 2

:venv_fail
echo.
echo ERROR: Python 3 was found, but creating the isolated build environment failed.
echo Try deleting this folder and run the builder again:
echo %BUILDROOT%
pause
popd
exit /b 3

:dependency_fail
echo.
echo ERROR: Python was accepted, but one or more build dependencies could not be installed.
echo The console output above should show which package failed for this Python version.
echo Your normal IceHaloStack runtime and system Python were not modified.
pause
popd
exit /b 4

:missing
echo.
echo ERROR: PyInstaller completed but IceHaloStack.exe was not found.
pause
popd
exit /b 1

:fail
echo.
echo ERROR: Build failed. Please keep the full console output for diagnosis.
pause
popd
exit /b 1
