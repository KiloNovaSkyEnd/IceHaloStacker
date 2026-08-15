@echo off
setlocal EnableExtensions
chcp 65001 >nul
pushd "%~dp0"
set "VENV=%LOCALAPPDATA%\IceHaloStackRuntime0946\venv"
set "PY=%VENV%\Scripts\python.exe"

echo ============================================================
echo IceHaloStack v0.9.4.2 - Environment Diagnostic
echo ============================================================
if not exist "%PY%" (
    echo IceHaloStack private environment does not exist yet.
    echo Run launch_IceHaloStack.bat first.
    pause
    popd
    exit /b 1
)
"%PY%" "%~dp0ihs_environment_check.py"
echo.
echo Runtime folder:
echo %VENV%
pause
popd
