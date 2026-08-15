@echo off
setlocal EnableExtensions
chcp 65001 >nul
pushd "%~dp0"
set "RUNTIME=%LOCALAPPDATA%\IceHaloStackRuntime0946"
set "VENV=%RUNTIME%\venv"

echo ============================================================
echo IceHaloStack v0.9.4.2 - Repair Private Python Environment
echo ============================================================
echo This removes ONLY IceHaloStack's private venv:
echo %VENV%
echo.
echo It does NOT remove your system Python or modify other applications.
echo ============================================================
echo.
choice /C YN /N /M "Rebuild the IceHaloStack environment now? [Y/N] "
if errorlevel 2 goto :cancel
if exist "%VENV%" rmdir /s /q "%VENV%"
if errorlevel 1 (
    echo ERROR: Could not remove the private venv. Close IceHaloStack and retry.
    pause
    popd
    exit /b 1
)
echo Private runtime removed. Starting Smart Launcher...
call "%~dp0launch_IceHaloStack.bat"
popd
exit /b %errorlevel%

:cancel
echo Cancelled.
popd
exit /b 0
