@echo off
setlocal EnableExtensions
chcp 65001 >nul
pushd "%~dp0"
set "VENV_PY=%LOCALAPPDATA%\IceHaloStackRuntime0946\venv\Scripts\python.exe"
if exist "%VENV_PY%" (
  "%VENV_PY%" "%~dp0repair_failed_video_export.py"
  popd
  exit /b %errorlevel%
)
where py >nul 2>&1
if not errorlevel 1 (
  py -3 "%~dp0repair_failed_video_export.py"
  popd
  exit /b %errorlevel%
)
where python >nul 2>&1
if not errorlevel 1 (
  python "%~dp0repair_failed_video_export.py"
  popd
  exit /b %errorlevel%
)
echo ERROR: Python runtime not found. Run launch_IceHaloStack.bat first.
pause
popd
exit /b 2
