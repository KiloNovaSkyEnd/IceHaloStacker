@echo off
setlocal EnableExtensions
chcp 65001 >nul
pushd "%~dp0"
set "BUILDVENV=%LOCALAPPDATA%\IceHaloStackBuild09415\venv"
set "PY=%BUILDVENV%\Scripts\python.exe"

if not exist "%PY%" (
    echo Build environment is not initialized yet.
    echo Run build_release.bat once first.
    pause
    popd
    exit /b 1
)

if exist "build_debug" rmdir /s /q "build_debug"
if exist "dist_debug" rmdir /s /q "dist_debug"

"%PY%" -m PyInstaller --noconfirm --clean --onedir --console ^
  --name IceHaloStack_Debug ^
  --icon "assets\icon\icehalostack.ico" ^
  --collect-all rawpy ^
  --collect-all cv2 ^
  --collect-all imageio_ffmpeg ^
  --collect-all tifffile ^
  --collect-all PIL ^
  --distpath dist_debug ^
  --workpath build_debug ^
  icehalostack.py
if errorlevel 1 (
  echo DEBUG BUILD FAILED
  pause
  popd
  exit /b 1
)
explorer "dist_debug\IceHaloStack_Debug"
pause
popd
