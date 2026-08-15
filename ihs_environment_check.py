from __future__ import annotations

import importlib
import importlib.metadata as metadata
import platform
import struct
import sys
from pathlib import Path

MIN_PY = (3, 12)
MAX_PY = (3, 14)


def dist_version(name: str) -> str:
    try:
        return metadata.version(name)
    except Exception:
        return "unknown"


def check_import(module_name: str, dist_name: str, label: str):
    try:
        importlib.import_module(module_name)
        return True, f"{label:<18} {dist_version(dist_name):<14} OK"
    except Exception as exc:
        return False, f"{label:<18} FAILED         {type(exc).__name__}: {exc}"


def main() -> int:
    bits = struct.calcsize("P") * 8
    pyver = sys.version_info[:3]
    supported = MIN_PY <= sys.version_info[:2] <= MAX_PY and bits == 64

    print("============================================================")
    print("IceHaloStack Environment Check")
    print("============================================================")
    print(f"Python             {pyver[0]}.{pyver[1]}.{pyver[2]} ({bits}-bit) {'OK' if supported else 'UNSUPPORTED'}")
    print(f"Executable         {sys.executable}")
    print(f"Platform           {platform.platform()}")

    results = []
    for module_name, dist_name, label in (
        ("numpy", "numpy", "NumPy"),
        ("PIL", "pillow", "Pillow"),
        ("tifffile", "tifffile", "tifffile"),
        ("rawpy", "rawpy", "rawpy / LibRaw"),
        ("cv2", "opencv-python-headless", "OpenCV"),
        ("imageio_ffmpeg", "imageio-ffmpeg", "imageio-ffmpeg"),
    ):
        ok, line = check_import(module_name, dist_name, label)
        results.append(ok)
        print(line)

    try:
        import tkinter as tk
        print(f"Tkinter / Tk       {tk.TkVersion:<14} OK")
        results.append(True)
    except Exception as exc:
        print(f"Tkinter / Tk       FAILED         {type(exc).__name__}: {exc}")
        results.append(False)

    # Extra practical checks used by the application.
    try:
        import rawpy
        print(f"LibRaw             {getattr(rawpy, 'libraw_version', 'unknown')}")
    except Exception:
        pass

    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"FFmpeg             {Path(ffmpeg_exe).name} OK")
    except Exception as exc:
        print(f"FFmpeg             FAILED         {type(exc).__name__}: {exc}")
        results.append(False)

    print("------------------------------------------------------------")
    if supported and all(results):
        print("ENVIRONMENT_READY")
        return 0

    if not supported:
        print("Required Python: CPython 3.12, 3.13, or 3.14, 64-bit.")
    print("ENVIRONMENT_NOT_READY")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
