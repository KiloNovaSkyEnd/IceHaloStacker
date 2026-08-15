# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

block_cipher = None

datas = []
binaries = []
hiddenimports = []

# Packages used dynamically by IceHaloStack and/or carrying native DLL/data files.
for package in (
    'numpy',
    'PIL',
    'tifffile',
    'rawpy',
    'cv2',
    'imageio_ffmpeg',
):
    try:
        d, b, h = collect_all(package)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        # Built-in PyInstaller hooks may already handle some packages.
        pass

hiddenimports += [
    'numpy',
    'PIL.Image',
    'PIL.ImageTk',
    'PIL.ImageFilter',
    'tifffile',
    'rawpy',
    'cv2',
    'imageio_ffmpeg',
]

# Avoid duplicate entries while keeping order stable.
def _dedup(seq):
    out = []
    seen = set()
    for item in seq:
        try:
            key = repr(item)
        except Exception:
            key = str(item)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out

datas = _dedup(datas)
binaries = _dedup(binaries)
hiddenimports = _dedup(hiddenimports)

a = Analysis(
    ['icehalostack.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib', 'pandas', 'scipy', 'IPython', 'jupyter',
        'pytest', 'setuptools.tests',
    ],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='IceHaloStack',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon/icehalostack.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='IceHaloStack',
)
