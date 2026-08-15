from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

APP='IceHaloStack Video Recovery'
EVEN_PAD=['-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:black']


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        p=imageio_ffmpeg.get_ffmpeg_exe()
        if p and Path(p).exists():
            return str(p)
    except Exception:
        pass
    p=shutil.which('ffmpeg')
    if p:
        return p
    local=Path(__file__).resolve().parent/('ffmpeg.exe' if os.name=='nt' else 'ffmpeg')
    return str(local) if local.exists() else None


def load_recipe(flow_dir: Path):
    p=flow_dir/'recipe.json'
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def find_video_frame_dirs(selected: Path):
    found=[]
    candidates=[]
    if selected.is_dir():
        candidates.extend([selected, selected/'_video_frames', selected/'sequence'])
        for p in selected.rglob('_video_frames'):
            candidates.append(p)
        for p in selected.rglob('sequence'):
            candidates.append(p)
    for p in candidates:
        if not p.is_dir() or p in found:
            continue
        if any(p.glob('frame_*.*')):
            found.append(p)
    return found


def run_checked(cmd):
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    p=subprocess.run(cmd,capture_output=True,text=True,creationflags=flags)
    if p.returncode!=0:
        raise RuntimeError((p.stderr or p.stdout or 'FFmpeg failed')[-4000:])


def encode_one(ff: str, frames: Path):
    flow=frames.parent
    recipe=load_recipe(flow)
    out=recipe.get('output',{}) if isinstance(recipe,dict) else {}
    fmt=str(out.get('video_format','MP4 H.264'))
    fps=str(max(0.1,float(out.get('fps',24.0))))
    files=sorted([p for p in frames.glob('frame_*.*') if p.is_file()])
    if not files:
        raise RuntimeError(f'没有找到可用于视频编码的序列帧：{frames}')
    ext=files[0].suffix.lower()
    pattern=str(frames/f'frame_%06d{ext}')
    name=flow.name
    if fmt=='MOV H.264':
        dest=flow/(name+'.mov')
        cmd=[ff,'-y','-framerate',fps,'-i',pattern]+EVEN_PAD+['-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p',str(dest)]
    elif fmt=='MOV ProRes':
        dest=flow/(name+'_ProRes.mov')
        cmd=[ff,'-y','-framerate',fps,'-i',pattern]+EVEN_PAD+['-c:v','prores_ks','-profile:v','3','-pix_fmt','yuv422p10le',str(dest)]
    elif fmt=='GIF':
        dest=flow/(name+'.gif')
        palette=frames/'palette_recovery.png'
        run_checked([ff,'-y','-framerate',fps,'-i',pattern,'-vf','palettegen=stats_mode=diff',str(palette)])
        cmd=[ff,'-y','-framerate',fps,'-i',pattern,'-i',str(palette),'-lavfi','paletteuse=dither=sierra2_4a',str(dest)]
    else:
        fmt='MP4 H.264'
        dest=flow/(name+'.mp4')
        cmd=[ff,'-y','-framerate',fps,'-i',pattern]+EVEN_PAD+['-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p',str(dest)]
    run_checked(cmd)
    return fmt, fps, dest


def main():
    root=tk.Tk();root.withdraw()
    chosen=filedialog.askdirectory(title='选择失败的 IceHaloStack_Timelapse 文件夹（或具体流程文件夹）')
    if not chosen:
        return 0
    base=Path(chosen)
    ff=ffmpeg_exe()
    if not ff:
        messagebox.showerror(APP,'未找到 FFmpeg。请先运行 launch_IceHaloStack.bat 初始化环境。')
        return 2
    frame_dirs=find_video_frame_dirs(base)
    if not frame_dirs:
        messagebox.showerror(APP,'没有找到可用于恢复的视频序列帧文件夹。\n\n请重新选择包含 sequence 或 _video_frames 的流程/延时输出目录。')
        return 3
    ok=[];fail=[]
    for frames in frame_dirs:
        try:
            fmt,fps,dest=encode_one(ff,frames)
            ok.append(f'{frames.parent.name}: {fmt} @ {fps} fps\n{dest}')
        except Exception as e:
            fail.append(f'{frames.parent}:\n{e}')
    msg=f'完成：{len(ok)} 个视频\n失败：{len(fail)} 个\n\n'
    if ok: msg+='成功：\n'+'\n\n'.join(ok[:12])
    if fail: msg+='\n\n失败：\n'+'\n\n'.join(fail[:5])
    msg+='\n\n为安全起见，sequence / _video_frames 序列帧没有自动删除。确认视频正常后可手动删除。'
    if fail:
        messagebox.showwarning(APP,msg)
        return 1
    messagebox.showinfo(APP,msg)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
