
# IceHaloStacker

堆栈固定机位拍摄的冰晕延时，进行平均值 / 最大值堆栈、节点式处理、实时预览与延时导出。

## 当前包内容

- `icehalostack.py`：主程序源码
- `launch_IceHaloStack.bat`：直接启动脚本
- `build_release.bat`：一键构建 Windows EXE
- `IceHaloStack.spec`：PyInstaller 打包配置
- `assets/icon/icehalostack.ico`：软件图标（ICO）
- `assets/icon/icehalostack_icon.png`：软件图标（PNG）

## 推荐仓库结构

- 将本文件夹全部内容上传到 GitHub 仓库根目录。
- `dist/IceHaloStack/` 下生成的内容适合打包到 GitHub Releases。

## 本地运行

双击 `launch_IceHaloStack.bat`。

## 构建 Windows EXE

双击 `build_release.bat`。

构建成功后，输出位置通常为：

```
dist/IceHaloStack/IceHaloStack.exe
```

## 图标设计说明

图标采用简洁冰晕主题：中心太阳、22° 晕环、两侧幻日、下方弧线提示大气光学结构。
