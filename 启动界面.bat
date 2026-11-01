@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在启动水印去除工具...
python wm.py gui
if errorlevel 1 (
  echo.
  echo 启动失败：请确认已安装 Python 和依赖（pip install -r requirements.txt）
  pause
)
