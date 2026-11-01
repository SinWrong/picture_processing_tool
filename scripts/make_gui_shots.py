"""生成 README 用的图形界面截图（docs/gui-*.png）。

用法：python scripts/make_gui_shots.py
需要图形桌面；无桌面环境时会报错退出。
"""

import ctypes
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk

from PIL import Image, ImageGrab

import wmkit.gui as gui

DOCS = "docs"
os.makedirs(DOCS, exist_ok=True)
OUT_W = 920          # 截图统一宽度（等比缩放，方便并排看）

for name in ("showinfo", "showerror", "showwarning"):
    setattr(gui.messagebox, name, lambda *a, **k: None)


def shot(root, path: str):
    """整屏抓图再按“物理/逻辑”比例裁出窗口（高 DPI 缩放也准）。"""
    from PIL import ImageStat

    img = None
    for _ in range(3):
        root.attributes("-topmost", True)
        root.lift()
        root.focus_force()
        root.update_idletasks()
        root.update()
        time.sleep(0.6)
        full = ImageGrab.grab()
        scale = full.width / float(root.winfo_screenwidth())
        x, y = root.winfo_rootx() * scale, root.winfo_rooty() * scale
        w, h = root.winfo_width() * scale, root.winfo_height() * scale
        img = full.crop((int(x), int(y), int(x + w), int(y + h)))
        # 抓黑了（窗口被遮/重绘中）就再试一次
        if ImageStat.Stat(img.convert("L")).stddev[0] > 3.0:
            break
    if img.width != OUT_W:
        img = img.resize((OUT_W, int(img.height * OUT_W / img.width)), Image.LANCZOS)
    img.save(path)
    print("saved", path, img.size)


root = tk.Tk()
root.geometry("820x620+80+40")
root.update()
app = gui.App(root)
root.attributes("-topmost", True)
root.lift()
root.focus_force()
for _ in range(4):
    time.sleep(0.3)
    root.update()

app.load_folder(os.path.join("tests", "data"))
root.update()

target = os.path.join("tests", "data", "doubao_target.png")
app.tree.selection_set(target)
app.on_tree_select()
root.update()

# 1) 画好选区（还没处理）
app.region = (860, 1340, 1080, 1440)      # (x0,y0,x1,y1)，右下角“豆包AI生成”
app.region_pct = (860 / 1080, 1340 / 1440, 220 / 1080, 100 / 1440)
app.status.set("已选择区域 (860,1340)-(1080,1440)，点“去除水印”。")
app.render()
root.update()
shot(root, os.path.join(DOCS, "gui-1-select.png"))

# 2) 勾“显示掩膜”：红色就是要擦掉的像素
app._ensure_analysis()
app.show_mask.set(True)
app.render()
root.update()
shot(root, os.path.join(DOCS, "gui-2-mask.png"))

# 3) 去除之后（关掉掩膜显示，看真实结果）
app.show_mask.set(False)
app.do_remove()
root.update()
shot(root, os.path.join(DOCS, "gui-3-done.png"))

# 4) 批量处理后的列表（真实跑一遍，再把状态刷成“完成”）
import shutil  # noqa: E402

from wmkit import pipeline  # noqa: E402

tmp_out = os.path.join(DOCS, "_tmp_out")
shutil.rmtree(tmp_out, ignore_errors=True)
files = list(app.files)
pipeline.process_many(files, tmp_out, auto=True)
for f in files:
    app._set_status(f, "done")
app.open_path(target)
app.region = (860, 1340, 1080, 1440)
app.status.set(f"批量完成：成功 {len(files)}，失败 0。输出目录：{tmp_out}")
app.render()
root.update()
shot(root, os.path.join(DOCS, "gui-4-batch.png"))
shutil.rmtree(tmp_out, ignore_errors=True)

# 5) 对比视图（换一张小图，左右两半都看得清）
corner = os.path.join("tests", "data", "wm_corner.png")
if os.path.exists(corner):
    app.tree.selection_set(corner)
    app.on_tree_select()
    root.update()
    app.region = (69, 39, 297, 141)       # 珊瑚上的“南海网”彩色 logo
    app.region_pct = None
    app.show_mask.set(False)
    app.compare.set(True)
    app.do_remove()
    root.update()
    shot(root, os.path.join(DOCS, "gui-5-compare.png"))

root.destroy()
print("done")
