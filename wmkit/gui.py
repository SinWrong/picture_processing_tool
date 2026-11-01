"""图形界面：打开图片 / 打开文件夹 → 框选或自动检测（或用样本定位）→ 去除水印 → 批量保存。

只依赖标准库 tkinter + Pillow。运行：python wm.py gui

界面分三块：
- 顶部工具栏：打开图片 / 打开文件夹 / 输出目录 / 自动检测 / 去除水印 / 还原 / 保存 /
  载入样本 / 选区存为样本 / 清除样本
- 左侧：当前文件夹里的图片列表（含处理状态），双击某一行可只处理这一张
- 右侧：预览画布，鼠标拖拽即框选水印区域

批量处理会按“输出目录”逐张处理：带了水印样本时先按样本定位；否则选区按**百分比**
套用到每张图，没画选区时自动检测。
"""

from __future__ import annotations

import os
import queue
import threading

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk

from . import analysis, imageio, pipeline, template

PREVIEW_MAX = 900
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

STATUS_TEXT = {"pending": "待处理", "running": "处理中…", "done": "完成", "fail": "失败"}
STATUS_TAG = {"pending": "muted", "running": "active", "done": "ok", "fail": "bad"}


class App:
    def __init__(self, root: tk.Tk, initial: str | None = None):
        self.root = root
        root.title("水印分析 / 去除工具 — wm")
        root.geometry("800x600")
        root.minsize(720, 520)

        # ---- 当前图片状态 ----
        self.path: str | None = None
        self.rgb: np.ndarray | None = None
        self.alpha = None
        self.mask = None
        self.analysis = None
        self.result: np.ndarray | None = None

        # ---- 文件夹 / 批量状态 ----
        self.folder: str | None = None
        self.out_dir: str | None = None
        self.files: list[str] = []
        self.items: dict[str, str] = {}          # path -> status
        self.batch_thread: threading.Thread | None = None
        self.batch_queue: queue.Queue = queue.Queue()
        self.stop_flag = threading.Event()

        # ---- 选区（像素 + 百分比）----
        self.region: tuple[int, int, int, int] | None = None
        self.region_pct: tuple[float, float, float, float] | None = None
        self.drag_start = None
        self.drag_rect = None

        # ---- 水印样本 ----
        self.sample = None                 # template.Sample
        self.sample_path: str | None = None

        self.compare = tk.BooleanVar(value=True)      # 处理后左右对比
        self.show_mask = tk.BooleanVar(value=False)
        self.apply_ratio = tk.BooleanVar(value=True)
        self.texture_var = tk.BooleanVar(value=False)
        self.scale = 1.0
        self.offset = (0, 0)
        self._photos = []

        self._build_ui()
        self._poll_batch()
        if initial:
            self.root.after(50, lambda: self.open_any(initial))

    # ================================================================== UI
    def _build_ui(self):
        # ---- 第一行：文件 + 处理 ----
        bar = ttk.Frame(self.root, padding=(6, 5, 6, 0))
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(bar, text="打开图片", command=self.open_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="打开文件夹", command=self.open_folder_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="输出目录…", command=self.choose_out_dir).pack(side=tk.LEFT, padx=2)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(bar, text="自动检测", command=self.auto_detect).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="去除水印", command=self.do_remove).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="还原", command=self.reset).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="保存结果", command=self.save).pack(side=tk.LEFT, padx=2)

        # ---- 第二行：样本 + 批量 ----
        bar1 = ttk.Frame(self.root, padding=(6, 3, 6, 0))
        bar1.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(bar1, text="载入样本…", command=self.load_sample_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar1, text="选区存为样本", command=self.save_sample_from_region).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar1, text="查看样本", command=self.show_sample_preview).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar1, text="清除样本", command=self.clear_sample).pack(side=tk.LEFT, padx=2)
        ttk.Separator(bar1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        self.btn_batch = ttk.Button(bar1, text="批量处理全部", command=self.batch_start)
        self.btn_batch.pack(side=tk.LEFT, padx=2)
        self.btn_stop = ttk.Button(bar1, text="停止", command=self.batch_stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=2)
        self.sample_label = tk.StringVar(value="样本：无")

        # ---- 第三行：参数 ----
        bar2 = ttk.Frame(self.root, padding=(6, 3, 6, 4))
        bar2.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(bar2, text="灵敏度").pack(side=tk.LEFT)
        self.sens = tk.DoubleVar(value=0.5)
        ttk.Scale(bar2, from_=0.0, to=1.0, variable=self.sens, length=100).pack(
            side=tk.LEFT, padx=(4, 8))

        ttk.Label(bar2, text="算法").pack(side=tk.LEFT)
        self.method = ttk.Combobox(bar2, values=["harmonic", "telea", "fast"], width=8, state="readonly")
        self.method.set("harmonic")
        self.method.pack(side=tk.LEFT, padx=(4, 8))

        ttk.Checkbutton(bar2, text="纹理", variable=self.texture_var).pack(side=tk.LEFT)
        ttk.Checkbutton(bar2, text="掩膜", variable=self.show_mask,
                        command=self.render).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Checkbutton(bar2, text="按比例", variable=self.apply_ratio).pack(
            side=tk.LEFT, padx=(6, 0))
        ttk.Checkbutton(bar2, text="对比", variable=self.compare,
                        command=self.render).pack(side=tk.LEFT, padx=(6, 0))

        self.progress = ttk.Progressbar(bar2, mode="determinate", length=120)
        self.progress.pack(side=tk.RIGHT, padx=(6, 0))

        # ---- 主体：左列表 + 右画布 ----
        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned, width=232)
        paned.add(left, weight=0)
        ttk.Label(left, text="文件夹中的图片", padding=(6, 4)).pack(side=tk.TOP, anchor="w")
        self.tree = ttk.Treeview(left, columns=("status", "name"), show="headings", height=14)
        self.tree.heading("status", text="状态")
        self.tree.heading("name", text="文件")
        self.tree.column("status", width=60, anchor="center", stretch=False)
        self.tree.column("name", width=160, anchor="w")
        self.tree.tag_configure("ok", foreground="#1e8449")
        self.tree.tag_configure("bad", foreground="#c0392b")
        self.tree.tag_configure("active", foreground="#1f6feb")
        self.tree.tag_configure("muted", foreground="#777777")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.tree.yview)
        scroll.pack(side=tk.LEFT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.tree.bind("<Double-1>", self.on_tree_double)

        self.canvas = tk.Canvas(paned, bg="#2b2b2b", highlightthickness=0)
        paned.add(self.canvas, weight=1)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Configure>", lambda e: self.render())

        # ---- 状态栏 ----
        self.status = tk.StringVar(value="打开一张图片或一个文件夹开始；拖拽框选水印区域，或点“自动检测”。")
        sb = ttk.Frame(self.root)
        sb.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(sb, textvariable=self.status, anchor="w", padding=(6, 3)).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(sb, textvariable=self.sample_label, foreground="#555555",
                  padding=(6, 3)).pack(side=tk.RIGHT)

    # ============================================================== 文件入口
    def open_any(self, path: str):
        if os.path.isdir(path):
            self.load_folder(path)
        else:
            self.open_path(path)

    def open_dialog(self):
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff"), ("全部", "*.*")],
        )
        if path:
            self.open_path(path)

    def open_folder_dialog(self):
        folder = filedialog.askdirectory(title="选择包含图片的文件夹")
        if folder:
            self.load_folder(folder)

    def load_folder(self, folder: str):
        names = sorted(
            n for n in os.listdir(folder)
            if os.path.splitext(n)[1].lower() in IMAGE_EXTS
        )
        if not names:
            messagebox.showinfo("文件夹", "这个文件夹里没有支持的图片。")
            return
        self.folder = folder
        self.files = [os.path.join(folder, n) for n in names]
        self.items = {p: "pending" for p in self.files}
        self.out_dir = os.path.join(folder, "out")
        self._refresh_tree()
        self.tree.selection_set(self.tree.get_children()[0])
        self.status.set(f"已载入 {len(self.files)} 张图片：{folder}；输出到 {self.out_dir}")

    def _refresh_tree(self):
        keep = None
        sel = self.tree.selection()
        if sel:
            keep = self.tree.item(sel[0], "values")[1]
        self.tree.delete(*self.tree.get_children())
        for path in self.files:
            status = self.items.get(path, "pending")
            self.tree.insert("", "end", iid=path,
                             values=(STATUS_TEXT[status], os.path.basename(path)),
                             tags=(STATUS_TAG[status],))
        if keep:
            for path in self.files:
                if os.path.basename(path) == keep:
                    self.tree.selection_set(path)
                    break

    def _set_status(self, path: str, status: str):
        self.items[path] = status
        if self.tree.exists(path):
            self.tree.item(path, values=(STATUS_TEXT[status], os.path.basename(path)),
                           tags=(STATUS_TAG[status],))

    def open_path(self, path: str):
        try:
            rgb, alpha, _ = imageio.load_image(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("打不开", f"{path}\n{exc}")
            return
        self.path, self.rgb, self.alpha = path, rgb, alpha
        self.result = None
        self.mask = None
        self.analysis = None
        self.region = None
        self.region_pct = None
        h, w = rgb.shape[:2]
        self.status.set(f"{os.path.basename(path)}  {w}×{h}  拖拽框选水印区域，或点“自动检测”。")
        self.render()

    # ================================================================ 渲染
    def render(self):
        self.canvas.delete("all")
        img = self.result if self.result is not None else self.rgb
        if img is None:
            self.canvas.create_text(
                max(1, self.canvas.winfo_width()) // 2, max(1, self.canvas.winfo_height()) // 2,
                text="打开图片或文件夹开始", fill="#888888", font=("Microsoft YaHei UI", 14),
            )
            return
        h, w = img.shape[:2]
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        if cw < 60 or ch < 60:
            cw, ch = PREVIEW_MAX, PREVIEW_MAX

        # ---- 处理后：左右对比 ----
        if self.result is not None and self.rgb is not None and self.compare.get():
            gap = 10
            label_h = 22
            pane_w = max(40, (cw - gap) // 2)
            pane_h = max(40, ch - label_h)
            scale = min(pane_w / w, pane_h / h, 1.0)
            dw, dh = max(1, int(w * scale)), max(1, int(h * scale))
            oy = label_h + (pane_h - dh) // 2
            ox1 = (pane_w - dw) // 2
            ox2 = pane_w + gap + (pane_w - dw) // 2

            before = self.rgb
            after = self.result
            if self.show_mask.get() and self.mask is not None:
                after = imageio.overlay_visual(self.result, self.mask)
            photos = []
            for arr, ox, title, color in ((before, ox1, "修改前", "#e8e8e8"),
                                          (after, ox2, "修改后", "#7ee0a0")):
                pil = Image.fromarray(arr.astype(np.uint8)).resize((dw, dh), Image.LANCZOS)
                photo = ImageTk.PhotoImage(pil)
                photos.append(photo)          # 必须留引用，否则被 GC 掉变空白
                self.canvas.create_image(ox, oy, anchor=tk.NW, image=photo)
                self.canvas.create_rectangle(ox - 1, oy - 1, ox + dw, oy + dh,
                                             outline="#c8c8c8")
                self.canvas.create_text(
                    ox + dw // 2, max(8, label_h // 2 + 2), text=title,
                    fill=color, font=("Microsoft YaHei UI", 9, "bold"),
                )
            self._photos = photos
            # 画布上的鼠标坐标按“修改前”那半算，框选仍然作用于原图
            self.scale = scale
            self.offset = (ox1, oy)
            if self.region:
                x0, y0, x1, y1 = self.region
                self.canvas.create_rectangle(
                    ox1 + x0 * scale, oy + y0 * scale,
                    ox1 + x1 * scale, oy + y1 * scale,
                    outline="#ffb300", width=2,
                )
            return

        # ---- 单图 ----
        self.scale = min(cw / w, ch / h, 1.0)
        dw, dh = max(1, int(w * self.scale)), max(1, int(h * self.scale))
        display = img
        if self.show_mask.get() and self.mask is not None:
            display = imageio.overlay_visual(img, self.mask)
        pil = Image.fromarray(display.astype(np.uint8)).resize((dw, dh), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(pil)
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        self.offset = (ox, oy)
        self.canvas.create_image(ox, oy, anchor=tk.NW, image=self.photo)

        if self.analysis is not None and self.result is None:
            for i, c in enumerate(self.analysis.candidates):
                x0, y0, x1, y1 = c.bbox
                color = "#00e5ff" if i == 0 else "#7f8c8d"
                self.canvas.create_rectangle(
                    ox + x0 * self.scale, oy + y0 * self.scale,
                    ox + x1 * self.scale, oy + y1 * self.scale,
                    outline=color, width=2, dash=(4, 3),
                )
        if self.region and self.result is None:
            x0, y0, x1, y1 = self.region
            self.canvas.create_rectangle(
                ox + x0 * self.scale, oy + y0 * self.scale,
                ox + x1 * self.scale, oy + y1 * self.scale,
                outline="#ffb300", width=2,
            )

    # ================================================================ 交互
    def _to_image(self, event):
        ox, oy = self.offset
        return int((event.x - ox) / self.scale), int((event.y - oy) / self.scale)

    def on_press(self, event):
        if self.rgb is None or self.batch_thread is not None:
            return
        self.drag_start = self._to_image(event)
        self.drag_rect = None

    def on_drag(self, event):
        if self.drag_start is None:
            return
        x0, y0 = self.drag_start
        x1, y1 = self._to_image(event)
        ox, oy = self.offset
        if self.drag_rect:
            self.canvas.delete(self.drag_rect)
        self.drag_rect = self.canvas.create_rectangle(
            ox + min(x0, x1) * self.scale, oy + min(y0, y1) * self.scale,
            ox + max(x0, x1) * self.scale, oy + max(y0, y1) * self.scale,
            outline="#ffd54f", width=2,
        )

    def on_release(self, event):
        if self.drag_start is None:
            return
        x0, y0 = self.drag_start
        x1, y1 = self._to_image(event)
        self.drag_start = None
        if self.drag_rect:
            self.canvas.delete(self.drag_rect)
            self.drag_rect = None
        h, w = self.rgb.shape[:2]
        rx0, ry0 = max(0, min(x0, x1)), max(0, min(y0, y1))
        rx1, ry1 = min(w, max(x0, x1)), min(h, max(y0, y1))
        if rx1 - rx0 < 8 or ry1 - ry0 < 8:
            return
        self.region = (rx0, ry0, rx1, ry1)
        self.region_pct = (rx0 / w, ry0 / h, (rx1 - rx0) / w, (ry1 - ry0) / h)
        self.result = None
        self.analysis = None
        self.status.set(f"已选择区域 ({rx0},{ry0})-({rx1},{ry1})"
                        f"（相对尺寸 {self.region_pct[2] * 100:.1f}%×{self.region_pct[3] * 100:.1f}%，"
                        f"批量处理会按这个比例套用），点“去除水印”。")
        self.render()

    def on_tree_select(self, _event=None):
        sel = self.tree.selection()
        if not sel or self.batch_thread is not None:
            return
        path = sel[0]
        if path == self.path:
            return
        try:
            rgb, alpha, _ = imageio.load_image(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("打不开", f"{path}\n{exc}")
            return
        self.path, self.rgb, self.alpha = path, rgb, alpha
        self.result = None
        self.mask = None
        self.analysis = None
        h, w = rgb.shape[:2]
        if self.region_pct:
            x, y, rw, rh = self.region_pct
            self.region = (int(x * w), int(y * h), int((x + rw) * w), int((y + rh) * h))
        else:
            self.region = None
        self.status.set(f"{os.path.basename(path)}  {w}×{h}")
        self.render()

    def on_tree_double(self, _event=None):
        if self.batch_thread is not None:
            return
        sel = self.tree.selection()
        if sel:
            self.process_one(sel[0])

    # ================================================================ 操作
    def _sensitivity(self) -> float:
        return float(self.sens.get())

    def auto_detect(self):
        if self.rgb is None:
            return
        try:
            a = pipeline.analyse(self.rgb, self.alpha, region=None,
                                 sensitivity=self._sensitivity(), max_candidates=3)
        except Exception as exc:  # noqa: BLE001
            messagebox.showinfo("自动检测", str(exc))
            return
        self.analysis = a
        self.region = a.region
        h, w = self.rgb.shape[:2]
        x0, y0, x1, y1 = a.region
        self.region_pct = (x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h)
        self.mask = a.mask_result.mask
        self.result = None
        if a.candidates:
            c = a.candidates[0]
            self.status.set(f"检测到 {len(a.candidates)} 个候选，最高置信度 {c.score:.2f}；"
                            f"可勾选“显示掩膜”核对，或直接点“去除水印”。")
        self.render()

    def _resolve_region(self):
        """当前图片上要用的像素区域：优先用画出来的，否则用比例换算。"""
        if self.region is not None:
            return self.region
        if self.region_pct is not None and self.rgb is not None:
            h, w = self.rgb.shape[:2]
            x, y, rw, rh = self.region_pct
            return (int(x * w), int(y * h), int((x + rw) * w), int((y + rh) * h))
        return None

    def _ensure_analysis(self) -> bool:
        if self.analysis is not None and self.mask is not None:
            return True
        region = self._resolve_region()
        if region is None:
            messagebox.showinfo("提示", "请先用鼠标框选水印区域，或点“自动检测”。")
            return False
        self.analysis = pipeline.analyse(self.rgb, self.alpha, region=region,
                                         sensitivity=self._sensitivity(), auto=False)
        self.region = self.analysis.region
        self.mask = self.analysis.mask_result.mask
        return True

    def do_remove(self):
        if self.rgb is None or self.batch_thread is not None:
            return
        if self.sample is not None:
            self._remove_with_sample()
            return
        self.analysis = None
        if not self._ensure_analysis():
            return
        try:
            self.status.set("正在修复…")
            self.root.update_idletasks()
            self.result = pipeline.remove(
                self.rgb, self.mask,
                method=self.method.get(),
                texture_refine=bool(self.texture_var.get()) if hasattr(self, "texture_var") else False,
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("修复失败", str(exc))
            return
        self.status.set("完成。可勾选“显示掩膜”核对，或点“保存结果”。")
        self.render()

    def _remove_with_sample(self):
        """按样本定位并去除。"""
        try:
            self.status.set("正在按样本定位水印…")
            self.root.update_idletasks()
            bg = analysis.median_background(self.rgb, 31)
            matches = pipeline.locate_sample(self.rgb, self.sample, bg=bg, min_score=0.5)
            if not matches:
                messagebox.showinfo("样本匹配", "这张图里没找到和样本匹配的水印（可以调低样本阈值或换个样本）。")
                self.status.set("未找到匹配的水印。")
                return
            mask = pipeline.sample_mask(self.rgb, self.sample, matches,
                                        bg=bg, sensitivity=self._sensitivity())
            self.mask = mask
            self.status.set(f"匹配到 {len(matches)} 处水印，正在修复…")
            self.root.update_idletasks()
            self.result = pipeline.remove(self.rgb, mask, method=self.method.get(),
                                          texture_refine=bool(self.texture_var.get()))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("样本去水印失败", str(exc))
            return
        boxes = ", ".join(f"({m['bbox'][0]},{m['bbox'][1]})" for m in matches[:4])
        self.status.set(f"完成：匹配 {len(matches)} 处 {boxes}，可保存或勾选“显示掩膜”核对。")
        self.render()

    # ---------------------------------------------------------------- 样本
    def load_sample_dialog(self):
        path = filedialog.askopenfilename(
            title="选择水印样本（.npz，或一张水印图片）",
            filetypes=[
                ("水印样本 / 图片", "*.npz *.png *.jpg *.jpeg *.webp *.bmp"),
                ("水印样本 (.npz)", "*.npz"),
                ("图片", "*.png *.jpg *.jpeg *.webp *.bmp"),
                ("全部", "*.*"),
            ],
        )
        if not path:
            return
        self.load_sample(path)

    def load_sample(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext not in (".npz", ".npy"):
            # 图片：太大的话不适合整张当样本，先打开它让用户框选
            try:
                with Image.open(path) as im:
                    w, h = im.size
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("打不开", f"{path}\n{exc}")
                return
            if w * h > 250_000:
                self.open_path(path)
                messagebox.showinfo(
                    "用图片当样本",
                    "这张图比较大，已经打开。\n请框选水印区域，再点「选区存为样本」。",
                )
                return
        try:
            self.sample = template.load_sample_any(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("样本打不开", f"{path}\n{exc}")
            return
        self.sample_path = path
        self._show_sample_label(path)
        self.status.set("已载入水印样本。点“去除水印/批量处理全部”会先定位同款水印再处理。")

    def _show_sample_label(self, path: str):
        info = self.sample.summary()
        name = os.path.basename(path)
        if len(name) > 16:
            name = name[:13] + "…"
        self.sample_label.set(
            f"样本：{name}  {info['size'][0]}×{info['size'][1]}  {info['color']}"
        )

    def clear_sample(self):
        """卸载当前水印样本（只清内存，不动磁盘文件）。"""
        if self.sample is None:
            messagebox.showinfo("清除样本", "当前没有载入样本。")
            return
        name = os.path.basename(self.sample_path or "")
        self.sample = None
        self.sample_path = None
        self.sample_label.set("样本：无")
        self.status.set(f"已卸载样本{f'（{name}）' if name else ''}；"
                        f"之后“去除水印/批量处理”会改用选区或自动检测。")

    def save_sample_from_region(self):
        if self.rgb is None:
            return
        region = self._resolve_region()
        if region is None:
            messagebox.showinfo("提示", "先框选水印区域，再存为样本。")
            return
        default = os.path.splitext(os.path.basename(self.path or "sample"))[0] + "_wm.npz"
        path = filedialog.asksaveasfilename(
            title="保存水印样本", defaultextension=".npz",
            initialfile=default, filetypes=[("水印样本", "*.npz")],
        )
        if not path:
            return
        try:
            s = template.build_sample(self.rgb, region, sensitivity=self._sensitivity(),
                                      name=os.path.basename(self.path or ""))
            template.save_sample(path, s)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("样本保存失败", str(exc))
            return
        self.sample = s
        self.sample_path = path
        self._show_sample_label(path)
        self.status.set(f"样本已保存：{path}")

    def reset(self):
        if self.rgb is None:
            return
        self.result = None
        self.render()
        self.status.set("已还原为原图。")

    def choose_out_dir(self):
        folder = filedialog.askdirectory(title="选择输出文件夹")
        if folder:
            self.out_dir = folder
            self.status.set(f"输出目录：{folder}")

    def _default_out_path(self) -> str:
        stem = os.path.splitext(os.path.basename(self.path or "output"))[0]
        base = self.out_dir or os.path.dirname(self.path or ".")
        return os.path.join(base, f"{stem}_clean.png")

    def save(self):
        if self.result is None:
            messagebox.showinfo("提示", "还没有结果可保存。")
            return
        path = self._default_out_path()
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            if os.path.exists(path):
                if not messagebox.askyesno("文件已存在", f"{path}\n要覆盖吗？"):
                    return
            imageio.save_image(path, self.result, alpha=self.alpha)
            self._set_status(self.path, "done")
            self.status.set(f"已保存: {path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("保存失败", str(exc))

    # ============================================================== 批量处理
    def show_sample_preview(self):
        """弹窗看“水印样本”长什么样（棋盘格底 = 透明部分）。"""
        if self.sample is None:
            messagebox.showinfo("水印样本", "还没有载入样本。")
            return
        info = self.sample.summary()
        scale = 3 if max(self.sample.alpha.shape) < 160 else 2
        img = template.stamp_preview(self.sample, scale=scale)
        win = tk.Toplevel(self.root)
        win.title(f"水印样本 — {info['size'][0]}×{info['size'][1]}")
        photo = ImageTk.PhotoImage(img)
        lbl = ttk.Label(win, image=photo)
        lbl.image = photo
        lbl.pack(padx=10, pady=(10, 4))
        ttk.Label(win, text=f"尺寸 {info['size'][0]}×{info['size'][1]}   "
                            f"叠加色 {info['color']}   笔画 {info['alpha_pixels']} px").pack(pady=(0, 10))

    def batch_start(self):
        if self.batch_thread is not None:
            return
        if not self.files:
            messagebox.showinfo("批量处理", "先点“打开文件夹”选一个图片文件夹。")
            return
        if self.sample is not None:
            desc = "使用已载入的水印样本，先定位再逐张去除。"
        elif self._resolve_region():
            desc = "使用当前选区（按比例套用到每张图）。"
        else:
            desc = "没有选区，将逐张自动检测水印区域。"
        if not messagebox.askyesno(
            "批量处理",
            f"将处理 {len(self.files)} 张图片，结果保存到：\n{self.out_dir}\n\n{desc}"
        ):
            return
        out_dir = self.out_dir or os.path.join(self.folder or ".", "out")
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir

        self.stop_flag.clear()
        self.progress.configure(maximum=len(self.files), value=0)
        self.btn_batch.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        for p in self.files:
            self._set_status(p, "pending")

        region = self._resolve_region() if self.apply_ratio.get() else self.region
        pct = None
        if region is not None and self.apply_ratio.get() and self.rgb is not None:
            h, w = self.rgb.shape[:2]
            x0, y0, x1, y1 = region
            pct = (x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h)
        elif region is not None and not self.apply_ratio.get():
            pct = ("px", region)

        args = dict(out_dir=out_dir, pct=pct, method=self.method.get(),
                    texture=bool(self.texture_var.get()),
                    sensitivity=self._sensitivity(),
                    sample=self.sample)
        files = list(self.files)
        self.batch_thread = threading.Thread(target=self._batch_worker, args=(files, args), daemon=True)
        self.batch_thread.start()
        self.status.set(f"开始批量处理 {len(files)} 张…")

    def _batch_worker(self, files: list[str], args: dict):
        q = self.batch_queue
        out_dir, pct = args["out_dir"], args["pct"]
        ok = fail = skipped = 0

        def _region_for(path):
            if pct is None:
                return None
            if pct[0] == "px":
                return pct[1]
            rgb, _alpha, _ = imageio.load_image(path)
            hh, ww = rgb.shape[:2]
            x, y, rw, rh = pct
            return (max(0, int(x * ww)), max(0, int(y * hh)),
                    max(1, min(ww, int((x + rw) * ww))),
                    max(1, min(hh, int((y + rh) * hh))))

        region = _region_for(files[0]) if files else None
        if self.stop_flag.is_set():
            q.put(("done", 0, 0, len(files), out_dir))
            return
        q.put(("start", files[0], 0))

        def _on_item(i, path, dst, report, error):
            nonlocal ok, fail, skipped
            if error:
                kind, msg = error
                if kind == "skip":
                    skipped += 1
                    q.put(("skip", path, msg))
                else:
                    fail += 1
                    q.put(("fail", path, msg))
            else:
                ok += 1
                q.put(("ok", path, dst))
            if i + 1 < len(files):
                q.put(("start", files[i + 1], i + 1))

        try:
            pipeline.process_many(
                files, out_dir, region=region,
                sensitivity=args["sensitivity"], method=args["method"],
                texture_refine=args["texture"], auto=True,
                sample=args.get("sample"), on_item=_on_item,
            )
        except Exception as exc:  # noqa: BLE001
            q.put(("fail", files[0], str(exc)))
        q.put(("done", ok, fail, skipped, out_dir))

    def _poll_batch(self):
        try:
            while True:
                msg = self.batch_queue.get_nowait()
                kind = msg[0]
                if kind == "start":
                    _k, path, i = msg
                    self._set_status(path, "running")
                    self.progress.configure(value=i)
                    self.status.set(f"正在处理 {os.path.basename(path)}（{i + 1}/{len(self.files)}）…")
                elif kind == "ok":
                    _k, path, _dst = msg
                    self._set_status(path, "done")
                elif kind == "fail":
                    _k, path, err = msg
                    self._set_status(path, "fail")
                    self.status.set(f"失败：{os.path.basename(path)} — {err}")
                elif kind == "skip":
                    _k, path, err = msg
                    self._set_status(path, "fail")
                    self.status.set(f"跳过：{os.path.basename(path)} — {err}")
                elif kind == "done":
                    _k, ok, fail, skipped, out_dir = msg
                    self.progress.configure(value=len(self.files))
                    self.btn_batch.configure(state=tk.NORMAL)
                    self.btn_stop.configure(state=tk.DISABLED)
                    self.batch_thread = None
                    text = f"批量完成：成功 {ok}，失败 {fail}"
                    if skipped:
                        text += f"，跳过 {skipped}"
                    text += f"。输出目录：{out_dir}"
                    self.status.set(text)
                    if not self.stop_flag.is_set():
                        messagebox.showinfo("批量处理", text)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_batch)

    def batch_stop(self):
        if self.batch_thread is not None:
            self.stop_flag.set()
            self.status.set("正在停止…（当前这张处理完就停）")


def main(initial: str | None = None):
    root = tk.Tk()
    try:
        root.iconphoto(True, ImageTk.PhotoImage(Image.new("RGB", (16, 16), "#2e86de")))
    except Exception:  # noqa: BLE001
        pass
    app = App(root, initial=initial)
    if not initial:
        app.render()
    root.mainloop()


if __name__ == "__main__":
    import sys as _sys
    main(_sys.argv[1] if len(_sys.argv) > 1 else None)
