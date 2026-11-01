"""样本匹配示例：从海报里截一个水印样本，再在一张“换过尺寸/亮度”的图上定位并去除。

用法（项目根目录）：
    D:\\anaconda3\\envs\\new_env\\python.exe scripts\\make_sample_demo.py
"""

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wmkit import imageio, pipeline, template  # noqa: E402

os.makedirs("samples", exist_ok=True)
os.makedirs("out", exist_ok=True)

SRC = "images/poster_icu.png"
SAMPLE = "samples/doubao_watermark.npz"
TARGET = "out/_sample_target.png"
CLEAN = "out/sample_demo_clean.png"

# 1) 从原图截取样本（水印所在区域）
rgb, _, _ = imageio.load_image(SRC)
sample = template.build_sample(rgb, (860, 1340, 220, 100), name=os.path.basename(SRC))
template.save_sample(SAMPLE, sample)
print("样本:", sample.summary())

# 2) 造一张“另一个来源”的图：放大 1.18、整体调亮一点
img = Image.open(SRC).convert("RGB")
scaled = img.resize((int(img.width * 1.18), int(img.height * 1.18)), Image.LANCZOS)
scaled = ImageEnhance.Brightness(scaled).enhance(1.06)
scaled.save(TARGET)

# 3) 用样本定位 + 去除
report = pipeline.process_sample(TARGET, CLEAN, sample)
with open("out/sample_demo_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("匹配:", report["matches"])

# 4) 画对比图：左边框出匹配位置，右边是结果
before = Image.open(TARGET).convert("RGB")
after = Image.open(CLEAN).convert("RGB")
d = ImageDraw.Draw(before)
for m in report["matches"]:
    x0, y0, x1, y1 = m["bbox"]
    d.rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4], outline=(255, 40, 40), width=4)
    d.text((x0, max(0, y0 - 16)), f"match {m['score']:.2f}", fill=(255, 40, 40))

w, h = before.size
scale = 900 / w
thumb_b = before.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
thumb_a = after.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
gap = 14
cmp_img = Image.new("RGB", (thumb_b.width * 2 + gap, thumb_b.height + 26), "white")
cmp_img.paste(thumb_b, (0, 26))
cmp_img.paste(thumb_a, (thumb_b.width + gap, 26))
dc = ImageDraw.Draw(cmp_img)
dc.text((8, 6), "BEFORE  (red = located by sample)", fill=(200, 30, 30))
dc.text((thumb_b.width + gap + 8, 6), "AFTER", fill=(20, 140, 60))
cmp_img.save("out/sample_demo_compare.png")
print("done ->", CLEAN, "out/sample_demo_compare.png")
