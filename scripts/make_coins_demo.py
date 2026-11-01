"""生成「硬币照片自动检测 + 去水印」的对比图（回归示例）。

用法：python scripts/make_coins_demo.py
"""

import json
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wmkit import pipeline  # noqa: E402

SRC = "tests/data/coins_photo.png"
OUT = "out/coins_clean.png"
CMP = "out/coins_auto_compare.png"

report = pipeline.process(SRC, OUT, method="harmonic")
with open("out/coins_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("候选:", [(c["bbox"], round(c["score"], 2)) for c in report["candidates"]])
print("分析区域:", report["region"], "掩膜:", report["removal"]["mask_pixels"])

before = Image.open(SRC).convert("RGB")
after = Image.open(OUT).convert("RGB")
d = ImageDraw.Draw(before)
for i, c in enumerate(report["candidates"]):
    x0, y0, x1, y1 = c["bbox"]
    d.rectangle([x0, y0, x1, y1], outline=(255, 170, 0) if i == 0 else (0, 200, 220), width=3)

x0, y0, x1, y1 = report["region"]
box = (max(0, x0 - 30), max(0, y0 - 30), min(before.width, x1 + 20), min(before.height, y1 + 20))
zb = before.crop(box).resize(((box[2] - box[0]) * 3, (box[3] - box[1]) * 3), Image.LANCZOS)
za = after.crop(box).resize(((box[2] - box[0]) * 3, (box[3] - box[1]) * 3), Image.LANCZOS)

top = Image.new("RGB", (before.width * 2 + 14, before.height), "white")
top.paste(before, (0, 0))
top.paste(after, (before.width + 14, 0))

bottom = Image.new("RGB", (zb.width + za.width + 14, max(zb.height, za.height)), "white")
bottom.paste(zb, (0, 0))
bottom.paste(za, (zb.width + 14, 0))

cmp_img = Image.new("RGB", (max(top.width, bottom.width), top.height + bottom.height + 30), "white")
cmp_img.paste(top, (0, 26))
cmp_img.paste(bottom, (0, top.height + 30))
dr = ImageDraw.Draw(cmp_img)
dr.text((8, 6), "BEFORE  (orange = chosen region, cyan = other candidates)", fill=(200, 120, 0))
dr.text((before.width + 22, 6), "AFTER", fill=(20, 140, 60))
cmp_img.save(CMP)
print("done ->", CMP)
