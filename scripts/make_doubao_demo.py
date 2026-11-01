"""生成「用 豆包AI生成.png 样本去除海报水印」的对比图（回归示例）。

用法：python scripts/make_doubao_demo.py
"""

import json
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wmkit import imageio, pipeline, template  # noqa: E402

SAMPLE = "tests/data/doubao_sample.png"
TARGET = "tests/data/doubao_target.png"
OUT = "out/doubao_clean.png"
CMP = "out/doubao_sample_removal_compare.png"

sample = template.load_sample_any(SAMPLE)
report = pipeline.process_sample(TARGET, OUT, sample, mask_path="out/doubao_mask.png",
                                 debug_path="out/doubao_mask_overlay.png")
with open("out/doubao_sample_removal_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("样本:", sample.summary())
print("匹配:", report["matches"])

before = Image.open(TARGET).convert("RGB")
after = Image.open(OUT).convert("RGB")
d = ImageDraw.Draw(before)
for m in report["matches"]:
    x0, y0, x1, y1 = m["bbox"]
    d.rectangle([x0 - 5, y0 - 5, x1 + 5, y1 + 5], outline=(230, 40, 40), width=4)

box = (820, 1320, 1080, 1440)
zb = before.crop(box).resize(((box[2] - box[0]) * 2, (box[3] - box[1]) * 2), Image.LANCZOS)
za = after.crop(box).resize(((box[2] - box[0]) * 2, (box[3] - box[1]) * 2), Image.LANCZOS)

w, h = before.size
scale = 0.42
tw, th = int(w * scale), int(h * scale)
top = Image.new("RGB", (tw * 2 + 14, th), "white")
top.paste(before.resize((tw, th), Image.LANCZOS), (0, 0))
top.paste(after.resize((tw, th), Image.LANCZOS), (tw + 14, 0))

bottom = Image.new("RGB", (zb.width + za.width + 14, max(zb.height, za.height)), "white")
bottom.paste(zb, (0, 0))
bottom.paste(za, (zb.width + 14, 0))

cmp_img = Image.new("RGB", (max(top.width, bottom.width), top.height + bottom.height + 30), "white")
cmp_img.paste(top, (0, 26))
cmp_img.paste(bottom, (0, top.height + 30))
dr = ImageDraw.Draw(cmp_img)
dr.text((8, 6), "BEFORE  (red = located by sample)", fill=(200, 30, 30))
dr.text((tw + 22, 6), "AFTER", fill=(20, 140, 60))
cmp_img.save(CMP)
print("done ->", CMP)
