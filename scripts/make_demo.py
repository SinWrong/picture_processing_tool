"""生成示例输出：单图结果、掩膜调试图、JSON 报告、前后对比图。

用法（项目根目录）：
    D:\\anaconda3\\envs\\new_env\\python.exe scripts\\make_demo.py
"""

import json
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wmkit import imageio, pipeline  # noqa: E402

os.makedirs("out", exist_ok=True)

CASES = [
    ("images/photo_bird.jpg", "out/photo_clean.png", "470,375,170,105", (0.70, 0.72)),
    ("images/poster_icu.png", "out/poster_clean.png", "860,1340,220,100", (0.72, 0.73)),
]


def zoom(img: Image.Image, box, factor=3) -> Image.Image:
    w, h = img.size
    crop = img.crop((int(w * box[0]), int(h * box[1]), w, h))
    return crop.resize((crop.width * factor, crop.height * factor), Image.LANCZOS)


for src, dst, region, corner in CASES:
    report = pipeline.process(src, dst, region=region, method="harmonic")
    base = os.path.splitext(os.path.basename(dst))[0]
    with open(f"out/{base}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    orig = Image.open(src).convert("RGB")
    clean = Image.open(dst).convert("RGB")

    rgb, alpha, _ = imageio.load_image(src)
    an = pipeline.analyse(rgb, alpha,
                          region=pipeline.parse_region(region, rgb.shape[1], rgb.shape[0]),
                          auto=False)
    imageio.save_mask_png(f"out/{base}_mask.png", an.mask_result.mask)
    imageio.save_image(f"out/{base}_mask_overlay.png",
                       imageio.overlay_visual(rgb, an.mask_result.mask))

    top = Image.new("RGB", (orig.width * 2 + 12, orig.height), "white")
    top.paste(orig, (0, 0))
    top.paste(clean, (orig.width + 12, 0))
    z1, z2 = zoom(orig, corner), zoom(clean, corner)
    bottom = Image.new("RGB", (z1.width + z2.width + 12, max(z1.height, z2.height)), "white")
    bottom.paste(z1, (0, 0))
    bottom.paste(z2, (z1.width + 12, 0))
    cmp_img = Image.new("RGB", (max(top.width, bottom.width), top.height + bottom.height + 12), "white")
    cmp_img.paste(top, (0, 0))
    cmp_img.paste(bottom, (0, top.height + 12))
    draw = ImageDraw.Draw(cmp_img)
    draw.text((8, 6), "BEFORE", fill=(255, 60, 60))
    draw.text((orig.width + 20, 6), "AFTER", fill=(0, 160, 60))
    cmp_img.save(f"out/{base}_compare.png")
    print("done", dst)
