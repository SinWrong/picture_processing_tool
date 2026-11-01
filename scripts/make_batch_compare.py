"""Compare "old batch output" vs "new batch output" at the watermark corner.

Usage: python scripts/make_batch_compare.py <src dir> <old out dir> <new out dir> [out png]
"""

import glob
import os
import sys

from PIL import Image, ImageDraw

src_dir = sys.argv[1]
old_dir = sys.argv[2]
new_dir = sys.argv[3]
out_path = sys.argv[4] if len(sys.argv) > 4 else "out/batch_compare.png"

files = sorted(glob.glob(os.path.join(src_dir, "*.jpeg"))) + \
    sorted(glob.glob(os.path.join(src_dir, "*.png"))) + \
    sorted(glob.glob(os.path.join(src_dir, "*.webp")))

PW, PH = 236, 132
cells = []
for f in files:
    base = os.path.splitext(os.path.basename(f))[0]
    imgs = [Image.open(f).convert("RGB")]
    for d in (old_dir, new_dir):
        for ext in (".png", ".webp"):
            p = os.path.join(d, base + "_clean" + ext)
            if os.path.exists(p):
                imgs.append(Image.open(p).convert("RGB"))
                break
        else:
            imgs.append(None)
    a = imgs[0]
    w, h = a.size
    box = (max(0, w - 320), max(0, h - 180), w, h)
    crops = [im.crop(box).resize((PW, PH), Image.LANCZOS) if im else None for im in imgs]
    cells.append((base[:8], crops))

cols = 5
cell_w, cell_h = PW * 3 + 24, PH + 30
rows = (len(cells) + cols - 1) // cols
sheet = Image.new("RGB", (cell_w * cols, cell_h * rows), "white")
d = ImageDraw.Draw(sheet)
for i, (name, crops) in enumerate(cells):
    cx, cy = (i % cols) * cell_w, (i // cols) * cell_h
    d.text((cx + 2, cy + 2), name, fill=(0, 0, 0))
    for j, c in enumerate(crops):
        if c:
            sheet.paste(c, (cx + j * (PW + 4), cy + 16))
    d.rectangle([cx + PW + 2, cy + 14, cx + PW * 2 + 2, cy + 16 + PH], outline=(190, 40, 40))
    d.rectangle([cx + PW * 2 + 6, cy + 14, cx + PW * 3 + 6, cy + 16 + PH], outline=(20, 140, 60))
sheet.save(out_path)
print("saved", out_path, sheet.size, "n", len(cells))
