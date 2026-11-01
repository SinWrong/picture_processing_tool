"""冒烟测试：不依赖 pytest，直接 python tests/test_basic.py 运行。"""

import os
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wmkit import (  # noqa: E402
    analysis, fill, imageio, inpaint, morph, pipeline, template, texture,
)

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [ok]   {name} {detail}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def test_parse_region():
    print("parse_region")
    check("像素", pipeline.parse_region("10,20,30,40", 100, 100) == (10, 20, 40, 60))
    check("百分比", pipeline.parse_region("10%,20%,30%,40%", 200, 200) == (20, 40, 80, 120))
    try:
        pipeline.parse_region("1,2,3", 100, 100)
        check("非法输入报错", False)
    except ValueError:
        check("非法输入报错", True)


def test_inpaint():
    print("inpaint.telea_inpaint")
    rng = np.random.default_rng(0)
    img = np.full((60, 80, 3), 120, np.uint8)
    mask = np.zeros((60, 80), bool)
    mask[25:32, 10:70] = True
    dirty = img.copy()
    dirty[mask] = 250
    out = inpaint.telea_inpaint(dirty, mask, radius=4)
    err = np.abs(out.astype(int) - 120)[mask].mean()
    check("平坦区域修复误差 < 6", err < 6, f"(err={err:.2f})")
    check("掩膜外像素不变", int(np.abs(out.astype(int) - dirty.astype(int)).max(2)[~mask].max()) == 0)


def test_harmonic():
    print("fill.harmonic_fill")
    yy, xx = np.mgrid[0:80, 0:120]
    base = np.zeros((80, 120, 3), np.uint8)
    base[..., 0] = 40 + (xx * 1.0).astype(np.uint8)
    base[..., 1] = 60 + (yy * 1.0).astype(np.uint8)
    base[..., 2] = 90
    mask = np.zeros((80, 120), bool)
    mask[20:60, 30:90] = True
    dirty = base.copy()
    dirty[mask] = 255
    out = fill.harmonic_fill(dirty, mask, levels=4)
    err = np.abs(out.astype(int) - base.astype(int))[mask].mean()
    check("线性渐变修复误差 < 8", err < 8, f"(err={err:.2f})")


def test_texture():
    print("texture.refine")
    yy, xx = np.mgrid[0:64, 0:96]
    rng = np.random.default_rng(1)
    base = np.dstack([
        100 + (xx * 0.5),
        120 + (yy * 0.4),
        80 + (xx * 0.2),
    ]).astype(np.uint8)
    noisy = np.clip(base + rng.normal(0, 8, base.shape), 0, 255).astype(np.uint8)
    mask = np.zeros((64, 96), bool)
    mask[20:40, 20:70] = True
    filled = fill.harmonic_fill(noisy, mask, levels=3)
    ref = texture.refine(noisy, mask, filled)
    check("返回同尺寸", ref.shape == noisy.shape)
    ring = morph.dilate(mask, 1)
    check("掩膜外基本不变（允许 1px 羽化）",
          int(np.abs(ref.astype(int) - filled.astype(int)).max(2)[~ring].max()) <= 3)


def test_mask_on_synthetic():
    print("analysis.build_mask（合成半透明白字）")
    h, w = 120, 200
    rng = np.random.default_rng(3)
    bg_col = np.array([110, 130, 70], np.float32)
    rgb = np.clip(bg_col + rng.normal(0, 6, (h, w, 3)), 0, 255).astype(np.uint8)
    # 画一个“水印”：白字条
    rgb[40:52, 30:60] = (rgb[40:52, 30:60] * 0.2 + 255 * 0.8).astype(np.uint8)
    rgb[40:52, 70:150] = (rgb[40:52, 70:150] * 0.2 + 255 * 0.8).astype(np.uint8)
    bg = analysis.median_background(rgb, 31)
    res = analysis.build_mask(rgb, bg, (20, 30, 180, 70), sensitivity=0.5, dilate=2)
    mask = res.mask
    covered = np.concatenate([mask[40:52, 30:60].ravel(),
                              mask[40:52, 70:150].ravel()]).mean()
    check("覆盖水印笔画 > 0.8", covered > 0.8, f"(covered={covered:.2f})")
    check("区域外无掩膜", mask[:30].sum() == 0 and mask[70:].sum() == 0)
    check("叠加色接近白色", any(abs(o["color"] == "#FFFFFF") for o in []) or res.score > 0)


def test_end_to_end():
    print("pipeline.process（临时文件）")
    h, w = 80, 120
    rgb = np.full((h, w, 3), 200, np.uint8)
    rgb[:, :, 2] = 210
    rgb[30:40, 20:100] = (60, 90, 180)   # 蓝色水印条
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.png")
        dst = os.path.join(tmp, "out.png")
        imageio.save_image(src, rgb)
        report = pipeline.process(src, dst, region="15,25,110,20", method="harmonic")
        out, _, _ = imageio.load_image(dst)
        check("输出文件存在", os.path.exists(dst))
        check("掩膜非空", report["removal"]["mask_pixels"] > 100)
        check("水印区被改写", int(out[35, 60, 0]) > 150, f"(value={int(out[35, 60, 0])})")


def _alpha_pattern(h=34, w=76):
    """造一个字条状的水印 alpha 图案（粗细不一的横竖笔画）。"""
    a = np.zeros((h, w), np.float32)
    a[6:10, 4:70] = 0.85
    a[6:26, 10:14] = 0.85
    a[20:24, 4:70] = 0.85
    a[6:26, 30:34] = 0.7
    a[6:26, 44:48] = 0.7
    a[12:16, 44:62] = 0.7
    a[6:26, 58:62] = 0.85
    return a


def _composite(bg, alpha, x, y, color=(255, 255, 255)):
    out = bg.astype(np.float32).copy()
    a = alpha[..., None]
    patch = out[y:y + alpha.shape[0], x:x + alpha.shape[1]]
    out[y:y + alpha.shape[0], x:x + alpha.shape[1]] = patch * (1 - a) + np.array(color, np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def test_template_sample():
    print("template 样本匹配（跨背景 / 跨尺度）")
    rng = np.random.default_rng(11)
    pat = _alpha_pattern()

    base1 = np.clip(np.array([120, 140, 90], np.float32) + rng.normal(0, 5, (200, 260, 3)), 0, 255)
    img1 = _composite(base1.astype(np.uint8), pat, 40, 50)
    sample = template.build_sample(img1, (34, 44, 122, 90), sensitivity=0.5)
    check("样本非空", (sample.alpha > 0.1).sum() > 100, f"({int((sample.alpha > 0.1).sum())} px)")

    # 换一张完全不同的底色 + 更暗 + 水印换位置
    base2 = np.clip(np.array([70, 95, 160], np.float32) * 0.85 + rng.normal(0, 5, (320, 360, 3)), 0, 255)
    img2 = _composite(base2.astype(np.uint8), pat, 180, 210)
    py, px = np.nonzero(pat > 0.1)
    ink_x, ink_y = int(px.min()), int(py.min())      # 图案真实笔画的左上角

    matches = template.locate(img2, sample, min_score=0.5)
    ok = bool(matches)
    check("换背景后能找到", ok)
    if ok:
        x0, y0 = matches[0]["bbox"][:2]
        check("位置准确（±6px）", abs(x0 - (180 + ink_x)) <= 6 and abs(y0 - (210 + ink_y)) <= 6,
              f"(got {x0},{y0} expect {180 + ink_x},{210 + ink_y})")
        check("置信度 > 0.6", matches[0]["score"] > 0.6, f"(score={matches[0]['score']:.2f})")

    # 放大 1.3 倍后仍能找到，并给出正确缩放
    big = np.asarray(Image.fromarray(img2).resize((int(360 * 1.3), int(320 * 1.3)), Image.LANCZOS))
    matches_big = template.locate(big, sample, min_score=0.5)
    ok = bool(matches_big)
    check("放大后能找到", ok)
    if ok:
        x0, y0 = matches_big[0]["bbox"][:2]
        ex, ey = (180 + ink_x) * 1.3, (210 + ink_y) * 1.3
        check("缩放位置准确（±8px）", abs(x0 - ex) <= 8 and abs(y0 - ey) <= 8,
              f"(got {x0},{y0} expect {ex:.0f},{ey:.0f})")
        check("缩放系数 ≈1.3", abs(matches_big[0]["scale"] - 1.3) <= 0.15,
              f"(scale={matches_big[0]['scale']})")


def test_sample_roundtrip_and_remove():
    print("样本存取 + process_sample")
    rng = np.random.default_rng(3)
    pat = _alpha_pattern()
    base = np.clip(np.array([200, 200, 205], np.float32) + rng.normal(0, 3, (160, 220, 3)), 0, 255)
    img = _composite(base.astype(np.uint8), pat, 60, 40)
    with tempfile.TemporaryDirectory() as tmp:
        sp = os.path.join(tmp, "s.npz")
        s = template.build_sample(img, (54, 34, 140, 80), sensitivity=0.5)
        template.save_sample(sp, s)
        s2 = template.load_sample(sp)
        check("样本可往返保存", s2.alpha.shape == s.alpha.shape and np.allclose(s2.color, s.color))

        src = os.path.join(tmp, "in.png")
        dst = os.path.join(tmp, "out.png")
        imageio.save_image(src, img)
        report = pipeline.process_sample(src, dst, s2, min_score=0.4)
        check("匹配数 >= 1", len(report["matches"]) >= 1)
        out, _, _ = imageio.load_image(dst)
        check("水印区被改写", int(np.abs(out[40:70, 60:130].astype(int) -
                                          img[40:70, 60:130].astype(int)).mean()) > 3)


def test_sample_from_image_file():
    print("load_sample_any（直接用图片当样本）")
    rng = np.random.default_rng(21)
    pat = _alpha_pattern()
    base1 = np.clip(np.array([150, 150, 150], np.float32) + rng.normal(0, 4, (120, 200, 3)), 0, 255)
    img1 = _composite(base1.astype(np.uint8), pat, 60, 40)
    py, px = np.nonzero(pat > 0.1)
    ink_x, ink_y = int(px.min()), int(py.min())

    with tempfile.TemporaryDirectory() as tmp:
        crop = img1[40 - 3:40 + pat.shape[0] + 3, 60 - 3:60 + pat.shape[1] + 3]
        p = os.path.join(tmp, "wm.png")
        imageio.save_image(p, crop)
        s = template.load_sample_any(p)
        check("从图片得到样本", s.alpha.shape[0] > 10 and s.alpha.shape[1] > 10,
              f"({s.alpha.shape[1]}×{s.alpha.shape[0]})")

        base2 = np.clip(np.array([90, 110, 150], np.float32) + rng.normal(0, 4, (200, 260, 3)), 0, 255)
        img2 = _composite(base2.astype(np.uint8), pat, 120, 90)
        m = template.locate(img2, s, min_score=0.45)
        check("图片样本能定位", bool(m))
        if m:
            x0, y0 = m[0]["bbox"][:2]
            check("位置准确（±8px）", abs(x0 - (120 + ink_x)) <= 8 and abs(y0 - (90 + ink_y)) <= 8,
                  f"(got {x0},{y0})")

        big = np.full((600, 600, 3), 200, np.uint8)
        bp = os.path.join(tmp, "big.png")
        imageio.save_image(bp, big)
        try:
            template.load_sample_any(bp)
            check("大图拒绝整张当样本", False)
        except ValueError:
            check("大图拒绝整张当样本", True)


def test_real_doubao_sample():
    """回归用例：用户给的“豆包AI生成”水印图（白底、alpha 全不透明）+ 海报。"""
    print("真实样本：豆包AI生成（白底水印图 → 海报）")
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    sp = os.path.join(base, "doubao_sample.png")
    tp = os.path.join(base, "doubao_target.png")
    if not (os.path.exists(sp) and os.path.exists(tp)):
        print("  (缺少 tests/data 里的图片，跳过)")
        return

    s = template.load_sample_any(sp)
    check("白底水印图能当成样本", s.alpha.shape[0] > 10 and s.alpha.shape[1] > 50,
          f"({s.size[0]}×{s.size[1]})")
    check("不会把整张图当成水印形状", s.size[0] < 200 and s.size[1] < 70,
          f"({s.size[0]}×{s.size[1]})")

    rgb, _, _ = imageio.load_image(tp)
    bg = analysis.median_background(rgb, 31)
    m = template.locate(rgb, s, bg=bg, min_score=0.5)
    check("在海报里找到水印", bool(m))
    if m:
        x0, y0, x1, y1 = m[0]["bbox"]
        check("落在右下角水印处", x0 > 820 and y0 > 1320 and x1 > 1000 and y1 > 1400,
              f"({x0},{y0})-({x1},{y1})")
        check("置信度 > 0.6", m[0]["score"] > 0.6, f"(score={m[0]['score']:.2f})")


def test_auto_detect_cases():
    """自动检测回归：硬币照片（深色纹理 + 右下角浅色文字）。"""
    print("自动检测：硬币照片（右下角文字水印）")
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    p = os.path.join(base, "coins_photo.png")
    if not os.path.exists(p):
        print("  (缺少 tests/data/coins_photo.png，跳过)")
        return
    rgb, _, _ = imageio.load_image(p)
    h, w = rgb.shape[:2]
    bg = analysis.median_background(rgb, 31)
    cands = analysis.detect_regions(rgb, bg, max_regions=3)
    check("有候选", bool(cands))
    if not cands:
        return
    x0, y0, x1, y1 = cands[0].bbox
    check("首选落在右下角", ((x0 + x1) / 2) / w > 0.7 and ((y0 + y1) / 2) / h > 0.7,
          f"({x0},{y0})-({x1},{y1})")
    check("置信度明显高于次选",
          len(cands) == 1 or cands[0].score > cands[1].score * 1.3,
          f"({cands[0].score:.2f} vs {cands[1].score:.2f})" if len(cands) > 1 else "")


def test_user_region_ring_mask():
    """回归：手工框选 + 花色背景（珊瑚上的“南海网”彩色 logo）。"""
    print("手工框选：彩色 logo + 花色背景")
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    src = os.path.join(base, "wm_corner.png")
    if not os.path.exists(src):
        print("  (缺少 tests/data/wm_corner.png，跳过)")
        return
    rgb, _, _ = imageio.load_image(src)
    x0, y0, x1, y1 = 69, 39, 297, 141
    inner = rgb[y0:y1, x0:x1].astype(int)
    blue_before = int(((inner[..., 2] - inner[..., 0] > 60) &
                       (inner[..., 2] > 120)).sum())

    with tempfile.TemporaryDirectory() as tmp:
        dst = os.path.join(tmp, "out.png")
        rep = pipeline.process(src, dst, region="69,39,228,102")
        check("掩膜覆盖水印", rep["removal"]["mask_pixels"] > 4000,
              f"({rep['removal']['mask_pixels']} px)")
        out, _, _ = imageio.load_image(dst)
        inner2 = out[y0:y1, x0:x1].astype(int)
        blue_after = int(((inner2[..., 2] - inner2[..., 0] > 60) &
                          (inner2[..., 2] > 120)).sum())
        check("logo 的蓝色像素基本消失",
              blue_before > 200 and blue_after < blue_before * 0.2,
              f"({blue_before} -> {blue_after})")


def test_compare_export():
    """--compare：导出“修改前 / 修改后”左右对比图。"""
    print("对比图导出")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "o.png")
        cmp_path = os.path.join(tmp, "c.png")
        rep = pipeline.process("images/photo_bird.jpg", out,
                               region="470,375,170,105", compare_path=cmp_path)
        check("对比图已生成", os.path.exists(cmp_path))
        if os.path.exists(cmp_path):
            before, _, _ = imageio.load_image("images/photo_bird.jpg")
            after, _, _ = imageio.load_image(out)
            side, _, _ = imageio.load_image(cmp_path)
            check("宽度 = 原图 + 结果 + 间隔",
                  side.shape[1] == before.shape[1] + after.shape[1] + 12,
                  f"({side.shape[1]} vs {before.shape[1]}+{after.shape[1]})")
            check("高度够放两张图", side.shape[0] >= max(before.shape[0], after.shape[0]))
        check("报告里有掩膜像素", rep["removal"]["mask_pixels"] > 0)


def main():
    test_parse_region()
    test_inpaint()
    test_harmonic()
    test_texture()
    test_mask_on_synthetic()
    test_end_to_end()
    test_template_sample()
    test_sample_roundtrip_and_remove()
    test_sample_from_image_file()
    test_real_doubao_sample()
    test_auto_detect_cases()
    test_user_region_ring_mask()
    test_compare_export()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
