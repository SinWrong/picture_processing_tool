#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""wm —— 图片水印分析 / 去除工具（命令行入口）。

示例
----
# 1) 只分析：找出水印候选区域，打印叠加色、透明度、覆盖率，并可导出 JSON
python wm.py analyze images/photo.jpg --json report.json --debug debug.png

# 2) 指定区域去水印（推荐，最快最准）
python wm.py remove images/photo.jpg -o out/photo.png --region 470,375,170,105

# 3) 自动检测 + 去除
python wm.py remove images/photo.jpg -o out/photo.png --auto

# 4) 批量处理
python wm.py batch images -o out --auto

# 5) 水印样本：截一次，之后靠样本自动定位
python wm.py sample images/poster.png -o samples/wm.npz --region 860,1340,220,100
python wm.py match  other.jpg --sample samples/wm.npz --debug match.png
python wm.py remove other.jpg -o out/other.png --sample samples/wm.npz
python wm.py batch  photos -o out --sample samples/wm.npz

# 6) 图形界面：可以打开单个文件，也可以打开一个文件夹做批量
python wm.py gui
python wm.py gui D:\照片\待处理
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

from wmkit import analysis, imageio, pipeline, template  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _fmt_color(rgb) -> str:
    return "#%02X%02X%02X" % tuple(int(round(max(0, min(255, c)))) for c in rgb)


def _print_matches(report: dict) -> None:
    s = report.get("sample") or {}
    if s:
        print(f"水印样本  : {s.get('size', ['?', '?'])[0]}×{s.get('size', ['?', '?'])[1]}  "
              f"叠加色 {s.get('color', '?')}  笔画像素 {s.get('alpha_pixels', '?')}")
    matches = report.get("matches") or []
    print(f"匹配结果  : 找到 {len(matches)} 处")
    for i, m in enumerate(matches, 1):
        x0, y0, x1, y1 = m["bbox"]
        print(f"  #{i} 置信度 {m['score']:.2f}  位置 ({x0},{y0})-({x1},{y1})  缩放 {m['scale']:.2f}")


def _print_analysis(report: dict, verbose: bool = True) -> None:
    img = report["image"]
    print(f"图像      : {img['path']}  {img['width']}×{img['height']}  "
          f"{'含透明通道' if img['has_alpha'] else 'RGB'}")
    if report.get("region"):
        x0, y0, x1, y1 = report["region"]
        print(f"分析区域  : ({x0},{y0}) - ({x1},{y1})   尺寸 {x1 - x0}×{y1 - y0}")

    cands = report.get("candidates") or []
    if cands:
        print(f"自动检测  : 共 {len(cands)} 个候选")
        for i, c in enumerate(cands, 1):
            x0, y0, x1, y1 = c["bbox"]
            colors = ", ".join(f"{o['color']}({o['name']})" for o in c["overlays"][:3])
            print(f"  #{i} 置信度 {c['score']:.2f}  区域 ({x0},{y0})-({x1},{y1})  "
                  f"掩膜 {c['mask_pixels']} px  叠加色 {colors}")

    m = report.get("mask")
    if m and verbose:
        print(f"像素掩膜  : {m['mask_pixels']} px，占区域 {m['coverage_in_region'] * 100:.1f}%")
        for o in m["overlays"]:
            alpha = o["mean_alpha"]
            kind = "不透明" if alpha > 0.85 else ("半透明" if alpha > 0.3 else "很淡")
            print(f"  - 叠加色 {o['color']}  来源 {o['name']:<6}  覆盖率 {o['coverage'] * 100:5.1f}%  "
                  f"平均 alpha {alpha:.2f}（{kind}）")


def cmd_analyze(args) -> int:
    rgb, alpha, _ = imageio.load_image(args.image)
    h, w = rgb.shape[:2]
    region = pipeline.parse_region(args.region, w, h) if args.region else None

    a = pipeline.analyse(rgb, alpha, region=region, sensitivity=args.sensitivity,
                         max_candidates=args.candidates, auto=not args.no_auto)
    report = pipeline.analysis_report(a, image_path=os.path.abspath(args.image))
    _print_analysis(report)

    if args.mask:
        imageio.save_mask_png(args.mask, a.mask_result.mask)
        print(f"掩膜已保存: {args.mask}")
    if args.debug:
        imageio.save_image(args.debug, imageio.overlay_visual(rgb, a.mask_result.mask))
        print(f"调试图已保存: {args.debug}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"报告已保存: {args.json}")
    return 0


def cmd_remove(args) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    if args.sample:
        sample = template.load_sample_any(args.sample)
        try:
            report = pipeline.process_sample(
                args.image, args.output, sample,
                min_score=args.min_score,
                method=args.method,
                texture_refine=args.texture,
                sensitivity=args.sensitivity,
                mode=args.sample_mode,
                mask_path=args.mask,
                debug_path=args.debug,
                compare_path=getattr(args, "compare", None),
            )
        except LookupError as exc:
            print(f"未处理: {exc}", file=sys.stderr)
            return 1
        _print_matches(report)
    else:
        report = pipeline.process(
            args.image, args.output,
            region=args.region,
            sensitivity=args.sensitivity,
            method=args.method,
            texture_refine=args.texture,
            mask_path=args.mask,
            debug_path=args.debug,
            compare_path=getattr(args, "compare", None),
            auto=not args.no_auto,
        )
        _print_analysis(report)
    rem = report["removal"]
    print(f"去除结果  : {report['output']['path']}  "
          f"（掩膜 {rem['mask_pixels']} px，占整图 {rem['mask_ratio'] * 100:.2f}%）")
    if args.mask:
        print(f"掩膜已保存: {args.mask}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"报告已保存: {args.json}")
    return 0


def cmd_sample(args) -> int:
    rgb, alpha, _ = imageio.load_image(args.image)
    h, w = rgb.shape[:2]
    if not args.region:
        print("请用 --region 指定水印所在的区域（x,y,w,h）", file=sys.stderr)
        return 2
    region = pipeline.parse_region(args.region, w, h)
    try:
        s = template.build_sample(rgb, region, sensitivity=args.sensitivity,
                                  name=os.path.basename(args.image))
    except ValueError as exc:
        print(f"截取失败: {exc}", file=sys.stderr)
        return 1
    template.save_sample(args.output, s)
    print(f"样本已保存: {args.output}")
    print(f"  尺寸 {s.size[0]}×{s.size[1]}  叠加色 {s.summary()['color']}  "
          f"笔画像素 {s.summary()['alpha_pixels']}")
    if args.debug:
        mask = np.zeros(rgb.shape[:2], bool)
        x0, y0, x1, y1 = s.bbox
        mask[y0:y1, x0:x1] = s.alpha > 0.1
        imageio.save_image(args.debug, imageio.overlay_visual(rgb, mask))
        print(f"样本调试图: {args.debug}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(s.summary(), f, ensure_ascii=False, indent=2)
        print(f"样本信息: {args.json}")
    return 0


def cmd_match(args) -> int:
    rgb, alpha, _ = imageio.load_image(args.image)
    sample = template.load_sample_any(args.sample)
    bg = analysis.median_background(rgb, 31)
    matches = template.locate(rgb, sample, bg=bg, min_score=args.min_score,
                              max_matches=args.max_matches)
    report = {
        "image": {"path": os.path.abspath(args.image), "width": int(rgb.shape[1]),
                  "height": int(rgb.shape[0])},
        "sample": sample.summary(),
        "matches": template.match_report(matches),
    }
    _print_matches(report)
    if args.debug:
        vis = rgb.copy()
        for m in matches:
            x0, y0, x1, y1 = m["bbox"]
            vis[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x0] = [255, 0, 0]
            vis[max(0, y0 - 2):y1 + 2, x1:x1 + 2] = [255, 0, 0]
            vis[max(0, y0 - 2):y0, max(0, x0 - 2):x1 + 2] = [255, 0, 0]
            vis[y1:y1 + 2, max(0, x0 - 2):x1 + 2] = [255, 0, 0]
        imageio.save_image(args.debug, vis)
        print(f"调试图已保存: {args.debug}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"报告已保存: {args.json}")
    return 0 if matches else 1


def cmd_batch(args) -> int:
    if not os.path.isdir(args.input):
        print(f"输入目录不存在: {args.input}", file=sys.stderr)
        return 2
    os.makedirs(args.output, exist_ok=True)
    files = sorted(
        f for f in os.listdir(args.input)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )
    if not files:
        print("目录里没有图片", file=sys.stderr)
        return 2
    sample = template.load_sample_any(args.sample) if args.sample else None
    paths = [os.path.join(args.input, name) for name in files]
    ok = skipped = 0

    def _report(i, path, dst, report, error):
        nonlocal ok, skipped
        name = os.path.basename(path)
        if error:
            kind, msg = error
            if kind == "skip":
                skipped += 1
                print(f"[SKIP] {name}: {msg}", file=sys.stderr)
            else:
                print(f"[FAIL] {name}: {msg}", file=sys.stderr)
            return
        rem = report["removal"]
        extra = ""
        if "matches" in report:
            extra = f" 匹配 {len(report['matches'])} 处"
        elif report.get("consensus"):
            extra = " 按同批多数区域"
        print(f"[OK] {name} -> {os.path.basename(dst)}  "
              f"掩膜 {rem['mask_pixels']} px ({rem['mask_ratio'] * 100:.2f}%){extra}")
        ok += 1

    pipeline.process_many(
        paths, args.output, region=args.region, sensitivity=args.sensitivity,
        method=args.method, texture_refine=args.texture, auto=not args.no_auto,
        sample=sample, on_item=_report,
    )
    print(f"完成 {ok}/{len(files)}" + (f"，跳过 {skipped}" if skipped else ""))
    return 0 if ok else 1


def cmd_gui(args) -> int:
    from wmkit.gui import main as gui_main
    gui_main(initial=args.image)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wm",
        description="图片水印分析 / 去除工具（纯 Python + numpy + Pillow，无需 OpenCV）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp, with_output=False):
        sp.add_argument("--region", default=None,
                        help="水印区域 x,y,w,h（像素）或 x%%,y%%,w%%,h%%（百分比）")
        sp.add_argument("--sensitivity", type=float, default=0.5,
                        help="分析灵敏度 0~1，越大越容易把淡色/细笔画算进来（默认 0.5）")
        sp.add_argument("--no-auto", action="store_true",
                        help="不做自动检测，必须用 --region 指定区域")
        sp.add_argument("--auto", action="store_true", default=True,
                        help="自动检测水印区域（默认就是开的，写上只是为了明确）")
        sp.add_argument("--method", default="harmonic",
                        choices=["harmonic", "telea", "fast"],
                        help="修复算法：harmonic=多尺度 Laplace（默认，无接缝）；"
                             "telea=快速行进；fast=小窗口 Telea")
        sp.add_argument("--texture", action="store_true",
                        help="叠加纹理细节（草地、树叶等高频背景更自然）")

    def add_sample(sp):
        sp.add_argument("--sample", default=None,
                        help="水印样本（.npz，或一张水印小图），按样本定位后去除")
        sp.add_argument("--min-score", type=float, default=0.5,
                        help="样本匹配的最低置信度 0~1（默认 0.5，找不到时可调低）")
        sp.add_argument("--max-matches", type=int, default=4,
                        help="最多处理几处水印（默认 4）")
        sp.add_argument("--sample-mode", default="stamp", choices=["stamp", "region"],
                        help="stamp=样本形状即水印，按它直接擦（默认）；"
                             "region=在匹配位置重新分析叠加层")

    a = sub.add_parser("analyze", help="只分析，不修改图片")
    a.add_argument("image")
    a.add_argument("--json", default=None, help="把分析结果写成 JSON")
    a.add_argument("--mask", default=None, help="导出掩膜 PNG")
    a.add_argument("--debug", default=None, help="导出掩膜叠加调试图")
    a.add_argument("--candidates", type=int, default=3, help="最多输出几个候选区域")
    add_common(a)
    a.set_defaults(func=cmd_analyze)

    r = sub.add_parser("remove", help="去除水印并保存")
    r.add_argument("image")
    r.add_argument("-o", "--output", required=True)
    r.add_argument("--json", default=None, help="把处理报告写成 JSON")
    r.add_argument("--mask", default=None, help="同时导出掩膜 PNG")
    r.add_argument("--debug", default=None, help="同时导出掩膜叠加调试图")
    r.add_argument("--compare", default=None, metavar="PNG",
                   help="同时导出“修改前 / 修改后”左右对比图")
    add_common(r)
    add_sample(r)
    r.set_defaults(func=cmd_remove)

    b = sub.add_parser("batch", help="批量处理目录")
    b.add_argument("input")
    b.add_argument("-o", "--output", required=True)
    add_common(b)
    add_sample(b)
    b.set_defaults(func=cmd_batch)

    s = sub.add_parser("sample", help="从图片里截取水印样本（.npz）")
    s.add_argument("image")
    s.add_argument("-o", "--output", required=True, help="样本保存路径，建议 .npz")
    s.add_argument("--region", required=True, help="水印区域 x,y,w,h 或百分比")
    s.add_argument("--sensitivity", type=float, default=0.5)
    s.add_argument("--json", default=None, help="输出样本信息 JSON")
    s.add_argument("--debug", default=None, help="导出样本掩膜调试图")
    s.set_defaults(func=cmd_sample)

    m = sub.add_parser("match", help="用样本在图片里找水印位置（不修改图片）")
    m.add_argument("image")
    m.add_argument("--sample", required=True, help="样本 .npz")
    m.add_argument("--min-score", type=float, default=0.5)
    m.add_argument("--max-matches", type=int, default=4)
    m.add_argument("--json", default=None)
    m.add_argument("--debug", default=None, help="导出框出匹配位置的图")
    m.set_defaults(func=cmd_match)

    g = sub.add_parser("gui", help="打开图形界面（可载入文件夹批量处理）")
    g.add_argument("image", nargs="?", default=None,
                   help="启动时载入的图片或文件夹")
    g.set_defaults(func=cmd_gui)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
