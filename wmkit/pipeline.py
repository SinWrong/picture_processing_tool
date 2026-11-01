"""端到端流程：分析 → 生成掩膜 → 修复 → 报告。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from . import analysis, fill, imageio, inpaint, morph, template, texture


# --------------------------------------------------------------------------- #
# 区域参数解析
# --------------------------------------------------------------------------- #
def parse_region(text: str, width: int, height: int) -> tuple[int, int, int, int]:
    """解析区域参数，返回 (x0, y0, x1, y1)。

    支持两种写法：
    - 像素：``x,y,w,h``，例如 ``470,375,170,105``
    - 百分比：``x%,y%,w%,h%``，例如 ``73%,78%,27%,22%``
    """
    parts = [p.strip() for p in str(text).replace("，", ",").split(",")]
    if len(parts) != 4:
        raise ValueError(f"区域参数需要 4 个数字（x,y,w,h），收到: {text!r}")
    pct = all(p.endswith("%") for p in parts)
    if pct:
        vals = [float(p.rstrip("%")) for p in parts]
        x, y = vals[0] / 100.0 * width, vals[1] / 100.0 * height
        w, h = vals[2] / 100.0 * width, vals[3] / 100.0 * height
    else:
        if any(p.endswith("%") for p in parts):
            raise ValueError("区域参数要么全用百分比，要么全用像素")
        x, y, w, h = (float(p) for p in parts)
    x0, y0 = int(round(x)), int(round(y))
    x1, y1 = int(round(x + w)), int(round(y + h))
    x0, y0 = max(0, min(x0, width - 1)), max(0, min(y0, height - 1))
    x1, y1 = max(x0 + 1, min(x1, width)), max(y0 + 1, min(y1, height))
    return x0, y0, x1, y1


def _expand(bbox, width: int, height: int, ratio: float = 0.12) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    pad_x = max(10, int((x1 - x0) * ratio))
    pad_y = max(10, int((y1 - y0) * ratio))
    return (max(0, x0 - pad_x), max(0, y0 - pad_y),
            min(width, x1 + pad_x), min(height, y1 + pad_y))


# --------------------------------------------------------------------------- #
# 分析
# --------------------------------------------------------------------------- #
@dataclass
class Analysis:
    rgb: np.ndarray
    alpha: np.ndarray | None
    bg: np.ndarray
    region: tuple[int, int, int, int] | None
    mask_result: analysis.MaskResult | None
    candidates: list[analysis.MaskResult] = field(default_factory=list)
    report: dict = field(default_factory=dict)


def _evidence_map(rgb, bg, region):
    """区域内“明显不属于背景”的像素（水印笔画、logo），用来给两种掩膜打分。"""
    x0, y0, x1, y1 = region
    sub = rgb[y0:y1, x0:x1].astype(np.float32)
    bgs = bg[y0:y1, x0:x1].astype(np.float32)
    dev = np.abs(sub - bgs).max(axis=2)
    tex = analysis.local_texture(rgb, bg)[y0:y1, x0:x1]
    return (dev > np.maximum(28.0, 2.5 * tex)) & (dev > 22.0)


def _mask_quality(mask, ev):
    """覆盖了多少“证据像素”，再按大小略作惩罚（宁小勿大）。mask/ev 尺寸需一致。"""
    if not mask.any():
        return -1.0
    tot = float(ev.sum())
    if tot <= 0:
        return 0.0
    hit = float((ev & mask).sum()) / tot
    size = float(mask.sum()) / max(1.0, mask.size)
    return hit - 0.25 * size


def _ring_region(rgb, region, *, sensitivity=0.5, rounds: int = 3):
    """用“框外当背景”取掩膜；框得太大就自动收拢，框切到水印就小步外扩。"""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in region]
    region = (max(0, x0), max(0, y0), min(w, x1), min(h, y1))
    rm = None
    for _ in range(rounds):
        x0, y0, x1, y1 = region
        area = (x1 - x0) * (y1 - y0)
        if area < 64:
            return None, region
        rm = analysis.ring_mask(rgb, region)
        cnt = int(rm.sum())
        if not (0.004 * area <= cnt <= 0.75 * area):
            return None, region
        bx = analysis.morph.bbox_of(rm)
        if bx is None:
            return None, region

        bw, bh = bx[2] - bx[0], bx[3] - bx[1]
        mask_area = bw * bh
        touched = (bx[0] <= x0 + 2 or bx[1] <= y0 + 2
                   or bx[2] >= x1 - 3 or bx[3] >= y1 - 3)
        if not touched and mask_area < 0.5 * area:
            # 框太大：收拢到墨迹附近，背景统计会更准
            pad_x = max(10, int(bw * 0.35))
            pad_y = max(10, int(bh * 0.45))
            tight = (max(0, bx[0] - pad_x), max(0, bx[1] - pad_y),
                     min(w, bx[2] + pad_x), min(h, bx[3] + pad_y))
            if tight != region and (tight[2] - tight[0]) > 10 and (tight[3] - tight[1]) > 10:
                region = tight
                continue
            break
        if not touched:
            break
        pad_x = int(min(40, max(10, (x1 - x0) * 0.12)))
        pad_y = int(min(40, max(10, (y1 - y0) * 0.12)))
        new_region = (max(0, x0 - pad_x), max(0, y0 - pad_y),
                      min(w, x1 + pad_x), min(h, y1 + pad_y))
        if new_region == region:
            break
        region = new_region
    return rm, region


def _best_mask(rgb, bg, region, *, sensitivity=0.5):
    """两种判据各出一张掩膜，按“证据覆盖”择优。

    1) build_mask：猜叠加色（纯色/低对比水印准）；
    2) ring_mask：框外当背景（纹理背景上的彩色 logo、细描边准）。
    """
    mask_bm = analysis.build_mask(rgb, bg, region, sensitivity=sensitivity).mask
    rm, region2 = _ring_region(rgb, region, sensitivity=sensitivity)

    cand = [(mask_bm, region, "mask")]
    if rm is not None:
        cand.append((rm, region2, "ring"))

    best, best_region, best_src, best_q = None, region, "none", -9.0
    for m, reg, src in cand:
        x0, y0, x1, y1 = reg
        sub = m[y0:y1, x0:x1]
        if not sub.any():
            q = -8.0
        else:
            ev = _evidence_map(rgb, bg, reg)
            q = _mask_quality(sub, ev)
        if q > best_q:
            best, best_region, best_src, best_q = m, reg, src, q
    return best, best_region, best_src


def analyse(rgb: np.ndarray, alpha, *, region=None, sensitivity: float = 0.5,
            max_candidates: int = 3, auto: bool = True) -> Analysis:
    h, w = rgb.shape[:2]
    bg = analysis.median_background(rgb, 31)
    user_region = region is not None

    candidates: list[analysis.MaskResult] = []
    if region is None and auto:
        candidates = analysis.detect_regions(rgb, bg, sensitivity=sensitivity,
                                             max_regions=max_candidates)
        if candidates:
            region = candidates[0].bbox
    if region is None:
        raise ValueError("没有指定水印区域，且自动检测没有找到候选区域。"
                         "请用 --region 指定（例如 --region 470,375,170,105）。")

    if user_region:
        # 手工框选：只用“框外当背景”这一个判据（框就是意图，宁可多擦）
        rm, region = _ring_region(rgb, region, sensitivity=sensitivity)
        mask_result = analysis.build_mask(rgb, bg, region, sensitivity=sensitivity)
        if rm is not None:
            mask_result.mask = rm
            mask_result.stats["mode"] = "ring"
        else:
            mask_result.stats["mode"] = "mask"
    else:
        mask, region, src = _best_mask(rgb, bg, region, sensitivity=sensitivity)
        mask_result = analysis.build_mask(rgb, bg, region, sensitivity=sensitivity)
        mask_result.mask = mask
        mask_result.stats["mode"] = src

    # 第一轮掩膜往往只抓到“最明显的那部分水印”，把区域收拢到它附近再分析一次：
    # 颜色统计会更聚焦，淡色/细笔画能补回来。
    bx0 = analysis.morph.bbox_of(mask_result.mask)
    if bx0 is not None:
        area_region = (region[2] - region[0]) * (region[3] - region[1])
        area_mask = (bx0[2] - bx0[0]) * (bx0[3] - bx0[1])
        if 0 < area_mask < area_region * 0.55:
            tight = _expand(bx0, w, h, ratio=0.10)
            if tight != region and (tight[2] - tight[0]) > 8 and (tight[3] - tight[1]) > 8:
                region = tight
                mask_result = analysis.build_mask(
                    rgb, bg, region, sensitivity=sensitivity
                )

    # 掩膜顶到区域边界，说明水印被切掉了一部分：小步外扩后重新分析。
    # 注意只扩“被顶住的那一侧”，且限制外扩幅度——区域放太大反而会让
    # 颜色统计被其他内容带偏。
    base_area = (region[2] - region[0]) * (region[3] - region[1])
    for _ in range(3):
        bx = analysis.morph.bbox_of(mask_result.mask)
        if bx is None:
            break
        x0, y0, x1, y1 = region
        pad_x = int(min(48, max(12, (x1 - x0) * 0.12)))
        pad_y = int(min(48, max(12, (y1 - y0) * 0.12)))
        nx0 = max(0, x0 - pad_x) if bx[0] <= x0 + 2 else x0
        ny0 = max(0, y0 - pad_y) if bx[1] <= y0 + 2 else y0
        nx1 = min(w, x1 + pad_x) if bx[2] >= x1 - 3 else x1
        ny1 = min(h, y1 + pad_y) if bx[3] >= y1 - 3 else y1
        new_region = (nx0, ny0, nx1, ny1)
        if new_region == region:
            break
        if (nx1 - nx0) * (ny1 - ny0) > base_area * 2.5:
            break
        region = new_region
        mask_result = analysis.build_mask(rgb, bg, region, sensitivity=sensitivity)

    return Analysis(rgb=rgb, alpha=alpha, bg=bg, region=region,
                    mask_result=mask_result, candidates=candidates)


def analysis_report(a: Analysis, *, image_path: str = "", extra: dict | None = None) -> dict:
    h, w = a.rgb.shape[:2]
    report = {
        "image": {"path": image_path, "width": int(w), "height": int(h),
                  "channels": int(a.rgb.shape[2]), "has_alpha": a.alpha is not None},
        "region": list(a.region) if a.region else None,
        "candidates": [c.summary() for c in a.candidates],
        "mask": a.mask_result.summary() if a.mask_result else None,
    }
    if extra:
        report.update(extra)
    return report


# --------------------------------------------------------------------------- #
# 去除
# --------------------------------------------------------------------------- #
def fill_holes(mask: np.ndarray, max_hole_ratio: float = 0.03) -> np.ndarray:
    """把掩膜内部的小孔补上（字腔、细缝），避免修复后留下残留小点。"""
    if not mask.any():
        return mask
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    y0, y1 = max(0, int(ys.min()) - 1), min(h, int(ys.max()) + 2)
    x0, x1 = max(0, int(xs.min()) - 1), min(w, int(xs.max()) + 2)
    sub = mask[y0:y1, x0:x1]
    inv = ~sub
    labels, n, areas = morph.label_components(inv)
    if n == 0:
        return mask
    border = set(labels[0, :].tolist()) | set(labels[-1, :].tolist()) \
        | set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
    max_hole = max(4, int(sub.size * max_hole_ratio))
    fill = np.zeros(n + 1, bool)
    for i, a in enumerate(areas, start=1):
        if i not in border and a <= max_hole:
            fill[i] = True
    sub = sub | fill[labels]
    out = mask.copy()
    out[y0:y1, x0:x1] = sub
    return out


def remove(rgb: np.ndarray, mask: np.ndarray, *, method: str = "harmonic",
           radius: int = 4, texture_refine: bool = False,
           texture_strength: float = 1.0) -> np.ndarray:
    """按掩膜去除水印，返回处理后的图像（uint8）。"""
    mask = fill_holes(mask)
    method = (method or "harmonic").lower()
    if method in ("harmonic", "laplace", "diffusion"):
        out = fill.harmonic_fill_region(rgb, mask, pad=max(32, int(min(rgb.shape[:2]) * 0.06)))
    elif method == "telea":
        out = inpaint.inpaint_region(rgb, mask, radius=radius,
                                     pad=max(24, int(min(rgb.shape[:2]) * 0.08)))
    elif method == "fast":
        out = inpaint.inpaint_region(rgb, mask, radius=3,
                                     pad=max(16, int(min(rgb.shape[:2]) * 0.05)))
    else:
        raise ValueError(f"未知的修复方法: {method!r}（可用 harmonic / telea / fast）")

    if texture_refine:
        base = fill.harmonic_fill_region(rgb, mask, pad=max(32, int(min(rgb.shape[:2]) * 0.06)))
        out = texture.refine(rgb, mask, base, sigma=1.2,
                             min_dist=int(min(rgb.shape[:2]) * 0.03),
                             max_dist=int(min(rgb.shape[:2]) * 0.25))
        if texture_strength < 1.0:
            blended = base.astype(np.float32) * (1 - texture_strength) + \
                out.astype(np.float32) * texture_strength
            out = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def process(image_path: str, out_path: str, *, region=None, sensitivity: float = 0.5,
            method: str = "harmonic", texture_refine: bool = False,
            mask_path: str | None = None, debug_path: str | None = None,
            compare_path: str | None = None,
            auto: bool = True, extra: dict | None = None) -> dict:
    rgb, alpha, _size = imageio.load_image(image_path)
    h, w = rgb.shape[:2]

    if region is not None and not isinstance(region, tuple):
        region = parse_region(str(region), w, h)

    a = analyse(rgb, alpha, region=region, sensitivity=sensitivity, auto=auto)
    mask = a.mask_result.mask
    out = remove(rgb, mask, method=method, texture_refine=texture_refine)

    imageio.save_image(out_path, out, alpha=alpha)
    if mask_path:
        imageio.save_mask_png(mask_path, mask)
    if debug_path:
        imageio.save_image(debug_path, imageio.overlay_visual(rgb, mask))
    if compare_path:
        imageio.save_side_by_side(compare_path, rgb, out, labels=("before", "after"))

    report = analysis_report(a, image_path=image_path, extra=extra)
    report["output"] = {"path": os.path.abspath(out_path), "method": method,
                        "texture_refine": bool(texture_refine)}
    report["removal"] = {
        "mask_pixels": int(mask.sum()),
        "mask_ratio": round(float(mask.sum()) / float(h * w), 5),
        "bbox": list(analysis.morph.bbox_of(mask) or (0, 0, 0, 0)),
    }
    return report


# --------------------------------------------------------------------------- #
# 按样本（模板）定位并去除
# --------------------------------------------------------------------------- #
def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = ((ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter)
    return inter / max(1.0, union)


def _median_region(boxes):
    xs0 = sorted(b[0] for b in boxes)
    ys0 = sorted(b[1] for b in boxes)
    xs1 = sorted(b[2] for b in boxes)
    ys1 = sorted(b[3] for b in boxes)
    mid = len(boxes) // 2
    return (xs0[mid], ys0[mid], xs1[mid], ys1[mid])


def process_many(paths, out_dir: str, *, region=None, sensitivity: float = 0.5,
                 method: str = "harmonic", texture_refine: bool = False,
                 auto: bool = True, sample=None, on_item=None):
    """批量处理：先逐张跑，再让同尺寸的图共享“多数派区域”。

    同一批图的水印位置通常一致；个别图自动检测跑偏时，用多数图的结果补一遍，
    比逐张各猜各的稳定得多。返回 [(path, dst, report|None, error|None), ...]
    """
    os.makedirs(out_dir, exist_ok=True)
    results = []

    def _dst_for(path):
        stem, ext = os.path.splitext(os.path.basename(path))
        ext = ext.lower()
        return os.path.join(out_dir, f"{stem}_clean{ext if ext in {'.png', '.webp'} else '.png'}")

    for i, path in enumerate(paths):
        dst = _dst_for(path)
        report, error = None, None
        try:
            if sample is not None:
                report = process_sample(path, dst, sample, method=method,
                                        texture_refine=texture_refine,
                                        sensitivity=sensitivity)
            else:
                report = process(path, dst, region=region, sensitivity=sensitivity,
                                 method=method, texture_refine=texture_refine, auto=auto)
        except LookupError as exc:
            error = ("skip", str(exc))
        except Exception as exc:  # noqa: BLE001
            error = ("fail", str(exc))
        results.append([path, dst, report, error])
        if on_item is not None:
            on_item(i, path, dst, report, error)

    # 同尺寸、自动检测、无样本时：用多数派区域校正离群图
    if auto and region is None and sample is None:
        groups: dict[tuple[int, int], list[int]] = {}
        for idx, (_p, _d, rep, err) in enumerate(results):
            if err or not rep:
                continue
            size = tuple(rep.get("image", {}).get("size") or ())
            if len(size) == 2:
                groups.setdefault(size, []).append(idx)
        for _size, idxs in groups.items():
            if len(idxs) < 3:
                continue
            good = []
            for idx in idxs:
                rep = results[idx][2]
                ratio = rep.get("removal", {}).get("mask_ratio", 0.0)
                cands = rep.get("candidates") or []
                if 0.0015 <= ratio <= 0.08 and cands:
                    good.append(tuple(cands[0]["bbox"]))
            if len(good) < 3:
                continue
            consensus = _median_region(good)
            for idx in idxs:
                rep = results[idx][2]
                cands = rep.get("candidates") or []
                if not cands:
                    continue
                if _iou(tuple(cands[0]["bbox"]), consensus) >= 0.45:
                    continue
                path, dst = results[idx][0], results[idx][1]
                try:
                    new_rep = process(path, dst, region=consensus,
                                      sensitivity=sensitivity, method=method,
                                      texture_refine=texture_refine, auto=False)
                    new_rep.setdefault("consensus", {})["region"] = list(consensus)
                    results[idx][2] = new_rep
                except Exception:  # noqa: BLE001
                    pass
    return results


def locate_sample(rgb: np.ndarray, sample, *, bg=None, min_score: float = 0.5,
                  max_matches: int = 4) -> list[dict]:
    """用样本在图中定位水印，返回匹配列表。"""
    if bg is None:
        bg = analysis.median_background(rgb, 31)
    return template.locate(rgb, sample, bg=bg, min_score=min_score,
                           max_matches=max_matches)


def sample_mask(rgb: np.ndarray, sample, matches, *, bg=None,
                sensitivity: float = 0.5, mode: str = "stamp") -> np.ndarray:
    """把多个匹配位置合成一张掩膜（默认按样本形状，直接当水印擦掉）。"""
    if bg is None:
        bg = analysis.median_background(rgb, 31)
    mask = np.zeros(rgb.shape[:2], bool)
    for m in matches:
        mask |= template.mask_at(rgb, sample, m, bg=bg,
                                 sensitivity=sensitivity, mode=mode)
    return mask


def process_sample(image_path: str, out_path: str, sample, *,
                   min_score: float = 0.5, max_matches: int = 4,
                   method: str = "harmonic", texture_refine: bool = False,
                   sensitivity: float = 0.5, mode: str = "stamp",
                   mask_path: str | None = None,
                   debug_path: str | None = None,
                   compare_path: str | None = None) -> dict:
    """用样本定位水印并去除，返回报告。

    mode="stamp"：样本就是水印，按它的形状直接擦（默认）；
    mode="region"：在匹配位置重新做叠加层分析，样本不太干净时更保险。
    """
    rgb, alpha, _size = imageio.load_image(image_path)
    h, w = rgb.shape[:2]
    bg = analysis.median_background(rgb, 31)
    matches = template.locate(rgb, sample, bg=bg,
                              min_score=min_score, max_matches=max_matches)
    if not matches:
        raise LookupError("没有在图片中找到与样本匹配的水印（可以调低 --min-score 试试）")

    mask = np.zeros((h, w), bool)
    for m in matches:
        mask |= template.mask_at(rgb, sample, m, bg=bg,
                                 sensitivity=sensitivity, mode=mode)
    out = remove(rgb, mask, method=method, texture_refine=texture_refine)

    imageio.save_image(out_path, out, alpha=alpha)
    if mask_path:
        imageio.save_mask_png(mask_path, mask)
    if compare_path:
        imageio.save_side_by_side(compare_path, rgb, out, labels=("before", "after"))
    if debug_path:
        imageio.save_image(debug_path, imageio.overlay_visual(rgb, mask))

    return {
        "image": {"path": os.path.abspath(image_path), "width": int(w),
                  "height": int(h), "has_alpha": alpha is not None},
        "sample": sample.summary(),
        "mode": mode,
        "matches": template.match_report(matches),
        "output": {"path": os.path.abspath(out_path), "method": method,
                   "texture_refine": bool(texture_refine)},
        "removal": {
            "mask_pixels": int(mask.sum()),
            "mask_ratio": round(float(mask.sum()) / float(h * w), 5),
            "bbox": list(analysis.morph.bbox_of(mask) or (0, 0, 0, 0)),
        },
    }
