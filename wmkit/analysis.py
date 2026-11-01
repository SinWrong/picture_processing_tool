"""水印分析核心。

流程：
1. 中值滤波估计“干净背景”，得到残差；
2. 在颜色直方图里找“叠加层颜色”（水印通常是少数几个固定颜色：
   白色文字、品牌色线框、灰字……）；
3. 对每个候选叠加色做透明度反解 I = a*C + (1-a)*B，用“三通道 alpha 一致”
   作为判据生成 matte；
4. 结合“相对纹理偏离度”和滞后阈值得到像素级掩膜，并给出覆盖率 / 透明度等指标；
5. 整图滑窗检测，给出水印候选区域与置信度。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter

from . import morph


# --------------------------------------------------------------------------- #
# 背景估计
# --------------------------------------------------------------------------- #
def median_background(rgb: np.ndarray, size: int = 31, max_dim: int = 900,
                      max_scale: float = 0.5) -> np.ndarray:
    """中值滤波估计背景。先降采样再滤波，最后双线性放大，兼顾速度。

    - 降采样后按比例缩小窗口，等效窗口仍约等于 ``size`` 像素；
    - 先做镜像填充，避免画面边缘（水印常贴边）污染背景估计。

    对 640×480 这种图，直接把中间结果缩到一半再滤波，速度快一个数量级，
    掩膜结果几乎不变（实测覆盖率差异 < 2%）。
    """
    h, w = rgb.shape[:2]
    scale = min(1.0, max_dim / float(max(h, w)))
    # 中小图再降一半：窗口会同步缩小，等效窗口大小不变
    if min(h, w) >= 360:
        scale = min(scale, max_scale)
    if scale < 0.999:
        small = np.asarray(
            Image.fromarray(rgb).resize(
                (max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR
            )
        )
    else:
        small = np.asarray(Image.fromarray(rgb))

    k = max(3, (int(size * scale) | 1))
    pad = k // 2
    padded = np.pad(small, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    chans = []
    for c in range(3):
        im = Image.fromarray(padded[..., c]).filter(ImageFilter.MedianFilter(size=k))
        chans.append(np.asarray(im, dtype=np.float32))
    bg = np.stack(chans, axis=-1)[pad:pad + small.shape[0], pad:pad + small.shape[1]]
    if scale < 0.999:
        bg = np.stack(
            [
                np.asarray(Image.fromarray(bg[..., c], mode="F").resize((w, h), Image.BILINEAR))
                for c in range(3)
            ],
            axis=-1,
        )
    return bg.astype(np.float32)


# --------------------------------------------------------------------------- #
# 颜色众数：画面里“成片出现且颜色固定”的颜色（水印的典型特征）
# --------------------------------------------------------------------------- #
def background_palette(bg: np.ndarray, q: int = 16, top: int = 4, min_frac: float = 0.02):
    """背景（中值滤波后的图）的主色调，用来判断某个颜色是不是“背景色”。"""
    I = bg.astype(np.int32)
    step = 256 // q
    idx = np.clip(I // step, 0, q - 1)
    flat = (idx[..., 0] * q + idx[..., 1]) * q + idx[..., 2]
    counts = np.bincount(flat.ravel(), minlength=q ** 3)
    need = max(1, int(min_frac * flat.size))
    order = np.argsort(counts)[::-1]
    out = []
    for b in order[:64]:
        if counts[b] < need or len(out) >= top:
            break
        out.append(np.array([b // (q * q), (b // q) % q, b % q], np.float32) * step + step / 2)
    return out


def color_modes(
    rgb: np.ndarray,
    bg: np.ndarray,
    *,
    q: int = 16,
    max_modes: int = 4,
    min_count_frac: float = 0.0012,
    min_count: int = 18,
    bg_distance: float = 46.0,
) -> list[np.ndarray]:
    """返回若干候选叠加色（RGB 0-255 float）。

    除了“出现最多”的颜色，还会补上“最鲜艳 / 最亮”的颜色：
    品牌 logo 往往只占很小面积、饱和度却远高于背景，只按频次取会把它漏掉
    （海水、珊瑚这类背景会把前几名频次全占满）。
    """
    I = rgb.astype(np.int32)
    step = 256 // q
    idx = np.clip(I // step, 0, q - 1)
    flat = (idx[..., 0] * q + idx[..., 1]) * q + idx[..., 2]
    area = flat.size
    counts = np.bincount(flat.ravel(), minlength=q ** 3)
    bg_med = np.median(bg.reshape(-1, 3), axis=0)
    need = max(min_count, int(min_count_frac * area))

    def _bin_color(b):
        return (np.array([b // (q * q), (b // q) % q, b % q], dtype=np.float32)
                * step + step / 2)

    def _accept(color, dist):
        if np.abs(color - bg_med).max() < bg_distance:
            return False
        return not any(np.abs(color - m).max() < dist for m in modes)

    modes: list[np.ndarray] = []
    for b in np.argsort(counts)[::-1][:400]:
        if counts[b] < need:
            break
        color = _bin_color(b)
        if not _accept(color, 26):
            continue
        modes.append(color)
        if len(modes) >= max_modes:
            break

    # 补：高饱和（logo / 彩色角标）与高亮（白色文字）
    mx = I.max(axis=2)
    mn = I.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1)
    for cond, extra in ((sat > 0.55, 3), (mx > 225, 2)):
        if int(cond.sum()) < max(min_count, 40):
            continue
        sub_counts = np.bincount(flat[cond], minlength=q ** 3)
        picked = 0
        for b in np.argsort(sub_counts)[::-1][:300]:
            if sub_counts[b] < max(need // 2, 20):
                break
            color = _bin_color(b)
            if not _accept(color, 40):
                continue
            modes.append(color)
            picked += 1
            if picked >= extra:
                break
    return modes


# --------------------------------------------------------------------------- #
# 透明度反解
# --------------------------------------------------------------------------- #
@dataclass
class Hypothesis:
    color: tuple[float, float, float]
    name: str
    alpha: np.ndarray
    mask: np.ndarray
    coverage: float = 0.0
    opacity: float = 0.0
    mean_alpha: float = 0.0
    score: float = 0.0
    thinness: float = 0.0


def matte(
    rgb,
    bg,
    color,
    *,
    min_alpha: float = 0.12,
    max_spread: float = 0.20,
    min_contrast: float = 22.0,
    min_delta: float = 5.0,
):
    """按 I = a*C + (1-a)*B 逐通道反解 alpha，返回 (low_mask, alpha)。"""
    I = rgb.astype(np.float32)
    B = bg.astype(np.float32)
    C = np.asarray(color, np.float32)
    d = C[None, None, :] - B
    delta = I - B
    valid = np.abs(d) > min_contrast
    cnt = valid.sum(axis=2)
    a = np.where(valid, delta / np.where(valid, d, 1.0), 0.0)
    mean_a = a.sum(2) / np.maximum(cnt, 1)
    amax = np.where(valid, a, -1e9).max(2)
    amin = np.where(valid, a, 1e9).min(2)
    spread = np.where(cnt >= 2, amax - amin, 0.0)
    mag = np.abs(delta).max(2)
    ok = (
        (cnt >= 2)
        & (mean_a >= min_alpha)
        & (mean_a <= 1.9)
        & (spread <= max_spread)
        & (mag >= min_delta)
    )
    alpha = np.where(ok, np.clip(mean_a, 0, 1.5), 0.0).astype(np.float32)
    return ok, alpha


def hysteresis(low: np.ndarray, core: np.ndarray, min_area: int = 6) -> np.ndarray:
    """滞后阈值：只保留包含“核心”像素、且面积够大的连通块。"""
    labels, n, areas = morph.label_components(low)
    if n == 0:
        return low
    flags = np.zeros(n + 1, bool)
    if core.any():
        flags[np.unique(labels[core])] = True
    flags[0] = False
    big = np.zeros(n + 1, bool)
    big[1:] = np.array(areas) >= min_area
    return (flags & big)[labels]


def local_texture(rgb: np.ndarray, bg: np.ndarray, radius: int = 5, cap: float = 30.0) -> np.ndarray:
    """局部纹理强度：邻域内 |Δ| 的均值（先截断，避免被水印本身拉高）。"""
    d = np.minimum(np.abs(rgb.astype(np.float32) - bg.astype(np.float32)).max(2), cap)
    ones = np.ones_like(d)
    return morph.box_sum(d, radius) / np.maximum(morph.box_sum(ones, radius), 1.0)


def _evaluate(
    rgb,
    bg,
    color,
    area,
    rel,
    dev,
    name="",
    core_alpha=0.45,
    min_area=7,
    rel_thr=2.2,
    max_spread=0.4,
    min_alpha=0.12,
    min_dev=18.0,
    max_area_frac=0.55,
) -> Hypothesis:
    C = np.asarray(color, np.float32)
    low, alpha = matte(rgb, bg, C, max_spread=max_spread, min_alpha=min_alpha)
    # 与背景的差异必须真实存在（否则“和背景同色”的假设会把自己算成 1.0 不透明）
    low &= dev >= min_dev
    # 强叠加（alpha 很高，说明就是该颜色本身）即便贴着纹理也可以通过
    low &= (rel > rel_thr) | (alpha >= 0.80)
    core = low & (alpha >= core_alpha)
    mask = hysteresis(low, core, min_area=min_area)
    h = Hypothesis(color=tuple(float(v) for v in C), name=name, alpha=alpha, mask=mask)
    n = int(mask.sum())
    if n == 0:
        return h
    h.coverage = n / float(area)
    sel = alpha[mask]
    h.mean_alpha = float(sel.mean())
    h.opacity = float((sel >= core_alpha).mean())
    h.thinness = 1.0 - float(morph.erode(mask, 2).sum()) / max(n, 1)
    # 叠加色必须明显区别于它覆盖的背景色，否则只是“背景本身的一种颜色”
    contrast = float(np.abs(C[None, None, :] - bg.astype(np.float32))[mask].max(axis=1).mean())
    if h.coverage > max_area_frac or h.coverage < 0.0015 or contrast < 45.0:
        return h
    h.score = (
        min(1.0, h.opacity / 0.5)
        * min(1.0, h.coverage / 0.05)
        * (0.25 + 0.75 * h.thinness)
    )
    return h


def _kmeans(values: np.ndarray, k: int, iters: int = 20) -> np.ndarray:
    """极简 k-means（只用于给背景颜色做聚类），values: (N, D) float。"""
    n = len(values)
    if n == 0:
        return np.zeros((0, values.shape[1]), np.float32)
    k = max(1, min(k, n))
    rng = np.random.default_rng(7)
    centers = [values[int(rng.integers(n))].copy()]
    for _ in range(k - 1):
        d = np.min(np.abs(values[:, None, :] - np.array(centers)[None]).max(axis=2), axis=1)
        centers.append(values[int(np.argmax(d))].copy())
    centers = np.array(centers, np.float32)
    for _ in range(iters):
        dist = np.abs(values[:, None, :] - centers[None]).max(axis=2)
        labels = dist.argmin(axis=1)
        for j in range(len(centers)):
            sel = labels == j
            if sel.any():
                centers[j] = values[sel].mean(axis=0)
    return centers


def ring_mask(rgb: np.ndarray, bbox, *, band: float = 0.18, k: int = 6,
              k_sigma: float = 1.8, min_dist: float = 20.0,
              max_dist: float = 95.0, dilate: int = 3) -> np.ndarray:
    """用“框外一圈”当背景模型，把框内不像背景的像素挑出来。

    适合用户手工框选的场景：框外是纯背景，框内是背景 + 水印/logo。
    比“猜叠加色”稳——背景再花，只要框外能代表它就行。
    """
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    out = np.zeros((h, w), bool)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return out

    padx = max(8, int((x1 - x0) * band))
    pady = max(8, int((y1 - y0) * band))
    ox0, oy0 = max(0, x0 - padx), max(0, y0 - pady)
    ox1, oy1 = min(w, x1 + padx), min(h, y1 + pady)

    outer = rgb[oy0:oy1, ox0:ox1].astype(np.float32)
    ring = np.ones(outer.shape[:2], bool)
    ring[y0 - oy0:y1 - oy0, x0 - ox0:x1 - ox0] = False
    vals = outer[ring]
    if len(vals) < 80:
        return out
    if len(vals) > 4000:
        vals = vals[:: max(1, len(vals) // 4000)]

    centers = _kmeans(vals, min(k, len(vals)))
    if len(centers) == 0:
        return out

    def _dist(arr):
        # 每个像素到最近背景中心的距离（通道差的最大值）
        d = np.abs(arr[..., None, :] - centers[None, None, :, :]).max(axis=3)
        return d.min(axis=2)

    d_ring = _dist(vals)
    thr = float(np.percentile(d_ring, 90) * k_sigma)
    thr = float(np.clip(thr, min_dist, max_dist))

    inner = rgb[y0:y1, x0:x1].astype(np.float32)
    d = _dist(inner)
    core = d > thr
    if not core.any():
        return out

    # 滞后阈值：淡色/细笔画（小字、光晕）只要和强像素连着就一起算进来
    low = d > thr * 0.55
    labels, n, areas = morph.label_components(low)
    if n:
        keep = np.zeros(n + 1, bool)
        keep[np.unique(labels[core])] = True
        keep[0] = False
        m = keep[labels]
    else:
        m = core

    m = morph.remove_small(m, max(13, int(0.0015 * m.size)))
    m = morph.erode(morph.dilate(m, 2), 2)     # 闭运算补缝
    if dilate:
        m = morph.dilate(m, dilate)
    out[y0:y1, x0:x1] = m
    return out


# --------------------------------------------------------------------------- #
# 区域掩膜
# --------------------------------------------------------------------------- #
@dataclass
class MaskResult:
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    hypotheses: list[Hypothesis] = field(default_factory=list)
    score: float = 0.0
    stats: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "bbox_size": [self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]],
            "mask_pixels": int(self.mask.sum()),
            "coverage_in_region": round(float(self.stats.get("coverage", 0.0)), 4),
            "score": round(float(self.score), 4),
            "overlays": [
                {
                    "color": "#%02X%02X%02X" % tuple(int(round(c)) for c in h.color),
                    "name": h.name,
                    "coverage": round(h.coverage, 4),
                    "opacity": round(h.opacity, 3),
                    "mean_alpha": round(h.mean_alpha, 3),
                }
                for h in self.hypotheses
                if h.score > 0
            ],
        }


def build_mask(
    rgb: np.ndarray,
    bg: np.ndarray,
    bbox,
    *,
    sensitivity: float = 0.5,
    extra_colors=(),
    dilate: int = 3,
    score_threshold: float = 0.12,
    max_colors: int = 7,
) -> MaskResult:
    """在给定矩形区域内分析叠加层，返回像素级掩膜。"""
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(W, int(x1)), min(H, int(y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"无效区域: {bbox}")

    roi = rgb[y0:y1, x0:x1]
    roi_bg = bg[y0:y1, x0:x1]
    area = roi.shape[0] * roi.shape[1]

    s = float(np.clip(sensitivity, 0.0, 1.0))
    min_alpha = 0.50 - 0.16 * s         # 0.50 .. 0.34
    max_spread = 0.30 + 0.30 * s        # 0.30 .. 0.60
    core_alpha = 0.62 - 0.14 * s        # 0.62 .. 0.48
    rel_thr = 1.8 - 1.2 * s             # 1.8 .. 0.6
    min_area = max(10, int(0.00015 * area))

    dev = np.abs(roi.astype(np.float32) - roi_bg.astype(np.float32)).max(2)
    tex = local_texture(roi, roi_bg)
    rel = dev / (tex + 6.0)

    colors: list[tuple[np.ndarray, str]] = []
    for c in extra_colors:
        colors.append((np.asarray(c, np.float32), "manual"))
    palette = background_palette(bg)
    modes: list[np.ndarray] = []
    for m in color_modes(roi, roi_bg):
        # 和背景主色调太近的颜色只是背景本身，不是水印
        if any(np.abs(m - p).max() < 34 for p in palette):
            continue
        modes.append(m)
    # 颜色太多会明显拖慢速度；白/黑兜底始终保留
    keep_modes = modes[: max(1, max_colors - len(extra_colors) - 2)]
    colors.extend((m, "mode") for m in keep_modes)
    colors.append((np.array([255.0, 255.0, 255.0]), "white"))
    colors.append((np.array([0.0, 0.0, 0.0]), "black"))

    hypotheses: list[Hypothesis] = []
    seen: list[np.ndarray] = []
    for color, name in colors:
        if any(np.abs(color - s2).max() < 26 for s2 in seen):
            continue
        seen.append(color)
        hypotheses.append(
            _evaluate(
                roi, roi_bg, color, area, rel, dev, name=name,
                core_alpha=core_alpha, min_area=min_area, rel_thr=rel_thr,
                max_spread=max_spread, min_alpha=min_alpha,
            )
        )

    keep = [h for h in hypotheses if h.score >= score_threshold]
    if not keep:
        keep = [max(hypotheses, key=lambda h: h.score)] if hypotheses else []

    mask = np.zeros(roi.shape[:2], bool)
    for h in keep:
        mask |= h.mask
    if dilate:
        mask = morph.dilate(mask, int(dilate))
    full = np.zeros((H, W), bool)
    full[y0:y1, x0:x1] = mask

    result = MaskResult(bbox=(x0, y0, x1, y1), mask=full, hypotheses=hypotheses)
    result.stats = {
        "region_area": area,
        "coverage": float(mask.sum()) / float(area),
        "core_alpha": core_alpha,
        "rel_thr": rel_thr,
        "min_alpha": min_alpha,
        "sensitivity": s,
    }
    result.score = max((h.score for h in keep), default=0.0)
    return result


# --------------------------------------------------------------------------- #
# 自动检测
# --------------------------------------------------------------------------- #
def detect_regions(
    rgb: np.ndarray,
    bg: np.ndarray,
    *,
    max_regions: int = 3,
    sensitivity: float = 0.5,
    scales=(0.50, 0.38, 0.28, 0.20),
    anchors=(0.0, 0.5, 1.0),
) -> list[MaskResult]:
    """滑窗检测水印候选区域，返回按置信度排序的结果。

    窗口按“贴边 / 贴角”的方式摆放（水印几乎总在画面边角），
    并用“文字排布特征”和“靠角程度”修正得分。
    """
    H, W = rgb.shape[:2]
    found: list[MaskResult] = []
    tried = set()
    for scale in scales:
        tw, th = max(48, int(W * scale)), max(48, int(H * scale))
        for fy in anchors:
            for fx in anchors:
                x0 = int(round((W - tw) * fx))
                y0 = int(round((H - th) * fy))
                x1, y1 = min(W, x0 + tw), min(H, y0 + th)
                key = (x0 // 16, y0 // 16, x1 // 16, y1 // 16)
                if key in tried:
                    continue
                tried.add(key)
                try:
                    res = build_mask(
                        rgb, bg, (x0, y0, x1, y1), sensitivity=sensitivity, dilate=0,
                        max_colors=5,
                    )
                except Exception:
                    continue
                cov = res.stats.get("coverage", 0.0)
                if not (0.004 <= cov <= 0.45):
                    continue
                res.score *= _layout_score(res.mask, (x0, y0, x1, y1), W, H)
                found.append(res)

    # 角窗：小 logo/小字几乎都贴在角上，网格常常盖不住（窗口太大或错位）。
    for fx, fy, boost in ((1.0, 1.0, 1.65), (0.5, 1.0, 1.15), (0.0, 1.0, 0.95),
                          (1.0, 0.0, 0.90), (0.0, 0.0, 0.90)):
        tw = max(64, int(W * 0.32))
        th = max(48, int(H * 0.18))
        x0 = 0 if fx == 0.0 else (W - tw if fx == 1.0 else int((W - tw) * 0.5))
        y0 = 0 if fy == 0.0 else H - th
        x1, y1 = min(W, x0 + tw), min(H, y0 + th)
        try:
            res = build_mask(rgb, bg, (x0, y0, x1, y1), sensitivity=sensitivity,
                             dilate=0, max_colors=5)
        except Exception:  # noqa: BLE001
            continue
        cov = res.stats.get("coverage", 0.0)
        if not (0.004 <= cov <= 0.45):
            continue
        res.score *= _layout_score(res.mask, (x0, y0, x1, y1), W, H) * boost
        found.append(res)

    found.sort(key=lambda r: r.score, reverse=True)
    merged: list[MaskResult] = []
    for res in found:
        if any(_iou(res.bbox, m.bbox) > 0.45 for m in merged):
            continue
        merged.append(res)
        if len(merged) >= max_regions:
            break
    return merged


def _layout_score(mask: np.ndarray, bbox, W: int, H: int) -> float:
    """按“水印的排版特征”修正候选得分：文字行、细结构、靠角。"""
    x0, y0, x1, y1 = bbox
    sub = mask[y0:y1, x0:x1]
    if not sub.any():
        return 0.05
    labels, n, areas = morph.label_components(sub)
    hh = y1 - y0
    ww = x1 - x0

    score = 1.0
    # 1) 文字行：多个高度相近的连通块（字）水平排列
    comps = []
    for i in range(1, n + 1):
        if areas[i - 1] < max(8, 0.0004 * ww * hh):
            continue
        ys, xs = np.nonzero(labels == i)
        comps.append((ys.min(), ys.max(), xs.min(), xs.max(), areas[i - 1]))
    if len(comps) >= 3:
        heights = np.array([c[1] - c[0] + 1 for c in comps], float)
        med = np.median(heights)
        same = np.mean(np.abs(heights - med) <= max(3, 0.35 * med))
        centers = np.array([(c[0] + c[1]) / 2 for c in comps], float)
        band = np.std(centers) <= max(4.0, 0.10 * hh)
        span = (max(c[3] for c in comps) - min(c[2] for c in comps)) / max(1.0, ww)
        if same >= 0.6 and band and span > 0.45:
            score *= 2.2          # 典型的一行字：字高一致、基线平、横向铺开
            if span > 0.85:
                score *= 0.75     # 铺满整个窗口的整行大标题多半是正文，不是水印
        elif same >= 0.5 and band:
            score *= 1.5
        elif same >= 0.4:
            score *= 1.1
        else:
            score *= 0.5          # 高度杂乱 → 纹理
        if len(comps) > 40:
            score *= 0.7          # 碎成一片 → 纹理
    elif len(comps) <= 2:
        score *= 0.8

    # 靠角程度：用水印“墨迹”的中心算（而不是窗口中心）——
    # 窗口常常为了贴边被拉得很长，窗口中心会误导判断
    ys_i, xs_i = np.nonzero(sub)
    ix0, ix1 = x0 + int(xs_i.min()), x0 + int(xs_i.max())
    iy0, iy1 = y0 + int(ys_i.min()), y0 + int(ys_i.max())
    cx, cy = (ix0 + ix1) / 2.0 / W, (iy0 + iy1) / 2.0 / H
    to_corner = min(
        (cx ** 2 + cy ** 2) ** 0.5,
        ((1 - cx) ** 2 + cy ** 2) ** 0.5,
        (cx ** 2 + (1 - cy) ** 2) ** 0.5,
        ((1 - cx) ** 2 + (1 - cy) ** 2) ** 0.5,
    )
    score *= max(0.75, 1.2 - 0.45 * to_corner)
    br = ((1 - cx) ** 2 + (1 - cy) ** 2) ** 0.5
    score *= 1.0 + 0.35 * max(0.0, 1.0 - br / 0.45)

    # 贴边数量：右下角的水印通常同时贴两条边
    edges = 0
    if ix0 <= 2:
        edges += 1
    if iy0 <= 2:
        edges += 1
    if ix1 >= W - 3:
        edges += 1
    if iy1 >= H - 3:
        edges += 1
    score *= 1.0 + 0.10 * max(0, edges - 1)

    # 掩膜不应铺满整个窗口（那是纹理，不是水印）
    density = float(sub.mean())
    if density > 0.32:
        score *= 0.7
    return score


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / max(ua, 1)
