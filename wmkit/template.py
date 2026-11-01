"""水印样本（模板）定位。

思路：同一个水印在不同图片里的**形状**是一样的，变的只是位置、大小和底图。
所以模板不放截图本身，而是放它的 **alpha 图**（水印笔画的不透明度），
在目标图上先用同样的叠加色反解出一张全图 alpha 图，再用归一化互相关（NCC）
做多尺度匹配 —— 换底图、换亮度都不影响。

对外接口：
    build_sample(rgb, bbox)      从一张图里截出样本
    save_sample / load_sample    存成 .npz
    locate(rgb, sample)          返回匹配到的位置 [{"bbox":..., "score":...}]
    mask_at(rgb, sample, match)  在匹配位置生成精确的像素掩膜
"""

from __future__ import annotations

from dataclasses import dataclass

import os

import numpy as np
from PIL import Image

from . import analysis, morph


# --------------------------------------------------------------------------- #
# 样本
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    alpha: np.ndarray          # (h, w) float32，水印不透明度
    color: np.ndarray          # (3,) float32，主叠加色
    crop: np.ndarray           # (h, w, 3) uint8，样本截图（便于查看）
    source: str = ""           # 来自哪张图
    bbox: tuple = (0, 0, 0, 0)  # 在原图中的位置
    colors: list | None = None  # 参与 alpha 反解的所有叠加色
    bg_luma: float = 0.0        # 采样时背景的中位亮度（用于抵消整体明暗差异）

    @property
    def size(self) -> tuple[int, int]:
        return int(self.alpha.shape[1]), int(self.alpha.shape[0])

    def all_colors(self) -> list:
        if self.colors:
            return [np.asarray(c, np.float32) for c in self.colors]
        return [np.asarray(self.color, np.float32)]

    def trim(self, keep: float = 0.05) -> "Sample":
        ys, xs = np.nonzero(self.alpha > keep)
        if len(xs) == 0:
            return self
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        if (x0, y0, x1, y1) == (0, 0, self.alpha.shape[1], self.alpha.shape[0]):
            return self
        return Sample(alpha=self.alpha[y0:y1, x0:x1],
                      color=self.color,
                      crop=self.crop[y0:y1, x0:x1] if self.crop.size else self.crop,
                      source=self.source, bbox=self.bbox, colors=self.colors,
                      bg_luma=self.bg_luma)

    def summary(self) -> dict:
        h, w = self.alpha.shape
        return {
            "size": [int(w), int(h)],
            "color": "#%02X%02X%02X" % tuple(int(round(max(0, min(255, c)))) for c in self.color),
            "colors": ["#%02X%02X%02X" % tuple(int(round(max(0, min(255, c)))) for c in col)
                       for col in self.all_colors()],
            "alpha_pixels": int((self.alpha > 0.1).sum()),
            "source": self.source,
        }


def build_sample(rgb: np.ndarray, bbox, *, sensitivity: float = 0.5,
                 bg: np.ndarray | None = None, name: str = "") -> Sample:
    """从 rgb 的 bbox 区域截取水印样本。

    alpha 取所有“被认可”的叠加色假设的最大值——水印常常不只一种颜色
    （比如白字 + 品牌色线框），匹配时也用同一组颜色去反解目标图。

    bbox 支持两种写法：``(x0, y0, x1, y1)`` 或 ``(x, y, 宽, 高)``
    （后者在 x1/x0 顺序反常时自动识别）。
    """
    x0, y0, x1, y1 = (int(v) for v in bbox)
    if x1 <= x0 or y1 <= y0:
        x0, y0, x1, y1 = x0, y0, x0 + max(1, x1), y0 + max(1, y1)
    bbox = (x0, y0, x1, y1)
    if bg is None:
        bg = analysis.median_background(rgb, 31)
    res = analysis.build_mask(rgb, bg, bbox, sensitivity=sensitivity, dilate=0)
    x0, y0, x1, y1 = res.bbox

    # 匹配用的“叠加色”：取最显著的几种（同色系去重），再补上纯白/纯黑兜底
    hyps = sorted((h for h in res.hypotheses if h.score > 0),
                  key=lambda h: h.score, reverse=True)
    colors: list[np.ndarray] = []
    for h in hyps:
        c = np.asarray(h.color, np.float32)
        if any(np.abs(c - c2).max() < 45 for c2 in colors):
            continue
        colors.append(c)
        if len(colors) >= 5:
            break
    for extra in (np.array([255.0, 255.0, 255.0], np.float32),):
        if not any(np.abs(extra - c2).max() < 20 for c2 in colors):
            colors.append(extra)
    if not hyps:
        raise ValueError("这块区域里没检测到明显的水印叠加层，换个更贴的框试试。")

    # 样本的形状：所有被认可的叠加层取最大（白字 + 品牌色框都算）
    alpha = np.zeros((y1 - y0, x1 - x0), np.float32)
    for h in hyps:
        np.maximum(alpha, h.alpha, out=alpha)

    main = colors[0]
    patch_bg = np.asarray(bg[y0:y1, x0:x1], np.float32)
    luma = 0.299 * patch_bg[..., 0] + 0.587 * patch_bg[..., 1] + 0.114 * patch_bg[..., 2]
    sample = Sample(alpha=alpha, color=main, crop=rgb[y0:y1, x0:x1].copy(),
                    source=name, bbox=(x0, y0, x1, y1), colors=colors,
                    bg_luma=float(np.median(luma)))
    return sample.trim()


def save_sample(path: str, sample: Sample) -> None:
    np.savez_compressed(path, alpha=sample.alpha.astype(np.float16),
                        color=sample.color.astype(np.float32),
                        colors=np.asarray(sample.all_colors(), np.float32),
                        crop=sample.crop,
                        source=np.array(sample.source),
                        bbox=np.array(sample.bbox, np.int32),
                        bg_luma=np.float32(sample.bg_luma))


def load_sample(path: str) -> Sample:
    data = np.load(path, allow_pickle=False)
    bbox = tuple(int(v) for v in data["bbox"]) if "bbox" in data.files else (0, 0, 0, 0)
    source = str(data["source"]) if "source" in data.files else ""
    colors = [c for c in data["colors"]] if "colors" in data.files else None
    return Sample(alpha=data["alpha"].astype(np.float32),
                  color=data["color"].astype(np.float32),
                  crop=data["crop"] if "crop" in data.files else np.zeros((0, 0, 3), np.uint8),
                  source=source, bbox=bbox, colors=colors,
                  bg_luma=float(data["bg_luma"]) if "bg_luma" in data.files else 0.0)


def load_sample_any(path: str) -> Sample:
    """载入样本：支持 .npz，也支持直接把一张图片当样本。

    - ``.npz``：之前用 sample 子命令存下的样本；
    - 图片（png/jpg/…）：
        * **真的带透明区**（alpha 有变化）→ 直接用 alpha 当水印形状；
        * 否则（比如白底上画的水印、alpha 全是 255）→ 当成“水印 + 背景”，
          现场反解 alpha。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".npz", ".npy"):
        return load_sample(path)

    from . import imageio  # 延迟导入，避免循环

    rgb, alpha, _size = imageio.load_image(path)
    name = os.path.basename(path)
    if alpha is not None and _has_real_transparency(alpha):
        a = (alpha.astype(np.float32) / 255.0)
        sel = a > 0.5
        color = (np.median(rgb[sel].astype(np.float32), axis=0) if sel.any()
                 else np.array([255.0, 255.0, 255.0], np.float32))
        return Sample(alpha=a, color=color, crop=rgb, source=name,
                      bbox=(0, 0, rgb.shape[1], rgb.shape[0]),
                      colors=[color], bg_luma=0.0)

    # 普通图片（白底/彩底的水印图），整张当作水印区域现场分析
    h, w = rgb.shape[:2]
    if h * w > 250_000:
        raise ValueError(
            "这张图太大了，不适合整张当样本。请先裁出只含水印的一小块，"
            "或在图形界面里载入图片→框选水印→“选区存为样本”。"
        )
    return build_sample(rgb, (0, 0, w, h), name=name)


def _has_real_transparency(alpha: np.ndarray) -> bool:
    """alpha 通道是否真的表达了“透明区”（而不是整张都不透明）。"""
    a = alpha.astype(np.float32) / 255.0
    if float(a.min()) > 0.9:          # 全不透明
        return False
    return float((a < 0.5).mean()) > 0.02   # 至少 2% 的像素是透明的


def _luma_scale(bg: np.ndarray, sample: Sample) -> float:
    """目标图与采样图的整体明暗比，用来把样本颜色缩放到同一“曝光”。"""
    if not sample.bg_luma:
        return 1.0
    b = np.asarray(bg, np.float32)
    luma = 0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]
    cur = float(np.median(luma))
    if cur <= 1:
        return 1.0
    return float(np.clip(cur / sample.bg_luma, 0.6, 1.6))


def _sample_alpha_map(rgb: np.ndarray, bg: np.ndarray, sample: Sample) -> np.ndarray:
    """在目标图上，用样本的叠加色反解出全图 alpha 图。

    - 多色取最大：样本里白字 + 品牌色线框都会参与；
    - 同时试“原色”和“按整体明暗缩放后的颜色”，取更强者——
      既能应对整体曝光变化，也能应对只换了底图、水印颜色不变的情况。
    """
    out = None
    k = _luma_scale(bg, sample)
    scales = [1.0] if abs(k - 1.0) < 0.02 else [1.0, k]
    for c0 in sample.all_colors():
        for kk in scales:
            c = np.asarray(c0, np.float32) * kk
            _ok, a = analysis.matte(rgb, bg, c, min_alpha=0.03, max_spread=0.6)
            if out is None:
                out = a.astype(np.float32)
            else:
                np.maximum(out, a, out=out)
    return out if out is not None else np.zeros(rgb.shape[:2], np.float32)


# --------------------------------------------------------------------------- #
# 多尺度归一化互相关
# --------------------------------------------------------------------------- #
def _window_sum(a: np.ndarray, th: int, tw: int) -> np.ndarray:
    ii = np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return ii[th:, tw:] - ii[:-th, tw:] - ii[th:, :-tw] + ii[:-th, :-tw]


def _box3(a: np.ndarray) -> np.ndarray:
    """3×3 均值模糊，给匹配留 1~2 像素的容差。"""
    p = np.pad(a, 1, mode="edge")
    return (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
            p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
            p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]) / 9.0


def _corr(image_fft: np.ndarray, kernel: np.ndarray, shape, th: int, tw: int) -> np.ndarray:
    H, W = shape
    K = np.fft.rfft2(kernel, s=(H, W))
    out = np.fft.irfft2(image_fft * np.conj(K), s=(H, W))
    return out[:H - th + 1, :W - tw + 1]


def _features(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """两种表示：模糊 alpha（看强弱）+ 模糊二值（看形状），抗曝光/压缩差异。"""
    soft = _box3(alpha)
    hard = _box3((alpha > 0.2).astype(np.float32))
    return soft, hard


class _Prepared:
    """把整图特征及其 FFT 预计算一次，供多个尺度复用。"""

    def __init__(self, alpha: np.ndarray):
        self.shape = alpha.shape
        self.soft, self.hard = _features(alpha)
        self.F_soft = np.fft.rfft2(self.soft)
        self.F2_soft = np.fft.rfft2(self.soft * self.soft)
        self.F_hard = np.fft.rfft2(self.hard)
        self.F2_hard = np.fft.rfft2(self.hard * self.hard)

    def ncc(self, tpl: np.ndarray) -> np.ndarray | None:
        th, tw = tpl.shape
        H, W = self.shape
        if th > H or tw > W or th < 4 or tw < 4:
            return None
        w = (tpl > 0.1).astype(np.float32)
        n = float(w.sum())
        if n < 12:
            return None
        t_soft, t_hard = _features(tpl)

        def _one(img, F, F2, T):
            t_mean = float((T * w).sum() / n)
            t0 = (T - t_mean) * w
            var_t = float((t0 * t0).sum() / n)
            if var_t < 1e-6:
                return None
            num = _corr(F, t0, (H, W), th, tw)
            s1 = _corr(F, w, (H, W), th, tw)
            s2 = _corr(F2, w, (H, W), th, tw)
            mean = s1 / n
            var_i = s2 / n - mean * mean
            return num / (n * np.sqrt(np.maximum(var_i, 1e-6) * var_t))

        r1 = _one(self.soft, self.F_soft, self.F2_soft, t_soft)
        r2 = _one(self.hard, self.F_hard, self.F2_hard, t_hard)
        if r1 is None and r2 is None:
            return None
        if r1 is None:
            return r2
        if r2 is None:
            return r1
        return 0.5 * r1 + 0.5 * r2


def _resize_mask(alpha: np.ndarray, size) -> np.ndarray:
    w, h = size
    if (w, h) == (alpha.shape[1], alpha.shape[0]):
        return alpha
    im = Image.fromarray(alpha.astype(np.float32), mode="F").resize((w, h), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def _down(arr: np.ndarray, f: int) -> np.ndarray:
    if f <= 1:
        return arr
    h, w = arr.shape[:2]
    h2, w2 = (h // f) * f, (w // f) * f
    a = arr[:h2, :w2]
    if a.ndim == 3:
        return a.reshape(h2 // f, f, w2 // f, f, a.shape[2]).mean(axis=(1, 3))
    return a.reshape(h2 // f, f, w2 // f, f).mean(axis=(1, 3))


def _local_alpha(rgb, bg, sample, box, pad: int):
    """只在一个小窗口里反解 alpha，返回 (y0, x0, alpha)。"""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, np.zeros((1, 1), np.float32)
    a = _sample_alpha_map(rgb[y0:y1, x0:x1], bg[y0:y1, x0:x1], sample)
    return y0, x0, a.astype(np.float32)


def locate(rgb: np.ndarray, sample: Sample, *, bg: np.ndarray | None = None,
           a_img: np.ndarray | None = None, scales=None, min_score: float = 0.5,
           max_matches: int = 6) -> list[dict]:
    """做多尺度模板匹配，返回 [{bbox, score, scale}, ...]。

    大图会先在降采样版本上粗扫（快 4 倍），再在候选位置用原分辨率微调。
    """
    if bg is None:
        bg = analysis.median_background(rgb, 31)
    if scales is None:
        scales = [round(0.5 + 0.1 * k, 2) for k in range(0, 12)]   # 0.5 .. 1.6

    h, w = rgb.shape[:2]
    ds = 1
    if a_img is not None:
        alpha = a_img
    else:
        ds = 2 if h * w > 700_000 else 1
        if ds > 1:
            alpha = _sample_alpha_map(_down(rgb, ds), _down(bg, ds), sample)
        else:
            alpha = _sample_alpha_map(rgb, bg, sample)
    H, W = alpha.shape
    prep = _Prepared(alpha)
    peaks: list[tuple[float, int, int, float, int, int]] = []
    for s in scales:
        tw = max(3, int(round(sample.alpha.shape[1] * s / ds)))
        th = max(3, int(round(sample.alpha.shape[0] * s / ds)))
        if th >= H or tw >= W:
            continue
        tpl = _resize_mask(sample.alpha, (tw, th))
        if (tpl > 0.15).sum() < 8:
            continue
        r = prep.ncc(tpl)
        if r is None:
            continue
        ys, xs = np.nonzero(r >= min_score)
        if len(xs) == 0:
            continue
        order = np.argsort(r[ys, xs])[::-1][:30]
        for i in order:
            peaks.append((float(r[ys[i], xs[i]]), int(xs[i]) * ds, int(ys[i]) * ds,
                          float(s), tw * ds, th * ds))

    peaks.sort(key=lambda p: p[0], reverse=True)
    chosen: list[dict] = []
    for score, x, y, s, tw, th in peaks:
        if len(chosen) >= max_matches:
            break
        rad_x, rad_y = max(8, tw // 3), max(8, th // 3)
        if any(abs(x - c["x0"]) < rad_x and abs(y - c["y0"]) < rad_y for c in chosen):
            continue
        chosen.append({"x0": x, "y0": y, "x1": x + tw, "y1": y + th,
                       "score": round(score, 4), "scale": round(s, 3),
                       "bbox": (x, y, x + tw, y + th)})

    # 在原分辨率上局部微调（±0.05 尺度、±30% 位置）
    for m in chosen:
        tw0, th0 = m["x1"] - m["x0"], m["y1"] - m["y0"]
        pad = max(16, int(max(tw0, th0) * 0.45))
        y0l, x0l, a_local = _local_alpha(rgb, bg, sample,
                                         (m["x0"], m["y0"], m["x1"], m["y1"]), pad)
        if a_local.size > 1:
            m_shift = dict(m)
            for key in ("x0", "y0", "x1", "y1"):
                m_shift[key] = m[key] - (x0l if key in ("x0", "x1") else y0l)
            m_shift["bbox"] = (m_shift["x0"], m_shift["y0"], m_shift["x1"], m_shift["y1"])
            m2 = _refine(a_local, sample, m_shift)
            if m2 is not None:
                for key in ("x0", "y0", "x1", "y1"):
                    m2[key] = m2[key] + (x0l if key in ("x0", "x1") else y0l)
                m2["bbox"] = (m2["x0"], m2["y0"], m2["x1"], m2["y1"])
                m.update(m2)

    # 用“局部 alpha 强度比”压掉弱匹配（真水印处强度接近模板）
    for m in chosen:
        y0l, x0l, a_local = _local_alpha(rgb, bg, sample,
                                         (m["x0"], m["y0"], m["x1"], m["y1"]),
                                         max(6, int(max(m["x1"] - m["x0"], m["y1"] - m["y0"]) * 0.1)))
        if a_local.size > 1:
            sub = {"bbox": (m["x0"] - x0l, m["y0"] - y0l,
                            m["x1"] - x0l, m["y1"] - y0l)}
            m["score"] = round(_ratio_adjusted(a_local, sample, sub) * m["score"], 4)

    if chosen:
        chosen.sort(key=lambda c: c["score"], reverse=True)
        best = chosen[0]["score"]
        chosen = [c for c in chosen if c["score"] >= max(min_score * 0.8, best - 0.12)]
    return chosen


def _ratio_adjusted(alpha_img: np.ndarray, sample: Sample, match: dict) -> float:
    """匹配处的 alpha 强度与模板是否一致，返回 0~1 的置信系数。"""
    x0, y0, x1, y1 = match["bbox"]
    tpl = _resize_mask(sample.alpha, (x1 - x0, y1 - y0))
    ink = tpl > 0.1
    if ink.sum() < 8:
        return 1.0
    patch = alpha_img[y0:y1, x0:x1]
    if patch.shape != tpl.shape:
        return 1.0
    ratio = float(patch[ink].mean() / max(1e-6, tpl[ink].mean()))
    if ratio >= 0.75:
        return 1.0
    return max(0.0, min(1.0, ratio / 0.75))


def _refine(alpha_img: np.ndarray, sample: Sample, match: dict) -> dict | None:
    x0, y0, x1, y1 = match["bbox"]
    tw0, th0 = x1 - x0, y1 - y0
    pad_x = max(8, int(tw0 * 0.3))
    pad_y = max(8, int(th0 * 0.3))
    H, W = alpha_img.shape
    wx0, wy0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    wx1, wy1 = min(W, x1 + pad_x), min(H, y1 + pad_y)
    win = alpha_img[wy0:wy1, wx0:wx1]
    if win.shape[0] < 12 or win.shape[1] < 12:
        return None
    prep = _Prepared(win)
    best = None
    for s in (match["scale"] - 0.06, match["scale"] - 0.03, match["scale"],
              match["scale"] + 0.03, match["scale"] + 0.06):
        tw = max(3, int(round(sample.alpha.shape[1] * s)))
        th = max(3, int(round(sample.alpha.shape[0] * s)))
        if th >= win.shape[0] or tw >= win.shape[1]:
            continue
        tpl = _resize_mask(sample.alpha, (tw, th))
        r = prep.ncc(tpl)
        if r is None:
            continue
        iy, ix = np.unravel_index(int(np.argmax(r)), r.shape)
        sc = float(r[iy, ix])
        if best is None or sc > best[0]:
            best = (sc, wx0 + int(ix), wy0 + int(iy), s, tw, th)
    if best is None:
        return None
    sc, nx, ny, s, tw, th = best
    return {"x0": nx, "y0": ny, "x1": nx + tw, "y1": ny + th,
            "score": round(sc, 4), "scale": round(s, 3),
            "bbox": (nx, ny, nx + tw, ny + th)}


def mask_at(rgb: np.ndarray, sample: Sample, match: dict, *,
            bg: np.ndarray | None = None, a_img: np.ndarray | None = None,
            sensitivity: float = 0.5, pad_ratio: float = 0.08, dilate: int = 3,
            mode: str = "stamp", threshold: float = 0.12) -> np.ndarray:
    """按“样本就是水印”生成掩膜。

    mode="stamp"（默认）：把样本的水印形状（alpha）按匹配到的位置和缩放铺上去，
    只在边缘 2 像素内借助目标图的 alpha 补一下抗锯齿——不重新“分析”这块区域，
    避免把旁边的草/其它内容也当成水印。
    mode="region"：旧做法，用区域分析重新找叠加层（样本不太干净时可用）。
    """
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = match["bbox"]
    mask = np.zeros((h, w), bool)

    if bg is None:
        bg = analysis.median_background(rgb, 31)

    if mode == "region":
        pw = int((x1 - x0) * pad_ratio) + 3
        ph = int((y1 - y0) * pad_ratio) + 3
        bx0, by0 = max(0, x0 - pw), max(0, y0 - ph)
        bx1, by1 = min(w, x1 + pw), min(h, y1 + ph)
        try:
            res = analysis.build_mask(rgb, bg, (bx0, by0, bx1, by1), sensitivity=sensitivity)
            mask |= res.mask
        except Exception:  # noqa: BLE001
            pass
        if not mask.any():
            warped = _resize_mask(sample.alpha, (x1 - x0, y1 - y0))
            mask[y0:y1, x0:x1] |= warped > threshold
        if dilate:
            mask = morph.dilate(mask, dilate)
        return mask

    # ---- 默认：样本形状即水印 ----
    tpl = _resize_mask(sample.alpha, (x1 - x0, y1 - y0))
    core = tpl > threshold
    sub = np.zeros_like(core)
    sub |= core
    if core.any():
        # 以样本形状为基准，再允许 5px 内“目标图确实有叠加”的像素并进来：
        # 抵消轻微缩放/位移误差，同时不会把旁边的无关内容算进来
        if a_img is None:
            pad = 8
            y0l, x0l, a_local = _local_alpha(rgb, bg, sample, (x0, y0, x1, y1), pad)
            local_a = a_local[y0 - y0l:y1 - y0l, x0 - x0l:x1 - x0l]
        else:
            local_a = a_img[y0:y1, x0:x1]
        band = morph.dilate(core, 5) & ~core
        sub |= band & (local_a > 0.12)
    mask[y0:y1, x0:x1] = sub

    # 样本形状定了“水印在哪”，再让目标图自己确认边界：
    # 在样本附近做一次区域分析，把确实有叠加的像素补进来（限定在样本周围 6px 内）
    try:
        pw = int((x1 - x0) * 0.10) + 4
        ph = int((y1 - y0) * 0.10) + 4
        bx0, by0 = max(0, x0 - pw), max(0, y0 - ph)
        bx1, by1 = min(w, x1 + pw), min(h, y1 + ph)
        res = analysis.build_mask(rgb, bg, (bx0, by0, bx1, by1), sensitivity=sensitivity)
        if res.mask.any():
            near_stamp = morph.dilate(mask, 6)
            mask |= res.mask & near_stamp
    except Exception:  # noqa: BLE001
        pass

    # 水印被画面边缘裁掉时，把贴边的那一小条也补上
    hh, ww = sub.shape
    edge = 3
    if y0 <= 1:
        cols = np.nonzero(sub[0])[0]
        mask[max(0, y0 - edge):y0, x0 + cols] = True
    if y1 >= h - 1 or y1 == h:
        rows = sub[hh - 1]
        cols = np.nonzero(rows)[0]
        mask[y1:min(h, y1 + 1 + edge), x0 + cols] = True
    if x0 <= 1:
        rows = np.nonzero(sub[:, 0])[0]
        mask[y0 + rows, max(0, x0 - edge):x0] = True
    if x1 >= w - 1 or x1 == w:
        rows = np.nonzero(sub[:, ww - 1])[0]
        mask[y0 + rows, x1:min(w, x1 + 1 + edge)] = True

    solid = morph.dilate(mask, 1)  # 闭合细缝
    mask |= solid
    if dilate:
        mask = morph.dilate(mask, dilate)
    return mask


def stamp_preview(sample: Sample, scale: int = 3) -> Image.Image:
    """把样本渲染成“透明水印图”的样子（棋盘格底 + alpha 合成的 RGB）。"""
    h, w = sample.alpha.shape
    cell = 8
    yy, xx = np.mgrid[0:h, 0:w]
    board = (((yy // cell) + (xx // cell)) % 2).astype(np.float32) * 55 + 150
    board = np.dstack([board, board, board])

    rgb = sample.crop
    if rgb.shape[:2] != (h, w):
        rgb = np.asarray(Image.fromarray(rgb).resize((w, h), Image.BILINEAR))
    a = np.clip(sample.alpha, 0, 1)[..., None]
    comp = rgb.astype(np.float32) * a + board * (1 - a)
    img = Image.fromarray(np.clip(comp, 0, 255).astype(np.uint8))
    if scale and scale > 1:
        img = img.resize((w * scale, h * scale), Image.NEAREST)
    return img


def match_report(matches: list[dict]) -> list[dict]:
    return [
        {"bbox": list(m["bbox"]), "score": m["score"], "scale": m["scale"]}
        for m in matches
    ]
