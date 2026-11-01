"""纹理细化：在 Telea 修复得到平滑基底之后，从邻近区域拷贝高频细节。

扩散式修复（Telea / 拉普拉斯）会把纹理糊掉；对草地、树叶这类高频背景，
把附近“同色系”位置的高频分量搬过来，观感会自然很多。
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from . import morph


def _box_blur(img: np.ndarray, r: int) -> np.ndarray:
    if r < 1:
        return img
    squeeze = img.ndim == 2
    if squeeze:
        img = img[..., None]
    pad = np.pad(img, ((r + 1, r), (r + 1, r), (0, 0)), mode="edge")
    ii = pad.cumsum(0).cumsum(1)
    k = 2 * r + 1
    out = (ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]) / float(k * k)
    return out[..., 0] if squeeze else out


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """三次盒式模糊近似高斯（纯 numpy，速度足够）。"""
    r = max(1, int(round(sigma * 0.9)))
    out = img.astype(np.float32)
    for _ in range(3):
        out = _box_blur(out, r)
    return out.astype(np.float32)


def _offsets(min_dist: int, max_dist: int, step: int = 6, count: int = 8):
    offs = []
    for d in range(min_dist, max_dist + 1, step):
        for a in range(count):
            ang = 2 * np.pi * a / count
            offs.append((int(round(np.cos(ang) * d)), int(round(np.sin(ang) * d))))
    # 去重
    return list(dict.fromkeys(offs))


def refine(rgb: np.ndarray, mask: np.ndarray, base: np.ndarray, *,
           sigma: float = 1.4, min_dist: int = 18, max_dist: int = 120,
           min_component: int = 60) -> np.ndarray:
    """把邻近纹理的高频细节补回 base 的掩膜区域。"""
    I = rgb.astype(np.float32)
    B = base.astype(np.float32)
    hp_src = I - _blur(I, sigma)          # 原图高频
    lp = _blur(B, sigma)                  # 修复后的低频

    h, w = mask.shape
    out = B.copy()
    labels, n, areas = morph.label_components(mask)
    for idx in range(1, n + 1):
        if areas[idx - 1] < min_component:
            continue
        comp = labels == idx
        ys, xs = np.nonzero(comp)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        best = None
        for dx, dy in _offsets(min_dist, max_dist):
            qy0, qy1 = y0 + dy, y1 + dy + 1
            qx0, qx1 = x0 + dx, x1 + dx + 1
            if qy0 < 0 or qx0 < 0 or qy1 > h or qx1 > w:
                continue
            donor_mask = mask[qy0:qy1, qx0:qx1]
            if donor_mask.any():
                continue
            diff = np.abs(lp[ys, xs] - lp[ys + dy, xs + dx]).mean()
            if best is None or diff < best[0]:
                best = (diff, dx, dy)
        if best is None:
            continue
        _score, dx, dy = best
        out[ys, xs] = lp[ys, xs] + hp_src[ys + dy, xs + dx]

    # 掩膜边缘羽化，避免出现硬边
    weight = np.zeros((h, w), np.float32)
    for _ in range(3):
        weight = np.maximum(weight, mask.astype(np.float32))
        weight = morph.dilate(weight > 0, 1).astype(np.float32) * mask
    inner = morph.erode(mask, 1)
    weight[inner] = 1.0
    weight = _blur(weight, 1.0)[..., None]
    return np.clip(B * (1 - weight) + out * weight, 0, 255).astype(np.uint8)
