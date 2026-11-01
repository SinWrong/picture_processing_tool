"""调和（Laplace）填补：多尺度求解，得到无方向性接缝的平滑底色。

与 Telea 的快速行进法相比，大洞不会被“四边推进”的先后顺序影响，
不会出现交叉状色带；代价是纯扩散会糊掉纹理，因此通常与 texture 模块配合。
"""

from __future__ import annotations

import numpy as np


def _downsample(a: np.ndarray) -> np.ndarray:
    h, w = a.shape[:2]
    h2, w2 = (h + 1) // 2, (w + 1) // 2
    pad = np.zeros((h2 * 2, w2 * 2) + a.shape[2:], dtype=a.dtype)
    pad[:h, :w] = a
    if a.ndim == 3:
        return pad.reshape(h2, 2, w2, 2, a.shape[2]).mean(axis=(1, 3))
    return pad.reshape(h2, 2, w2, 2).mean(axis=(1, 3))


def _upsample(a: np.ndarray, shape) -> np.ndarray:
    h, w = shape
    img = a
    if a.ndim == 3:
        out = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)
        return out[:h, :w]
    out = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)
    return out[:h, :w]


def _jacobi(u: np.ndarray, known: np.ndarray, iters: int) -> np.ndarray:
    """在未知区域做 Jacobi 迭代（已知区域保持不动）。"""
    u = u.copy()
    k = known[..., None] if u.ndim == 3 else known
    for _ in range(iters):
        avg = (
            np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1)
        ) / 4.0
        u = np.where(k, u, avg)
    return u


def harmonic_fill(rgb: np.ndarray, mask: np.ndarray, levels: int = 4,
                  coarse_iters: int = 400, fine_iters: int = 40) -> np.ndarray:
    """用多尺度 Laplace 求解填补掩膜区域。"""
    img = rgb.astype(np.float32)
    hole = mask.astype(bool)
    if not hole.any():
        return rgb.astype(np.uint8).copy()

    pyr_img = [img]
    pyr_mask = [hole]
    for _ in range(levels):
        h, w = pyr_mask[-1].shape[:2]
        if min(h, w) <= 8 or pyr_mask[-1].sum() < 16:
            break
        small_img = _downsample(pyr_img[-1])
        small_mask = _downsample(pyr_mask[-1].astype(np.float32)) > 0.25
        pyr_img.append(small_img)
        pyr_mask.append(small_mask)

    u = pyr_img[-1].copy()
    known = ~pyr_mask[-1]
    if not known.all():
        # 用已知像素的均值初始化
        init = u[known].mean(axis=0) if known.any() else np.zeros(3, np.float32)
        u[~known] = init
        u = _jacobi(u, known, coarse_iters)

    for level in range(len(pyr_img) - 2, -1, -1):
        target = pyr_img[level]
        u = _upsample(u, target.shape[:2])
        known = ~pyr_mask[level]
        u = np.where(known[..., None], target, u)
        u = _jacobi(u, known, fine_iters)

    out = rgb.astype(np.float32).copy()
    out[hole] = u[hole]
    return np.clip(out, 0, 255).astype(np.uint8)


def harmonic_fill_region(rgb: np.ndarray, mask: np.ndarray, pad: int = 40,
                         levels: int = 5) -> np.ndarray:
    """只修复掩膜附近的局部窗口（速度快很多，边缘取自原图）。"""
    if not mask.any():
        return rgb.astype(np.uint8).copy()
    h, w = mask.shape[:2]
    ys, xs = np.nonzero(mask)
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + 1 + pad)
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + 1 + pad)
    sub = rgb[y0:y1, x0:x1].astype(np.uint8)
    sub_mask = mask[y0:y1, x0:x1]
    fixed = harmonic_fill(sub, sub_mask, levels=levels)
    out = rgb.copy()
    out[y0:y1, x0:x1] = fixed
    return out
