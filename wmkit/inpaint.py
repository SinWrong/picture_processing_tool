"""图像修复（inpainting）：Telea 快速行进法（FMM）的 numpy 实现。

纯 numpy + heapq，不依赖 OpenCV。修复质量接近 cv2.inpaint(INPAINT_TELEA)，
足够应付半透明文字水印、角标、细线框这类细长结构。
"""

from __future__ import annotations

import heapq

import numpy as np

KNOWN, BAND, INSIDE = 0, 1, 2
_BIG = 1e6

_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _disc_offsets(radius: int) -> np.ndarray:
    offs = [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if (dy or dx) and dy * dy + dx * dx <= radius * radius
    ]
    return np.array(offs, dtype=np.int32)


def _solve_t(T: np.ndarray, h: int, w: int, y: int, x: int) -> float:
    """解程函方程，得到像素 (y,x) 的距离场取值。"""
    def g(yy, xx):
        if 0 <= yy < h and 0 <= xx < w:
            v = T[yy, xx]
            return v if v < _BIG else _BIG
        return _BIG

    a1 = min(g(y - 1, x), g(y + 1, x))
    a2 = min(g(y, x - 1), g(y, x + 1))
    if a1 >= _BIG and a2 >= _BIG:
        return _BIG
    if a1 >= _BIG:
        return a2 + 1.0
    if a2 >= _BIG:
        return a1 + 1.0
    if abs(a1 - a2) >= 1.0:
        return min(a1, a2) + 1.0
    return 0.5 * (a1 + a2 + np.sqrt(max(0.0, 2.0 - (a1 - a2) ** 2)))


def telea_inpaint(rgb: np.ndarray, mask: np.ndarray, radius: int = 4) -> np.ndarray:
    """按掩膜修复图像。rgb: HxWx3，mask: HxW bool。返回修复后的 uint8 图像。"""
    img = np.asarray(rgb, dtype=np.float32).copy()
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    if not mask.any():
        return np.clip(img, 0, 255).astype(np.uint8)

    flags = np.where(mask, INSIDE, KNOWN).astype(np.uint8)
    T = np.where(mask, _BIG, 0.0).astype(np.float32)

    heap: list[tuple[float, int, int]] = []
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        boundary = (
            (y > 0 and flags[y - 1, x] == KNOWN)
            or (y < h - 1 and flags[y + 1, x] == KNOWN)
            or (x > 0 and flags[y, x - 1] == KNOWN)
            or (x < w - 1 and flags[y, x + 1] == KNOWN)
        )
        if boundary:
            flags[y, x] = BAND
            T[y, x] = 1.0
            heapq.heappush(heap, (1.0, y, x))

    offs = _disc_offsets(radius)
    offs_y = offs[:, 0]
    offs_x = offs[:, 1]
    dist2 = (offs_y ** 2 + offs_x ** 2).astype(np.float32)

    while heap:
        t, y, x = heapq.heappop(heap)
        if flags[y, x] != BAND:
            continue
        if t > T[y, x] + 1e-4:
            continue

        # ---- 用邻域已知像素估计当前像素颜色 ----
        # 注意：先算颜色、后标记 KNOWN，否则邻域梯度会读到当前像素的旧值
        ny = y + offs_y
        nx = x + offs_x
        ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        ny_c, nx_c = np.clip(ny, 0, h - 1), np.clip(nx, 0, w - 1)
        ok &= flags[ny_c, nx_c] == KNOWN
        if ok.any():
            yy, xx = ny[ok], nx[ok]
            dy = -offs_y[ok].astype(np.float32)
            dx = -offs_x[ok].astype(np.float32)
            d2 = dist2[ok]
            lev = 1.0 / (1.0 + np.abs(T[yy, xx] - T[y, x]))
            wgt = lev / d2
            vals = img[yy, xx]
            if T[y, x] <= 2.5:
                # 只在贴着洞边缘时使用梯度项（结构延续），大洞内部误用会发散
                gy, gx = _gradient(img, flags, yy, xx, h, w)
                corr = gx * dx[:, None] + gy * dy[:, None]
                np.clip(corr, -48.0, 48.0, out=corr)
                vals = vals + corr
            total = float(wgt.sum())
            if total > 0:
                img[y, x] = (vals * wgt[:, None]).sum(0) / total
        flags[y, x] = KNOWN

        # ---- 更新四个方向上的未知邻居的距离场 ----
        for dy, dx in _FOUR:
            nyy, nxx = y + dy, x + dx
            if 0 <= nyy < h and 0 <= nxx < w and flags[nyy, nxx] != KNOWN:
                flags[nyy, nxx] = BAND
                tn = _solve_t(T, h, w, nyy, nxx)
                T[nyy, nxx] = tn
                heapq.heappush(heap, (tn, nyy, nxx))

    return np.clip(img, 0, 255).astype(np.uint8)


def _gradient(img, flags, ys, xs, h, w):
    """对一组像素求中心差分梯度（未知邻域退化为中心差分不可用时取 0）。"""
    n = len(ys)
    gy = np.zeros((n, 3), np.float32)
    gx = np.zeros((n, 3), np.float32)
    for i, (y, x) in enumerate(zip(ys.tolist(), xs.tolist())):
        if y > 0 and y < h - 1 and flags[y - 1, x] == KNOWN and flags[y + 1, x] == KNOWN:
            gy[i] = (img[y + 1, x] - img[y - 1, x]) * 0.5
        if x > 0 and x < w - 1 and flags[y, x - 1] == KNOWN and flags[y, x + 1] == KNOWN:
            gx[i] = (img[y, x + 1] - img[y, x - 1]) * 0.5
    return gy, gx


def inpaint_region(rgb: np.ndarray, mask: np.ndarray, radius: int = 4, pad: int = 24) -> np.ndarray:
    """只修复掩膜所在的局部窗口，速度快很多（边缘取自原图）。"""
    if not mask.any():
        return rgb.astype(np.uint8).copy()
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + 1 + pad)
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + 1 + pad)
    sub = rgb[y0:y1, x0:x1].astype(np.uint8)
    sub_mask = mask[y0:y1, x0:x1]
    fixed = telea_inpaint(sub, sub_mask, radius=radius)
    out = rgb.copy()
    out[y0:y1, x0:x1] = fixed
    return out
