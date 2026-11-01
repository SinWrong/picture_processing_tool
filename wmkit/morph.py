"""轻量形态学 / 连通域工具（仅依赖 numpy，避免引入 OpenCV）。"""

from __future__ import annotations

from collections import deque

import numpy as np


def shift(mask: np.ndarray, dy: int, dx: int, fill: bool = False) -> np.ndarray:
    """平移布尔数组，越界部分用 fill 填充（不循环）。"""
    out = np.full_like(mask, fill)
    h, w = mask.shape
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    ys0s, xs0s = max(0, -dy), max(0, -dx)
    ys1s, xs1s = ys0s + (ys1 - ys0), xs0s + (xs1 - xs0)
    out[ys0:ys1, xs0:xs1] = mask[ys0s:ys1s, xs0s:xs1s]
    return out


def dilate(mask: np.ndarray, radius: int = 1, square: bool = False) -> np.ndarray:
    """圆形（或方形）结构元膨胀。"""
    if radius <= 0:
        return mask.copy()
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if not square and dy * dy + dx * dx > radius * radius:
                continue
            out |= shift(mask, dy, dx)
    return out


def erode(mask: np.ndarray, radius: int = 1, square: bool = False) -> np.ndarray:
    """圆形（或方形）结构元腐蚀。"""
    if radius <= 0:
        return mask.copy()
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if not square and dy * dy + dx * dx > radius * radius:
                continue
            out &= shift(mask, dy, dx, fill=True)
    return out


def label_components(mask: np.ndarray, connectivity: int = 8):
    """8 邻域连通域标记。

    返回 (labels HxW int32, count, areas list[int])，标签从 1 开始，0 为背景。
    """
    h, w = mask.shape
    labels = np.zeros((h, w), np.int32)
    neighbors = (
        [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        if connectivity == 8
        else [(-1, 0), (1, 0), (0, -1), (0, 1)]
    )
    ys, xs = np.nonzero(mask)
    visited = np.zeros_like(mask, dtype=bool)
    count = 0
    areas: list[int] = []
    todo = list(zip(ys.tolist(), xs.tolist()))
    for y0, x0 in todo:
        if visited[y0, x0]:
            continue
        count += 1
        q = deque([(y0, x0)])
        visited[y0, x0] = True
        labels[y0, x0] = count
        area = 0
        while q:
            y, x = q.popleft()
            area += 1
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    labels[ny, nx] = count
                    q.append((ny, nx))
        areas.append(area)
    return labels, count, areas


def remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    """去掉面积小于 min_area 的连通块。"""
    if min_area <= 1:
        return mask
    labels, n, areas = label_components(mask)
    keep = np.zeros(n + 1, dtype=bool)
    for i, a in enumerate(areas, start=1):
        keep[i] = a >= min_area
    return keep[labels]


def keep_components(mask: np.ndarray, min_area: int = 1, max_area: int | None = None):
    """保留指定面积范围内的连通块，返回 (mask, labels, areas)。"""
    labels, n, areas = label_components(mask)
    keep = np.zeros(n + 1, dtype=bool)
    for i, a in enumerate(areas, start=1):
        if a < min_area:
            continue
        if max_area is not None and a > max_area:
            continue
        keep[i] = True
    return keep[labels], labels, areas


def box_sum(values: np.ndarray, radius: int) -> np.ndarray:
    """盒式窗口求和（积分图实现，O(1)/像素）。"""
    a = np.asarray(values, dtype=np.float64)
    ii = a.cumsum(0).cumsum(1)
    ii = np.pad(ii, ((1, 0), (1, 0)))
    h, w = a.shape
    ys = np.arange(h)
    xs = np.arange(w)
    y0 = np.clip(ys - radius, 0, h)
    y1 = np.clip(ys + radius + 1, 0, h)
    x0 = np.clip(xs - radius, 0, w)
    x1 = np.clip(xs + radius + 1, 0, w)
    return (
        ii[y1][:, x1]
        - ii[y0][:, x1]
        - ii[y1][:, x0]
        + ii[y0][:, x0]
    )


def bbox_of(mask: np.ndarray):
    """返回掩膜的 (x0, y0, x1, y1) 闭区间包围盒，空掩膜返回 None。"""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
