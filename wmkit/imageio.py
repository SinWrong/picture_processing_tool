"""图像读取 / 保存工具。"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image


def load_image(path: str):
    """读取图片，返回 (rgb uint8 HxWx3, alpha uint8 HxW 或 None, 原始尺寸 tuple)。"""
    im = Image.open(path)
    size = im.size
    has_alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
    rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    alpha = None
    if has_alpha:
        alpha = np.asarray(im.convert("RGBA"), dtype=np.uint8)[..., 3]
    return rgb, alpha, (size[0], size[1])


def save_image(path: str, rgb: np.ndarray, alpha: np.ndarray | None = None) -> None:
    """保存 RGB(A) 图像，按扩展名推断格式（jpg 会以高质量保存）。"""
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if alpha is not None:
        im = Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA")
    else:
        im = Image.fromarray(rgb, mode="RGB")
    if ext in (".jpg", ".jpeg"):
        im = im.convert("RGB")
        im.save(path, quality=95, subsampling=0)
    else:
        im.save(path)


def save_mask_png(path: str, mask: np.ndarray) -> None:
    """把掩膜保存为黑白 PNG（白色 = 待修复区域）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def overlay_visual(rgb: np.ndarray, mask: np.ndarray, color=(255, 40, 40), alpha=0.65) -> np.ndarray:
    """把掩膜半透明地叠在图像上，用于调试查看。"""
    out = rgb.astype(np.float32)
    c = np.array(color, dtype=np.float32)
    m = mask[..., None].astype(np.float32) * alpha
    out = out * (1 - m) + c * m
    return np.clip(out, 0, 255).astype(np.uint8)


def save_side_by_side(path: str, before: np.ndarray, after: np.ndarray,
                      *, gap: int = 12, labels=("before", "after"),
                      bg: tuple[int, int, int] = (255, 255, 255)) -> None:
    """把“修改前 / 修改后”左右拼成一张图保存（两边等高，不足处留白）。"""
    a = np.asarray(before, dtype=np.uint8)
    b = np.asarray(after, dtype=np.uint8)
    h = max(a.shape[0], b.shape[0])
    w = a.shape[1] + b.shape[1] + gap
    canvas = np.full((h, w, 3), bg, np.uint8)
    canvas[: a.shape[0], : a.shape[1]] = a[..., :3]
    canvas[: b.shape[0], a.shape[1] + gap: a.shape[1] + gap + b.shape[1]] = b[..., :3]
    save_image(path, canvas)
