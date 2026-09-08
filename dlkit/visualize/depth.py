"""深度图可视化：彩色化与叠加（不依赖 matplotlib）。

彩色图约定与 YOLO26-Depth 一致：
- mode='metric'    按米线性归一化，暖色=远
- mode='disparity' 按逆深度 1/d 归一化，暖色=近
无效像素（<=0）渲染为黑色。
"""

import numpy as np
from PIL import Image


def _jet_lut(n=256):
    lut = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        x = i / (n - 1)
        r = min(max(1.5 - abs(4.0 * x - 3.0), 0.0), 1.0)
        g = min(max(1.5 - abs(4.0 * x - 2.0), 0.0), 1.0)
        b = min(max(1.5 - abs(4.0 * x - 1.0), 0.0), 1.0)
        lut[i] = [int(255 * r), int(255 * g), int(255 * b)]
    return lut


def _hot_lut(n=256):
    lut = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        x = i / (n - 1)
        r = min(3.0 * x, 1.0)
        g = min(max(3.0 * x - 1.0, 0.0), 1.0)
        b = min(max(3.0 * x - 2.0, 0.0), 1.0)
        lut[i] = [int(255 * r), int(255 * g), int(255 * b)]
    return lut


def _gray_lut(n=256):
    ramp = np.linspace(0, 255, n).astype(np.uint8)
    return np.stack([ramp] * 3, axis=1)


_COLORMAPS = {
    'jet': _jet_lut(),
    'hot': _hot_lut(),
    'gray': _gray_lut(),
}


def colorize_depth(depth, cmap='jet', vmin=None, vmax=None, mode='metric'):
    """把 (H,W) 深度图（米）转成 (H,W,3) uint8 彩色图。

    vmin/vmax 用于锁定颜色范围（米），缺省取 2%/98% 分位数。
    """
    depth = np.asarray(depth, dtype=np.float64)
    valid = depth > 0
    out = np.zeros(depth.shape + (3,), dtype=np.uint8)
    if not valid.any():
        return out
    if cmap not in _COLORMAPS:
        raise ValueError('未知 colormap %r，可选: %s' % (cmap, sorted(_COLORMAPS)))
    lut = _COLORMAPS[cmap]

    if mode == 'disparity':
        disp = 1.0 / np.maximum(depth, 1e-8)
        if vmin is None or vmax is None:
            lo, hi = np.percentile(disp[valid], 2), np.percentile(disp[valid], 98)
        else:
            lo, hi = 1.0 / vmax, 1.0 / vmin
        norm = np.where(valid, (disp - lo) / max(hi - lo, 1e-8), 0.0)
    else:
        if vmin is None or vmax is None:
            lo, hi = np.percentile(depth[valid], 2), np.percentile(depth[valid], 98)
        else:
            lo, hi = vmin, vmax
        norm = np.where(valid, (depth - lo) / max(hi - lo, 1e-8), 0.0)

    idx = np.clip((norm * (len(lut) - 1)).round().astype(np.int64), 0, len(lut) - 1)
    out = lut[idx]
    out[~valid] = 0
    return out


def overlay_depth(image, depth, alpha=0.5, **colorize_kwargs):
    """把彩色深度图半透明叠加到原图上，返回 (H,W,3) uint8。"""
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 1) * 255.0
        image = image.astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    color = colorize_depth(depth, **colorize_kwargs)
    if color.shape[:2] != image.shape[:2]:
        color = np.asarray(
            Image.fromarray(color).resize((image.shape[1], image.shape[0]), Image.BILINEAR)
        )
    return (image.astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha).astype(np.uint8)
