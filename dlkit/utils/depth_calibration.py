"""log-深度头的尺度校准（YOLO26-Depth 同款思路）。

log-深度头预测的是「相对」log-深度：形状（相对结构）是对的，
绝对米数差一个全局尺度。用少量带真值深度的样本做闭式拟合，
得到双参数对数仿射变换 depth = exp(a·log d + b)，无需梯度训练。
"""

import numpy as np


def fit_log_affine(pred_log, gt):
    """拟合校准参数 (a, b)。

    pred_log: 模型预测的 log 深度（任意形状）
    gt:       真实深度（米，任意形状）
    """
    pred_log = np.asarray(pred_log, dtype=np.float64).ravel()
    gt = np.asarray(gt, dtype=np.float64).ravel()
    valid = np.isfinite(pred_log) & np.isfinite(gt) & (gt > 0)
    x = pred_log[valid]
    y = np.log(gt[valid])
    if x.size < 2:
        return 1.0, 0.0
    design = np.stack([x, np.ones_like(x)], axis=1)
    a, b = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(a), float(b)


def apply_log_affine(pred_log, a=1.0, b=0.0):
    """应用校准：depth = exp(a·log d + b)。"""
    return np.exp(np.asarray(pred_log, dtype=np.float64) * a + b)


def calibrate_from_pairs(pred_logs, gts):
    """从多张预测/真值对的列表拟合校准参数 (a, b)。"""
    p = np.concatenate([np.asarray(x, dtype=np.float64).ravel() for x in pred_logs])
    g = np.concatenate([np.asarray(x, dtype=np.float64).ravel() for x in gts])
    return fit_log_affine(p, g)


def median_scale_align(pred, gt, min_depth=1e-3, min_valid_pixels=10):
    """逐样本 median scaling 对齐：把无绝对尺度的预测深度对齐到真值尺度（米）。

    这是内镜/单目深度评估的标准协议（与 DepthMetrics 内部一致），
    也可用于推理时把相对深度转成近似米制。
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    valid = (gt > min_depth) & np.isfinite(gt) & np.isfinite(pred)
    if valid.sum() < min_valid_pixels:
        return pred.astype(np.float32)
    scale = np.median(gt[valid]) / max(np.median(pred[valid]), 1e-8)
    return (pred * scale).astype(np.float32)
