import numpy as np

from dlkit.registry import METRICS


@METRICS.register()
class DepthMetrics:
    """单目深度指标（YOLO26-Depth 官方同款协议）。

    delta1/2/3（阈值 1.25 的比值比例，越高越好）、abs_rel（越低越好）、
    rmse（米，越低越好）、silog（尺度不变对数误差，越低越好）。

    每个样本先做中位数对齐（median scaling），再算指标，最后按样本平均，
    与 Depth Anything / YOLO26 的评估方式一致。
    """

    def __init__(self, max_depth=None, min_depth=1e-3, min_valid_pixels=10, log_space=True):
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.min_valid_pixels = min_valid_pixels
        self.log_space = log_space
        self.reset()

    def reset(self):
        self._delta1 = []
        self._delta2 = []
        self._delta3 = []
        self._abs_rel = []
        self._rmse = []
        self._silog = []

    def update(self, logits, batch):
        if self.log_space:
            pred = np.exp(logits.detach().cpu().numpy())
        else:
            pred = logits.detach().cpu().numpy()
        gt = batch['mask'].detach().cpu().numpy()
        pred = pred[:, 0] if pred.ndim == 4 else pred
        gt = gt[:, 0] if gt.ndim == 4 else gt

        for p, g in zip(pred, gt):
            p = p.astype(np.float64)
            g = g.astype(np.float64)
            valid = (g > self.min_depth) & np.isfinite(g) & np.isfinite(p)
            if self.max_depth is not None:
                valid &= g < self.max_depth
            if valid.sum() < self.min_valid_pixels:
                continue

            scale = np.median(g[valid]) / max(np.median(p[valid]), 1e-8)
            p_aligned = p * scale

            ratio = np.maximum(p_aligned[valid] / g[valid], g[valid] / p_aligned[valid])
            self._delta1.append(float((ratio < 1.25).mean()))
            self._delta2.append(float((ratio < 1.25 ** 2).mean()))
            self._delta3.append(float((ratio < 1.25 ** 3).mean()))
            self._abs_rel.append(float((np.abs(p_aligned[valid] - g[valid]) / g[valid]).mean()))
            self._rmse.append(float(np.sqrt(((p_aligned[valid] - g[valid]) ** 2).mean())))

            d = np.log(p_aligned[valid]) - np.log(g[valid])
            self._silog.append(float(np.sqrt(max((d ** 2).mean() - (d.mean() ** 2), 0.0))))

    def compute(self):
        def mean(x):
            return float(np.mean(x)) if x else 0.0

        return {
            'delta1': mean(self._delta1),
            'delta2': mean(self._delta2),
            'delta3': mean(self._delta3),
            'abs_rel': mean(self._abs_rel),
            'rmse': mean(self._rmse),
            'silog': mean(self._silog),
        }
