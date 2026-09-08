import torch
import torch.nn as nn

from dlkit.registry import LOSSES


@LOSSES.register()
class DepthLoss(nn.Module):
    """单目深度损失：SILog（尺度不变对数损失）+ 梯度一致性损失。

    logits 是模型预测的 log(深度)，target 是以米为单位的深度图。
    SILog 在 log 空间计算，天然对全局尺度不敏感，与 log-深度头配套。
    """

    def __init__(self, silog_weight=0.5, grad_weight=0.5, variance_focus=0.85,
                 min_depth=1e-3, weight=1.0):
        super().__init__()
        self.silog_weight = silog_weight
        self.grad_weight = grad_weight
        self.variance_focus = variance_focus
        self.min_depth = min_depth
        self.weight = weight

    def forward(self, logits, target):
        if logits.dim() == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        log_target = torch.log(target.clamp(min=self.min_depth))
        d = logits - log_target
        valid = target > 0

        silog = self._silog(d, valid)
        grad = self._grad_loss(d, valid)
        return self.weight * (self.silog_weight * silog + self.grad_weight * grad)

    def _silog(self, d, valid):
        d = d[valid]
        if d.numel() == 0:
            return d.sum() * 0.0
        mean_d = d.mean()
        var_term = (d * d).mean() - self.variance_focus * (mean_d * mean_d)
        return torch.sqrt(torch.clamp(var_term, min=0.0))

    def _grad_loss(self, d, valid):
        dx = d[:, 1:, :] - d[:, :-1, :]
        dy = d[:, :, 1:] - d[:, :, :-1]
        wx = valid[:, 1:, :] & valid[:, :-1, :]
        wy = valid[:, :, 1:] & valid[:, :, :-1]
        if dx[wx].numel() == 0:
            return d.sum() * 0.0
        return dx[wx].abs().mean() + dy[wy].abs().mean()
