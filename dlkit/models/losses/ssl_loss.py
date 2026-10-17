"""自监督对比损失:NT-Xent / InfoNCE(SimCLR)。

预测是 ``(2B, D)`` 的投影向量:前 B 个是一个视图,后 B 个是另一个视图,
同一张图的两个视图互为唯一正样本。公式::

    L = CE( (z_i · z_j / tau) , positive_index )

与 Trainer 的通用契约兼容:``forward(predictions, batch)``,``batch`` 不参与计算。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dlkit.registry import LOSSES


@LOSSES.register(
    tags=('self-supervised', 'contrastive'),
    description='NT-Xent / InfoNCE:双视图对比学习损失',
)
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, eps=1e-8):
        super().__init__()
        if float(temperature) <= 0:
            raise ValueError('temperature must be > 0')
        self.temperature = float(temperature)
        self.eps = float(eps)

    def forward(self, predictions, batch=None):  # noqa: ARG002  保持损失统一签名
        if isinstance(predictions, dict):
            predictions = predictions.get('embeddings')
        if predictions is None or predictions.dim() != 2:
            raise ValueError(
                'NTXentLoss expects (2B, D) embeddings, got %r'
                % (None if predictions is None else tuple(predictions.shape),))
        count = predictions.size(0)
        if count % 2 != 0:
            raise ValueError(
                'NTXentLoss expects an even batch (two views), got %d' % count)

        z = F.normalize(predictions, dim=1, eps=self.eps)
        logits = z @ z.t() / self.temperature
        mask = torch.eye(count, dtype=torch.bool, device=z.device)
        logits = logits.masked_fill(mask, float('-inf'))
        labels = (torch.arange(count, device=z.device) + count // 2) % count
        return F.cross_entropy(logits, labels)


__all__ = ['NTXentLoss']
