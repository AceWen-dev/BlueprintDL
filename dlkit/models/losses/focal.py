import torch
import torch.nn as nn
import torch.nn.functional as F

from dlkit.registry import LOSSES


@LOSSES.register()
class FocalLoss(nn.Module):
    def __init__(self, weight=1.0, gamma=2.0, alpha=None, ignore_index=None):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        ce = -logp
        pt = (-ce).exp()
        focal = (1.0 - pt) ** self.gamma * ce

        if self.alpha is not None:
            if isinstance(self.alpha, (int, float)):
                alpha = self.alpha
            else:
                alpha = torch.as_tensor(self.alpha, device=logits.device)
            if isinstance(alpha, torch.Tensor):
                alpha = alpha.view(1, -1, 1, 1)
            focal = alpha * focal

        if self.ignore_index is not None:
            valid = (target != self.ignore_index).unsqueeze(1).float()
            if valid.sum() == 0:
                return torch.zeros((), device=logits.device)
            focal = focal * valid
            return self.weight * focal.sum() / valid.sum()
        return self.weight * focal.mean()
