import torch
import torch.nn as nn
import torch.nn.functional as F

from dlkit.registry import LOSSES


@LOSSES.register()
class DiceLoss(nn.Module):
    def __init__(self, weight=1.0, ignore_index=None, smooth=1.0, apply_softmax=True,
                 target_key='mask'):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.apply_softmax = apply_softmax
        self.target_key = target_key

    def forward(self, logits, batch):
        target = batch[self.target_key]
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1) if self.apply_softmax else logits

        t = target.clone()
        valid = None
        if self.ignore_index is not None:
            valid = (t != self.ignore_index)
            t = t.clamp(min=0)
        onehot = F.one_hot(t, num_classes).permute(0, 3, 1, 2).float()

        if valid is not None:
            valid = valid.unsqueeze(1).float()
            if valid.sum() == 0:
                return torch.zeros((), device=logits.device)
            onehot = onehot * valid
            probs = probs * valid

        inter = (probs * onehot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return self.weight * (1.0 - dice.mean())
