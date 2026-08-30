import torch
import torch.nn as nn

from dlkit.registry import LOSSES


@LOSSES.register()
class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=1.0, ignore_index=None, class_weight=None, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        kwargs = {}
        if ignore_index is not None:
            kwargs['ignore_index'] = ignore_index
        if class_weight is not None:
            kwargs['weight'] = torch.as_tensor(class_weight, dtype=torch.float32)
        try:
            self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, **kwargs)
        except TypeError:
            self.ce = nn.CrossEntropyLoss(**kwargs)

    def forward(self, logits, target):
        return self.weight * self.ce(logits, target)
