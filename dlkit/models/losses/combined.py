import torch.nn as nn

from dlkit.registry import LOSSES, build_from_cfg


@LOSSES.register()
class CombinedLoss(nn.Module):
    def __init__(self, losses=None):
        super().__init__()
        built = []
        for loss in losses or []:
            built.append(build_from_cfg(loss) if isinstance(loss, dict) else loss)
        self.losses = nn.ModuleList(built)

    def forward(self, preds, batch):
        total = None
        for loss in self.losses:
            value = loss(preds, batch)
            total = value if total is None else total + value
        if total is None:
            raise RuntimeError('CombinedLoss has no sub-losses')
        return total
