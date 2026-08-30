import torch.nn as nn

from dlkit.registry import HEADS


@HEADS.register()
class SegHead(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.0):
        super().__init__()
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.Conv2d(in_channels, num_classes, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)
