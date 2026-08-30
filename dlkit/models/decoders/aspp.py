import torch
import torch.nn as nn
import torch.nn.functional as F

from dlkit.registry import DECODERS
from dlkit.models.decoders.unet_decoder import ConvBnRelu


@DECODERS.register()
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, atrous_rates=(6, 12, 18)):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(ConvBnRelu(in_channels, out_channels, 1))
        for rate in atrous_rates:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 1, rate, rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(self.convs) + 1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_channels

    def forward(self, x):
        outs = [conv(x) for conv in self.convs]
        pool = self.image_pool(x)
        pool = F.interpolate(pool, size=x.shape[2:], mode='bilinear', align_corners=False)
        outs.append(pool)
        return self.project(torch.cat(outs, dim=1))
