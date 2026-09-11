import torch.nn as nn

from dlkit.registry import BACKBONES


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


@BACKBONES.register()
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, channels=(16, 32, 64, 128), num_blocks=(2, 2, 2, 2)):
        super().__init__()
        stages = []
        prev = in_channels
        for out_c, n in zip(channels, num_blocks):
            blocks = []
            for i in range(n):
                stride = 2 if i == 0 else 1
                blocks.append(ConvBlock(prev, out_c, 3, stride))
                prev = out_c
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.out_channels = list(channels)
        # 每个 stage 首块 stride=2，故第 i 层相对输入的下采样倍率为 2^(i+1)
        self.strides = [2 ** (i + 1) for i in range(len(channels))]

    def forward(self, x):
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats
