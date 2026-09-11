import torch.nn as nn
import torch.nn.functional as F

from dlkit.registry import DECODERS


@DECODERS.register()
class FPN(nn.Module):
    """特征金字塔网络（Feature Pyramid Network），检测模型的标准 neck。

    取 backbone 最后三层特征（C3/C4/C5），自顶向下融合出 P3/P4/P5，
    再下采样生成 P6/P7，共 5 层输出（FCOS 使用的多尺度）。
    """

    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        in_channels = list(in_channels)
        if len(in_channels) < 3:
            raise ValueError('FPN needs at least 3 backbone levels, got %d' % len(in_channels))
        self.out_channels = out_channels
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels[-3:]])
        self.fpn_conv = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, 1, 1) for _ in in_channels[-3:]])
        self.p6 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats):
        feats = feats[-3:]  # C3, C4, C5
        laterals = [conv(f) for conv, f in zip(self.lateral, feats)]

        p5 = self.fpn_conv[2](laterals[2])
        p4 = self.fpn_conv[1](
            laterals[1] + F.interpolate(p5, size=laterals[1].shape[2:], mode='nearest')
        )
        p3 = self.fpn_conv[0](
            laterals[0] + F.interpolate(p4, size=laterals[0].shape[2:], mode='nearest')
        )
        p6 = self.p6(laterals[2])
        p7 = self.relu(self.p7(p6))
        return [p3, p4, p5, p6, p7]
