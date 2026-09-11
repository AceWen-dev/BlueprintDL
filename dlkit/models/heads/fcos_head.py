import math

import torch.nn as nn

from dlkit.registry import HEADS


def _tower(in_channels, mid_channels, num_convs):
    layers = []
    for i in range(num_convs):
        in_c = in_channels if i == 0 else mid_channels
        layers += [
            nn.Conv2d(in_c, mid_channels, 3, 1, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        ]
    return nn.Sequential(*layers)


@HEADS.register()
class FCOSHead(nn.Module):
    """FCOS 检测头（anchor-free，全卷积）。

    对 FPN 的每一层特征分别预测：
      - cls    分类分数 (C,)
      - reg    到 box 四边的距离 (l, t, r, b)，即 (4,)
      - center 中心度 (1,)，抑制偏离物体中心的低质量预测

    各层共享同一组卷积权重（FCOS 的 shared head）。
    """

    def __init__(self, in_channels=256, num_classes=3, num_convs=4):
        super().__init__()
        self.num_classes = num_classes
        self.cls_tower = _tower(in_channels, in_channels, num_convs)
        self.reg_tower = _tower(in_channels, in_channels, num_convs)
        self.center_tower = _tower(in_channels, in_channels, num_convs)

        self.cls_out = nn.Conv2d(in_channels, num_classes, 3, 1, 1)
        self.reg_out = nn.Conv2d(in_channels, 4, 3, 1, 1)
        self.center_out = nn.Conv2d(in_channels, 1, 3, 1, 1)

        self._init_bias()

    def _init_bias(self):
        # 分类输出偏置设为 -log((1-pi)/pi)，pi=0.01，缓解训练初期正样本不足
        pi = 0.01
        bias = -math.log((1.0 - pi) / pi)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.reg_out.weight)
        nn.init.zeros_(self.center_out.weight)
        nn.init.constant_(self.cls_out.bias, bias)

    def forward(self, feats):
        cls_out, reg_out, center_out = [], [], []
        for x in feats:
            cls_out.append(self.cls_out(self.cls_tower(x)))
            reg_out.append(self.reg_out(self.reg_tower(x)))
            center_out.append(self.center_out(self.center_tower(x)))
        return cls_out, reg_out, center_out
