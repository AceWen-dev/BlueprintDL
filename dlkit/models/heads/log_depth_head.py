import torch.nn as nn

from dlkit.registry import HEADS


@HEADS.register()
class LogDepthHead(nn.Module):
    """YOLO26-Depth 风格的 log-深度头。

    预测 1 通道的 log(深度)，无界输出（不像 sigmoid×max_depth 那样有上限）。
    真实米数 = exp(logit)，绝对尺度由外部校准（log-affine: exp(a·log d + b)）。
    """

    def __init__(self, in_channels, mid_channels=64, out_channels=1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 1),
        )
        self.out_channels = out_channels

    def forward(self, x):
        return self.head(x)
