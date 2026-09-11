import torch.nn as nn

from dlkit.registry import HEADS


@HEADS.register()
class GlobalPoolHead(nn.Module):
    """分类头：全局池化 -> 展平 -> 全连接。

    输入 backbone 最后一层特征 (B, C, H, W)，输出分类 logits (B, num_classes)。
    """

    def __init__(self, in_channels, num_classes, pool='avg', dropout=0.0):
        super().__init__()
        if pool == 'avg':
            pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool == 'max':
            pool_layer = nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError('pool must be "avg" or "max", got %r' % (pool,))

        layers = [pool_layer, nn.Flatten()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_channels, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)
