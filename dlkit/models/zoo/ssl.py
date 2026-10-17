"""自监督(对比学习)模型:backbone + 投影头。

输入既支持 ``(B, 2, C, H, W)`` 的双视图 batch(TwoViewDataset 的输出),
也支持 ``(N, C, H, W)`` 的单视图(推理/线性探针)。输出投影向量 ``(N, D)``。
"""

import torch.nn as nn
import torch.nn.functional as F

from dlkit.builders import build_from_cfg
from dlkit.registry import MODELS
from dlkit.models.base import BaseModel
from dlkit.models.backbones.simple_cnn import SimpleCNN


@MODELS.register(tags=('self-supervised', 'contrastive'))
class SSLModel(BaseModel):
    """SimCLR 风格编码器:backbone 提特征 -> 全局池化 -> MLP 投影头。"""

    _manual_build = True

    def __init__(self, in_channels=3, backbone=None, projection_dim=128,
                 hidden_dim=256, dropout=0.0):
        super().__init__()
        self.backbone = _as(backbone, in_channels=in_channels) or SimpleCNN(
            in_channels=in_channels)
        feat_ch = self.backbone.out_channels[-1]

        layers = [nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                  nn.Linear(feat_ch, hidden_dim, bias=False),
                  nn.BatchNorm1d(hidden_dim),
                  nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, projection_dim))
        self.projector = nn.Sequential(*layers)

        if not getattr(self.backbone, 'pretrained', False):
            self.init_weights()

    def _flatten_views(self, x):
        if x.dim() == 5:  # (B, V, C, H, W)
            batch, views = x.shape[:2]
            return x.reshape(batch * views, *x.shape[2:])
        if x.dim() == 4:  # (N, C, H, W)
            return x
        raise ValueError(
            'SSLModel expects (B, V, C, H, W) or (N, C, H, W), got %s'
            % (tuple(x.shape),))

    def encode(self, x):
        """返回 backbone 的全局池化特征 (N, feat_ch),供线性探针/下游任务使用。"""

        feats = self.backbone(self._flatten_views(x))
        return F.adaptive_avg_pool2d(feats[-1], 1).flatten(1)

    def forward(self, x):
        feats = self.backbone(self._flatten_views(x))
        return self.projector(feats[-1])


def _as(obj, **defaults):
    if isinstance(obj, dict):
        cfg = dict(obj)
        params = dict(cfg.get('params', {}))
        for key, value in defaults.items():
            params.setdefault(key, value)
        cfg['params'] = params
        return build_from_cfg(cfg)
    return obj
