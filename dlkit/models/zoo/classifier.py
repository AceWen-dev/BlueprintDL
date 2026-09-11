from dlkit.registry import MODELS, build_from_cfg
from dlkit.models.base import BaseModel
from dlkit.models.backbones.simple_cnn import SimpleCNN
from dlkit.models.heads.classification_head import GlobalPoolHead


@MODELS.register()
class Classifier(BaseModel):
    """分类网络（backbone + 全局池化分类头）。

    复用三段式思想：backbone 提取特征，head 取最后一层特征做全局池化后分类。
    """

    _manual_build = True

    def __init__(self, in_channels=3, num_classes=1000, backbone=None, head=None):
        super().__init__()
        self.backbone = _as(backbone, in_channels=in_channels) or SimpleCNN(in_channels=in_channels)
        feat_ch = self.backbone.out_channels[-1]

        if head is None:
            head = GlobalPoolHead(feat_ch, num_classes)
        else:
            head = _as(head, in_channels=feat_ch, num_classes=num_classes)
        self.head = head

        if not getattr(self.backbone, 'pretrained', False):
            self.init_weights()

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats[-1])


def _as(obj, **defaults):
    if isinstance(obj, dict):
        cfg = dict(obj)
        params = dict(cfg.get('params', {}))
        for k, v in defaults.items():
            params.setdefault(k, v)
        cfg['params'] = params
        return build_from_cfg(cfg)
    return obj
