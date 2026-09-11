from dlkit.registry import MODELS, build_from_cfg
from dlkit.models.base import BaseModel
from dlkit.models.backbones.simple_cnn import SimpleCNN
from dlkit.models.decoders.fpn import FPN
from dlkit.models.heads.fcos_head import FCOSHead


@MODELS.register()
class FCOS(BaseModel):
    """FCOS：anchor-free 单阶段目标检测器（backbone + FPN neck + FCOS head）。

    forward 返回 dict:
        cls / reg / center: 每层一个张量的列表
        strides: 每层相对输入的下采样倍率（供损失与解码使用）
    """

    _manual_build = True

    def __init__(self, in_channels=3, num_classes=3, backbone=None, neck=None, head=None,
                 strides=None):
        super().__init__()
        self.backbone = _as(backbone, in_channels=in_channels) or SimpleCNN(in_channels=in_channels)
        neck_in = self.backbone.out_channels[-3:]

        if neck is None:
            neck = FPN(in_channels=neck_in)
        else:
            neck = _as(neck, in_channels=neck_in)
        self.neck = neck

        if head is None:
            head = FCOSHead(in_channels=neck.out_channels, num_classes=num_classes)
        else:
            head = _as(head, in_channels=neck.out_channels, num_classes=num_classes)
        self.head = head

        self.num_classes = num_classes

        if strides is None:
            s = list(getattr(self.backbone, 'strides', [4, 8, 16, 32]))[-3:]
            strides = [s[0], s[1], s[2], s[2] * 2, s[2] * 4]
        self.strides = list(strides)

        if not getattr(self.backbone, 'pretrained', False):
            self.init_weights()
            # init_weights 会把 Conv2d 的 bias 清零，需重新恢复分类头的偏置先验
            if hasattr(self.head, '_init_bias'):
                self.head._init_bias()

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        cls, reg, center = self.head(feats)
        return {'cls': cls, 'reg': reg, 'center': center, 'strides': self.strides}


def _as(obj, **defaults):
    if isinstance(obj, dict):
        cfg = dict(obj)
        params = dict(cfg.get('params', {}))
        for k, v in defaults.items():
            params.setdefault(k, v)
        cfg['params'] = params
        return build_from_cfg(cfg)
    return obj
