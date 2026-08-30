import torch.nn.functional as F

from dlkit.registry import MODELS, build_from_cfg
from dlkit.models.base import BaseModel
from dlkit.models.backbones.simple_cnn import SimpleCNN
from dlkit.models.decoders.aspp import ASPP
from dlkit.models.heads.seg_head import SegHead


@MODELS.register()
class DeepLabV3(BaseModel):
    _manual_build = True

    def __init__(self, in_channels=3, num_classes=1, backbone=None, decoder=None, head=None):
        super().__init__()
        self.backbone = _as(backbone, in_channels=in_channels) or SimpleCNN(in_channels=in_channels)

        if decoder is None:
            decoder = ASPP(in_channels=self.backbone.out_channels[-1])
        else:
            decoder = _as(decoder, in_channels=self.backbone.out_channels[-1])
        self.decoder = decoder

        if head is None:
            head = SegHead(self.decoder.out_channels, num_classes)
        else:
            head = _as(head, in_channels=self.decoder.out_channels, num_classes=num_classes)
        self.head = head

        if not getattr(self.backbone, 'pretrained', False):
            self.init_weights()

    def forward(self, x):
        input_size = x.shape[2:]
        feats = self.backbone(x)
        out = self.decoder(feats[-1])
        logits = self.head(out)
        if logits.shape[2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        return logits


def _as(obj, **defaults): #这个函数用来补充参数
    if isinstance(obj, dict):
        cfg = dict(obj) #复制obj这个字典
        params = dict(cfg.get('params', {}))
        for k, v in defaults.items():
            params.setdefault(k, v)#如果对应的k存在v值就不动，如果不存在就设置成v
        cfg['params'] = params
        return build_from_cfg(cfg)
    return obj #不是字典直接返回
