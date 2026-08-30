import torch.nn.functional as F

from dlkit.registry import MODELS, build_from_cfg
from dlkit.models.base import BaseModel
from dlkit.models.backbones.simple_cnn import SimpleCNN
from dlkit.models.decoders.unet_decoder import UNetDecoder
from dlkit.models.heads.seg_head import SegHead


@MODELS.register()
class UNet(BaseModel):
    _manual_build = True

    def __init__(self, in_channels=3, num_classes=1, backbone=None, decoder=None, head=None):
        super().__init__()
        self.backbone = _as(backbone, in_channels=in_channels) or SimpleCNN(in_channels=in_channels)
        encoder_channels = list(self.backbone.out_channels)

        if decoder is None:
            decoder = UNetDecoder(encoder_channels=encoder_channels)
        else:
            decoder = _as(decoder, encoder_channels=encoder_channels)
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
        out = self.decoder(feats)
        logits = self.head(out)
        if logits.shape[2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        return logits


def _as(obj, **defaults):
    if isinstance(obj, dict):
        cfg = dict(obj)
        params = dict(cfg.get('params', {}))
        for k, v in defaults.items():
            params.setdefault(k, v)
        cfg['params'] = params
        return build_from_cfg(cfg)
    return obj
