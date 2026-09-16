import torch
import torch.nn as nn
import torch.nn.functional as F

from dlkit.builders import build_from_cfg
from dlkit.registry import MODELS
from dlkit.models.base import BaseModel
from dlkit.models.backbones.simple_cnn import SimpleCNN
from dlkit.models.decoders.aspp import ASPP
from dlkit.models.decoders.unet_decoder import ConvBnRelu
from dlkit.models.heads.seg_head import SegHead


@MODELS.register()
class DeepLabV3Plus(BaseModel):
    _manual_build = True

    def __init__(self, in_channels=3, num_classes=1, backbone=None, decoder=None,
                 head=None, low_level_index=-2, low_level_channels=48, fuse_channels=256):
        super().__init__()
        self.backbone = _as(backbone, in_channels=in_channels) or SimpleCNN(in_channels=in_channels)

        if decoder is None:
            decoder = ASPP(in_channels=self.backbone.out_channels[-1])
        else:
            decoder = _as(decoder, in_channels=self.backbone.out_channels[-1])
        self.decoder = decoder

        self.low_level_index = low_level_index
        low_in = self.backbone.out_channels[low_level_index]
        self.low_proj = ConvBnRelu(low_in, low_level_channels, 1)
        self.fuse = nn.Sequential(
            ConvBnRelu(decoder.out_channels + low_level_channels, fuse_channels, 3),
            ConvBnRelu(fuse_channels, fuse_channels, 3),
        )

        if head is None:
            head = SegHead(fuse_channels, num_classes)
        else:
            head = _as(head, in_channels=fuse_channels, num_classes=num_classes)
        self.head = head

        if not getattr(self.backbone, 'pretrained', False):
            self.init_weights()

    def forward(self, x):
        input_size = x.shape[2:]
        feats = self.backbone(x)
        low = feats[self.low_level_index]
        high = feats[-1]

        out = self.decoder(high)
        if out.shape[2:] != low.shape[2:]:
            out = F.interpolate(out, size=low.shape[2:], mode='bilinear', align_corners=False)
        out = torch.cat([self.low_proj(low), out], dim=1)
        out = self.fuse(out)
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
