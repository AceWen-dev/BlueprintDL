import torch
import torch.nn.functional as F

from dlkit.registry import MODELS, build_from_cfg
from dlkit.models.base import BaseModel
from dlkit.models.backbones.simple_cnn import SimpleCNN
from dlkit.models.decoders.unet_decoder import UNetDecoder
from dlkit.models.heads.log_depth_head import LogDepthHead


@MODELS.register()
class DepthNet(BaseModel):
    """YOLO26-Depth 风格的单目深度估计网络（三段式）。

    backbone 提取多尺度特征 -> decoder 恢复分辨率 -> LogDepthHead 预测 log(深度)。
    forward 输出 log-深度 (B,1,H,W)；真实米数 = exp(logits)，
    绝对尺度可通过 log-affine 校准（exp(a·log d + b)）获得。
    """

    _manual_build = True

    def __init__(self, in_channels=3, backbone=None, decoder=None, head=None):
        super().__init__()
        self.backbone = _as(backbone, in_channels=in_channels) or SimpleCNN(in_channels=in_channels)
        encoder_channels = list(self.backbone.out_channels)

        if decoder is None:
            decoder = UNetDecoder(encoder_channels=encoder_channels)
        else:
            decoder = _as(decoder, encoder_channels=encoder_channels)
        self.decoder = decoder

        if head is None:
            head = LogDepthHead(self.decoder.out_channels)
        else:
            head = _as(head, in_channels=self.decoder.out_channels)
        self.head = head

        if not getattr(self.backbone, 'pretrained', False):
            self.init_weights()

    def forward(self, x):
        input_size = x.shape[2:]
        feats = self.backbone(x)
        out = self.decoder(feats)
        logit = self.head(out)
        if logit.shape[2:] != input_size:
            logit = F.interpolate(logit, size=input_size, mode='bilinear', align_corners=False)
        return logit

    def predict_depth(self, x):
        """推理接口：返回以米为单位的深度图 (B,1,H,W)。"""
        with torch.no_grad():
            return torch.exp(self.forward(x))


def _as(obj, **defaults):
    if isinstance(obj, dict):
        cfg = dict(obj)
        params = dict(cfg.get('params', {}))
        for k, v in defaults.items():
            params.setdefault(k, v)
        cfg['params'] = params
        return build_from_cfg(cfg)
    return obj
