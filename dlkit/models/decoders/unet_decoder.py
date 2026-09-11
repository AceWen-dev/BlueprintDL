import torch
import torch.nn as nn
import torch.nn.functional as F

from dlkit.registry import DECODERS


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = ConvBnRelu(in_channels + skip_channels, out_channels, 3)
        self.conv2 = ConvBnRelu(out_channels, out_channels, 3)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        x = self.conv1(x)
        return self.conv2(x)


@DECODERS.register()
class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels=None):
        super().__init__()
        self.encoder_channels = list(encoder_channels)
        if decoder_channels is None:
            decoder_channels = list(reversed(self.encoder_channels))[:-1]
        self.decoder_channels = list(decoder_channels)

        stages = []
        prev = self.encoder_channels[-1]
        for skip_c, out_c in zip(self.encoder_channels[-2::-1], self.decoder_channels):
            stages.append(UpBlock(prev, skip_c, out_c))
            prev = out_c
        self.stages = nn.ModuleList(stages)
        self.out_channels = self.decoder_channels[-1] if self.decoder_channels else self.encoder_channels[-1]

    def forward(self, feats):
        x = feats[-1]
        skips = feats[:-1][::-1]
        for stage, skip in zip(self.stages, skips):
            x = stage(x, skip)
        return x
    
if __name__ == '__main__':
    # 调试入口：VS Code 里用 .vscode/launch.json 的配置按 F5，或终端
    #     PYTHONPATH=. python dlkit/models/decoders/unet_decoder.py
    # （不要用 python -m：父包 __init__ 会先 import 本模块，导致重复注册）
    decoder = UNetDecoder(encoder_channels=[16, 32, 64, 128])
    feats = [
        torch.randn(1, c, 64 // (2 ** i), 64 // (2 ** i))
        for i, c in enumerate([16, 32, 64, 128])
    ]
    out = decoder(feats)
    print('最终输出:', tuple(out.shape))   # 通道由 decoder_channels[-1] 决定，这里是 32
