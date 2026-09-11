import torch.nn as nn

from dlkit.registry import BACKBONES

try:
    import torchvision.models as tvm
    HAS_TORCHVISION = True
except ImportError:
    tvm = None
    HAS_TORCHVISION = False


_CHANNEL_MAP = {
    'resnet18': [64, 128, 256, 512],
    'resnet34': [64, 128, 256, 512],
    'resnet50': [256, 512, 1024, 2048],
    'resnet101': [256, 512, 1024, 2048],
}


@BACKBONES.register()
class TorchvisionBackbone(nn.Module):
    def __init__(self, name='resnet18', pretrained=True, output_stride=32, in_channels=3):
        super().__init__()
        if not HAS_TORCHVISION:
            raise RuntimeError('torchvision is required for TorchvisionBackbone')
        if name not in _CHANNEL_MAP:
            raise ValueError('TorchvisionBackbone supports %s' % sorted(_CHANNEL_MAP))
        self.name = name
        self.pretrained = pretrained
        weights = 'DEFAULT' if pretrained else None
        model = getattr(tvm, name)(weights=weights)

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        if output_stride == 8:
            self._dilate(self.layer3, 2)
            self._dilate(self.layer4, 4)
        elif output_stride == 16:
            self._dilate(self.layer4, 2)
        elif output_stride != 32:
            raise ValueError('output_stride must be 8, 16 or 32')

        self.out_channels = _CHANNEL_MAP[name]
        self.strides = [4, 8, 16, 32]

    def _dilate(self, stage, dilation):
        for m in stage.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):
                    m.dilation = (dilation, dilation)
                    m.padding = (dilation, dilation)
                if m.stride == (2, 2):
                    m.stride = (1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        return [l1, l2, l3, l4]
