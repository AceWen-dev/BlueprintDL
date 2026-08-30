import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch.nn as nn

from dlkit.registry import BACKBONES


class BasicBlock(nn.Module):
    expansion = 1     #通道膨胀的倍率，这个expansion定义在init之外是类属性不是实例属性，通过BasicBlock.expansion访问，self.expansion是实例属性

    def __init__(self, in_planes, planes, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, dilation, dilation, bias=False)# 尺寸不变
        self.bn1 = nn.BatchNorm2d(planes) #传入参数为特征通道数
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, dilation, dilation, bias=False)#尺寸不变
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out + identity)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, dilation, dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity() #如果x和需要残差连接的x一模一样就什么都不用做
        if stride != 1 or in_planes != planes * self.expansion:  #如果x和需要残差连接的x不一样比如通道数变了或者hw减半则通过下面函数进行对其后相加
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out + identity)
        return out


def _dilate_stage(stage, dilation):#把一个阶段的普通卷积改成空洞卷积同时取消下采样，因为普通卷积特征图太小小物体就丢失了
    for m in stage.modules():
        if isinstance(m, nn.Conv2d):
            if m.kernel_size == (3, 3):
                m.dilation = (dilation, dilation)
                m.padding = (dilation, dilation)
            if m.stride == (2, 2):
                m.stride = (1, 1)


@BACKBONES.register()
class ResNet(nn.Module):
    #用什么积木不同类型的resent，各个stage堆叠对少个blocks(basicblock或者bottleneck)
    ARCH = {
        18: (BasicBlock, [2, 2, 2, 2]),
        34: (BasicBlock, [3, 4, 6, 3]),
        50: (Bottleneck, [3, 4, 6, 3]),
        101: (Bottleneck, [3, 4, 23, 3]),
    }

    def __init__(self, depth=18, in_channels=3, replace_stride=False, pretrained=False):
        super().__init__()
        if depth not in self.ARCH:
            raise ValueError('ResNet depth must be one of %s' % sorted(self.ARCH))
        block, layers = self.ARCH[depth]
        self.depth = depth
        self.pretrained = pretrained

        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        base_widths = [64, 128, 256, 512]#基础通道数
        #由于历史遗留问题，这里叫self.stage1其实是stage2，self.stage2是stage3，self.stage3是stage4，self.stage4是stage5
        self.stage1 = self._make_stage(block, 64, base_widths[0], layers[0], stride=1)
        self.stage2 = self._make_stage(block, base_widths[0] * block.expansion, base_widths[1], layers[1], stride=2)
        self.stage3 = self._make_stage(block, base_widths[1] * block.expansion, base_widths[2], layers[2], stride=2)
        self.stage4 = self._make_stage(block, base_widths[2] * block.expansion, base_widths[3], layers[3], stride=2)

        if replace_stride: 
            _dilate_stage(self.stage4, 2)

        self.out_channels = [w * block.expansion for w in base_widths]   #四个stage输出的通道数

        if pretrained: #加载与训练权重
            self._load_torchvision_state()
    #构建stage的函数
    def _make_stage(self, block, in_planes, planes, blocks, stride):
        layers = [block(in_planes, planes, stride)]
        for _ in range(1, blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _load_torchvision_state(self):
        try:
            import torchvision.models as tvm
        except ImportError:
            raise RuntimeError('torchvision is required for pretrained weights')
        factory = getattr(tvm, 'resnet%d' % self.depth, None)
        if factory is None:
            raise RuntimeError('no torchvision model resnet%d' % self.depth)
        tv = factory(weights='DEFAULT')
        tv_state = tv.state_dict()
        my_keys = set(self.state_dict().keys())
        filtered = {k: v for k, v in tv_state.items() if k in my_keys}
        self.load_state_dict(filtered, strict=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        l1 = self.stage1(x)
        l2 = self.stage2(l1)
        l3 = self.stage3(l2)
        l4 = self.stage4(l3)
        return [l1, l2, l3, l4] #返回的是四个不同stage的特征图张量列表


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='用 torchinfo 打印 ResNet 结构')
    parser.add_argument('--depth', type=int, default=18, choices=[18, 34, 50, 101], help='ResNet 深度')
    parser.add_argument('--input-size', type=int, nargs='+', default=[1, 3, 256, 256], help='输入形状 (B C H W)')
    parser.add_argument('--max-depth', type=int, default=3, help='torchinfo 显示的最大嵌套深度')
    args = parser.parse_args()

    try:
        from torchinfo import summary
    except ImportError:
        raise SystemExit('未安装 torchinfo，请先执行: pip install torchinfo')

    model = ResNet(depth=args.depth)
    summary(
        model,
        input_size=tuple(args.input_size),
        depth=args.max_depth,
        col_names=('input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'),
        verbose=1,
    )
    
