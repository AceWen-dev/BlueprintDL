import torch.nn as nn


def init_weights(module):
    if isinstance(module, nn.Conv2d):#如果是卷积层就使用kaiming初始化
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):#对于批归一化层
        nn.init.ones_(module.weight) #权重初始化为1
        nn.init.zeros_(module.bias) #偏执初始化为0  


class BaseModel(nn.Module):
    def init_weights(self):
        self.apply(init_weights)
        return self

    def num_parameters(self, trainable_only=True):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
