"""任务无关的 batch 工具。

Trainer 不再假设 batch 的具体结构（分割的 mask / 分类的 label / 检测的
boxes+labels 各不相同），而是用 `to_device` 递归地把 batch 里所有的张量
搬到目标设备，把「如何取标签」这件事完全交给 loss 和 metric。
"""

import torch


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj
