"""自监督双视图数据集(SimCLR 风格)。

每张图片独立应用两次 ``view_transform``(强增强),把两个视图堆叠成
``(2, C, H, W)``,默认 collate 后 Trainer 看到的是 ``(B, 2, C, H, W)``。
配套模型 ``SSLModel`` 会把 batch 摊平成 2B 张图,损失 ``NTXentLoss``
把同一张图的两个视图当作正样本对。
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dlkit.registry import DATASETS
from dlkit.data.dataset import _build_transform


@DATASETS.register(tags=('self-supervised', 'contrastive'))
class TwoViewDataset(Dataset):
    """无标签图片目录 -> 双增强视图。

    参数:
    - ``image_dir``:图片目录(不需要标签);
    - ``view_transform``:单视图变换配置,必须以 ToTensor(+Normalize) 结尾;
    - ``image_suffix``:图片后缀(默认常见图片格式)。
    """

    def __init__(self, image_dir, view_transform=None, image_suffix=None):
        self.image_dir = image_dir
        self.image_suffix = (
            tuple(image_suffix) if image_suffix else ('.png', '.jpg', '.jpeg', '.bmp')
        )
        self.transform = _build_transform(view_transform)
        if not os.path.isdir(self.image_dir):
            raise ValueError('directory not found: %s' % self.image_dir)
        self.items = sorted(
            os.path.join(self.image_dir, name)
            for name in os.listdir(self.image_dir)
            if name.lower().endswith(self.image_suffix)
        )
        if not self.items:
            raise ValueError('no images found in %s' % self.image_dir)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path = self.items[index]
        image = np.asarray(Image.open(path).convert('RGB'))
        name = os.path.splitext(os.path.basename(path))[0]

        views = []
        for _ in range(2):
            sample = self.transform({'image': image.copy(), 'name': name})
            view = sample['image']
            if not isinstance(view, torch.Tensor):
                raise TypeError(
                    'TwoViewDataset view_transform must end with ToTensor; got %s'
                    % type(view).__name__
                )
            views.append(view)

        return {'image': torch.stack(views, dim=0), 'name': name}
