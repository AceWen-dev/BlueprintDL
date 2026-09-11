import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from dlkit.registry import DATASETS
from dlkit.data.dataset import _build_transform


@DATASETS.register()
class ClassificationDataset(Dataset):
    """ImageFolder 风格分类数据集。

    目录结构（每类一个子目录，子目录名即类别名）:
        image_dir/class_a/xxx.jpg
        image_dir/class_b/yyy.jpg

    class_names 按子目录名排序自动生成；也可通过 ``names`` 显式指定顺序。
    sample 结构: {'image': ndarray(H,W,3), 'label': int, 'name': str}
    """

    def __init__(self, image_dir, transform=None, image_suffix=None, names=None):
        self.image_dir = image_dir
        self.image_suffix = tuple(image_suffix) if image_suffix else ('.png', '.jpg', '.jpeg', '.bmp')
        self.transform = _build_transform(transform)

        subdirs = [
            d for d in sorted(os.listdir(image_dir))
            if os.path.isdir(os.path.join(image_dir, d))
        ]
        if names is not None:
            missing = [n for n in names if n not in subdirs]
            if missing:
                raise ValueError('names %s not found under %s' % (missing, image_dir))
            self.classes = list(names)
        else:
            self.classes = subdirs
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.items = []
        for cls in self.classes:
            cls_dir = os.path.join(image_dir, cls)
            for f in sorted(os.listdir(cls_dir)):
                if f.lower().endswith(self.image_suffix):
                    self.items.append((os.path.join(cls_dir, f), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_path, label = self.items[idx]
        image = np.asarray(Image.open(image_path).convert('RGB'))
        sample = {
            'image': image,
            'label': label,
            'name': os.path.splitext(os.path.basename(image_path))[0],
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
