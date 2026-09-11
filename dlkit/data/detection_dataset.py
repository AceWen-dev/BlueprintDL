import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dlkit.registry import DATASETS, TRANSFORMS
from dlkit.data.dataset import _build_transform


@TRANSFORMS.register()
class DetResize:
    """检测版 Resize：缩放图像的同时按比例缩放边界框。"""

    def __init__(self, size):
        if isinstance(size, (int, float)):
            self.size = (int(size), int(size))
        else:
            self.size = (int(size[0]), int(size[1]))

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]
        th, tw = self.size
        sx, sy = tw / w, th / h
        image = Image.fromarray(image).resize((tw, th), Image.BILINEAR)
        sample['image'] = np.asarray(image)

        boxes = sample['boxes']
        if boxes is not None and len(boxes):
            boxes = boxes.copy()
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
            sample['boxes'] = boxes
        return sample


@TRANSFORMS.register()
class DetRandomHorizontalFlip:
    """检测版水平翻转：图像与边界框同步镜像。"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = np.ascontiguousarray(sample['image'][:, ::-1])
            boxes = sample['boxes']
            if boxes is not None and len(boxes):
                w = sample['image'].shape[1]
                boxes = boxes.copy()
                x1 = boxes[:, 0].copy()
                x2 = boxes[:, 2].copy()
                boxes[:, 0] = w - x2
                boxes[:, 2] = w - x1
                sample['boxes'] = boxes
        return sample


@TRANSFORMS.register()
class DetToTensor:
    """检测版 ToTensor：图像 -> (C,H,W)/255，boxes/labels -> 张量。"""

    def __call__(self, sample):
        image = sample['image']
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255.0
        sample['image'] = image
        sample['boxes'] = torch.as_tensor(sample['boxes'], dtype=torch.float32)
        sample['labels'] = torch.as_tensor(sample['labels'], dtype=torch.int64)
        return sample


@DATASETS.register()
class DetectionDataset(Dataset):
    """目标检测数据集：图像 + 同名 txt 标注。

    目录结构:
        image_dir/xxx.png
        label_dir/xxx.txt   （每行: cls x1 y1 x2 y2，绝对像素坐标）

    sample 结构: {'image': (H,W,3), 'boxes': (N,4), 'labels': (N,), 'name': str}
    由于每张图的物体数量不同，提供自定义 collate_fn 把 boxes/labels 打包成
    list-of-tensors（Trainer/loss/metric 逐图处理）。
    """

    def __init__(self, image_dir, label_dir, transform=None, image_suffix=None, label_suffix='.txt'):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.label_suffix = label_suffix
        self.image_suffix = tuple(image_suffix) if image_suffix else ('.png', '.jpg', '.jpeg', '.bmp')
        self.transform = _build_transform(transform)
        self.items = self._collect_pairs()

    def _collect_pairs(self):
        pairs = []
        for f in sorted(os.listdir(self.image_dir)):
            if not f.lower().endswith(self.image_suffix):
                continue
            stem = os.path.splitext(f)[0]
            label_path = os.path.join(self.label_dir, stem + self.label_suffix)
            if os.path.exists(label_path):
                pairs.append((os.path.join(self.image_dir, f), label_path))
        if len(pairs) < len([f for f in os.listdir(self.image_dir) if f.lower().endswith(self.image_suffix)]):
            print('[DetectionDataset] paired %d images with labels in %s' % (len(pairs), self.label_dir))
        return pairs

    @staticmethod
    def _load_labels(path):
        boxes, labels = [], []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                labels.append(int(parts[0]))
                boxes.append([float(v) for v in parts[1:5]])
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(labels, dtype=np.int64)
        return boxes, labels

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_path, label_path = self.items[idx]
        image = np.asarray(Image.open(image_path).convert('RGB'))
        boxes, labels = self._load_labels(label_path)
        sample = {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'name': os.path.splitext(os.path.basename(image_path))[0],
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    @staticmethod
    def collate_fn(batch):
        images = []
        for s in batch:
            img = s['image']
            if not isinstance(img, torch.Tensor):
                img = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1) / 255.0
            images.append(img)
        images = torch.stack(images, dim=0)
        boxes = [
            s['boxes'] if isinstance(s['boxes'], torch.Tensor)
            else torch.as_tensor(s['boxes'], dtype=torch.float32)
            for s in batch
        ]
        labels = [
            s['labels'] if isinstance(s['labels'], torch.Tensor)
            else torch.as_tensor(s['labels'], dtype=torch.int64)
            for s in batch
        ]
        names = [s['name'] for s in batch]
        return {'image': images, 'boxes': boxes, 'labels': labels, 'name': names}
