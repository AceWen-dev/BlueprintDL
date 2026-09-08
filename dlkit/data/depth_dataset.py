import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from dlkit.registry import DATASETS
from dlkit.data.dataset import _build_transform


@DATASETS.register()
class DepthDataset(Dataset):
    """RGB 图像 + 深度图配对数据集（YOLO26-Depth 官方目录格式）。

    目录结构:
        images/xxx.png   <->   depth/xxx.png  (uint16 PNG，像素值=毫米)
        images/xxx.png   <->   depth/xxx.npy  (float32 NPY，像素值=米)

    深度图以米为单位存放在 sample['mask'] 中（沿用 Trainer 的约定），
    无效像素约定为 0，损失与指标会自动跳过。
    """

    def __init__(self, image_dir, depth_dir, image_suffix=None, depth_suffix=None,
                 depth_scale=None, transform=None, names=None):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.image_suffix = tuple(image_suffix) if image_suffix else ('.png', '.jpg', '.jpeg', '.bmp')
        self.depth_suffix = tuple(depth_suffix) if depth_suffix else ('.png', '.npy')
        self.depth_scale = depth_scale
        self.transform = _build_transform(transform)
        self.items = self._collect_pairs(self._load_names(names))

    def _load_names(self, names):
        if names is None:
            return None
        if isinstance(names, str):
            with open(names, 'r', encoding='utf-8') as f:
                names = [line.strip() for line in f if line.strip()]
        return {os.path.splitext(os.path.basename(n))[0] for n in names}

    def _collect_pairs(self, name_set):
        images = [
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(self.image_suffix)
        ]
        depths = {
            os.path.splitext(f)[0]: f
            for f in os.listdir(self.depth_dir)
            if f.lower().endswith(self.depth_suffix)
        }
        pairs = []
        for f in sorted(images):
            stem = os.path.splitext(f)[0]
            if name_set is not None and stem not in name_set:
                continue
            if stem not in depths:
                continue
            pairs.append(
                (os.path.join(self.image_dir, f), os.path.join(self.depth_dir, depths[stem]))
            )
        if len(pairs) < len(images):
            print(
                '[DepthDataset] paired %d/%d images with depth maps in %s'
                % (len(pairs), len(images), self.depth_dir)
            )
        return pairs

    def _load_depth(self, path):
        if path.lower().endswith('.npy'):
            depth = np.load(path).astype(np.float32)
            scale = self.depth_scale if self.depth_scale is not None else 1.0
            return depth * scale
        arr = np.asarray(Image.open(path))
        scale = self.depth_scale if self.depth_scale is not None else 0.001
        return arr.astype(np.float32) * scale

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_path, depth_path = self.items[idx]
        image = np.asarray(Image.open(image_path).convert('RGB'))
        depth = self._load_depth(depth_path)
        sample = {
            'image': image,
            'mask': depth,
            'name': os.path.splitext(os.path.basename(image_path))[0],
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
