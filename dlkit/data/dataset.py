import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from dlkit.builders import build_from_cfg
from dlkit.registry import DATASETS, TRANSFORMS


def _build_transform(transform):
    if transform is None:
        return TRANSFORMS.get('Compose')([])
    if isinstance(transform, dict):
        return build_from_cfg(transform)
    if isinstance(transform, (list, tuple)):
        return TRANSFORMS.get('Compose')(transform)
    return transform


@DATASETS.register()
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_suffix=None, mask_suffix=None,
                 transform=None, names=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_suffix = tuple(image_suffix) if image_suffix else ('.png', '.jpg', '.jpeg', '.bmp')
        self.mask_suffix = tuple(mask_suffix) if mask_suffix else ('.png',)
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
        masks = {
            os.path.splitext(f)[0]: f
            for f in os.listdir(self.mask_dir)
            if f.lower().endswith(self.mask_suffix)
        }
        pairs = []
        for f in sorted(images):
            stem = os.path.splitext(f)[0]
            if name_set is not None and stem not in name_set:
                continue
            if stem not in masks:
                continue
            pairs.append(
                (os.path.join(self.image_dir, f), os.path.join(self.mask_dir, masks[stem]))
            )
        if len(pairs) < len(images):
            print(
                '[SegmentationDataset] paired %d/%d images with masks in %s'
                % (len(pairs), len(images), self.mask_dir)
            )
        return pairs

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_path, mask_path = self.items[idx]
        image = np.asarray(Image.open(image_path).convert('RGB'))
        mask = np.asarray(Image.open(mask_path))
        sample = {
            'image': image,
            'mask': mask,
            'name': os.path.splitext(os.path.basename(image_path))[0],
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
