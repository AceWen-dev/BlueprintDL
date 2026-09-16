import random

import numpy as np
import torch
from PIL import Image

from dlkit.builders import build_from_cfg
from dlkit.registry import TRANSFORMS


def _has_mask(sample):
    return 'mask' in sample and sample['mask'] is not None


@TRANSFORMS.register()
class Compose:
    def __init__(self, transforms):
        self.transforms = [
            build_from_cfg(t) if isinstance(t, dict) else t for t in transforms
        ]

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self):
        return 'Compose(' + ', '.join(repr(t) for t in self.transforms) + ')'


@TRANSFORMS.register()
class Resize:
    def __init__(self, size):
        if isinstance(size, (int, float)):
            self.size = (int(size), int(size))
        else:
            self.size = (int(size[0]), int(size[1]))

    def __call__(self, sample):
        image = Image.fromarray(sample['image'])
        image = image.resize((self.size[1], self.size[0]), Image.BILINEAR)
        sample['image'] = np.asarray(image)
        if _has_mask(sample):
            mask = Image.fromarray(sample['mask'])
            mask = mask.resize((self.size[1], self.size[0]), Image.NEAREST)
            sample['mask'] = np.asarray(mask)
        return sample


@TRANSFORMS.register()
class CenterCrop:
    def __init__(self, size):
        if isinstance(size, (int, float)):
            self.size = (int(size), int(size))
        else:
            self.size = (int(size[0]), int(size[1]))

    def __call__(self, sample):
        image, mask = sample['image'], sample.get('mask')
        h, w = image.shape[:2]
        ch, cw = self.size
        top = max(0, (h - ch) // 2)
        left = max(0, (w - cw) // 2)
        sample['image'] = image[top:top + ch, left:left + cw]
        if mask is not None:
            sample['mask'] = mask[top:top + ch, left:left + cw]
        return sample


@TRANSFORMS.register()
class RandomCrop:
    def __init__(self, size, pad_if_needed=True):
        if isinstance(size, (int, float)):
            self.size = (int(size), int(size))
        else:
            self.size = (int(size[0]), int(size[1]))
        self.pad_if_needed = pad_if_needed

    def __call__(self, sample):
        image, mask = sample['image'], sample.get('mask')
        h, w = image.shape[:2]
        ch, cw = self.size
        if self.pad_if_needed and (h < ch or w < cw):
            ph = max(0, ch - h)
            pw = max(0, cw - w)
            image = np.pad(image, ((0, ph), (0, pw), (0, 0)))
            if mask is not None:
                mask = np.pad(mask, ((0, ph), (0, pw)))
        h, w = image.shape[:2]
        top = random.randint(0, h - ch)
        left = random.randint(0, w - cw)
        sample['image'] = image[top:top + ch, left:left + cw]
        if mask is not None:
            sample['mask'] = mask[top:top + ch, left:left + cw]
        return sample


@TRANSFORMS.register()
class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = np.ascontiguousarray(sample['image'][:, ::-1])
            if _has_mask(sample):
                sample['mask'] = np.ascontiguousarray(sample['mask'][:, ::-1])
        return sample


@TRANSFORMS.register()
class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = np.ascontiguousarray(sample['image'][::-1])
            if _has_mask(sample):
                sample['mask'] = np.ascontiguousarray(sample['mask'][::-1])
        return sample


@TRANSFORMS.register()
class RandomRotate90:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            k = random.randint(1, 3)
            sample['image'] = np.ascontiguousarray(np.rot90(sample['image'], k, (0, 1)))
            if _has_mask(sample):
                sample['mask'] = np.ascontiguousarray(np.rot90(sample['mask'], k, (0, 1)))
        return sample


@TRANSFORMS.register()
class RandomBrightnessContrast:
    def __init__(self, alpha=(0.8, 1.2), beta=(-0.1, 0.1)):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, sample):
        image = sample['image']
        a = random.uniform(*self.alpha)
        b = random.uniform(*self.beta)
        if isinstance(image, torch.Tensor):
            image = image * a + b * 255.0
            image = torch.clamp(image, 0.0, 255.0)
        else:
            image = image.astype(np.float32) * a + b * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        sample['image'] = image
        return sample


@TRANSFORMS.register()
class ToTensor:
    def __call__(self, sample):
        image = sample['image']
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.array(image, copy=True)).float()
            if image.dim() == 2:
                image = image.unsqueeze(0)
            else:
                image = image.permute(2, 0, 1)
            image = image / 255.0
        sample['image'] = image
        mask = sample.get('mask')
        if mask is not None and not isinstance(mask, torch.Tensor):
            sample['mask'] = torch.from_numpy(np.array(mask, copy=True)).long()
        return sample


@TRANSFORMS.register()
class ToTensorDepth:
    """深度版 ToTensor：图像 -> (C,H,W) float 张量，深度图 -> (1,H,W) float 张量（米）。

    与 ToTensor 的区别：深度图是连续值，不能转 long。
    """

    def __call__(self, sample):
        image = sample['image']
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.array(image, copy=True)).float()
            if image.dim() == 2:
                image = image.unsqueeze(0)
            else:
                image = image.permute(2, 0, 1)
            image = image / 255.0
        depth = sample['mask']
        if not isinstance(depth, torch.Tensor):
            depth = torch.from_numpy(np.array(depth, copy=True)).float()
            if depth.dim() == 2:
                depth = depth.unsqueeze(0)
        sample['image'] = image
        sample['mask'] = depth
        return sample


@TRANSFORMS.register()
class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample['image']
        c = image.shape[0]
        if len(self.mean) != c or len(self.std) != c:
            raise ValueError(
                'Normalize expects %d channels but got mean/std of length %d/%d'
                % (c, len(self.mean), len(self.std))
            )
        mean = torch.as_tensor(self.mean, dtype=image.dtype, device=image.device).view(c, 1, 1)
        std = torch.as_tensor(self.std, dtype=image.dtype, device=image.device).view(c, 1, 1)
        sample['image'] = (image - mean) / std
        return sample
