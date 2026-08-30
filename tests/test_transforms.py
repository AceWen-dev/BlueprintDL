import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from dlkit.registry import build_from_cfg
import dlkit.data  # noqa: F401


def _make_sample(h=96, w=96):
    image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    mask = np.random.randint(0, 4, (h, w), dtype=np.int64)
    return {'image': image, 'mask': mask, 'name': 'x'}


def test_transform_pipeline():
    if not HAS_TORCH:
        return
    cfg = {
        'type': 'Compose',
        'params': {
            'transforms': [
                {'type': 'RandomCrop', 'params': {'size': [64, 64]}},
                {'type': 'RandomHorizontalFlip', 'params': {'p': 0.5}},
                {'type': 'ToTensor'},
                {'type': 'Normalize', 'params': {'mean': [0.5, 0.5, 0.5], 'std': [0.25, 0.25, 0.25]}},
            ],
        },
    }
    transform = build_from_cfg(cfg)
    sample = transform(_make_sample())
    assert sample['image'].shape == (3, 64, 64)
    assert sample['mask'].shape == (64, 64)
    assert sample['image'].dtype == torch.float32
    assert sample['mask'].dtype == torch.int64


def run_all():
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print('PASS %s' % name)
            except Exception as e:
                failed += 1
                print('FAIL %s: %s' % (name, e))
    print('done, %d failed' % failed)
    return failed


if __name__ == '__main__':
    sys.exit(run_all())
