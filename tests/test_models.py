import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import dlkit.models  # noqa: F401
from dlkit.registry import build_from_cfg


def _build(cfg):
    if not HAS_TORCH:
        return
    model = build_from_cfg(cfg)
    model.eval()
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, cfg['params']['num_classes'], 128, 128), out.shape


def test_unet():
    _build({
        'type': 'UNet',
        'params': {
            'in_channels': 3,
            'num_classes': 4,
            'backbone': {'type': 'SimpleCNN', 'params': {'channels': [16, 32, 64, 128]}},
        },
    })


def test_deeplabv3():
    _build({
        'type': 'DeepLabV3',
        'params': {
            'in_channels': 3,
            'num_classes': 4,
            'backbone': {'type': 'SimpleCNN', 'params': {'channels': [16, 32, 64, 128]}},
            'decoder': {'type': 'ASPP', 'params': {'out_channels': 128, 'atrous_rates': [3, 6, 9]}},
        },
    })


def test_deeplabv3plus():
    _build({
        'type': 'DeepLabV3Plus',
        'params': {
            'in_channels': 3,
            'num_classes': 4,
            'backbone': {'type': 'SimpleCNN', 'params': {'channels': [16, 32, 64, 128]}},
            'decoder': {'type': 'ASPP', 'params': {'out_channels': 128, 'atrous_rates': [3, 6, 9]}},
            'low_level_channels': 32,
            'fuse_channels': 128,
        },
    })


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
