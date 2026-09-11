import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import dlkit.models  # noqa: F401
from dlkit.registry import build_from_cfg
from dlkit.utils.depth_calibration import median_scale_align


def _fake_impl_factory():
    class FakeImpl:
        @classmethod
        def from_pretrained(cls, path, device='cuda'):
            return cls()

        def predict_depth(self, video, **kwargs):
            n = video.shape[0]
            h, w = video.shape[-2:]
            depth = np.ones((n, h, w), dtype=np.float32) * 2.0
            disparity = np.ones((n, h, w), dtype=np.float32) * 0.5
            return depth, disparity

    return FakeImpl


def test_build_and_forward_with_fake_impl():
    if not HAS_TORCH:
        return
    import dlkit.models.zoo.coloncrafter as cc

    orig = cc._import_coloncrafter
    cc._import_coloncrafter = lambda d: _fake_impl_factory()
    try:
        model = build_from_cfg({
            'type': 'ColonCrafter',
            'params': {'model_path': 'fake/repo', 'coloncrafter_dir': '/fake/dir', 'device': 'cpu'},
        })
        out = model(torch.rand(3, 3, 32, 32), num_inference_steps=1)
        assert out.shape == (3, 32, 32), out.shape
        assert float(out.mean()) == 2.0
    finally:
        cc._import_coloncrafter = orig


def test_missing_dependency_gives_guidance():
    if not HAS_TORCH:
        return
    model = build_from_cfg({
        'type': 'ColonCrafter',
        'params': {'model_path': 'fake/repo'},
    })
    try:
        model(torch.rand(2, 3, 32, 32))
    except RuntimeError as e:
        msg = str(e)
        assert 'git clone' in msg, msg
        return
    raise AssertionError('expected RuntimeError with installation guidance')


def test_nonexistent_dir_gives_guidance():
    if not HAS_TORCH:
        return
    model = build_from_cfg({
        'type': 'ColonCrafter',
        'params': {'model_path': 'fake/repo', 'coloncrafter_dir': '/no/such/dir'},
    })
    try:
        model(torch.rand(2, 3, 32, 32))
    except RuntimeError as e:
        assert '不存在' in str(e), e
        return
    raise AssertionError('expected RuntimeError about missing directory')


def test_median_scale_align():
    pred = np.ones((8, 8), dtype=np.float32) * 0.5   # 相对深度，中位数 0.5
    gt = np.ones((8, 8), dtype=np.float32) * 2.0     # 真值 2 米，中位数 2.0
    aligned = median_scale_align(pred, gt)
    assert abs(float(aligned.mean()) - 2.0) < 1e-3, aligned.mean()


def test_median_scale_align_invalid_pixels():
    pred = np.ones((4, 4), dtype=np.float32)
    gt = np.zeros((4, 4), dtype=np.float32)  # 全无效
    aligned = median_scale_align(pred, gt)
    assert float(aligned.mean()) == 1.0  # 不缩放，原样返回


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
