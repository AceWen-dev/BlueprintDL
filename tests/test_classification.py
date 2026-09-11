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
import dlkit.data    # noqa: F401
import dlkit.metrics  # noqa: F401
from dlkit.registry import build_from_cfg
from dlkit.metrics.classification_metrics import Accuracy


def test_classifier_forward():
    if not HAS_TORCH:
        return
    model = build_from_cfg({
        'type': 'Classifier',
        'params': {
            'num_classes': 4,
            'backbone': {'type': 'SimpleCNN', 'params': {'channels': [16, 32, 64, 128]}},
        },
    })
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 4), out.shape


def test_classifier_global_pool_head():
    if not HAS_TORCH:
        return
    model = build_from_cfg({
        'type': 'Classifier',
        'params': {
            'num_classes': 10,
            'backbone': {'type': 'SimpleCNN', 'params': {'channels': [16, 32]}},
            'head': {'type': 'GlobalPoolHead', 'params': {'pool': 'max', 'dropout': 0.0}},
        },
    })
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 10), out.shape


def test_accuracy_metric():
    if not HAS_TORCH:
        return
    acc = Accuracy(topk=(1, 2))
    logits = torch.tensor([
        [3.0, 1.0, 0.0, 0.0],
        [0.5, 2.0, 0.2, 0.1],
        [0.1, 0.1, 0.1, 2.0],
    ])
    batch = {'label': torch.tensor([0, 1, 3])}
    acc.update(logits, batch)
    result = acc.compute()
    assert result['Acc@1'] == 1.0, result
    assert result['Acc@2'] == 1.0, result


def test_classification_dataset(tmpdir=None):
    if not HAS_TORCH:
        return
    import tempfile
    from PIL import Image
    from dlkit.data.classification_dataset import ClassificationDataset

    with tempfile.TemporaryDirectory() as root:
        for cls in ['cat', 'dog']:
            os.makedirs(os.path.join(root, cls), exist_ok=True)
            for i in range(3):
                arr = (np.random.rand(16, 16, 3) * 255).astype(np.uint8)
                Image.fromarray(arr).save(os.path.join(root, cls, '%d.png' % i))

        ds = ClassificationDataset(root)
        assert len(ds) == 6
        assert ds.classes == ['cat', 'dog']
        sample = ds[0]
        assert sample['label'] in (0, 1)
        assert sample['image'].shape == (16, 16, 3)


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
