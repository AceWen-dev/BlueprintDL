"""自监督(SimCLR 风格)组件与端到端训练测试(需要 torch,缺失时自动跳过)。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    HAS_TORCH = True
except Exception:  # ImportError 或损坏安装(如 c10.dll 加载失败)
    HAS_TORCH = False

if HAS_TORCH:
    import json
    import tempfile
    from pathlib import Path

    import numpy as np
    from PIL import Image

    import dlkit.data  # noqa: F401
    import dlkit.models  # noqa: F401
    from dlkit.builders import build_from_cfg
    from dlkit.engine.trainer import Trainer
    from dlkit.registry import DATASETS, LOSSES, MODELS
    from dlkit.utils.checkpoint import load_model_weights


def _make_toy_images(image_dir, count=8, size=32):
    os.makedirs(image_dir, exist_ok=True)
    rng = np.random.RandomState(11)
    for i in range(count):
        array = (rng.rand(size, size, 3) * 255).astype(np.uint8)
        Image.fromarray(array).save(os.path.join(image_dir, '%d.png' % i))


def _view_transform():
    return {
        'type': 'Compose',
        'params': {
            'transforms': [
                {'type': 'RandomCrop', 'params': {'size': [16, 16]}},
                {'type': 'RandomHorizontalFlip', 'params': {'p': 0.5}},
                {'type': 'ToTensor'},
                {'type': 'Normalize', 'params': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}},
            ]
        },
    }


def _ssl_config(root, runs_root):
    return {
        'task': 'self_supervised',
        'seed': 42,
        'device': 'cpu',
        'project': 'unittest',
        'name': 'ssl_toy',
        'runs_root': str(runs_root),
        'data': {
            'train': {
                'type': 'TwoViewDataset',
                'params': {
                    'image_dir': os.path.join(root, 'train', 'images'),
                    'view_transform': _view_transform(),
                },
            },
            'val': {
                'type': 'TwoViewDataset',
                'params': {
                    'image_dir': os.path.join(root, 'val', 'images'),
                    'view_transform': _view_transform(),
                },
            },
            'loader': {'batch_size': 4, 'num_workers': 0},
        },
        'model': {
            'type': 'SSLModel',
            'params': {
                'in_channels': 3,
                'projection_dim': 8,
                'hidden_dim': 16,
                'backbone': {'type': 'SimpleCNN', 'params': {'channels': [8, 16]}},
            },
        },
        'loss': {'type': 'NTXentLoss', 'params': {'temperature': 0.5}},
        'optimizer': {'type': 'AdamW', 'params': {'lr': 0.001}},
        'train': {
            'epochs': 1,
            'amp': False,
            'log_interval': 1,
            'save_best': {'name': 'val_loss', 'mode': 'min'},
        },
        'metrics': [],
    }


if HAS_TORCH:

    def test_ssl_components_registered():
        assert 'TwoViewDataset' in DATASETS
        assert 'SSLModel' in MODELS
        assert 'NTXentLoss' in LOSSES

    def test_two_view_dataset_shapes():
        from torch.utils.data import DataLoader

        with tempfile.TemporaryDirectory() as root:
            image_dir = os.path.join(root, 'images')
            _make_toy_images(image_dir, count=5)
            dataset = build_from_cfg({
                'type': 'TwoViewDataset',
                'params': {'image_dir': image_dir, 'view_transform': _view_transform()},
            })
            assert len(dataset) == 5
            sample = dataset[0]
            assert sample['image'].shape == (2, 3, 16, 16)
            assert sample['image'].dtype == torch.float32

            loader = DataLoader(dataset, batch_size=2)
            batch = next(iter(loader))
            assert batch['image'].shape == (2, 2, 3, 16, 16)

    def test_ssl_model_forward_and_encode():
        model = build_from_cfg({
            'type': 'SSLModel',
            'params': {
                'in_channels': 3,
                'projection_dim': 8,
                'hidden_dim': 16,
                'backbone': {'type': 'SimpleCNN', 'params': {'channels': [8, 16]}},
            },
        })
        model.eval()
        with torch.no_grad():
            two_view = model(torch.randn(2, 2, 3, 32, 32))
            single = model(torch.randn(3, 3, 32, 32))
            features = model.encode(torch.randn(3, 3, 32, 32))
        assert two_view.shape == (4, 8)
        assert single.shape == (3, 8)
        assert features.shape == (3, 16)

    def test_ntxent_loss_prefers_aligned_views():
        loss_fn = build_from_cfg({
            'type': 'NTXentLoss',
            'params': {'temperature': 0.1},
        })
        aligned = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        mismatched = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])

        low = loss_fn(aligned, None)
        high = loss_fn(mismatched, None)
        assert float(low) < 0.01
        assert float(high) > 1.0

        embeddings = torch.randn(4, 3, requires_grad=True)
        loss = loss_fn(embeddings, None)
        loss.backward()
        assert embeddings.grad is not None

    def test_ssl_trainer_end_to_end():
        with tempfile.TemporaryDirectory() as root, \
                tempfile.TemporaryDirectory() as runs_root:
            _make_toy_images(os.path.join(root, 'train', 'images'), count=8)
            _make_toy_images(os.path.join(root, 'val', 'images'), count=4)
            cfg = _ssl_config(root, runs_root)

            trainer = Trainer(cfg)
            run_dir = Path(trainer.work_dir)
            history = trainer.train()

            assert len(history) == 1
            assert isinstance(history[0]['val_loss'], float)
            assert np.isfinite(history[0]['val_loss'])
            assert (run_dir / 'last.pth').is_file()
            assert (run_dir / 'best.pth').is_file()

            manifest = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
            assert manifest['status'] == 'finished'
            assert manifest['best']['name'] == 'val_loss'
            assert manifest['final']['total_epochs'] == 1

    def test_ssl_checkpoint_initializes_supervised_backbone():
        """预训练权重 -> 监督分类器:backbone 全量匹配,投影头被忽略。"""

        with tempfile.TemporaryDirectory() as root:
            ssl_model = build_from_cfg({
                'type': 'SSLModel',
                'params': {
                    'in_channels': 3,
                    'projection_dim': 8,
                    'hidden_dim': 16,
                    'backbone': {'type': 'SimpleCNN', 'params': {'channels': [8, 16]}},
                },
            })
            checkpoint = Path(root) / 'ssl.pth'
            torch.save({'model': ssl_model.state_dict()}, checkpoint)

            classifier = build_from_cfg({
                'type': 'Classifier',
                'params': {
                    'num_classes': 3,
                    'backbone': {'type': 'SimpleCNN', 'params': {'channels': [8, 16]}},
                },
            })
            result = load_model_weights(checkpoint, classifier, device='cpu', strict=False)

            missing = [key for key in result.missing_keys if not key.startswith('head.')]
            assert missing == []
            assert all(key.startswith('projector.') for key in result.unexpected_keys)

            ssl_state = ssl_model.state_dict()
            classifier_state = classifier.state_dict()
            key = 'backbone.stages.0.0.block.0.weight'
            assert torch.equal(ssl_state[key], classifier_state[key])
