"""Trainer 运行身份系统端到端测试(需要 torch,缺失时自动跳过)。"""

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

    from dlkit.engine.trainer import Trainer

    ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_toy_seg_data(image_dir, mask_dir, count=4, size=32):
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    rng = np.random.RandomState(7)
    for i in range(count):
        arr = (rng.rand(size, size, 3) * 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(image_dir, '%d.png' % i))
        mask = (rng.rand(size, size) * 3).astype(np.uint8)
        Image.fromarray(mask).save(os.path.join(mask_dir, '%d.png' % i))


def _base_config(root, runs_root):
    return {
        'task': 'segmentation',
        'seed': 42,
        'device': 'cpu',
        'project': 'unittest',
        'name': 'toy_seg',
        'runs_root': str(runs_root),
        'data': {
            'train': {
                'type': 'SegmentationDataset',
                'params': {
                    'image_dir': os.path.join(root, 'train', 'images'),
                    'mask_dir': os.path.join(root, 'train', 'masks'),
                    'transform': {'type': 'Compose', 'params': {'transforms': [
                        {'type': 'Resize', 'params': {'size': [64, 64]}},
                        {'type': 'ToTensor'},
                    ]}},
                },
            },
            'val': {
                'type': 'SegmentationDataset',
                'params': {
                    'image_dir': os.path.join(root, 'val', 'images'),
                    'mask_dir': os.path.join(root, 'val', 'masks'),
                    'transform': {'type': 'Compose', 'params': {'transforms': [
                        {'type': 'Resize', 'params': {'size': [64, 64]}},
                        {'type': 'ToTensor'},
                    ]}},
                },
            },
            'loader': {'batch_size': 2, 'num_workers': 0},
        },
        'model': {
            'type': 'UNet',
            'params': {
                'num_classes': 3,
                'backbone': {'type': 'SimpleCNN', 'params': {'channels': [8, 16]}},
            },
        },
        'loss': {'type': 'CrossEntropyLoss', 'params': {'ignore_index': 255}},
        'optimizer': {'type': 'AdamW', 'params': {'lr': 0.001}},
        'train': {
            'epochs': 1,
            'amp': False,
            'log_interval': 1,
            'save_best': {'name': 'mIoU', 'mode': 'max'},
        },
        'metrics': [
            {'type': 'SegMetrics', 'params': {'num_classes': 3, 'ignore_index': 255}},
        ],
    }


if HAS_TORCH:

    def test_trainer_run_identity_and_manifest():
        with tempfile.TemporaryDirectory() as root, \
                tempfile.TemporaryDirectory() as runs_root:
            _make_toy_seg_data(os.path.join(root, 'train', 'images'), os.path.join(root, 'train', 'masks'))
            _make_toy_seg_data(os.path.join(root, 'val', 'images'), os.path.join(root, 'val', 'masks'))
            cfg = _base_config(root, runs_root)

            trainer = Trainer(cfg, opts=['train.epochs=1'])
            run_dir = Path(trainer.work_dir)
            assert run_dir.parent.name == 'toy_seg'
            assert run_dir.parent.parent.name == 'unittest'
            assert run_dir.is_relative_to(Path(runs_root).resolve())

            manifest_path = run_dir / 'run.json'
            assert manifest_path.is_file()
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            assert manifest['status'] == 'running'
            assert manifest['overrides'].get('train.epochs') == '1'
            assert manifest['experiment'] == 'toy_seg'
            assert manifest['environment']['torch'] is not None

            # 先跑 1 epoch 产生 last.pth,再验证 resume=latest 的谱系续接
            trainer.train()
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            assert manifest['status'] == 'finished'

            previous = run_dir

            # resume=latest 应续接同一谱系目录(不再新建时间戳目录)
            trainer2 = Trainer(dict(cfg), resume_from='latest')
            assert Path(trainer2.work_dir) == previous
            assert trainer2.resume_from == str(previous / 'last.pth')
            trainer2.close()

            # 换实验名则不相干
            other_cfg = dict(cfg)
            other_cfg['name'] = 'toy_other'
            trainer3 = Trainer(other_cfg)
            assert Path(trainer3.work_dir) != previous
            trainer3.close()

    def test_trainer_full_training_finalize_and_curves():
        with tempfile.TemporaryDirectory() as root, \
                tempfile.TemporaryDirectory() as runs_root:
            _make_toy_seg_data(os.path.join(root, 'train', 'images'), os.path.join(root, 'train', 'masks'))
            _make_toy_seg_data(os.path.join(root, 'val', 'images'), os.path.join(root, 'val', 'masks'))
            cfg = _base_config(root, runs_root)

            trainer = Trainer(cfg, opts=['train.epochs=1'])
            run_dir = Path(trainer.work_dir)

            from tools.train import register_curve_plot

            register_curve_plot(trainer)
            trainer.train()

            manifest = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
            assert manifest['status'] == 'finished'
            assert manifest['final']['total_epochs'] == 1
            assert manifest['best']['name'] == 'mIoU'
            assert (run_dir / 'vis' / 'curves.png').is_file()
            assert (run_dir / 'metrics.jsonl').is_file()

    def test_trainer_init_from_weights_only():
        with tempfile.TemporaryDirectory() as root, \
                tempfile.TemporaryDirectory() as runs_root:
            _make_toy_seg_data(os.path.join(root, 'train', 'images'), os.path.join(root, 'train', 'masks'))
            _make_toy_seg_data(os.path.join(root, 'val', 'images'), os.path.join(root, 'val', 'masks'))

            reference = Trainer(_base_config(root, runs_root))
            reference._build()
            state = {
                key: (value + 0.05 if value.is_floating_point() else value.clone())
                for key, value in reference.model.state_dict().items()
            }
            checkpoint = Path(runs_root) / 'pretrained.pth'
            torch.save({'model': state}, checkpoint)

            cfg = _base_config(root, runs_root)
            cfg['train']['init_from'] = str(checkpoint)
            trainer = Trainer(cfg)
            trainer._build()

            assert trainer.start_epoch == 0  # 仅加载权重,不续接 epoch
            current = trainer.model.state_dict()
            for key, value in state.items():
                assert torch.equal(current[key].cpu(), value.cpu()), key

            manifest = json.loads(
                (Path(trainer.work_dir) / 'run.json').read_text(encoding='utf-8'))
            assert manifest['init_from'] == str(checkpoint)

            trainer.close()
            reference.close()

    def test_trainer_ema_recipe():
        with tempfile.TemporaryDirectory() as root, \
                tempfile.TemporaryDirectory() as runs_root:
            _make_toy_seg_data(os.path.join(root, 'train', 'images'), os.path.join(root, 'train', 'masks'))
            _make_toy_seg_data(os.path.join(root, 'val', 'images'), os.path.join(root, 'val', 'masks'))
            cfg = _base_config(root, runs_root)
            cfg['recipe'] = {
                'type': 'EmaTeacher',
                'params': {'decay': 0.5, 'validate_teacher': True},
            }

            trainer = Trainer(cfg)
            trainer._build()
            assert trainer.ema_model is not None
            initial = next(trainer.ema_model.parameters()).detach().clone()

            trainer.train()

            teacher_after = next(trainer.ema_model.parameters()).detach().clone()
            assert not torch.equal(initial, teacher_after)
            assert (Path(trainer.work_dir) / 'last.pth').is_file()

    def test_trainer_loads_project_plugins():
        with tempfile.TemporaryDirectory() as root, \
                tempfile.TemporaryDirectory() as runs_root:
            _make_toy_seg_data(os.path.join(root, 'train', 'images'), os.path.join(root, 'train', 'masks'))
            _make_toy_seg_data(os.path.join(root, 'val', 'images'), os.path.join(root, 'val', 'masks'))
            cfg = _base_config(root, runs_root)
            extensions = Path(root) / 'extensions'
            extensions.mkdir(parents=True, exist_ok=True)
            marker = Path(root) / 'plugin_loaded.txt'
            marker.unlink(missing_ok=True)
            (extensions / 'demo_plugin.py').write_text(
                'from pathlib import Path\n'
                'from dlkit.registry import METRICS\n\n'
                'Path(%r).write_text("loaded", encoding="utf-8")\n\n'
                '@METRICS.register(name="PluginMarkerMetric")\n'
                'class PluginMarkerMetric:\n'
                '    def reset(self):\n'
                '        pass\n'
                '    def update(self, preds, batch):\n'
                '        pass\n'
                '    def compute(self):\n'
                '        return {}\n' % str(marker),
                encoding='utf-8',
            )
            cfg['imports'] = [str(extensions)]
            cfg['metrics'].append({'type': 'PluginMarkerMetric'})

            trainer = Trainer(cfg)
            trainer._build()

            from dlkit.registry import METRICS

            try:
                assert 'PluginMarkerMetric' in METRICS
                assert marker.is_file()

                manifest = json.loads(
                    (Path(trainer.work_dir) / 'run.json').read_text(encoding='utf-8'))
                assert any('demo_plugin.py' in path for path in manifest.get('plugins', []))
            finally:
                trainer.close()
                METRICS.unregister('PluginMarkerMetric')

    def test_trainer_supports_forward_batch_protocol():
        """多模态模型:实现 forward_batch(batch) 时由模型自己取 batch 字段。"""

        with tempfile.TemporaryDirectory() as root, \
                tempfile.TemporaryDirectory() as runs_root:
            _make_toy_seg_data(os.path.join(root, 'train', 'images'), os.path.join(root, 'train', 'masks'))
            _make_toy_seg_data(os.path.join(root, 'val', 'images'), os.path.join(root, 'val', 'masks'))
            cfg = _base_config(root, runs_root)

            trainer = Trainer(cfg)
            trainer._build()
            calls = []
            base_forward = trainer.model.forward

            def forward_batch(batch):
                calls.append(1)
                return base_forward(batch['image'])

            trainer.model.forward_batch = forward_batch
            trainer._train_one_epoch(0, trainer.cfg['train'])
            trainer._validate()
            assert len(calls) >= 2
            trainer.close()
