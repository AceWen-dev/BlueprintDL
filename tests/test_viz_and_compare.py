"""TrainingCurves 与实验聚合(experiments 纯逻辑层)的测试。

- experiments.py 通过文件路径加载(纯 stdlib+matplotlib,零 torch),
  在 torch 损坏的开发机上借助 `_stub_dlkit` 照常运行;
- curves/context 涉及 dlkit 包链(torch),坏环境下跳过,uv 环境始终执行。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))       # tests/ 下取 _stub_dlkit
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util
import json
import csv
import tempfile
from pathlib import Path

import pytest

import _stub_dlkit

_stub_dlkit.install()

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_experiments():
    spec = importlib.util.spec_from_file_location(
        'dlkit_utils_experiments_under_test', os.path.join(_ROOT, 'dlkit', 'utils', 'experiments.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


experiments = _load_experiments()

try:
    from dlkit.viz.curves import plot_training_curves
    from dlkit.viz.context import resolve_norm_stats
    _VIZ_OK = True
except Exception:  # 环境问题(无 torch 或 torch 损坏),uv 环境不受影响
    _VIZ_OK = False

requires_viz = pytest.mark.skipif(not _VIZ_OK, reason='dlkit not importable in this environment')


@requires_viz
def test_plot_training_curves_outputs_grid():
    history = [
        {'epoch': i, 'train_loss': 1.0 - i * 0.05, 'val_loss': 0.9 - i * 0.04,
         'val_mIoU': 0.1 + i * 0.06, 'val_Dice': 0.2 + i * 0.05}
        for i in range(6)
    ]
    with tempfile.TemporaryDirectory() as root:
        run_dir = Path(root)
        out = plot_training_curves(history, run_dir / 'vis' / 'curves.png')
        assert Path(out).is_file()
        assert Path(out).stat().st_size > 0


@requires_viz
def test_plot_training_curves_rejects_empty_history():
    with pytest.raises(ValueError):
        plot_training_curves([], 'unused.png')


@requires_viz
def test_resolve_norm_stats_prefers_config_values():
    config = {
        'data': {
            'val': {
                'transform': {
                    'type': 'Compose',
                    'params': {
                        'transforms': [
                            {'type': 'ToTensor'},
                            {'type': 'Normalize',
                             'params': {'mean': [0.5, 0.5, 0.5], 'std': [0.25, 0.25, 0.25]}},
                        ]
                    },
                }
            }
        }
    }
    mean, std = resolve_norm_stats(config)
    assert mean == (0.5, 0.5, 0.5)
    assert std == (0.25, 0.25, 0.25)

    mean, std = resolve_norm_stats({'data': {}})
    assert len(mean) == 3 and len(std) == 3


@requires_viz
def test_visualizer_components_instantiate():
    """Registry 里存的是类,调用方必须先实例化(曾经漏掉导致曲线 hook 报错)。"""

    import dlkit.viz  # noqa: F401
    from dlkit.registry import VISUALIZERS

    for name in ('TrainingCurves', 'SegOverlay', 'ConfusionMatrix',
                 'DetBoxes', 'DepthColormap'):
        component = VISUALIZERS.get(name)
        instance = component() if isinstance(component, type) else component
        assert hasattr(instance, 'render')


def _write_fake_run(runs_root, project, experiment, tag, idx):
    run_dir = runs_root / project / experiment / tag
    run_dir.mkdir(parents=True)
    manifest = {
        'schema_version': 1,
        'status': 'finished',
        'project': project,
        'experiment': experiment,
        'run_id': tag,
        'created_at': '2024-01-0%dT00:00:00' % (idx + 1),
        'config': {
            'model': {'type': 'UNet'},
            'loss': {'type': 'CombinedLoss'},
            'optimizer': {'type': 'AdamW', 'params': {'lr': 0.001 + idx * 0.001}},
            'train': {'epochs': 10 + idx},
        },
        'environment': {'device': 'cuda', 'git_commit': 'a' * 40},
        'final': {'total_epochs': 10 + idx, 'val_loss': 0.5 - idx * 0.1,
                  'val_mIoU': 0.4 + idx * 0.1},
        'best': {'name': 'val_mIoU', 'value': 0.4 + idx * 0.1, 'epoch': idx * 2 + 2},
    }
    (run_dir / 'run.json').write_text(
        json.dumps(manifest, ensure_ascii=False), encoding='utf-8')
    history = [
        {'epoch': e, 'val_loss': 0.9 - e * 0.02, 'val_mIoU': 0.2 + e * 0.02}
        for e in range(10 + idx)
    ]
    (run_dir / 'metrics.jsonl').write_text(
        '\n'.join(json.dumps(r) for r in history) + '\n', encoding='utf-8')
    return run_dir


def test_compare_outputs_markdown_csv_plot():
    with tempfile.TemporaryDirectory() as root:
        runs_root = Path(root)
        for exp in ('exp_a', 'exp_b'):
            for idx, tag in enumerate(('20240101-000000', '20240102-000000')):
                _write_fake_run(runs_root, 'proj', exp, tag, idx)

        rows = experiments.collect_rows(runs_root, 'proj')
        assert len(rows) == 4
        assert rows[0]['status'] == 'finished'
        assert rows[0]['config']['model']['type'] == 'UNet'

        # 旧目录没有 run.json 也不崩:显示 unknown 并尽力回读
        legacy_run = runs_root / 'proj' / 'exp_a' / '20480101-000000'
        legacy_run.mkdir(parents=True, exist_ok=True)
        (legacy_run / 'config.yaml').write_text('model: {type: SimpleCNN}\n', encoding='utf-8')
        (legacy_run / 'metrics.jsonl').write_text('{"epoch":0,"val_loss":0.7}\n', encoding='utf-8')

        rows2 = experiments.collect_rows(runs_root, 'proj')
        legacy = [row for row in rows2 if row['run_id'] == '20480101-000000'][0]
        assert legacy['status'] == 'unknown'
        assert legacy['final'].get('val_loss') == 0.7
        assert legacy['config']['model']['type'] == 'SimpleCNN'

        out_dir = runs_root / 'reports'
        metric = experiments.default_metric(rows)
        assert metric == 'val_mIoU'  # 4 个 run 的 save_best 同名

        outputs = experiments.write_outputs(rows, 'proj', out_dir, metric)
        names = {Path(p).name for p in outputs}
        assert {'EXPERIMENTS.md', 'EXPERIMENTS.csv', 'curves_val_mIoU.png'} <= names

        text = (out_dir / 'EXPERIMENTS.md').read_text(encoding='utf-8')
        assert '# Experiments: proj' in text
        assert '| experiment |' in text
        assert 'val_mIoU' in text

        with (out_dir / 'EXPERIMENTS.csv').open('r', encoding='utf-8-sig') as f:
            table = list(csv.reader(f))
        assert len(table) == 5  # 表头 + 4 行
        assert table[1][0] == 'exp_a'
        assert table[0][-1] == 'val_mIoU'


def test_compare_table_metric_union_across_manifests():
    """不同 run 的 final 指标名不同时,取并集列且不丢值。"""

    with tempfile.TemporaryDirectory() as root:
        runs_root = Path(root)
        run_a = _write_fake_run(runs_root, 'proj', 'exp', '20240101-000000', 0)
        (run_a / 'run.json').write_text(
            (run_a / 'run.json').read_text(encoding='utf-8').replace(
                '"val_loss": 0.5,', '"val_accuracy": 0.9,'),
            encoding='utf-8')  # 粗暴造一个不同指标列
        rows = experiments.collect_rows(runs_root, 'proj')
        headers, table = experiments.build_table(rows)
        assert 'val_mIoU' in headers and 'val_accuracy' in headers
        assert table[0][headers.index('val_accuracy')] != '-'
