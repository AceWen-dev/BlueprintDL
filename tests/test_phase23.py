"""Phase 2/3 纯逻辑测试:数据体检(prepare_data 核心)与项目生命周期(project.py)。

- data_audit.py 零 dlkit 依赖,直接路径加载;
- project.py 的 archive 依赖 runs/experiments 的惰性导入,坏 torch 环境由
  `_stub_dlkit` 提供同名模块,正常环境用真实 dlkit。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))       # tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import _stub_dlkit

_stub_dlkit.install()

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, *relpath.split('/')))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load('dlkit_utils_data_audit_under_test', 'dlkit/utils/data_audit.py')
project = _load('tools_project_under_test', 'tools/exp/project.py')


# ---------- 数据体检 ----------


def _seg_split(base, count=4, zero_mask=False):
    images = base / 'images'
    masks = base / 'masks'
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(11)
    for i in range(count):
        Image.fromarray((rng.rand(24, 16, 3) * 255).astype(np.uint8)).save(images / ('%d.png' % i))
        values = np.zeros((24, 16), dtype=np.uint8) if zero_mask else (rng.rand(24, 16) * 3).astype(np.uint8)
        Image.fromarray(values).save(masks / ('%d.png' % i))
    return images, masks


def test_audit_segmentation_reports_issues_and_plots():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data = root / 'data' / 'demo-seg'
        train_images, train_masks = _seg_split(data / 'train')
        val_images, val_masks = _seg_split(data / 'val')
        # val 里放一张没有 mask 的图 + 一张全背景 mask
        Image.fromarray((np.random.rand(24, 16, 3) * 255).astype(np.uint8)).save(val_images / 'lonely.png')
        Image.fromarray(np.zeros((24, 16), dtype=np.uint8)).save(val_masks / '1.png')  # 覆盖为全 0

        meta = audit.audit_dataset(
            'seg',
            {'image_dir': train_images, 'mask_dir': train_masks},
            {'image_dir': val_images, 'mask_dir': val_masks},
            num_classes=3,
            sample_limit=10,
        )
        assert meta['task'] == 'segmentation'
        train_report = meta['train']
        assert train_report['num_images'] == 4
        assert train_report['matched_pairs'] == 4
        assert train_report['background_only_masks'] == 0
        assert meta['val']['matched_pairs'] == len(list(val_images.glob('*.png'))) - 1
        assert any(issue['reason'] == 'missing_mask' for issue in meta['val']['issues'])
        # 全背景 mask 计入统计(不是 issue),但像素分布里包含 0 类
        assert '0' in train_report['class_pixel_counts']

        out_dir = root / 'report'
        meta_path = audit.write_meta(meta, out_dir)
        plots = audit.render_report_plots(meta, out_dir)
        assert meta_path.is_file()
        assert json.loads(meta_path.read_text(encoding='utf-8'))['task'] == 'segmentation'
        names = {p.name for p in plots}
        assert 'class_distribution.png' in names
        for path in plots:
            assert path.is_file() and path.stat().st_size > 0


def test_audit_classification_detection_depth():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # classification: 两类 + 一个空类
        cls_root = root / 'ds-cls' / 'train'
        (cls_root / 'apple').mkdir(parents=True)
        (cls_root / 'banana').mkdir(parents=True)
        (cls_root / 'cherry').mkdir(parents=True)
        for i in range(3):
            Image.fromarray((np.random.rand(20, 20, 3) * 255).astype(np.uint8)).save(cls_root / 'apple' / ('a%d.png' % i))
        Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)).save(cls_root / 'banana' / 'b0.png')
        report = audit.audit_dataset('cls', {'root': cls_root})
        assert report['train']['num_classes'] == 3
        assert report['train']['class_counts']['apple'] == 3
        assert any(issue['reason'] == 'empty_class' for issue in report['train']['issues'])

        # detection: 1 张正常、1 张越界坐标、1 张缺 label、孤儿 label
        det_root = root / 'ds-det' / 'train'
        (det_root / 'images').mkdir(parents=True)
        (det_root / 'labels').mkdir(parents=True)
        Image.fromarray(np.zeros((24, 16, 3), dtype=np.uint8)).save(det_root / 'images' / '0.png')
        Image.fromarray(np.zeros((24, 16, 3), dtype=np.uint8)).save(det_root / 'images' / '1.png')
        (det_root / 'labels' / '0.txt').write_text('0 2 1 10 8\n1 30 0 5 5\noops\n', encoding='utf-8')
        (det_root / 'labels' / 'orphan.txt').write_text('0 1 1 4 4\n', encoding='utf-8')
        report = audit.audit_dataset('det', {'image_dir': det_root / 'images', 'label_dir': det_root / 'labels'}, num_classes=2)
        detection_report = report['train']
        assert detection_report['num_images'] == 2
        assert detection_report['matched_pairs'] == 1
        assert detection_report['num_classes_est'] == 2
        reasons = {issue['reason'].split(' ')[0] for issue in detection_report['issues']}
        assert {'box_out_of_range', 'bad_line', 'label_without_image', 'image_without_boxes'} & reasons

        # depth: npy(米)+ uint16 png(毫米)+ 缺配对
        dep_root = root / 'ds-dep' / 'train'
        (dep_root / 'images').mkdir(parents=True)
        (dep_root / 'depth').mkdir(parents=True)
        Image.fromarray((np.random.rand(12, 12, 3) * 255).astype(np.uint8)).save(dep_root / 'images' / '0.png')
        Image.fromarray((np.random.rand(12, 12, 3) * 255).astype(np.uint8)).save(dep_root / 'images' / '1.png')
        depth = (np.random.rand(12, 12).astype(np.float32) + 0.5)  # 0.5~1.5 m
        np.save(dep_root / 'depth' / '0.npy', depth)
        Image.fromarray((np.random.rand(12, 12) * 2000).astype(np.uint16)).save(dep_root / 'depth' / '1.png')
        Image.fromarray(np.zeros((12, 12, 3), dtype=np.uint8)).save(dep_root / 'images' / '2.png')
        report = audit.audit_dataset(
            'depth',
            {'image_dir': dep_root / 'images', 'depth_dir': dep_root / 'depth', 'depth_scale': 0.001},
            sample_limit=10,
        )
        depth_report = report['train']
        assert depth_report['matched_pairs'] == 2
        assert any(issue['reason'] == 'missing_depth' for issue in depth_report['issues'])
        rng = depth_report['depth_range']
        assert rng['train_max'] is not None and rng['train_max'] < 5.0
        assert depth_report['sampled_values'] is not None


def test_audit_consistency_flags_unseen_classes():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train_root = root / 'cls-tr'
        (train_root / 'cat').mkdir(parents=True)
        (train_root / 'dog').mkdir()
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(train_root / 'cat' / 'c.png')
        val_root = root / 'cls-vl'
        (val_root / 'cat').mkdir(parents=True)
        (val_root / 'lizard').mkdir()
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(val_root / 'lizard' / 'l.png')
        meta = audit.audit_dataset('classification', {'root': train_root}, {'root': val_root})
        consistency = meta['consistency']
        assert consistency['val_only_classes'] == ['lizard']


# ---------- 项目生命周期 ----------


def _fake_manifest(project, experiment, tag, idx=0):
    return {
        'schema_version': 1,
        'status': 'finished',
        'project': project,
        'experiment': experiment,
        'run_id': tag,
        'created_at': '2024-01-01T00:00:0%d' % idx,
        'config': {'model': {'type': 'UNet'}, 'loss': {'type': 'CrossEntropyLoss'},
                   'optimizer': {'type': 'AdamW', 'params': {'lr': 0.001}},
                   'train': {'epochs': 5 + idx}},
        'environment': {'device': 'cpu', 'git_commit': None},
        'final': {'total_epochs': 5 + idx, 'val_loss': 0.6 - idx * 0.1, 'val_mIoU': 0.5 + idx * 0.1},
        'best': {'name': 'val_mIoU', 'value': 0.5 + idx * 0.1, 'epoch': idx},
    }


def _write_run(runs_root, experiment, tag, idx=0):
    run_dir = runs_root / 'demo-seg' / experiment / tag
    run_dir.mkdir(parents=True)
    (run_dir / 'run.json').write_text(
        json.dumps(_fake_manifest('demo-seg', experiment, tag, idx), ensure_ascii=False), encoding='utf-8')
    (run_dir / 'metrics.jsonl').write_text(
        '{"epoch":0,"val_loss":0.9,"val_mIoU":0.3}\n{"epoch":1,"val_loss":0.7,"val_mIoU":0.55}\n',
        encoding='utf-8')
    return run_dir


def test_create_project_skeleton_and_id_validation():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        projects = root / 'projects'
        proj = project.create_project(projects, 'demo-seg', display_name='Demo Seg', task='segmentation')

        assert (proj / 'project.yaml').is_file()
        assert (proj / 'configs' / '_template.yaml').is_file()
        assert (proj / 'configs' / 'README.md').is_file()
        assert (proj / 'reports' / 'README.md').is_file()
        assert (proj / 'data' / 'README.md').is_file()
        assert (proj / 'extensions' / 'README.md').is_file()

        doc = project.load_project(proj)
        assert doc['id'] == 'demo-seg'
        assert doc['display_name'] == 'Demo Seg'
        assert doc['status'] == 'active'

        template = (proj / 'configs' / '_template.yaml').read_text(encoding='utf-8')
        assert 'project: demo-seg' in template
        assert 'data/demo-seg/train' in template
        assert 'imports:' in template
        assert 'projects/demo-seg/extensions' in template

        with pytest.raises(FileExistsError):
            project.create_project(projects, 'demo-seg')
        with pytest.raises(ValueError):
            project.create_project(projects, '2bad-id')
        assert project.valid_project_id('ok_id-1')
        assert not project.valid_project_id('Bad ID')


def test_summarize_and_status_update():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        projects = root / 'projects'
        runs = root / 'runs'
        project.create_project(projects, 'demo-seg', task='segmentation')

        summary_rows = project.summarize_projects(projects, runs)
        assert len(summary_rows) == 1
        assert summary_rows[0]['runs']['num_runs'] == 0

        _write_run(runs, 'unet_base', '20240101-000000')
        _write_run(runs, 'unet_base', '20240102-000000', idx=1)
        summary_rows = project.summarize_projects(projects, runs)
        assert summary_rows[0]['runs']['num_runs'] == 2
        assert summary_rows[0]['runs']['experiments'] == ['unet_base']
        assert summary_rows[0]['runs']['latest'].endswith('20240102-000000')

        doc = project.set_project_status(projects / 'demo-seg', 'active')
        assert doc['status'] == 'active'
        assert 'archived_at' not in doc


def test_archive_writes_closure_and_keeps_runs():
    from dlkit.utils.experiments import collect_rows

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        projects = root / 'projects'
        runs = root / 'runs'
        proj = project.create_project(projects, 'demo-seg', display_name='Demo Seg', task='segmentation')
        _write_run(runs, 'unet_base', '20240101-000000', idx=0)
        _write_run(runs, 'unet_base', '20240102-000000', idx=1)

        doc = project.load_project(proj)
        rows = collect_rows(runs, 'demo-seg')
        outputs = project.write_closure_report(proj, doc, rows, metric='val_mIoU')

        reports = proj / 'reports'
        assert {'EXPERIMENTS.md', 'EXPERIMENTS.csv', 'curves_val_mIoU.png'} <= {p.name for p in outputs}
        closure = [p for p in outputs if p.name.startswith('closure_') and p.suffix == '.md'][0]
        text = closure.read_text(encoding='utf-8')
        assert 'Project closure: Demo Seg' in text
        assert 'unet_base' in text
        assert 'Best result per experiment' in text
        assert 'scratch/' in text

        # 状态写入 archived,runs 数据一个没动
        doc = project.set_project_status(proj, 'archived')
        assert doc['status'] == 'archived'
        assert (proj / 'project.yaml').read_text(encoding='utf-8').count('archived') >= 1
        assert project.summarize_projects(projects, runs)[0]['runs']['num_runs'] == 2
        assert collect_rows(runs, 'demo-seg')
