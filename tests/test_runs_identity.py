"""runs.runs 工具的单元测试(不依赖 torch)。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_runs_module():
    spec = importlib.util.spec_from_file_location(
        'dlkit_utils_runs_under_test', os.path.join(_ROOT, 'dlkit', 'utils', 'runs.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runs_module = _load_runs_module()

MANIFEST_NAME = runs_module.MANIFEST_NAME
collect_environment_info = runs_module.collect_environment_info
finalize_run_manifest = runs_module.finalize_run_manifest
init_run_manifest = runs_module.init_run_manifest
iter_run_dirs = runs_module.iter_run_dirs
latest_run = runs_module.latest_run
make_run_dir = runs_module.make_run_dir
read_run_manifest = runs_module.read_run_manifest


def _touch(run_dir, name):
    path = run_dir / name
    path.write_text('{}', encoding='utf-8')
    return path


def test_make_run_dir_is_unique_under_same_second():
    import tempfile

    with tempfile.TemporaryDirectory() as root:
        first = make_run_dir(root, 'examples', 'unet_toy')
        second = make_run_dir(root, 'examples', 'unet_toy')
        third = make_run_dir(root, 'examples', 'other_exp')

        assert first.parent == second.parent
        assert first != second
        assert third.parent.name == 'other_exp'
        assert first.parent.parent.name == 'examples'


def test_iter_and_latest_run_filters():
    import tempfile

    with tempfile.TemporaryDirectory() as root:
        old = make_run_dir(root, 'proj', 'exp')
        # 人为指定新旧顺序
        old.rename(old.parent / '20240101-000000')
        old = old.parent / '20240101-000000'
        new = make_run_dir(root, 'proj', 'exp')
        assert new.name >= '20260101' or new != old

        empty = new.parent / new.name  # 无 checkpoint
        _touch(old, 'last.pth')

        directories = iter_run_dirs(root, project='proj', experiment='exp')
        assert old in directories and empty in directories
        assert latest_run(root, 'proj', 'exp') == old  # 只有 old 有 last.pth

        _touch(empty, 'last.pth')
        assert latest_run(root, 'proj', 'exp') == empty  # 新目录排在最后

        # 跨实验过滤
        assert latest_run(root, 'proj', 'missing_exp') is None
        assert latest_run(root, 'proj', 'exp') is not None


def test_manifest_roundtrip_and_finalize():
    import tempfile

    with tempfile.TemporaryDirectory() as root:
        run_dir = make_run_dir(root, 'proj', 'exp')
        assert read_run_manifest(run_dir) is None

        init_run_manifest(
            run_dir,
            project='proj',
            experiment='exp',
            config={'name': 'exp', 'train': {'epochs': 3}},
            overrides={'train.epochs': '3'},
            raw_opts=['train.epochs=3'],
            resume_from=None,
        )
        manifest = read_run_manifest(run_dir)
        assert manifest is not None
        assert manifest['status'] == 'running'
        assert manifest['project'] == 'proj'
        assert manifest['run_id'] == run_dir.name
        assert manifest['config']['train']['epochs'] == 3
        assert manifest['overrides'] == {'train.epochs': '3'}
        assert 'environment' in manifest

        finalize_run_manifest(
            run_dir,
            total_epochs=3,
            last_metrics={'val_loss': 0.25, 'val_mIoU': 0.7},
            save_best={'name': 'val_mIoU', 'value': 0.7, 'epoch': 2},
        )
        manifest = read_run_manifest(run_dir)
        assert manifest['status'] == 'finished'
        assert manifest['final']['total_epochs'] == 3
        assert manifest['final']['val_loss'] == 0.25
        assert manifest['best']['name'] == 'val_mIoU'


def test_corrupt_manifest_reads_none():
    import tempfile

    with tempfile.TemporaryDirectory() as root:
        run_dir = make_run_dir(root, 'proj', 'exp')
        _touch(run_dir, MANIFEST_NAME)
        # 内容是字符串 "{}",json 有效但顶层非 dict
        (run_dir / MANIFEST_NAME).write_text('[]', encoding='utf-8')
        assert read_run_manifest(run_dir) is None


def test_environment_info_is_best_effort():
    info = collect_environment_info(device='cpu', torch_version='2.x', cuda_available=False)
    assert info['device'] == 'cpu'
    assert info['torch'] == '2.x'
    assert info['cuda_available'] is False
    # git 字段允许为 None(无 git 环境),有 git 时是 40 位哈希或 None
    assert info['git_commit'] is None or len(info['git_commit']) > 6
