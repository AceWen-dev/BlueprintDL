"""run catalog 的纯文件层测试（不启动训练、不依赖 GPU）。"""

import csv
import json

import yaml

from dlkit.utils.catalog import (
    audit_catalog,
    build_lineage,
    collect_catalog,
    default_metric,
    filter_records,
    select_best,
    summarize_catalog,
    write_catalog,
)


def _config(dataset='data/toy', model='UNet'):
    return {
        'task': 'segmentation',
        'model': {'type': model},
        'loss': {'type': 'CombinedLoss'},
        'data': {
            'train': {
                'type': 'SegmentationDataset',
                'params': {
                    'image_dir': '%s/train/images' % dataset,
                    'mask_dir': '%s/train/masks' % dataset,
                },
            },
            'val': {
                'type': 'SegmentationDataset',
                'params': {
                    'image_dir': '%s/val/images' % dataset,
                    'mask_dir': '%s/val/masks' % dataset,
                },
            },
        },
    }


def _write_run(
    runs_root, project, experiment, run_id, *,
    status='finished', config=None, final=None, best=None, init_from=None,
    manifest=True, last=True, best_file=True, vis=1,
):
    run_dir = runs_root / project / experiment / run_id
    run_dir.mkdir(parents=True)
    config = config or _config()
    if manifest:
        doc = {
            'schema_version': 1,
            'status': status,
            'project': project,
            'experiment': experiment,
            'run_id': run_id,
            'created_at': '2024-01-0%sT00:00:00' % run_id[7:8],
            'config': config,
            'environment': {'device': 'cuda', 'git_commit': 'a' * 40},
            'init_from': init_from,
            'resume_from': None,
            'final': final or {},
            'best': best,
        }
        (run_dir / 'run.json').write_text(
            json.dumps(doc, ensure_ascii=False), encoding='utf-8')
    else:
        (run_dir / 'config.yaml').write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')

    if last:
        (run_dir / 'last.pth').write_text('last', encoding='utf-8')
    if best_file:
        (run_dir / 'best.pth').write_text('best', encoding='utf-8')
    if vis:
        (run_dir / 'vis').mkdir()
        for index in range(vis):
            (run_dir / 'vis' / ('sample_%d.png' % index)).write_bytes(b'png')
    if final is not None:
        (run_dir / 'metrics.jsonl').write_text(
            json.dumps({'epoch': 0, **final}) + '\n', encoding='utf-8')
    return run_dir


def _fixture_runs(tmp_path):
    runs_root = tmp_path / 'runs'
    run1 = _write_run(
        runs_root, 'proj', 'exp_a', '20240101-000000',
        final={'val_loss': 0.2, 'val_mIoU': 0.65},
        best={'name': 'val_mIoU', 'value': 0.7, 'epoch': 2},
    )
    _write_run(
        runs_root, 'proj', 'exp_a', '20240102-000000',
        final={'val_loss': 0.1, 'val_mIoU': 0.8},
        best=None,
        init_from=str(run1 / 'best.pth'),
        best_file=False,
        vis=0,
    )
    _write_run(
        runs_root, 'proj', 'exp_b', '20240103-000000',
        config=_config(model='SimpleCNN'),
        final={'val_loss': 0.7},
        manifest=False,
        best_file=False,
        vis=1,
    )
    _write_run(
        runs_root, 'proj2', 'exp_c', '20240104-000000',
        config=_config(dataset='data/other'),
        final={'val_loss': 0.3, 'val_mIoU': 0.85},
        best={'name': 'val_mIoU', 'value': 0.9, 'epoch': 4},
    )
    return runs_root, run1


def test_collect_catalog_reads_manifest_and_legacy(tmp_path):
    runs_root, _ = _fixture_runs(tmp_path)
    records = collect_catalog(runs_root)

    assert [row['key'] for row in records] == [
        'proj/exp_a/20240101-000000',
        'proj/exp_a/20240102-000000',
        'proj/exp_b/20240103-000000',
        'proj2/exp_c/20240104-000000',
    ]
    assert records[0]['source'] == 'manifest'
    assert records[0]['dataset'] == 'data/toy'
    assert records[0]['model'] == 'UNet'
    assert records[0]['best']['value'] == 0.7
    assert records[0]['artifacts']['best'] is True

    legacy = records[2]
    assert legacy['source'] == 'legacy'
    assert legacy['status'] == 'unknown'
    assert legacy['model'] == 'SimpleCNN'
    assert legacy['final']['val_loss'] == 0.7
    assert 'legacy' in legacy['flags']


def test_filter_best_and_summary(tmp_path):
    runs_root, _ = _fixture_runs(tmp_path)
    records = collect_catalog(runs_root)

    assert [row['key'] for row in filter_records(records, project='proj')] == [
        'proj/exp_a/20240101-000000',
        'proj/exp_a/20240102-000000',
        'proj/exp_b/20240103-000000',
    ]
    assert [row['key'] for row in filter_records(records, model='simplecnn')] == [
        'proj/exp_b/20240103-000000',
    ]
    assert [row['key'] for row in filter_records(records, dataset='toy')] == [
        'proj/exp_a/20240101-000000',
        'proj/exp_a/20240102-000000',
        'proj/exp_b/20240103-000000',
    ]

    assert default_metric(records) == 'val_mIoU'
    best = select_best(records, metric='val_mIoU', mode='max')
    assert [row['key'] for row in best] == [
        'proj/exp_a/20240102-000000',
        'proj2/exp_c/20240104-000000',
    ]
    assert best[0]['selected_source'] == 'final'
    assert best[0]['selected_value'] == 0.8

    summary = summarize_catalog(records)
    assert summary['runs'] == 4
    assert summary['projects'] == 2
    assert summary['experiments'] == 3
    assert summary['by_status']['finished'] == 3
    assert summary['by_source']['legacy'] == 1


def test_audit_and_lineage(tmp_path):
    runs_root, run1 = _fixture_runs(tmp_path)
    records = collect_catalog(runs_root)

    audit = audit_catalog(records)
    assert audit['summary']['legacy'] == 1
    assert audit['summary']['missing_vis'] == 1
    assert audit['issues']['missing_vis'] == ['proj/exp_a/20240102-000000']

    down = build_lineage(records, 'proj/exp_a/20240101-000000', direction='down')
    assert [row['key'] for row in down] == [
        'proj/exp_a/20240101-000000',
        'proj/exp_a/20240102-000000',
    ]
    assert down[1]['depth'] == 1

    # 从 checkpoint 路径也能反查到 run
    up = build_lineage(records, str(run1 / 'best.pth'), direction='up')
    assert up[0]['key'] == 'proj/exp_a/20240101-000000'


def test_write_catalog_outputs(tmp_path):
    runs_root, _ = _fixture_runs(tmp_path)
    records = collect_catalog(runs_root)
    out_dir = tmp_path / 'reports'

    outputs = write_catalog(records, out_dir, runs_root=runs_root)
    names = {path.name for path in outputs}
    assert names == {'catalog.json', 'catalog.md', 'catalog.csv'}

    payload = json.loads((out_dir / 'catalog.json').read_text(encoding='utf-8'))
    assert payload['schema_version'] == 1
    assert payload['summary']['runs'] == 4
    assert len(payload['runs']) == 4

    markdown = (out_dir / 'catalog.md').read_text(encoding='utf-8')
    assert '# Run catalog' in markdown
    assert 'proj/exp_a/20240101-000000' in markdown
    assert '## Latest run per experiment' in markdown

    with (out_dir / 'catalog.csv').open('r', encoding='utf-8-sig') as f:
        table = list(csv.reader(f))
    assert len(table) == 5
    assert table[0][0] == 'key'


def test_empty_catalog_is_valid(tmp_path):
    runs_root = tmp_path / 'runs'
    runs_root.mkdir()
    records = collect_catalog(runs_root)
    assert records == []

    out_dir = tmp_path / 'reports'
    outputs = write_catalog(records, out_dir, runs_root=runs_root)
    assert {path.name for path in outputs} == {'catalog.json', 'catalog.md', 'catalog.csv'}
    payload = json.loads((out_dir / 'catalog.json').read_text(encoding='utf-8'))
    assert payload['runs'] == []
    assert payload['summary']['runs'] == 0
