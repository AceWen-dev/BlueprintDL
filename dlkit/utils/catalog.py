"""跨项目 run 总目录(catalog)的纯逻辑层。

``compare`` 只回答“某个项目里谁更好”；catalog 回答的是更基础的问题：

- 我一共有哪些 run？分布在哪些项目/实验？
- 某个数据集、模型、状态下的 run 有哪些？
- 每个实验线目前最好的是哪一次？
- 哪些 run 缺 checkpoint / vis / 指标，或者根本没有 run.json？
- 某个权重是从哪次 run 开始微调的？

本模块只读文件，不依赖 torch；``tools/exp/catalog.py`` 是它的 CLI 入口。
"""

from __future__ import annotations

import csv
import datetime
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from dlkit.utils.runs import iter_run_dirs, read_run_manifest

CATALOG_SCHEMA_VERSION = 1
CATALOG_JSON = 'catalog.json'
CATALOG_MD = 'catalog.md'
CATALOG_CSV = 'catalog.csv'

_DATA_PATH_KEYS = ('image_dir', 'mask_dir', 'depth_dir', 'root', 'data_dir', 'list_file')
_PATH_TAIL_NAMES = {
    'train', 'val', 'test', 'images', 'masks', 'depth', 'labels', 'annotations',
}
_CHECKPOINT_NAMES = ('last.pth', 'last.pt')
_BEST_CHECKPOINT_NAMES = ('best.pth', 'best.pt')


# ---------- 单条 run 记录 ----------


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _nested(node: Any, keys: Iterable[str]) -> Any:
    current = node
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        return value
    return None


def _as_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(',') if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(part).strip() for part in value if str(part).strip()]
    return [str(value).strip()]


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except Exception:
        return {}
    try:
        with path.open('r', encoding='utf-8') as f:
            doc = yaml.safe_load(f) or {}
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _read_last_metrics(path: Path) -> dict:
    if not path.is_file():
        return {}
    last: dict = {}
    try:
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict):
                    last = record
    except OSError:
        return {}
    last.pop('epoch', None)
    if not isinstance(last, dict):
        return {}
    return last


def _timestamp_from_dir(name: str) -> str | None:
    """从 ``20240101-120000`` 时间戳目录还原一个可排序的 ISO 时间。"""

    try:
        return datetime.datetime.strptime(name[:15], '%Y%m%d-%H%M%S').isoformat(timespec='seconds')
    except (TypeError, ValueError):
        return None


def _has_file(run_dir: Path, names: Iterable[str]) -> bool:
    return any((run_dir / name).is_file() for name in names)


def _vis_count(run_dir: Path) -> int:
    vis_dir = run_dir / 'vis'
    if not vis_dir.is_dir():
        return 0
    suffixes = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
    return sum(1 for item in vis_dir.iterdir() if item.is_file() and item.suffix.lower() in suffixes)


def _infer_dataset(config: dict) -> str | None:
    """从 data.*.params 的路径推断一个“数据集提示”。

    这不是 dataset registry，只是让人能在 catalog 里按数据根目录粗筛。
    显式写了 ``dataset`` 时永远优先使用显式值。
    """

    data = _as_dict(config.get('data'))
    paths: list[str] = []
    for split in ('train', 'val', 'test'):
        params = _as_dict(_as_dict(data.get(split)).get('params'))
        for key in _DATA_PATH_KEYS:
            value = params.get(key)
            if isinstance(value, str) and value.strip():
                paths.append(value.replace('\\', '/').strip('/'))
    if not paths:
        return None

    columns = [path.split('/') for path in paths]
    common: list[str] = []
    for parts in zip(*columns):
        if len(set(parts)) != 1:
            break
        common.append(parts[0])
    while common and common[-1].lower() in _PATH_TAIL_NAMES:
        common.pop()
    if not common:
        return None
    return '/'.join(common)


def _normalize_best(value: Any) -> dict | None:
    if not isinstance(value, dict):
        return None
    name = value.get('name')
    metric_value = value.get('value')
    if name is None or not isinstance(metric_value, (int, float)):
        return None
    return {
        'name': str(name),
        'value': float(metric_value),
        'epoch': value.get('epoch'),
    }


def build_record(run_dir: str | Path, runs_root: str | Path = 'runs') -> dict:
    """把一个 run 目录读成 catalog 记录；缺 run.json 时降级为 legacy。"""

    run_dir = Path(run_dir)
    runs_root = Path(runs_root)
    project = run_dir.parent.parent.name
    experiment = run_dir.parent.name
    run_id = run_dir.name
    manifest = read_run_manifest(run_dir)

    config: dict = {}
    config_source = None
    if manifest is not None:
        config = _as_dict(manifest.get('config'))
        if config:
            config_source = 'run.json'
    if not config:
        config_path = run_dir / 'config.yaml'
        if config_path.is_file():
            config = _load_yaml(config_path)
            config_source = 'config.yaml'

    if manifest is not None:
        final = _as_dict(manifest.get('final'))
        best = _normalize_best(manifest.get('best'))
        status = str(manifest.get('status') or 'unknown')
        created_at = manifest.get('created_at')
        environment = _as_dict(manifest.get('environment'))
        init_from = manifest.get('init_from')
        resume_from = manifest.get('resume_from')
        parent_run = manifest.get('parent_run')
        source = 'manifest'
    else:
        final = _read_last_metrics(run_dir / 'metrics.jsonl')
        best = None
        status = 'unknown'
        created_at = _timestamp_from_dir(run_id)
        environment = {}
        init_from = None
        resume_from = None
        parent_run = None
        source = 'legacy'

    try:
        relative_dir = run_dir.relative_to(runs_root).as_posix()
    except ValueError:
        relative_dir = run_dir.as_posix()

    dataset = _first(
        manifest.get('dataset') if manifest else None,
        config.get('dataset'),
        _nested(config, ('data', 'dataset')),
        _infer_dataset(config),
    )
    split = _first(
        manifest.get('split') if manifest else None,
        config.get('split'),
        _nested(config, ('data', 'split')),
    )
    tags = list(dict.fromkeys(
        _as_tags(manifest.get('tags') if manifest else None)
        + _as_tags(config.get('tags'))
        + _as_tags(_nested(config, ('experiment', 'tags')))
    ))

    artifacts = {
        'last': _has_file(run_dir, _CHECKPOINT_NAMES),
        'best': _has_file(run_dir, _BEST_CHECKPOINT_NAMES),
        'vis': _vis_count(run_dir),
        'metrics': (run_dir / 'metrics.jsonl').is_file(),
        'config': config_source is not None,
    }
    flags: list[str] = []
    if source == 'legacy':
        flags.append('legacy')
    if not artifacts['last'] and not artifacts['best']:
        flags.append('missing-checkpoint')
    if best and not artifacts['best']:
        flags.append('missing-best-ckpt')
    if status == 'finished' and artifacts['vis'] == 0:
        flags.append('missing-vis')
    if config_source is None:
        flags.append('missing-config')
    if not final and not best:
        flags.append('no-metrics')

    device = environment.get('device') or '-'
    commit = environment.get('git_commit') or ''
    return {
        'key': '%s/%s/%s' % (project, experiment, run_id),
        'project': project,
        'experiment': experiment,
        'run_id': run_id,
        'run_dir': relative_dir,
        'source': source,
        'created_at': created_at,
        'status': status,
        'task': config.get('task'),
        'model': _nested(config, ('model', 'type')),
        'loss': _nested(config, ('loss', 'type')),
        'dataset': None if dataset is None else str(dataset),
        'split': None if split is None else str(split),
        'tags': tags,
        'git': commit[:7] if commit else '-',
        'device': device,
        'final': final,
        'best': best,
        'init_from': init_from,
        'resume_from': resume_from,
        'parent_run': parent_run,
        'artifacts': artifacts,
        'flags': flags,
    }


def collect_catalog(
    runs_root: str | Path = 'runs',
    project: str | None = None,
) -> list[dict]:
    """扫描 runs 根目录，返回按项目/实验/时间排序的 catalog 记录。"""

    root = Path(runs_root)
    records = [build_record(run_dir, root) for run_dir in iter_run_dirs(root, project=project)]
    records.sort(key=lambda row: (
        str(row.get('project', '')).casefold(),
        str(row.get('experiment', '')).casefold(),
        str(row.get('created_at') or ''),
        str(row.get('run_id') or ''),
    ))
    return records


# ---------- 查询 / 最佳 / 审计 / 谱系 ----------


def filter_records(
    records: Iterable[dict],
    *,
    project: str | None = None,
    experiment: str | None = None,
    status: str | None = None,
    model: str | None = None,
    dataset: str | None = None,
    split: str | None = None,
    tag: str | None = None,
    source: str | None = None,
) -> list[dict]:
    """按字段过滤；文本字段用不区分大小写的包含匹配。"""

    def contains(value: Any, needle: str | None) -> bool:
        return needle is None or needle.casefold() in str(value or '').casefold()

    def exact(value: Any, needle: str | None) -> bool:
        return needle is None or str(value or '').casefold() == needle.casefold()

    result = []
    for row in records:
        if not exact(row.get('project'), project):
            continue
        if not exact(row.get('experiment'), experiment):
            continue
        if not exact(row.get('status'), status):
            continue
        if not contains(row.get('model'), model):
            continue
        if not contains(row.get('dataset'), dataset):
            continue
        if not contains(row.get('split'), split):
            continue
        if source is not None and not exact(row.get('source'), source):
            continue
        if tag is not None and not any(tag.casefold() in str(item).casefold() for item in row.get('tags') or []):
            continue
        result.append(row)
    return result


def _metric_value(record: dict, metric: str) -> tuple[float | None, str | None]:
    best = record.get('best') or {}
    if best.get('name') == metric and isinstance(best.get('value'), (int, float)):
        return float(best['value']), 'best'
    final = record.get('final') or {}
    value = final.get(metric)
    if isinstance(value, (int, float)):
        return float(value), 'final'
    return None, None


def default_metric(records: Iterable[dict]) -> str | None:
    """优先使用出现次数最多的 save_best 指标，否则 val_loss。"""

    names: Counter = Counter()
    finals: Counter = Counter()
    for row in records:
        best = row.get('best') or {}
        if best.get('name'):
            names[str(best['name'])] += 1
        for key, value in (row.get('final') or {}).items():
            if key == 'total_epochs' or not isinstance(value, (int, float)):
                continue
            finals[str(key)] += 1
    if names:
        return names.most_common(1)[0][0]
    if 'val_loss' in finals:
        return 'val_loss'
    return finals.most_common(1)[0][0] if finals else None


def select_best(
    records: Iterable[dict],
    metric: str | None = None,
    mode: str = 'max',
) -> list[dict]:
    """每个 (project, experiment) 选一个代表 run。

    ``best`` 字段优先；没有 best 时退回 final 指标。同一数值并列时取更新的 run。
    返回的每条记录带 ``selected_metric`` / ``selected_value`` / ``selected_source``。
    """

    if mode not in ('max', 'min'):
        raise ValueError("mode must be 'max' or 'min'")
    records = list(records)
    metric = metric or default_metric(records)
    if not metric:
        return []

    groups: dict[tuple[str, str], list[tuple[float, str, dict]]] = defaultdict(list)
    for row in records:
        value, source = _metric_value(row, metric)
        if value is None:
            continue
        groups[(str(row['project']), str(row['experiment']))].append((value, source or 'final', row))

    selected = []
    for key in sorted(groups):
        entries = groups[key]
        target = max(value for value, _, _ in entries) if mode == 'max' else min(
            value for value, _, _ in entries)
        candidates = [entry for entry in entries if entry[0] == target]
        value, source, row = max(
            candidates,
            key=lambda entry: (str(entry[2].get('created_at') or ''), str(entry[2].get('run_id') or '')),
        )
        chosen = dict(row)
        chosen['selected_metric'] = metric
        chosen['selected_value'] = value
        chosen['selected_source'] = source
        selected.append(chosen)
    return selected


def audit_catalog(records: Iterable[dict]) -> dict:
    """列出需要人工看一眼的 run，不修改任何文件。"""

    issues: dict[str, list[str]] = {
        'legacy': [],
        'running': [],
        'missing_checkpoint': [],
        'missing_best_checkpoint': [],
        'missing_vis': [],
        'missing_config': [],
        'no_metrics': [],
    }
    for row in records:
        key = str(row['key'])
        if row.get('source') == 'legacy':
            issues['legacy'].append(key)
        if row.get('status') == 'running':
            issues['running'].append(key)
        artifacts = row.get('artifacts') or {}
        if not artifacts.get('last') and not artifacts.get('best'):
            issues['missing_checkpoint'].append(key)
        if row.get('best') and not artifacts.get('best'):
            issues['missing_best_checkpoint'].append(key)
        if row.get('status') == 'finished' and not artifacts.get('vis'):
            issues['missing_vis'].append(key)
        if not artifacts.get('config'):
            issues['missing_config'].append(key)
        if not (row.get('final') or row.get('best')):
            issues['no_metrics'].append(key)
    return {
        'summary': {name: len(keys) for name, keys in issues.items()},
        'issues': issues,
    }


def _normalize_ref(value: Any) -> str:
    return str(value).replace('\\', '/').strip().rstrip('/')


def _match_run_key(records: Iterable[dict], ref: Any) -> str | None:
    """把 run_id、run 目录、checkpoint 路径或 key 统一映射到 catalog key。"""

    if ref is None or not str(ref).strip():
        return None
    records = list(records)
    text = _normalize_ref(ref)
    by_key = {str(row['key']): row for row in records}
    if text in by_key:
        return text

    candidates = [text]
    path = Path(text)
    if path.name.lower() in {'last.pth', 'last.pt', 'best.pth', 'best.pt', 'run.json', 'config.yaml'}:
        candidates.append(_normalize_ref(path.parent))

    for candidate in candidates:
        for row in records:
            key = str(row['key'])
            if candidate == key or candidate.endswith('/' + key):
                return key

    # 允许只写 run_id；只有唯一匹配时才接受，避免跨项目误连。
    matches = [row for row in records if str(row['run_id']) == text]
    if len(matches) == 1:
        return str(matches[0]['key'])
    return None


def build_lineage(
    records: Iterable[dict],
    start: Any,
    direction: str = 'down',
) -> list[dict]:
    """沿 init_from / resume_from / parent_run 追踪 run 的前驱或后继。"""

    if direction not in ('down', 'up'):
        raise ValueError("direction must be 'down' or 'up'")
    records = list(records)
    by_key = {str(row['key']): row for row in records}
    start_key = _match_run_key(records, start)
    if start_key is None:
        raise KeyError('run not found in catalog: %s' % start)

    parent_of: dict[str, str] = {}
    for row in records:
        for ref in (row.get('parent_run'), row.get('init_from'), row.get('resume_from')):
            parent_key = _match_run_key(records, ref)
            if parent_key and parent_key != row['key']:
                parent_of[str(row['key'])] = parent_key
                break

    if direction == 'up':
        result = []
        current: str | None = start_key
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            result.append({**by_key[current], 'depth': len(result)})
            current = parent_of.get(current)
        return result

    children: dict[str, list[str]] = defaultdict(list)
    for child, parent in parent_of.items():
        children[parent].append(child)
    result = []
    queue: list[tuple[str, int]] = [(start_key, 0)]
    seen: set[str] = set()
    while queue:
        key, depth = queue.pop(0)
        if key in seen:
            continue
        seen.add(key)
        result.append({**by_key[key], 'depth': depth})
        for child in sorted(children.get(key, [])):
            queue.append((child, depth + 1))
    return result


# ---------- 汇总 / 输出 ----------


def summarize_catalog(records: Iterable[dict]) -> dict:
    records = list(records)
    projects: Counter = Counter(str(row.get('project')) for row in records)
    experiments: Counter = Counter('%s/%s' % (row.get('project'), row.get('experiment')) for row in records)
    statuses: Counter = Counter(str(row.get('status')) for row in records)
    sources: Counter = Counter(str(row.get('source')) for row in records)
    flags: Counter = Counter(flag for row in records for flag in (row.get('flags') or []))
    return {
        'runs': len(records),
        'projects': len(projects),
        'experiments': len(experiments),
        'by_project': dict(sorted(projects.items())),
        'by_status': dict(sorted(statuses.items())),
        'by_source': dict(sorted(sources.items())),
        'flags': dict(sorted(flags.items())),
    }


def _best_text(row: dict) -> str:
    best = row.get('best') or {}
    if not best:
        return '-'
    return '%s=%.4f@%s' % (best.get('name'), float(best.get('value') or 0.0), best.get('epoch', '-'))


def _latest_per_experiment(records: Iterable[dict]) -> list[dict]:
    latest: dict[tuple[str, str], dict] = {}
    for row in records:
        key = (str(row['project']), str(row['experiment']))
        current = latest.get(key)
        if current is None or (
            str(row.get('created_at') or ''), str(row.get('run_id') or '')
        ) > (
            str(current.get('created_at') or ''), str(current.get('run_id') or '')
        ):
            latest[key] = row
    return [latest[key] for key in sorted(latest)]


def _md_cell(value: Any) -> str:
    return str(value if value is not None else '-').replace('|', '\\|')


def _write_markdown(
    path: Path,
    records: list[dict],
    summary: dict,
    generated_at: str,
    runs_root: str,
) -> Path:
    lines = [
        '# Run catalog',
        '',
        '- generated_at: `%s`' % generated_at,
        '- runs_root: `%s`' % runs_root,
        '- runs: %d | projects: %d | experiments: %d' % (
            summary['runs'], summary['projects'], summary['experiments']),
        '',
        '## Status',
        '',
        '| status | runs |',
        '| --- | ---: |',
    ]
    for name, count in summary['by_status'].items():
        lines.append('| %s | %d |' % (_md_cell(name), count))
    lines += ['', '## Projects', '', '| project | runs |', '| --- | ---: |']
    for name, count in summary['by_project'].items():
        lines.append('| %s | %d |' % (_md_cell(name), count))
    lines += [
        '',
        '## Runs',
        '',
        '| key | created | status | model | task | dataset | split | best | source | flags |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in records:
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            _md_cell(row.get('key')),
            _md_cell(row.get('created_at')),
            _md_cell(row.get('status')),
            _md_cell(row.get('model')),
            _md_cell(row.get('task')),
            _md_cell(row.get('dataset')),
            _md_cell(row.get('split')),
            _md_cell(_best_text(row)),
            _md_cell(row.get('source')),
            _md_cell(', '.join(row.get('flags') or []) or '-'),
        ))
    lines += [
        '',
        '## Latest run per experiment',
        '',
        '| experiment | latest run | best |',
        '| --- | --- | --- |',
    ]
    for row in _latest_per_experiment(records):
        lines.append('| %s/%s | %s | %s |' % (
            _md_cell(row.get('project')),
            _md_cell(row.get('experiment')),
            _md_cell(row.get('key')),
            _md_cell(_best_text(row)),
        ))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def _csv_row(row: dict) -> dict:
    best = row.get('best') or {}
    artifacts = row.get('artifacts') or {}
    return {
        'key': row.get('key'),
        'project': row.get('project'),
        'experiment': row.get('experiment'),
        'run_id': row.get('run_id'),
        'created_at': row.get('created_at'),
        'status': row.get('status'),
        'source': row.get('source'),
        'task': row.get('task'),
        'model': row.get('model'),
        'loss': row.get('loss'),
        'dataset': row.get('dataset'),
        'split': row.get('split'),
        'tags': ';'.join(row.get('tags') or []),
        'git': row.get('git'),
        'device': row.get('device'),
        'best_name': best.get('name'),
        'best_value': best.get('value'),
        'best_epoch': best.get('epoch'),
        'init_from': row.get('init_from'),
        'last_ckpt': artifacts.get('last'),
        'best_ckpt': artifacts.get('best'),
        'vis_count': artifacts.get('vis'),
        'flags': ';'.join(row.get('flags') or []),
    }


def write_catalog(
    records: Iterable[dict],
    out_dir: str | Path,
    *,
    runs_root: str | Path = 'runs',
    formats: Iterable[str] = ('json', 'md', 'csv'),
) -> list[Path]:
    """写出 catalog.json / catalog.md / catalog.csv，返回文件路径列表。"""

    records = list(records)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = {str(fmt).lower() for fmt in formats}
    generated_at = datetime.datetime.now().isoformat(timespec='seconds')
    summary = summarize_catalog(records)
    outputs: list[Path] = []

    if 'json' in formats:
        path = out_dir / CATALOG_JSON
        with path.open('w', encoding='utf-8') as f:
            json.dump({
                'schema_version': CATALOG_SCHEMA_VERSION,
                'generated_at': generated_at,
                'runs_root': str(runs_root),
                'summary': summary,
                'runs': records,
            }, f, indent=2, ensure_ascii=False)
            f.write('\n')
        outputs.append(path)

    if 'md' in formats:
        outputs.append(_write_markdown(
            out_dir / CATALOG_MD, records, summary, generated_at, str(runs_root)))

    if 'csv' in formats:
        path = out_dir / CATALOG_CSV
        headers = list(_csv_row(records[0]).keys()) if records else [
            'key', 'project', 'experiment', 'run_id', 'created_at', 'status', 'source',
            'task', 'model', 'loss', 'dataset', 'split', 'tags', 'git', 'device',
            'best_name', 'best_value', 'best_epoch', 'init_from', 'last_ckpt',
            'best_ckpt', 'vis_count', 'flags',
        ]
        with path.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in records:
                writer.writerow(_csv_row(row))
        outputs.append(path)

    return outputs


__all__ = [
    'CATALOG_CSV',
    'CATALOG_JSON',
    'CATALOG_MD',
    'CATALOG_SCHEMA_VERSION',
    'audit_catalog',
    'build_lineage',
    'build_record',
    'collect_catalog',
    'default_metric',
    'filter_records',
    'select_best',
    'summarize_catalog',
    'write_catalog',
]
