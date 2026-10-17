"""实验聚合与对比(纯逻辑,不依赖 torch)。

从 runs/<project>/<experiment>/<时间戳>/ 的 run.json 与 metrics.jsonl 聚合
行记录,生成 EXPERIMENTS.md / CSV / 指标曲线叠图。tools/exp/compare.py 是它的
CLI 入口,tools/exp/project.py 的收尾报告也复用这里的逻辑。
"""

from __future__ import annotations

import csv
import datetime
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIXED_COLUMNS = ['experiment', 'run_id', 'created', 'status', 'device', 'git', 'init_from']


# ---------- 行记录采集 ----------


def collect_rows(runs_root: str | Path, project: str) -> list[dict]:
    """把项目下所有 run 读成行记录(带完整 final/best 数据)。"""

    from dlkit.utils.runs import iter_run_dirs, read_run_manifest

    rows = []
    for run_dir in iter_run_dirs(runs_root, project=project):
        manifest = read_run_manifest(run_dir)
        row = {
            'run_dir': run_dir,
            'experiment': run_dir.parent.name,
            'run_id': run_dir.name,
            'created': None,
            'status': 'unknown',
            'device': '-',
            'git': '-',
            'init_from': None,
            'config': {},
            'final': {},
            'best': None,
        }
        if manifest:
            row['created'] = manifest.get('created_at')
            row['status'] = manifest.get('status', '?')
            env = manifest.get('environment') or {}
            row['device'] = env.get('device') or '-'
            commit = env.get('git_commit')
            row['git'] = (commit or '')[:7] or '-'
            row['init_from'] = manifest.get('init_from')
            row['config'] = manifest.get('config') or {}
            row['final'] = manifest.get('final') or {}
            row['best'] = manifest.get('best')
        else:
            # 旧版 run 目录没有 manifest:从 config.yaml 做最佳努力
            config_path = run_dir / 'config.yaml'
            if config_path.is_file():
                from dlkit.utils.config import load_config

                try:
                    row['config'] = load_config(str(config_path))
                except Exception:
                    row['config'] = {}
            metrics_path = run_dir / 'metrics.jsonl'
            if metrics_path.is_file():
                row['final'] = _read_last_metrics(metrics_path)
        rows.append(row)
    rows.sort(key=lambda r: (str(r['experiment']).casefold(), str(r['run_id'])))
    return rows


def _read_last_metrics(path: Path) -> dict:
    last = {}
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except ValueError:
                continue
    if isinstance(last, dict):
        last.pop('epoch', None)
        return last
    return {}


def default_metric(rows: list[dict]) -> str | None:
    """优先项目内统一的 save_best 指标,否则 val_loss。"""

    from collections import Counter

    names = Counter()
    for row in rows:
        best = row.get('best')
        if best and best.get('name'):
            names[best['name']] += 1
    if names:
        return names.most_common(1)[0][0]
    if any('val_loss' in (row.get('final') or {}) for row in rows):
        return 'val_loss'
    return None


# ---------- 表格与产物 ----------


def metric_names(rows: list[dict]) -> list[str]:
    names = []
    for row in rows:
        for key, value in (row.get('final') or {}).items():
            if key in ('epoch', 'total_epochs') or not isinstance(value, (int, float)):
                continue
            if key not in names:
                names.append(key)
    return sorted(names)


def build_table(rows: list[dict]) -> tuple[list[str], list[list[str]]]:
    headers = (
        FIXED_COLUMNS
        + ['model', 'loss', 'lr', 'plan_epochs', 'total_epochs']
        + metric_names(rows)
    )
    table = []
    for row in rows:
        config = row.get('config') or {}
        record = [
            str(row['experiment']),
            str(row['run_id']),
            str(row['created'] or '-'),
            str(row['status']),
            str(row['device']),
            str(row['git']),
            _short_source(row.get('init_from')),
        ]
        for keys in (
            ('model', 'type'),
            ('loss', 'type'),
            ('optimizer', 'params', 'lr'),
            ('train', 'epochs'),
        ):
            record.append(_fmt(_get_nested(config, keys)))
        record.append(_fmt((row.get('final') or {}).get('total_epochs')))
        for name in metric_names(rows):
            record.append(_fmt((row.get('final') or {}).get(name)))
        table.append(record)
    return headers, table


def _get_nested(config: dict, keys):
    node = config
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _fmt(value) -> str:
    if value is None:
        return '-'
    if isinstance(value, float):
        return '%.4f' % value
    return str(value)


def _short_source(path) -> str:
    """init_from 只展示 run 目录名(保留实验/时间戳信息,避免长路径)。"""

    if not path:
        return '-'
    parts = Path(str(path)).parts
    return '/'.join(parts[-3:]) if len(parts) >= 3 else str(path)


def write_outputs(
    rows: list[dict],
    project: str,
    out_dir: str | Path,
    metric: str | None,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    headers, table = build_table(rows)
    outputs.append(_write_markdown(out_dir / 'EXPERIMENTS.md', project, headers, table, rows, metric))
    outputs.append(_write_csv(out_dir / 'EXPERIMENTS.csv', headers, table))
    if metric:
        outputs.extend(plot_metric_curves(rows, project, metric, out_dir))
    return outputs


def _write_markdown(path: Path, project: str, headers, table, rows, metric) -> Path:
    lines = [
        '# Experiments: %s' % project,
        '',
        '_generated at %s by dlkit.utils.experiments_' % datetime.datetime.now().isoformat(timespec='seconds'),
        '',
    ]
    if metric:
        lines.append('preferred metric: `%s`' % metric)
        lines.append('')
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('|' + '|'.join(['---'] * len(headers)) + '|')
    # 同一实验内最新 run 在前
    order = sorted(range(len(table)), key=lambda i: table[i][1], reverse=True)
    order.sort(key=lambda i: table[i][0])
    for i in order:
        lines.append('| ' + ' | '.join(str(v) for v in table[i]) + ' |')
    lines.append('')
    lines.append('## Run directories')
    lines.append('')
    for row in rows:
        best = row.get('best')
        note = ''
        if best:
            note = ' best@%s=%.4f (epoch %s)' % (
                best.get('name'), float(best.get('value') or 0.0), best.get('epoch'),
            )
        lines.append('- `%s`%s' % (row['run_dir'], note))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def _write_csv(path: Path, headers, table) -> Path:
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(table)
    return path


def plot_metric_curves(
    rows: list[dict],
    project: str,
    metric: str,
    out_dir: str | Path,
    max_lines: int = 20,
) -> list[Path]:
    """画一个指标的全部 run 叠图,超出 max_lines 只保留每实验最新一次。"""

    series: list[tuple[str, list[tuple[int, float]]]] = []
    for row in rows:
        metrics_path = row['run_dir'] / 'metrics.jsonl'
        if not metrics_path.is_file():
            continue
        points = []
        with metrics_path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                value = record.get(metric)
                if isinstance(value, (int, float)):
                    points.append((int(record.get('epoch', len(points))), float(value)))
        if points:
            series.append(('%s@%s' % (row['experiment'], row['run_id']), points))
    if not series:
        return []

    if len(series) > max_lines:
        keep: dict[str, tuple[str, list[tuple[int, float]]]] = {}
        for label, points in series:
            experiment = label.split('@', 1)[0]
            if experiment not in keep or points[-1][0] >= keep[experiment][1][-1][0]:
                keep[experiment] = (label, points)
        series = list(keep.values())

    figure, ax = plt.subplots(figsize=(9, 5.2))
    for label, points in series:
        ax.plot(*zip(*points), label=label, linewidth=1.2)
    ax.set_xlabel('epoch')
    ax.set_ylabel(metric)
    ax.set_title('%s — %s across runs (%d lines)' % (project, metric, len(series)))
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=7)
    figure.tight_layout()
    out_path = Path(out_dir) / ('curves_%s.png' % metric)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return [out_path]


__all__ = [
    'FIXED_COLUMNS',
    'build_table',
    'collect_rows',
    'default_metric',
    'metric_names',
    'plot_metric_curves',
    'write_outputs',
]
