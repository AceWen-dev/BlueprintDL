"""跨项目 run 总目录 CLI。

    uv run python tools/exp/catalog.py build
    uv run python tools/exp/catalog.py query --project examples --status finished
    uv run python tools/exp/catalog.py best --metric val_mIoU
    uv run python tools/exp/catalog.py audit
    uv run python tools/exp/catalog.py lineage --run runs/examples/unet_toy/<ts>

``build`` 在 ``reports/`` 下生成 catalog.json / catalog.md / catalog.csv；
``query`` / ``best`` / ``audit`` / ``lineage`` 直接扫描 runs，不要求先 build。
所有产品都是派生数据，可以随时重建。
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import json


def _print_table(headers, rows, limit=None):
    rows = [[('' if value is None else str(value)) for value in row] for row in rows]
    if limit is not None:
        rows = rows[:limit]

    def shrink(value, width=52):
        return value if len(value) <= width else value[:width - 3] + '...'

    rendered = [[shrink(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    print('  '.join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print('  '.join('-' * width for width in widths))
    for row in rendered:
        print('  '.join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _query_rows(rows):
    return [[
        row.get('key'),
        row.get('created_at'),
        row.get('status'),
        row.get('model'),
        row.get('dataset'),
        (row.get('best') or {}).get('name') or '-',
        (row.get('best') or {}).get('value') if row.get('best') else '-',
        ','.join(row.get('flags') or []) or '-',
    ] for row in rows]


def _cmd_build(args):
    from dlkit.utils.catalog import collect_catalog, summarize_catalog, write_catalog

    records = collect_catalog(args.runs_root)
    formats = [part.strip().lower() for part in args.formats.split(',') if part.strip()]
    invalid = sorted(set(formats) - {'json', 'md', 'csv'})
    if invalid:
        print('ERROR: unsupported format(s): %s (use json,md,csv)' % ', '.join(invalid))
        return 2
    outputs = write_catalog(records, args.out_dir, runs_root=args.runs_root, formats=formats)
    summary = summarize_catalog(records)
    print('Runs: %d | projects: %d | experiments: %d' % (
        summary['runs'], summary['projects'], summary['experiments']))
    for path in outputs:
        print('Wrote %s' % path)
    return 0


def _cmd_query(args):
    from dlkit.utils.catalog import collect_catalog, filter_records

    records = filter_records(
        collect_catalog(args.runs_root),
        project=args.project,
        experiment=args.experiment,
        status=args.status,
        model=args.model,
        dataset=args.dataset,
        split=args.split,
        tag=args.tag,
        source=args.source,
    )
    if args.json:
        print(json.dumps(records, indent=2, ensure_ascii=False))
        return 0
    if not records:
        print('No runs matched.')
        return 0
    _print_table(
        ['key', 'created', 'status', 'model', 'dataset', 'best', 'value', 'flags'],
        _query_rows(records),
        limit=args.limit,
    )
    if args.limit is not None and len(records) > args.limit:
        print('... %d more (raise --limit or use --json)' % (len(records) - args.limit))
    return 0


def _cmd_best(args):
    from dlkit.utils.catalog import collect_catalog, filter_records, select_best

    records = filter_records(
        collect_catalog(args.runs_root),
        project=args.project,
        experiment=args.experiment,
        model=args.model,
        dataset=args.dataset,
        split=args.split,
        tag=args.tag,
    )
    selected = select_best(records, metric=args.metric, mode=args.mode)
    if args.json:
        print(json.dumps(selected, indent=2, ensure_ascii=False))
        return 0
    if not selected:
        print('No metric found. Pass --metric explicitly.')
        return 0
    rows = [[
        row.get('key'),
        row.get('selected_metric'),
        '%.4f' % float(row.get('selected_value')),
        row.get('selected_source'),
        row.get('created_at'),
    ] for row in selected]
    print('metric: %s | mode: %s' % (selected[0]['selected_metric'], args.mode))
    _print_table(['key', 'metric', 'value', 'from', 'created'], rows)
    return 0


def _cmd_audit(args):
    from dlkit.utils.catalog import audit_catalog, collect_catalog

    result = audit_catalog(collect_catalog(args.runs_root))
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print('Audit summary:')
        for name, count in result['summary'].items():
            print('  %-24s %d' % (name, count))
        for name, keys in result['issues'].items():
            if not keys:
                continue
            print('\n%s:' % name)
            for key in keys[:args.limit]:
                print('  %s' % key)
            if len(keys) > args.limit:
                print('  ... %d more' % (len(keys) - args.limit))
    return 1 if args.strict and any(result['summary'].values()) else 0


def _cmd_lineage(args):
    from dlkit.utils.catalog import build_lineage, collect_catalog

    records = collect_catalog(args.runs_root)
    try:
        rows = build_lineage(records, args.run, direction=args.direction)
    except KeyError as exc:
        print('ERROR: %s' % exc)
        return 1
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0
    print('lineage: %s (%s)' % (args.run, args.direction))
    for row in rows:
        indent = '  ' * int(row.get('depth') or 0)
        print('%s%s  [%s]' % (indent, row.get('key'), row.get('status')))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description='Cross-project run catalog')
    sub = parser.add_subparsers(dest='command')

    def common(child):
        child.add_argument('--runs-root', default='runs', help='runs root (default: runs)')

    build = sub.add_parser('build', help='scan runs and write catalog.json/md/csv')
    common(build)
    build.add_argument('--out-dir', default='reports', help='output dir (default: reports)')
    build.add_argument('--formats', default='json,md,csv', help='comma-separated: json,md,csv')
    build.set_defaults(func=_cmd_build)

    query = sub.add_parser('query', help='filter runs by project/experiment/model/dataset/...')
    common(query)
    query.add_argument('--project')
    query.add_argument('--experiment')
    query.add_argument('--status')
    query.add_argument('--model')
    query.add_argument('--dataset')
    query.add_argument('--split')
    query.add_argument('--tag')
    query.add_argument('--source', choices=['manifest', 'legacy'])
    query.add_argument('--limit', type=int, default=None)
    query.add_argument('--json', action='store_true')
    query.set_defaults(func=_cmd_query)

    best = sub.add_parser('best', help='best run per experiment for one metric')
    common(best)
    best.add_argument('--metric', default=None)
    best.add_argument('--mode', choices=['max', 'min'], default='max')
    best.add_argument('--project')
    best.add_argument('--experiment')
    best.add_argument('--model')
    best.add_argument('--dataset')
    best.add_argument('--split')
    best.add_argument('--tag')
    best.add_argument('--json', action='store_true')
    best.set_defaults(func=_cmd_best)

    audit = sub.add_parser('audit', help='list missing checkpoints/vis/metrics/legacy runs')
    common(audit)
    audit.add_argument('--json', action='store_true')
    audit.add_argument('--strict', action='store_true', help='exit 1 when any issue exists')
    audit.add_argument('--limit', type=int, default=20, help='max keys printed per issue')
    audit.set_defaults(func=_cmd_audit)

    lineage = sub.add_parser('lineage', help='trace init_from/resume_from/parent_run')
    common(lineage)
    lineage.add_argument('--run', required=True, help='run key, run dir, run_id or checkpoint path')
    lineage.add_argument('--direction', choices=['down', 'up'], default='down')
    lineage.add_argument('--json', action='store_true')
    lineage.set_defaults(func=_cmd_lineage)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, 'command', None):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
