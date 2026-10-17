"""跨实验对比 CLI:聚合一个项目下所有 run,生成对比表(md/csv)与训练曲线对比图。

    uv run python tools/exp/compare.py --project examples
    uv run python tools/exp/compare.py --project polyp --metric mIoU

聚合逻辑在 `dlkit/utils/experiments.py`(纯逻辑层,`tools/exp/project.py` 的收尾
报告复用同一实现)。输出到 `projects/<project>/reports/`:

    EXPERIMENTS.md        对比表(按实验分组,最新在前)+ run 目录索引
    EXPERIMENTS.csv       同表 CSV
    curves_<metric>.png   全部 run 的指定指标曲线叠图
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description='Compare runs of one project')
    parser.add_argument('--project', required=True, help='project id directory under runs/')
    parser.add_argument('--runs-root', default='runs', help='runs root (default: runs)')
    parser.add_argument('--out-dir', default=None,
                        help='output dir (default: projects/<project>/reports)')
    parser.add_argument('--metric', default=None,
                        help='metric name to plot (default: common save_best metric, else val_loss)')
    parser.add_argument('--no-plot', action='store_true', help='skip curve image')
    return parser


def main():
    args = build_parser().parse_args()

    from dlkit.utils.experiments import collect_rows, default_metric, write_outputs

    runs_root = Path(args.runs_root).resolve()
    rows = collect_rows(runs_root, args.project)
    if not rows:
        print('No runs found under %s/%s' % (runs_root, args.project))
        return 1

    out_dir = Path(args.out_dir or (Path('projects') / args.project / 'reports')).resolve()
    metric = None if args.no_plot else (args.metric or default_metric(rows))
    outputs = write_outputs(rows, args.project, out_dir, metric)
    print('Runs: %d | metric: %s' % (len(rows), metric or '-'))
    for path in outputs:
        print('Wrote %s' % path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
