"""对一次 run 生成标准可视化产物(vis/ 目录)。

用法:
    uv run python tools/exp/visualize.py --run runs/<project>/<experiment>/<ts>
    uv run python tools/exp/visualize.py --run runs/... --visualizer SegOverlay --num-samples 6

默认按 config 快照的 task 自动选择组件:曲线(必画)+ 任务专属可视化。
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
from pathlib import Path

import dlkit.viz  # noqa: F401  完成组件注册
from dlkit.registry import VISUALIZERS
from dlkit.viz.context import build_run_context

# 每个 task 默认运行的组件(task='curves' 只画曲线)
_TASK_VISUALIZERS = {
    'segmentation': ['SegOverlay'],
    'classification': ['ConfusionMatrix'],
    'detection': ['DetBoxes'],
    'depth': ['DepthColormap'],
    'depth_estimation': ['DepthColormap'],
}


def main():
    parser = argparse.ArgumentParser(description='Generate standard visualizations for one run')
    parser.add_argument('--run', required=True, help='run directory containing config/checkpoint')
    parser.add_argument('--device', default=None, help='cpu / cuda')
    parser.add_argument('--visualizer', action='append', default=None,
                        help='visualizer name; repeat to select several (default: task auto)')
    parser.add_argument('--num-samples', type=int, default=4,
                        help='max number of val samples for prediction visualizations')
    args = parser.parse_args()

    run_dir = Path(args.run).resolve()
    if not run_dir.is_dir():
        raise SystemExit('run directory not found: %s' % run_dir)

    # 先轻量读取配置/历史,决定要跑哪些组件;只有任务专属可视化才重建模型与数据
    ctx = build_run_context(run_dir, load_model=False, device=args.device,
                            max_samples=args.num_samples)
    task = str(ctx.config.get('task') or '').strip().lower()
    if args.visualizer:
        selected = list(args.visualizer)
    else:
        selected = ['TrainingCurves'] + _TASK_VISUALIZERS.get(task, [])

    if any(name != 'TrainingCurves' for name in selected):
        try:
            ctx = build_run_context(
                run_dir,
                load_model=True,
                device=args.device,
                max_samples=args.num_samples,
            )
        except Exception as exc:
            print('WARN: full context unavailable (%s); prediction visualizers will be skipped' % exc)

    if not ctx.history:
        print('WARN: no metrics.jsonl in run; curves will be skipped if selected here')

    for name in selected:
        if name not in VISUALIZERS:
            print('WARN: unknown visualizer %r; available: %s'
                  % (name, VISUALIZERS.keys()))
            continue
        component = VISUALIZERS.get(name)
        try:
            instance = component() if isinstance(component, type) else component
            outputs = instance.render(ctx)
            for path in outputs:
                print('Wrote %s' % path)
        except RuntimeError as exc:
            print('SKIP %s: %s' % (name, exc))
        except Exception as exc:  # 单个组件失败不阻塞其他组件
            print('FAIL %s: %r' % (name, exc))


if __name__ == '__main__':
    main()
