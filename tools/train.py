import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

from dlkit.utils.config import load_config, apply_overrides
from dlkit.engine.trainer import Trainer


def build_parser():
    parser = argparse.ArgumentParser(description='Train a model from a config file')
    parser.add_argument('--config', required=True, help='path to yaml config')
    parser.add_argument('--work-dir', default=None, help='override output directory')
    parser.add_argument(
        '--resume',
        nargs='?',
        default=None,
        const='latest',
        help='checkpoint path, or no value / "latest" to continue the latest run of this experiment',
    )
    parser.add_argument('--device', default=None, help='cpu, cuda or auto')
    parser.add_argument('--opts', nargs='*', default=[], help='override config key=value, e.g. train.epochs=10')
    return parser


def main():
    args = build_parser().parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, args.opts)
    # 实验身份:project 来自配置,name 默认取配置文件名,保证 run 目录可读
    cfg.setdefault('name', Path(args.config).stem)
    cfg.setdefault('project', 'default')
    cfg.setdefault('runs_root', 'runs')

    trainer = Trainer(
        cfg,
        work_dir=args.work_dir,
        device=args.device,
        resume_from=args.resume,
        opts=args.opts,
    )
    register_curve_plot(trainer)
    print('Run dir: %s' % trainer.work_dir)
    trainer.train()


def register_curve_plot(trainer):
    """训练结束自动在 run 目录下生成标准训练曲线图。"""

    try:
        import dlkit.viz  # noqa: F401  注册可视化组件
        from dlkit.viz import build_run_context
        from dlkit.registry import VISUALIZERS
    except ImportError as exc:  # 可视化依赖不可用时静默降级
        trainer.logger.warning('visualization unavailable, skip curves: %s' % exc)
        return

    def _plot_after_train(trainer, **_):
        try:
            ctx = build_run_context(
                trainer.work_dir, load_model=False, device=str(trainer.device),
            )
            ctx.history = trainer.metric_history
            ctx.config = trainer.cfg
            ctx.project = str(trainer.cfg.get('project') or 'default')
            ctx.experiment = str(trainer.cfg.get('name') or 'unnamed')
            curves = VISUALIZERS.get('TrainingCurves')
            curves = curves() if isinstance(curves, type) else curves
            for path in curves.render(ctx):
                trainer.logger.info('Saved training curves: %s', path)
        except Exception as exc:  # 曲线失败不影响训练主流程
            trainer.logger.warning('training curves failed: %s' % exc)

    trainer.register_hook('after_train', _plot_after_train)


if __name__ == '__main__':
    main()
