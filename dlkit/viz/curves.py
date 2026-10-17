"""训练曲线可视化:val_loss / 各评价指标随 epoch 的标准曲线图。"""

from __future__ import annotations

import matplotlib

matplotlib.use('Agg')  # 无窗口环境统一后端

import os
from pathlib import Path

import matplotlib.pyplot as plt

from dlkit.registry import VISUALIZERS

from .context import RunContext


def plot_training_curves(history, output_path: str | Path) -> str:
    """把 (逐 epoch 指标字典) 的历史画成标准曲线图,返回输出路径。

    - 训练损失 train_loss 与验证损失 val_loss 画在同一张子图;
    - 其余每个 metric 一张子图;
    - history 为空时不画。
    """

    if not history:
        raise ValueError('history is empty, nothing to plot')

    epochs = [row.get('epoch', i) for i, row in enumerate(history)]
    metric_keys = sorted(
        {
            key
            for row in history
            for key, value in row.items()
            if key not in ('epoch', 'train_loss', 'val_loss')
            and isinstance(value, (int, float))
        }
    )
    n_cols = min(2, max(1, len(metric_keys) + 1))
    n_subplots = len(metric_keys) + 1
    n_rows = (n_subplots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.6 * n_rows), squeeze=False)
    axes = [ax for row in axes for ax in row]

    ax = axes[0]
    ax.plot(epochs, _column(history, 'train_loss'), label='train_loss', color='tab:orange')
    ax.plot(epochs, _column(history, 'val_loss'), label='val_loss', color='tab:blue')
    ax.set_title('loss')
    ax.set_xlabel('epoch')
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=8)

    for index, key in enumerate(metric_keys, start=1):
        ax = axes[index]
        ax.plot(epochs, _column(history, key), color='tab:green')
        ax.set_title(key)
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.3)

    for index in range(n_subplots, len(axes)):
        axes[index].set_visible(False)

    fig.suptitle('Training curves', fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def _column(history, key):
    return [float(row[key]) for row in history if isinstance(row.get(key), (int, float))]


@VISUALIZERS.register(tags=('curves', 'all'), description='训练/验证损失与指标曲线')
class TrainingCurves:
    """run 内训练曲线渲染(不依赖模型与数据)。"""

    def render(self, ctx: RunContext) -> list[str]:
        out_dir = Path(ctx.run_dir) / 'vis'
        out_path = out_dir / 'curves.png'
        return [plot_training_curves(ctx.history, out_path)]


__all__ = ['TrainingCurves', 'plot_training_curves']
