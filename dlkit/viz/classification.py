"""分类任务可视化:混淆矩阵热力图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from dlkit.registry import VISUALIZERS
from dlkit.utils.batch import to_device

from .context import RunContext


@VISUALIZERS.register(tags=('classification',), description='val 集混淆矩阵')
class ConfusionMatrix:
    """输出 vis/confusion_matrix.png。label 键遵循分类协议 batch['label']。"""

    def render(self, ctx: RunContext) -> list[str]:
        if ctx.model is None or ctx.val_loader is None:
            raise RuntimeError('ConfusionMatrix needs model and val_loader in context')

        all_pred: list[int] = []
        all_gt: list[int] = []
        for batch in ctx.val_loader:
            batch = to_device(batch, ctx.device)
            with torch.no_grad():
                preds = ctx.model(batch['image'])
            logits = preds if not isinstance(preds, dict) else preds.get('logits')
            if logits is None:
                raise RuntimeError('ConfusionMatrix: model output has no logits')
            all_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            all_gt.extend(batch['label'].detach().cpu().numpy().tolist())

        if not all_gt:
            raise RuntimeError('ConfusionMatrix found no samples')

        guessed = max(max(all_pred), max(all_gt)) + 1
        num_classes = max(ctx.num_classes or 0, guessed)
        matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        for pred, gt in zip(all_pred, all_gt):
            matrix[int(gt), int(pred)] += 1

        figure, ax = plt.subplots(figsize=(6.5, 5.6))
        with np.errstate(invalid='ignore', divide='ignore'):
            row_norm = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
        image = ax.imshow(row_norm, cmap='Blues', vmin=0, vmax=1)
        figure.colorbar(image, ax=ax, fraction=0.046)
        ax.set_xlabel('predicted')
        ax.set_ylabel('ground truth')
        ax.set_title('confusion matrix (row-normalized)')
        ticks = list(range(matrix.shape[0]))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        for i in ticks:
            for j in ticks:
                value = int(matrix[i, j])
                if value:
                    ax.text(j, i, str(value), ha='center', va='center', fontsize=7,
                            color='white' if row_norm[i, j] > 0.5 else 'black')
        figure.tight_layout()
        out_path = Path(ctx.run_dir) / 'vis' / 'confusion_matrix.png'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out_path, dpi=150)
        plt.close(figure)
        return [str(out_path)]


__all__ = ['ConfusionMatrix']
