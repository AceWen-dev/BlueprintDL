"""分割任务可视化:原图 / GT / 预测 的三联叠加图。"""

from __future__ import annotations

import matplotlib

matplotlib.use('Agg')

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dlkit.registry import VISUALIZERS
from dlkit.utils.batch import to_device

from .context import RunContext, denormalize_image, resolve_norm_stats

_DEFAULT_CMAP = plt.get_cmap('tab10')


def _default_palette(num_classes: int) -> list[tuple]:
    return [_DEFAULT_CMAP(i % _DEFAULT_CMAP.N) for i in range(max(1, num_classes))]


def color_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """把类别索引图映射成固定调色板的 RGB 图 (uint8)。"""

    palette = _default_palette(num_classes)
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls in range(num_classes):
        color = np.array(palette[cls])[:3]
        color = (color * 255).astype(np.uint8)
        out[mask == cls] = color
    return out


@VISUALIZERS.register(
    tags=('segmentation',), description='val 样本 原图/GT/预测 三联图'
)
class SegOverlay:
    """输出 vis/seg_<i>.png,对少量 val 样本并排展示原图、GT 与预测。"""

    def render(self, ctx: RunContext) -> list[str]:
        if ctx.model is None or ctx.val_loader is None:
            raise RuntimeError('SegOverlay needs model and val_loader in context')

        num_classes = ctx.num_classes or 4
        mean, std = resolve_norm_stats(ctx.config)
        out_paths = []
        count = 0
        for batch in ctx.val_loader:
            batch = to_device(batch, ctx.device)
            images = batch['image']
            masks = batch['mask']
            with torch.no_grad():
                preds = ctx.model(images)
            if isinstance(preds, dict):
                preds = preds.get('seg') or preds.get('pred')
            pred_classes = preds.argmax(dim=1).cpu().numpy()
            gt_classes = masks.cpu().numpy()
            raw_images = (
                images.cpu().numpy().transpose(0, 2, 3, 1)
            )
            raw = denormalize_image(raw_images, mean, std)

            for i in range(images.shape[0]):
                if count >= ctx.max_samples:
                    return out_paths
                fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
                axes[0].imshow(np.clip(raw[i], 0, 1))
                axes[0].set_title('image')
                axes[1].imshow(color_mask(gt_classes[i], num_classes), interpolation='nearest')
                axes[1].set_title('ground truth')
                axes[2].imshow(color_mask(pred_classes[i], num_classes), interpolation='nearest')
                axes[2].set_title('prediction')
                for ax in axes:
                    ax.axis('off')
                fig.tight_layout()
                out_path = Path(ctx.run_dir) / 'vis' / ('seg_%d.png' % count)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_path, dpi=150)
                plt.close(fig)
                out_paths.append(str(out_path))
                count += 1
        return out_paths


__all__ = ['SegOverlay', 'color_mask']
