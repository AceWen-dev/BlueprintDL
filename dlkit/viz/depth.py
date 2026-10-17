"""深度估计可视化:GT / 预测 / 绝对误差 的 colormap 图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from dlkit.registry import VISUALIZERS
from dlkit.utils.batch import to_device

from .context import RunContext, denormalize_image, resolve_norm_stats

_CMAP = plt.get_cmap('viridis')


@VISUALIZERS.register(tags=('depth',), description='val 样本 GT/预测/误差深度图')
class DepthColormap:
    """输出 vis/depth_<i>.png。GT 深度(米)约定在 batch['mask'],0 视为无效像素。

    深度模型 forward 返回 log 深度,predict_depth 约定 exp(forward)。
    """

    def render(self, ctx: RunContext) -> list[str]:
        if ctx.model is None or ctx.val_loader is None:
            raise RuntimeError('DepthColormap needs model and val_loader in context')

        mean, std = resolve_norm_stats(ctx.config)
        out_paths = []
        count = 0
        for batch in ctx.val_loader:
            batch = to_device(batch, ctx.device)
            images = batch['image']
            gt = batch['mask'].float()
            with torch.no_grad():
                preds = ctx.model(images)
            pred = torch.exp(preds) if preds.shape[1] == 1 else torch.exp(preds[:, :1])
            err = (pred - gt.unsqueeze(1)).abs()

            inter = float(
                torch.nanquantile(pred[(gt > 0) & torch.isfinite(pred)].float(), 0.98).item()
            ) if bool(((gt > 0) & torch.isfinite(pred)).any()) else 0.0
            vmax = max(inter, 1e-6)

            pred_np = pred.squeeze(1).cpu().numpy()
            gt_np = gt.cpu().numpy()
            err_np = err.squeeze(1).cpu().numpy()
            raw = denormalize_image(images.cpu().numpy().transpose(0, 2, 3, 1), mean, std)

            for i in range(images.shape[0]):
                if count >= ctx.max_samples:
                    return out_paths
                figure, axes = plt.subplots(1, 4, figsize=(16, 4.2))
                panel(axes[0], np.clip(raw[i], 0, 1), None, 'image')
                panel(axes[1], gt_np[i], vmax, 'ground truth (m)',
                      cmap=_CMAP, interpolation='nearest')
                panel(axes[2], np.clip(pred_np[i], 0, vmax), vmax, 'prediction (m)',
                      cmap=_CMAP, interpolation='nearest')
                panel(axes[3], np.clip(err_np[i], 0, vmax * 0.25), vmax * 0.25,
                      '|error| (m)', cmap='magma', interpolation='nearest')
                for ax in axes:
                    ax.axis('off')
                figure.tight_layout()
                out_path = Path(ctx.run_dir) / 'vis' / ('depth_%d.png' % count)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                figure.savefig(out_path, dpi=150)
                plt.close(figure)
                out_paths.append(str(out_path))
                count += 1
        return out_paths


def panel(ax, data, vmax, title, **kwargs):
    if vmax is not None:
        kwargs.setdefault('vmin', 0)
        kwargs.setdefault('vmax', vmax)
    ax.imshow(np.squeeze(data), **kwargs)
    ax.set_title(title, fontsize=10)


__all__ = ['DepthColormap']
