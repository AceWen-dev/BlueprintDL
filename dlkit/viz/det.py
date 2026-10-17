"""检测任务可视化:GT 框(绿色)与预测框(红色)对比图。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from dlkit.registry import VISUALIZERS
from dlkit.utils.batch import to_device
from dlkit.utils.det import decode_fcos

from .context import RunContext, resolve_norm_stats

GT_COLOR = (0, 200, 0)
PRED_COLOR = (230, 30, 30)


@VISUALIZERS.register(tags=('detection',), description='val 样本 GT/预测框对比')
class DetBoxes:
    """输出 vis/det_<i>.png,decode 流程与 tools/predict_det.py 保持一致。"""

    def render(self, ctx: RunContext) -> list[str]:
        if ctx.model is None or ctx.val_loader is None:
            raise RuntimeError('DetBoxes needs model and val_loader in context')

        mean, std = resolve_norm_stats(ctx.config)
        out_paths = []
        count = 0
        for batch in ctx.val_loader:
            batch = to_device(batch, ctx.device)
            images = batch['image']
            gt_boxes = batch['boxes']
            gt_labels = batch['labels']
            with torch.no_grad():
                preds = ctx.model(images)

            for i in range(images.shape[0]):
                if count >= ctx.max_samples:
                    return out_paths
                boxes, scores, labels = decode_fcos(
                    [p[i] for p in preds['cls']],
                    [p[i] for p in preds['reg']],
                    [p[i] for p in preds['center']],
                    preds['strides'],
                    score_threshold=0.3,
                    nms_threshold=0.5,
                )
                out_path = Path(ctx.run_dir) / 'vis' / ('det_%d.png' % count)
                _save_overlay(
                    images[i], gt_boxes[i], gt_labels[i], boxes, scores, labels,
                    mean, std, out_path,
                )
                out_paths.append(str(out_path))
                count += 1
        return out_paths


def _save_overlay(image, gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels,
                  mean, std, out_path: Path):
    rgb = np.clip(denormalize(image.cpu().numpy(), mean, std) * 255.0, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(rgb).convert('RGB')
    draw = ImageDraw.Draw(canvas)
    for box, _label in zip(np.asarray(gt_boxes).reshape(-1, 4), np.asarray(gt_labels).reshape(-1)):
        draw.rectangle([box[0], box[1], box[2], box[3]], outline=GT_COLOR, width=2)
    boxes = pred_boxes.cpu().numpy()
    scores = pred_scores.cpu().numpy()
    for i in range(len(scores)):
        x1, y1, x2, y2 = boxes[i]
        draw.rectangle([x1, y1, x2, y2], outline=PRED_COLOR, width=2)
        draw.text((x1, max(0, y1 - 10)), '%.2f' % scores[i], fill=PRED_COLOR)
    draw.rectangle([0, 0, 92, 30], fill=(255, 255, 255), outline=(200, 200, 200))
    draw.rectangle([4, 4, 16, 12], outline=GT_COLOR, width=3)
    draw.text((20, 4), 'GT', fill=(0, 0, 0))
    draw.rectangle([4, 16, 16, 24], outline=PRED_COLOR, width=3)
    draw.text((20, 16), 'pred', fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def denormalize(image: np.ndarray, mean, std) -> np.ndarray:
    """CHW -> HWC 并去归一化到 [0,1]。"""

    image = np.transpose(image, (1, 2, 0))
    return np.clip(image * np.asarray(std) + np.asarray(mean), 0.0, 1.0)


__all__ = ['DetBoxes']
