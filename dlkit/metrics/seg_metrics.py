import numpy as np

from dlkit.registry import METRICS


@METRICS.register()
class SegMetrics:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, logits, target):
        preds = logits.detach().argmax(dim=1).cpu().numpy().astype(np.int64)
        target = target.detach().cpu().numpy().astype(np.int64)
        if self.ignore_index is None:
            valid = np.ones_like(target, dtype=bool)
        else:
            valid = target != self.ignore_index
        idx = target[valid] * self.num_classes + preds[valid]
        self.cm += np.bincount(idx, minlength=self.num_classes * self.num_classes).reshape(
            self.num_classes, self.num_classes
        )

    def compute(self):
        cm = self.cm
        with np.errstate(divide='ignore', invalid='ignore'):
            union = cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm)
            iou = np.diag(cm) / (union + 1e-10)
            acc = np.diag(cm) / (cm.sum(axis=1) + 1e-10)
            dice = 2.0 * np.diag(cm) / (cm.sum(axis=0) + cm.sum(axis=1) + 1e-10)
        total = cm.sum()
        pix_acc = np.diag(cm).sum() / (total + 1e-10)
        freq = cm.sum(axis=1) / (total + 1e-10)
        fwiou = float(np.nansum(freq * iou))

        return {
            'mIoU': float(np.nanmean(iou)),
            'mDice': float(np.nanmean(dice)),
            'mAcc': float(np.nanmean(acc)),
            'PixAcc': float(pix_acc),
            'FWIoU': fwiou,
            'IoU': [float(v) for v in iou],
            'Dice': [float(v) for v in dice],
            'Acc': [float(v) for v in acc],
        }
