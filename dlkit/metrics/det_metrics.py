import numpy as np

from dlkit.registry import METRICS
from dlkit.utils.det import decode_fcos, box_iou_np


@METRICS.register()
class DetMetrics:
    """检测 mAP（Pascal VOC 风格 mAP@0.5，11 点插值）。

    update 接收模型输出的 dict 和 batch，内部完成 FCOS 解码 + NMS，
    再与真值按 IoU 贪心匹配统计各类 AP。
    """

    def __init__(self, num_classes=None, iou_threshold=0.5, score_threshold=0.05,
                 nms_threshold=0.5, topk=100):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.topk = topk
        self.reset()

    def reset(self):
        self._gts = {}   # cls -> list of (4,) ndarray
        self._dets = {}  # cls -> (list of scores, list of (4,) ndarray)

    def update(self, preds, batch):
        cls_preds = preds['cls']
        reg_preds = preds['reg']
        center_preds = preds['center']
        strides = preds['strides']
        boxes_list = batch['boxes']
        labels_list = batch['labels']

        n_levels = len(cls_preds)
        for b in range(len(boxes_list)):
            cls_b = [cls_preds[l][b] for l in range(n_levels)]
            reg_b = [reg_preds[l][b] for l in range(n_levels)]
            cen_b = [center_preds[l][b] for l in range(n_levels)]
            boxes, scores, labels = decode_fcos(
                cls_b, reg_b, cen_b, strides,
                self.score_threshold, self.nms_threshold, self.topk,
            )
            boxes = boxes.cpu().numpy()
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()

            for i in range(len(labels)):
                cls = int(labels[i])
                entry = self._dets.setdefault(cls, ([], []))
                entry[0].append(float(scores[i]))
                entry[1].append(boxes[i])

            gt_boxes = boxes_list[b].cpu().numpy()
            gt_labels = labels_list[b].cpu().numpy()
            for i in range(len(gt_labels)):
                cls = int(gt_labels[i])
                self._gts.setdefault(cls, []).append(gt_boxes[i])

    def _ap(self, gts, det_scores, det_boxes):
        order = np.argsort(-np.asarray(det_scores))
        det_boxes = [det_boxes[i] for i in order]
        matched = [False] * len(gts)

        tp = np.zeros(len(det_boxes), dtype=np.float64)
        fp = np.zeros(len(det_boxes), dtype=np.float64)
        for i, db in enumerate(det_boxes):
            best_iou, best_j = 0.0, -1
            for j, gb in enumerate(gts):
                if matched[j]:
                    continue
                iou = box_iou_np(db, gb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou >= self.iou_threshold:
                matched[best_j] = True
                tp[i] = 1.0
            else:
                fp[i] = 1.0

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recalls = tp / max(len(gts), 1)
        precisions = tp / np.maximum(tp + fp, 1e-9)

        ap = 0.0
        for t in np.linspace(0.0, 1.0, 11):
            p = precisions[recalls >= t].max() if (recalls >= t).any() else 0.0
            ap += p / 11.0
        return float(ap)

    def compute(self):
        classes = sorted(set(self._gts) | set(self._dets))
        aps = []
        for cls in classes:
            gts = self._gts.get(cls, [])
            scores, boxes = self._dets.get(cls, ([], []))
            aps.append(self._ap(gts, scores, boxes))
        mAP = float(np.mean(aps)) if aps else 0.0
        return {'mAP@0.5': mAP}
