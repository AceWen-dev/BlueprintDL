"""检测通用算子：IoU / NMS / FCOS 预测解码。

box 统一使用 [x1, y1, x2, y2] 的绝对像素坐标（浮点），
方便在数据集、损失、后处理之间流转。
"""

import numpy as np
import torch


def box_iou(boxes1, boxes2):
    """计算两组 box 的两两 IoU。

    boxes1: (N, 4), boxes2: (M, 4) -> (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6)


def box_iou_np(boxes1, boxes2):
    """numpy 版 box IoU（用于指标统计，避免跨设备搬运）。"""
    area1 = (boxes1[2] - boxes1[0]) * (boxes1[3] - boxes1[1])
    area2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])
    lt = np.maximum(boxes1[:2], boxes2[:2])
    rb = np.minimum(boxes1[2:], boxes2[2:])
    wh = np.maximum(rb - lt, 0)
    inter = wh[0] * wh[1]
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def nms(boxes, scores, iou_threshold):
    """非极大值抑制。boxes (N,4), scores (N,) -> 保留索引 (K,)。"""
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def per_class_nms(boxes, scores, labels, iou_threshold):
    """按类别分别做 NMS，返回全局保留索引。"""
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    keep = []
    for cls in labels.unique():
        m = labels == cls
        idx = torch.nonzero(m, as_tuple=False).squeeze(1)
        k = nms(boxes[idx], scores[idx], iou_threshold)
        keep.append(idx[k])
    return torch.cat(keep) if keep else torch.zeros(0, dtype=torch.long, device=boxes.device)


def decode_fcos(cls_preds, reg_preds, center_preds, strides,
                score_threshold=0.05, nms_threshold=0.5, topk=100):
    """把 FCOS 的多层预测解码成最终检测框（单张图）。

    cls_preds / reg_preds / center_preds: 每层一个张量，单图形状分别为
    (C,H,W) / (4,H,W) / (1,H,W)。
    返回 (boxes (M,4), scores (M,), labels (M,))，已做 NMS + topk。
    """
    all_boxes, all_scores, all_labels = [], [], []
    for cls_p, reg_p, cen_p, stride in zip(cls_preds, reg_preds, center_preds, strides):
        C, H, W = cls_p.shape
        ys = (torch.arange(H, device=cls_p.device, dtype=torch.float32) + 0.5) * stride
        xs = (torch.arange(W, device=cls_p.device, dtype=torch.float32) + 0.5) * stride

        probs = cls_p.sigmoid()                     # (C,H,W)
        center = cen_p.sigmoid().squeeze(0)          # (H,W)
        max_score, max_cls = probs.max(dim=0)        # (H,W)
        mask = max_score > score_threshold
        if not mask.any():
            continue

        score = (max_score[mask] * center[mask]).sqrt()
        labels = max_cls[mask]

        # reg 分支输出在 log 空间，exp 后得到非负的距离（FCOS 标准做法）
        ltrb = reg_p.permute(1, 2, 0)[mask].exp() * stride  # (K,4)
        y = ys.view(H, 1).expand(H, W)[mask]
        x = xs.view(1, W).expand(H, W)[mask]
        l, t, r, b = ltrb[:, 0], ltrb[:, 1], ltrb[:, 2], ltrb[:, 3]
        boxes = torch.stack([x - l, y - t, x + r, y + b], dim=1)  # [x1,y1,x2,y2]

        all_boxes.append(boxes)
        all_scores.append(score)
        all_labels.append(labels)

    if not all_boxes:
        dev = cls_preds[0].device
        return (
            torch.zeros((0, 4), device=dev),
            torch.zeros((0,), device=dev),
            torch.zeros((0,), dtype=torch.long, device=dev),
        )

    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)

    keep = per_class_nms(boxes, scores, labels, nms_threshold)
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if len(scores) > topk:
        idx = scores.topk(topk).indices
        boxes, scores, labels = boxes[idx], scores[idx], labels[idx]
    return boxes, scores, labels
