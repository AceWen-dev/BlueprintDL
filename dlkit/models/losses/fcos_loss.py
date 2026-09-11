import torch
import torch.nn as nn
import torch.nn.functional as F

from dlkit.registry import LOSSES


def sigmoid_focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Sigmoid focal loss（RetinaNet/FCOS 风格），逐元素。

    pred:  logits，target: 0/1 标签，形状相同。
    """
    p = torch.sigmoid(pred)
    ce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    p_t = p * target + (1.0 - p) * (1.0 - target)
    loss = ce * ((1.0 - p_t) ** gamma)
    if alpha is not None:
        alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
        loss = alpha_t * loss
    return loss


@LOSSES.register()
class FCOSLoss(nn.Module):
    """FCOS 训练损失：目标分配 + 分类 focal loss + 回归 smooth-L1 + 中心度 BCE。

    输入 preds 是模型 forward 的 dict，batch 里是 boxes/labels（list，逐图）。
    目标分配：每个特征点若落在某个 box 内即为正样本（重叠时分配给面积最小的 box），
    回归目标为点到四边的距离 (l,t,r,b)（除以 stride 归一化），
    中心度目标 = sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b))。
    """

    def __init__(self, cls_weight=1.0, reg_weight=1.0, center_weight=1.0,
                 alpha=0.25, gamma=2.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.center_weight = center_weight
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, batch):
        cls_preds = preds['cls']
        reg_preds = preds['reg']
        center_preds = preds['center']
        strides = preds['strides']
        boxes_list = batch['boxes']
        labels_list = batch['labels']

        dev = cls_preds[0].device
        B = cls_preds[0].shape[0]
        C = cls_preds[0].shape[1]

        cls_loss_sum = torch.zeros((), device=dev)
        reg_loss_sum = torch.zeros((), device=dev)
        center_loss_sum = torch.zeros((), device=dev)
        num_pos = torch.zeros((), device=dev, dtype=torch.float32)

        for cls_p, reg_p, cen_p, stride in zip(cls_preds, reg_preds, center_preds, strides):
            _, _, H, W = cls_p.shape
            ys = (torch.arange(H, device=dev, dtype=torch.float32) + 0.5) * stride
            xs = (torch.arange(W, device=dev, dtype=torch.float32) + 0.5) * stride
            pts_x = xs.repeat(H)
            pts_y = ys.repeat_interleave(W)

            for b in range(B):
                boxes = boxes_list[b]
                labels = labels_list[b]

                cls_target = torch.zeros((H * W, C), device=dev)
                reg_target = torch.zeros((H * W, 4), device=dev)
                center_target = torch.zeros((H * W,), device=dev)
                assigned = torch.zeros((H * W,), dtype=torch.bool, device=dev)

                if boxes.numel() > 0:
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    for idx in torch.argsort(areas):  # 小 box 优先，处理重叠
                        x1, y1, x2, y2 = boxes[idx]
                        l = pts_x - x1
                        r = x2 - pts_x
                        t = pts_y - y1
                        bb = y2 - pts_y
                        inside = (l > 0) & (r > 0) & (t > 0) & (bb > 0)
                        free = inside & ~assigned
                        if not free.any():
                            continue
                        assigned[free] = True
                        cls_target[free, labels[idx]] = 1.0
                        # 回归目标用 log 空间（与 decode 时的 exp 对应，保证距离非负）
                        reg_target[free, 0] = torch.log((l[free] / stride).clamp(min=1e-4))
                        reg_target[free, 1] = torch.log((t[free] / stride).clamp(min=1e-4))
                        reg_target[free, 2] = torch.log((r[free] / stride).clamp(min=1e-4))
                        reg_target[free, 3] = torch.log((bb[free] / stride).clamp(min=1e-4))
                        lf, rf = l[free], r[free]
                        tf, bf = t[free], bb[free]
                        center_target[free] = torch.sqrt(
                            (torch.min(lf, rf) / torch.max(lf, rf))
                            * (torch.min(tf, bf) / torch.max(tf, bf))
                        )

                n_pos_b = assigned.sum().float()
                num_pos = num_pos + n_pos_b

                cls_pred = cls_p[b].permute(1, 2, 0).reshape(H * W, C)
                cls_loss_sum = cls_loss_sum + sigmoid_focal_loss(
                    cls_pred, cls_target, self.alpha, self.gamma
                ).sum()

                if assigned.any():
                    reg_pred = reg_p[b].permute(1, 2, 0).reshape(H * W, 4)[assigned]
                    reg_loss_sum = reg_loss_sum + F.smooth_l1_loss(
                        reg_pred, reg_target[assigned], beta=1.0, reduction='sum'
                    )
                    cen_pred = cen_p[b].reshape(-1)[assigned]
                    center_loss_sum = center_loss_sum + F.binary_cross_entropy_with_logits(
                        cen_pred, center_target[assigned], reduction='sum'
                    )

        denom = num_pos.clamp(min=1.0)
        cls_loss = cls_loss_sum / denom
        reg_loss = reg_loss_sum / denom
        center_loss = center_loss_sum / denom
        return self.cls_weight * cls_loss + self.reg_weight * reg_loss + self.center_weight * center_loss
