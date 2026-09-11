import torch

from dlkit.registry import METRICS


@METRICS.register()
class Accuracy:
    """分类 Top-K 准确率。

    update 接收 (logits, batch)，batch['label'] 是 (B,) 的类别索引。
    """

    def __init__(self, topk=(1,)):
        self.topk = tuple(topk)
        self.reset()

    def reset(self):
        self._correct = {k: 0 for k in self.topk}
        self._total = 0
        self._num_classes = None

    def update(self, logits, batch):
        target = batch['label']
        if self._num_classes is None:
            self._num_classes = logits.size(1)
        maxk = min(max(self.topk), logits.size(1))
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        for k in self.topk:
            if k > logits.size(1):
                continue
            self._correct[k] += correct[:k].reshape(-1).float().sum().item()
        self._total += target.numel()

    def compute(self):
        total = max(self._total, 1)
        result = {}
        for k in self.topk:
            if self._num_classes is not None and k > self._num_classes:
                continue
            result['Acc@%d' % k] = float(self._correct[k] / total)
        return result
