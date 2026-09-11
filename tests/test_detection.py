import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import dlkit.models  # noqa: F401
import dlkit.data    # noqa: F401
import dlkit.metrics  # noqa: F401
from dlkit.registry import build_from_cfg
from dlkit.models.losses.fcos_loss import FCOSLoss
from dlkit.utils.det import nms, box_iou, decode_fcos, per_class_nms


def _build_fcos(num_classes=3):
    return build_from_cfg({
        'type': 'FCOS',
        'params': {
            'num_classes': num_classes,
            'backbone': {'type': 'SimpleCNN', 'params': {'channels': [16, 32, 64, 128]}},
        },
    })


def test_fcos_forward():
    if not HAS_TORCH:
        return
    model = _build_fcos()
    model.eval()
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert set(out) == {'cls', 'reg', 'center', 'strides'}
    assert len(out['cls']) == 5
    assert out['cls'][0].shape[1] == 3
    assert len(out['strides']) == 5


def test_fcos_loss_scalar_and_grad():
    if not HAS_TORCH:
        return
    model = _build_fcos()
    model.train()
    x = torch.randn(1, 3, 128, 128)
    preds = model(x)
    batch = {
        'boxes': [torch.tensor([[10., 10., 50., 40.]])],
        'labels': [torch.tensor([1])],
    }
    loss = FCOSLoss()(preds, batch)
    assert loss.dim() == 0, loss.shape
    loss.backward()
    assert model.backbone.stages[0][0].block[0].weight.grad is not None


def test_box_iou_and_nms():
    if not HAS_TORCH:
        return
    boxes = torch.tensor([[0., 0., 10., 10.], [5., 5., 15., 15.], [100., 100., 110., 110.]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    iou = box_iou(boxes[:1], boxes[:2])
    # 前两个框 IoU = 25 / 175 = 0.1429
    assert abs(float(iou[0, 1]) - 25.0 / 175.0) < 1e-4
    # threshold 0.1：框 1 与框 0 的 IoU 0.1429 > 0.1 被抑制，保留 0 和 2
    keep = nms(boxes, scores, 0.1)
    assert set(keep.tolist()) == {0, 2}, keep


def test_decode_fcos_recovers_box():
    if not HAS_TORCH:
        return
    # 构造单层预测，让某个位置预测出一个已知 box
    H = W = 4
    stride = 4
    # 位置 (1,1) 中心点 = (6,6)；l=6,r=6,t=4,b=4 -> box [0,2,12,10]
    cls_p = torch.full((1, H, W), -10.0)  # C=1，低分背景
    cls_p[0, 1, 1] = 5.0                   # 仅该位置高分
    reg_p = torch.zeros((4, H, W))
    reg_p[0, 1, 1] = np.log(6.0 / stride)
    reg_p[1, 1, 1] = np.log(4.0 / stride)
    reg_p[2, 1, 1] = np.log(6.0 / stride)
    reg_p[3, 1, 1] = np.log(4.0 / stride)
    cen_p = torch.ones((1, H, W)) * 5.0

    boxes, scores, labels = decode_fcos([cls_p], [reg_p], [cen_p], [stride], score_threshold=0.05)
    assert len(boxes) == 1, boxes
    assert labels[0] == 0
    x1, y1, x2, y2 = boxes[0].tolist()
    assert abs(x1 - 0.0) < 1.0 and abs(y1 - 2.0) < 1.0, boxes[0]
    assert abs(x2 - 12.0) < 1.0 and abs(y2 - 10.0) < 1.0, boxes[0]


def test_detection_dataset(tmpdir=None):
    if not HAS_TORCH:
        return
    import tempfile
    from PIL import Image
    from dlkit.data.detection_dataset import DetectionDataset

    with tempfile.TemporaryDirectory() as root:
        img_dir = os.path.join(root, 'images')
        lbl_dir = os.path.join(root, 'labels')
        os.makedirs(img_dir)
        os.makedirs(lbl_dir)
        for i in range(2):
            arr = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(img_dir, '%d.png' % i))
            with open(os.path.join(lbl_dir, '%d.txt' % i), 'w') as f:
                f.write('0 4 4 12 12\n1 16 16 24 24\n')

        ds = DetectionDataset(img_dir, lbl_dir)
        assert len(ds) == 2
        sample = ds[0]
        assert sample['boxes'].shape == (2, 4)
        assert sample['labels'].shape == (2,)

        batch = DetectionDataset.collate_fn([ds[0], ds[1]])
        assert batch['image'].shape == (2, 3, 32, 32)
        assert len(batch['boxes']) == 2
        assert batch['labels'][0].dtype == torch.int64


def run_all():
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print('PASS %s' % name)
            except Exception as e:
                failed += 1
                print('FAIL %s: %s' % (name, e))
    print('done, %d failed' % failed)
    return failed


if __name__ == '__main__':
    sys.exit(run_all())
