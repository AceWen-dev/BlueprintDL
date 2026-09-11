import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import dlkit.models  # noqa: F401
from dlkit.registry import build_from_cfg
from dlkit.utils.config import load_config
from dlkit.utils.checkpoint import load_checkpoint
from dlkit.utils.det import decode_fcos
from dlkit.data.dataset import _build_transform

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _load_meta(path):
    names = None
    colors = None
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        names = data.get('names')
        colors = {int(k): v for k, v in (data.get('colors') or {}).items()}
    return names, colors


def preprocess(path, device):
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img)
    t = torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    mean = torch.as_tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (t - mean) / std, arr


def main():
    parser = argparse.ArgumentParser(description='FCOS 目标检测推理 + 可视化')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True, help='图片文件或目录')
    parser.add_argument('--output-dir', default='runs/predict_det')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--score-threshold', type=float, default=0.3)
    parser.add_argument('--nms-threshold', type=float, default=0.5)
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--classes', default=None, help='classes.json 路径')
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = build_from_cfg(cfg['model']).to(args.device).eval()
    load_checkpoint(args.checkpoint, model=model, device=args.device)
    names, colors = _load_meta(args.classes)

    files = []
    if os.path.isdir(args.input):
        files = [
            os.path.join(args.input, f) for f in sorted(os.listdir(args.input))
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
    else:
        files = [args.input]

    os.makedirs(args.output_dir, exist_ok=True)
    for f in files:
        t, original = preprocess(f, args.device)
        with torch.no_grad():
            preds = model(t)
        boxes, scores, labels = decode_fcos(
            [p[0] for p in preds['cls']],
            [p[0] for p in preds['reg']],
            [p[0] for p in preds['center']],
            preds['strides'],
            score_threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,
            topk=args.topk,
        )
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()

        vis = Image.fromarray(original).convert('RGB')
        draw = ImageDraw.Draw(vis)
        for i in range(len(labels)):
            x1, y1, x2, y2 = boxes[i]
            cls = int(labels[i])
            color = tuple(colors[cls]) if colors and cls in colors else (255, 0, 0)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            label = names[cls] if names and cls < len(names) else str(cls)
            draw.text((x1, max(0, y1 - 10)), '%s %.2f' % (label, scores[i]), fill=color)

        stem = os.path.splitext(os.path.basename(f))[0]
        vis.save(os.path.join(args.output_dir, stem + '_det.png'))
        print('%s -> %d boxes' % (f, len(labels)))
        for i in range(len(labels)):
            print('  cls=%d score=%.3f box=[%.1f %.1f %.1f %.1f]'
                  % (labels[i], scores[i], *boxes[i]))


if __name__ == '__main__':
    main()
