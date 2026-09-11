import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CLASS_NAMES = ['circle', 'square', 'triangle']
CLASS_COLORS = {
    0: [220, 80, 60],
    1: [60, 180, 90],
    2: [80, 110, 230],
}


def _jitter(color, rng, amount=30):
    return [int(np.clip(c + rng.uniform(-amount, amount), 0, 255)) for c in color]


def make_sample(size, rng):
    """生成一张含若干形状的图，返回 (image, boxes, labels)。"""
    bg = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
    image = Image.fromarray(bg).filter(ImageFilter.GaussianBlur(1.5))
    draw = ImageDraw.Draw(image)

    boxes, labels = [], []
    for _ in range(rng.randint(1, 4)):
        cls = rng.randint(0, 2)
        margin = int(size * 0.05)
        s = rng.randint(int(size * 0.12), int(size * 0.32))
        x = rng.randint(margin, size - s - margin)
        y = rng.randint(margin, size - s - margin)
        color = tuple(_jitter(CLASS_COLORS[cls], rng))
        if cls == 0:
            draw.ellipse([x, y, x + s, y + s], fill=color)
        elif cls == 1:
            draw.rectangle([x, y, x + s, y + s], fill=color)
        else:
            pts = [(x, y + s), (x + s, y + s), (x + s // 2, y)]
            draw.polygon(pts, fill=color)
        boxes.append([x, y, x + s, y + s])
        labels.append(cls)
    return image, boxes, labels


def main():
    parser = argparse.ArgumentParser(description='生成合成检测数据集（图像 + txt 标注）')
    parser.add_argument('--root', default='data/det_toy')
    parser.add_argument('--samples', type=int, default=120)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    n_val = max(1, int(args.samples * args.val_ratio))
    n_train = args.samples - n_val

    for split, n in [('train', n_train), ('val', n_val)]:
        img_dir = os.path.join(args.root, split, 'images')
        lbl_dir = os.path.join(args.root, split, 'labels')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for i in range(n):
            image, boxes, labels = make_sample(args.size, rng)
            name = '%04d' % i
            image.save(os.path.join(img_dir, name + '.png'))
            with open(os.path.join(lbl_dir, name + '.txt'), 'w', encoding='utf-8') as f:
                for cls, (x1, y1, x2, y2) in zip(labels, boxes):
                    f.write('%d %d %d %d %d\n' % (cls, x1, y1, x2, y2))
        print('生成 %s: %d 张' % (split, n))

    meta = {
        'names': CLASS_NAMES,
        'colors': {str(k): v for k, v in CLASS_COLORS.items()},
    }
    with open(os.path.join(args.root, 'classes.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print('Generated %d detection samples in %s (train=%d, val=%d)'
          % (args.samples, args.root, n_train, n_val))


if __name__ == '__main__':
    main()
