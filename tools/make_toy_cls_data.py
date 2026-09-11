import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CLASS_NAMES = ['background', 'circle', 'square', 'triangle']
CLASS_COLORS = {
    0: [25, 25, 45],
    1: [220, 80, 60],
    2: [60, 180, 90],
    3: [80, 110, 230],
}


def _jitter(color, rng, amount=30):
    return [int(np.clip(c + rng.uniform(-amount, amount), 0, 255)) for c in color]


def make_sample(size, cls, rng):
    """生成一张只含单个形状的图（label=cls），背景是随机噪声。"""
    bg = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
    image = Image.fromarray(bg).filter(ImageFilter.GaussianBlur(1.5))
    draw = ImageDraw.Draw(image)

    if cls != 0:
        margin = int(size * 0.08)
        s = rng.randint(int(size * 0.25), int(size * 0.55))
        x = rng.randint(margin, size - s - margin)
        y = rng.randint(margin, size - s - margin)
        color = tuple(_jitter(CLASS_COLORS[cls], rng))
        if cls == 1:
            draw.ellipse([x, y, x + s, y + s], fill=color)
        elif cls == 2:
            draw.rectangle([x, y, x + s, y + s], fill=color)
        elif cls == 3:
            pts = [(x, y + s), (x + s, y + s), (x + s // 2, y)]
            draw.polygon(pts, fill=color)
    return image


def main():
    parser = argparse.ArgumentParser(description='生成合成分类数据集（ImageFolder 风格）')
    parser.add_argument('--root', default='data/cls_toy')
    parser.add_argument('--samples-per-class', type=int, default=30)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    n_val = max(1, int(args.samples_per_class * args.val_ratio))
    n_train = args.samples_per_class - n_val

    for split, n in [('train', n_train), ('val', n_val)]:
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            cls_dir = os.path.join(args.root, split, cls_name)
            os.makedirs(cls_dir, exist_ok=True)
            for i in range(n):
                image = make_sample(args.size, cls_idx, rng)
                image.save(os.path.join(cls_dir, '%04d.png' % i))

    meta = {
        'names': CLASS_NAMES,
        'colors': {str(k): v for k, v in CLASS_COLORS.items()},
    }
    os.makedirs(args.root, exist_ok=True)
    with open(os.path.join(args.root, 'classes.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print('Generated %d samples/class in %s (train=%d, val=%d per class)'
          % (args.samples_per_class, args.root, n_train, n_val))


if __name__ == '__main__':
    main()
