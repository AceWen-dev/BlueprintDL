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
    return [
        int(np.clip(c + rng.uniform(-amount, amount), 0, 255)) for c in color
    ]


def _draw_shape(draw, mask_draw, cls, size, rng):
    margin = int(size * 0.05)
    s = rng.randint(int(size * 0.12), int(size * 0.35))
    x = rng.randint(margin, size - s - margin)
    y = rng.randint(margin, size - s - margin)
    color = tuple(_jitter(CLASS_COLORS[cls], rng))
    if cls == 1:
        draw.ellipse([x, y, x + s, y + s], fill=color)
        mask_draw.ellipse([x, y, x + s, y + s], fill=cls)
    elif cls == 2:
        draw.rectangle([x, y, x + s, y + s], fill=color)
        mask_draw.rectangle([x, y, x + s, y + s], fill=cls)
    elif cls == 3:
        pts = [(x, y + s), (x + s, y + s), (x + s // 2, y)]
        draw.polygon(pts, fill=color)
        mask_draw.polygon(pts, fill=cls)


def make_sample(size, rng):
    bg = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
    image = Image.fromarray(bg).filter(ImageFilter.GaussianBlur(1.5))
    mask = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)
    for _ in range(rng.randint(1, 4)):
        cls = rng.randint(1, 3)
        _draw_shape(draw, mask_draw, cls, size, rng)
    return image, mask


def main():
    parser = argparse.ArgumentParser(description='Generate a synthetic toy segmentation dataset')
    parser.add_argument('--root', default='data/toy')
    parser.add_argument('--samples', type=int, default=120)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--size', type=int, default=160)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    splits = {'train': [], 'val': []}
    n_val = max(1, int(args.samples * args.val_ratio))
    for i in range(args.samples):
        split = 'val' if i < n_val else 'train'
        splits[split].append(i)

    for split, indices in splits.items():
        image_dir = os.path.join(args.root, split, 'images')
        mask_dir = os.path.join(args.root, split, 'masks')
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        for i in indices:
            image, mask = make_sample(args.size, rng)
            image.save(os.path.join(image_dir, '%04d.png' % i))
            mask.save(os.path.join(mask_dir, '%04d.png' % i))

    meta = {
        'names': CLASS_NAMES,
        'colors': {str(k): v for k, v in CLASS_COLORS.items()},
    }
    with open(os.path.join(args.root, 'classes.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print('Generated %d samples in %s (train=%d, val=%d)'
          % (args.samples, args.root, len(splits['train']), len(splits['val'])))


if __name__ == '__main__':
    main()
