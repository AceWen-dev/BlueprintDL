import argparse
import os
import random

import numpy as np
from PIL import Image


def make_scene(size, rng, shape_range=(1, 5), depth_range=(0.5, 6.0)):
    """生成一张合成场景：背景是径向渐变深度，上面叠几个已知深度的形状。"""
    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = size / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (size / 2)

    depth = 1.5 + 6.5 * r ** 1.5 + np.random.uniform(-0.1, 0.1, (size, size))
    image = np.zeros((size, size, 3), dtype=np.float32)
    base = np.array([120.0, 150.0, 180.0])
    image[:] = base * np.clip(1.0 - r, 0.15, 1.0)[..., None]

    objs = []
    for _ in range(rng.randint(*shape_range)):
        d = rng.uniform(*depth_range)
        kind = rng.choice(['circle', 'rect'])
        ox = rng.randint(0, size)
        oy = rng.randint(0, size)
        rad = rng.randint(size // 8, size // 4)
        objs.append((d, kind, ox, oy, rad))
    objs.sort(key=lambda o: -o[0])  # 远的先画，近的盖在上面（遮挡一致）

    for d, kind, ox, oy, rad in objs:
        color = np.array([rng.randint(60, 230) for _ in range(3)], dtype=np.float32)
        color = color * np.clip(1.2 - d * 0.12, 0.35, 1.1)
        if kind == 'circle':
            mask = (xx - ox) ** 2 + (yy - oy) ** 2 < rad ** 2
        else:
            half = rad // 2
            mask = (np.abs(xx - ox) < rad) & (np.abs(yy - oy) < half)
        image[mask] = color
        depth[mask] = d

    image = np.clip(image + np.random.normal(0, 6, image.shape), 0, 255).astype(np.uint8)
    return image, depth


def save_depth(depth, path, fmt):
    if fmt == 'png':
        arr = np.clip(depth, 0, 65.0)
        Image.fromarray((arr * 1000.0).astype(np.uint16)).save(path)
    else:
        np.save(path, depth.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description='生成合成 RGB+深度 数据（YOLO26-Depth 格式）')
    parser.add_argument('--root', default='data/depth_toy')
    parser.add_argument('--samples', type=int, default=120)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--size', type=int, default=192)
    parser.add_argument('--format', default='png', choices=['png', 'npy'],
                        help='深度图格式：png=uint16毫米(官方默认) / npy=float32米')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    n_val = max(1, int(args.samples * args.val_ratio))
    n_train = args.samples - n_val

    for split, n in [('train', n_train), ('val', n_val)]:
        img_dir = os.path.join(args.root, split, 'images')
        dep_dir = os.path.join(args.root, split, 'depth')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(dep_dir, exist_ok=True)
        for i in range(n):
            image, depth = make_scene(args.size, rng)
            name = '%04d' % i
            Image.fromarray(image).save(os.path.join(img_dir, name + '.png'))
            save_depth(depth, os.path.join(dep_dir, name + ('.png' if args.format == 'png' else '.npy')), args.format)
        print('生成 %s: %d 对' % (split, n))

    scale_note = '0.001（uint16 PNG，毫米）' if args.format == 'png' else '1.0（float32 NPY，米）'
    print('深度图格式: %s -> 配置里 depth_scale 写 %s' % (args.format, scale_note))


if __name__ == '__main__':
    main()
