import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np
import torch
from PIL import Image

import dlkit.models  # noqa: F401  触发模型注册
from dlkit.registry import build_from_cfg
from dlkit.utils.config import load_config
from dlkit.utils.depth_calibration import median_scale_align
from dlkit.visualize.depth import colorize_depth, overlay_depth
from dlkit.metrics.depth_metrics import DepthMetrics

_IMG_SUFFIX = ('.png', '.jpg', '.jpeg', '.bmp')


def _list_frames(input_dir):
    files = sorted(
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if f.lower().endswith(_IMG_SUFFIX)
    )
    if not files:
        raise SystemExit('输入目录里没有图片帧: %s' % input_dir)
    return files


def load_frames(input_dir, size=None):
    """读帧序列 -> (N,3,H,W) float32 [0,1] 张量 + 文件名列表。"""
    files = _list_frames(input_dir)
    frames = []
    for f in files:
        img = Image.open(f).convert('RGB')
        if size is not None:
            img = img.resize((size, size), Image.BILINEAR)
        frames.append(np.asarray(img, dtype=np.float32) / 255.0)
    arr = np.stack(frames, axis=0)  # (N,H,W,3)
    return torch.from_numpy(arr).permute(0, 3, 1, 2), files


def main():
    parser = argparse.ArgumentParser(description='ColonCrafter 结肠镜视频深度估计（框架封装）')
    parser.add_argument('--config', required=True, help='配置（含 model: {type: ColonCrafter}）')
    parser.add_argument('--input', required=True, help='帧序列目录（按文件名排序）')
    parser.add_argument('--output-dir', default='runs/predict_coloncrafter')
    parser.add_argument('--size', type=int, default=None, help='输入尺寸（正方形，建议 16 的倍数）')
    parser.add_argument('--num-inference-steps', type=int, default=1)
    parser.add_argument('--window-size', type=int, default=16)
    parser.add_argument('--overlap', type=int, default=8)
    parser.add_argument('--guidance-scale', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gt-dir', default=None,
                        help='真值深度目录（如 C3VD 的 depth/*.tiff），用于尺度对齐与评估')
    parser.add_argument('--gt-scale', type=float, default=100.0 / 65535.0 / 1000.0,
                        help='真值缩放系数：value * gt-scale = 米。C3VD 为 uint16 且 0-100mm 线性映射到 '
                             '0-65535，故默认 100/65535/1000 ≈ 1.526e-6')
    parser.add_argument('--scale', type=float, default=None, help='手动全局尺度系数（无真值时把相对深度转米）')
    parser.add_argument('--cmap', default='jet', choices=['jet', 'hot', 'gray'])
    parser.add_argument('--mode', default='metric', choices=['metric', 'disparity'])
    args = parser.parse_args()

    cfg = load_config(args.config)
    infer_cfg = cfg.get('infer', {})
    model = build_from_cfg(cfg['model'])

    video, files = load_frames(args.input, size=args.size or infer_cfg.get('size'))
    print('输入 %d 帧, 形状 %s' % (len(files), tuple(video.shape)))

    with torch.no_grad():
        depth = model(
            video,
            num_inference_steps=args.num_inference_steps or infer_cfg.get('num_inference_steps', 1),
            window_size=args.window_size or infer_cfg.get('window_size', 16),
            overlap=args.overlap if args.overlap is not None else infer_cfg.get('overlap', 8),
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
    depth = depth.cpu().numpy()  # (N,H,W) 相对深度
    if depth.ndim == 4:
        depth = depth[:, 0]

    os.makedirs(args.output_dir, exist_ok=True)

    gt_available = args.gt_dir is not None
    metric = DepthMetrics(log_space=False) if gt_available else None

    for i, f in enumerate(files):
        stem = os.path.splitext(os.path.basename(f))[0]
        d = depth[i].astype(np.float32)

        if args.gt_dir is not None:
            gt_file = None
            for ext in ('.tiff', '.tif', '.png', '.npy'):
                cand = os.path.join(args.gt_dir, stem + ext)
                if os.path.exists(cand):
                    gt_file = cand
                    break
            if gt_file is None:
                print('[warn] 找不到 %s 的真值，跳过该帧对齐' % stem)
            else:
                if gt_file.lower().endswith('.npy'):
                    gt = np.load(gt_file).astype(np.float32) * args.gt_scale
                else:
                    gt = np.asarray(Image.open(gt_file), dtype=np.float32) * args.gt_scale
                if gt.shape != d.shape:
                    gt = np.asarray(Image.fromarray(gt.astype(np.float32)).resize(
                        (d.shape[1], d.shape[0]), Image.BILINEAR))
                aligned = median_scale_align(d, gt)
                np.save(os.path.join(args.output_dir, stem + '_depth_m.npy'), aligned)
                if metric is not None:
                    metric.update(torch.from_numpy(d)[None, None], {'mask': torch.from_numpy(gt)[None, None]})
        elif args.scale is not None:
            d = d * args.scale
            np.save(os.path.join(args.output_dir, stem + '_depth_m.npy'), d)

        np.save(os.path.join(args.output_dir, stem + '_depth.npy'), depth[i])
        color = colorize_depth(d, cmap=args.cmap, mode=args.mode)
        Image.fromarray(color).save(os.path.join(args.output_dir, stem + '_depth_colored.png'))
        original = np.asarray(Image.open(f).convert('RGB'))
        if original.shape[:2] != d.shape[:2]:
            original = np.asarray(Image.fromarray(original).resize((d.shape[1], d.shape[0]), Image.BILINEAR))
        blended = overlay_depth(original, d, alpha=0.5, cmap=args.cmap, mode=args.mode)
        Image.fromarray(blended).save(os.path.join(args.output_dir, stem + '_overlay.png'))

    if metric is not None:
        print('评估指标（逐帧 median scaling 对齐后）:')
        for k, v in metric.compute().items():
            print('  %s = %.4f' % (k, v))

    print('完成，输出目录: %s' % args.output_dir)


if __name__ == '__main__':
    main()
