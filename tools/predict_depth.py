import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np
import torch
from PIL import Image

import dlkit.models  # noqa: F401

from dlkit.registry import build_from_cfg
from dlkit.utils.config import load_config
from dlkit.utils.depth_calibration import fit_log_affine, apply_log_affine, calibrate_from_pairs
from dlkit.visualize.depth import colorize_depth, overlay_depth

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_model(cfg, checkpoint, device):
    model = build_from_cfg(cfg['model'])
    state = torch.load(checkpoint, map_location=device)
    if 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    return model.to(device).eval()


def preprocess(path, device):
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    mean = torch.as_tensor(IMAGENET_MEAN, dtype=t.dtype, device=device).view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, dtype=t.dtype, device=device).view(1, 3, 1, 1)
    return (t - mean) / std


def fit_calibration(model, image_dir, depth_dir, device):
    from dlkit.data.depth_dataset import DepthDataset

    dataset = DepthDataset(image_dir=image_dir, depth_dir=depth_dir)
    pred_logs, gts = [], []
    for sample in dataset:
        image = torch.from_numpy(np.asarray(sample['image'], dtype=np.float32) / 255.0)
        image = image.permute(2, 0, 1).unsqueeze(0).to(device)
        mean = torch.as_tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.as_tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        image = (image - mean) / std
        with torch.no_grad():
            logit = model(image)
        pred_logs.append(logit[0, 0].cpu().numpy())
        gts.append(np.asarray(sample['mask'], dtype=np.float32))
    return calibrate_from_pairs(pred_logs, gts)


def main():
    parser = argparse.ArgumentParser(description='单目深度估计推理（YOLO26-Depth 风格）')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True, help='图片文件或目录')
    parser.add_argument('--output-dir', default='runs/predict_depth')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--calibration', default=None, help='校准参数 "a,b"，如 "0.98,0.12"')
    parser.add_argument('--fit-calibration', nargs=2, metavar=('IMAGE_DIR', 'DEPTH_DIR'),
                        default=None, help='用 (图片目录, 深度真值目录) 拟合校准参数')
    parser.add_argument('--cmap', default='jet', choices=['jet', 'hot', 'gray'])
    parser.add_argument('--mode', default='metric', choices=['metric', 'disparity'])
    parser.add_argument('--vmin', type=float, default=None, help='彩色图锁定范围下限（米）')
    parser.add_argument('--vmax', type=float, default=None, help='彩色图锁定范围上限（米）')
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = load_model(cfg, args.checkpoint, args.device)

    a, b = 1.0, 0.0
    if args.fit_calibration is not None:
        a, b = fit_calibration(model, args.fit_calibration[0], args.fit_calibration[1], args.device)
        print('拟合校准参数: a=%.4f b=%.4f' % (a, b))
    elif args.calibration is not None:
        a, b = [float(v) for v in args.calibration.split(',')]
        print('使用校准参数: a=%.4f b=%.4f' % (a, b))

    if os.path.isdir(args.input):
        files = [
            os.path.join(args.input, f) for f in sorted(os.listdir(args.input))
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
    else:
        files = [args.input]

    os.makedirs(args.output_dir, exist_ok=True)
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        original = np.asarray(Image.open(f).convert('RGB'))
        t = preprocess(f, args.device)
        with torch.no_grad():
            logit = model(t)
        log_depth = logit[0, 0].cpu().numpy()
        depth = apply_log_affine(log_depth, a, b).astype(np.float32)

        np.save(os.path.join(args.output_dir, stem + '_depth.npy'), depth)
        color = colorize_depth(depth, cmap=args.cmap, vmin=args.vmin, vmax=args.vmax, mode=args.mode)
        Image.fromarray(color).save(os.path.join(args.output_dir, stem + '_depth_colored.png'))
        blended = overlay_depth(original, depth, alpha=0.5, cmap=args.cmap,
                                vmin=args.vmin, vmax=args.vmax, mode=args.mode)
        Image.fromarray(blended).save(os.path.join(args.output_dir, stem + '_overlay.png'))
        print('%s -> %s (深度范围 %.2f~%.2f m)' % (f, args.output_dir, float(depth.min()), float(depth.max())))


if __name__ == '__main__':
    main()
