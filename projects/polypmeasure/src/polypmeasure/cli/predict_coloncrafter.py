"""Run ColonCrafter depth prediction through the PolypMeasure plugin."""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from dlkit.builders import build_from_cfg
from dlkit.metrics.depth_metrics import DepthMetrics
from dlkit.utils.config import load_config
from dlkit.utils.depth_calibration import median_scale_align
from dlkit.visualize.depth import colorize_depth, overlay_depth
from polypmeasure.bootstrap import register

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def _list_frames(input_dir):
    files = sorted(
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if name.lower().endswith(_IMAGE_SUFFIXES)
    )
    if not files:
        raise SystemExit("输入目录里没有图片帧: %s" % input_dir)
    return files


def load_frames(input_dir, size=None):
    """Load an image sequence as an ``(N, 3, H, W)`` float tensor."""
    files = _list_frames(input_dir)
    frames = []
    for path in files:
        image = Image.open(path).convert("RGB")
        if size is not None:
            image = image.resize((size, size), Image.BILINEAR)
        frames.append(np.asarray(image, dtype=np.float32) / 255.0)
    array = np.stack(frames, axis=0)
    return torch.from_numpy(array).permute(0, 3, 1, 2), files


def _find_ground_truth(directory, stem):
    for extension in (".tiff", ".tif", ".png", ".npy"):
        candidate = os.path.join(directory, stem + extension)
        if os.path.exists(candidate):
            return candidate
    return None


def build_parser():
    parser = argparse.ArgumentParser(
        description="ColonCrafter 结肠镜视频深度估计（PolypMeasure 插件）"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, help="按文件名排序的帧目录")
    parser.add_argument("--output-dir", default="runs/predict_coloncrafter")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gt-dir", default=None, help="用于尺度对齐与评估的真值目录")
    parser.add_argument(
        "--gt-scale",
        type=float,
        default=100.0 / 65535.0 / 1000.0,
        help="value * gt-scale = 米；默认值适用于 C3VD uint16 深度图",
    )
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--cmap", default="jet", choices=["jet", "hot", "gray"])
    parser.add_argument(
        "--mode", default="metric", choices=["metric", "disparity"]
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    register()
    cfg = load_config(args.config)
    infer_cfg = cfg.get("infer", {})
    model = build_from_cfg(cfg["model"])

    video, files = load_frames(args.input, size=args.size or infer_cfg.get("size"))
    print("输入 %d 帧, 形状 %s" % (len(files), tuple(video.shape)))
    inference_args = {
        "num_inference_steps": args.num_inference_steps
        if args.num_inference_steps is not None
        else infer_cfg.get("num_inference_steps", 1),
        "window_size": args.window_size
        if args.window_size is not None
        else infer_cfg.get("window_size", 16),
        "overlap": args.overlap
        if args.overlap is not None
        else infer_cfg.get("overlap", 8),
        "guidance_scale": args.guidance_scale
        if args.guidance_scale is not None
        else infer_cfg.get("guidance_scale", 1.0),
        "seed": args.seed if args.seed is not None else infer_cfg.get("seed", 42),
    }
    with torch.no_grad():
        depth = model(video, **inference_args)
    depth = depth.cpu().numpy()
    if depth.ndim == 4:
        depth = depth[:, 0]

    os.makedirs(args.output_dir, exist_ok=True)
    metric = DepthMetrics(log_space=False) if args.gt_dir else None
    for index, source_path in enumerate(files):
        stem = os.path.splitext(os.path.basename(source_path))[0]
        current_depth = depth[index].astype(np.float32)
        if args.gt_dir:
            ground_truth_path = _find_ground_truth(args.gt_dir, stem)
            if ground_truth_path is None:
                print("[warn] 找不到 %s 的真值，跳过该帧对齐" % stem)
            else:
                if ground_truth_path.lower().endswith(".npy"):
                    ground_truth = np.load(ground_truth_path).astype(np.float32)
                else:
                    ground_truth = np.asarray(
                        Image.open(ground_truth_path), dtype=np.float32
                    )
                ground_truth *= args.gt_scale
                if ground_truth.shape != current_depth.shape:
                    ground_truth = np.asarray(
                        Image.fromarray(ground_truth).resize(
                            (current_depth.shape[1], current_depth.shape[0]),
                            Image.BILINEAR,
                        )
                    )
                current_depth = median_scale_align(current_depth, ground_truth)
                np.save(
                    os.path.join(args.output_dir, stem + "_depth_m.npy"),
                    current_depth,
                )
                metric.update(
                    torch.from_numpy(current_depth)[None, None],
                    {"mask": torch.from_numpy(ground_truth)[None, None]},
                )
        elif args.scale is not None:
            current_depth *= args.scale
            np.save(
                os.path.join(args.output_dir, stem + "_depth_m.npy"),
                current_depth,
            )

        np.save(os.path.join(args.output_dir, stem + "_depth.npy"), depth[index])
        color = colorize_depth(current_depth, cmap=args.cmap, mode=args.mode)
        Image.fromarray(color).save(
            os.path.join(args.output_dir, stem + "_depth_colored.png")
        )
        original = np.asarray(Image.open(source_path).convert("RGB"))
        if original.shape[:2] != current_depth.shape[:2]:
            original = np.asarray(
                Image.fromarray(original).resize(
                    (current_depth.shape[1], current_depth.shape[0]), Image.BILINEAR
                )
            )
        blended = overlay_depth(
            original,
            current_depth,
            alpha=0.5,
            cmap=args.cmap,
            mode=args.mode,
        )
        Image.fromarray(blended).save(
            os.path.join(args.output_dir, stem + "_overlay.png")
        )

    if metric is not None:
        print("评估指标（逐帧 median scaling 对齐后）:")
        for name, value in metric.compute().items():
            print("  %s = %.4f" % (name, value))
    print("完成，输出目录: %s" % args.output_dir)


if __name__ == "__main__":
    main()
