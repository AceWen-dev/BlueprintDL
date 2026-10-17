"""数据集体检:为一个数据集生成 dataset_meta.json 与分布图(数据档案)。

用法(按任务选择参数组):

    # 分割: images/ + masks/
    uv run python tools/data/prepare_data.py --task segmentation --image-dir data/myseg/train/images --mask-dir data/myseg/train/masks --val-image-dir data/myseg/val/images --val-mask-dir data/myseg/val/masks --num-classes 4

    # 分类: <root>/<class_name>/
    uv run python tools/data/prepare_data.py --task classification --train-dir data/mycls/train --val-dir data/mycls/val

    # 检测: images/ + labels/*.txt
    uv run python tools/data/prepare_data.py --task detection --image-dir data/mydet/train/images --label-dir data/mydet/train/labels

    # 深度: images/ + depth/(uint16 毫米 或 npy 米)
    uv run python tools/data/prepare_data.py --task depth --image-dir data/mydep/train/images --depth-dir data/mydep/train/depth --depth-scale 0.001

输出(--report-dir 默认取 image_dir/train_dir 的父目录):
    dataset_meta.json    样本数、配对率、类别分布、issues、train/val 一致性
    datameta/*.png       类别/框/深度分布图

分割任务可用 --write-classes 顺便生成 classes.json(调色板,供 predict --palette 使用)。
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import json
from pathlib import Path

from dlkit.utils.data_audit import (
    audit_dataset,
    normalize_task,
    render_report_plots,
    write_meta,
)


def build_parser():
    parser = argparse.ArgumentParser(description='Audit a dataset and write dataset_meta.json + distribution plots')
    parser.add_argument('--task', required=True,
                        help='segmentation/classification/detection/depth (aliases: seg/cls/det)')
    parser.add_argument('--image-dir', default=None, help='train images dir (seg/det/depth)')
    parser.add_argument('--mask-dir', default=None, help='train masks dir (seg)')
    parser.add_argument('--label-dir', default=None, help='train labels dir (det, *.txt)')
    parser.add_argument('--depth-dir', default=None, help='train depth dir (seg for depth task)')
    parser.add_argument('--depth-scale', type=float, default=None,
                        help='explicit depth scale; default follows dataset convention (npy=meters, image=uint16 millimeters)')
    parser.add_argument('--train-dir', default=None, help='train root dir (classification)')
    parser.add_argument('--val-image-dir', default=None, help='val images dir')
    parser.add_argument('--val-mask-dir', default=None, help='val masks dir (seg)')
    parser.add_argument('--val-label-dir', default=None, help='val labels dir (det)')
    parser.add_argument('--val-depth-dir', default=None, help='val depth dir')
    parser.add_argument('--val-dir', default=None, help='val root dir (classification)')
    parser.add_argument('--num-classes', type=int, default=None, help='expected class count for issue checks')
    parser.add_argument('--sample-limit', type=int, default=300,
                        help='max samples for pixel/size statistics (default 300)')
    parser.add_argument('--report-dir', default=None,
                        help='output dir (default: parent of image/train dir)')
    parser.add_argument('--write-classes', action='store_true',
                        help='segmentation only: write a default classes.json palette')
    return parser


def _split_spec(args):
    task = normalize_task(args.task)
    if task == 'segmentation':
        if not (args.image_dir and args.mask_dir):
            raise SystemExit('segmentation needs --image-dir and --mask-dir')
        train = {'image_dir': args.image_dir, 'mask_dir': args.mask_dir}
        val = {'image_dir': args.val_image_dir, 'mask_dir': args.val_mask_dir} if args.val_image_dir else None
        return task, train, val
    if task == 'classification':
        if not args.train_dir:
            raise SystemExit('classification needs --train-dir')
        train = {'root': args.train_dir}
        val = {'root': args.val_dir} if args.val_dir else None
        return task, train, val
    if task == 'detection':
        if not (args.image_dir and args.label_dir):
            raise SystemExit('detection needs --image-dir and --label-dir')
        train = {'image_dir': args.image_dir, 'label_dir': args.label_dir}
        val = {'image_dir': args.val_image_dir, 'label_dir': args.val_label_dir} if args.val_image_dir else None
        return task, train, val
    # depth
    if not (args.image_dir and args.depth_dir):
        raise SystemExit('depth needs --image-dir and --depth-dir')
    train = {'image_dir': args.image_dir, 'depth_dir': args.depth_dir, 'depth_scale': args.depth_scale}
    val = {'image_dir': args.val_image_dir, 'depth_dir': args.val_depth_dir,
           'depth_scale': args.depth_scale} if args.val_image_dir else None
    return task, train, val


def _default_report_dir(task, args):
    anchor = args.image_dir or args.train_dir
    if not anchor:
        raise SystemExit('cannot infer report dir: provide --report-dir')
    return Path(anchor).parent


def _write_classes_json(meta, path: Path) -> Path:
    """用 tab10 色环生成默认调色板,类别数取实际观测到的最大类 + 1。"""

    counts = meta['train'].get('class_pixel_counts') or meta['train'].get('class_counts') or {}
    observed = max((int(k) for k in counts), default=-1)
    num_classes = int(meta.get('num_classes') or 0) or (observed + 1)
    if num_classes <= 0:
        raise SystemExit('cannot infer class count for classes.json; pass --num-classes')

    import matplotlib.pyplot as plt

    colormap = plt.get_cmap('tab10')
    names = ['class_%d' % i for i in range(num_classes)]
    payload = {
        'names': names,
        'colors': {str(i): [int(c * 255) for c in colormap(i % colormap.N)[:3]] for i in range(num_classes)},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return path


def main():
    args = build_parser().parse_args()

    task, train, val = _split_spec(args)
    report_dir = Path(args.report_dir) if args.report_dir else _default_report_dir(task, args)

    meta = audit_dataset(task, train, val, num_classes=args.num_classes, sample_limit=args.sample_limit)
    meta['train']['spec'] = {key: str(value) for key, value in train.items()}
    if val:
        meta['val']['spec'] = {key: str(value) for key, value in val.items()}

    meta_path = write_meta(meta, report_dir)
    plots = render_report_plots(meta, report_dir)
    meta['artifacts'] = {
        'meta': meta_path.name,
        'plots': [path.name for path in plots],
    }
    # 补写 artifacts 字段(轻量二次写入,保持 meta 自描述)
    write_meta(meta, report_dir)

    if task == 'segmentation' and args.write_classes:
        classes_path = _write_classes_json(meta, report_dir / 'classes.json')
        print('Wrote %s' % classes_path)

    train_report = meta['train']
    print('Task: %s | train: %s samples, %s matched pairs | val: %s' % (
        task,
        train_report.get('num_images', '-'),
        train_report.get('matched_pairs', '-'),
        (meta['val'] or {}).get('num_images', '-') if meta['val'] else 'none',
    ))
    issues = len((meta.get('consistency') or {}).get('issues') or [])
    issues += len(train_report.get('issues') or [])
    if meta['val']:
        issues += len(meta['val'].get('issues') or [])
    print('Issues: %d | Meta: %s' % (issues, meta_path))
    for path in plots:
        print('Wrote %s' % path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
