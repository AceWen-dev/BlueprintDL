"""数据集体检(data audit):为一次实验的数据生成 dataset_meta.json 与分布图。

布局约定与训练用 Dataset 完全一致:

- segmentation: ``images/`` 与 ``masks/``(同名 stem 配对,mask 任意常见图片后缀);
- classification: ``<root>/<类别名>/*.图片``;
- detection: ``images/`` 与 ``labels/``(同名 txt,每行 `cls x1 y1 x2 y2` 绝对像素坐标);
- depth: ``images/`` 与 ``depth/``(uint16 PNG=毫米 或 npy=米;depth_scale 只作用于 uint16)。

产物:
- ``dataset_meta.json``:样本数、配对率、类别/像素分布、尺寸分布、逐条 issues、train/val 一致性;
- 分布图(class 分布柱状图 / 检测框统计 / 深度直方图)写入 ``datameta/``。

本模块刻意不依赖 torch,体检能力在纯文件层面即可测试。
"""

from __future__ import annotations

import datetime
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

TASK_ALIASES = {
    'segmentation': 'segmentation',
    'seg': 'segmentation',
    'classification': 'classification',
    'cls': 'classification',
    'detection': 'detection',
    'det': 'detection',
    'depth': 'depth',
    'depth_estimation': 'depth',
}

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
MAX_REPORTED_ISSUES = 200


# ---------- 通用小工具 ----------


def normalize_task(task: str) -> str:
    normalized = TASK_ALIASES.get(str(task).strip().lower())
    if normalized is None:
        raise ValueError('unknown task %r (choose from %s)' % (task, sorted(set(TASK_ALIASES.values()))))
    return normalized


def list_images(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise ValueError('directory not found: %s' % root)
    return sorted(
        (p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: p.name.casefold(),
    )


def _read_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as handle:
        return handle.size  # (w, h)


def _resolve_pair(folder: str | Path, stem: str, extra_exts: tuple[str, ...]) -> Path | None:
    root = Path(folder)
    for ext in extra_exts:
        candidate = root / (stem + ext)
        if candidate.is_file():
            return candidate
    return None


def _size_summary(counter: Counter) -> dict[str, Any]:
    top = counter.most_common(5)
    return {
        'num_unique': len(counter),
        'top_sizes': ['%dx%d: %d' % (w, h, n) for (w, h), n in top],
    }


def _clip_issues(issues: list[dict]) -> list[dict]:
    if len(issues) <= MAX_REPORTED_ISSUES:
        return issues
    clipped = list(issues[:MAX_REPORTED_ISSUES])
    clipped.append({'file': '...', 'reason': 'issues truncated, total=%d' % len(issues)})
    return clipped


def _validate_num_classes(maximum: int, num_classes: int | None, issues: list[dict], path: Path) -> None:
    if num_classes is not None and maximum >= int(num_classes):
        issues.append({'file': path.name, 'reason': 'label >= num_classes (max=%d, num_classes=%d)' % (maximum, num_classes)})


# ---------- 各任务的单 split 审计 ----------


def audit_segmentation(spec: dict, *, sample_limit: int = 300, num_classes: int | None = None) -> dict:
    """images/ + masks/ 同名配对抽查:配对率、类别像素占比、尺寸一致性。"""

    images = list_images(spec['image_dir'])
    mask_dir = Path(spec['mask_dir'])
    if not mask_dir.is_dir():
        raise ValueError('directory not found: %s' % mask_dir)

    issues: list[dict] = []
    class_counter: Counter = Counter()
    size_counter: Counter = Counter()
    matched = 0
    sampled = 0
    background_only = 0

    for image_path in images:
        mask_path = _resolve_pair(mask_dir, image_path.stem, ('.png', '.bmp', '.jpg', '.jpeg', '.tif'))
        if mask_path is None:
            issues.append({'file': image_path.name, 'reason': 'missing_mask'})
            continue
        matched += 1
        try:
            image_size = _read_size(image_path)
            mask_size = _read_size(mask_path)
        except Exception as exc:
            issues.append({'file': image_path.name, 'reason': 'unreadable: %r' % exc})
            continue
        if image_size != mask_size:
            issues.append({'file': image_path.name,
                           'reason': 'size_mismatch image=%dx%d mask=%dx%d' % (*image_size, *mask_size)})
        else:
            size_counter[image_size] += 1

        if sampled < int(sample_limit):
            sampled += 1
            with Image.open(mask_path) as handle:
                mask = np.asarray(handle.convert('L'))
            values = np.unique(mask)
            if values.size and int(values.max()) == 0:
                background_only += 1
            counts = Counter(np.asarray(mask).reshape(-1).tolist())
            class_counter.update(counts)
            _validate_num_classes(int(values.max()) if values.size else 0, num_classes, issues, mask_path)

    return {
        'num_images': len(images),
        'matched_pairs': matched,
        'sampled_masks': sampled,
        'class_pixel_counts': {str(key): value for key, value in sorted(class_counter.items())},
        'sizes': _size_summary(size_counter),
        'background_only_masks': background_only,
        'issues': _clip_issues(issues),
    }


def audit_classification(spec: dict, *, sample_limit: int = 300, num_classes: int | None = None) -> dict:
    """<root>/<类别名>/ 图片目录:类别清单、样本数、可读性抽查。"""

    root = Path(spec['root'])
    if not root.is_dir():
        raise ValueError('directory not found: %s' % root)

    classes = sorted((d.name for d in root.iterdir() if d.is_dir()), key=str.casefold)
    issues: list[dict] = []
    class_counts: dict[str, int] = {}
    size_counter: Counter = Counter()
    sampled = 0
    total = 0

    for class_name in classes:
        files = list_images(root / class_name)
        class_counts[class_name] = len(files)
        total += len(files)
        if not files:
            issues.append({'file': class_name + '/', 'reason': 'empty_class'})
        for image_path in files:
            if sampled < int(sample_limit):
                sampled += 1
                try:
                    size_counter[_read_size(image_path)] += 1
                except Exception as exc:
                    issues.append({'file': image_path.name, 'reason': 'unreadable: %r' % exc})

    if num_classes is not None and len(classes) != int(num_classes):
        issues.append({'file': '(classes)', 'reason': 'num_classes mismatch: found=%d, expected=%d' % (len(classes), num_classes)})

    return {
        'num_classes': len(classes),
        'classes': classes,
        'class_counts': class_counts,
        'num_images': total,
        'sampled_images': sampled,
        'sizes': _size_summary(size_counter),
        'issues': _clip_issues(issues),
    }


def audit_detection(spec: dict, *, sample_limit: int = 300, num_classes: int | None = None) -> dict:
    """images/ + labels/*.txt(`cls x1 y1 x2 y2` 绝对坐标):配对、坐标范围、框统计。"""

    images = list_images(spec['image_dir'])
    label_dir = Path(spec['label_dir'])
    if not label_dir.is_dir():
        raise ValueError('directory not found: %s' % label_dir)

    issues: list[dict] = []
    boxes_per_class: Counter = Counter()
    boxes_per_image: list[int] = []
    size_counter: Counter = Counter()
    matched = 0
    size_checked = 0

    label_files = sorted(label_dir.glob('*.txt'), key=lambda p: p.name.casefold())
    image_stems = {p.stem for p in images}
    orphan_labels = [p.name for p in label_files if p.stem not in image_stems]
    for name in orphan_labels[: MAX_REPORTED_ISSUES]:
        issues.append({'file': name, 'reason': 'label_without_image'})

    for image_path in images:
        label_path = label_dir / (image_path.stem + '.txt')
        if not label_path.is_file():
            issues.append({'file': image_path.name, 'reason': 'missing_label'})
            continue
        matched += 1

        boxes = 0
        image_size = None
        if size_checked < int(sample_limit):
            size_checked += 1
            try:
                image_size = _read_size(image_path)
                size_counter[image_size] += 1
            except Exception as exc:
                issues.append({'file': image_path.name, 'reason': 'unreadable: %r' % exc})

        for line_no, raw in enumerate(label_path.read_text(encoding='utf-8').splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                issues.append({'file': '%s:%d' % (label_path.name, line_no), 'reason': 'bad_line: %r' % line})
                continue
            try:
                cls = int(float(parts[0]))
                x1, y1, x2, y2 = (float(v) for v in parts[1:])
            except ValueError as exc:
                issues.append({'file': label_path.name, 'reason': 'unparsable_line_%d: %r' % (line_no, exc)})
                continue
            boxes += 1
            boxes_per_class[cls] += 1
            if num_classes is not None and cls >= int(num_classes):
                issues.append({'file': label_path.name, 'reason': 'cls >= num_classes (cls=%d)' % cls})
            if image_size is not None:
                width, height = image_size
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    issues.append({'file': label_path.name, 'reason': 'box_out_of_range line %d' % line_no})

        boxes_per_image.append(boxes)
        if boxes == 0:
            issues.append({'file': image_path.name, 'reason': 'image_without_boxes'})

    summary = _stats(boxes_per_image)
    return {
        'num_images': len(images),
        'matched_pairs': matched,
        'num_classes_est': (max(boxes_per_class) + 1) if boxes_per_class else 0,
        'boxes_per_class': {str(key): value for key, value in sorted(boxes_per_class.items())},
        'boxes_per_image': summary,
        'sizes': _size_summary(size_counter),
        'issues': _clip_issues(issues),
    }


def audit_depth(spec: dict, *, sample_limit: int = 300, num_classes: int | None = None) -> dict:  # noqa: ARG001 统一签名
    """images/ + depth/(uint16 毫米 PNG 或 npy 米):配对、有效率、范围统计。"""

    images = list_images(spec['image_dir'])
    depth_dir = Path(spec['depth_dir'])
    if not depth_dir.is_dir():
        raise ValueError('directory not found: %s' % depth_dir)
    explicit_scale = spec.get('depth_scale')
    # 与 DepthDataset._load_depth 的约定一致:
    # npy 默认米(×1.0),图片默认 uint16 毫米(×0.001);显式 depth_scale 优先。
    default_scale = 1.0
    depth_scale = float(explicit_scale) if explicit_scale is not None else default_scale

    issues: list[dict] = []
    size_counter: Counter = Counter()
    pooled: list[np.ndarray] = []
    matched = 0
    valid_fractions: list[float] = []
    mins: list[float] = []
    maxs: list[float] = []

    for image_path in images:
        depth_path = _resolve_pair(depth_dir, image_path.stem, ('.png', '.npy', '.bmp', '.jpg', '.jpeg'))
        if depth_path is None:
            issues.append({'file': image_path.name, 'reason': 'missing_depth'})
            continue
        matched += 1
        try:
            if sampled_size_ok(size_counter, sample_limit):
                size_counter[_read_size(image_path)] += 1
        except Exception as exc:
            issues.append({'file': image_path.name, 'reason': 'unreadable: %r' % exc})

        try:
            if depth_path.suffix.lower() == '.npy':
                depth = np.load(depth_path, allow_pickle=False).astype(np.float32)
                scale_for_file = float(explicit_scale) if explicit_scale is not None else 1.0
            else:
                with Image.open(depth_path) as handle:
                    depth = np.asarray(handle).astype(np.float32)
                scale_for_file = float(explicit_scale) if explicit_scale is not None else 0.001
            depth = depth * scale_for_file
        except Exception as exc:
            issues.append({'file': depth_path.name, 'reason': 'unreadable: %r' % exc})
            continue

        finite = np.isfinite(depth)
        valid = finite & (depth > 0)
        total = depth.size
        fraction = float(valid.sum()) / max(total, 1)
        valid_fractions.append(fraction)
        if fraction == 0.0:
            issues.append({'file': depth_path.name, 'reason': 'all_invalid_depth'})
            continue
        values = depth[valid]
        mins.append(float(values.min()))
        maxs.append(float(values.max()))
        if len(pooled) < int(sample_limit):
            pooled.append(values.astype(np.float32).reshape(-1))
            if sum(v.size for v in pooled) > 200_000:
                pooled = [np.concatenate(pooled)[-200_000:]]

    histogram = None
    if pooled:
        values = np.concatenate(pooled)
        counts, edges = np.histogram(values, bins=32)
        histogram = {
            'values_min': float(values.min()),
            'values_max': float(values.max()),
            'bin_edges': [round(float(e), 4) for e in edges],
            'counts': [int(c) for c in counts],
        }
    return {
        'num_images': len(images),
        'matched_pairs': matched,
        'depth_scale': depth_scale,
        'valid_fraction': _stats(valid_fractions),
        'depth_range': {
            'train_min': min(mins) if mins else None,
            'train_max': max(maxs) if maxs else None,
        },
        'sampled_values': histogram,
        'issues': _clip_issues(issues),
    }


def sampled_size_ok(counter: Counter, sample_limit: int) -> bool:
    """尺寸读取的软上限(避免超大集合全量读头)。"""
    return sum(counter.values()) < int(sample_limit)


def _stats(values: list[int | float]) -> dict:
    if not values:
        return {'count': 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        'count': int(array.size),
        'mean': float(array.mean()),
        'min': float(array.min()),
        'max': float(array.max()),
        'p95': float(np.percentile(array, 95)),
    }


# ---------- train/val 一致性 ----------


def audit_consistency(task: str, train_report: dict, val_report: dict | None) -> dict | None:
    if val_report is None:
        return None

    result: dict[str, Any] = {'issues': []}

    if task == 'classification':
        train_classes = set(train_report.get('classes') or [])
        val_classes = set(val_report.get('classes') or [])
        result['train_only_classes'] = sorted(train_classes - val_classes)
        result['val_only_classes'] = sorted(val_classes - train_classes)
        if result['val_only_classes']:
            result['issues'].append({'file': '(val)', 'reason': 'val classes unseen in train: %s' % result['val_only_classes']})

    if task == 'segmentation':
        train_keys = set(train_report.get('class_pixel_counts') or {})
        val_keys = set(val_report.get('class_pixel_counts') or {})
        result['val_only_classes'] = sorted(val_keys - train_keys, key=lambda s: (len(s), s))
        if result['val_only_classes']:
            result['issues'].append({'file': '(val)', 'reason': 'val mask classes unseen in train (pixel counts): %s' % result['val_only_classes']})

    if task == 'detection':
        train_classes = set(train_report.get('boxes_per_class') or {})
        val_classes = set(val_report.get('boxes_per_class') or {})
        result['val_only_classes'] = sorted(val_classes - train_classes)
        if result['val_only_classes']:
            result['issues'].append({'file': '(val)', 'reason': 'val classes unseen in train: %s' % result['val_only_classes']})

    if task == 'depth':
        train_range = (train_report.get('depth_range') or {})
        val_range = (val_report.get('depth_range') or {})
        result['train_range'] = train_range
        result['val_range'] = val_range
        if (
            train_range.get('train_max') is not None
            and val_range.get('train_min') is not None
            and (val_range['train_min'] > train_range['train_max'] or val_range['train_max'] < train_range.get('train_min', 0))
        ):
            result['issues'].append({'file': '(val)', 'reason': 'depth ranges disjoint between train and val'})

    return result


# ---------- 总入口 ----------


_AUDIT_FUNCS = {
    'segmentation': audit_segmentation,
    'classification': audit_classification,
    'detection': audit_detection,
    'depth': audit_depth,
}


def audit_dataset(
    task: str,
    train: dict,
    val: dict | None = None,
    *,
    num_classes: int | None = None,
    sample_limit: int = 300,
) -> dict:
    """对 train(+val) splits 做体检,返回可 json 序列化的报告。"""

    task = normalize_task(task)
    func = _AUDIT_FUNCS[task]
    train_report = func(train, sample_limit=sample_limit, num_classes=num_classes)
    val_report = func(val, sample_limit=sample_limit, num_classes=num_classes) if val else None
    return {
        'schema_version': 1,
        'task': task,
        'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'num_classes': num_classes,
        'sample_limit': sample_limit,
        'train': train_report,
        'val': val_report,
        'consistency': audit_consistency(task, train_report, val_report),
    }


# ---------- 分布图 ----------


def render_report_plots(meta: dict, report_dir: str | Path) -> list[Path]:
    """按任务输出分布图到 <report_dir>/datameta/,返回文件列表。"""

    task = meta['task']
    out_dir = Path(report_dir) / 'datameta'
    out_dir.mkdir(parents=True, exist_ok=True)
    plots: list[Path] = []

    if task == 'classification':
        classes = meta['train'].get('class_counts') or {}
        series = {'train': list(classes.values())}
        if meta.get('val'):
            val_counts = meta['val'].get('class_counts') or {}
            series['val'] = [val_counts.get(name, 0) for name in classes]
        plots.append(_grouped_bar(
            list(classes.keys()), series, 'images per class', out_dir / 'class_distribution.png'))

    elif task == 'segmentation':
        counts = meta['train'].get('class_pixel_counts') or {}
        keys = sorted(counts, key=_class_sort)
        series = {'train': [counts[k] for k in keys]}
        if meta.get('val'):
            val_counts = meta['val'].get('class_pixel_counts') or {}
            series['val'] = [val_counts.get(k, 0) for k in keys]
        plots.append(_grouped_bar(
            keys, series, 'pixels per class', out_dir / 'class_distribution.png'))

    elif task == 'detection':
        counts = meta['train'].get('boxes_per_class') or {}
        keys = sorted(counts, key=lambda s: int(s))
        series = {'train': [counts[k] for k in keys]}
        if meta.get('val'):
            val_counts = meta['val'].get('boxes_per_class') or {}
            series['val'] = [val_counts.get(k, 0) for k in keys]
        plots.append(_grouped_bar(
            keys, series, 'boxes per class', out_dir / 'boxes_distribution.png'))

    elif task == 'depth':
        plots.append(_depth_hist(meta, out_dir / 'depth_histogram.png'))

    return plots


def _class_sort(key: str):
    try:
        return (0, int(key), '')
    except ValueError:
        return (1, 0, key)


def _grouped_bar(categories: list[str], series: dict[str, list], ylabel: str, out_path: Path) -> Path:
    positions = np.arange(len(categories))
    width = 0.8 / max(1, len(series))
    figure, ax = plt.subplots(figsize=(max(6, 0.6 * len(categories)), 4.2))
    colors = ('tab:blue', 'tab:orange', 'tab:green')
    for index, (name, values) in enumerate(series.items()):
        ax.bar(positions + index * width, values, width * 0.9, label=name, color=colors[index % len(colors)])
    ax.set_xticks(positions + width * (len(series) - 1) / 2)
    if len(categories) <= 16:
        ax.set_xticklabels(categories)
    else:
        step = max(1, len(categories) // 16)
        ax.set_xticklabels(categories[::step])
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def _depth_hist(meta: dict, out_path: Path) -> Path:
    figure, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = ('tab:blue', 'tab:orange')
    for index, split in enumerate(('train', 'val')):
        sampled = ((meta.get(split) or {}).get('sampled_values')) or None
        if not sampled:
            continue
        edges = np.asarray(sampled['bin_edges'], dtype=np.float64)
        counts = np.asarray(sampled['counts'], dtype=np.float64)
        centers = (edges[:-1] + edges[1:]) / 2
        share = counts / max(counts.sum(), 1)
        ax.bar(centers, share, width=np.diff(edges).mean() * 0.95, color=colors[index % len(colors)], alpha=0.55, label=split)
    ax.set_xlabel('depth (m)')
    ax.set_ylabel('fraction')
    ax.set_title('depth value distribution')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def write_meta(meta: dict, report_dir: str | Path) -> Path:
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'dataset_meta.json'
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return path


__all__ = [
    'IMAGE_EXTS',
    'TASK_ALIASES',
    'audit_classification',
    'audit_consistency',
    'audit_dataset',
    'audit_depth',
    'audit_detection',
    'audit_segmentation',
    'normalize_task',
    'render_report_plots',
    'write_meta',
]
