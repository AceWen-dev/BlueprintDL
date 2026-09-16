import csv
import hashlib
import json
import os

import numpy as np
from PIL import Image

from dlkit.builders import build_from_cfg
from dlkit.registry import CLEANERS

Image.MAX_IMAGE_PIXELS = None


def _iter_pairs(image_dir, mask_dir, image_suffix, mask_suffix):
    images = sorted(
        f for f in os.listdir(image_dir) if f.lower().endswith(image_suffix)
    )
    masks = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(mask_dir)
        if f.lower().endswith(mask_suffix)
    }
    for f in images:
        stem = os.path.splitext(f)[0]
        if stem not in masks:
            yield os.path.join(image_dir, f), None, stem, 'missing mask'
            continue
        yield os.path.join(image_dir, f), os.path.join(mask_dir, masks[stem]), stem, None


class BaseCleaner:
    def begin(self):
        pass

    def check(self, image_path, mask_path):
        raise NotImplementedError


@CLEANERS.register()
class CorruptImage(BaseCleaner):
    def check(self, image_path, mask_path):
        for p in (image_path, mask_path):
            if p is None:
                continue
            try:
                with Image.open(p) as im:
                    im.load()
            except (OSError, ValueError, SyntaxError):
                return False, 'cannot open image %s' % p
        return True, None


@CLEANERS.register()
class SizeMismatch(BaseCleaner):
    def check(self, image_path, mask_path):
        if mask_path is None:
            return False, 'mask missing'
        with Image.open(image_path) as a, Image.open(mask_path) as b:
            if a.size != b.size:
                return False, 'size mismatch image=%s mask=%s' % (a.size, b.size)
        return True, None


@CLEANERS.register()
class MaskValues(BaseCleaner):
    def __init__(self, num_classes, ignore_values=None):
        self.allowed = set(range(num_classes)) | set(ignore_values or [255])

    def check(self, image_path, mask_path):
        if mask_path is None:
            return False, 'mask missing'
        mask = np.asarray(Image.open(mask_path))
        bad = sorted(set(np.unique(mask).tolist()) - self.allowed)
        if bad:
            return False, 'illegal mask values %s' % bad
        return True, None


@CLEANERS.register()
class MinSize(BaseCleaner):
    def __init__(self, min_size=32):
        self.min_size = min_size

    def check(self, image_path, mask_path):
        with Image.open(image_path) as im:
            if min(im.size) < self.min_size:
                return False, 'image too small %s' % (im.size,)
        return True, None


@CLEANERS.register()
class Duplicate(BaseCleaner):
    def begin(self):
        self.seen = set()

    def check(self, image_path, mask_path):
        with open(image_path, 'rb') as f:
            digest = hashlib.md5(f.read()).hexdigest()
        if digest in self.seen:
            return False, 'duplicate image md5=%s' % digest
        self.seen.add(digest)
        return True, None


def run_cleaning(cfg):
    image_dir = cfg['image_dir']
    mask_dir = cfg['mask_dir']
    output_dir = cfg.get('output_dir', 'runs/clean_report')
    action = cfg.get('action', 'report')
    image_suffix = tuple(cfg.get('image_suffix', ('.png', '.jpg', '.jpeg', '.bmp')))
    mask_suffix = tuple(cfg.get('mask_suffix', ('.png',)))
    os.makedirs(output_dir, exist_ok=True)

    cleaners = [build_from_cfg(c) for c in cfg.get('cleaners', [])]
    for cleaner in cleaners:
        cleaner.begin()

    issues = []
    for image_path, mask_path, stem, pair_reason in _iter_pairs(
        image_dir, mask_dir, image_suffix, mask_suffix
    ):
        if pair_reason:
            issues.append({'image': image_path, 'mask': '', 'cleaner': 'Pairing', 'reason': pair_reason})
            continue
        for cleaner in cleaners:
            ok, reason = cleaner.check(image_path, mask_path)
            if not ok:
                issues.append({
                    'image': image_path,
                    'mask': mask_path,
                    'cleaner': cleaner.__class__.__name__,
                    'reason': reason,
                })

    summary = {'total_issues': len(issues), 'issues': issues}
    with open(os.path.join(output_dir, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, 'report.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'mask', 'cleaner', 'reason'])
        for it in issues:
            writer.writerow([it['image'], it['mask'], it['cleaner'], it['reason']])

    if action in ('move', 'delete'):
        quarantine = os.path.join(output_dir, 'quarantine')
        os.makedirs(quarantine, exist_ok=True)
        handled_images = set()
        for it in issues:
            if it['image'] in handled_images:
                continue
            handled_images.add(it['image'])
            if action == 'move':
                os.replace(it['image'], os.path.join(quarantine, os.path.basename(it['image'])))
                if it['mask']:
                    os.replace(it['mask'], os.path.join(quarantine, os.path.basename(it['mask'])))
            elif action == 'delete':
                os.remove(it['image'])
                if it['mask'] and os.path.exists(it['mask']):
                    os.remove(it['mask'])

    return summary
