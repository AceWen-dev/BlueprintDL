import json
import os

import numpy as np
import torch
from PIL import Image

import dlkit.models  # noqa: F401
from dlkit.registry import build_from_cfg
from dlkit.data.dataset import _build_transform
from dlkit.utils.device import resolve_device
from dlkit.utils.checkpoint import load_checkpoint
from dlkit.visualize.seg import colorize, save_mask, save_overlay


def _load_palette(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if 'colors' in data:
        data = data['colors']
    return {int(k): v for k, v in data.items()}


class Inferencer:
    def __init__(self, cfg, checkpoint, device=None, palette=None, class_names=None):
        self.device = resolve_device(device or cfg.get('device', 'auto'))
        self.model = build_from_cfg(cfg['model']).to(self.device)
        load_checkpoint(checkpoint, model=self.model, device=self.device)
        self.model.eval()

        transforms = cfg.get('data', {}).get('transforms', {})
        infer_cfg = transforms.get('infer') or transforms.get('val')
        self.transform = _build_transform(infer_cfg)

        self.palette = None
        if palette:
            self.palette = _load_palette(palette)
        self.class_names = class_names

    def _preprocess(self, image):
        if isinstance(image, str):
            image = np.asarray(Image.open(image).convert('RGB'))
        else:
            image = np.asarray(image)
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        sample = self.transform({'image': image, 'mask': mask, 'name': 'input'})
        tensor = sample['image'].unsqueeze(0).to(self.device)
        return tensor, sample['image'].shape[1:]

    def predict(self, image):
        tensor, out_size = self._preprocess(image)
        with torch.no_grad():
            logits = self.model(tensor)
        logits = logits[0].cpu()
        probs = torch.softmax(logits, dim=0)
        pred = probs.argmax(dim=0).numpy()
        if pred.shape != out_size:
            from PIL import Image as _Image
            pred = np.asarray(
                _Image.fromarray(pred.astype(np.uint8)).resize((out_size[1], out_size[0]), _Image.NEAREST)
            )
        return {'logits': logits, 'probs': probs.numpy(), 'mask': pred}

    def predict_file(self, image_path, output_dir=None, save_mask_image=True, save_overlay_image=True):
        image = np.asarray(Image.open(image_path).convert('RGB'))
        result = self.predict(image)
        outputs = {'mask': result['mask']}
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(image_path))[0]
            if self.palette is not None and save_mask_image:
                outputs['mask_file'] = save_mask(
                    result['mask'], self.palette, os.path.join(output_dir, stem + '_mask.png')
                )
            if self.palette is not None and save_overlay_image:
                outputs['overlay_file'] = save_overlay(
                    image, result['mask'], self.palette, os.path.join(output_dir, stem + '_overlay.png')
                )
        return outputs
