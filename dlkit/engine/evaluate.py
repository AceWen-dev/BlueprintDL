import json

import torch

import dlkit.data  # noqa: F401
import dlkit.models  # noqa: F401
import dlkit.metrics  # noqa: F401
from dlkit.builders import build_from_cfg, build_loader
from dlkit.utils.seed import set_seed
from dlkit.utils.device import resolve_device
from dlkit.utils.checkpoint import load_checkpoint
from dlkit.utils.batch import to_device


def evaluate(cfg, checkpoint, split='val', device=None, output_json=None):
    set_seed(cfg.get('seed', 42))
    dev = resolve_device(device or cfg.get('device', 'auto'))

    loader = build_loader(cfg['data'], split)
    model = build_from_cfg(cfg['model']).to(dev)
    load_checkpoint(checkpoint, model=model, device=dev)

    criterion = build_from_cfg(cfg['loss']).to(dev) if 'loss' in cfg else None
    metrics = [build_from_cfg(m) for m in cfg.get('metrics', [])]

    model.eval()
    for m in metrics:
        m.reset()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, dev)
            images = batch['image']
            preds = model(images)
            if criterion is not None:
                total_loss += float(criterion(preds, batch).detach().item())
            n += 1
            for m in metrics:
                m.update(preds, batch)

    result = {}
    if criterion is not None:
        result['val_loss'] = float(total_loss / max(1, n))
    for m in metrics:
        for k, v in m.compute().items():
            if isinstance(v, float):
                result[k] = float(v)

    if output_json:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result
