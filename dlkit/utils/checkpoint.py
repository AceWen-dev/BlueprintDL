import torch


def load_checkpoint(path, model=None, optimizer=None, scheduler=None, device=None):
    map_location = device if device is not None else 'cpu'
    ckpt = torch.load(path, map_location=map_location)
    if model is not None:
        sd = ckpt.get('model') if isinstance(ckpt, dict) else ckpt
        if sd is None:
            raise KeyError('Checkpoint %s has no model weights' % (path,))
        model.load_state_dict(sd)
    if optimizer is not None and isinstance(ckpt, dict) and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and isinstance(ckpt, dict) and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt
