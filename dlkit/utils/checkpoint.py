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


def load_model_weights(path, model, device=None, strict=False):
    """仅加载模型权重(迁移初始化/微调),不恢复 optimizer、epoch 等训练状态。

    与 `load_checkpoint` 的区别:用于"预训练 -> 微调"或冷启动权重初始化,
    不会把 epoch 续到预训练轮次,也不会恢复优化器动量。

    返回 ``torch.nn.modules.module._IncompatibleKeys``(missing_keys/unexpected_keys)。
    """

    map_location = device if device is not None else 'cpu'
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt.get('model') if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    if not isinstance(state, dict):
        raise ValueError('Checkpoint %s does not contain model weights' % (path,))
    return model.load_state_dict(state, strict=strict)
