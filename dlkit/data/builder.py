#此段代码用于构建训练集和测试集的数据加载器，主要包括两个函数：`build_loader` 和 `build_dataloaders`。
from torch.utils.data import DataLoader

from dlkit.registry import build_from_cfg


def build_loader(data_cfg, split, shuffle=None):
    if split not in data_cfg:
        raise KeyError(
            'split %r not in data config; available splits: %s'
            % (split, [k for k in data_cfg if k != 'loader'])
        )
    dataset_cfg = data_cfg[split]
    if not isinstance(dataset_cfg, dict) or 'type' not in dataset_cfg:
        raise ValueError('dataset config for split %r must be a dict with a "type" key' % split)
    dataset = build_from_cfg(dataset_cfg)

    # 任务可自定义 collate：检测等变长标签任务需要把 boxes/labels 打包成
    # list-of-tensors，而不是 torch 默认的堆叠。优先用数据集自带的 collate_fn。
    collate_fn = getattr(dataset, 'collate_fn', None)

    loader_cfg = dict(data_cfg.get('loader') or {})
    if shuffle is None:
        shuffle = loader_cfg.pop('shuffle', True) if split == 'train' else False
    else:
        loader_cfg.pop('shuffle', None)

    batch_size = loader_cfg.pop('batch_size', 1)
    num_workers = loader_cfg.pop('num_workers', 0)
    pin_memory = loader_cfg.pop('pin_memory', False)
    drop_last = loader_cfg.pop('drop_last', False)
    persistent_workers = loader_cfg.pop('persistent_workers', False)
    if loader_cfg:
        raise ValueError('unknown loader options: %s' % sorted(loader_cfg))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )


def build_dataloaders(data_cfg):
    return build_loader(data_cfg, 'train'), build_loader(data_cfg, 'val')
