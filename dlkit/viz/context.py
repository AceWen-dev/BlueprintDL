"""可视化运行上下文:把一次 run 的配置、历史、模型与验证加载器打包,供可视化组件使用。"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

# 组件注册副作用:重建 Dataset/模型前必须先导入
import dlkit.data  # noqa: F401
import dlkit.models  # noqa: F401

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclasses.dataclass
class RunContext:
    """一次 run 的可视化上下文。

    - 必有:run_dir / config
    - 曲线类可视化只需要 history;
    - 预测类可视化(分割叠加、混淆矩阵、检测框、深度图)依赖 model 与
      val_loader,二者缺失时相关组件应跳过并提示。
    """

    run_dir: Path
    config: dict[str, Any]
    history: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    model: Any = None
    val_loader: Any = None
    device: Any = None
    checkpoint_path: Path | None = None
    experiment: str | None = None
    project: str | None = None
    num_classes: int | None = None
    max_samples: int = 4
    mean: tuple = IMAGENET_MEAN
    std: tuple = IMAGENET_STD


def _find_normalize_params(node: Any, key: str):
    """在(嵌套)配置里查找最近的 Normalize mean/std 参数。"""

    if isinstance(node, dict):
        if (node.get('type') or '').lower() == 'normalize' and key in node.get(
            'params', {}
        ):
            return node['params'][key]
        for value in node.values():
            found = _find_normalize_params(value, key)
            if found is not None:
                return found
    elif isinstance(node, (list, tuple)):
        for item in node:
            found = _find_normalize_params(item, key)
            if found is not None:
                return found
    return None


def resolve_norm_stats(config: dict) -> tuple[tuple, tuple]:
    """从配置的 val transform 中提取 Normalize mean/std,缺省用 ImageNet 统计。"""

    data_cfg = config.get('data') or {}
    mean = _find_normalize_params(data_cfg.get('val') or data_cfg, 'mean') or IMAGENET_MEAN
    std = _find_normalize_params(data_cfg.get('val') or data_cfg, 'std') or IMAGENET_STD
    return tuple(float(v) for v in mean), tuple(float(v) for v in std)


def denormalize_image(image: np.ndarray, mean: tuple, std: tuple) -> np.ndarray:
    """把归一化的 HWC 图像还原到 [0,1] 显示区间。"""

    if image.ndim == 3 and image.shape[-1] == 3:
        return np.clip(image * np.asarray(std) + np.asarray(mean), 0.0, 1.0)
    return np.clip(image, 0.0, 1.0)


def build_run_context(
    run_dir: str | Path,
    *,
    load_model: bool = False,
    device: Any = None,
    max_samples: int = 4,
) -> RunContext:
    """从 run 目录构建完整可视化上下文。

    run_dir 需要包含 config.yaml(或 run.json 内嵌 config)。
    load_model=True 时尝试构建模型并加载 checkpoint(best.pth 优先),
    用真实配置重建 val DataLoader;任何失败会抛出异常,由调用方决定降级。
    """

    import json

    from dlkit.utils.config import load_config
    from dlkit.utils.runs import read_run_manifest

    run_dir = Path(run_dir)
    config = {}
    manifest = read_run_manifest(run_dir)
    if manifest and isinstance(manifest.get('config'), dict):
        config = dict(manifest['config'])
    elif (run_dir / 'config.yaml').is_file():
        config = load_config(str(run_dir / 'config.yaml'))

    history: list[dict[str, Any]] = []
    metrics_path = run_dir / 'metrics.jsonl'
    if metrics_path.is_file():
        with metrics_path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    history.append(row)

    model = None
    val_loader = None
    checkpoint_path = None
    if load_model and config:
        checkpoint_path = _pick_checkpoint(run_dir)
        if checkpoint_path is not None:
            from dlkit.builders import build_dataloaders, build_from_cfg
            from dlkit.utils.checkpoint import load_checkpoint
            from dlkit.utils.plugins import load_plugin_modules

            load_plugin_modules(config.get('imports') or [])

            if device is None:
                import torch

                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            loaders = build_dataloaders(config['data'])
            val_loader = loaders[1] if isinstance(loaders, (tuple, list)) else loaders
            model = build_from_cfg(config['model']).to(device).eval()
            load_checkpoint(checkpoint_path, model=model, device=device)

    model_cfg = (config.get('model') or {}).get('params') or {}
    return RunContext(
        run_dir=run_dir,
        config=config,
        history=history,
        model=model,
        val_loader=val_loader,
        device=device,
        checkpoint_path=checkpoint_path,
        experiment=manifest.get('experiment') if manifest else run_dir.parent.name,
        project=manifest.get('project') if manifest else run_dir.parent.parent.name,
        num_classes=model_cfg.get('num_classes'),
        max_samples=max_samples,
    )


def _pick_checkpoint(run_dir: Path) -> Path | None:
    best = run_dir / 'best.pth'
    last = run_dir / 'last.pth'
    return best if best.is_file() else (last if last.is_file() else None)


__all__ = [
    'IMAGENET_MEAN',
    'IMAGENET_STD',
    'RunContext',
    'build_run_context',
    'denormalize_image',
    'resolve_norm_stats',
]
