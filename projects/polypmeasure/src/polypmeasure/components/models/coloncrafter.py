"""ColonCrafter endoscopic depth-estimation plugin for PolypMeasure.

The upstream implementation and model assets are loaded lazily. Importing this
module only declares the plugin; it does not download weights or initialize CUDA.
The upstream project is research-only/non-commercial, so review its licenses
before distributing or using this integration.
"""

from contextlib import contextmanager
import importlib
import os
import sys

import torch
import torch.nn as nn

from dlkit.registry import MODELS


@contextmanager
def _temporary_import_paths(paths):
    added = []
    for path in reversed(tuple(paths)):
        if path not in sys.path:
            sys.path.insert(0, path)
            added.append(path)
    importlib.invalidate_caches()
    try:
        yield
    finally:
        for path in added:
            try:
                sys.path.remove(path)
            except ValueError:
                pass
        importlib.invalidate_caches()


def _import_coloncrafter(coloncrafter_dir):
    """Lazily import the upstream inference class with actionable guidance."""
    if coloncrafter_dir is None:
        raise RuntimeError(
            "ColonCrafter 需要官方实现。请先执行:\n"
            "  git clone https://github.com/rajpurkarlab/ColonCrafter <dir>\n"
            "  cd <dir> && git submodule update --init --recursive\n"
            "然后在 model.params 中设置 coloncrafter_dir: <dir>"
        )
    coloncrafter_dir = os.path.abspath(os.fspath(coloncrafter_dir))
    if not os.path.isdir(coloncrafter_dir):
        raise RuntimeError("ColonCrafter 目录不存在: %s" % coloncrafter_dir)
    depthcrafter_dir = os.path.join(
        coloncrafter_dir, "src", "submodules", "DepthCrafter"
    )
    if not os.path.isdir(depthcrafter_dir):
        raise RuntimeError(
            "DepthCrafter submodule 未初始化，请执行: cd %s && "
            "git submodule update --init --recursive" % coloncrafter_dir
        )
    source_dir = os.path.join(coloncrafter_dir, "src")
    try:
        with _temporary_import_paths((coloncrafter_dir, source_dir)):
            module = importlib.import_module("coloncrafter")
            return module.ColonCrafterInference
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "导入官方 ColonCrafter 失败。请根据其 README，在项目自己的 GPU "
            "环境中安装匹配的依赖；不要把目标机器的 CUDA 组合写死到 Core。\n"
            "原始错误: %s" % exc
        ) from exc


@MODELS.register(
    provider="polypmeasure",
    tags=("depth-estimation", "endoscopy", "research"),
    description="ColonCrafter relative-depth inference adapter for PolypMeasure.",
)
class ColonCrafter(nn.Module):
    """Lazy adapter around the upstream ColonCrafter implementation."""

    def __init__(
        self,
        model_path="romainhardy/coloncrafter",
        coloncrafter_dir=None,
        device="cuda",
    ):
        super().__init__()
        self.model_path = model_path
        self.coloncrafter_dir = coloncrafter_dir
        self.device = device
        self._impl = None

    def _lazy_build(self):
        if self._impl is None:
            implementation = _import_coloncrafter(self.coloncrafter_dir)
            self._impl = implementation.from_pretrained(
                self.model_path, device=self.device
            )
        return self._impl

    @torch.inference_mode()
    def forward(
        self,
        video,
        num_inference_steps=1,
        window_size=16,
        overlap=8,
        guidance_scale=1.0,
        seed=42,
    ):
        """Predict relative depth for an ``(N, C, H, W)`` video tensor."""
        video = video.to(self.device)
        depth, _ = self._lazy_build().predict_depth(
            video,
            num_inference_steps=num_inference_steps,
            window_size=window_size,
            overlap=overlap,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return torch.as_tensor(depth, dtype=torch.float32)

    def predict_depth(self, video, **kwargs):
        """Expose the upstream ``(depth, disparity)`` result as tensors."""
        depth, disparity = self._lazy_build().predict_depth(video, **kwargs)
        return (
            torch.as_tensor(depth, dtype=torch.float32),
            torch.as_tensor(disparity, dtype=torch.float32),
        )
