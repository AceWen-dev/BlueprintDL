"""ColonCrafter 内镜深度估计组件（框架封装）。

ColonCrafter (arXiv 2509.13525) 是扩散先验驱动的结肠镜视频深度估计模型，
本质 = 腾讯 DepthCrafter UNet（SVD-XT 视频扩散架构）+ LoRA 微调 + 1 步去噪 +
16 帧滑动窗口推理，输出时序一致的相对深度图。

注意：
- 官方实现不在 pip 上，需要先 clone 官方仓库并初始化 DepthCrafter submodule，
  再通过 ``coloncrafter_dir`` 指定其路径；本模块对该依赖做**惰性导入**，
  不影响框架其他任务的轻量性。
- 官方代码与 DepthCrafter 均为非商用许可，仅限研究/学术用途。
- 输出为相对深度（无绝对尺度），用于米级测量前需用真值做尺度对齐（见
  tools/predict_coloncrafter.py 的 --gt-dir / --scale 参数）。
"""

import os
import sys

import torch
import torch.nn as nn

from dlkit.registry import MODELS


def _import_coloncrafter(coloncrafter_dir):
    """惰性导入官方 ColonCrafterInference，并给出清晰的环境指引。"""
    if coloncrafter_dir is None:
        raise RuntimeError(
            'ColonCrafter 需要官方实现。请先执行:\n'
            '  git clone https://github.com/rajpurkarlab/ColonCrafter <dir>\n'
            '  cd <dir> && git submodule update --init --recursive\n'
            '然后在配置 model.params 里设置 coloncrafter_dir: <dir>'
        )
    if not os.path.isdir(coloncrafter_dir):
        raise RuntimeError('ColonCrafter 目录不存在: %s' % coloncrafter_dir)
    depthcrafter_dir = os.path.join(coloncrafter_dir, 'src', 'submodules', 'DepthCrafter')
    if not os.path.isdir(depthcrafter_dir):
        raise RuntimeError(
            'DepthCrafter submodule 未初始化，请执行: cd %s && git submodule update --init --recursive'
            % coloncrafter_dir
        )
    src_dir = os.path.join(coloncrafter_dir, 'src')
    for p in (coloncrafter_dir, src_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from coloncrafter import ColonCrafterInference
    except ImportError as e:
        raise RuntimeError(
            '导入官方 ColonCrafter 失败，请先按其 README 安装依赖\n'
            '（torch 2.1+cu118、xformers、diffusers、peft 等）: pip install -r %s/requirements.txt\n'
            '原始错误: %s' % (coloncrafter_dir, e)
        )
    return ColonCrafterInference


@MODELS.register()
class ColonCrafter(nn.Module):
    """ColonCrafter 深度估计推理组件。

    forward(video) 输入 (N,C,H,W) 的 [0,1] 帧序列，输出 (N,H,W) 相对深度图。
    模型权重在首次调用时惰性加载（HF repo id 或本地目录）。
    """

    def __init__(self, model_path='romainhardy/coloncrafter', coloncrafter_dir=None,
                 device=None):
        super().__init__()
        self.model_path = model_path
        self.coloncrafter_dir = coloncrafter_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._impl = None

    def _lazy_build(self):
        if self._impl is None:
            cls = _import_coloncrafter(self.coloncrafter_dir)
            self._impl = cls.from_pretrained(self.model_path, device=self.device)
        return self._impl

    @torch.inference_mode()
    def forward(self, video, num_inference_steps=1, window_size=16, overlap=8,
                guidance_scale=1.0, seed=42):
        """预测深度。

        video: (N, C, H, W)，值域 [0,1]（CPU/GPU 均可，内部自动搬到模型设备）。
        返回 depth: (N, H, W)，相对深度（无绝对尺度）。
        """
        video = video.to(self.device)
        impl = self._lazy_build()
        depth, disparity = impl.predict_depth(
            video,
            num_inference_steps=num_inference_steps,
            window_size=window_size,
            overlap=overlap,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return torch.as_tensor(depth, dtype=torch.float32)

    def predict_depth(self, video, **kwargs):
        """对齐官方接口：返回 (depth, disparity)。"""
        impl = self._lazy_build()
        depth, disparity = impl.predict_depth(video, **kwargs)
        return torch.as_tensor(depth, dtype=torch.float32), torch.as_tensor(disparity, dtype=torch.float32)
