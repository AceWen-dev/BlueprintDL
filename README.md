# 从 0 手搓深度学习项目 —— 模块化即插即用框架 (dlkit)

一个以 **注册中心 (Registry) + 配置驱动 (Config)** 为核心的模块化深度学习流水线框架。
你可以像搭积木一样自由组合数据、模型、损失、指标、优化器，快速搭建 UNet、DeepLabV3(+)、
PSPNet 等任意分割/分类网络，并完成 **数据清洗 → 训练 → 评估 → 可视化 → 导出部署** 的一条龙流水线。

## 目录结构

```
├── configs/                  # 流水线配置文件（yaml）
│   ├── unet_toy.yaml
│   ├── deeplabv3_toy.yaml
│   ├── deeplabv3plus_toy.yaml
│   └── clean_toy.yaml
├── dlkit/                    # 核心框架包
│   ├── registry.py           # 注册中心：即插即用的核心
│   ├── data/                 # 数据：清洗 / 数据集 / 增强 / 加载器
│   ├── models/               # 模型：backbone / decoder / head / loss / 完整网络(zoo)
│   ├── metrics/              # 评价指标（IoU / Dice / Acc / FWIoU）
│   ├── engine/               # 训练引擎 + 回调钩子 + 评估
│   ├── visualize/            # 训练曲线 / 分割结果可视化
│   └── deploy/               # ONNX / TorchScript 导出与推理
├── tools/                    # 命令行入口脚本
│   ├── train.py  evaluate.py  predict.py
│   ├── clean_data.py  export.py  make_toy_data.py
└── tests/                    # 测试
```

## 快速开始

```bash
# 1. 安装依赖（CPU 版 PyTorch，Windows）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. 生成一个合成数据集（4 类：背景/圆/方块/三角）
python tools/make_toy_data.py --root data/toy --samples 120

# 3. 数据清洗（示例：检测损坏/尺寸不匹配/非法标签/重复）
python tools/clean_data.py --config configs/clean_toy.yaml

# 4. 训练 UNet
python tools/train.py --config configs/unet_toy.yaml

# 5. 评估
python tools/evaluate.py --config configs/unet_toy.yaml --checkpoint runs/unet_toy/best.pth

# 6. 推理并可视化
python tools/predict.py --config configs/unet_toy.yaml \
       --checkpoint runs/unet_toy/best.pth \
       --input data/toy/val/images --palette data/toy/classes.json \
       --output-dir runs/predict

# 7. 导出 ONNX / TorchScript
python tools/export.py --config configs/unet_toy.yaml \
       --checkpoint runs/unet_toy/best.pth --format onnx \
       --output runs/unet_toy/model.onnx
```

训练曲线绘制：

```python
from dlkit.visualize.curves import plot_training
plot_training('runs/unet_toy/metrics.jsonl', 'runs/unet_toy/curves.png')
```

## 核心概念：注册中心（即插即用）

所有可替换的组件都通过 `Registry` 注册，配置文件中只用 `type + params` 声明，框架自动实例化。

```python
from dlkit.registry import BACKBONES
import torch.nn as nn

@BACKBONES.register()
class MyBackbone(nn.Module):
    def __init__(self, in_channels=3):
        ...
        self.out_channels = [32, 64, 128]   # 各阶段输出通道，供 decoder 自动拼接
```

之后即可在配置里直接引用：

```yaml
model:
  type: UNet
  params:
    backbone:
      type: MyBackbone          # 即插即用
```

内置注册表：`BACKBONES` `DECODERS` `HEADS` `MODELS` `LOSSES` `DATASETS`
`TRANSFORMS` `CLEANERS` `METRICS` `OPTIMIZERS` `SCHEDULERS`。

`build_from_cfg` 会**递归**解析配置里所有 `type/params`，例如 loss 里嵌套多个子 loss、
model 里嵌套 backbone/decoder/head，都会自动按顺序构建，无需手写装配代码。

## 模型组装：UNet / DeepLabV3 / DeepLabV3+

模型统一为 `backbone + decoder + head` 三段式，通过配置自由组合：

| 网络 | backbone | decoder | head | 说明 |
|------|----------|---------|------|------|
| UNet | 任意 | UNetDecoder（上采样+跳跃连接） | SegHead | 编码-解码 |
| DeepLabV3 | 任意 | ASPP（空洞空间金字塔池化） | SegHead | 多尺度上下文 |
| DeepLabV3+ | 任意 | ASPP + 低层特征融合 | SegHead | 编码-解码 + ASPP |

内置 backbone：`SimpleCNN`（手写）、`ResNet`（手写，支持 18/34/50/101，可加载 torchvision 预训练）、
`TorchvisionBackbone`（torchvision 预训练，支持 output_stride 8/16/32）。

```yaml
# 换成 torchvision 预训练 ResNet50
model:
  type: DeepLabV3Plus
  params:
    num_classes: 19
    backbone:
      type: TorchvisionBackbone
      params: { name: resnet50, pretrained: true, output_stride: 16 }
```

## 配置结构说明

`type` 必须与注册名一致；`params` 为构造函数参数，内部可继续嵌套 `type/params`。
常用 `--opts` 命令行覆盖：`--opts train.epochs=50 optimizer.params.lr=0.0001`。

## 训练引擎与钩子（Hook）

`Trainer` 内置训练/验证/checkpoint 保存（`best.pth` / `last.pth`）、训练日志（`metrics.jsonl`）。
通过钩子注入自定义行为：

```python
from dlkit.engine.trainer import Trainer
from dlkit.engine.callbacks import EarlyStopping

trainer = Trainer(cfg)
EarlyStopping(metric='mIoU', mode='max', patience=10).attach(trainer)

def my_hook(trainer, epoch=None, **kwargs):
    print('epoch', epoch)

trainer.register_hook('after_epoch', my_hook)
trainer.train()
```

可用事件：`before_train` `after_train` `before_epoch` `after_epoch`
`before_train_step` `after_train_step`。

## 测试

```bash
python tests/test_registry.py     # 纯 Python，无需 torch
python tests/test_transforms.py
python tests/test_models.py       # 需要 torch
```

## 可选依赖

```bash
pip install -r requirements-optional.txt   # onnx / onnxruntime / tensorboard
```
