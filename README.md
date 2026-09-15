# BlueprintDL —— 模块化即插即用深度学习框架 (dlkit)

一个以 **注册中心 (Registry) + 配置驱动 (Config)** 为核心的模块化深度学习流水线框架。
你可以像搭积木一样自由组合数据、模型、损失、指标、优化器，快速搭建 **分割 / 分类 / 检测 / 深度估计**
等任意网络，并完成 **数据清洗 → 训练 → 评估 → 可视化 → 导出部署** 的一条龙流水线。

## 项目锻造仓库

BlueprintDL 现在也可以作为具体深度学习项目的孵化仓库。框架能力继续保留在 `dlkit/`，具体项目放在 `projects/<project_id>/`，从创建开始就拥有独立的源码包、配置、测试、依赖声明和交付清单。

首个项目骨架是 `projects/polypmeasure/`。Forge 提供以下入口：

```bash
# 查看仓库中的项目
uv run python tools/forge.py list

# 查看项目身份以及由它提供的注册组件
uv run python tools/forge.py inspect polypmeasure

# 查看 Core 提供的组件及其来源
uv run python tools/forge.py components --provider blueprintdl

# 检查包边界、组件归属、插件入口和交付路径
uv run python tools/forge.py audit polypmeasure

# 创建另一个独立项目骨架
uv run python tools/forge.py new another-project --display-name "Another Project"

# 通过 project.yaml 中的 delivery.include 导出独立项目
uv run python tools/forge.py export polypmeasure dist/projects
```

项目组件使用 `provider=<project_id>` 登记来源，因此可以追踪某个项目提供了哪些 Dataset、Model、Loss、Metric 或其他插件。具体项目默认不加入根 uv workspace，各自保留环境与依赖边界，避免不同 PyTorch/GPU 方案互相锁定。

完整约束见 [`standards/PROJECT_STANDARD.md`](standards/PROJECT_STANDARD.md)。

框架的核心是 **任务无关（task-agnostic）**：`Trainer` 不关心你训练的是分割还是检测，
数据、模型、损失、指标都遵循统一的接口约定，新任务 = 往注册表里加组件 + 写 yaml。

## 目录结构

```
├── configs/                  # 流水线配置文件（yaml）
│   ├── unet_toy.yaml  deeplabv3_toy.yaml  deeplabv3plus_toy.yaml   # 分割
│   ├── cls_toy.yaml                                           # 分类
│   ├── det_toy.yaml                                           # 检测(FCOS)
│   ├── depth_toy.yaml                                         # 深度估计
│   ├── coloncrafter.yaml                                      # 内镜深度(ColonCrafter)
│   └── clean_toy.yaml                                         # 数据清洗
├── dlkit/                    # 核心框架包
│   ├── registry.py           # 注册中心：即插即用的核心
│   ├── data/                 # 数据：清洗 / 数据集 / 增强 / 加载器
│   ├── models/               # 模型：backbone / decoder(neck) / head / loss / 完整网络(zoo)
│   ├── metrics/              # 评价指标（分割 IoU / 分类 Acc / 检测 mAP / 深度 RMSE）
│   ├── engine/               # 训练引擎 + 回调钩子 + 评估
│   ├── visualize/            # 训练曲线 / 分割 / 深度可视化
│   ├── utils/                # checkpoint / device / logger / seed / batch / 检测算子 / 深度校准
│   └── deploy/               # ONNX / TorchScript 导出与推理
├── tools/                    # 命令行入口脚本
│   ├── train.py  evaluate.py  predict.py  predict_det.py  predict_depth.py
│   ├── clean_data.py  export.py
│   └── make_toy_data.py  make_toy_cls_data.py  make_toy_det_data.py  make_toy_depth_data.py
└── tests/                    # 测试（纯 Python / 需 torch 分开）
```

## 快速开始

```bash
# 1. 安装 uv 后，同步项目环境
uv sync

# 项目面向 GPU；具体 PyTorch/CUDA 构建由部署环境选择，框架本身不写死硬件后端
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# ---- 分割 ----
uv run python tools/make_toy_data.py --root data/toy --samples 120
uv run python tools/train.py --config configs/unet_toy.yaml

# ---- 分类 ----
uv run python tools/make_toy_cls_data.py --root data/cls_toy --samples-per-class 30
uv run python tools/train.py --config configs/cls_toy.yaml

# ---- 检测（FCOS）----
uv run python tools/make_toy_det_data.py --root data/det_toy --samples 120
uv run python tools/train.py --config configs/det_toy.yaml
uv run python tools/predict_det.py --config configs/det_toy.yaml \
       --checkpoint runs/det_toy/best.pth --input data/det_toy/val/images \
       --classes data/det_toy/classes.json --output-dir runs/predict_det

# ---- 深度估计 ----
uv run python tools/make_toy_depth_data.py --root data/depth_toy --samples 120
uv run python tools/train.py --config configs/depth_toy.yaml

# ---- 内镜深度（ColonCrafter，仅推理，研究用途）----
# 先按 configs/coloncrafter.yaml 注释 clone 官方仓库并装依赖
uv run python tools/predict_coloncrafter.py --config configs/coloncrafter.yaml \
       --input data/c3vd/cecum_t1_a/color \
       --gt-dir data/c3vd/cecum_t1_a/depth \
       --output-dir runs/predict_coloncrafter

# 评估 / 推理 / 导出（以分割为例）
uv run python tools/evaluate.py --config configs/unet_toy.yaml --checkpoint runs/unet_toy/best.pth
uv run python tools/predict.py --config configs/unet_toy.yaml \
       --checkpoint runs/unet_toy/best.pth --input data/toy/val/images \
       --palette data/toy/classes.json --output-dir runs/predict
uv run python tools/export.py --config configs/unet_toy.yaml \
       --checkpoint runs/unet_toy/best.pth --format onnx --output runs/unet_toy/model.onnx
```

## 核心概念：任务无关协议

这是框架最关键的设计。`Trainer` 不硬编码任何任务，而是遵循四条约定，
从而让分割 / 分类 / 检测 / 深度「搭积木」式接入：

| 组件 | 约定 | 分割 | 分类 | 检测 |
|------|------|------|------|------|
| Dataset 样本键 | dict，至少含 `image` | `mask` | `label` | `boxes`+`labels` |
| collate | 默认 torch；变长标签由数据集提供 `collate_fn` | - | - | ✓ |
| model 输出 | `forward(image) -> preds` | tensor logits | tensor logits | dict |
| loss 签名 | `loss(preds, batch) -> 标量` | ✓ | ✓ | ✓ |
| metric 签名 | `metric.update(preds, batch)` | ✓ | ✓ | ✓ |

也就是说：**每个任务只是「数据集 + 模型 + 损失 + 指标」的一组组件**，Trainer 永远不变。
新增任务不需要改引擎，只需写组件并注册。

## 注册中心（即插即用）

所有可替换组件都通过 `Registry` 注册，配置里只用 `type + params` 声明，框架自动实例化：

```python
from dlkit.registry import BACKBONES
import torch.nn as nn

@BACKBONES.register()
class MyBackbone(nn.Module):
    def __init__(self, in_channels=3):
        ...
        self.out_channels = [32, 64, 128]   # 各阶段输出通道，供 decoder 自动拼接
        self.strides = [4, 8, 16]           # 各阶段下采样倍率，供检测推算
```

```yaml
model:
  type: UNet
  params:
    backbone:
      type: MyBackbone          # 即插即用
```

内置注册表：`BACKBONES` `DECODERS` `HEADS` `MODELS` `LOSSES` `DATASETS`
`TRANSFORMS` `CLEANERS` `METRICS` `OPTIMIZERS` `SCHEDULERS`。

`build_from_cfg` 会**递归**解析配置里所有 `type/params`，loss 里嵌套子 loss、
model 里嵌套 backbone/neck/head，都会自动按顺序构建。

## 模型组装

模型统一为三段式 `backbone + decoder(neck) + head`，通过配置自由组合：

| 任务 | 模型 | backbone | decoder/neck | head |
|------|------|----------|--------------|------|
| 分割 | UNet | 任意 | UNetDecoder | SegHead |
| 分割 | DeepLabV3 | 任意 | ASPP | SegHead |
| 分割 | DeepLabV3+ | 任意 | ASPP + 低层融合 | SegHead |
| 分类 | Classifier | 任意 | - | GlobalPoolHead |
| 检测 | FCOS | 任意 | FPN | FCOSHead |
| 深度 | DepthNet | 任意 | UNetDecoder | LogDepthHead |
| 内镜深度 | ColonCrafter | - | - | -（扩散模型，仅推理） |

内置 backbone：`SimpleCNN`（手写轻量）、`ResNet`（手写，18/34/50/101，可加载 torchvision 预训练）、
`TorchvisionBackbone`（torchvision 预训练）。

```yaml
# 换 torchvision 预训练 ResNet50 做分割
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
uv run pytest

# 也可以单独运行某个测试文件
uv run python tests/test_registry.py
```

## 可选依赖

```bash
# ONNX 导出/推理
uv sync --extra export

# TensorBoard
uv sync --extra tensorboard

# 安装所有可选依赖
uv sync --all-extras
```
