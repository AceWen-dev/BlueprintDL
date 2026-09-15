# BlueprintDL：配置驱动的深度学习框架与项目锻造仓库

BlueprintDL 是一个面向学习、实验和项目孵化的深度学习框架。它通过 Registry、可插拔组件、Builder、依赖注入和薄 Pipeline，把“选什么组件”“如何构造对象”和“训练流程如何运行”分开。

这个仓库同时承担两种职责：

- `dlkit/` 提供可复用的 BlueprintDL Core；
- `projects/` 在统一工程规则下孵化具体项目，并在成熟后独立导出交付。

框架面向 GPU 使用场景，但不会在通用代码中固定某个 CUDA、PyTorch wheel、驱动或显卡型号。硬件选择属于运行与部署环境，应用配置只表达 `auto`、`cuda`、AMP 等逻辑策略。

## 设计思路

```text
YAML 配置
   │
   ▼
Builder / Factory ───── 注入 model.parameters()、optimizer、device 等运行时对象
   │
   ▼
Registry ────────────── 根据稳定名称找到组件实现
   │
   ▼
Dataset / Model / Loss / Metric / Optimizer / Scheduler
   │
   ▼
Trainer / Pipeline ──── 只编排训练、验证、checkpoint 和 hook
```

四层分别负责：

| 层 | 负责 | 不负责 |
| --- | --- | --- |
| Registry | 名称到实现的映射、重复检查、组件来源 | 解析整份配置、训练和资源初始化 |
| Component | Dataset、Model、Loss、Metric 等具体能力 | 中央选择逻辑和训练流程 |
| Builder | 校验配置、构造组件、注入运行时依赖 | 长时间训练循环 |
| Pipeline / Trainer | 生命周期、执行顺序和数据流 | 维护组件名字表和具体类型分支 |

理想情况下，新增一个组件只需要修改：

```text
组件实现 + 注册 + 配置 + 测试
```

## 仓库结构

```text
BlueprintDL/
├── dlkit/                         # BlueprintDL Core
│   ├── registry.py                # Registry、构造入口和组件来源记录
│   ├── data/                      # Dataset、Transform、Cleaner、DataLoader Builder
│   ├── models/                    # Backbone、Decoder、Head、Loss、完整模型
│   ├── metrics/                   # 分类、分割、检测和深度指标
│   ├── engine/                    # Trainer、Optimizer、Scheduler、Callback、Evaluate
│   ├── deploy/                    # ONNX / TorchScript 导出和推理
│   ├── forge/                     # 项目清单、审计、创建和导出能力
│   └── utils/                     # 配置、device、checkpoint、日志等工具
├── configs/                       # Core 示例配置
├── tools/                         # 训练、评估、推理、导出和 Forge 命令
├── tests/                         # Core 测试
├── standards/                     # 项目锻造和交付规则
├── projects/                      # 独立的具体项目
│   └── polypmeasure/
├── pyproject.toml                 # Core 依赖与打包声明
└── uv.lock                        # Core 可复现依赖锁
```

依赖方向固定为：

```text
具体项目 -> BlueprintDL Core 公共 API
Core      -X-> 具体项目
项目 A    -X-> 项目 B 内部代码
交付物    -X-> 仓库中未声明的隐藏文件
```

## 快速开始

### 1. 准备环境

安装 uv 后，在仓库根目录执行：

```powershell
uv sync
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

`uv sync` 根据 `pyproject.toml` 和 `uv.lock` 创建 `.venv`。项目面向 GPU，训练前应确认当前环境中的 PyTorch 与目标 GPU 驱动兼容；框架不会自动替你下载或切换 CUDA。

更完整的依赖与 GPU 环境说明见 [uv 使用说明](UV使用说明.md)。

### 2. 跑通第一个分割任务

```powershell
uv run python tools/make_toy_data.py --root data/toy --samples 120
uv run python tools/train.py --config configs/unet_toy.yaml
uv run python tools/evaluate.py --config configs/unet_toy.yaml --checkpoint runs/unet_toy/best.pth
uv run python tools/predict.py --config configs/unet_toy.yaml --checkpoint runs/unet_toy/best.pth --input data/toy/val/images --palette data/toy/classes.json --output-dir runs/predict
```

想先验证流程，可以临时把训练轮数降为 1：

```powershell
uv run python tools/train.py --config configs/unet_toy.yaml --opts train.epochs=1
```

`--opts` 使用点号覆盖嵌套配置，例如：

```powershell
uv run python tools/train.py --config configs/unet_toy.yaml --opts train.epochs=50 optimizer.params.lr=0.0001
```

## 配置驱动构造

所有普通组件使用 `type + params`：

```yaml
model:
  type: UNet
  params:
    in_channels: 3
    num_classes: 4
    backbone:
      type: SimpleCNN
      params:
        channels: [16, 32, 64, 128]

optimizer:
  type: AdamW
  params:
    lr: 0.001
    weight_decay: 0.0001
```

`build_from_cfg` 将注册名解析成类，并递归构造普通嵌套组件。需要根据上游组件推导参数的组合模型可以声明 `_manual_build = True`，由模型内部的局部组装逻辑补充通道数等派生参数。

Optimizer 和 Scheduler 仍然由配置选择，但通过专用构造步骤注入运行时依赖：

```text
先构造 Model
  -> 构造 Optimizer，并注入 model.parameters()
  -> 构造 Scheduler，并注入 optimizer
```

`model.parameters()` 和 optimizer 实例不是可序列化数据，因此不应写进 YAML。

## 任务无关训练协议

`Trainer` 通过统一协议支持分割、分类、检测和深度估计：

| 组件 | 协议 |
| --- | --- |
| Dataset | 返回至少包含 `image` 的字典；标签键由任务定义 |
| Model | `forward(image) -> predictions` |
| Loss | `loss(predictions, batch) -> scalar` |
| Metric | `reset()`、`update(predictions, batch)`、`compute() -> dict` |
| DataLoader | 变长标注可由 Dataset 提供 `collate_fn` |

当前示例配置：

| 任务 | 配置 |
| --- | --- |
| 分割 | `configs/unet_toy.yaml`、`deeplabv3_toy.yaml`、`deeplabv3plus_toy.yaml` |
| 分类 | `configs/cls_toy.yaml` |
| 检测 | `configs/det_toy.yaml` |
| 深度估计 | `configs/depth_toy.yaml` |
| 数据清洗 | `configs/clean_toy.yaml` |
| 研究型内镜深度推理 | `configs/coloncrafter.yaml` |

可以从注册表动态查看内置组件，不需要手工维护容易过期的列表：

```powershell
uv run blueprintdl-forge components --provider blueprintdl
uv run blueprintdl-forge components --provider blueprintdl --registry models
uv run blueprintdl-forge components --provider blueprintdl --registry losses --json
```

## 注册新组件

Core 组件实现在 `dlkit/` 内，来源会自动记录为 `provider=blueprintdl`：

```python
import torch.nn as nn

from dlkit.registry import LOSSES


@LOSSES.register(name="MyLoss", tags=("segmentation",))
class MyLoss(nn.Module):
    def forward(self, predictions, batch):
        ...
```

具体项目组件必须放在对应项目包中，并显式声明项目 ID：

```python
@LOSSES.register(
    name="BoundaryLoss",
    provider="polypmeasure",
    tags=("segmentation", "boundary"),
)
class BoundaryLoss:
    ...
```

项目通过 `project.yaml` 中的 bootstrap 显式加载插件。注册阶段只能导入和登记组件，不能训练、读数据、下载权重或占用 GPU。

## 使用项目锻造仓库

### 查看和审计项目

```powershell
uv run blueprintdl-forge list
uv run blueprintdl-forge inspect polypmeasure
uv run blueprintdl-forge audit polypmeasure
```

`inspect` 从真实注册记录生成组件清单；`audit` 检查项目身份、Python 元数据、源码包、bootstrap、交付白名单、跨项目导入和 provider 归属。

### 创建新项目

```powershell
uv run blueprintdl-forge new retinal-seg --display-name "Retinal Segmentation"
```

生成的项目位于 `projects/retinal-seg/`，拥有自己的：

- `project.yaml`；
- `pyproject.toml`；
- `src/<package>/`；
- `configs/`；
- `tests/`；
- `README.md`。

具体项目默认不加入根 uv workspace。它们可以独立选择 Python、PyTorch 和 GPU 依赖策略，避免不同客户或硬件环境互相锁定。

### 导出独立项目

```powershell
uv run blueprintdl-forge audit polypmeasure
uv run blueprintdl-forge export polypmeasure dist/projects
```

Forge 会创建 `dist/projects/polypmeasure/`，只复制 `project.yaml` 中 `delivery.include` 允许的内容，并生成 `.blueprintdl-export.json` 审计回执。已有目标目录不会被覆盖。

当前导出模型是“具体项目 + 版本化 BlueprintDL Core 依赖”。交付前需要保证目标环境能够安装项目声明的兼容 Core 版本。

完整规则见 [BlueprintDL 项目锻造标准](standards/PROJECT_STANDARD.md)。

## 训练、评估、续训和导出

```powershell
# 训练
uv run python tools/train.py --config configs/unet_toy.yaml

# 从 last.pth 续训
uv run python tools/train.py --config configs/unet_toy.yaml --resume runs/unet_toy/last.pth

# 评估
uv run python tools/evaluate.py --config configs/unet_toy.yaml --checkpoint runs/unet_toy/best.pth

# 使用 GPU 导出 ONNX
uv run python tools/export.py --config configs/unet_toy.yaml --checkpoint runs/unet_toy/best.pth --format onnx --device cuda --output runs/unet_toy/model.onnx
```

训练目录默认包含配置快照、`metrics.jsonl`、`last.pth` 和按配置选出的 `best.pth`。

## 测试和构建

```powershell
# Core 测试
uv run pytest -q

# 检查依赖锁
uv lock --check

# 构建 BlueprintDL Core
uv build

# 构建具体项目
uv build --project projects/polypmeasure
```

Core 与具体项目应分别测试和构建。Core 的 wheel/sdist 不会自动包含 `projects/`。首次构建可能需要由 uv 获取 `hatchling` 构建后端。

## 文档导航

- [完整框架教程](教程.md)：从配置到自定义组件、具体项目和交付的完整学习路径；
- [项目锻造标准](standards/PROJECT_STANDARD.md)：Core/项目边界、provider、bootstrap 和交付门禁；
- [uv 使用说明](UV使用说明.md)：环境、依赖、锁文件和 GPU 安装边界；
- [Git 使用说明](Git使用说明.md)：提交、分支、同步和 Forge 项目改动建议；
- [Docker 使用说明](Docker使用说明.md)：容器基础、GPU 运行与独立项目镜像；
- [AI 辅助项目开发指南](AI辅助项目开发指南.md)：如何用工程约束让 Coding Agent 稳定协作；
- [具体项目说明](projects/README.md)：仓库中的项目如何创建、开发、审计和导出。

## 当前状态

- BlueprintDL Core 已提供注册表、配置构造、通用训练引擎和多任务示例；
- Registry 已支持 `provider`、模块、限定名称、标签和描述等来源记录；
- Forge 已支持 `list`、`components`、`inspect`、`audit`、`new` 和 `export`；
- `polypmeasure` 已建立独立项目边界，业务 Dataset、Model、Loss 和 Metric 将在需求明确后迁入该项目。
