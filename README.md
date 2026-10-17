# BlueprintDL：配置驱动的深度学习实验框架

BlueprintDL 面向学习、实验和模型对比。它通过 Registry、可插拔组件、Builder、依赖注入和薄 Pipeline,把"选什么组件""如何构造对象"和"训练流程如何运行"分开。

你可以在 `configs/` 下为每个实验写一份 YAML 配置,在 `dlkit/` 中注册自己的模型、损失和指标,然后用统一的 `tools/` 脚本训练、评估、推理和导出,并在 `runs/` 下对比各次实验的结果。

框架面向 GPU 使用场景,但不会在 `dlkit/` 代码或 `uv.lock` 中固定某个 CUDA、PyTorch wheel、驱动或显卡型号。硬件选择属于运行与部署环境,应用配置只表达 `auto`、`cuda`、AMP 等逻辑策略;GPU 版 PyTorch 通过本机脚本 `scripts/setup_gpu.ps1` 安装,不进入通用锁文件。

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

四层分别负责:

| 层 | 负责 | 不负责 |
| --- | --- | --- |
| Registry | 名称到实现的映射、重复检查、组件来源 | 解析整份配置、训练和资源初始化 |
| Component | Dataset、Model、Loss、Metric 等具体能力 | 中央选择逻辑和训练流程 |
| Builder | 校验配置、构造组件、注入运行时依赖 | 长时间训练循环 |
| Pipeline / Trainer | 生命周期、执行顺序和数据流 | 维护组件名字表和具体类型分支 |

理想情况下,新增一个组件只需要修改:

```text
组件实现 + 注册 + 配置 + 测试
```

## 仓库结构

```text
BlueprintDL/
├── dlkit/                         # 核心
│   ├── registry.py                # Registry、构造入口和组件来源记录
│   ├── data/                      # Dataset、Transform、Cleaner、DataLoader Builder
│   ├── models/                    # Backbone、Decoder、Head、Loss、完整模型
│   ├── metrics/                   # 分类、分割、检测和深度指标
│   ├── engine/                    # Trainer、Optimizer、Scheduler、Callback、Evaluate
│   ├── deploy/                    # ONNX / TorchScript 导出和推理
│   ├── viz/                       # 标准可视化组件(曲线、三联图、混淆矩阵、检测框、深度图)
│   └── utils/                     # 配置、device、checkpoint、run 归档、日志等工具
├── configs/                       # 框架教学示例配置(project: examples)
├── tools/                         # 命令行入口
│   ├── train.py / evaluate.py     # 训练与评估
│   ├── exp/                       # 实验治理: visualize(可视化)、compare(对比)、project(生命周期)
│   ├── data/                      # 数据侧: prepare_data(体检)、clean_data(清洗)、make_toy_*(示例数据)
│   └── deploy/                    # 推理与导出: predict*.py、export.py
├── scripts/                       # 本机环境辅助脚本(如 setup_gpu.ps1,不属于组件体系)
├── projects/<project_id>/         # 真实实验项目:project.yaml + configs/ + extensions/ + reports/
├── runs/                          # 运行产物(不进 Git):runs/<project>/<experiment>/<时间戳>/
├── tests/                         # Core 测试
├── pyproject.toml                 # 依赖与打包声明
└── uv.lock                        # 可复现依赖锁
```

新写的组件直接放进 `dlkit/` 对应目录并注册;只属于某个项目的组件/Recipe 放 `projects/<id>/extensions/`,由配置的 `imports:` 加载。一次性分析脚本放 `tools/` 对应分组或 `scratch/`,不要让 `dlkit/` 反向依赖实验代码。

真实项目(如导师给的医学分割项目)在 `projects/<id>/` 下建立自己的 `configs/`;`configs/` 只保留框架教学示例。同时在项目配置文件头部声明实验身份:

```yaml
project: polyp        # 项目 id,决定 runs/<project>/ 分组
name: unet_base       # 实验名,决定 runs/<project>/<name>/ 分组
```

## 快速开始

### 1. 准备环境

安装 uv 后,在仓库根目录执行:

```powershell
uv sync
# Windows + NVIDIA GPU:把锁文件里的 CPU 版 torch 换成 CUDA 版
powershell -ExecutionPolicy Bypass -File scripts/setup_gpu.ps1
```

`uv.lock` 保持硬件中立(Windows 上 PyPI 的 torch 轮子默认是 CPU 版),GPU 版 PyTorch 属于本机环境,不写入锁文件。访问 PyTorch 官方源慢时,可使用镜像:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_gpu.ps1 -IndexUrl https://mirror.sjtu.edu.cn/pytorch-wheels/cu130
```

注意:`uv sync` 和普通的 `uv run` 会按锁文件恢复 CPU 版 torch。安装过 GPU 版之后,日常使用 `uv run --no-sync ...`(或先执行 `$env:UV_NO_SYNC = '1'`),修改依赖后再 `uv sync` 并重跑 `scripts/setup_gpu.ps1`。验证:

```powershell
uv run --no-sync python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

可选依赖按需安装:`uv sync --extra export`(ONNX)、`--extra tensorboard`、`--extra multimodal`(SPM-Net 的 smp + transformers)。GPU 机器装完 extras 后重跑一次 `scripts/setup_gpu.ps1 -SkipSync`。

> 已安装 CUDA 版 torch 的机器,下面所有 `uv run` 命令请改用 `uv run --no-sync`,否则 uv 会按锁文件把 torch 换回 CPU 版。

### 2. 跑通第一个分割任务

```powershell
uv run python tools/data/make_toy_data.py --root data/toy --samples 120
uv run python tools/train.py --config configs/unet_toy.yaml
uv run python tools/evaluate.py --config configs/unet_toy.yaml --checkpoint runs/examples/unet_toy/<timestamp>/best.pth
uv run python tools/deploy/predict.py --config configs/unet_toy.yaml --checkpoint runs/examples/unet_toy/<timestamp>/best.pth --input data/toy/val/images --palette data/toy/classes.json --output-dir runs/predict
```

想先验证流程,可以临时把训练轮数降为 1:

```powershell
uv run python tools/train.py --config configs/unet_toy.yaml --opts train.epochs=1
```

`--opts` 使用点号覆盖嵌套配置,例如:

```powershell
uv run python tools/train.py --config configs/unet_toy.yaml --opts train.epochs=50 optimizer.params.lr=0.0001
```

## 配置驱动构造

所有普通组件使用 `type + params`:

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

`build_from_cfg` 将注册名解析成类,并递归构造普通嵌套组件。需要根据上游组件推导参数的组合模型可以声明 `_manual_build = True`,由模型内部的局部组装逻辑补充通道数等派生参数。

Optimizer 和 Scheduler 仍然由配置选择,但通过专用构造步骤注入运行时依赖:

```text
先构造 Model
  -> 构造 Optimizer,并注入 model.parameters()
  -> 构造 Scheduler,并注入 optimizer
```

`model.parameters()` 和 optimizer 实例不是可序列化数据,因此不应写进 YAML。

## 任务无关训练协议

`Trainer` 通过统一协议支持分割、分类、检测和深度估计:

| 组件 | 协议 |
| --- | --- |
| Dataset | 返回至少包含 `image` 的字典;标签键由任务定义 |
| Model | `forward(image) -> predictions`;多模态模型可实现 `forward_batch(batch) -> predictions` 自行取模态 |
| Loss | `loss(predictions, batch) -> scalar` |
| Metric | `reset()`、`update(predictions, batch)`、`compute() -> dict` |
| DataLoader | 变长标注可由 Dataset 提供 `collate_fn` |

多模态模型(如 `spm-net`)的 `forward_batch` 会从 batch 读取 `immuno` / `clinical` / `text` 等键,具体约定见教程 8.4;单模态模型无需改动。

当前示例配置:

| 任务 | 配置 |
| --- | --- |
| 分割 | `configs/unet_toy.yaml`、`deeplabv3_toy.yaml`、`deeplabv3plus_toy.yaml` |
| 分类 | `configs/cls_toy.yaml` |
| 检测 | `configs/det_toy.yaml` |
| 深度估计 | `configs/depth_toy.yaml` |
| 自监督预训练(SimCLR) | `configs/ssl_toy.yaml` |
| 多模态分割 | 模型 `spm-net`(需项目数据,见教程 8.4) |
| 数据清洗 | `configs/clean_toy.yaml` |

查看已注册的内置组件:

```powershell
uv run python -c "from dlkit.registry import iter_component_records; [print(f'{r.registry}.{r.name} | {r.module}.{r.qualname}') for r in sorted(iter_component_records(provider='blueprintdl'), key=lambda x: (x.registry, x.name))]"
```

## 注册新组件

组件实现在 `dlkit/` 内并注册到对应 Registry:

```python
import torch.nn as nn

from dlkit.registry import LOSSES


@LOSSES.register(name="MyLoss", tags=("segmentation",))
class MyLoss(nn.Module):
    def forward(self, predictions, batch):
        ...
```

然后在组件所属包的导入入口(`__init__.py`)中显式导入该模块。注册阶段只登记组件,不训练、不读数据、不下载权重、不占用 GPU。添加一份新的 `configs/*.yaml` 即可把它用于实验。

## 训练产出治理:每个实验一个唯一 run

同一份配置反复训练,每次都会得到一个独立的时间戳目录,**永不覆盖**:

```text
runs/<project>/<experiment>/<YYYYmmdd-HHMMSS>/
├── run.json         # 身份:配置快照、--opts 覆盖、git commit、设备、torch 版本、final/best 指标
├── config.yaml      # 本次训练生效的完整配置
├── metrics.jsonl    # 逐 epoch 损失与指标
├── last.pth / best.pth
├── log.txt
└── vis/             # 标准可视化产物(训练结束自动画曲线)
```

- 续训:`--resume` 不带参数或 `--resume latest` 自动接本实验最近一次 run(谱系延续,metrics 历史不分裂);显式传 checkpoint 路径则按原语义;
- 显式覆盖输出目录:`--work-dir`(兼容旧用法);
- run.json 是对比实验(`tools/exp/compare.py`)的唯一事实来源。

## 可视化标准件

可视化由 `VISUALIZERS` Registry 驱动,与模型/损失组件同哲学:

- `TrainingCurves` — 训练/验证曲线(**训练结束自动生成** `vis/curves.png`);
- `SegOverlay` — val 样本 原图/GT/预测 三联图;
- `ConfusionMatrix` — 行归一化混淆矩阵;
- `DetBoxes` — GT(绿)与预测(红)框对比;
- `DepthColormap` — GT/预测/误差三联深度 colormap。

```powershell
# 对一次 run 生成全部标准可视化(按 config 的 task 自动选择)
uv run python tools/exp/visualize.py --run runs/examples/unet_toy/<timestamp>

# 只画曲线 / 指定组件 / 改样本数
uv run python tools/exp/visualize.py --run runs/... --visualizer SegOverlay --num-samples 6
```

自定义新可视化 = 在 `dlkit/viz/` 实现 `render(ctx)` 并注册,`tools/exp/visualize.py` 无需修改。

## 跨实验对比

```powershell
uv run python tools/exp/compare.py --project polyp              # 汇总并生成报告
uv run python tools/exp/compare.py --project polyp --metric mIoU  # 指定对比指标
```

输出到 `projects/<project>/reports/`:

- `EXPERIMENTS.md` — 所有 run 的对比表(模型/损失/学习率/轮数 + 各指标 last 值与 best),按实验分组、最新在前,底部附 run 目录索引;
- `EXPERIMENTS.csv` — 同表的 CSV 版本(方便贴进 Excel);
- `curves_<metric>.png` — 全部 run 的指定指标曲线叠图(默认选项目内常用 save_best 指标,否则 val_loss;run 超 20 条自动只画每实验最新一次)。

训练完某一组实验后跑一次 compare,即可把"这组到底谁好"沉淀为项目报告,而不是靠翻文件夹。

## 数据体检:一次体检,一份档案

新数据接进来先跑体检,把`样本数/配对率/类别像素分布/尺寸一致性/零值掩膜/坐标越界`等问题在训练前暴露,并落为数据档案:

```powershell
# 分割 images/ + masks/(可带 val 一致性检查)
uv run python tools/data/prepare_data.py --task segmentation --image-dir data/polyp/train/images --mask-dir data/polyp/train/masks --val-image-dir data/polyp/val/images --val-mask-dir data/polyp/val/masks --num-classes 4

# 分类 / 检测 / 深度同理(--train-dir 或 --image-dir+--label-dir/--depth-dir)
uv run python tools/data/prepare_data.py --task depth --image-dir data/depth/train/images --depth-dir data/depth/train/depth
```

输出到数据根目录:`dataset_meta.json`(可进 Git,作为该数据集的"体检报告")+ `datameta/*.png` 分布图。分割任务加 `--write-classes` 可顺便生成默认调色板 `classes.json`,供 `predict.py --palette` 直接使用。

## 项目生命周期:new → status → archive

```powershell
# 1) 建项目:生成 projects/polyp/(project.yaml + configs/_template.yaml + reports/ + data 说明)
uv run python tools/exp/project.py new polyp --display-name "Polyp Seg" --task segmentation

# 2) 随时看全局:list 显示每个项目的状态、配置数、run 数与最新 run
uv run python tools/exp/project.py list
uv run python tools/exp/project.py status polyp

# 3) 收尾:只生成收尾报告并置 archived,runs 产物全部保留(不移动不删除)
uv run python tools/exp/project.py archive polyp
```

`archive` 会把对比产物(EXPERIMENTS.md/CSV/曲线图,复用 compare 的同一套逻辑)与 `closure_<时间>.md`(每个实验的最佳结果表 + 关项 checklist)写进 `projects/<id>/reports/`,并把 project.yaml 状态置为 `archived`。想重新启用项目,把状态改回 `active` 即可。

## 想法如何落地:配置 → 组件 → Recipe

实验过程中不断冒出的新想法,每种只有一个归宿——**不要再为单个想法新建训练脚本**:

| 想法类型 | 归宿 | 产出 |
| --- | --- | --- |
| 超参/数据/损失权重 | YAML 或 `--opts` | 1 个配置 |
| 新 Dataset/Loss/Model/Metric | 注册组件(通用进 `dlkit/`,项目专属进 `projects/<id>/extensions/`) | 每个组件 1 个文件 |
| 训练流程变化(一致性、EMA、自蒸馏、交替优化) | 注册 Recipe(`RECIPES`) | 1 个 recipe 文件 |
| 一次性探索 | `scratch/`(gitignore),归档前清理或提升 | 临时脚本/notebook |

项目专属扩展由配置声明,训练/评估/推理自动加载(来源记入 run.json):

```yaml
imports: [projects/<id>/extensions]
```

流程型想法写成 Recipe,而不是复制一份训练脚本:

```yaml
recipe:
  type: EmaTeacher          # 内置模板:EMA 教师
  params: { decay: 0.999, validate_teacher: true }
```

预训练 → 微调用 `train.init_from`(只加载模型权重,不恢复 epoch/优化器),与 `--resume` 的"续训"语义严格区分:

```yaml
train:
  init_from: runs/<project>/<pretrain_exp>/<timestamp>/best.pth
```

`tools/` 只放全局 CLI;单个想法的实现永远不进 `tools/`。

自监督预训练已作为完整示例提供:

```powershell
uv run python tools/data/make_toy_data.py --root data/toy
uv run python tools/train.py --config configs/ssl_toy.yaml   # TwoViewDataset + SSLModel + NTXentLoss
```

预训练完成后,在目标任务配置里用 `train.init_from` 接上(只加载 backbone,投影头自动忽略):

```yaml
train:
  init_from: runs/examples/ssl_toy/<timestamp>/best.pth
```

## 训练、评估、续训和导出

```powershell
# 训练
uv run python tools/train.py --config configs/unet_toy.yaml

# 从 last.pth 续训
uv run python tools/train.py --config configs/unet_toy.yaml --resume latest            # 自动接本实验最近一次 run 的 last.pth

# 评估
uv run python tools/evaluate.py --config configs/unet_toy.yaml --checkpoint runs/examples/unet_toy/<timestamp>/best.pth

# 使用 GPU 导出 ONNX
uv run python tools/deploy/export.py --config configs/unet_toy.yaml --checkpoint runs/examples/unet_toy/<timestamp>/best.pth --format onnx --device cuda --output runs/examples/unet_toy/<timestamp>/model.onnx
```

训练目录默认包含配置快照、`metrics.jsonl`、`last.pth` 和按配置选出的 `best.pth`。

## 测试和构建

```powershell
# Core 测试
uv run pytest -q

# 检查依赖锁
uv lock --check

# 构建
uv build
```

首次构建可能需要由 uv 获取 `hatchling` 构建后端。

## 文档导航

- [完整框架教程](教程.md):从配置到自定义组件、实验对比的完整学习路径;
- [AI 辅助项目开发指南](AI辅助项目开发指南.md):如何用工程约束让 Coding Agent 稳定协作。

## 当前状态

- Registry 已支持注册名、模块、限定名称、标签、描述和来源记录(含 VISUALIZERS 与 RECIPES);
- 配置构造、通用训练引擎和多任务(分割、分类、检测、深度估计、自监督)示例已可用;
- 内置 SimCLR 自监督示例:`TwoViewDataset` + `SSLModel` + `NTXentLoss`,`configs/ssl_toy.yaml` 可直接跑;
- 内置多模态分割模型 `spm-net`(SPM-Net V3,图像 + 结构化特征 + 文本),依赖通过 `uv sync --extra multimodal` 按需安装;
- 每个 run 有唯一时间戳目录与 run.json 身份,支持 `--resume latest` 谱延续与 `train.init_from` 权重初始化;
- 支持项目插件(`imports`)与流程型 Recipe(内置 EMA 教师),项目专属扩展放在 `projects/<id>/extensions/`;
- `tools/data/prepare_data.py` 提供数据体检与 dataset_meta.json 档案;`tools/exp/project.py` 提供项目生命周期(new/list/status/archive,收尾保留全部 runs);
- `tools/` 同时提供生成数据、训练、评估、推理、导出、数据清洗、可视化(visualize)与对比(compare);
- 计划中的实验按"新组件 + 新配置 + 新测试"的流程加入;真实项目放进 `projects/<id>/`。
