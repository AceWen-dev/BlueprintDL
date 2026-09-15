# BlueprintDL 具体项目区

`projects/` 用来孵化遵守 BlueprintDL 规则的具体深度学习项目。每个一级子目录都是一个独立 Python 项目，而不是 Core 的示例配置或随意堆放业务代码的目录。

## 项目与 Core 的边界

```text
projects/<project_id> -> dlkit 公共 API
dlkit                 -X-> projects/<project_id>
project-a             -X-> project-b 内部代码
```

适合放在具体项目中的内容：

- 客户或数据集特有的 Dataset 与标注解析；
- 项目专用模型、Loss、Metric 和后处理；
- 项目训练、评估、推理和交付入口；
- 项目自己的配置、测试与文档。

一个组件只有经过真实跨项目复用、接口稳定且不携带项目假设后，才考虑提升到 `dlkit` Core。

## 每个项目必须拥有

```text
projects/<project_id>/
├── project.yaml                 # 身份、Core 兼容范围、bootstrap、交付白名单
├── pyproject.toml               # 独立 Python 包和依赖
├── README.md                    # 任务、数据协议、运行与交付说明
├── configs/                     # 声明式配置
├── src/<package>/               # 项目源码
│   ├── bootstrap.py             # 显式插件注册入口
│   └── components/              # 项目组件
└── tests/                       # 项目测试
```

## 创建项目

在 BlueprintDL 仓库根目录执行：

```powershell
uv run blueprintdl-forge new retinal-seg --display-name "Retinal Segmentation"
```

项目 ID 使用小写字母、数字和连字符。Forge 会生成合法的 Python 包名，并拒绝覆盖已有项目。

## 注册项目组件

项目组件必须显式声明 `provider=<project_id>`：

```python
from dlkit.registry import MODELS


@MODELS.register(name="RetinalNet", provider="retinal-seg")
class RetinalNet:
    ...
```

在项目的 `components.load_components()` 中显式导入组件模块。不要扫描目录，也不要在注册阶段读数据、下载权重、初始化 CUDA 或开始训练。

组件清单从真实 Registry 记录生成，不在 `project.yaml` 中重复维护。

## 查看与审计

```powershell
uv run blueprintdl-forge list
uv run blueprintdl-forge inspect polypmeasure
uv run blueprintdl-forge audit polypmeasure
```

`audit` 会检查项目身份、Python 元数据、源码包、bootstrap、交付白名单、跨项目导入和 provider 归属。

## 独立环境

具体项目默认不加入根 uv workspace。不同项目可以拥有不同 Python、PyTorch、CUDA/ROCm 和客户依赖策略。

项目独立安装前，需要确保 `pyproject.toml` 声明的 BlueprintDL Core 版本可以从包源或交付构建物获得。仓库内的 Forge 检查会显式暴露当前项目源码和本地 Core，但这不应成为交付项目的长期依赖方式。

## 导出项目

```powershell
uv run blueprintdl-forge audit polypmeasure
uv run blueprintdl-forge export polypmeasure dist/projects
```

导出结果位于 `dist/projects/polypmeasure/`。Forge 只复制 `project.yaml` 中 `delivery.include` 允许的路径，并写入 `.blueprintdl-export.json`。缓存、虚拟环境、运行结果、数据集和仓库其他项目不会默认进入交付物。

完整规则见 [BlueprintDL 项目锻造标准](../standards/PROJECT_STANDARD.md)，完整操作见 [框架教程](../教程.md)。
