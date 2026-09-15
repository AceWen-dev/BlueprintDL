# PolypMeasure

PolypMeasure 是在 BlueprintDL 项目锻造标准下创建的独立深度学习项目包。当前版本已经建立项目身份、Python 包边界、显式插件入口、测试目录和交付白名单；真实 Dataset、Model、Loss、Metric 与测量 Pipeline 将在业务需求明确后继续实现。

## 项目定位

项目专属代码留在 `polypmeasure` 包内，可复用能力通过 BlueprintDL Core 的公共 API 获得：

```text
polypmeasure -> blueprintdl
blueprintdl  -X-> polypmeasure
```

不要因为某段代码“以后可能复用”就提前放进 Core。只有经过多个项目真实使用、接口稳定且不包含 PolypMeasure 假设后，才考虑提升。

## 目录职责

```text
project.yaml                    项目身份、Core 版本、bootstrap 和交付白名单
pyproject.toml                  项目包与标准依赖
src/polypmeasure/               项目源码
src/polypmeasure/bootstrap.py   显式注册入口
src/polypmeasure/components/    项目 Dataset、Model、Loss、Metric 等组件
configs/                        训练、评估、推理和导出配置
tests/                          项目测试
```

## 注册组件

例如新增项目专用模型：

```python
from dlkit.registry import MODELS


@MODELS.register(
    name="PolypNet",
    provider="polypmeasure",
    tags=("segmentation", "polyp"),
)
class PolypNet:
    ...
```

然后在 `src/polypmeasure/components/__init__.py` 的 `load_components()` 中显式导入实现模块：

```python
def load_components():
    from . import models

    return (models.__name__,)
```

`provider` 必须与 `project.yaml` 中的 `project.id` 一致。bootstrap 只完成轻量导入与注册，不读取数据、不下载权重、不初始化 GPU，也不开始训练。

## 在 Forge 中检查

回到 BlueprintDL 仓库根目录：

```powershell
uv run blueprintdl-forge inspect polypmeasure
uv run blueprintdl-forge audit polypmeasure
```

当前尚未迁入业务组件，因此 `inspect` 显示 `none registered yet` 是正常状态。实现组件并加入 bootstrap 后，Forge 会根据真实 Registry 记录显示它们的类别、名称和代码位置。

## 配置与运行入口

项目配置放在 `configs/`，只保存可序列化信息：

- 组件注册名；
- 构造参数；
- 数据标识或相对路径；
- 训练轮数、batch size、device 逻辑策略等普通值。

`model.parameters()`、optimizer、scheduler、device 对象、logger、分布式上下文和凭据由项目入口或 Builder 在运行时注入。

项目开始训练开发时，应在 `polypmeasure` 包内添加自己的 Composition Root：先执行 `polypmeasure.bootstrap:register`，再加载配置、构造对象并启动 Trainer 或测量 Pipeline。不要让根目录通用训练脚本扫描所有项目寻找插件。

## 环境边界

PolypMeasure 面向 GPU 使用场景，但不会固定 CUDA、PyTorch wheel、驱动或 GPU 型号。它拥有独立 `pyproject.toml`，可以在交付环境中选择与目标机器兼容的依赖方案。

项目没有加入根 uv workspace。独立安装前，应确保声明的 `blueprintdl>=0.1,<0.2` 可以从包源或随交付提供的 Core 构建物获得。

## 构建和交付

在仓库根目录构建项目包：

```powershell
uv build --project projects/polypmeasure
```

审计并导出：

```powershell
uv run blueprintdl-forge audit polypmeasure
uv run blueprintdl-forge export polypmeasure dist/projects
```

Forge 会生成 `dist/projects/polypmeasure/` 和审计回执，并拒绝覆盖已有目标。交付前还需要在导出目录或隔离环境中验证项目安装、配置解析、测试和真实入口。

仓库级规则见 [项目锻造标准](../../standards/PROJECT_STANDARD.md)，完整学习路径见 [BlueprintDL 教程](../../教程.md)。
