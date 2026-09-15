# BlueprintDL 项目锻造标准

BlueprintDL 同时承担两种职责：

1. 提供稳定的深度学习框架能力与工程规则；
2. 作为多个具体项目的孵化仓库，直到项目满足独立交付条件。

具体项目必须从创建当天起保持清晰边界。项目在仓库内部开发，并不意味着它可以依赖仓库中任意文件。

## 四个层次

```text
标准层 standards
    定义项目规则、接口契约、检查项和交付条件

框架层 dlkit
    提供 Registry、Builder、训练引擎、通用组件和公共 API

项目层 projects/<project_id>
    保存具体业务的数据集、模型、损失、指标、配置、测试和入口

交付层 dist/projects/<project_id>
    由 Forge 根据项目清单生成的独立交付物
```

## 依赖规则

允许：

```text
项目 -> dlkit 公共 API
项目组件 -> 项目内部协议或 dlkit 公共协议
项目配置 -> 已显式注册的框架组件或项目组件
```

禁止：

```text
dlkit -> 任意具体项目
项目 A -> 项目 B 的内部模块
项目 -> 仓库根目录下未声明的临时文件
项目 -> 另一个项目的 data、runs 或私有配置
```

## 项目必须拥有的文件

每个 `projects/<project_id>` 至少包含：

- `project.yaml`：项目身份、框架兼容范围、插件入口和交付清单；
- `pyproject.toml`：可独立构建的 Python 项目与标准依赖；
- `src/<package>/`：项目源码包；
- `src/<package>/bootstrap.py`：显式组件注册入口；
- `configs/`：项目自己的声明式配置；
- `tests/`：组件、配置和最小流程测试；
- `README.md`：项目用途、数据契约、训练/推理和交付说明。

项目不得依赖运行时修改 `sys.path` 才能工作。仓库工具可以在开发检查时显式暴露当前项目源码并使用本地 BlueprintDL Core，但项目的标准依赖声明必须在离开仓库后仍然有效。具体项目不加入根 uv workspace，可以拥有独立环境和锁文件，以允许不同项目采用不同的 PyTorch 与 GPU 依赖策略。

## 项目身份

`project.yaml` 是项目身份和交付边界的权威来源：

```yaml
schema_version: 1
project:
  id: polypmeasure
  display_name: PolypMeasure
  package: polypmeasure
  version: 0.1.0
framework:
  package: blueprintdl
  version: ">=0.1,<0.2"
plugins:
  bootstrap:
    - polypmeasure.bootstrap:register
delivery:
  include:
    - pyproject.toml
    - project.yaml
    - README.md
    - src
    - configs
    - tests
```

清单只声明项目级信息和显式插件入口。不要在清单中手写一份组件名称列表；组件归属从注册表记录生成，避免清单与代码漂移。

## 组件归属

注册表中的项目组件必须声明 `provider=<project_id>`：

```python
@MODELS.register(name="PolypNet", provider="polypmeasure")
class PolypNet:
    ...
```

组件来源记录至少包含：

- 所属 Registry；
- 注册名称；
- provider；
- Python 模块和限定名称；
- 可选标签与说明。

Forge 通过这些记录回答：

- 仓库里有哪些项目；
- 某个项目提供哪些组件；
- 某个组件由哪个项目或框架模块提供；
- 项目配置引用了哪些共享组件。

Core 中由 `dlkit.*` 模块实现的组件记录为 `provider=blueprintdl`；具体项目必须显式声明自己的 provider，审计不会把缺失归属的项目组件自动当成 Core。

## 组件放置规则

只服务一个项目的能力留在该项目中，例如专用 Dataset、标注解析、模型结构、业务 Loss、Metric 和后处理。

满足以下条件后，才考虑提升到 `dlkit` 或公共扩展包：

1. 已经被两个以上项目实际复用；
2. 接口稳定，不携带某个项目的业务假设；
3. 有独立测试与清楚文档；
4. 提升后不会迫使 Core 依赖具体项目。

## 插件加载

每个项目通过 `plugins.bootstrap` 列出的 `module:function` 显式加载组件。

- 不扫描整个文件系统；
- 不从不可信配置导入任意模块；
- bootstrap 只完成轻量注册；
- bootstrap 不启动训练、下载权重、读取数据或占用 GPU；
- 项目根包被导入时不应自动启动工作流。

## 配置与运行时对象

配置保存组件名称和普通参数。以下对象由 Builder 或 Pipeline setup 在运行时注入：

- `model.parameters()`；
- optimizer 和 scheduler 实例；
- device、分布式上下文和 AMP 状态；
- logger、客户端、打开的文件与凭据；
- 数据加载器计算出的总步数。

框架面向 GPU，但项目清单不得写死 CUDA wheel、驱动版本或显卡型号。具体安装环境由交付环境决定。

## 独立交付条件

项目只有同时满足以下条件，才可以从 Forge 导出：

- 清单结构合法且所有交付路径存在；
- Python 包可以独立构建；
- 只依赖 `dlkit` 公共 API；
- 没有跨项目导入；
- bootstrap 可以显式加载；
- 注册组件均带正确 provider；
- 配置可解析；
- 单元测试与最小集成测试通过；
- README 说明数据格式、训练、评估、推理和环境边界；
- 标准依赖中声明兼容的 BlueprintDL 版本。

导出过程必须创建新目录并拒绝覆盖已有交付物。仓库中的 `data/`、`runs/`、缓存、虚拟环境和本机 CUDA 环境不属于默认交付内容。
