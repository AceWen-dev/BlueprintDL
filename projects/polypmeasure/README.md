# PolypMeasure

PolypMeasure 是在 BlueprintDL 项目锻造标准下创建的独立项目包。当前提交建立项目边界、插件入口和交付清单；具体数据协议、模型与测量算法将在明确需求后放入本项目，而不是写入 `dlkit` Core。

## 目录职责

```text
project.yaml                 项目身份、框架兼容范围、插件入口和交付清单
src/polypmeasure/            项目源码
src/polypmeasure/components/ 项目专属 Dataset、Model、Loss、Metric 等组件
configs/                     项目配置
tests/                       项目测试
```

## 注册项目组件

在 `src/polypmeasure/components/` 中实现组件，声明项目来源：

```python
from dlkit.registry import MODELS

@MODELS.register(name="PolypNet", provider="polypmeasure")
class PolypNet:
    pass
```

然后在 `components.load_components()` 中显式导入该模块。不要扫描目录或在注册阶段加载数据、权重与 GPU 资源。

## 环境边界

项目面向 GPU 使用场景，但不固定 CUDA、PyTorch wheel、驱动或 GPU 型号。仓库内检查时 Forge 使用当前 BlueprintDL Core 并显式加载本项目源码；独立交付后按照 `pyproject.toml` 中声明的兼容版本安装 BlueprintDL。项目不加入根 uv workspace，可以独立锁定自己的 GPU 依赖。

在 BlueprintDL 仓库根目录可以检查当前骨架：

```bash
uv run python tools/forge.py inspect polypmeasure
uv run python tools/forge.py audit polypmeasure
```
