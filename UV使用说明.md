# uv 使用说明（新手版）

这份说明针对当前项目 `BlueprintDL` 编写。项目使用 `pyproject.toml` 声明依赖，使用 `uv.lock` 锁定具体版本。

如果只记住三条命令：

```powershell
uv sync
uv run pytest
uv run python tools/train.py --config configs/unet_toy.yaml
```

## 1. uv 是什么

uv 是 Python 的项目和包管理工具，可以统一完成 Python 版本管理、虚拟环境创建、依赖安装、锁文件管理，以及在正确环境中运行脚本和测试。

当前项目中的几个关键文件：

| 文件/目录 | 作用 |
| --- | --- |
| `pyproject.toml` | 项目元数据、Python 版本范围、核心依赖和可选依赖 |
| `uv.lock` | uv锁定版本因为toml文件时版本是一个范围  |
| `.python-version` | 当前项目优先使用的 Python 版本，本项目为 3.12 |
| `.venv/` | 项目虚拟环境，不要提交到 Git |

官方项目结构说明见 [uv 项目指南](https://docs.astral.sh/uv/guides/projects/)。

## 2. 安装 uv（Windows）

推荐使用 WinGet，在 PowerShell 中执行：

```powershell
winget install --id=astral-sh.uv -e
```

安装完成后关闭并重新打开终端，再确认：

```powershell
uv --version
uv --help
```

也可以使用官方安装脚本：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

更多方式见 [uv 安装文档](https://docs.astral.sh/uv/getting-started/installation/)。

## 3. 第一次使用本项目

先进入能看到 `pyproject.toml` 的项目根目录：

```powershell
cd "A:\从0手搓深度学习项目"
```

同步环境：

```powershell
uv sync
```

这个命令会读取项目配置和锁文件，创建 `.venv`，安装依赖，并以可编辑模式安装当前项目。首次安装会下载较大的 PyTorch 文件，需要一些时间。

当前项目只声明 `torch` 和 `torchvision` 这两个框架依赖，不在核心配置中固定 CPU、CUDA 或 ROCm 源。它是一个面向 GPU 的通用学习框架，具体硬件后端属于部署环境，由使用者根据自己的驱动和平台选择。

检查环境：

```powershell
uv run python --version
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

在配置了 GPU 版 PyTorch、驱动和运行时的机器上，第二条命令通常应输出 `True`。如果输出 `False`，优先检查部署环境，而不是修改框架代码。

## 4. `uv run`：运行项目命令

推荐始终使用 `uv run`，这样不需要手动激活虚拟环境：

```powershell
uv run python tools/make_toy_data.py --root data/toy --samples 120
uv run python tools/train.py --config configs/unet_toy.yaml
uv run pytest
```

也可以执行一次性的 Python 代码：

```powershell
uv run python -c "import dlkit; print(dlkit.__version__)"
```

`uv run` 会在执行前检查锁文件和环境是否需要更新，并确保命令使用项目依赖。详细说明见 [uv run 文档](https://docs.astral.sh/uv/concepts/projects/run/)。

如果确实需要手动激活，PowerShell 使用：

```powershell
.\.venv\Scripts\Activate.ps1
```

日常开发仍建议使用 `uv run`，这样不容易误用其他 Python 环境。

## 5. 添加和删除依赖

不要在这个项目中把 `pip install xxx` 作为日常依赖管理方式。应该让 uv 同时更新配置、锁文件和环境。

添加运行时依赖：

```powershell
uv add requests
uv add "requests>=2.32"
```

添加开发依赖：

```powershell
uv add --dev ruff
uv add --dev pytest
```

添加可选依赖到指定 extra：

```powershell
uv add --optional export onnx
```

删除依赖：

```powershell
uv remove requests
uv remove --dev ruff
```

这些命令通常会自动更新 `uv.lock` 并同步环境；也可以手动执行 `uv sync`。依赖声明方式见 [uv 依赖管理文档](https://docs.astral.sh/uv/concepts/projects/dependencies/)。

## 6. 本项目中的依赖分类

### 核心依赖

训练、推理和数据处理所需的依赖，例如 `torch`、`torchvision`、`numpy`、`pillow`、`pyyaml`、`matplotlib` 和 `tqdm`。执行 `uv sync` 会默认安装。

### 开发依赖

`pytest` 位于 `dev` 组中，默认随 `uv sync` 安装：

```powershell
uv sync
uv sync --no-dev
uv sync --only-dev
```

### 可选依赖（extras）

当前项目提供两个 extra：

| extra | 内容 | 命令 |
| --- | --- | --- |
| `export` | `onnx`、`onnxruntime` | `uv sync --extra export` |
| `tensorboard` | `tensorboard` | `uv sync --extra tensorboard` |

安装所有可选依赖：

```powershell
uv sync --all-extras
```

可选依赖默认不安装，避免基础训练下载不需要的包。

## 7. 更新依赖和查看依赖树

普通 `uv sync` 会优先使用已有锁文件的版本。升级所有依赖：

```powershell
uv lock --upgrade
uv sync
```

只升级一个依赖：

```powershell
uv lock --upgrade-package numpy
uv sync
```

升级到指定版本：

```powershell
uv lock --upgrade-package numpy==2.5.3
uv sync
```

查看依赖树：

```powershell
uv tree
```

检查配置和锁文件是否一致：

```powershell
uv lock --check
```

`uv.lock` 由 uv 管理，不建议手动编辑。`pyproject.toml` 描述允许的版本范围，`uv.lock` 记录实际解析出的精确版本；锁文件应该提交到 Git。

## 8. 测试和训练命令

运行全部测试：

```powershell
uv run pytest
```

运行单个测试文件：

```powershell
uv run pytest tests/test_models.py
uv run python tests/test_registry.py
```

生成 toy 数据并训练分割模型：

```powershell
uv run python tools/make_toy_data.py --root data/toy --samples 120
uv run python tools/train.py --config configs/unet_toy.yaml
```

分类、检测和深度估计同理，只需替换对应脚本和配置：

```powershell
uv run python tools/make_toy_cls_data.py --root data/cls_toy --samples-per-class 30
uv run python tools/train.py --config configs/cls_toy.yaml

uv run python tools/make_toy_det_data.py --root data/det_toy --samples 120
uv run python tools/train.py --config configs/det_toy.yaml

uv run python tools/make_toy_depth_data.py --root data/depth_toy --samples 120
uv run python tools/train.py --config configs/depth_toy.yaml
```

## 9. Python 版本管理

查看 uv 能找到的 Python：

```powershell
uv python list
```

安装 Python 3.12：

```powershell
uv python install 3.12
```

将当前项目固定到 Python 3.12：

```powershell
uv python pin 3.12
```

本项目已经有 `.python-version`，通常不需要重复执行 `pin`。uv 也能在需要时自动下载和管理 Python；参考 [Python 管理文档](https://docs.astral.sh/uv/guides/install-python/)。

## 10. 常见问题

### `uv` 不是内部或外部命令

关闭当前终端并重新打开 PowerShell。若仍然不行，重新执行安装命令，或检查 WinGet 是否安装成功。

### 为什么 `python` 找不到项目包？

系统的 `python` 不一定指向 `.venv`。使用 `uv run python your_script.py`，或者先执行 `.\.venv\Scripts\Activate.ps1`。

### 改了 `pyproject.toml` 后怎么办？

不要手动修改锁文件，执行：

```powershell
uv lock
uv sync
```

### 为什么 `uv sync` 删除了我手动安装的包？

`uv sync` 默认会把环境同步为锁文件描述的状态，未声明的包可能被移除。需要长期使用的包应该执行 `uv add package-name`。

### 下载依赖很慢或报错怎么办？

先确认网络可以访问 PyPI 和 PyTorch 包源。查看缓存位置：

```powershell
uv cache dir
```

如果缓存损坏，可以清理后重试：

```powershell
uv cache clean
uv sync
```

### 如何选择 GPU/CUDA 版 PyTorch？

本项目的核心配置不会替你选择硬件后端。请根据 NVIDIA 驱动或其他硬件环境，在 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)选择对应构建，并在部署环境中配置对应的 PyTorch 包源。完成环境配置后，再执行：

```powershell
uv lock
uv sync
```

不要把某台机器的 CUDA 源硬编码进通用框架配置。不同机器需要不同后端时，应使用部署专用的依赖覆盖、容器镜像或 CI 配置，并让对应环境单独解析锁文件；框架本身继续保持 `torch`/`torchvision` 的抽象依赖。

## 11. 推荐的日常流程

拿到项目或切换分支后：

```powershell
uv sync
uv run pytest
```

增加功能时：

```powershell
uv add package-name
uv run pytest
git add pyproject.toml uv.lock
```

升级某个依赖时：

```powershell
uv lock --upgrade-package package-name
uv sync
uv run pytest
```

提交代码时，通常提交 `pyproject.toml`、`uv.lock` 和源代码，不提交 `.venv`、缓存、训练数据和模型权重。

## 官方资料

- [uv 官方文档](https://docs.astral.sh/uv/)
- [安装 uv](https://docs.astral.sh/uv/getting-started/installation/)
- [uv 项目指南](https://docs.astral.sh/uv/guides/projects/)
- [依赖管理](https://docs.astral.sh/uv/concepts/projects/dependencies/)
- [锁定与同步环境](https://docs.astral.sh/uv/concepts/projects/sync/)
- [运行项目命令](https://docs.astral.sh/uv/concepts/projects/run/)
