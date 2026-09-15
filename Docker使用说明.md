# Docker 使用说明（新手版）

这份教程针对当前的 `BlueprintDL` 项目编写。Docker 可以把程序、Python、依赖和系统环境一起打包成镜像，让项目在不同电脑上更容易复现。

当前项目已经使用 uv 管理 Python 依赖，但还没有 `Dockerfile` 或 Compose 文件。因此本文先讲清楚 Docker 的基本概念，再提供一套适合本项目的示例配置。

## 1. Docker 是什么

Docker 中最常见的几个概念：

| 概念 | 含义 |
| --- | --- |
| 镜像（image） | 程序运行环境的只读模板 |
| 容器（container） | 镜像启动后的运行实例 |
| Dockerfile | 描述如何构建镜像的文本文件 |
| Compose | 用 YAML 文件描述和管理容器 |
| 数据卷/挂载 | 让容器和宿主机共享或持久保存数据 |
| Registry | 存放和下载镜像的仓库，例如 Docker Hub |

可以这样理解：

```text
Dockerfile  --构建-->  Image  --运行-->  Container
```

容器删除后，容器内部临时写入的数据可能消失；数据集、日志和模型权重等重要内容应该使用目录挂载或命名卷保存。

## 2. 安装 Docker Desktop（Windows）

在 Windows 上推荐安装 Docker Desktop。安装完成后启动 Docker Desktop，等待状态显示为运行中。

验证 Docker：

```powershell
docker --version
```

```powershell
docker compose version
```

```powershell
docker run --rm hello-world
```

`hello-world` 能正常输出欢迎信息，说明 Docker 引擎可以工作。

Windows 上建议使用 WSL 2 后端。官方资料：

- [Docker Desktop 安装](https://docs.docker.com/desktop/setup/install/windows-install/)
- [Docker 入门教程](https://docs.docker.com/get-started/)

## 3. 常用 Docker 命令

查看本地镜像：

```powershell
docker image ls
```

查看运行中的容器：

```powershell
docker ps
```

查看所有容器：

```powershell
docker ps -a
```

下载镜像：

```powershell
docker pull python:3.12-slim
```

运行临时容器：

```powershell
docker run --rm python:3.12-slim python --version
```

其中 `--rm` 表示容器退出后自动删除。

停止容器：

```powershell
docker stop <container-id-or-name>
```

删除停止的容器：

```powershell
docker rm <container-id-or-name>
```

删除镜像：

```powershell
docker rmi <image-name-or-id>
```

查看磁盘占用：

```powershell
docker system df
```

`docker system prune` 会清理未使用的资源，使用前要确认不会影响其他项目。

## 4. 为本项目准备 Dockerfile

在项目根目录新建名为 `Dockerfile` 的文件，注意文件名没有扩展名。下面是框架通用镜像示例：

```dockerfile
FROM python:3.12-slim

# 使用官方 uv 镜像中的 uv 程序
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 先复制依赖文件，便于 Docker 复用构建缓存
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

# 再复制项目源代码和配置
COPY dlkit ./dlkit
COPY tools ./tools
COPY configs ./configs

# 安装当前项目本身
RUN uv sync --frozen --no-dev

# 默认只做一个轻量导入检查，不自动开始训练
CMD ["uv", "run", "python", "-c", "import dlkit; print(dlkit.__version__)"]
```

这个 Dockerfile 使用 Python 3.12、项目中的 `pyproject.toml` 和 `uv.lock`，并关闭开发依赖。它不在框架镜像中硬编码 CPU、CUDA 或 ROCm 后端；实际部署时应根据目标机器构建对应的运行镜像。正式项目中可以把 uv 的 `latest` 换成固定版本，以获得更稳定的构建结果。

## 5. 创建 .dockerignore

在项目根目录新建 `.dockerignore`，避免把本地虚拟环境、数据和训练结果发送到 Docker 构建上下文：

```text
.git
.venv
__pycache__
.pytest_cache
*.py[cod]
data
runs
*.pth
*.pt
*.onnx
*.log
.env
```

`data/` 和 `runs/` 不会被打包进镜像，运行容器时再通过目录挂载使用它们。这样镜像更小，也不会把本地数据或模型权重意外写入镜像。

## 6. 构建和运行本项目镜像

在包含 `Dockerfile` 的项目根目录执行构建：

```powershell
docker build -t blueprintdl:framework .
```

查看镜像：

```powershell
docker image ls blueprintdl
```

运行默认导入检查：

```powershell
docker run --rm blueprintdl:framework
```

在容器中运行完整测试：

```powershell
docker run --rm blueprintdl:framework uv run pytest
```

如果 Dockerfile 没有复制 `tests/`，需要额外加入：

```dockerfile
COPY tests ./tests
```

然后重新构建镜像。修改 Dockerfile 或依赖后，需要重新构建：

```powershell
docker build --no-cache -t blueprintdl:framework .
```

通常不要一开始就使用 `--no-cache`；它会放弃所有构建缓存，速度会明显变慢。

## 7. 在容器中训练

容器内的 `/app` 是项目目录。宿主机的 `data/` 和 `runs/` 应挂载进去，否则容器删除后数据和输出可能丢失。

PowerShell 示例：

```powershell
docker run --rm -it `
  -v "${PWD}\data:/app/data" `
  -v "${PWD}\runs:/app/runs" `
  blueprintdl:framework `
  uv run python tools/train.py --config configs/unet_toy.yaml
```

如果只想先生成 toy 数据：

```powershell
docker run --rm -it `
  -v "${PWD}\data:/app/data" `
  blueprintdl:framework `
  uv run python tools/make_toy_data.py --root data/toy --samples 120
```

如果使用其他终端，`${PWD}` 的写法可能不同；核心思想是把宿主机的 `data` 挂载到容器内的 `/app/data`，把训练输出挂载到 `/app/runs`。

## 8. 使用 Docker Compose

如果经常运行同一套参数，可以在项目根目录创建 `compose.yaml`：

```yaml
services:
  blueprintdl:
    build: .
    image: blueprintdl:framework
    working_dir: /app
    volumes:
      - ./data:/app/data
      - ./runs:/app/runs
    command:
      - uv
      - run
      - python
      - tools/train.py
      - --config
      - configs/unet_toy.yaml
```

构建并运行：

```powershell
docker compose up --build
```

后台运行：

```powershell
docker compose up --build -d
```

如果这是一次性训练任务，更适合使用：

```powershell
docker compose run --rm blueprintdl
```

训练完成后容器会退出，但通过挂载保存的 `runs/` 内容仍然保留在宿主机。

查看服务状态：

```powershell
docker compose ps
```

查看日志：

```powershell
docker compose logs -f blueprintdl
```

停止并删除容器：

```powershell
docker compose down
```

Compose 适合把构建参数、目录挂载和启动命令保存到版本控制中。官方说明见 [Docker Compose 入门](https://docs.docker.com/compose/gettingstarted/)。

## 9. GPU 和 CUDA

当前项目面向 GPU，但框架配置不固定 CPU、CUDA 或 ROCm。前面的通用 Dockerfile 只负责打包项目和依赖；如果要运行 GPU 训练，应基于目标机器兼容的 CUDA/ROCm 运行时制作部署镜像，并在该环境中安装对应的 PyTorch 构建。

使用 GPU 容器需要同时满足：

1. 宿主机安装兼容的 NVIDIA 驱动；
2. Docker 已配置 GPU 支持；
3. 已安装并配置 NVIDIA Container Toolkit；
4. 镜像中的 PyTorch 是与 CUDA 兼容的版本。

先用官方 CUDA 镜像验证 GPU 是否能被 Docker 看到：

```powershell
docker run --rm --gpus all nvidia/cuda:<compatible-tag> nvidia-smi
```

`<compatible-tag>` 要替换成与你的驱动兼容的 CUDA 镜像标签。NVIDIA 官方示例使用 `--gpus all` 暴露所有 GPU，也可以指定某一块 GPU。

GPU 配置不是只在 `docker run` 后面增加 `--gpus all` 就结束了。还需要让部署镜像中的 PyTorch 与宿主机驱动和 CUDA 版本匹配；这些属于部署层，不应写死到通用框架配置。

验证项目容器能否看到 GPU：

```powershell
docker run --rm --gpus all blueprintdl:framework uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

如果输出 `False`，请依次检查 NVIDIA 驱动、Docker GPU 支持、PyTorch wheel 和 CUDA 兼容性。

参考：[NVIDIA Container Toolkit 示例](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html)。

## 10. 数据卷和目录挂载

绑定挂载把宿主机目录映射到容器目录：

```powershell
docker run --rm -it -v "${PWD}\data:/app/data" blueprintdl:framework
```

左边是宿主机目录，右边是容器内目录。容器中的程序访问 `/app/data` 时，实际读写的是宿主机的 `data` 目录。

当前项目常用的挂载关系：

| 宿主机目录 | 容器目录 | 用途 |
| --- | --- | --- |
| `data/` | `/app/data` | 数据集和 toy 数据 |
| `runs/` | `/app/runs` | 日志、预测结果和 checkpoint |

不要把数据和模型权重直接写死在镜像中。变化频繁的数据和训练输出最好挂载出来。

## 11. 进入容器和查看日志

后台启动容器：

```powershell
docker run -d --name blueprintdl-dev blueprintdl:framework
```

查看日志：

```powershell
docker logs blueprintdl-dev
docker logs -f blueprintdl-dev
```

进入正在运行的容器：

```powershell
docker exec -it blueprintdl-dev sh
```

进入后可以检查：

```sh
pwd
ls
uv run python --version
```

退出容器输入 `exit`。停止并删除开发容器：

```powershell
docker stop blueprintdl-dev
docker rm blueprintdl-dev
```

## 12. 常见问题

### `docker` 不是内部或外部命令

确认 Docker Desktop 已安装并启动，然后关闭当前终端并重新打开。如果 Docker Desktop 正在启动，等待其状态变为 Running。

### `Cannot connect to the Docker daemon`

通常是 Docker Desktop 没有启动，或者当前用户没有访问 Docker Engine 的权限。先启动 Docker Desktop，再重新执行命令。

### 构建很慢

首次构建需要下载 Python、PyTorch 和其他依赖。后续构建会复用层缓存。不要频繁使用 `docker build --no-cache`。

如果只改了 Python 源码，正确的 Dockerfile 分层可以避免重新安装所有依赖；依赖文件改变时才会重新执行 uv 同步层。

### 容器删除后数据没有了

这是因为数据写在容器内部。把数据集和训练输出挂载到宿主机：

```powershell
docker run --rm --gpus all -v "${PWD}\data:/app/data" -v "${PWD}\runs:/app/runs" blueprintdl:framework
```

### Windows 挂载目录报权限错误

确认 Docker Desktop 允许访问该磁盘，并尽量使用当前项目目录下的 `data` 和 `runs`。路径中有空格时要使用引号。

### 为什么容器里没有本地 `.venv`

本地 `.venv` 是 Windows 环境创建的，不能直接复制到 Linux 容器。Dockerfile 应该在镜像内部重新执行 `uv sync`，这也是 `.dockerignore` 忽略 `.venv` 的原因。

### 为什么容器中找不到数据

检查挂载参数、容器内路径和脚本参数是否一致：

```powershell
docker inspect <container-id-or-name>
```

训练命令中的 `data/toy` 是相对于 `/app` 的路径，因此宿主机应把 `data` 挂载为 `/app/data`。

## 13. 推荐的日常流程

修改 Dockerfile 或依赖后：

```powershell
docker build -t blueprintdl:framework .
docker run --rm blueprintdl:framework uv run pytest
```

训练时挂载数据和输出：

```powershell
docker run --rm -it `
  -v "${PWD}\data:/app/data" `
  -v "${PWD}\runs:/app/runs" `
  blueprintdl:framework `
  uv run python tools/train.py --config configs/unet_toy.yaml
```

确认运行结果后，再提交 Docker 配置文件：

```powershell
git add Dockerfile .dockerignore compose.yaml
git commit -m "添加 Docker 运行配置"
```

如果暂时只是在学习 Docker，也可以先只提交本教程，不必立刻把 Dockerfile 加入项目。

## 参考资料

- [Docker 官方文档](https://docs.docker.com/)
- [Dockerfile 参考](https://docs.docker.com/reference/dockerfile/)
- [Docker Compose 入门](https://docs.docker.com/compose/gettingstarted/)
- [Docker 与 uv 集成](https://docs.astral.sh/uv/guides/integration/docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/)
