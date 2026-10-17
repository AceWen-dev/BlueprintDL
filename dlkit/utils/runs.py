"""运行(Run)身份与产物归档工具。

约定一次训练的完整产物位于:

    runs/<project>/<experiment>/<YYYYmmdd-HHMMSS>/

- `<project_id>` 与 `<experiment>` 取自配置文件(缺失时由调用方/Trainer 提供回退值);
- 时间戳目录保证同一实验重跑永不覆盖,直接为实验留档;
- 目录内维护一份 ``run.json`` 清单:配置快照、命令行覆盖、git 提交、
  环境、最终与 best 指标,是跨实验对比(compare)的唯一事实来源。

本模块刻意不依赖 torch,保证清单能力在纯文件层面即可测试。
"""

from __future__ import annotations

import datetime
import json
import os
import re
import socket
import subprocess
import getpass
import sys
from pathlib import Path
from typing import Any

RUN_SCHEMA_VERSION = 1
MANIFEST_NAME = 'run.json'
RUNS_ROOT_DEFAULT = 'runs'
_RUN_DIR_RE = re.compile(r'^\d{8}-\d{6}(?:-\d+)?$')


# ---------- run 目录管理 ----------


def make_run_dir(runs_root: str | Path, project: str, experiment: str) -> Path:
    """为一次训练创建唯一的时间戳目录,并返回其路径。"""

    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    base = Path(runs_root) / project / experiment
    base.mkdir(parents=True, exist_ok=True)
    run_dir = base / stamp
    counter = 1
    while run_dir.exists():
        counter += 1
        run_dir = base / ('%s-%d' % (stamp, counter))
    run_dir.mkdir()
    return run_dir


def iter_run_dirs(
    runs_root: str | Path = RUNS_ROOT_DEFAULT,
    project: str | None = None,
    experiment: str | None = None,
) -> list[Path]:
    """枚举 run 目录(只认时间戳形态),按项目/实验/时间排序。"""

    root = Path(runs_root)
    if not root.is_dir():
        return []
    result: list[Path] = []
    project_root = root / project if project else root
    for dirpath, dirnames, _ in os.walk(str(project_root)):
        parent = Path(dirpath)
        folder_name = parent.name
        dirnames[:] = [d for d in dirnames]
        depth = len(parent.relative_to(root).parts)
        if depth >= 2 and _RUN_DIR_RE.match(folder_name):
            if experiment is not None and parent.parent.name != experiment:
                continue
            result.append(parent)
    return sorted(
        result,
        key=lambda p: (
            p.parent.parent.name.casefold(),
            p.parent.name.casefold(),
            p.name,
        ),
    )


def latest_run(
    runs_root: str | Path,
    project: str,
    experiment: str,
    *,
    requires_checkpoint: bool = True,
) -> Path | None:
    """返回指定实验最近一次 run 目录(默认要求存在 last.pth)。"""

    candidates = list(
        reversed(
            iter_run_dirs(runs_root, project=project, experiment=experiment)
        )
    )
    for run_dir in candidates:
        if requires_checkpoint and not (run_dir / 'last.pth').is_file():
            continue
        return run_dir
    return None


# ---------- run 清单(run.json) ----------


def read_run_manifest(run_dir: str | Path) -> dict[str, Any] | None:
    """读取 run.json;不存在或损坏时返回 None。"""

    path = Path(run_dir) / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_run_manifest(run_dir: str | Path, manifest: dict[str, Any]) -> Path:
    path = Path(run_dir) / MANIFEST_NAME
    with path.open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write('\n')
    return path


def init_run_manifest(
    run_dir: str | Path,
    *,
    project: str,
    experiment: str,
    config: dict[str, Any],
    overrides: dict[str, Any] | None = None,
    raw_opts: list[str] | None = None,
    resume_from: str | None = None,
    init_from: str | None = None,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """建立一份新的 run.json 清单(初始状态 running)。"""

    manifest = {
        'schema_version': RUN_SCHEMA_VERSION,
        'status': 'running',
        'project': project,
        'experiment': experiment,
        'run_id': Path(run_dir).name,
        'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'config': config,
        'overrides': dict(overrides or {}),
        'raw_opts': list(raw_opts or []),
        'environment': environment or collect_environment_info(),
        'resume_from': resume_from,
        'init_from': init_from,
        'final': None,
        'best': None,
    }
    write_run_manifest(run_dir, manifest)
    return manifest


def finalize_run_manifest(
    run_dir: str | Path,
    *,
    total_epochs: int,
    last_metrics: dict[str, Any] | None = None,
    save_best: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """训练结束后回填 run.json 的 final/best 字段并置 finished。"""

    manifest = read_run_manifest(run_dir)
    if manifest is None:
        return None
    manifest['status'] = 'finished'
    manifest.setdefault('finished_at', datetime.datetime.now().isoformat(timespec='seconds'))
    manifest['final'] = {
        'total_epochs': total_epochs,
        **(last_metrics or {}),
    }
    if save_best:
        manifest['best'] = save_best
    write_run_manifest(run_dir, manifest)
    return manifest


# ---------- 环境信息(best effort) ----------


def _git_info() -> dict[str, Any]:
    """尽力获取当前 git 提交信息;不在 git 仓库或 git 不可用时返回空。"""

    info: dict[str, Any] = {'commit': None, 'dirty': None}
    try:
        head = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if head.returncode == 0:
            info['commit'] = head.stdout.strip()
            dirty = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if dirty.returncode == 0:
                info['dirty'] = bool(dirty.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def collect_environment_info(
    *,
    device: str | None = None,
    torch_version: str | None = None,
    cuda_available: bool | None = None,
) -> dict[str, Any]:
    """收集可复现所需的环境摘要。torch 相关字段由调用方(Trainer)注入。"""

    info = _git_info()
    return {
        'git_commit': info['commit'],
        'git_dirty': info['dirty'],
        'hostname': socket.gethostname(),
        'user': getpass.getuser() or None,
        'python': _short(sys.version),
        'device': device,
        'torch': torch_version,
        'cuda_available': cuda_available,
    }


def _short(version_string: str) -> str:
    first = version_string.split(' ', 1)[0]
    return first.split('|')[0].strip() or version_string[:32]


__all__ = [
    'MANIFEST_NAME',
    'RUNS_ROOT_DEFAULT',
    'RUN_SCHEMA_VERSION',
    'collect_environment_info',
    'finalize_run_manifest',
    'init_run_manifest',
    'iter_run_dirs',
    'latest_run',
    'make_run_dir',
    'read_run_manifest',
    'write_run_manifest',
]
