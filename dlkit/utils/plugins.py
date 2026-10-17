"""项目插件加载:把 ``config['imports']`` 声明的 .py 模块导入并完成注册。

约定:
- 条目可以是单个 ``.py`` 文件,也可以是目录(导入目录下所有 ``*.py``,不递归);
- 相对路径按当前工作目录解析(在仓库根运行工具是标准做法);
- 同一文件在进程内只执行一次(按解析后的绝对路径缓存),避免重复注册;
- 目录不存在会报错(暴露配置拼写错误),空目录容忍(项目骨架默认
  ``projects/<id>/extensions/`` 允许为空)。

典型用途:项目专属 Dataset/Loss/Metric/Model 组件与 Recipe(流程插件)。
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

_LOADED: dict[str, str] = {}


def load_plugin_modules(entries, *, base_dir=None) -> list[str]:
    """导入配置声明的插件模块,返回本次新加载文件的展示路径列表。"""

    loaded: list[str] = []
    base = Path(base_dir) if base_dir is not None else Path.cwd()

    for raw in entries or []:
        entry = Path(str(raw))
        if not entry.is_absolute():
            entry = base / entry
        if not entry.exists():
            raise FileNotFoundError('imports entry not found: %s' % entry)

        files = (
            sorted(entry.glob('*.py'), key=lambda p: p.name.casefold())
            if entry.is_dir()
            else [entry]
        )
        for path in files:
            resolved = str(path.resolve())
            if resolved in _LOADED:
                continue
            module_name = 'dlkit_plugin_' + hashlib.sha1(
                resolved.encode('utf-8')).hexdigest()[:12]
            spec = importlib.util.spec_from_file_location(module_name, resolved)
            if spec is None or spec.loader is None:
                raise ImportError('cannot load plugin module: %s' % path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            _LOADED[resolved] = module_name
            loaded.append(display_path(Path(resolved)))

    return loaded


def display_path(path: Path) -> str:
    """尽量转成相对 CWD 的 posix 路径(便于写进 run.json 与报告)。"""

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


__all__ = ['display_path', 'load_plugin_modules']
