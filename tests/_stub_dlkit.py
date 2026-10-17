"""测试环境的 dlkit 降级助手。

在 torch 可用的正常环境(如 uv)里,什么都不做——真实 dlkit 优先。
在 torch 损坏的环境(如本仓库开发机的 Anaconda)里,把 dlkit 命名空间与
`dlkit.utils.runs` / `dlkit.utils.config` / `dlkit.utils.experiments` 按文件
路径加载成可用模块,使这些纯逻辑函数照常测试(chromium 各测试自带守卫)。
"""

import importlib.util
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PATH_MODULES = (
    ('dlkit.utils.runs', 'dlkit/utils/runs.py'),
    ('dlkit.utils.config', 'dlkit/utils/config.py'),
    ('dlkit.utils.experiments', 'dlkit/utils/experiments.py'),
    ('dlkit.utils.data_audit', 'dlkit/utils/data_audit.py'),
    ('dlkit.utils.plugins', 'dlkit/utils/plugins.py'),
)


def _load_by_path(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, *relpath.split('/')))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install(force: bool = False) -> bool:
    """需要时安装路径级 stub;返回 True 表示安装了 stub,False 表示真实 dlkit 可用。"""

    if not force:
        try:
            import dlkit  # noqa: F401

            return False
        except Exception:
            pass

    for name in ('dlkit', 'dlkit.utils'):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module

    for name, relpath in _PATH_MODULES:
        if name not in sys.modules:
            _load_by_path(name, relpath)
    return True
