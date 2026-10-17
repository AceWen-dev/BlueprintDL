"""插件加载器测试(纯 stdlib,可在无 torch 环境运行)。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))       # tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import _stub_dlkit

_stub_dlkit.install()

from dlkit.utils.plugins import load_plugin_modules  # noqa: E402


def _plugin_code(marker, label):
    return (
        'from pathlib import Path\n'
        'marker = Path(%r)\n'
        'count = 0\n'
        'if marker.exists():\n'
        '    count = int(marker.read_text() or "0")\n'
        'marker.write_text(str(count + 1))\n'
        'LABEL = %r\n'
        % (str(marker), label)
    )


def test_load_directory_and_dedupe(tmp_path):
    extensions = tmp_path / 'extensions'
    extensions.mkdir()
    marker = tmp_path / 'counter.txt'
    (extensions / 'a_plugin.py').write_text(_plugin_code(marker, 'a'), encoding='utf-8')
    (extensions / 'b_plugin.py').write_text(_plugin_code(marker, 'b'), encoding='utf-8')

    loaded = load_plugin_modules([str(extensions)])
    assert len(loaded) == 2
    assert all(path.endswith('.py') for path in loaded)
    assert marker.read_text(encoding='utf-8') == '2'

    # 进程内幂等:同一文件不会重复执行(避免 Registry 重复注册)
    assert load_plugin_modules([str(extensions)]) == []
    assert marker.read_text(encoding='utf-8') == '2'

    # 单文件条目与目录条目共享缓存
    assert load_plugin_modules([extensions / 'a_plugin.py']) == []


def test_single_file_entry(tmp_path):
    marker = tmp_path / 'counter2.txt'
    plugin = tmp_path / 'solo.py'
    plugin.write_text(_plugin_code(marker, 'solo'), encoding='utf-8')
    loaded = load_plugin_modules([plugin])
    assert len(loaded) == 1
    assert marker.read_text(encoding='utf-8') == '1'


def test_missing_entry_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_plugin_modules([str(tmp_path / 'missing_plugins')])


def test_empty_directory_is_tolerated(tmp_path):
    empty = tmp_path / 'empty_extensions'
    empty.mkdir()
    assert load_plugin_modules([str(empty)]) == []


def test_relative_entry_resolves_against_base_dir(tmp_path):
    base = tmp_path / 'repo'
    extensions = base / 'projects' / 'demo' / 'extensions'
    extensions.mkdir(parents=True)
    marker = tmp_path / 'counter3.txt'
    (extensions / 'rel_plugin.py').write_text(_plugin_code(marker, 'rel'), encoding='utf-8')

    loaded = load_plugin_modules(
        ['projects/demo/extensions'], base_dir=base,
    )
    assert len(loaded) == 1
    assert marker.read_text(encoding='utf-8') == '1'
