from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from dlkit.forge import (
    ManifestError,
    ProjectManifest,
    discover_project_dirs,
    inspect_project,
    run_plugin_bootstraps,
)
import dlkit.registry as registry_module


VALID_MANIFEST = """\
schema_version: 1
project:
  id: demo
  display_name: Demo Project
  package: demo_project
  version: 0.1.0
framework:
  package: blueprintdl
  version: ">=0.1"
plugins:
  bootstrap:
    - demo_project.plugins:register_plugins
delivery:
  include:
    - configs/**
    - README.md
"""


def _valid_mapping(*, bootstrap=None, include=None):
    return {
        "schema_version": 1,
        "project": {
            "id": "demo",
            "display_name": "Demo Project",
            "package": "demo_project",
            "version": "0.1.0",
        },
        "framework": {"package": "blueprintdl", "version": ">=0.1"},
        "plugins": {
            "bootstrap": bootstrap
            if bootstrap is not None
            else ["demo_project.plugins:register_plugins"]
        },
        "delivery": {
            "include": include
            if include is not None
            else ["configs/**", "README.md"]
        },
    }


def test_load_valid_manifest_from_file_or_project_dir(tmp_path):
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    manifest_path = project_dir / "project.yaml"
    manifest_path.write_text(VALID_MANIFEST, encoding="utf-8")

    from_file = ProjectManifest.load(manifest_path)
    from_dir = ProjectManifest.load(project_dir)

    assert from_file == from_dir
    assert from_file.schema_version == 1
    assert from_file.project.id == "demo"
    assert from_file.project.display_name == "Demo Project"
    assert from_file.framework.package == "blueprintdl"
    assert from_file.plugins.bootstrap == (
        "demo_project.plugins:register_plugins",
    )
    assert from_file.delivery.include == ("configs/**", "README.md")

    data = from_file.to_dict()
    assert json.loads(json.dumps(data)) == data
    data["plugins"]["bootstrap"].append("mutated:locally")
    assert from_file.plugins.bootstrap == (
        "demo_project.plugins:register_plugins",
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(schema_version=2), "schema_version"),
        (lambda value: value.update(schema_version=True), "schema_version"),
        (lambda value: value.pop("delivery"), "missing 'delivery'"),
        (lambda value: value.update(extra={}), "unknown 'extra'"),
        (lambda value: value["project"].pop("package"), "project"),
        (lambda value: value["framework"].update(extra="x"), "framework"),
        (
            lambda value: value["plugins"].update(
                bootstrap="demo_project.plugins:register_plugins"
            ),
            "plugins.bootstrap",
        ),
        (
            lambda value: value["delivery"].update(include=["configs", ""]),
            r"delivery\.include\[1\]",
        ),
        (lambda value: value["project"].update(id="../escape"), "project.id"),
        (
            lambda value: value["project"].update(package="Invalid-Package"),
            "project.package",
        ),
        (
            lambda value: value["plugins"].update(bootstrap=[]),
            "plugins.bootstrap",
        ),
        (
            lambda value: value["delivery"].update(include=[]),
            "delivery.include",
        ),
        (
            lambda value: value["delivery"].update(include=["../secret"]),
            r"delivery\.include\[0\]",
        ),
        (
            lambda value: value["delivery"].update(include=[".venv"]),
            r"delivery\.include\[0\]",
        ),
    ],
)
def test_manifest_schema_is_strict(mutate, message):
    value = _valid_mapping()
    mutate(value)

    with pytest.raises(ManifestError, match=message):
        ProjectManifest.from_mapping(value)


@pytest.mark.parametrize(
    "reference",
    [
        "demo_project.plugins",
        "demo_project.plugins:register:again",
        "bad-module.plugins:register",
        "demo_project.plugins:not.a.function",
    ],
)
def test_bootstrap_reference_must_be_module_colon_function(reference):
    with pytest.raises(ManifestError, match=r"plugins\.bootstrap\[0\]"):
        ProjectManifest.from_mapping(_valid_mapping(bootstrap=[reference]))


def test_safe_yaml_rejects_python_tags_and_duplicate_keys(tmp_path):
    unsafe = tmp_path / "unsafe.yaml"
    unsafe.write_text(
        "schema_version: !!python/object/new:tuple []\n", encoding="utf-8"
    )
    with pytest.raises(ManifestError, match="invalid YAML"):
        ProjectManifest.load(unsafe)

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        VALID_MANIFEST.replace(
            "schema_version: 1", "schema_version: 1\nschema_version: 1", 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="duplicate key"):
        ProjectManifest.load(duplicate)


def test_run_plugin_bootstraps_imports_only_explicit_references_in_order():
    manifest = ProjectManifest.from_mapping(
        _valid_mapping(
            bootstrap=[
                "demo_project.plugins:register_plugins",
                "extra_plugin.bootstrap:setup",
            ]
        )
    )
    events = []
    modules = {
        "demo_project.plugins": SimpleNamespace(
            register_plugins=lambda: events.append("demo")
        ),
        "extra_plugin.bootstrap": SimpleNamespace(
            setup=lambda: events.append("extra")
        ),
    }

    def importer(module_name):
        events.append(f"import:{module_name}")
        return modules[module_name]

    before_sys_path = tuple(sys.path)
    invoked = run_plugin_bootstraps(manifest, importer=importer)

    assert invoked == (
        "demo_project.plugins:register_plugins",
        "extra_plugin.bootstrap:setup",
    )
    assert events == [
        "import:demo_project.plugins",
        "demo",
        "import:extra_plugin.bootstrap",
        "extra",
    ]
    assert tuple(sys.path) == before_sys_path


def test_run_plugin_bootstraps_rejects_non_callable_target():
    manifest = ProjectManifest.from_mapping(_valid_mapping())

    with pytest.raises(ManifestError, match="non-callable"):
        run_plugin_bootstraps(
            manifest,
            importer=lambda _: SimpleNamespace(register_plugins="not callable"),
        )


def test_discover_project_dirs_is_shallow_sorted_and_read_only(tmp_path):
    projects_dir = tmp_path / "workspace" / "projects"
    projects_dir.mkdir(parents=True)
    alpha = projects_dir / "Alpha"
    zeta = projects_dir / "zeta"
    ignored = projects_dir / "ignored"
    nested = ignored / "nested"
    for directory in (alpha, zeta, nested):
        directory.mkdir(parents=True)
        (directory / "project.yaml").write_text(
            VALID_MANIFEST, encoding="utf-8"
        )
    (projects_dir / "project.yaml").write_text(VALID_MANIFEST, encoding="utf-8")

    before_sys_path = tuple(sys.path)
    found = discover_project_dirs(projects_dir)

    assert found == (alpha, zeta)
    assert tuple(sys.path) == before_sys_path
    assert discover_project_dirs(tmp_path / "missing") == ()


@dataclass
class _FakeComponentRecord:
    registry: str
    name: str
    target: object
    provider: str
    module: str
    qualname: str
    tags: tuple[str, ...]
    description: str | None


def test_inspect_project_filters_by_provider_and_omits_live_targets(monkeypatch):
    manifest = ProjectManifest.from_mapping(_valid_mapping())
    target = object()
    calls = []

    def iter_component_records(*, provider=None, registry=None):
        calls.append({"provider": provider, "registry": registry})
        return (
            _FakeComponentRecord(
                registry="models",
                name="z_model",
                target=target,
                provider="demo",
                module="demo_project.models",
                qualname="ZModel",
                tags=("vision", "example"),
                description="A model",
            ),
            _FakeComponentRecord(
                registry="datasets",
                name="a_dataset",
                target=target,
                provider="demo",
                module="demo_project.data",
                qualname="ADataset",
                tags=(),
                description=None,
            ),
        )

    monkeypatch.setattr(
        registry_module,
        "iter_component_records",
        iter_component_records,
        raising=False,
    )

    result = inspect_project(manifest)

    assert calls == [{"provider": "demo", "registry": None}]
    assert [item["name"] for item in result["components"]] == [
        "a_dataset",
        "z_model",
    ]
    assert set(result["components"][0]) == {
        "registry",
        "name",
        "provider",
        "module",
        "qualname",
        "tags",
        "description",
    }
    assert result["components"][1]["tags"] == ["vision", "example"]
    assert json.loads(json.dumps(result)) == result
