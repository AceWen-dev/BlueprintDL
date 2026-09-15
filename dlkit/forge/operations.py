"""State-changing and repository-aware operations for BlueprintDL Forge."""

from __future__ import annotations

import ast
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import re
import shutil
import sys
import tomllib
from typing import Iterator
from uuid import uuid4

import yaml

from dlkit.forge.manifest import (
    ProjectManifest,
    discover_project_dirs,
    inspect_project,
    load_project_manifest,
    run_plugin_bootstraps,
)


_PROJECT_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_PACKAGE_RE = re.compile(r"^[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*$")
_DEPENDENCY_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)")
_GLOB_CHARS = frozenset("*?[")
_DELIVERY_IGNORE = shutil.ignore_patterns(
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "*.pyc",
    "*.pyo",
    "*.egg-info",
)


class ForgeError(RuntimeError):
    """Raised when a Forge operation cannot complete safely."""


@dataclass(frozen=True, slots=True)
class AuditCheck:
    """One observable project-boundary check."""

    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Serializable result of auditing one forged project."""

    project_id: str
    project_root: str
    checks: tuple[AuditCheck, ...]
    components: tuple[dict[str, object], ...] = ()

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "project_root": self.project_root,
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
            "components": [dict(component) for component in self.components],
        }


def _normalize_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _dependency_names(values: object) -> set[str]:
    if not isinstance(values, list):
        return set()
    result = set()
    for value in values:
        if not isinstance(value, str):
            continue
        match = _DEPENDENCY_NAME_RE.match(value)
        if match:
            result.add(_normalize_distribution_name(match.group(1)))
    return result


def _safe_delivery_entry(project_root: Path, entry: str) -> tuple[Path, ...]:
    relative = Path(entry)
    if relative.is_absolute() or ".." in relative.parts:
        raise ForgeError(
            f"delivery.include entry must stay inside the project: {entry!r}"
        )

    if any(char in entry for char in _GLOB_CHARS):
        candidates = tuple(project_root.glob(entry))
    else:
        candidates = (project_root / relative,)

    root_resolved = project_root.resolve()
    safe = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root_resolved):
            raise ForgeError(
                f"delivery.include entry escaped the project: {entry!r}"
            )
        safe.append(candidate)
    return tuple(safe)


def _delivery_sources(
    project_root: Path, manifest: ProjectManifest
) -> tuple[Path, ...]:
    sources: dict[str, Path] = {}
    for entry in manifest.delivery.include:
        candidates = _safe_delivery_entry(project_root, entry)
        if not candidates:
            raise ForgeError(f"delivery.include matched nothing: {entry!r}")
        for candidate in candidates:
            if not candidate.exists():
                raise ForgeError(f"delivery.include path does not exist: {entry!r}")
            relative = candidate.relative_to(project_root)
            sources[relative.as_posix()] = candidate

    selected = []
    for relative, candidate in sorted(sources.items()):
        if any(
            relative != parent
            and relative.startswith(parent.rstrip("/") + "/")
            for parent in sources
            if (project_root / parent).is_dir()
        ):
            continue
        selected.append(candidate)
    return tuple(selected)


def _read_project_metadata(pyproject_path: Path) -> dict[str, object]:
    try:
        with pyproject_path.open("rb") as stream:
            data = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ForgeError(f"could not read {pyproject_path}: {exc}") from exc
    project = data.get("project")
    if not isinstance(project, dict):
        raise ForgeError(f"{pyproject_path}: missing [project] table")
    return project


def _sibling_packages(project_root: Path) -> set[str]:
    packages = set()
    for sibling in discover_project_dirs(project_root.parent):
        if sibling.resolve() == project_root.resolve():
            continue
        try:
            manifest = load_project_manifest(sibling)
        except ValueError:
            continue
        packages.add(manifest.project.package.split(".", 1)[0])
    return packages


def _cross_project_imports(project_root: Path) -> tuple[str, ...]:
    forbidden = _sibling_packages(project_root)
    if not forbidden:
        return ()

    violations = []
    for source_path in sorted((project_root / "src").rglob("*.py")):
        try:
            tree = ast.parse(
                source_path.read_text(encoding="utf-8"),
                filename=str(source_path),
            )
        except (OSError, SyntaxError) as exc:
            violations.append(f"{source_path.relative_to(project_root)}: {exc}")
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                root_name = name.split(".", 1)[0]
                if root_name in forbidden:
                    violations.append(
                        f"{source_path.relative_to(project_root)}:{node.lineno} imports {name}"
                    )
    return tuple(violations)


@contextmanager
def _project_source_path(project_root: Path) -> Iterator[None]:
    """Expose one declared src tree only while Forge loads its bootstrap."""

    source = str((project_root / "src").resolve())
    sys.path.insert(0, source)
    importlib.invalidate_caches()
    try:
        yield
    finally:
        try:
            sys.path.remove(source)
        except ValueError:
            pass
        importlib.invalidate_caches()


def audit_project(
    project_path: str | Path,
    *,
    load_plugins: bool = False,
) -> AuditReport:
    """Audit manifest, packaging, ownership, and delivery boundaries."""

    project_root = Path(project_path).resolve()
    manifest = load_project_manifest(project_root)
    checks = []

    checks.append(
        AuditCheck(
            "project-directory",
            project_root.name == manifest.project.id,
            (
                "directory matches project.id"
                if project_root.name == manifest.project.id
                else f"directory {project_root.name!r} must equal {manifest.project.id!r}"
            ),
        )
    )

    pyproject_path = project_root / "pyproject.toml"
    try:
        metadata = _read_project_metadata(pyproject_path)
        actual_name = metadata.get("name")
        actual_version = metadata.get("version")
        dependencies = _dependency_names(metadata.get("dependencies"))
        framework_name = _normalize_distribution_name(manifest.framework.package)
        package_ok = (
            actual_name == manifest.project.package
            and actual_version == manifest.project.version
            and framework_name in dependencies
        )
        package_detail = (
            "pyproject identity and framework dependency match the manifest"
            if package_ok
            else (
                f"expected name={manifest.project.package!r}, "
                f"version={manifest.project.version!r}, dependency="
                f"{manifest.framework.package!r}; got name={actual_name!r}, "
                f"version={actual_version!r}"
            )
        )
    except ForgeError as exc:
        package_ok = False
        package_detail = str(exc)
    checks.append(AuditCheck("python-metadata", package_ok, package_detail))

    package_path = project_root / "src"
    for part in manifest.project.package.split("."):
        package_path /= part
    package_ok = package_path.is_dir() and (package_path / "__init__.py").is_file()
    checks.append(
        AuditCheck(
            "source-package",
            package_ok,
            (
                f"source package exists at {package_path.relative_to(project_root)}"
                if package_ok
                else f"missing source package {package_path.relative_to(project_root)}"
            ),
        )
    )

    bootstrap_prefix = manifest.project.package + "."
    bootstrap_ok = all(
        reference.split(":", 1)[0].startswith(bootstrap_prefix)
        for reference in manifest.plugins.bootstrap
    )
    checks.append(
        AuditCheck(
            "bootstrap-ownership",
            bootstrap_ok,
            (
                "all bootstrap modules belong to the project package"
                if bootstrap_ok
                else "bootstrap modules must live under the project package"
            ),
        )
    )

    try:
        delivery_sources = _delivery_sources(project_root, manifest)
        delivery_ok = True
        delivery_detail = f"{len(delivery_sources)} delivery roots are safe and present"
    except ForgeError as exc:
        delivery_ok = False
        delivery_detail = str(exc)
    checks.append(AuditCheck("delivery-boundary", delivery_ok, delivery_detail))

    violations = _cross_project_imports(project_root)
    checks.append(
        AuditCheck(
            "cross-project-imports",
            not violations,
            "no sibling project imports found" if not violations else "; ".join(violations),
        )
    )

    components: tuple[dict[str, object], ...] = ()
    if load_plugins:
        try:
            with _project_source_path(project_root):
                run_plugin_bootstraps(manifest)
            inspected = inspect_project(manifest)
            components = tuple(inspected["components"])
            plugin_ok = True
            plugin_detail = (
                f"loaded {len(manifest.plugins.bootstrap)} bootstrap(s); "
                f"found {len(components)} owned component(s)"
            )
        except Exception as exc:
            plugin_ok = False
            plugin_detail = str(exc)
        checks.append(AuditCheck("plugin-bootstrap", plugin_ok, plugin_detail))

        if plugin_ok:
            from dlkit.registry import iter_component_records

            package_prefix = manifest.project.package + "."
            missing_provider = [
                f"{record.registry}.{record.name}"
                for record in iter_component_records()
                if (
                    record.module == manifest.project.package
                    or record.module.startswith(package_prefix)
                )
                and record.provider != manifest.project.id
            ]
            ownership_ok = not missing_provider
            ownership_detail = (
                "all project-module components declare the project provider"
                if ownership_ok
                else "missing or incorrect provider: " + ", ".join(missing_provider)
            )
            checks.append(
                AuditCheck(
                    "component-ownership",
                    ownership_ok,
                    ownership_detail,
                )
            )

    return AuditReport(
        project_id=manifest.project.id,
        project_root=str(project_root),
        checks=tuple(checks),
        components=components,
    )


def inspect_project_source(project_path: str | Path) -> dict[str, object]:
    """Load one project's explicit bootstrap and return serializable inventory."""

    project_root = Path(project_path).resolve()
    manifest = load_project_manifest(project_root)
    with _project_source_path(project_root):
        run_plugin_bootstraps(manifest)
    return inspect_project(manifest)


def export_project(
    project_path: str | Path,
    destination_dir: str | Path,
) -> Path:
    """Copy the declared delivery boundary into a new, non-overwritten directory."""

    project_root = Path(project_path).resolve()
    manifest = load_project_manifest(project_root)
    report = audit_project(project_root, load_plugins=True)
    if not report.passed:
        failures = "; ".join(
            f"{check.name}: {check.detail}"
            for check in report.checks
            if not check.passed
        )
        raise ForgeError(f"project audit failed: {failures}")

    destination_root = Path(destination_dir).resolve()
    if destination_root == project_root or destination_root.is_relative_to(project_root):
        raise ForgeError("delivery destination must be outside the project directory")
    destination_root.mkdir(parents=True, exist_ok=True)
    target = destination_root / manifest.project.id
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing delivery: {target}")

    temporary = destination_root / f".{manifest.project.id}.tmp-{uuid4().hex}"
    temporary.mkdir()
    try:
        for source in _delivery_sources(project_root, manifest):
            relative = source.relative_to(project_root)
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, destination, ignore=_DELIVERY_IGNORE)
            else:
                shutil.copy2(source, destination)

        audit_receipt = report.to_dict()
        audit_receipt["project_root"] = "."
        receipt = {
            "forge_schema_version": 1,
            "project": manifest.to_dict(),
            "audit": audit_receipt,
        }
        (temporary / ".blueprintdl-export.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.rename(target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return target


def _validate_scaffold_identity(project_id: str, package: str) -> None:
    if not _PROJECT_ID_RE.fullmatch(project_id):
        raise ForgeError(
            "project_id must use lowercase letters, digits, and single hyphens"
        )
    if not _PACKAGE_RE.fullmatch(package):
        raise ForgeError(
            "package must be a lowercase Python package name, optionally dotted"
        )


def scaffold_project(
    projects_dir: str | Path,
    project_id: str,
    *,
    display_name: str | None = None,
    package: str | None = None,
    version: str = "0.1.0",
    framework_version: str = ">=0.1,<0.2",
    python_version: str = "3.12",
) -> Path:
    """Create an independently packaged project skeleton without overwriting."""

    normalized_id = project_id.strip()
    package_name = (package or normalized_id.replace("-", "_")).strip()
    title = (display_name or normalized_id).strip()
    _validate_scaffold_identity(normalized_id, package_name)
    if not title or "\n" in title or "\r" in title:
        raise ForgeError("display_name must be a non-empty single line")
    if not version or "\n" in version or "\r" in version:
        raise ForgeError("version must be a non-empty single line")
    if not framework_version or "\n" in framework_version or "\r" in framework_version:
        raise ForgeError("framework_version must be a non-empty single line")
    if not python_version or "\n" in python_version or "\r" in python_version:
        raise ForgeError("python_version must be a non-empty single line")

    root = Path(projects_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / normalized_id
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing project: {target}")
    temporary = root / f".{normalized_id}.tmp-{uuid4().hex}"
    temporary.mkdir()

    package_path = Path("src", *package_name.split("."))
    manifest = {
        "schema_version": 1,
        "project": {
            "id": normalized_id,
            "display_name": title,
            "package": package_name,
            "version": version,
        },
        "framework": {
            "package": "blueprintdl",
            "version": framework_version,
        },
        "plugins": {"bootstrap": [f"{package_name}.bootstrap:register"]},
        "delivery": {
            "include": [
                "pyproject.toml",
                "project.yaml",
                "README.md",
                "src",
                "configs",
                "tests",
            ]
        },
    }

    def write(relative: str | Path, content: str) -> None:
        path = temporary / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    try:
        write(
            "project.yaml",
            yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        )
        write(
            "pyproject.toml",
            "[build-system]\n"
            'requires = ["hatchling"]\n'
            'build-backend = "hatchling.build"\n\n'
            "[project]\n"
            f"name = {json.dumps(package_name)}\n"
            f"version = {json.dumps(version)}\n"
            f"description = {json.dumps(title + ' project forged with BlueprintDL.')}\n"
            'readme = "README.md"\n'
            'requires-python = ">=3.10"\n'
            f"dependencies = [{json.dumps('blueprintdl' + framework_version)}]\n\n"
            "[dependency-groups]\n"
            'dev = ["pytest>=8.0"]\n\n'
            "[tool.hatch.build.targets.wheel]\n"
            f"packages = [{json.dumps(package_path.as_posix())}]\n\n"
            "[tool.pytest.ini_options]\n"
            'testpaths = ["tests"]\n'
            'pythonpath = ["src"]\n',
        )
        write(".python-version", python_version + "\n")
        write(
            ".gitignore",
            "__pycache__/\n*.py[cod]\n.venv/\n.pytest_cache/\n"
            "*.egg-info/\nbuild/\ndist/\nruns/\ndata/\n"
            "*.pth\n*.pt\n*.onnx\n",
        )
        write(
            "README.md",
            f"# {title}\n\n"
            f"Project `{normalized_id}` is forged under the BlueprintDL project standard.\n",
        )
        write(
            package_path / "__init__.py",
            f'PROJECT_ID = "{normalized_id}"\n'
            f"__version__ = {json.dumps(version)}\n\n"
            '__all__ = ["PROJECT_ID", "__version__"]\n',
        )
        write(
            package_path / "bootstrap.py",
            '"""Explicit project plugin bootstrap."""\n\n'
            "def register():\n"
            f"    from {package_name}.components import load_components\n\n"
            "    return load_components()\n",
        )
        write(
            package_path / "components" / "__init__.py",
            '"""Explicit project-owned component imports."""\n\n'
            "def load_components():\n"
            "    return ()\n\n"
            '__all__ = ["load_components"]\n',
        )
        write(
            "configs/README.md",
            "# Configurations\n\nKeep serializable project configurations here.\n",
        )
        write(
            "tests/test_project_contract.py",
            f"from {package_name} import PROJECT_ID, __version__\n"
            f"from {package_name}.bootstrap import register\n\n\n"
            "def test_project_identity():\n"
            f'    assert PROJECT_ID == "{normalized_id}"\n'
            f'    assert __version__ == "{version}"\n\n\n'
            "def test_bootstrap_is_repeatable():\n"
            "    assert register() == ()\n"
            "    assert register() == ()\n",
        )
        temporary.rename(target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return target


__all__ = [
    "AuditCheck",
    "AuditReport",
    "ForgeError",
    "audit_project",
    "export_project",
    "inspect_project_source",
    "scaffold_project",
]
