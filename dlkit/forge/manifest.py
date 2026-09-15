"""Project manifest loading and read-only Forge inspection helpers.

The module intentionally keeps project discovery separate from Python import
discovery.  Only bootstrap callables explicitly listed in ``project.yaml`` are
imported, and no helper in this module mutates ``sys.path``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import importlib
from os import PathLike
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver


class ManifestError(ValueError):
    """Raised when a project manifest is missing, unsafe, or invalid."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """A SafeLoader variant that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


_BOOTSTRAP_MODULE_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)
_BOOTSTRAP_FUNCTION_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PROJECT_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_PACKAGE_RE = re.compile(
    r"^[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*$"
)


def _require_non_empty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{path}: expected a non-empty string")
    return value.strip()


def _require_mapping(value: object, path: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{path}: expected a mapping")
    return value


def _format_keys(keys: Iterable[object]) -> str:
    return ", ".join(sorted((repr(key) for key in keys), key=str.casefold))


def _require_exact_keys(
    value: Mapping[object, object],
    expected: set[str],
    path: str,
) -> None:
    actual = set(value)
    missing = expected - actual
    unknown = actual - expected
    details = []
    if missing:
        details.append(f"missing {_format_keys(missing)}")
    if unknown:
        details.append(f"unknown {_format_keys(unknown)}")
    if details:
        raise ManifestError(f"{path}: {'; '.join(details)}")


def _require_string_list(
    value: object,
    path: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ManifestError(f"{path}: expected a list")

    result = []
    seen = set()
    for index, item in enumerate(value):
        normalized = _require_non_empty_string(item, f"{path}[{index}]")
        if normalized in seen:
            raise ManifestError(
                f"{path}[{index}]: duplicate value {normalized!r}"
            )
        seen.add(normalized)
        result.append(normalized)
    if not result and not allow_empty:
        raise ManifestError(f"{path}: expected at least one entry")
    return tuple(result)


def _validate_delivery_entry(entry: str, path: str) -> None:
    for path_type in (PurePosixPath, PureWindowsPath):
        candidate = path_type(entry)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ManifestError(f"{path}: path must stay inside the project")
    if entry in (".", "./", ".\\"):
        raise ManifestError(f"{path}: project root is too broad")
    first = re.split(r"[/\\]", entry, maxsplit=1)[0]
    if first in {".git", ".venv", "__pycache__", ".pytest_cache"}:
        raise ManifestError(f"{path}: generated or private path is not deliverable")


def _split_bootstrap_reference(reference: str, path: str) -> tuple[str, str]:
    if reference.count(":") != 1:
        raise ManifestError(f"{path}: expected 'module:function'")
    module_name, function_name = reference.split(":", 1)
    if not _BOOTSTRAP_MODULE_RE.fullmatch(module_name):
        raise ManifestError(f"{path}: invalid bootstrap module {module_name!r}")
    if not _BOOTSTRAP_FUNCTION_RE.fullmatch(function_name):
        raise ManifestError(
            f"{path}: invalid bootstrap function {function_name!r}"
        )
    return module_name, function_name


@dataclass(frozen=True, slots=True)
class ProjectSpec:
    """Serializable identity and package metadata for one project."""

    id: str
    display_name: str
    package: str
    version: str

    def __post_init__(self) -> None:
        project_id = _require_non_empty_string(self.id, "project.id")
        if not _PROJECT_ID_RE.fullmatch(project_id):
            raise ManifestError(
                "project.id: expected lowercase letters, digits, and single hyphens"
            )
        _require_non_empty_string(self.display_name, "project.display_name")
        package = _require_non_empty_string(self.package, "project.package")
        if not _PACKAGE_RE.fullmatch(package):
            raise ManifestError("project.package: expected a lowercase Python package")
        _require_non_empty_string(self.version, "project.version")


@dataclass(frozen=True, slots=True)
class FrameworkSpec:
    """Framework package compatibility declared by a project."""

    package: str
    version: str

    def __post_init__(self) -> None:
        _require_non_empty_string(self.package, "framework.package")
        _require_non_empty_string(self.version, "framework.version")


@dataclass(frozen=True, slots=True)
class PluginSpec:
    """Explicit, ordered plugin bootstrap references."""

    bootstrap: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.bootstrap, tuple):
            raise ManifestError("plugins.bootstrap: expected an immutable tuple")
        for index, reference in enumerate(self.bootstrap):
            normalized = _require_non_empty_string(
                reference, f"plugins.bootstrap[{index}]"
            )
            _split_bootstrap_reference(
                normalized, f"plugins.bootstrap[{index}]"
            )


@dataclass(frozen=True, slots=True)
class DeliverySpec:
    """Project-relative delivery entries retained as declarative data."""

    include: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.include, tuple):
            raise ManifestError("delivery.include: expected an immutable tuple")
        for index, entry in enumerate(self.include):
            normalized = _require_non_empty_string(
                entry, f"delivery.include[{index}]"
            )
            _validate_delivery_entry(
                normalized, f"delivery.include[{index}]"
            )


@dataclass(frozen=True, slots=True)
class ProjectManifest:
    """Validated schema-version-1 representation of ``project.yaml``."""

    schema_version: int
    project: ProjectSpec
    framework: FrameworkSpec
    plugins: PluginSpec
    delivery: DeliverySpec

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ManifestError("schema_version: expected integer 1")
        if not isinstance(self.project, ProjectSpec):
            raise ManifestError("project: expected ProjectSpec")
        if not isinstance(self.framework, FrameworkSpec):
            raise ManifestError("framework: expected FrameworkSpec")
        if not isinstance(self.plugins, PluginSpec):
            raise ManifestError("plugins: expected PluginSpec")
        if not isinstance(self.delivery, DeliverySpec):
            raise ManifestError("delivery: expected DeliverySpec")

    @classmethod
    def from_mapping(cls, value: object) -> "ProjectManifest":
        """Validate a parsed YAML value without mutating caller-owned data."""

        root = _require_mapping(value, "manifest")
        _require_exact_keys(
            root,
            {"schema_version", "project", "framework", "plugins", "delivery"},
            "manifest",
        )

        schema_version = root["schema_version"]
        if type(schema_version) is not int or schema_version != 1:
            raise ManifestError("schema_version: expected integer 1")

        project = _require_mapping(root["project"], "project")
        _require_exact_keys(
            project,
            {"id", "display_name", "package", "version"},
            "project",
        )

        framework = _require_mapping(root["framework"], "framework")
        _require_exact_keys(framework, {"package", "version"}, "framework")

        plugins = _require_mapping(root["plugins"], "plugins")
        _require_exact_keys(plugins, {"bootstrap"}, "plugins")

        delivery = _require_mapping(root["delivery"], "delivery")
        _require_exact_keys(delivery, {"include"}, "delivery")

        bootstrap = _require_string_list(
            plugins["bootstrap"], "plugins.bootstrap", allow_empty=False
        )
        for index, reference in enumerate(bootstrap):
            _split_bootstrap_reference(
                reference, f"plugins.bootstrap[{index}]"
            )

        return cls(
            schema_version=1,
            project=ProjectSpec(
                id=_require_non_empty_string(project["id"], "project.id"),
                display_name=_require_non_empty_string(
                    project["display_name"], "project.display_name"
                ),
                package=_require_non_empty_string(
                    project["package"], "project.package"
                ),
                version=_require_non_empty_string(
                    project["version"], "project.version"
                ),
            ),
            framework=FrameworkSpec(
                package=_require_non_empty_string(
                    framework["package"], "framework.package"
                ),
                version=_require_non_empty_string(
                    framework["version"], "framework.version"
                ),
            ),
            plugins=PluginSpec(bootstrap=bootstrap),
            delivery=DeliverySpec(
                include=_require_string_list(
                    delivery["include"],
                    "delivery.include",
                    allow_empty=False,
                )
            ),
        )

    @classmethod
    def load(cls, path: str | PathLike[str]) -> "ProjectManifest":
        """Safely load a manifest file, or ``project.yaml`` within a directory."""

        manifest_path = Path(path)
        if manifest_path.is_dir():
            manifest_path = manifest_path / "project.yaml"

        try:
            with manifest_path.open("r", encoding="utf-8") as stream:
                value = yaml.load(stream, Loader=_UniqueKeySafeLoader)
        except OSError as exc:
            raise ManifestError(
                f"could not read manifest {str(manifest_path)!r}: {exc}"
            ) from exc
        except yaml.YAMLError as exc:
            raise ManifestError(
                f"invalid YAML in manifest {str(manifest_path)!r}: {exc}"
            ) from exc

        try:
            return cls.from_mapping(value)
        except ManifestError as exc:
            raise ManifestError(f"{manifest_path}: {exc}") from exc

    def to_dict(self) -> dict[str, object]:
        """Return a fresh, serialization-friendly representation."""

        return {
            "schema_version": self.schema_version,
            "project": {
                "id": self.project.id,
                "display_name": self.project.display_name,
                "package": self.project.package,
                "version": self.project.version,
            },
            "framework": {
                "package": self.framework.package,
                "version": self.framework.version,
            },
            "plugins": {"bootstrap": list(self.plugins.bootstrap)},
            "delivery": {"include": list(self.delivery.include)},
        }


def load_project_manifest(path: str | PathLike[str]) -> ProjectManifest:
    """Load ``project.yaml`` through :class:`ProjectManifest`."""

    return ProjectManifest.load(path)


def run_plugin_bootstraps(
    manifest: ProjectManifest,
    *,
    importer: Callable[[str], object] | None = None,
) -> tuple[str, ...]:
    """Import and invoke the manifest's explicit ``module:function`` hooks.

    Hooks are resolved in manifest order.  No packages are scanned and no
    search paths are modified.
    """

    if not isinstance(manifest, ProjectManifest):
        raise TypeError("manifest must be a ProjectManifest")
    import_module = importer or importlib.import_module
    invoked = []

    for index, reference in enumerate(manifest.plugins.bootstrap):
        path = f"plugins.bootstrap[{index}]"
        module_name, function_name = _split_bootstrap_reference(reference, path)
        try:
            module = import_module(module_name)
        except Exception as exc:
            raise ManifestError(
                f"{path}: could not import module {module_name!r}: {exc}"
            ) from exc

        try:
            hook = getattr(module, function_name)
        except AttributeError as exc:
            raise ManifestError(
                f"{path}: module {module_name!r} has no function "
                f"{function_name!r}"
            ) from exc
        if not callable(hook):
            raise ManifestError(
                f"{path}: {reference!r} resolved to a non-callable object"
            )

        try:
            hook()
        except Exception as exc:
            raise ManifestError(
                f"{path}: bootstrap callable {reference!r} failed: {exc}"
            ) from exc
        invoked.append(reference)

    return tuple(invoked)


def discover_project_dirs(
    projects_dir: str | PathLike[str],
) -> tuple[Path, ...]:
    """Return immediate child directories containing ``project.yaml``."""

    root = Path(projects_dir)
    if not root.exists():
        return ()
    if not root.is_dir():
        raise NotADirectoryError(str(root))

    return tuple(
        sorted(
            (
                child
                for child in root.iterdir()
                if child.is_dir() and (child / "project.yaml").is_file()
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )
    )


_COMPONENT_RECORD_FIELDS = (
    "registry",
    "name",
    "provider",
    "module",
    "qualname",
    "tags",
    "description",
)
_MISSING = object()


def _record_field(record: object, field: str) -> object:
    if isinstance(record, Mapping):
        return record.get(field, _MISSING)
    return getattr(record, field, _MISSING)


def _component_record_to_dict(record: object) -> dict[str, object]:
    result = {}
    for field in _COMPONENT_RECORD_FIELDS:
        value = _record_field(record, field)
        if value is _MISSING:
            raise TypeError(f"component record is missing field {field!r}")
        result[field] = value

    for field in ("registry", "name", "provider", "module", "qualname"):
        if not isinstance(result[field], str):
            raise TypeError(f"component record field {field!r} must be a string")

    tags = result["tags"]
    if isinstance(tags, (str, bytes)) or not isinstance(tags, Iterable):
        raise TypeError("component record field 'tags' must be an iterable of strings")
    normalized_tags = list(tags)
    if any(not isinstance(tag, str) for tag in normalized_tags):
        raise TypeError("component record field 'tags' must contain only strings")
    result["tags"] = normalized_tags

    description = result["description"]
    if description is not None and not isinstance(description, str):
        raise TypeError(
            "component record field 'description' must be a string or None"
        )
    return result


def inspect_project(
    manifest: ProjectManifest,
    *,
    records_provider: Callable[..., Iterable[object]] | None = None,
) -> dict[str, object]:
    """Return manifest and registered component metadata as pure data.

    Registry records are requested with ``provider=manifest.project.id``.  The
    live ``target`` object is deliberately omitted from the returned data.
    """

    if not isinstance(manifest, ProjectManifest):
        raise TypeError("manifest must be a ProjectManifest")

    if records_provider is None:
        from dlkit import registry as registry_module

        records_provider = getattr(
            registry_module, "iter_component_records", None
        )
        if records_provider is None:
            raise RuntimeError(
                "dlkit.registry.iter_component_records is unavailable"
            )
    if not callable(records_provider):
        raise TypeError("records_provider must be callable")

    records = records_provider(provider=manifest.project.id)
    components = [_component_record_to_dict(record) for record in records]
    components.sort(
        key=lambda record: (
            str(record["registry"]).casefold(),
            str(record["name"]).casefold(),
            str(record["module"]).casefold(),
            str(record["qualname"]).casefold(),
        )
    )
    return {"manifest": manifest.to_dict(), "components": components}


__all__ = [
    "DeliverySpec",
    "FrameworkSpec",
    "ManifestError",
    "PluginSpec",
    "ProjectManifest",
    "ProjectSpec",
    "discover_project_dirs",
    "inspect_project",
    "load_project_manifest",
    "run_plugin_bootstraps",
]
