"""Command-line interface for the BlueprintDL project forge."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

from dlkit.forge import (
    ForgeError,
    ManifestError,
    audit_project,
    discover_project_dirs,
    export_project,
    inspect_project_source,
    load_project_manifest,
    scaffold_project,
)
from dlkit.registry import iter_component_records


def _default_projects_dir() -> Path:
    return Path.cwd() / "projects"


def _project_path(value: str, projects_dir: Path) -> Path:
    direct = Path(value)
    if direct.is_dir() or (direct / "project.yaml").is_file():
        return direct
    return projects_dir / value


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _load_core_components() -> None:
    for module_name in (
        "dlkit.data",
        "dlkit.models",
        "dlkit.metrics",
        "dlkit.engine.optimizers",
        "dlkit.engine.schedulers",
    ):
        importlib.import_module(module_name)


def _list_components(
    *,
    provider: str | None,
    registry: str | None,
    as_json: bool,
) -> int:
    _load_core_components()
    records = sorted(
        iter_component_records(provider=provider, registry=registry),
        key=lambda item: (
            item.registry.casefold(),
            item.name.casefold(),
            item.module.casefold(),
        ),
    )
    components = [
        {
            "registry": record.registry,
            "name": record.name,
            "provider": record.provider,
            "module": record.module,
            "qualname": record.qualname,
            "tags": list(record.tags),
            "description": record.description,
        }
        for record in records
    ]
    if as_json:
        _print_json({"components": components})
    elif not components:
        print("No matching components found")
    else:
        for component in components:
            owner = component["provider"] or "unowned"
            print(
                f"{component['registry']}.{component['name']} | "
                f"provider={owner} | "
                f"{component['module']}.{component['qualname']}"
            )
    return 0


def _list_projects(projects_dir: Path, as_json: bool) -> int:
    projects = []
    failed = False
    for project_dir in discover_project_dirs(projects_dir):
        try:
            manifest = load_project_manifest(project_dir)
        except ManifestError as exc:
            projects.append(
                {
                    "path": str(project_dir.resolve()),
                    "valid": False,
                    "error": str(exc),
                }
            )
            failed = True
            continue
        projects.append(
            {
                "id": manifest.project.id,
                "display_name": manifest.project.display_name,
                "package": manifest.project.package,
                "version": manifest.project.version,
                "path": str(project_dir.resolve()),
                "valid": True,
            }
        )

    if as_json:
        _print_json({"projects": projects})
    elif not projects:
        print(f"No forged projects found in {projects_dir.resolve()}")
    else:
        for item in projects:
            if item["valid"]:
                print(
                    f"{item['id']} {item['version']} | "
                    f"{item['display_name']} | {item['path']}"
                )
            else:
                print(f"INVALID | {item['path']} | {item['error']}")
    return 1 if failed else 0


def _inspect(project: Path, as_json: bool) -> int:
    result = inspect_project_source(project)
    if as_json:
        _print_json(result)
        return 0

    project_data = result["manifest"]["project"]
    print(
        f"{project_data['display_name']} ({project_data['id']}) "
        f"v{project_data['version']}"
    )
    components = result["components"]
    if not components:
        print("Owned components: none registered yet")
        return 0
    print("Owned components:")
    for component in components:
        print(
            f"  {component['registry']}.{component['name']} "
            f"<- {component['module']}.{component['qualname']}"
        )
    return 0


def _audit(project: Path, as_json: bool, load_plugins: bool) -> int:
    report = audit_project(project, load_plugins=load_plugins)
    if as_json:
        _print_json(report.to_dict())
    else:
        status = "PASS" if report.passed else "FAIL"
        print(f"{status} {report.project_id} | {report.project_root}")
        for check in report.checks:
            marker = "PASS" if check.passed else "FAIL"
            print(f"  {marker} {check.name}: {check.detail}")
    return 0 if report.passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create, inspect, audit, and export BlueprintDL projects"
    )
    parser.add_argument(
        "--projects-dir",
        type=Path,
        default=_default_projects_dir(),
        help="project warehouse directory (default: ./projects)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list forged projects")
    list_parser.add_argument("--json", action="store_true")

    components_parser = subparsers.add_parser(
        "components", help="list registered components and their providers"
    )
    components_parser.add_argument("--provider")
    components_parser.add_argument("--registry")
    components_parser.add_argument("--json", action="store_true")

    inspect_parser = subparsers.add_parser(
        "inspect", help="load a project and show its owned components"
    )
    inspect_parser.add_argument("project", help="project id or directory")
    inspect_parser.add_argument("--json", action="store_true")

    audit_parser = subparsers.add_parser(
        "audit", help="check project and delivery boundaries"
    )
    audit_parser.add_argument("project", help="project id or directory")
    audit_parser.add_argument("--json", action="store_true")
    audit_parser.add_argument(
        "--skip-plugins",
        action="store_true",
        help="validate files without importing project bootstrap modules",
    )

    new_parser = subparsers.add_parser("new", help="create a project skeleton")
    new_parser.add_argument("project_id")
    new_parser.add_argument("--display-name")
    new_parser.add_argument("--package")

    export_parser = subparsers.add_parser(
        "export", help="copy the declared project boundary for delivery"
    )
    export_parser.add_argument("project", help="project id or directory")
    export_parser.add_argument("destination", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    projects_dir = args.projects_dir.resolve()

    try:
        if args.command == "list":
            return _list_projects(projects_dir, args.json)
        if args.command == "components":
            return _list_components(
                provider=args.provider,
                registry=args.registry,
                as_json=args.json,
            )
        if args.command == "inspect":
            return _inspect(
                _project_path(args.project, projects_dir), args.json
            )
        if args.command == "audit":
            return _audit(
                _project_path(args.project, projects_dir),
                args.json,
                not args.skip_plugins,
            )
        if args.command == "new":
            created = scaffold_project(
                projects_dir,
                args.project_id,
                display_name=args.display_name,
                package=args.package,
            )
            print(created)
            return 0
        if args.command == "export":
            exported = export_project(
                _project_path(args.project, projects_dir), args.destination
            )
            print(exported)
            return 0
    except (ForgeError, ManifestError, FileExistsError, NotADirectoryError) as exc:
        print(f"forge error: {exc}", file=sys.stderr)
        return 2

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
