"""List registered components and their provenance records."""

from __future__ import annotations

import argparse
import importlib
import json
from typing import Any

from dlkit.registry import iter_component_records


def _load_registered_modules() -> None:
    for module_name in (
        "dlkit.data",
        "dlkit.models",
        "dlkit.metrics",
        "dlkit.engine.optimizers",
        "dlkit.engine.schedulers",
        "dlkit.engine.recipes",
        "dlkit.viz",
    ):
        importlib.import_module(module_name)


def list_components(
    *,
    provider: str | None = None,
    registry: str | None = None,
    as_json: bool = False,
) -> list[dict[str, Any]]:
    _load_registered_modules()
    records = sorted(
        iter_component_records(provider=provider, registry=registry),
        key=lambda item: (
            item.registry.casefold(),
            item.name.casefold(),
            item.module.casefold(),
        ),
    )
    return [
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List registered components and their providers",
    )
    parser.add_argument("--provider", help="filter by provider")
    parser.add_argument("--registry", help="filter by registry name")
    parser.add_argument("--json", action="store_true", help="print JSON output")
    args = parser.parse_args()

    components = list_components(
        provider=args.provider,
        registry=args.registry,
        as_json=args.json,
    )
    if args.json:
        print(json.dumps({"components": components}, indent=2, ensure_ascii=False))
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


__all__ = ["list_components"]


if __name__ == "__main__":
    raise SystemExit(main())
