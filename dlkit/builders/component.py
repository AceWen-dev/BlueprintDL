"""Generic construction from serializable ``type``/``params`` configuration."""

from __future__ import annotations

from typing import Any

from dlkit.registry import iter_registries


def resolve_component(type_name: str):
    """Resolve one registered implementation by qualified or unique name."""

    if isinstance(type_name, str) and "." in type_name:
        registry_name, component_name = type_name.split(".", 1)
        for registry in iter_registries():
            if registry.name == registry_name:
                return registry.get(component_name)
        raise KeyError(
            "Unknown registry name %r in qualified type %r"
            % (registry_name, type_name)
        )

    matches = [registry for registry in iter_registries() if type_name in registry]
    if not matches:
        available = sorted(
            {key for registry in iter_registries() for key in registry.keys()}
        )
        raise KeyError(
            "%r is not registered in any registry. Available: %s"
            % (type_name, available)
        )
    if len(matches) > 1:
        names = ", ".join(
            "%s.%s" % (registry.name, type_name) for registry in matches
        )
        raise KeyError(
            "%r is ambiguous, found in %d registries. Use a qualified name: %s"
            % (type_name, len(matches), names)
        )
    return matches[0].get(type_name)


def build_from_cfg(cfg: Any):
    """Build registered components without mutating caller-owned configuration."""

    if isinstance(cfg, (list, tuple)):
        return [build_from_cfg(item) for item in cfg]
    if not isinstance(cfg, dict):
        return cfg
    if "type" not in cfg:
        return {key: build_from_cfg(value) for key, value in cfg.items()}

    item = resolve_component(cfg["type"])
    raw_params = cfg.get("params", {})
    if raw_params is None:
        raw_params = {}
    if not isinstance(raw_params, dict):
        raise TypeError(
            "params for component %r must be a mapping, got %s"
            % (cfg["type"], type(raw_params).__name__)
        )
    params = dict(raw_params)
    if not getattr(item, "_manual_build", False):
        params = build_from_cfg(params)
    return item(**params)
