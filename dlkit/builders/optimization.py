"""Role-specific builders for optimizers and learning-rate schedulers."""

from __future__ import annotations

from collections.abc import Iterable

from dlkit.registry import OPTIMIZERS, SCHEDULERS

from .component import build_from_cfg


def _component_params(cfg, *, reserved):
    if not isinstance(cfg, dict) or "type" not in cfg:
        raise ValueError('component config must be a dict with a "type" key')
    raw_params = cfg.get("params", {})
    if raw_params is None:
        raw_params = {}
    if not isinstance(raw_params, dict):
        raise TypeError("component params must be a mapping")
    params = build_from_cfg(dict(raw_params))
    collisions = sorted(set(params) & set(reserved))
    if collisions:
        raise ValueError(
            "configured arguments collide with injected dependencies: %s"
            % collisions
        )
    return params


def build_optimizer(cfg, *, parameters: Iterable):
    """Build an optimizer selected by config and inject model parameters."""

    params = _component_params(cfg, reserved=("params",))
    optimizer_cls = OPTIMIZERS.get(cfg["type"])
    return optimizer_cls(parameters, **params)


def build_scheduler(cfg, *, optimizer, total_steps=None):
    """Build a scheduler and inject its optimizer and derived schedule size."""

    if not cfg:
        return None
    params = _component_params(cfg, reserved=("optimizer",))
    scheduler_cls = SCHEDULERS.get(cfg["type"])
    if getattr(scheduler_cls, "requires_total_steps", False):
        if "max_iters" not in params:
            if total_steps is None:
                raise ValueError(
                    "%s requires total_steps to be injected" % cfg["type"]
                )
            params["max_iters"] = total_steps
    return scheduler_cls(optimizer, **params)
