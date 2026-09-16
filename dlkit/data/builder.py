"""Backward-compatible data builder imports.

New code should import these functions from :mod:`dlkit.builders`.
"""

from dlkit.builders.data import build_dataloaders, build_loader

__all__ = ["build_loader", "build_dataloaders"]
