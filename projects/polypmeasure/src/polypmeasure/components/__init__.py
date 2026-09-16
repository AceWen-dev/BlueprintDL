"""Explicit imports for PolypMeasure-owned plugin components."""


def load_components():
    """Load project components without starting work or acquiring resources."""
    from . import models

    return (models.__name__,)


__all__ = ["load_components"]
