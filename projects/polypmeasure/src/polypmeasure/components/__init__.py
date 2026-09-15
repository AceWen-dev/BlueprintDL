"""PolypMeasure-owned plugin components.

Add explicit imports inside ``load_components`` as components are implemented.
Directory scanning is intentionally avoided so registration remains deterministic.
"""


def load_components():
    """Load project components without starting work or acquiring resources."""
    component_modules = ()
    return component_modules


__all__ = ["load_components"]
