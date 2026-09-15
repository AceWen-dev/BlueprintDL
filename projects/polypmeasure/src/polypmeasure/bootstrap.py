"""Explicit, lightweight registration entry point for PolypMeasure."""


def register():
    """Import the project's declared component modules and return their names."""
    from polypmeasure.components import load_components

    return load_components()
