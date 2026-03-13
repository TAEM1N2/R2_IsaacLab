"""Locomotion environments for legged robots."""

try:
    from . import robots
except ModuleNotFoundError:
    # Some locomotion configs still depend on optional external assets/packages.
    # Keep the main pongbot_r2 package importable even when those extras are absent.
    robots = None
