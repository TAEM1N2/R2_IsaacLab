import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .gait_command import GaitCommand  # Import the GaitCommand class


@configclass
class UniformGaitCommandCfg(CommandTermCfg):
    """Configuration for the gait command generator."""

    class_type: type = GaitCommand  # Specify the class type for dynamic instantiation

    @configclass
    class Ranges:
        """Uniform distribution ranges for the gait parameters."""

        frequencies: tuple[float, float] = MISSING
        """Range for gait frequencies [Hz]."""
        offsets: tuple[float, float] = MISSING
        """Range for phase offsets [0-1]."""
        durations: tuple[float, float] = MISSING
        """Range for contact durations [0-1]."""
        swing_height: tuple[float, float] = MISSING
        """Range for contact durations [0-1]."""

    ranges: Ranges = MISSING
    """Distribution ranges for the gait parameters."""

    profiles: tuple[tuple[float, float, float, float], ...] | None = None
    """Optional gait command anchors as ``(frequency, offset, duration, swing_height)``."""

    profile_probabilities: tuple[float, ...] | None = None
    """Sampling probability for each profile. Uniform sampling is used when unset."""

    jitter: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    """Uniform jitter half-width around each sampled profile."""

    resampling_time_range: tuple[float, float] = MISSING
    """Time interval for resampling the gait (in seconds)."""
