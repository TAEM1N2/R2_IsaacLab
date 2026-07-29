"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import UniformGaitCommandCfg


class GaitCommand(CommandTerm):
    """Command generator that generates gait frequency, phase offset and contact duration."""

    cfg: UniformGaitCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformGaitCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # create buffers to store the command
        # command format: [frequency, phase offset, contact duration]
        self.gait_command = torch.zeros(self.num_envs, 4, device=self.device)
        # self.gait_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.profile_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._profile_commands = None
        self._profile_probabilities = None
        self._profile_jitter = None
        self._init_profiles()
        # create metrics dictionary for logging
        self.metrics = {}

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "GaitCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The gait command. Shape is (num_envs, 3)."""
        return self.gait_command

    def _update_metrics(self):
        """Update the metrics based on the current state.

        In this implementation, we don't track any specific metrics.
        """
        pass

    def _init_profiles(self):
        """Initialize optional profile-based gait sampling."""
        if not self.cfg.profiles:
            return

        profile_commands = torch.tensor(self.cfg.profiles, dtype=torch.float, device=self.device)
        if profile_commands.ndim != 2 or profile_commands.shape[1] != 4:
            raise ValueError("'profiles' must be a tuple of (frequency, offset, duration, swing_height) tuples.")

        if self.cfg.profile_probabilities is None:
            profile_probabilities = torch.ones(profile_commands.shape[0], dtype=torch.float, device=self.device)
        else:
            profile_probabilities = torch.tensor(
                self.cfg.profile_probabilities, dtype=torch.float, device=self.device
            )
            if profile_probabilities.shape[0] != profile_commands.shape[0]:
                raise ValueError("'profile_probabilities' must match the number of gait profiles.")
            if torch.any(profile_probabilities < 0.0):
                raise ValueError("'profile_probabilities' cannot contain negative values.")

        probability_sum = torch.sum(profile_probabilities)
        if probability_sum <= 0.0:
            raise ValueError("'profile_probabilities' must have a positive sum.")

        profile_jitter = torch.tensor(self.cfg.jitter, dtype=torch.float, device=self.device)
        if profile_jitter.shape[0] != 4:
            raise ValueError("'jitter' must contain four values.")
        if torch.any(profile_jitter < 0.0):
            raise ValueError("'jitter' cannot contain negative values.")

        self._profile_commands = profile_commands
        self._profile_probabilities = profile_probabilities / probability_sum
        self._profile_jitter = profile_jitter

    def _resample_command(self, env_ids):
        """Resample the gait command for specified environments."""
        if len(env_ids) == 0:
            return

        if self._profile_commands is not None:
            num_resampled = len(env_ids)
            profile_ids = torch.multinomial(self._profile_probabilities, num_resampled, replacement=True)
            commands = self._profile_commands[profile_ids].clone()

            if torch.any(self._profile_jitter > 0.0):
                noise = (2.0 * torch.rand(num_resampled, 4, device=self.device) - 1.0) * self._profile_jitter
                commands += noise

            commands[:, 0].clamp_(min=1.0e-3)
            commands[:, 1].clamp_(0.0, 1.0)
            commands[:, 2].clamp_(1.0e-3, 1.0 - 1.0e-3)
            commands[:, 3].clamp_(min=0.0)

            self.gait_command[env_ids, :] = commands
            self.profile_ids[env_ids] = profile_ids
            return

        # sample gait parameters
        r = torch.empty(len(env_ids), device=self.device)
        # -- frequency
        self.gait_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.frequencies)
        # -- phase offset
        self.gait_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.offsets)
        # -- contact duration
        self.gait_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.durations)
        # -- swing height
        self.gait_command[env_ids, 3] = r.uniform_(*self.cfg.ranges.swing_height)

    def _update_command(self):
        """Update the command. No additional processing needed in this implementation."""
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.

        In this implementation, we don't provide any debug visualization.
        """
        pass

    def _debug_vis_callback(self, event):
        """Callback for debug visualization.

        In this implementation, we don't provide any debug visualization.
        """
        pass
