from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import TerminationTermCfg


class DelayedIllegalContact(ManagerTermBase):
    """Terminate only after contact has persisted for a configured duration."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.threshold = float(cfg.params["threshold"])
        self.delay_s = float(cfg.params["delay_s"])
        self.contact_sensor: ContactSensor = env.scene.sensors[self.sensor_cfg.name]
        self.delay_steps = max(1, math.ceil(self.delay_s / env.step_dt))
        self.contact_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def __call__(self, env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg, delay_s: float) -> torch.Tensor:
        # Reset term state at the start of freshly reset episodes.
        reset_envs = env.episode_length_buf <= 1
        self.contact_counter[reset_envs] = 0

        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        contact_now = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self.sensor_cfg.body_ids], dim=-1), dim=1)[0] > self.threshold,
            dim=1,
        )

        self.contact_counter[contact_now] += 1
        self.contact_counter[~contact_now] = 0
        return self.contact_counter >= self.delay_steps
