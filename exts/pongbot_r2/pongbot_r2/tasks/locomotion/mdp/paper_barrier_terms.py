"""MDP terms for the R2 reproduction of barrier-based style rewards.

@version 0.0.6
@update 2026-07-13: Add progress-gated rough-terrain sampling and persistent calf-support suppression.
@update 2026-07-13: Read explicit R2 actuator limits and retain the latest applied action in policy history.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp import UniformVelocityCommand, reset_joints_by_offset, reset_root_state_uniform
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import CurriculumTermCfg, ObservationTermCfg, RewardTermCfg


FOOT_NAMES = ("FL_TIP", "FR_TIP", "RL_TIP", "RR_TIP")
THIGH_NAMES = ("FL_THIGH", "FR_THIGH", "RL_THIGH", "RR_THIGH")
FOOT_SCANNER_NAMES = (
    "fl_foot_scanner",
    "fr_foot_scanner",
    "rl_foot_scanner",
    "rr_foot_scanner",
)
PAPER_ROLE_ANCHOR = 0
PAPER_ROLE_FRONTIER = 1
PAPER_ROLE_PROBE = 2


def _as_env_ids(env: ManagerBasedRLEnv, env_ids: Sequence[int] | slice | None) -> torch.Tensor:
    """Return environment indices as a one-dimensional device tensor."""
    all_ids = torch.arange(env.num_envs, device=env.device)
    if env_ids is None:
        return all_ids
    if isinstance(env_ids, slice):
        return all_ids[env_ids]
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long).view(-1)


def _paper_hard_reset_mask(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
    """Select probe and mature-frontier environments for full reset disturbances."""
    roles = getattr(env, "_paper_curriculum_role", None)
    competence = getattr(env, "_paper_competence_level", None)
    if roles is None or competence is None:
        return torch.zeros(env_ids.numel(), dtype=torch.bool, device=env.device)
    return (roles[env_ids] == PAPER_ROLE_PROBE) | (
        (roles[env_ids] == PAPER_ROLE_FRONTIER) & (competence[env_ids] >= 7)
    )


class PaperVelocityCommand(UniformVelocityCommand):
    """Sample moderate discovery commands while retaining full-range probe environments."""

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).view(-1)
        if ids.numel() == 0:
            return

        roles = getattr(self._env, "_paper_curriculum_role", None)
        competence = getattr(self._env, "_paper_competence_level", None)
        if roles is None or competence is None:
            roles = torch.full((self.num_envs,), PAPER_ROLE_FRONTIER, device=self.device, dtype=torch.long)
            competence = torch.full((self.num_envs,), 3, device=self.device, dtype=torch.long)

        discovery_mask = roles[ids] != PAPER_ROLE_PROBE
        discovery_ids = ids[discovery_mask]
        if discovery_ids.numel() == 0:
            return

        # Frontier commands expand smoothly from a forward-biased discovery range
        # to the paper's full omnidirectional range as terrain competence grows.
        progress = torch.clamp((competence[discovery_ids].float() - 3.0) / 7.0, 0.0, 1.0)
        progress = torch.where(
            roles[discovery_ids] == PAPER_ROLE_ANCHOR,
            torch.zeros_like(progress),
            progress,
        )
        random = torch.rand(discovery_ids.numel(), 3, device=self.device)
        lower = torch.stack(
            (
                0.25 + progress * (-1.0 - 0.25),
                -0.20 + progress * (-1.0 + 0.20),
                -0.30 + progress * (-1.0 + 0.30),
            ),
            dim=1,
        )
        upper = torch.stack(
            (
                0.80 + progress * (1.50 - 0.80),
                0.20 + progress * (1.0 - 0.20),
                0.30 + progress * (1.0 - 0.30),
            ),
            dim=1,
        )
        self.vel_command_b[discovery_ids] = lower + random * (upper - lower)


def paper_reset_root_state_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | slice | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Use mild discovery resets while preserving full disturbance probes."""
    ids = _as_env_ids(env, env_ids)
    hard_mask = _paper_hard_reset_mask(env, ids)
    mild_ids = ids[~hard_mask]
    hard_ids = ids[hard_mask]
    pose_range = {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-math.pi, math.pi)}
    if mild_ids.numel():
        reset_root_state_uniform(
            env,
            mild_ids,
            pose_range=pose_range,
            velocity_range={
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.1, 0.1),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
            asset_cfg=asset_cfg,
        )
    if hard_ids.numel():
        reset_root_state_uniform(
            env,
            hard_ids,
            pose_range=pose_range,
            velocity_range={
                "x": (-1.0, 1.0),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.7, 0.7),
                "pitch": (-0.7, 0.7),
                "yaw": (-0.7, 0.7),
            },
            asset_cfg=asset_cfg,
        )


def paper_reset_joints_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | slice | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Use small joint-state perturbations except on robustness probes."""
    ids = _as_env_ids(env, env_ids)
    hard_mask = _paper_hard_reset_mask(env, ids)
    mild_ids = ids[~hard_mask]
    hard_ids = ids[hard_mask]
    if mild_ids.numel():
        reset_joints_by_offset(
            env,
            mild_ids,
            position_range=(-0.10, 0.10),
            velocity_range=(-0.5, 0.5),
            asset_cfg=asset_cfg,
        )
    if hard_ids.numel():
        reset_joints_by_offset(
            env,
            hard_ids,
            position_range=(-0.20, 0.20),
            velocity_range=(-2.5, 2.5),
            asset_cfg=asset_cfg,
        )


def _episode_phase_time(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return per-environment episode time, including manager construction."""
    episode_length = getattr(env, "episode_length_buf", None)
    if episode_length is None:
        episode_length = torch.zeros(env.num_envs, device=env.device)
    return episode_length * env.step_dt


def _ordered_body_ids(asset: Articulation, names: Sequence[str]) -> list[int]:
    """Resolve body names while preserving the requested leg order."""
    body_ids = []
    for name in names:
        ids, _ = asset.find_bodies(name, preserve_order=True)
        if len(ids) != 1:
            raise ValueError(f"Expected exactly one body named {name!r}, got ids={ids}")
        body_ids.append(int(ids[0]))
    return body_ids


def _ordered_joint_ids(asset: Articulation, suffix: str) -> list[int]:
    """Resolve one joint suffix for FL, FR, RL, RR."""
    joint_ids = []
    for leg in ("FL", "FR", "RL", "RR"):
        name = f"{leg}_{suffix}"
        ids, _ = asset.find_joints(name, preserve_order=True)
        if len(ids) != 1:
            raise ValueError(f"Expected exactly one joint named {name!r}, got ids={ids}")
        joint_ids.append(int(ids[0]))
    return joint_ids


def _foot_positions_b(asset: Articulation, foot_ids: Sequence[int]) -> torch.Tensor:
    """Return ordered foot positions in the root body frame."""
    relative_w = asset.data.body_pos_w[:, foot_ids] - asset.data.root_pos_w.unsqueeze(1)
    root_quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, len(foot_ids), -1)
    return math_utils.quat_apply_inverse(root_quat.reshape(-1, 4), relative_w.reshape(-1, 3)).reshape(
        asset.num_instances, len(foot_ids), 3
    )


def _local_terrain_heights(
    env: ManagerBasedRLEnv,
    feet_positions_w: torch.Tensor,
    radius: float = 0.05,
) -> torch.Tensor:
    """Return the highest finite terrain hit within a radius of each foot."""
    # The estimator target, standard reward, and barrier reward all request
    # this same four-foot terrain-relative quantity during one simulation
    # tick.  RayCaster hit tensors are unchanged until the next physics step,
    # so reuse the result within that tick instead of scanning all rays three
    # times.  The integer common_step_counter also invalidates the cache after
    # reset/physics advancement and avoids any tensor->CPU synchronization.
    step = getattr(env, "common_step_counter", None)
    cache = getattr(env, "_paper_terrain_height_cache", None)
    if step is not None and cache is not None and cache[0] == step:
        return cache[1]
    heights = torch.zeros_like(feet_positions_w[..., 2])
    for foot_idx, sensor_name in enumerate(FOOT_SCANNER_NAMES):
        scanner: RayCaster = env.scene.sensors[sensor_name]
        hits = scanner.data.ray_hits_w
        finite = torch.isfinite(hits[..., 2])
        distance = torch.linalg.norm(hits[..., :2] - feet_positions_w[:, foot_idx : foot_idx + 1, :2], dim=-1)
        valid = finite & (distance <= radius + 1.0e-6)
        candidate_z = torch.where(valid, hits[..., 2], torch.full_like(hits[..., 2], -torch.inf))
        local_max = torch.max(candidate_z, dim=1).values

        nearest_distance = torch.where(finite, distance, torch.inf)
        nearest_idx = torch.argmin(nearest_distance, dim=1)
        nearest_z = torch.gather(hits[..., 2], 1, nearest_idx.unsqueeze(1)).squeeze(1)
        nearest_z = torch.where(torch.isfinite(nearest_z), nearest_z, torch.zeros_like(nearest_z))
        heights[:, foot_idx] = torch.where(torch.isfinite(local_max), local_max, nearest_z)
    if step is not None:
        env._paper_terrain_height_cache = (step, heights)
    return heights


def _foot_contact_mask(
    env: ManagerBasedRLEnv,
    sensor: ContactSensor,
    foot_ids: Sequence[int],
) -> torch.Tensor:
    """Return stable binary foot contacts using the calibrated force threshold."""
    forces = torch.linalg.norm(sensor.data.net_forces_w[:, foot_ids], dim=-1)
    threshold = float(getattr(env, "_paper_contact_force_on", 1.0))
    return forces > threshold


def _effective_foot_radius(env: ManagerBasedRLEnv) -> float:
    """Return the R2 foot radius used by terrain-relative quantities."""
    return float(getattr(env, "_paper_foot_radius", 0.03))


def relaxed_log_barrier(z: torch.Tensor, delta: float | torch.Tensor) -> torch.Tensor:
    """Evaluate the relaxed logarithmic barrier from the paper."""
    delta_tensor = torch.as_tensor(delta, device=z.device, dtype=z.dtype)
    safe_z = torch.clamp(z, min=torch.finfo(z.dtype).tiny)
    quadratic = torch.log(delta_tensor) - 0.5 * (torch.square((z - 2.0 * delta_tensor) / delta_tensor) - 1.0)
    return torch.where(z > delta_tensor, torch.log(safe_z), quadratic)


def _bounded_barrier(
    value: torch.Tensor,
    lower: float | torch.Tensor,
    upper: float | torch.Tensor,
    delta: float | torch.Tensor,
) -> torch.Tensor:
    """Return lower and upper relaxed barriers for a bounded variable."""
    return relaxed_log_barrier(value - lower, delta) + relaxed_log_barrier(upper - value, delta)


class PaperProprioception(ManagerTermBase):
    """Construct the proprioceptive observation and sparse 20-ms histories."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.foot_ids = _ordered_body_ids(asset, FOOT_NAMES)
        self.error_history = torch.zeros(env.num_envs, 7, asset.num_joints, device=env.device)
        self.velocity_history = torch.zeros_like(self.error_history)
        self.desired_history = torch.zeros_like(self.error_history)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.error_history[env_ids] = 0.0
        self.velocity_history[env_ids] = 0.0
        self.desired_history[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        action_scale: float,
        gait_period: float,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        raw_action = env.action_manager.action
        desired = asset.data.default_joint_pos + action_scale * raw_action
        position_error = desired - asset.data.joint_pos

        self.error_history = torch.roll(self.error_history, shifts=1, dims=1)
        self.velocity_history = torch.roll(self.velocity_history, shifts=1, dims=1)
        self.desired_history = torch.roll(self.desired_history, shifts=1, dims=1)
        self.error_history[:, 0] = position_error
        self.velocity_history[:, 0] = asset.data.joint_vel
        self.desired_history[:, 0] = desired

        commands = env.command_manager.get_command("base_velocity")
        stand = torch.linalg.norm(commands, dim=1) < 0.2
        phase = torch.remainder(_episode_phase_time(env) / gait_period, 1.0)
        phase_sin = torch.sin(2.0 * math.pi * phase)
        phase_cos = torch.cos(2.0 * math.pi * phase)
        phase_sin = torch.where(stand, torch.zeros_like(phase_sin), phase_sin)
        phase_cos = torch.where(stand, torch.zeros_like(phase_cos), phase_cos)

        # action_manager.action is the action most recently applied to the simulation.
        # Keep it and the action before it, matching the paper's previous-action observation.
        desired_relative = self.desired_history[:, [0, 1]] - asset.data.default_joint_pos.unsqueeze(1)
        sparse_errors = self.error_history[:, [2, 4, 6]]
        sparse_velocities = self.velocity_history[:, [2, 4, 6]] * 0.05
        feet_b = _foot_positions_b(asset, self.foot_ids)

        return torch.cat(
            (
                asset.data.projected_gravity_b,
                asset.data.root_ang_vel_b * 0.25,
                asset.data.joint_pos - asset.data.default_joint_pos,
                asset.data.joint_vel * 0.05,
                desired_relative.flatten(1),
                sparse_errors.flatten(1),
                sparse_velocities.flatten(1),
                feet_b.flatten(1),
                phase_sin.unsqueeze(1),
                phase_cos.unsqueeze(1),
                stand.float().unsqueeze(1),
            ),
            dim=1,
        )


def paper_estimator_target(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return base velocity, foot clearance, and contact supervision."""
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_ids = _ordered_body_ids(asset, FOOT_NAMES)
    sensor_foot_ids = _ordered_body_ids_from_sensor(sensor, FOOT_NAMES)
    feet_w = asset.data.body_pos_w[:, foot_ids]
    terrain_z = _local_terrain_heights(env, feet_w)
    foot_height = feet_w[..., 2] - terrain_z - _effective_foot_radius(env)
    foot_height = torch.clamp(foot_height, -0.25, 0.75)
    contacts = _foot_contact_mask(env, sensor, sensor_foot_ids).float()
    return torch.cat((asset.data.root_lin_vel_b, foot_height, contacts), dim=1)


def _ordered_body_ids_from_sensor(sensor: ContactSensor, names: Sequence[str]) -> list[int]:
    """Resolve ordered body indices in a contact sensor."""
    ids = []
    for name in names:
        matching = [idx for idx, body_name in enumerate(sensor.body_names) if body_name == name]
        if len(matching) != 1:
            raise ValueError(f"Expected sensor body {name!r}, got matches={matching}")
        ids.append(matching[0])
    return ids


class PaperStandardReward(ManagerTermBase):
    """Standard regularization reward used beside the paper barrier stream."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.foot_ids = _ordered_body_ids(asset, FOOT_NAMES)
        self.thigh_ids = _ordered_body_ids(asset, THIGH_NAMES)
        self.sensor_foot_ids = _ordered_body_ids_from_sensor(sensor, FOOT_NAMES)
        self.body_sensor_ids = [idx for idx, name in enumerate(sensor.body_names) if name == "BODY"]
        self.thigh_sensor_ids = [idx for idx, name in enumerate(sensor.body_names) if "THIGH" in name]
        self.calf_sensor_ids = [idx for idx, name in enumerate(sensor.body_names) if "CALF" in name]
        self.illegal_sensor_ids = self.body_sensor_ids + self.thigh_sensor_ids
        self.previous_action = torch.zeros(env.num_envs, asset.num_joints, device=env.device)
        self.previous_previous_action = torch.zeros_like(self.previous_action)
        self.calf_contact_duration = torch.zeros(
            env.num_envs, len(self.calf_sensor_ids), device=env.device
        )
        self.nominal_foot_pos_b: torch.Tensor | None = None
        legs_actuator = asset.actuators.get("legs")
        if legs_actuator is None:
            raise RuntimeError("PaperStandardReward requires the R2 'legs' actuator group.")
        effort_limits = getattr(legs_actuator, "effort_limit", None)
        if effort_limits is None or effort_limits.shape != self.previous_action.shape:
            raise RuntimeError(
                "PaperStandardReward requires per-environment R2 actuator effort limits matching applied torque."
            )
        if not torch.isfinite(effort_limits).all() or not torch.all(effort_limits > 0.0):
            raise RuntimeError("PaperStandardReward received non-positive or non-finite joint effort limits.")
        self.effort_limits = effort_limits.detach().clone()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.previous_action[env_ids] = 0.0
        self.previous_previous_action[env_ids] = 0.0
        self.calf_contact_duration[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        action_scale: float,
        foot_position_weight: float,
        height_difference_weight: float,
        torque_normalized_weight: float,
        action_rate_weight: float = 2.5,
        action_acceleration_weight: float = 1.2,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        commands = env.command_manager.get_command("base_velocity")

        linear_error = commands[:, :2] - asset.data.root_lin_vel_b[:, :2]
        angular_error = commands[:, 2] - asset.data.root_ang_vel_b[:, 2]
        positive = 3.0 * torch.exp(-torch.sum(torch.square(linear_error), dim=1))
        positive += 3.0 * torch.exp(-1.5 * torch.square(angular_error))

        contacts = _foot_contact_mask(env, sensor, self.sensor_foot_ids).float()
        foot_velocity_xy = asset.data.body_lin_vel_w[:, self.foot_ids, :2]
        slip = -0.08 * torch.sum(contacts * torch.sum(torch.square(foot_velocity_xy), dim=-1), dim=1)
        torque_usage = asset.data.applied_torque / self.effort_limits
        torque = -abs(torque_normalized_weight) * torch.sum(torch.square(torque_usage), dim=1)

        current_action = env.action_manager.action
        desired = asset.data.default_joint_pos + action_scale * current_action
        previous_desired = asset.data.default_joint_pos + action_scale * self.previous_action
        previous_previous_desired = asset.data.default_joint_pos + action_scale * self.previous_previous_action
        smooth_one = -abs(action_rate_weight) * torch.sum(torch.square(desired - previous_desired), dim=1)
        smooth_two = -abs(action_acceleration_weight) * torch.sum(
            torch.square(desired - 2.0 * previous_desired + previous_previous_desired), dim=1
        )

        feet_b = _foot_positions_b(asset, self.foot_ids)
        calibrated_nominal = getattr(env, "_paper_nominal_foot_pos_b", None)
        if calibrated_nominal is None:
            if self.nominal_foot_pos_b is None:
                self.nominal_foot_pos_b = feet_b.detach().mean(dim=0)
            calibrated_nominal = self.nominal_foot_pos_b
        nominal = calibrated_nominal.to(device=env.device).unsqueeze(0)
        foot_position = -abs(foot_position_weight) * torch.sum(torch.square(feet_b - nominal), dim=(1, 2))

        feet_w = asset.data.body_pos_w[:, self.foot_ids]
        terrain_z = _local_terrain_heights(env, feet_w)
        thigh_z = asset.data.body_pos_w[:, self.thigh_ids, 2]
        front_height = torch.mean(thigh_z[:, :2] - terrain_z[:, :2], dim=1)
        hind_height = torch.mean(thigh_z[:, 2:] - terrain_z[:, 2:], dim=1)
        calibrated_delta = float(getattr(env, "_paper_nominal_height_difference", 0.0))
        height_difference = -abs(height_difference_weight) * torch.square(
            (front_height - hind_height) - calibrated_delta
        )

        negative = slip + torque + smooth_one + smooth_two + foot_position + height_difference
        standard = positive * torch.exp(torch.clamp(0.2 * negative, min=-60.0, max=0.0))
        illegal_force = torch.linalg.norm(sensor.data.net_forces_w[:, self.illegal_sensor_ids], dim=-1)
        illegal_contact = torch.any(illegal_force > 1.0, dim=1)
        body_force = torch.linalg.norm(sensor.data.net_forces_w[:, self.body_sensor_ids], dim=-1).amax(dim=1)
        thigh_force = torch.linalg.norm(sensor.data.net_forces_w[:, self.thigh_sensor_ids], dim=-1).amax(dim=1)
        calf_force = torch.linalg.norm(sensor.data.net_forces_w[:, self.calf_sensor_ids], dim=-1).amax(dim=1)
        calf_forces = torch.linalg.norm(sensor.data.net_forces_w[:, self.calf_sensor_ids], dim=-1)
        body_contact = body_force > 1.0
        thigh_contact = thigh_force > 1.0
        calf_contact = calf_force > 1.0
        loaded_calf_contact = calf_forces > 10.0
        self.calf_contact_duration = torch.where(
            loaded_calf_contact,
            self.calf_contact_duration + env.step_dt,
            torch.zeros_like(self.calf_contact_duration),
        )
        persistent_calf_support = self.calf_contact_duration > 0.12
        calf_support_penalty = -0.25 * persistent_calf_support.float().sum(dim=1)
        standard = standard - 10.0 * illegal_contact.float()
        standard = standard + calf_support_penalty

        self.previous_previous_action.copy_(self.previous_action)
        self.previous_action.copy_(current_action)
        env._paper_standard_metrics = {
            "Diagnostics/positive": positive.detach(),
            "Diagnostics/negative": negative.detach(),
            "Diagnostics/illegal_contact": illegal_contact.float().detach(),
            "StandardPenalty/foot_slip": slip.detach(),
            "StandardPenalty/joint_torque": torque.detach(),
            "StandardPenalty/action_rate": smooth_one.detach(),
            "StandardPenalty/action_acceleration": smooth_two.detach(),
            "StandardPenalty/foot_position": foot_position.detach(),
            "StandardPenalty/front_hind_height_difference": height_difference.detach(),
            "StandardPenalty/calf_support": calf_support_penalty.detach(),
            "Contact/body_rate": body_contact.float().detach(),
            "Contact/thigh_rate": thigh_contact.float().detach(),
            "Contact/calf_rate": calf_contact.float().detach(),
            "Contact/calf_persistent_rate": persistent_calf_support.float().mean(dim=1).detach(),
            "Contact/calf_max_duration": self.calf_contact_duration.amax(dim=1).detach(),
            "Contact/foot_rate": contacts.mean(dim=1).detach(),
            "Contact/body_force_max": body_force.detach(),
            "Contact/thigh_force_max": thigh_force.detach(),
            "Contact/calf_force_max": calf_force.detach(),
            "Motion/applied_torque_rms": torch.sqrt(
                torch.mean(torch.square(asset.data.applied_torque), dim=1)
            ).detach(),
            "Motion/torque_usage_rms": torch.sqrt(torch.mean(torch.square(torque_usage), dim=1)).detach(),
            "Motion/effort_limit_min": torch.amin(self.effort_limits, dim=1).detach(),
            "Motion/effort_limit_max": torch.amax(self.effort_limits, dim=1).detach(),
            "Motion/joint_velocity_rms": torch.sqrt(torch.mean(torch.square(asset.data.joint_vel), dim=1)).detach(),
            "Motion/action_rms": torch.sqrt(torch.mean(torch.square(current_action), dim=1)).detach(),
            "Motion/action_rate_rms": torch.sqrt(
                torch.mean(torch.square(current_action - self.previous_previous_action), dim=1)
            ).detach(),
        }
        return standard


class PaperBarrierReward(ManagerTermBase):
    """Relaxed barrier style reward for quadruped rough-terrain locomotion."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.foot_ids = _ordered_body_ids(asset, FOOT_NAMES)
        self.thigh_ids = _ordered_body_ids(asset, THIGH_NAMES)
        self.sensor_foot_ids = _ordered_body_ids_from_sensor(sensor, FOOT_NAMES)
        self.hr_ids = _ordered_joint_ids(asset, "HR_JOINT")
        self.hp_ids = _ordered_joint_ids(asset, "HP_JOINT")
        self.kn_ids = _ordered_joint_ids(asset, "KN_JOINT")
        self.phase_offsets = torch.tensor((0.0, 0.5, 0.5, 0.0), device=env.device)
        self.asset = asset
        self.env = env
        self.episode_moving_error_sum = torch.zeros(env.num_envs, device=env.device)
        self.episode_moving_steps = torch.zeros(env.num_envs, device=env.device)
        self.episode_commanded_distance = torch.zeros(env.num_envs, device=env.device)
        self.episode_projected_progress = torch.zeros(env.num_envs, device=env.device)
        env._paper_progress_reward = torch.zeros(env.num_envs, device=env.device)
        self.previous_root_pos_w = asset.data.root_pos_w[:, :2].clone()
        env._paper_episode_moving_error_sum = self.episode_moving_error_sum
        env._paper_episode_moving_steps = self.episode_moving_steps
        env._paper_episode_commanded_distance = self.episode_commanded_distance
        env._paper_episode_projected_progress = self.episode_projected_progress

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.episode_moving_error_sum[env_ids] = 0.0
        self.episode_moving_steps[env_ids] = 0.0
        self.episode_commanded_distance[env_ids] = 0.0
        self.episode_projected_progress[env_ids] = 0.0
        self.env._paper_progress_reward[env_ids] = 0.0
        # A reset can change foot positions without advancing the global
        # common-step counter; never reuse the previous episode's ray heights.
        self.env._paper_terrain_height_cache = None
        self.previous_root_pos_w[env_ids] = self.asset.data.root_pos_w[env_ids, :2]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        gait_period: float,
        alpha: float,
        progress_weight: float = 0.0,
        underspeed_weight: float = 0.0,
        moving_command_threshold: float = 0.20,
        progress_clip: float = 1.50,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        commands = env.command_manager.get_command("base_velocity")
        stand = torch.linalg.norm(commands, dim=1) < 0.2
        scale = float(getattr(env, "_paper_length_scale", 1.0))

        phase = _episode_phase_time(env) / gait_period
        gait = torch.sin(2.0 * math.pi * (phase.unsqueeze(1) + self.phase_offsets.unsqueeze(0)))
        contacts = _foot_contact_mask(env, sensor, self.sensor_foot_ids)
        gait_variable = torch.where(contacts, gait, -gait)
        gait_variable = torch.where(
            stand.unsqueeze(1), torch.where(contacts, torch.ones_like(gait), -torch.ones_like(gait)), gait_variable
        )
        gait_reward = _bounded_barrier(gait_variable, -0.6, 2.0, 0.1).sum(dim=1)

        feet_w = asset.data.body_pos_w[:, self.foot_ids]
        terrain_z = _local_terrain_heights(env, feet_w)
        foot_height = feet_w[..., 2] - terrain_z - _effective_foot_radius(env)
        desired_clearance = 0.15 * scale
        clearance_variable = foot_height - desired_clearance
        swing_enforced = gait <= -0.6
        clearance_variable = torch.where(swing_enforced & ~stand.unsqueeze(1), clearance_variable, 0.0)
        clearance_reward = _bounded_barrier(
            clearance_variable,
            -0.08 * scale,
            1.0 * scale,
            0.01 * scale,
        ).sum(dim=1)

        relative_joint_position = asset.data.joint_pos - asset.data.default_joint_pos
        joint_reward = _bounded_barrier(
            relative_joint_position[:, self.hr_ids], -math.pi / 6.0, math.pi / 6.0, 0.08
        ).sum(dim=1)
        joint_reward += _bounded_barrier(
            relative_joint_position[:, self.hp_ids], -math.pi / 4.0, math.pi / 4.0, 0.08
        ).sum(dim=1)
        joint_reward += _bounded_barrier(
            relative_joint_position[:, self.kn_ids], -2.0 * math.pi / 5.0, math.pi / 4.0, 0.08
        ).sum(dim=1)

        thigh_z = asset.data.body_pos_w[:, self.thigh_ids, 2]
        front_height = torch.mean(thigh_z[:, :2] - terrain_z[:, :2], dim=1)
        hind_height = torch.mean(thigh_z[:, 2:] - terrain_z[:, 2:], dim=1)
        centers = getattr(env, "_paper_body_height_centers", (0.59 * scale, 0.59 * scale))
        half_width = 0.07 * scale
        height_delta = 0.04 * scale
        front_reward = _bounded_barrier(front_height, centers[0] - half_width, centers[0] + half_width, height_delta)
        hind_reward = _bounded_barrier(hind_height, centers[1] - half_width, centers[1] + half_width, height_delta)

        velocity_error = commands - torch.stack(
            (asset.data.root_lin_vel_b[:, 0], asset.data.root_lin_vel_b[:, 1], asset.data.root_ang_vel_b[:, 2]),
            dim=1,
        )
        traversal_command = torch.linalg.norm(commands[:, :2], dim=1) >= 0.2
        max_xy_error = torch.amax(torch.abs(velocity_error[:, :2]), dim=1)
        command_b = torch.cat((commands[:, :2], torch.zeros_like(commands[:, :1])), dim=1)
        command_w = math_utils.quat_apply_yaw(asset.data.root_quat_w, command_b)[:, :2]
        command_direction_w = command_w / torch.clamp(
            torch.linalg.norm(command_w, dim=1, keepdim=True), min=1.0e-6
        )
        root_displacement_w = asset.data.root_pos_w[:, :2] - self.previous_root_pos_w
        projected_progress = torch.sum(root_displacement_w * command_direction_w, dim=1)
        command_speed = torch.linalg.norm(commands[:, :2], dim=1)
        projected_speed = projected_progress / max(float(env.step_dt), 1.0e-6)
        moving_for_progress = command_speed >= float(moving_command_threshold)
        progress_reward = float(progress_weight) * torch.clamp(
            projected_speed, min=-0.5, max=float(progress_clip)
        ) * moving_for_progress.float()
        underspeed_error = torch.clamp(command_speed - projected_speed, min=0.0, max=1.0)
        underspeed_penalty = -float(underspeed_weight) * torch.square(underspeed_error) * moving_for_progress.float()
        self.episode_moving_error_sum += torch.where(
            traversal_command, max_xy_error, torch.zeros_like(max_xy_error)
        )
        self.episode_moving_steps += traversal_command.float()
        self.episode_commanded_distance += torch.where(
            traversal_command,
            torch.linalg.norm(commands[:, :2], dim=1) * env.step_dt,
            torch.zeros_like(max_xy_error),
        )
        self.episode_projected_progress += torch.where(
            traversal_command, projected_progress, torch.zeros_like(projected_progress)
        )
        self.previous_root_pos_w.copy_(asset.data.root_pos_w[:, :2])
        velocity_reward = _bounded_barrier(velocity_error, -0.4, 0.4, 0.2).sum(dim=1)
        base_motion = torch.stack(
            (asset.data.root_ang_vel_b[:, 0], asset.data.root_ang_vel_b[:, 1], asset.data.root_lin_vel_b[:, 2]),
            dim=1,
        )
        base_lower = torch.tensor((-0.3, -0.3, -0.2 * scale), device=env.device)
        base_upper = torch.tensor((0.3, 0.3, 0.2 * scale), device=env.device)
        base_delta = torch.tensor((0.3, 0.3, 0.2 * scale), device=env.device)
        base_reward = _bounded_barrier(base_motion, base_lower, base_upper, base_delta).sum(dim=1)
        joint_velocity_reward = _bounded_barrier(asset.data.joint_vel, -8.0, 8.0, 2.0).sum(dim=1)

        joint_position_violation = torch.cat(
            (
                (relative_joint_position[:, self.hr_ids] < -math.pi / 6.0)
                | (relative_joint_position[:, self.hr_ids] > math.pi / 6.0),
                (relative_joint_position[:, self.hp_ids] < -math.pi / 4.0)
                | (relative_joint_position[:, self.hp_ids] > math.pi / 4.0),
                (relative_joint_position[:, self.kn_ids] < -2.0 * math.pi / 5.0)
                | (relative_joint_position[:, self.kn_ids] > math.pi / 4.0),
            ),
            dim=1,
        )
        front_height_violation = (front_height < centers[0] - half_width) | (front_height > centers[0] + half_width)
        hind_height_violation = (hind_height < centers[1] - half_width) | (hind_height > centers[1] + half_width)
        base_motion_violation = (base_motion < base_lower) | (base_motion > base_upper)
        active_swing = swing_enforced & ~stand.unsqueeze(1)
        swing_count = torch.clamp(active_swing.float().sum(dim=1), min=1.0)
        mean_swing_clearance = torch.sum(torch.where(active_swing, foot_height, 0.0), dim=1) / swing_count

        barrier = alpha * (
            gait_reward
            + clearance_reward
            + joint_reward
            + front_reward
            + hind_reward
            + velocity_reward
            + base_reward
            + joint_velocity_reward
        )
        # Keep the barrier stream a constraint stream; progress shaping belongs
        # to the standard locomotion objective and is added separately below.
        # The caller combines the two reward terms, so adding it here would
        # incorrectly scale it by alpha.
        barrier = torch.nan_to_num(barrier, nan=-1.0e4, posinf=1.0e4, neginf=-1.0e4)

        env._paper_barrier_metrics = {
            "Diagnostics/gait_violation": torch.mean((gait_variable < -0.6).float(), dim=1).detach(),
            "Diagnostics/clearance_violation": torch.mean(
                ((clearance_variable < -0.08 * scale) & swing_enforced & ~stand.unsqueeze(1)).float(), dim=1
            ).detach(),
            "Diagnostics/velocity_violation": torch.mean((torch.abs(velocity_error) > 0.4).float(), dim=1).detach(),
            "Diagnostics/joint_velocity_violation": torch.mean(
                (torch.abs(asset.data.joint_vel) > 8.0).float(), dim=1
            ).detach(),
            "ConstraintViolation/joint_position": joint_position_violation.float().mean(dim=1).detach(),
            "ConstraintViolation/front_body_height": front_height_violation.float().detach(),
            "ConstraintViolation/hind_body_height": hind_height_violation.float().detach(),
            "ConstraintViolation/base_motion": base_motion_violation.float().mean(dim=1).detach(),
            "BarrierTerm/gait": (alpha * gait_reward).detach(),
            "BarrierTerm/foot_clearance": (alpha * clearance_reward).detach(),
            "BarrierTerm/joint_position": (alpha * joint_reward).detach(),
            "BarrierTerm/front_body_height": (alpha * front_reward).detach(),
            "BarrierTerm/hind_body_height": (alpha * hind_reward).detach(),
            "BarrierTerm/target_velocity": (alpha * velocity_reward).detach(),
            "BarrierTerm/base_motion": (alpha * base_reward).detach(),
            "BarrierTerm/joint_velocity": (alpha * joint_velocity_reward).detach(),
            "Tracking/lin_vel_xy_error": torch.linalg.norm(velocity_error[:, :2], dim=1).detach(),
            "Tracking/lin_vel_xy_max_abs_error": torch.amax(
                torch.abs(velocity_error[:, :2]), dim=1
            ).detach(),
            "Tracking/yaw_rate_error": torch.abs(velocity_error[:, 2]).detach(),
            "Tracking/command_xy_speed": torch.linalg.norm(commands[:, :2], dim=1).detach(),
            "Tracking/actual_xy_speed": torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1).detach(),
            "Tracking/projected_speed": projected_speed.detach(),
            "Tracking/progress_reward": progress_reward.detach(),
            "Tracking/underspeed_penalty": underspeed_penalty.detach(),
            "Tracking/is_moving_command": (~stand).float().detach(),
            "Tracking/is_traversal_command": traversal_command.float().detach(),
            "Foot/mean_swing_clearance": mean_swing_clearance.detach(),
        }
        env._paper_progress_reward = progress_reward + underspeed_penalty
        return barrier


class PaperTerrainDifficultyCurriculum(ManagerTermBase):
    """Maintain anchor/frontier/probe terrain roles using moving traversal success."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        env_ids = torch.arange(env.num_envs, device=env.device)
        role_bucket = torch.remainder(env_ids, 10)
        self.roles = torch.where(
            role_bucket < 3,
            torch.full_like(role_bucket, PAPER_ROLE_ANCHOR),
            torch.where(
                role_bucket < 8,
                torch.full_like(role_bucket, PAPER_ROLE_FRONTIER),
                torch.full_like(role_bucket, PAPER_ROLE_PROBE),
            ),
        )
        self.competence = torch.full((env.num_envs,), 3, device=env.device, dtype=torch.long)
        self.last_progress_ratio = torch.zeros(env.num_envs, device=env.device)
        self.last_mean_error = torch.zeros(env.num_envs, device=env.device)
        self.last_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._paper_curriculum_role = self.roles
        env._paper_competence_level = self.competence
        env._paper_curriculum_term = self

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        anchor_max_level: int,
        probe_min_level: int,
        success_progress_ratio: float,
        failure_progress_ratio: float,
        success_mean_error: float,
        failure_mean_error: float,
    ) -> torch.Tensor:
        terrain = env.scene.terrain
        if terrain.terrain_origins is None:
            return torch.zeros((), device=env.device)

        env_ids_tensor = _as_env_ids(env, env_ids)
        if env_ids_tensor.numel() == 0:
            return torch.zeros((), device=env.device)

        if getattr(env, "_paper_calibrating", False):
            sampled_levels = torch.zeros_like(env_ids_tensor)
            terrain.terrain_levels[env_ids_tensor] = sampled_levels
            terrain.env_origins[env_ids_tensor] = terrain.terrain_origins[
                sampled_levels, terrain.terrain_types[env_ids_tensor]
            ]
            return torch.zeros((), device=env.device)

        max_level = terrain.max_terrain_level - 1
        moving_steps = getattr(env, "_paper_episode_moving_steps", None)
        error_sum = getattr(env, "_paper_episode_moving_error_sum", None)
        commanded_distance = getattr(env, "_paper_episode_commanded_distance", None)
        projected_progress = getattr(env, "_paper_episode_projected_progress", None)
        if (
            moving_steps is not None
            and error_sum is not None
            and commanded_distance is not None
            and projected_progress is not None
        ):
            ids = env_ids_tensor
            expected_distance = torch.clamp(commanded_distance[ids], min=0.1)
            progress_ratio = projected_progress[ids] / expected_distance
            mean_error = error_sum[ids] / torch.clamp(moving_steps[ids], min=1.0)
            traversal_episode = moving_steps[ids] >= 0.5 * torch.clamp(
                env.episode_length_buf[ids].float(), min=1.0
            )
            timeout = env.termination_manager.time_outs[ids]
            success = (
                traversal_episode
                & timeout
                & (progress_ratio >= success_progress_ratio)
                & (mean_error <= success_mean_error)
            )
            failure = traversal_episode & (
                (~timeout)
                | (progress_ratio < failure_progress_ratio)
                | (mean_error > failure_mean_error)
            )
            frontier = self.roles[ids] == PAPER_ROLE_FRONTIER
            delta = success.long() - failure.long()
            self.competence[ids] = torch.where(
                frontier,
                torch.clamp(self.competence[ids] + delta, min=1, max=max_level),
                self.competence[ids],
            )
            self.last_progress_ratio[ids] = progress_ratio
            self.last_mean_error[ids] = mean_error
            self.last_success[ids] = success

        roles = self.roles[env_ids_tensor]
        anchor = roles == PAPER_ROLE_ANCHOR
        frontier = roles == PAPER_ROLE_FRONTIER
        num_ids = env_ids_tensor.numel()
        anchor_levels = torch.randint(
            0,
            min(anchor_max_level, max_level) + 1,
            (num_ids,),
            device=env.device,
        )
        frontier_levels = torch.clamp(
            self.competence[env_ids_tensor]
            + torch.randint(-1, 2, (num_ids,), device=env.device),
            min=0,
            max=max_level,
        )
        probe_lower = torch.clamp(
            self.competence[env_ids_tensor] + 2,
            min=min(probe_min_level, max_level),
            max=max_level,
        )
        probe_width = (max_level - probe_lower + 1).float()
        probe_levels = probe_lower + torch.floor(torch.rand_like(probe_width) * probe_width).long()
        sampled_levels = torch.where(
            anchor,
            anchor_levels,
            torch.where(frontier, frontier_levels, probe_levels),
        )

        terrain.terrain_levels[env_ids_tensor] = sampled_levels
        terrain.env_origins[env_ids_tensor] = terrain.terrain_origins[
            sampled_levels, terrain.terrain_types[env_ids_tensor]
        ]
        return terrain.terrain_levels.float().mean()

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return curriculum state for checkpoint continuation."""
        return {
            "competence": self.competence.detach().cpu().clone(),
            "last_progress_ratio": self.last_progress_ratio.detach().cpu().clone(),
            "last_mean_error": self.last_mean_error.detach().cpu().clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore curriculum state when shapes match the current environment count."""
        for name in ("competence", "last_progress_ratio", "last_mean_error"):
            value = state.get(name)
            target = getattr(self, name)
            if value is not None and value.shape == target.shape:
                target.copy_(value.to(device=target.device, dtype=target.dtype))


__all__ = [
    "FOOT_NAMES",
    "FOOT_SCANNER_NAMES",
    "PaperBarrierReward",
    "PaperProprioception",
    "PaperStandardReward",
    "PaperTerrainDifficultyCurriculum",
    "PaperVelocityCommand",
    "paper_reset_joints_curriculum",
    "paper_reset_root_state_curriculum",
    "paper_estimator_target",
    "relaxed_log_barrier",
]
