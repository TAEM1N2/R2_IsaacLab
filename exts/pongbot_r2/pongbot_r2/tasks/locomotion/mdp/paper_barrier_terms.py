"""MDP terms for the R2 reproduction of barrier-based style rewards.

@version 0.0.4
@update 2026-07-13: Normalize standard torque regularization by the R2 actuator effort limits.
@update 2026-07-13: Expose rollout diagnostics for every standard and barrier component.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
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

        desired_relative = self.desired_history[:, [1, 2]] - asset.data.default_joint_pos.unsqueeze(1)
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
        self.nominal_foot_pos_b: torch.Tensor | None = None
        effort_limits = getattr(asset.data, "joint_effort_limits", None)
        if effort_limits is None or effort_limits.shape != self.previous_action.shape:
            raise RuntimeError(
                "PaperStandardReward requires per-environment joint_effort_limits matching applied torque."
            )
        if not torch.isfinite(effort_limits).all() or not torch.all(effort_limits > 0.0):
            raise RuntimeError("PaperStandardReward received non-positive or non-finite joint effort limits.")
        self.effort_limits = effort_limits.detach().clone()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.previous_action[env_ids] = 0.0
        self.previous_previous_action[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        action_scale: float,
        foot_position_weight: float,
        height_difference_weight: float,
        torque_normalized_weight: float,
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
        smooth_one = -2.5 * torch.sum(torch.square(desired - previous_desired), dim=1)
        smooth_two = -1.2 * torch.sum(
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
        body_contact = body_force > 1.0
        thigh_contact = thigh_force > 1.0
        calf_contact = calf_force > 1.0
        standard = standard - 10.0 * illegal_contact.float()

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
            "Contact/body_rate": body_contact.float().detach(),
            "Contact/thigh_rate": thigh_contact.float().detach(),
            "Contact/calf_rate": calf_contact.float().detach(),
            "Contact/foot_rate": contacts.mean(dim=1).detach(),
            "Contact/body_force_max": body_force.detach(),
            "Contact/thigh_force_max": thigh_force.detach(),
            "Contact/calf_force_max": calf_force.detach(),
            "Motion/applied_torque_rms": torch.sqrt(torch.mean(torch.square(asset.data.applied_torque), dim=1)).detach(),
            "Motion/torque_usage_rms": torch.sqrt(torch.mean(torch.square(torque_usage), dim=1)).detach(),
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

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        gait_period: float,
        alpha: float,
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
            "Tracking/is_moving_command": (~stand).float().detach(),
            "Foot/mean_swing_clearance": mean_swing_clearance.detach(),
        }
        return barrier


class PaperTerrainDifficultyCurriculum(ManagerTermBase):
    """Expand the globally available terrain rows without success-based demotion."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        initial_difficulty: float,
        ramp_start_iteration: int,
        ramp_end_iteration: int,
    ) -> dict[str, float]:
        terrain = env.scene.terrain
        if terrain.terrain_origins is None:
            return {"difficulty": 0.0, "max_level": 0.0}

        env_ids_tensor = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).view(-1)
        if env_ids_tensor.numel() == 0:
            return {"difficulty": 0.0, "max_level": 0.0}

        iteration = int(getattr(env, "_paper_learning_iteration", 0))
        if iteration <= ramp_start_iteration:
            difficulty = initial_difficulty
        elif iteration >= ramp_end_iteration:
            difficulty = 1.0
        else:
            fraction = (iteration - ramp_start_iteration) / (ramp_end_iteration - ramp_start_iteration)
            difficulty = initial_difficulty + fraction * (1.0 - initial_difficulty)

        max_level = max(0, min(terrain.max_terrain_level - 1, round(difficulty * (terrain.max_terrain_level - 1))))
        sampled_levels = torch.randint(0, max_level + 1, (env_ids_tensor.numel(),), device=env.device)
        terrain.terrain_levels[env_ids_tensor] = sampled_levels
        terrain.env_origins[env_ids_tensor] = terrain.terrain_origins[
            sampled_levels, terrain.terrain_types[env_ids_tensor]
        ]
        return {"difficulty": float(difficulty), "max_level": float(max_level)}


__all__ = [
    "FOOT_NAMES",
    "FOOT_SCANNER_NAMES",
    "PaperBarrierReward",
    "PaperProprioception",
    "PaperStandardReward",
    "PaperTerrainDifficultyCurriculum",
    "paper_estimator_target",
    "relaxed_log_barrier",
]
