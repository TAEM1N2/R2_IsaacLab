"""This sub-module contains the reward functions that can be used for LimX Point Foot's locomotion task.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import os
import numpy as np
import torch
from torch import distributions
from typing import TYPE_CHECKING, Optional
from pathlib import Path

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


def _tensor_stats(tensor: torch.Tensor) -> dict:
    finite = torch.isfinite(tensor)
    stats = {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "num_bad": int((~finite).sum().item()),
    }
    if finite.any():
        finite_values = tensor[finite]
        stats.update(
            {
                "min": float(finite_values.min().item()),
                "max": float(finite_values.max().item()),
                "mean": float(finite_values.mean().item()),
                "std": float(finite_values.std().item()) if finite_values.numel() > 1 else 0.0,
            }
        )
    return stats


def _reward_debug_dir() -> Path:
    debug_dir = Path(os.getcwd()) / "nan_debug_rewards"
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def _check_reward_finite(
    env: ManagerBasedRLEnv,
    reward_name: str,
    reward: torch.Tensor,
    extra_tensors: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    if torch.isfinite(reward).all():
        return reward

    bad_mask = ~torch.isfinite(reward)
    bad_indices = bad_mask.nonzero(as_tuple=False).detach().cpu()
    payload = {
        "reward_name": reward_name,
        "step_dt": float(env.step_dt),
        "common_step_counter": int(getattr(env, "common_step_counter", -1)),
        "episode_length_buf_bad": env.episode_length_buf[bad_mask].detach().cpu(),
        "bad_indices": bad_indices,
        "reward_stats": _tensor_stats(reward),
        "reward": reward.detach().cpu(),
    }
    if extra_tensors:
        payload["extra_stats"] = {name: _tensor_stats(tensor) for name, tensor in extra_tensors.items()}
        payload["extra_tensors"] = {name: tensor.detach().cpu() for name, tensor in extra_tensors.items()}

    dump_path = _reward_debug_dir() / (
        f"{reward_name}_step{payload['common_step_counter']}_env"
        f"{bad_indices[0].item() if bad_indices.numel() > 0 else -1}.pt"
    )
    torch.save(payload, dump_path)
    print(f"[REWARD_NAN_DEBUG] Invalid values detected in reward term '{reward_name}'. dump={dump_path}", flush=True)
    print(
        f"[REWARD_NAN_DEBUG] bad_indices={bad_indices.squeeze(-1).tolist()} "
        f"stats={payload['reward_stats']}",
        flush=True,
    )
    if os.environ.get("PONGBOT_PLAY_IGNORE_REWARD_NAN", "0") == "1":
        print(
            f"[REWARD_NAN_DEBUG] Ignoring invalid reward term '{reward_name}' during play and replacing NaN/Inf with 0.0.",
            flush=True,
        )
        return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
    raise RuntimeError(f"Reward term '{reward_name}' contains NaN/Inf")


def _maybe_log_swing_height_debug(
    env: ManagerBasedRLEnv,
    feet_positions: torch.Tensor,
    ray_hits_w: torch.Tensor,
    nearest_hit_idx: torch.Tensor,
    nearest_terrain_z: torch.Tensor,
    forward_terrain_z: torch.Tensor,
    reference_terrain_z: torch.Tensor,
    feet_height: torch.Tensor,
    swing_height_target: torch.Tensor,
    swing_mask: torch.Tensor,
    step_rise: torch.Tensor,
    log_interval: int = 1,
) -> None:
    """Periodically print the actual height-scanner values used by pen_swing_height_error."""
    step = int(getattr(env, "common_step_counter", -1))
    if step < 0 or step % log_interval != 0 or env.num_envs == 0:
        return

    env_idx = 0
    ray_hits_env = ray_hits_w[env_idx]
    valid_hits = torch.isfinite(ray_hits_env[:, 2])
    nearest_idx_env = nearest_hit_idx[env_idx]
    selected_hits = ray_hits_env[nearest_idx_env].detach().cpu()
    valid_hit_z = ray_hits_env[valid_hits, 2]

    if valid_hit_z.numel() > 0:
        valid_hit_z_cpu = valid_hit_z.detach().cpu()
        hit_stats = {
            "min": round(float(valid_hit_z_cpu.min().item()), 4),
            "max": round(float(valid_hit_z_cpu.max().item()), 4),
            "mean": round(float(valid_hit_z_cpu.mean().item()), 4),
        }
    else:
        hit_stats = {"min": None, "max": None, "mean": None}

    # print(
    #     "[SWING_HEIGHT_DEBUG]",
    #     f"step={step}",
    #     f"feet_xyz={feet_positions[env_idx].detach().cpu().tolist()}",
    #     f"nearest_hit_idx={nearest_idx_env.detach().cpu().tolist()}",
    #     f"selected_hit_xyz={selected_hits.tolist()}",
    #     f"nearest_terrain_z={nearest_terrain_z[env_idx].detach().cpu().tolist()}",
    #     f"forward_terrain_z={forward_terrain_z[env_idx].detach().cpu().tolist()}",
    #     f"reference_terrain_z={reference_terrain_z[env_idx].detach().cpu().tolist()}",
    #     f"step_rise={step_rise[env_idx].detach().cpu().tolist()}",
    #     f"feet_height={feet_height[env_idx].detach().cpu().tolist()}",
    #     f"swing_target={swing_height_target[env_idx].detach().cpu().tolist()}",
    #     f"swing_mask={swing_mask[env_idx].detach().cpu().tolist()}",
    #     f"height_scanner_z_stats={hit_stats}",
    #     flush=True,
    # )


def lin_vel_error(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(lin_vel_error / std**2)


def ang_vel_error(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(ang_vel_error / std**2)



def stay_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for staying alive."""
    return torch.ones(env.num_envs, device=env.device)


def action_rate_l2_clamped(
    env: ManagerBasedRLEnv, min_value: float = 0.0, max_value: float | None = None
) -> torch.Tensor:
    """Penalize the action-rate magnitude with optional output clamping."""
    penalty = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    if max_value is None:
        return torch.clamp(penalty, min=min_value)
    return torch.clamp(penalty, min=min_value, max=max_value)

def foot_landing_vel(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        foot_radius: float,
        about_landing_threshold: float,
) -> torch.Tensor:
    """Penalize high foot landing velocities"""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    z_vels = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 0.1

    foot_positions = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    terrain_z = 0.0
    if "height_scanner" in env.scene.sensors:
        height_scanner: RayCaster = env.scene.sensors["height_scanner"]
        ray_hits_w = height_scanner.data.ray_hits_w
        valid_hits = torch.isfinite(ray_hits_w[..., 2])
        if torch.any(valid_hits):
            foot_xy = foot_positions[..., :2]
            ray_xy = ray_hits_w[..., :2]
            dist_sq = torch.sum(
                torch.square(foot_xy.unsqueeze(2) - ray_xy.unsqueeze(1)),
                dim=-1,
            )
            dist_sq = torch.where(valid_hits.unsqueeze(1), dist_sq, torch.inf)
            nearest_hit_idx = torch.argmin(dist_sq, dim=2, keepdim=True)
            terrain_z = torch.gather(ray_hits_w[..., 2], 1, nearest_hit_idx.squeeze(-1))
            terrain_z = torch.where(torch.isfinite(terrain_z), terrain_z, torch.zeros_like(terrain_z))

    foot_heights = torch.clip(foot_positions[..., 2] - terrain_z - foot_radius, 0, 1)

    about_to_land = (foot_heights < about_landing_threshold) & (~contacts) & (z_vels < 0.0)
    landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
    reward = torch.sum(torch.square(landing_z_vels), dim=1)
    return _check_reward_finite(
        env,
        "foot_landing_vel",
        reward,
        {
            "z_vels": z_vels,
            "foot_heights": foot_heights,
            "terrain_z": terrain_z if torch.is_tensor(terrain_z) else torch.tensor(terrain_z, device=reward.device),
        },
    )

def joint_powers_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint powers on the articulation using L1-kernel"""

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(torch.mul(asset.data.applied_torque, asset.data.joint_vel)), dim=1)


def joint_powers_var(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_group_names: tuple[str, ...] = (".*HR_JOINT", ".*HP_JOINT", ".*KN_JOINT"),
) -> torch.Tensor:
    """Penalize imbalance in power usage within each joint type.

    The variance is computed separately for each joint group, e.g. left/right HR,
    left/right HP, left/right KN, and then summed. This avoids comparing inherently
    different joint types against each other.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    joint_power = torch.abs(asset.data.applied_torque * asset.data.joint_vel)

    penalty = torch.zeros(env.num_envs, device=env.device)
    for joint_name in joint_group_names:
        joint_ids = asset.find_joints(joint_name)[0]
        if len(joint_ids) == 0:
            continue
        penalty += torch.var(joint_power[:, joint_ids], dim=1, unbiased=False)

    return penalty

def no_fly(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Reward if only one foot is in contact with the ground."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    latest_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, :, 2]

    contacts = latest_contact_forces > threshold
    single_contact = torch.sum(contacts.float(), dim=1) == 1

    return 1.0 * single_contact


def unbalance_feet_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize if the feet air time variance exceeds the balance threshold."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    return torch.var(contact_sensor.data.last_air_time[:, sensor_cfg.body_ids], dim=-1)


def unbalance_feet_height(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the variance of feet maximum height using sensor positions."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    feet_positions = contact_sensor.data.pos_w[:, sensor_cfg.body_ids]

    if feet_positions is None:
        return torch.zeros(env.num_envs)

    feet_heights = feet_positions[:, :, 2]
    max_feet_heights = torch.max(feet_heights, dim=-1)[0]
    height_variance = torch.var(max_feet_heights, dim=-1)
    return height_variance


# def feet_distance(
#     env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Penalize if the distance between feet is below a minimum threshold."""

#     asset: Articulation = env.scene[asset_cfg.name]

#     feet_positions = asset.data.joint_pos[sensor_cfg.body_ids]

#     if feet_positions is None:
#         return torch.zeros(env.num_envs)

#     # feet distance on x-y plane
#     feet_distance = torch.norm(feet_positions[0, :2] - feet_positions[1, :2], dim=-1)

#     return torch.clamp(0.1 - feet_distance, min=0.0)


def feet_distance(env: ManagerBasedRLEnv,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                  feet_links_name: list[str]=[".*TIP"],
                  min_feet_distance: float = 0.1,
                  max_feet_distance: float = 1.0,)-> torch.Tensor:
    # Penalize base height away from target
    asset: Articulation = env.scene[asset_cfg.name]
    feet_links_idx = asset.find_bodies(feet_links_name)[0]
    feet_pos = asset.data.body_link_pos_w[:,feet_links_idx]
    # feet distance on x-y plane
    feet_distance = torch.norm(feet_pos[:, 0, :2] - feet_pos[:, 1, :2], dim=-1)
    reward = torch.clip(min_feet_distance - feet_distance, 0, 1)
    reward += torch.clip(feet_distance - max_feet_distance, 0, 1)
    return reward

def nominal_foot_position(env: ManagerBasedRLEnv, command_name: str,
                          base_height_target: float,
                           asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Compute the nominal foot position"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    # assert (compute_rotation_distance(asset.data.root_com_quat_w, asset.data.root_link_quat_w) < 0.1).all()
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = math_utils.quat_rotate_inverse(
        base_quat,
        feet_pos_w - base_pos,
    )
    feet_center_b = torch.mean(feet_pos_b[:, :, :3], dim=1)
    base_height_error = torch.abs((feet_center_b[:, 2] - env._foot_radius + base_height_target))

    reward = torch.exp(-base_height_error / std**2)
    return reward

def leg_symmetry(env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """Reward regulate abad joint position."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    # assert (compute_rotation_distance(asset.data.root_com_quat_w, asset.data.root_link_quat_w) < 0.1).all()
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = math_utils.quat_rotate_inverse(
        base_quat,
        feet_pos_w - base_pos,
    )
    leg_symmetry_err = torch.abs(feet_pos_b[:, 0, 1]) - torch.abs(feet_pos_b[:, 1, 1])

    return torch.exp(-leg_symmetry_err ** 2 / std**2)

def same_feet_x_position(env: ManagerBasedRLEnv,
                  asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward regulate abad joint position."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    # assert (compute_rotation_distance(asset.data.root_com_quat_w, asset.data.root_link_quat_w) < 0.1).all()
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = math_utils.quat_rotate_inverse(
        base_quat,
        feet_pos_w - base_pos,
    )
    feet_x_distance = torch.abs(feet_pos_b[:, 0, 0] - feet_pos_b[:, 1, 0])
    # return torch.exp(-feet_x_distance / 0.2)
    return feet_x_distance

def keep_ankle_pitch_zero_in_air(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor", body_names=["ankle_[LR]_Link"]),
    force_threshold: float = 2.0,
    pitch_scale: float = 0.2
) -> torch.Tensor:
    """Reward for keeping ankle pitch angle close to zero when foot is in the air.
    
    Args:
        env: The environment object.
        asset_cfg: Configuration for the robot asset containing DOF positions.
        sensor_cfg: Configuration for the contact force sensor.
        force_threshold: Threshold value for contact detection (in Newtons).
        pitch_scale: Scaling factor for the exponential reward.
        
    Returns:
        The computed reward tensor.
    """
    asset = env.scene[asset_cfg.name]
    contact_forces_history = env.scene.sensors[sensor_cfg.name].data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    current_contact = torch.norm(contact_forces_history[:, -1], dim=-1) > force_threshold
    last_contact = torch.norm(contact_forces_history[:, -2], dim=-1) > force_threshold
    contact_filt = torch.logical_or(current_contact, last_contact)
    ankle_pitch_left = torch.abs(asset.data.joint_pos[:, 3]) * ~contact_filt[:, 0]
    ankle_pitch_right = torch.abs(asset.data.joint_pos[:, 7]) * ~contact_filt[:, 1]
    weighted_ankle_pitch = ankle_pitch_left + ankle_pitch_right
    return torch.exp(-weighted_ankle_pitch / pitch_scale)

def no_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Penalize if both feet are not in contact with the ground.
    """

    # Access the contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Get the latest contact forces in the z direction (upward direction)
    latest_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, :, 2]  # shape: (env_num, 2)

    # Determine if each foot is in contact
    contacts = latest_contact_forces > 1.0  # Returns a boolean tensor where True indicates contact

    return (torch.sum(contacts.float(), dim=1) == 0).float()


def stand_still(
    env, lin_threshold: float = 0.05, ang_threshold: float = 0.05, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    penalizing linear and angular motion when command velocities are near zero.
    """

    asset = env.scene[asset_cfg.name]
    base_lin_vel = asset.data.root_lin_vel_w[:, :2]
    base_ang_vel = asset.data.root_ang_vel_w[:, -1]

    commands = env.command_manager.get_command("base_velocity")

    lin_commands = commands[:, :2]
    ang_commands = commands[:, 2]

    reward_lin = torch.sum(
        torch.abs(base_lin_vel) * (torch.norm(lin_commands, dim=1, keepdim=True) < lin_threshold), dim=-1
    )

    reward_ang = torch.abs(base_ang_vel) * (torch.abs(ang_commands) < ang_threshold)

    total_reward = reward_lin + reward_ang
    return _check_reward_finite(
        env,
        "stand_still",
        total_reward,
        {"base_lin_vel": base_lin_vel, "base_ang_vel": base_ang_vel, "commands": commands},
    )


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    lin_threshold: float = 0.05,
    ang_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from the default joint pose only when the velocity command is near zero."""

    asset: Articulation = env.scene[asset_cfg.name]
    commands = env.command_manager.get_command("base_velocity")

    standing_mask = (
        (torch.norm(commands[:, :2], dim=1) < lin_threshold)
        & (torch.abs(commands[:, 2]) < ang_threshold)
    ).float()

    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    joint_deviation = torch.abs(
        asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    )
    reward = torch.sum(joint_deviation, dim=1) * standing_mask

    return _check_reward_finite(
        env,
        "stand_still_joint_deviation_l1",
        reward,
        {"commands": commands, "standing_mask": standing_mask, "joint_deviation": joint_deviation},
    )


# def feet_regulation(
#     env: ManagerBasedRLEnv,
#     sensor_cfg: SceneEntityCfg,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     desired_body_height: float = 0.65,
# ) -> torch.Tensor:
#     """Penalize if the feet are not in contact with the ground.

#     Args:
#         env: The environment object.
#         sensor_cfg: The configuration of the contact sensor.
#         desired_body_height: The desired body height used for normalization.

#     Returns:
#         A tensor representing the feet regulation penalty for each environment.
#     """

#     asset: Articulation = env.scene[asset_cfg.name]

#     feet_positions_z = asset.data.joint_pos[sensor_cfg.body_ids, 2]

#     feet_vel_xy = asset.data.joint_vel[sensor_cfg.body_ids, :2]

#     vel_norms_xy = torch.norm(feet_vel_xy, dim=-1)

#     exp_term = torch.exp(-feet_positions_z / (0.025 * desired_body_height))

#     r_fr = torch.sum(vel_norms_xy**2 * exp_term, dim=-1)

#     return r_fr

def feet_regulation(env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    foot_radius: float,
    base_height_target: float,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    feet_positions = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    terrain_z = 0.0
    if "height_scanner" in env.scene.sensors:
        height_scanner: RayCaster = env.scene.sensors["height_scanner"]
        ray_hits_w = height_scanner.data.ray_hits_w
        valid_hits = torch.isfinite(ray_hits_w[..., 2])
        if torch.any(valid_hits):
            dist_sq = torch.sum(
                torch.square(feet_positions[..., :2].unsqueeze(2) - ray_hits_w[..., :2].unsqueeze(1)),
                dim=-1,
            )
            dist_sq = torch.where(valid_hits.unsqueeze(1), dist_sq, torch.inf)
            nearest_hit_idx = torch.argmin(dist_sq, dim=2)
            terrain_z = torch.gather(ray_hits_w[..., 2], 1, nearest_hit_idx)
            terrain_z = torch.where(torch.isfinite(terrain_z), terrain_z, torch.zeros_like(terrain_z))

    feet_height = torch.clip(feet_positions[..., 2] - terrain_z - foot_radius, 0, 1)
    feet_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]

    height_scale = torch.exp(-feet_height / base_height_target)
    reward = torch.sum(height_scale * torch.square(torch.norm(feet_vel_xy, dim=-1)), dim=1)
    return _check_reward_finite(
        env,
        "feet_regulation",
        reward,
        {
            "feet_positions": feet_positions,
            "terrain_z": terrain_z if torch.is_tensor(terrain_z) else torch.tensor(terrain_z, device=reward.device),
            "feet_height": feet_height,
            "feet_vel_xy": feet_vel_xy,
            "height_scale": height_scale,
        },
    )


def pen_swing_height_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    foot_radius: float,
) -> torch.Tensor:
    """Penalize insufficient swing-foot clearance using a forward-aware terrain reference."""
    asset: Articulation = env.scene[asset_cfg.name]
    gait_params = env.command_manager.get_command(command_name)
    base_velocity_cmd = env.command_manager.get_command("base_velocity")
    clearance_margin = 0.03
    forward_window = 0.30
    lateral_window = 0.10

    frequencies = gait_params[:, 0]
    offsets = gait_params[:, 1]
    durations = torch.cat(
        [gait_params[:, 2].view(env.num_envs, 1), gait_params[:, 2].view(env.num_envs, 1)],
        dim=1,
    )
    # swing_height_target = gait_params[:, 3].unsqueeze(1)
    

    gait_indices = torch.remainder(env.episode_length_buf * env.step_dt * frequencies, 1.0)
    foot_indices = torch.remainder(
        torch.cat(
            [gait_indices.view(env.num_envs, 1), (gait_indices + offsets + 1.0).view(env.num_envs, 1)],
            dim=1,
        ),
        1.0,
    )
    swing_mask = foot_indices > durations

    feet_positions = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    nearest_terrain_z = torch.zeros_like(feet_positions[..., 2])
    forward_terrain_z = torch.zeros_like(feet_positions[..., 2])
    reference_terrain_z = torch.zeros_like(feet_positions[..., 2])
    if "height_scanner" in env.scene.sensors:
        height_scanner: RayCaster = env.scene.sensors["height_scanner"]
        ray_hits_w = height_scanner.data.ray_hits_w
        valid_hits = torch.isfinite(ray_hits_w[..., 2])
        if torch.any(valid_hits):
            dist_sq = torch.sum(
                torch.square(feet_positions[..., :2].unsqueeze(2) - ray_hits_w[..., :2].unsqueeze(1)),
                dim=-1,
            )
            dist_sq = torch.where(valid_hits.unsqueeze(1), dist_sq, torch.inf)
            nearest_hit_idx = torch.argmin(dist_sq, dim=2)
            nearest_terrain_z = torch.gather(ray_hits_w[..., 2], 1, nearest_hit_idx)
            nearest_terrain_z = torch.where(
                torch.isfinite(nearest_terrain_z), nearest_terrain_z, torch.zeros_like(nearest_terrain_z)
            )

            cmd_xy = base_velocity_cmd[:, :2]
            cmd_xy_norm = torch.norm(cmd_xy, dim=1, keepdim=True)
            cmd_direction = cmd_xy / torch.clamp(cmd_xy_norm, min=1.0e-6)
            body_forward = math_utils.quat_apply_yaw(
                asset.data.root_quat_w, torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
            )[:, :2]
            body_forward = body_forward / torch.clamp(torch.norm(body_forward, dim=1, keepdim=True), min=1.0e-6)
            use_body_forward = cmd_xy_norm.squeeze(-1) <= 0.1
            move_direction = torch.where(use_body_forward.unsqueeze(-1), body_forward, cmd_direction)
            lateral_direction = torch.stack([-move_direction[:, 1], move_direction[:, 0]], dim=1)

            hit_delta_xy = ray_hits_w[..., :2].unsqueeze(1) - feet_positions[..., :2].unsqueeze(2)
            forward_dist = torch.sum(hit_delta_xy * move_direction.unsqueeze(1).unsqueeze(1), dim=-1)
            lateral_dist = torch.abs(torch.sum(hit_delta_xy * lateral_direction.unsqueeze(1).unsqueeze(1), dim=-1))
            forward_candidates = (
                valid_hits.unsqueeze(1)
                & (forward_dist > 0.0)
                & (forward_dist <= forward_window)
                & (lateral_dist <= lateral_window)
            )
            forward_hit_z = ray_hits_w[..., 2].unsqueeze(1).expand(-1, feet_positions.shape[1], -1)
            forward_hit_z = torch.where(forward_candidates, forward_hit_z, torch.full_like(forward_hit_z, -torch.inf))
            forward_terrain_z = torch.max(forward_hit_z, dim=2).values
            forward_terrain_z = torch.where(
                torch.isfinite(forward_terrain_z), forward_terrain_z, nearest_terrain_z
            )
            reference_terrain_z = torch.maximum(nearest_terrain_z, forward_terrain_z)
            debug_swing_mask = swing_mask
            debug_step_rise = torch.clamp(reference_terrain_z - nearest_terrain_z, min=0.0)
            debug_swing_height_target = gait_params[:, 3:4].expand(-1, feet_positions.shape[1]) + clearance_margin + debug_step_rise
            if debug_swing_mask.shape[1] != feet_positions.shape[1]:
                repeat_factor = (feet_positions.shape[1] + debug_swing_mask.shape[1] - 1) // debug_swing_mask.shape[1]
                debug_swing_mask = debug_swing_mask.repeat(1, repeat_factor)[:, : feet_positions.shape[1]]
                debug_swing_height_target = debug_swing_height_target.repeat(1, repeat_factor)[:, : feet_positions.shape[1]]
                debug_step_rise = debug_step_rise.repeat(1, repeat_factor)[:, : feet_positions.shape[1]]
            _maybe_log_swing_height_debug(
                env,
                feet_positions=feet_positions,
                ray_hits_w=ray_hits_w,
                nearest_hit_idx=nearest_hit_idx,
                nearest_terrain_z=nearest_terrain_z,
                forward_terrain_z=forward_terrain_z,
                reference_terrain_z=reference_terrain_z,
                feet_height=torch.clamp(feet_positions[..., 2] - nearest_terrain_z - foot_radius, min=0.0),
                swing_height_target=debug_swing_height_target,
                swing_mask=debug_swing_mask,
                step_rise=debug_step_rise,
            )

    if not torch.any(reference_terrain_z):
        reference_terrain_z = nearest_terrain_z
    feet_height = torch.clamp(feet_positions[..., 2] - nearest_terrain_z - foot_radius, min=0.0)
    step_rise = torch.clamp(reference_terrain_z - nearest_terrain_z, min=0.0)
    swing_height_target = gait_params[:, 3:4].expand(-1, feet_height.shape[1]) + clearance_margin + step_rise

    if swing_mask.shape[1] != feet_height.shape[1]:
        repeat_factor = (feet_height.shape[1] + swing_mask.shape[1] - 1) // swing_mask.shape[1]
        swing_mask = swing_mask.repeat(1, repeat_factor)[:, : feet_height.shape[1]]
        swing_height_target = swing_height_target.repeat(1, repeat_factor)[:, : feet_height.shape[1]]
        step_rise = step_rise.repeat(1, repeat_factor)[:, : feet_height.shape[1]]

    clearance_error = torch.clamp(swing_height_target - feet_height, min=0.0)
    height_error = torch.square(clearance_error)
    reward = torch.sum(torch.where(swing_mask, height_error, torch.zeros_like(height_error)), dim=1)
    reward *= torch.norm(base_velocity_cmd[:, :2], dim=1) > 0.1

    return _check_reward_finite(
        env,
        "pen_swing_height_error",
        reward,
        {
            "gait_params": gait_params,
            "base_velocity_cmd": base_velocity_cmd,
            "swing_mask": swing_mask,
            "feet_height": feet_height,
            "swing_height_target": swing_height_target,
            "nearest_terrain_z": nearest_terrain_z,
            "forward_terrain_z": forward_terrain_z,
            "reference_terrain_z": reference_terrain_z,
            "step_rise": step_rise,
        },
    )


def base_height_rough_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        Currently, it assumes a flat terrain, i.e. the target height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    height = asset.data.root_pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[:, :, 2]
    # sensor.data.ray_hits_w can be inf, so we clip it to avoid NaN
    height = torch.nan_to_num(height, nan=target_height, posinf=target_height, neginf=target_height)
    reward = torch.square(height.mean(dim=1) - target_height)
    return _check_reward_finite(
        env,
        "base_height_rough_l2",
        reward,
        {"height": height, "ray_hits_z": sensor.data.ray_hits_w[:, :, 2]},
    )


def base_com_height(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits_z = sensor.data.ray_hits_w[..., 2]
        finite_hits = torch.isfinite(ray_hits_z)
        sanitized_hits = torch.where(finite_hits, ray_hits_z, torch.zeros_like(ray_hits_z))
        hit_counts = finite_hits.sum(dim=1).clamp(min=1)
        adjusted_target_height = target_height + sanitized_hits.sum(dim=1) / hit_counts
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
        ray_hits_z = None
    # Compute the L2 squared penalty
    reward = torch.abs(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    extra = {
        "root_pos_z": asset.data.root_pos_w[:, 2],
        "adjusted_target_height": adjusted_target_height,
    }
    if ray_hits_z is not None:
        extra["ray_hits_z"] = ray_hits_z
    return _check_reward_finite(env, "base_com_height", reward, extra)


class GaitReward(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)

        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]

        # extract the used quantities (to enable type-hinting)
        self.contact_sensor: ContactSensor = env.scene.sensors[self.sensor_cfg.name]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

        # Store configuration parameters
        self.force_scale = float(cfg.params["tracking_contacts_shaped_force"])
        self.vel_scale = float(cfg.params["tracking_contacts_shaped_vel"])
        self.force_sigma = cfg.params["gait_force_sigma"]
        self.vel_sigma = cfg.params["gait_vel_sigma"]
        self.kappa_gait_probs = cfg.params["kappa_gait_probs"]
        self.command_name = cfg.params["command_name"]
        self.dt = env.step_dt

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        tracking_contacts_shaped_force,
        tracking_contacts_shaped_vel,
        gait_force_sigma,
        gait_vel_sigma,
        kappa_gait_probs,
        command_name,
        sensor_cfg,
        asset_cfg,
    ) -> torch.Tensor:
        """Compute the reward.

        The reward combines force-based and velocity-based terms to encourage desired gait patterns.

        Args:
            env: The RL environment instance.

        Returns:
            The reward value.
        """

        gait_params = env.command_manager.get_command(self.command_name)
        base_velocity_cmd = env.command_manager.get_command("base_velocity")
        standing_cmd = torch.norm(base_velocity_cmd[:, :2], dim=1) <= 0.1

        # Update contact targets
        desired_contact_states = self.compute_contact_targets(gait_params)

        # Force-based reward
        foot_forces = torch.norm(self.contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids], dim=-1)
        desired_contact_states = self._match_contact_shape(desired_contact_states, foot_forces.shape[1])
        desired_contact_states = torch.where(
            standing_cmd.unsqueeze(1), torch.ones_like(desired_contact_states), desired_contact_states
        )
        force_reward = self._compute_force_reward(foot_forces, desired_contact_states)

        # Velocity-based reward
        foot_velocities = torch.norm(self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids], dim=-1)
        desired_contact_states = self._match_contact_shape(desired_contact_states, foot_velocities.shape[1])
        velocity_reward = self._compute_velocity_reward(foot_velocities, desired_contact_states)

        # Combine rewards
        total_reward = force_reward + velocity_reward
        return _check_reward_finite(
            env,
            "GaitReward",
            total_reward,
            {
                "gait_params": gait_params,
                "desired_contact_states": desired_contact_states,
                "foot_forces": foot_forces,
                "foot_velocities": foot_velocities,
                "force_reward": force_reward,
                "velocity_reward": velocity_reward,
            },
        )

    def compute_contact_targets(self, gait_params):
        """Calculate desired contact states for the current timestep."""
        frequencies = gait_params[:, 0]
        offsets = gait_params[:, 1]
        durations = torch.cat(
            [
                gait_params[:, 2].view(self.num_envs, 1),
                gait_params[:, 2].view(self.num_envs, 1),
            ],
            dim=1,
        )

        assert torch.all(frequencies > 0), "Frequencies must be positive"
        assert torch.all((offsets >= 0) & (offsets <= 1)), "Offsets must be between 0 and 1"
        assert torch.all((durations > 0) & (durations < 1)), "Durations must be between 0 and 1"

        gait_indices = torch.remainder(self._env.episode_length_buf * self.dt * frequencies, 1.0)

        # Calculate foot indices
        foot_indices = torch.remainder(
            torch.cat(
                [gait_indices.view(self.num_envs, 1), (gait_indices + offsets + 1).view(self.num_envs, 1)],
                dim=1,
            ),
            1.0,
        )

        # Determine stance and swing phases
        stance_idxs = foot_indices < durations
        swing_idxs = foot_indices > durations

        # Adjust foot indices based on phase
        foot_indices[stance_idxs] = torch.remainder(foot_indices[stance_idxs], 1) * (0.5 / durations[stance_idxs])
        foot_indices[swing_idxs] = 0.5 + (torch.remainder(foot_indices[swing_idxs], 1) - durations[swing_idxs]) * (
            0.5 / (1 - durations[swing_idxs])
        )

        # Calculate desired contact states using von mises distribution
        smoothing_cdf_start = distributions.normal.Normal(0, self.kappa_gait_probs).cdf
        desired_contact_states = smoothing_cdf_start(foot_indices) * (
            1 - smoothing_cdf_start(foot_indices - 0.5)
        ) + smoothing_cdf_start(foot_indices - 1) * (1 - smoothing_cdf_start(foot_indices - 1.5))

        return desired_contact_states

    def _match_contact_shape(self, desired_contacts: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Expand or trim desired contact states to match the number of tracked feet.

        The gait command encodes two phases. For quadrupeds we map them to diagonal pairs:
        [phase_a, phase_b] -> [phase_a, phase_b, phase_b, phase_a].
        """
        if desired_contacts.shape[1] == target_dim:
            return desired_contacts

        if desired_contacts.shape[1] == 2 and target_dim == 4:
            return torch.stack(
                [
                    desired_contacts[:, 0],
                    desired_contacts[:, 1],
                    desired_contacts[:, 1],
                    desired_contacts[:, 0],
                ],
                dim=1,
            )

        if desired_contacts.shape[1] > target_dim:
            return desired_contacts[:, :target_dim]

        repeat_factor = (target_dim + desired_contacts.shape[1] - 1) // desired_contacts.shape[1]
        expanded = desired_contacts.repeat(1, repeat_factor)
        return expanded[:, :target_dim]

    def _compute_force_reward(self, forces: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """Compute force-based reward component."""
        reward = torch.zeros_like(forces[:, 0])
        if self.force_scale < 0:  # Negative scale means penalize unwanted contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * (1 - torch.exp(-forces[:, i] ** 2 / self.force_sigma))
        else:  # Positive scale means reward desired contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * torch.exp(-forces[:, i] ** 2 / self.force_sigma)

        return (reward / forces.shape[1]) * self.force_scale

    def _compute_velocity_reward(self, velocities: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """Compute velocity-based reward component."""
        reward = torch.zeros_like(velocities[:, 0])
        if self.vel_scale < 0:  # Negative scale means penalize movement during contact
            for i in range(velocities.shape[1]):
                reward += desired_contacts[:, i] * (1 - torch.exp(-velocities[:, i] ** 2 / self.vel_sigma))
        else:  # Positive scale means reward movement during swing
            for i in range(velocities.shape[1]):
                reward += desired_contacts[:, i] * torch.exp(-velocities[:, i] ** 2 / self.vel_sigma)

        return (reward / velocities.shape[1]) * self.vel_scale


class ActionSmoothnessPenalty(ManagerTermBase):
    """
    A reward term for penalizing large instantaneous changes in the network action output.
    This penalty encourages smoother actions over time.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward term.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.dt = env.step_dt
        self.prev_prev_action = None
        self.prev_action = None
        # self.__name__ = "action_smoothness_penalty"

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute the action smoothness penalty.

        Args:
            env: The RL environment instance.

        Returns:
            The penalty value based on the action smoothness.
        """
        # Get the current action from the environment's action manager
        current_action = env.action_manager.action.clone()

        # If this is the first call, initialize the previous actions
        if self.prev_action is None:
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        if self.prev_prev_action is None:
            self.prev_prev_action = self.prev_action
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        # Compute the smoothness penalty
        penalty = torch.sum(torch.square(current_action - 2 * self.prev_action + self.prev_prev_action), dim=1)

        # Update the previous actions for the next call
        self.prev_prev_action = self.prev_action
        self.prev_action = current_action

        # Apply a condition to ignore penalty during the first few episodes
        startup_env_mask = env.episode_length_buf < 3
        penalty[startup_env_mask] = 0

        # Return the penalty scaled by the configured weight
        return _check_reward_finite(env, "ActionSmoothnessPenalty", penalty, {"current_action": current_action})
