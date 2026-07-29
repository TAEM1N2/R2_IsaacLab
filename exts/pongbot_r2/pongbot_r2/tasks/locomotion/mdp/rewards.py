"""This sub-module contains the reward functions that can be used for LimX Point Foot's locomotion task.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.

@version 0.0.3
@update 2026-07-13: Add linear tracking advantage over the stationary baseline.
@update 2026-07-13: Detect TIP obstacles and add capped staged recovery credit.
"""

from __future__ import annotations

import os
import re
import torch
from torch import distributions
from typing import TYPE_CHECKING, Optional
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
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
    log_interval: int = 7,
) -> None:
    """Keep the old debug hook inert during training and play."""
    return


def _maybe_visualize_swing_height_debug(
    env: ManagerBasedRLEnv,
    nearest_hit_xyz: torch.Tensor,
    highest_hit_xyz: torch.Tensor,
    feet_positions: torch.Tensor,
    nearest_terrain_z: torch.Tensor,
    swing_height_target: torch.Tensor,
    foot_radius: float,
) -> None:
    """Visualize the terrain references and swing-height target for env 0."""
    swing_markers = os.environ.get("PONGBOT_SWING_HEIGHT_MARKERS", "0") == "1"
    foot_markers = os.environ.get("PONGBOT_FOOT_SCANNER_MARKERS", "0") == "1"
    if not (swing_markers or foot_markers) or env.num_envs != 1:
        return

    if not hasattr(env, "_pongbot_swing_height_markers"):
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/PongBotSwingHeightDebug",
            markers={
                "nearest_terrain": sim_utils.SphereCfg(
                    radius=0.02,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
                "highest_forward_terrain": sim_utils.SphereCfg(
                    radius=0.025,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
                "swing_height_target": sim_utils.SphereCfg(
                    radius=0.025,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.25, 1.0)),
                ),
            },
        )
        env._pongbot_swing_height_markers = VisualizationMarkers(marker_cfg)
        env._pongbot_swing_height_markers.set_visibility(True)
        print(
            "[SWING_HEIGHT_MARKERS] enabled: green=nearest terrain, red=highest forward terrain, blue=swing target",
            flush=True,
        )

    env_idx = 0
    target_xyz = feet_positions[env_idx].detach().clone()
    if torch.is_tensor(foot_radius):
        radius_env = foot_radius[env_idx].reshape(-1)
    else:
        radius_env = torch.full_like(nearest_terrain_z[env_idx], float(foot_radius))
    target_xyz[:, 2] = nearest_terrain_z[env_idx] + radius_env + swing_height_target[env_idx]

    translations = torch.cat(
        [
            nearest_hit_xyz[env_idx],
            highest_hit_xyz[env_idx],
            target_xyz,
        ],
        dim=0,
    )
    marker_indices = torch.cat(
        [
            torch.zeros(nearest_hit_xyz.shape[1], dtype=torch.long, device=env.device),
            torch.ones(highest_hit_xyz.shape[1], dtype=torch.long, device=env.device),
            torch.full((target_xyz.shape[0],), 2, dtype=torch.long, device=env.device),
        ],
        dim=0,
    )
    env._pongbot_swing_height_markers.visualize(translations=translations, marker_indices=marker_indices)


def _maybe_visualize_foot_height_scanners(env: ManagerBasedRLEnv) -> None:
    """Visualize foot-centered height scanner ray hits for env 0."""
    if env.num_envs != 1 or os.environ.get("PONGBOT_FOOT_SCANNER_MARKERS", "0") != "1":
        return

    sensor_names = (
        "fl_foot_height_scanner",
        "fr_foot_height_scanner",
        "rl_foot_height_scanner",
        "rr_foot_height_scanner",
    )
    if not all(sensor_name in env.scene.sensors for sensor_name in sensor_names):
        return

    if not hasattr(env, "_pongbot_foot_scanner_markers"):
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/PongBotFootHeightScanners",
            markers={
                "fl": sim_utils.SphereCfg(
                    radius=0.01,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
                "fr": sim_utils.SphereCfg(
                    radius=0.01,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
                "rl": sim_utils.SphereCfg(
                    radius=0.01,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.25, 1.0)),
                ),
                "rr": sim_utils.SphereCfg(
                    radius=0.01,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                ),
            },
        )
        env._pongbot_foot_scanner_markers = VisualizationMarkers(marker_cfg)
        env._pongbot_foot_scanner_markers.set_visibility(True)
        print(
            "[FOOT_SCANNER_MARKERS] enabled: red=FL, green=FR, blue=RL, yellow=RR",
            flush=True,
        )

    hit_batches = []
    marker_batches = []
    for marker_idx, sensor_name in enumerate(sensor_names):
        hits = env.scene.sensors[sensor_name].data.ray_hits_w.reshape(-1, 3)
        finite_hits = hits[torch.isfinite(hits[:, 2])]
        if finite_hits.numel() == 0:
            continue
        finite_hits = finite_hits.clone()
        finite_hits[:, 2] += 0.015
        hit_batches.append(finite_hits)
        marker_batches.append(torch.full((finite_hits.shape[0],), marker_idx, dtype=torch.long, device=env.device))

    if len(hit_batches) == 0:
        return

    env._pongbot_foot_scanner_markers.visualize(
        translations=torch.cat(hit_batches, dim=0),
        marker_indices=torch.cat(marker_batches, dim=0),
    )


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


def nonzero_lin_vel_tracking_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    lin_threshold: float = 0.1,
    max_error: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared xy velocity tracking error, active only for nonzero linear velocity commands."""
    asset: RigidObject = env.scene[asset_cfg.name]
    commands_xy = env.command_manager.get_command(command_name)[:, :2]
    error = torch.sum(torch.square(commands_xy - asset.data.root_lin_vel_b[:, :2]), dim=1)
    if max_error is not None:
        error = torch.clamp(error, max=max_error)
    moving_mask = torch.norm(commands_xy, dim=1) > lin_threshold
    error = error * moving_mask.float()
    return _check_reward_finite(
        env,
        "nonzero_lin_vel_tracking_error_l2",
        error,
        {"commands_xy": commands_xy, "root_lin_vel_b": asset.data.root_lin_vel_b[:, :2], "moving_mask": moving_mask},
    )


def track_lin_vel_xy_advantage_over_stationary(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    command_threshold: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward velocity tracking relative to remaining stationary.

    Moving commands receive the exponential tracking reward minus the reward
    that a zero-velocity robot would receive. Zero commands keep the original
    exponential reward so stable standing remains positively reinforced.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command_xy = env.command_manager.get_command(command_name)[:, :2]
    velocity_xy = asset.data.root_lin_vel_b[:, :2]
    tracking_error = torch.sum(torch.square(command_xy - velocity_xy), dim=1)
    stationary_error = torch.sum(torch.square(command_xy), dim=1)
    tracking_reward = torch.exp(-tracking_error / std**2)
    stationary_reward = torch.exp(-stationary_error / std**2)
    moving = torch.linalg.vector_norm(command_xy, dim=1) > command_threshold
    reward = torch.where(moving, tracking_reward - stationary_reward, tracking_reward)
    return _check_reward_finite(
        env,
        "track_lin_vel_xy_advantage_over_stationary",
        reward,
        {
            "command_xy": command_xy,
            "velocity_xy": velocity_xy,
            "tracking_error": tracking_error,
            "stationary_reward": stationary_reward,
        },
    )


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


def _action_columns_for_joints(
    env: ManagerBasedRLEnv,
    asset: Articulation,
    joint_name_expr: str | tuple[str, ...] | list[str],
    action_term_name: str,
) -> list[int]:
    """Resolve action columns corresponding to named robot joints."""
    joint_names = [joint_name_expr] if isinstance(joint_name_expr, str) else list(joint_name_expr)
    joint_ids = asset.find_joints(joint_names)[0]
    if len(joint_ids) == 0:
        raise ValueError(f"Failed to resolve joint names for action penalty: {joint_names}")

    action_term = env.action_manager.get_term(action_term_name)
    action_joint_ids = getattr(action_term, "_joint_ids", None)
    if action_joint_ids is None:
        if env.action_manager.action.shape[1] == asset.num_joints:
            action_joint_ids = list(range(asset.num_joints))
        else:
            raise RuntimeError(
                f"Action term '{action_term_name}' does not expose _joint_ids and action dimension "
                f"{env.action_manager.action.shape[1]} does not match robot joints {asset.num_joints}."
            )

    if isinstance(action_joint_ids, slice):
        action_joint_ids = list(range(asset.num_joints))[action_joint_ids]
    elif torch.is_tensor(action_joint_ids):
        action_joint_ids = action_joint_ids.detach().cpu().tolist()
    else:
        action_joint_ids = list(action_joint_ids)

    joint_id_set = {int(joint_id) for joint_id in joint_ids}
    action_columns = [idx for idx, joint_id in enumerate(action_joint_ids) if int(joint_id) in joint_id_set]
    if len(action_columns) == 0:
        raise ValueError(f"Joints {joint_names} are not part of action term '{action_term_name}'.")
    return action_columns


def joint_action_delta_excess_l2(
    env: ManagerBasedRLEnv,
    joint_name_expr: str | tuple[str, ...] | list[str],
    threshold: float,
    action_term_name: str = "joint_pos",
    max_value: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize only large action jumps for selected joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    action_columns = _action_columns_for_joints(env, asset, joint_name_expr, action_term_name)
    action = env.action_manager.action[:, action_columns]
    prev_action = env.action_manager.prev_action[:, action_columns]
    excess = torch.relu(torch.abs(action - prev_action) - threshold)
    penalty = torch.sum(torch.square(excess), dim=1)
    if max_value is not None:
        penalty = torch.clamp(penalty, max=max_value)
    return _check_reward_finite(
        env,
        "joint_action_delta_excess_l2",
        penalty,
        {"action": action, "prev_action": prev_action, "excess": excess},
    )


def joint_action_abs_excess_l2(
    env: ManagerBasedRLEnv,
    joint_name_expr: str | tuple[str, ...] | list[str],
    threshold: float,
    action_term_name: str = "joint_pos",
    max_value: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize selected joints only when absolute action approaches saturation."""
    asset: Articulation = env.scene[asset_cfg.name]
    action_columns = _action_columns_for_joints(env, asset, joint_name_expr, action_term_name)
    action = env.action_manager.action[:, action_columns]
    excess = torch.relu(torch.abs(action) - threshold)
    penalty = torch.sum(torch.square(excess), dim=1)
    if max_value is not None:
        penalty = torch.clamp(penalty, max=max_value)
    return _check_reward_finite(
        env,
        "joint_action_abs_excess_l2",
        penalty,
        {"action": action, "excess": excess},
    )


def moving_stall_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.2,
    velocity_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize near-zero body velocity when a nonzero xy command is requested."""
    asset: RigidObject = env.scene[asset_cfg.name]
    commands_xy = env.command_manager.get_command(command_name)[:, :2]
    base_vel_xy = asset.data.root_lin_vel_b[:, :2]
    moving_cmd = torch.norm(commands_xy, dim=1) > command_threshold
    stalled = torch.norm(base_vel_xy, dim=1) < velocity_threshold
    penalty = (moving_cmd & stalled).float()
    return _check_reward_finite(
        env,
        "moving_stall_penalty",
        penalty,
        {"commands_xy": commands_xy, "base_vel_xy": base_vel_xy, "moving_cmd": moving_cmd, "stalled": stalled},
    )


def _match_quadruped_phase_shape(phase_values: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Expand two gait phases to foot order FL, FR, RL, RR for a trot pattern."""
    if phase_values.shape[1] == target_dim:
        return phase_values

    if phase_values.shape[1] == 2 and target_dim == 4:
        return torch.stack(
            [
                phase_values[:, 0],
                phase_values[:, 1],
                phase_values[:, 1],
                phase_values[:, 0],
            ],
            dim=1,
        )

    if phase_values.shape[1] > target_dim:
        return phase_values[:, :target_dim]

    repeat_factor = (target_dim + phase_values.shape[1] - 1) // phase_values.shape[1]
    return phase_values.repeat(1, repeat_factor)[:, :target_dim]


_FOOT_HEIGHT_SCANNER_NAMES = (
    "fl_foot_height_scanner",
    "fr_foot_height_scanner",
    "rl_foot_height_scanner",
    "rr_foot_height_scanner",
)


def _get_foot_scanner_hits(env: ManagerBasedRLEnv, num_feet: int) -> torch.Tensor | None:
    """Return foot-centered terrain hits ordered FL, FR, RL, RR when available."""
    if num_feet != len(_FOOT_HEIGHT_SCANNER_NAMES):
        return None
    if not all(sensor_name in env.scene.sensors for sensor_name in _FOOT_HEIGHT_SCANNER_NAMES):
        return None
    return torch.stack(
        [env.scene.sensors[sensor_name].data.ray_hits_w for sensor_name in _FOOT_HEIGHT_SCANNER_NAMES],
        dim=1,
    )


def _nearest_terrain_z_from_foot_hits(feet_positions: torch.Tensor, foot_ray_hits_w: torch.Tensor) -> torch.Tensor:
    """Select the local terrain height closest to each foot in the matching foot scanner."""
    valid_hits = torch.isfinite(foot_ray_hits_w[..., 2])
    dist_sq = torch.sum(
        torch.square(feet_positions[..., :2].unsqueeze(2) - foot_ray_hits_w[..., :2]),
        dim=-1,
    )
    dist_sq = torch.where(valid_hits, dist_sq, torch.full_like(dist_sq, float("inf")))
    nearest_hit_idx = torch.argmin(dist_sq, dim=-1)
    terrain_z = torch.gather(foot_ray_hits_w[..., 2], 2, nearest_hit_idx.unsqueeze(-1)).squeeze(-1)
    return torch.where(torch.isfinite(terrain_z), terrain_z, torch.zeros_like(terrain_z))


def _nearest_terrain_z_from_body_hits(
    feet_positions: torch.Tensor, ray_hits_w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select each foot's nearest terrain hit from the body-centered height scanner."""
    valid_hits = torch.isfinite(ray_hits_w[..., 2])
    dist_sq = torch.sum(
        torch.square(feet_positions[..., :2].unsqueeze(2) - ray_hits_w[..., :2].unsqueeze(1)),
        dim=-1,
    )
    dist_sq = torch.where(valid_hits.unsqueeze(1), dist_sq, torch.full_like(dist_sq, float("inf")))
    nearest_hit_idx = torch.argmin(dist_sq, dim=-1)
    terrain_z = torch.gather(ray_hits_w[..., 2], 1, nearest_hit_idx)
    terrain_z = torch.where(torch.isfinite(terrain_z), terrain_z, torch.zeros_like(terrain_z))
    return terrain_z, nearest_hit_idx


def _split_front_rear_body_local_ids(asset: Articulation | RigidObject, body_ids: list[int]) -> tuple[list[int], list[int]]:
    """Split selected body ids into local front/rear indices using body names when possible."""
    front_tokens = {"FL", "FR", "FRONT"}
    rear_tokens = {"RL", "RR", "REAR"}
    front_local_ids = []
    rear_local_ids = []
    body_names = getattr(asset, "body_names", None) or getattr(asset.data, "body_names", None) or []

    for local_id, body_id in enumerate(body_ids):
        body_id = int(body_id)
        body_name = body_names[body_id] if body_id < len(body_names) else ""
        tokens = set(re.split(r"[^A-Za-z0-9]+", body_name.upper()))
        if tokens & front_tokens:
            front_local_ids.append(local_id)
        elif tokens & rear_tokens:
            rear_local_ids.append(local_id)

    if front_local_ids and rear_local_ids:
        return front_local_ids, rear_local_ids

    midpoint = len(body_ids) // 2
    return list(range(midpoint)), list(range(midpoint, len(body_ids)))


def _effective_foot_radius(env: ManagerBasedRLEnv, foot_radius: float, reference: torch.Tensor) -> torch.Tensor:
    """Return env-specific effective foot radius, falling back to the configured nominal value."""
    radius = getattr(env, "_foot_radius", foot_radius)
    if torch.is_tensor(radius):
        radius = radius.to(device=reference.device, dtype=reference.dtype)
        if radius.ndim == 0:
            radius = radius.expand(reference.shape[0])
        elif radius.shape[0] != reference.shape[0]:
            radius = radius.reshape(-1)[:1].expand(reference.shape[0])
    else:
        radius = torch.full((reference.shape[0],), float(radius), device=reference.device, dtype=reference.dtype)

    while radius.ndim < reference.ndim:
        radius = radius.unsqueeze(-1)
    return radius


def _terrain_refs_from_foot_hits(
    feet_positions: torch.Tensor,
    foot_ray_hits_w: torch.Tensor,
    move_direction: torch.Tensor,
    forward_window: float,
    lateral_window: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-foot nearest and forward-high terrain references from TIP scanners."""
    valid_hits = torch.isfinite(foot_ray_hits_w[..., 2])
    dist_sq = torch.sum(
        torch.square(feet_positions[..., :2].unsqueeze(2) - foot_ray_hits_w[..., :2]),
        dim=-1,
    )
    dist_sq = torch.where(valid_hits, dist_sq, torch.full_like(dist_sq, float("inf")))
    nearest_hit_idx = torch.argmin(dist_sq, dim=-1)
    nearest_hit_xyz = torch.gather(
        foot_ray_hits_w,
        2,
        nearest_hit_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3),
    ).squeeze(2)
    nearest_terrain_z = torch.where(
        torch.isfinite(nearest_hit_xyz[..., 2]),
        nearest_hit_xyz[..., 2],
        torch.zeros_like(nearest_hit_xyz[..., 2]),
    )
    nearest_hit_xyz = torch.where(
        torch.isfinite(nearest_hit_xyz),
        nearest_hit_xyz,
        feet_positions,
    )

    lateral_direction = torch.stack([-move_direction[:, 1], move_direction[:, 0]], dim=1)
    hit_delta_xy = foot_ray_hits_w[..., :2] - feet_positions[..., :2].unsqueeze(2)
    forward_dist = torch.sum(hit_delta_xy * move_direction[:, None, None, :], dim=-1)
    lateral_dist = torch.abs(torch.sum(hit_delta_xy * lateral_direction[:, None, None, :], dim=-1))
    forward_candidates = (
        valid_hits
        & (forward_dist > 0.0)
        & (forward_dist <= forward_window)
        & (lateral_dist <= lateral_window)
    )
    forward_hit_z = torch.where(
        forward_candidates,
        foot_ray_hits_w[..., 2],
        torch.full_like(foot_ray_hits_w[..., 2], -torch.inf),
    )
    forward_terrain_z = torch.max(forward_hit_z, dim=2).values
    highest_hit_idx = torch.argmax(forward_hit_z, dim=2)
    highest_hit_xyz = torch.gather(
        foot_ray_hits_w,
        2,
        highest_hit_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3),
    ).squeeze(2)
    highest_hit_xyz = torch.where(
        torch.isfinite(forward_terrain_z).unsqueeze(-1),
        highest_hit_xyz,
        nearest_hit_xyz,
    )
    forward_terrain_z = torch.where(
        torch.isfinite(forward_terrain_z),
        forward_terrain_z,
        nearest_terrain_z,
    )
    reference_terrain_z = torch.maximum(nearest_terrain_z, forward_terrain_z)
    return nearest_terrain_z, forward_terrain_z, reference_terrain_z, nearest_hit_xyz, highest_hit_xyz


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
    foot_ray_hits_w = _get_foot_scanner_hits(env, foot_positions.shape[1])
    terrain_z = torch.zeros_like(foot_positions[..., 2])
    if foot_ray_hits_w is not None:
        terrain_z = _nearest_terrain_z_from_foot_hits(foot_positions, foot_ray_hits_w)
    elif "height_scanner" in env.scene.sensors:
        height_scanner: RayCaster = env.scene.sensors["height_scanner"]
        terrain_z, _ = _nearest_terrain_z_from_body_hits(foot_positions, height_scanner.data.ray_hits_w)

    effective_foot_radius = _effective_foot_radius(env, foot_radius, foot_positions[..., 2])
    foot_heights = torch.clip(foot_positions[..., 2] - terrain_z - effective_foot_radius, 0, 1)

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
            "effective_foot_radius": effective_foot_radius,
        },
    )

def joint_powers_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint powers on the articulation using L1-kernel"""

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(torch.mul(asset.data.applied_torque, asset.data.joint_vel)), dim=1)


def _expand_joint_effort_limit(limit: torch.Tensor, torque: torch.Tensor) -> torch.Tensor | None:
    """Return effort limits with the same shape as applied torque when compatible."""
    limit = torch.as_tensor(limit, device=torque.device, dtype=torque.dtype)
    if limit.shape == torque.shape:
        expanded_limit = limit
    elif limit.ndim == 1 and limit.shape[0] == torque.shape[1]:
        expanded_limit = limit.unsqueeze(0).expand_as(torque)
    elif limit.ndim == 2 and limit.shape[0] == 1 and limit.shape[1] == torque.shape[1]:
        expanded_limit = limit.expand_as(torque)
    else:
        return None

    if torch.isfinite(expanded_limit).all() and torch.all(expanded_limit > 0.0):
        return expanded_limit
    return None


def _joint_effort_limits_from_asset_data(asset: Articulation, torque: torch.Tensor) -> torch.Tensor | None:
    """Read joint effort limits from Isaac Lab articulation data across API variants."""
    for attr_name in (
        "soft_joint_effort_limits",
        "joint_effort_limits",
        "default_joint_effort_limits",
        "joint_effort_limit",
    ):
        limit = getattr(asset.data, attr_name, None)
        if limit is None:
            continue
        expanded_limit = _expand_joint_effort_limit(limit, torque)
        if expanded_limit is not None:
            return expanded_limit
    return None


def _joint_effort_limits_from_actuator_cfg(asset: Articulation, torque: torch.Tensor) -> torch.Tensor | None:
    """Build joint effort limits from actuator config regex patterns as a fallback."""
    joint_names = getattr(asset, "joint_names", None)
    actuator_cfgs = getattr(getattr(asset, "cfg", None), "actuators", None)
    if not joint_names or not actuator_cfgs:
        return None

    effort_limits = torch.full((len(joint_names),), torch.nan, device=torque.device, dtype=torque.dtype)
    for actuator_cfg in actuator_cfgs.values():
        effort_limit = getattr(actuator_cfg, "effort_limit", None)
        if effort_limit is None:
            continue
        if isinstance(effort_limit, dict):
            for pattern, limit in effort_limit.items():
                for joint_id, joint_name in enumerate(joint_names):
                    if re.fullmatch(pattern, joint_name):
                        effort_limits[joint_id] = float(limit)
        else:
            effort_limits[:] = float(effort_limit)

    if torch.isfinite(effort_limits).all():
        return effort_limits.unsqueeze(0).expand_as(torque)
    return None


def _joint_effort_limits(asset: Articulation, torque: torch.Tensor) -> torch.Tensor:
    """Return positive joint effort limits matching applied torque shape."""
    effort_limits = _joint_effort_limits_from_actuator_cfg(asset, torque)
    if effort_limits is None:
        effort_limits = _joint_effort_limits_from_asset_data(asset, torque)
    if effort_limits is None:
        raise RuntimeError(
            "Unable to determine joint effort limits for combined_torque_penalty. "
            "Expected effort limits on asset.data or asset.cfg.actuators."
        )
    return torch.clamp(torch.abs(effort_limits), min=1.0e-6)


def combined_torque_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    torque_safe_ratio: float = 0.7,
    power: float = 6.0,
    high_usage_weight: float = 0.1,
) -> torch.Tensor:
    """Penalize torque only past a safe usage band, with extra pressure near effort limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    torque = torch.abs(asset.data.applied_torque)
    effort_limits = _joint_effort_limits(asset, torque)
    if asset_cfg.joint_ids is not None:
        torque = torque[:, asset_cfg.joint_ids]
        effort_limits = effort_limits[:, asset_cfg.joint_ids]
    torque_usage = torque / effort_limits

    soft_penalty = torch.sum(torch.square(torch.relu(torque_usage - torque_safe_ratio)), dim=1)
    high_usage_penalty = torch.sum(torch.pow(torque_usage, power), dim=1)
    reward = soft_penalty + high_usage_weight * high_usage_penalty
    return _check_reward_finite(
        env,
        "combined_torque_penalty",
        reward,
        {
            "torque": torque,
            "effort_limits": effort_limits,
            "torque_usage": torque_usage,
            "soft_penalty": soft_penalty,
            "high_usage_penalty": high_usage_penalty,
        },
    )


class JointTorqueRateExcessPenalty(ManagerTermBase):
    """Penalize only large per-step torque changes for selected joints."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.prev_torque = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        threshold: float,
        max_value: float | None = None,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        torque_all = asset.data.applied_torque
        effort_limits_all = _joint_effort_limits(asset, torch.abs(torque_all))

        if asset_cfg.joint_ids is not None:
            torque = torque_all[:, asset_cfg.joint_ids]
            effort_limits = effort_limits_all[:, asset_cfg.joint_ids]
        else:
            torque = torque_all
            effort_limits = effort_limits_all

        if self.prev_torque is None or self.prev_torque.shape != torque.shape:
            self.prev_torque = torque.detach().clone()
            return torch.zeros(torque.shape[0], device=torque.device)

        delta_usage = torch.abs(torque - self.prev_torque) / torch.clamp(effort_limits, min=1.0e-6)
        excess = torch.relu(delta_usage - threshold)
        penalty = torch.sum(torch.square(excess), dim=1)
        if max_value is not None:
            penalty = torch.clamp(penalty, max=max_value)

        startup_env_mask = env.episode_length_buf < 3
        penalty[startup_env_mask] = 0.0
        self.prev_torque = torque.detach().clone()

        return _check_reward_finite(
            env,
            "JointTorqueRateExcessPenalty",
            penalty,
            {"torque": torque, "effort_limits": effort_limits, "delta_usage": delta_usage, "excess": excess},
        )


class ObstacleContactRecoveryReward(ManagerTermBase):
    """Give staged recovery credit after a high-confidence obstacle contact."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        num_envs = env.num_envs
        device = env.device
        self.contact_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.pending = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.release_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.release_pos = torch.zeros(num_envs, 2, device=device)
        self.cooldown = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.short_paid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.episode_credit = torch.zeros(num_envs, device=device)
        self.recontact_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.contact_steps[env_ids] = 0
        self.pending[env_ids] = False
        self.release_steps[env_ids] = 0
        self.release_pos[env_ids] = 0.0
        self.cooldown[env_ids] = 0
        self.short_paid[env_ids] = False
        self.episode_credit[env_ids] = 0.0
        self.recontact_steps[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        force_threshold: float,
        tip_horizontal_ratio: float,
        tip_stall_speed: float,
        minimum_contact_steps: int,
        recovery_window_steps: int,
        short_distance: float,
        full_distance: float,
        release_credit: float,
        short_credit: float,
        full_credit: float,
        episode_credit_cap: float,
        recontact_tolerance_steps: int,
        cooldown_steps: int,
        command_threshold: float,
    ) -> torch.Tensor:
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        asset: Articulation = env.scene[asset_cfg.name]
        if isinstance(sensor_cfg.body_ids, slice):
            selected_ids = torch.arange(len(sensor.body_names), device=env.device)
        else:
            selected_ids = torch.as_tensor(sensor_cfg.body_ids, device=env.device, dtype=torch.long)
        selected_names = [sensor.body_names[index] for index in selected_ids.tolist()]
        forces = sensor.data.net_forces_w[:, selected_ids, :]
        tip_local = [index for index, name in enumerate(selected_names) if "TIP" in name.upper()]
        leg_local = [index for index, name in enumerate(selected_names) if "CALF" in name.upper() or "THIGH" in name.upper()]
        tip_force = forces[:, tip_local, :] if tip_local else forces[:, :0, :]
        leg_force = forces[:, leg_local, :] if leg_local else forces[:, :0, :]
        tip_horizontal = torch.linalg.vector_norm(tip_force[..., :2], dim=-1)
        tip_vertical = torch.abs(tip_force[..., 2])
        tip_ids, _ = asset.find_bodies(".*TIP")
        tip_speed = torch.linalg.vector_norm(asset.data.body_lin_vel_w[:, tip_ids, :2], dim=-1)
        tip_count = min(tip_horizontal.shape[1], tip_speed.shape[1])
        tip_obstacle = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if tip_count > 0:
            horizontal_hit = tip_horizontal[:, :tip_count] > force_threshold
            horizontal_dominant = tip_horizontal[:, :tip_count] > tip_horizontal_ratio * tip_vertical[:, :tip_count]
            stalled = tip_speed[:, :tip_count] < tip_stall_speed
            tip_obstacle = torch.any(horizontal_hit & horizontal_dominant & stalled, dim=1)
        leg_contact = torch.any(torch.linalg.vector_norm(leg_force, dim=-1) > force_threshold, dim=1) if leg_local else torch.zeros_like(tip_obstacle)
        contact = tip_obstacle | leg_contact
        command = env.command_manager.get_command(command_name)[:, :2]
        speed = torch.linalg.vector_norm(command, dim=1)
        moving = speed > command_threshold
        direction = command / speed.unsqueeze(1).clamp(min=1.0e-6)

        self.cooldown.clamp_min_(0)
        self.cooldown.sub_(1).clamp_min_(0)
        active_contact = contact & moving & ~self.pending & (self.cooldown == 0)
        self.contact_steps = torch.where(active_contact, self.contact_steps + 1, self.contact_steps)
        released = ~contact & moving & ~self.pending & (self.contact_steps >= minimum_contact_steps)
        self.pending |= released
        self.release_steps = torch.where(released, torch.zeros_like(self.release_steps), self.release_steps)
        self.release_pos[released] = asset.data.root_pos_w[released, :2]
        self.short_paid[released] = False
        self.contact_steps[released] = 0
        idle_release = ~contact & ~self.pending & ~released
        self.contact_steps[idle_release] = 0
        self.release_steps = torch.where(self.pending, self.release_steps + 1, self.release_steps)

        displacement = asset.data.root_pos_w[:, :2] - self.release_pos
        path_distance = torch.linalg.vector_norm(displacement, dim=1)
        progress = torch.sum(displacement * direction, dim=1)
        self.recontact_steps = torch.where(self.pending & contact, self.recontact_steps + 1, torch.zeros_like(self.recontact_steps))
        reward = released.float() * release_credit
        short = self.pending & ~self.short_paid & (path_distance >= short_distance) & (progress >= 0.5 * short_distance)
        reward += short.float() * short_credit
        self.short_paid |= short
        recovered = self.pending & (path_distance >= full_distance) & (progress >= 0.5 * full_distance)
        reward += recovered.float() * full_credit
        expired = self.pending & ((self.release_steps > recovery_window_steps) | ~moving | (self.recontact_steps > recontact_tolerance_steps))
        finished = recovered | expired
        self.pending[finished] = False
        self.release_steps[finished] = 0
        self.recontact_steps[finished] = 0
        self.cooldown[finished] = cooldown_steps
        remaining = torch.clamp(episode_credit_cap - self.episode_credit, min=0.0)
        reward = torch.minimum(reward, remaining)
        self.episode_credit += reward
        reward[env.episode_length_buf < 3] = 0.0
        return reward


def _split_front_rear_joint_ids(joint_ids: list[int], joint_names: list[str]) -> tuple[list[int], list[int]]:
    front_tokens = {"FL", "FR"}
    rear_tokens = {"RL", "RR"}
    front_ids = []
    rear_ids = []

    for joint_id, joint_name in zip(joint_ids, joint_names):
        tokens = set(re.split(r"[^A-Za-z0-9]+", joint_name.upper()))
        if tokens & front_tokens:
            front_ids.append(joint_id)
        elif tokens & rear_tokens:
            rear_ids.append(joint_id)

    if len(front_ids) >= 2 and len(rear_ids) >= 2:
        return front_ids, rear_ids

    midpoint = len(joint_ids) // 2
    return joint_ids[:midpoint], joint_ids[midpoint:]


def joint_powers_var(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_group_names: tuple[str, ...] = (".*HR_JOINT", ".*HP_JOINT", ".*KN_JOINT"),
) -> torch.Tensor:
    """Penalize front/rear left-right power imbalance within each joint type."""

    asset: Articulation = env.scene[asset_cfg.name]
    joint_power = torch.abs(asset.data.applied_torque * asset.data.joint_vel)

    penalty = torch.zeros(env.num_envs, device=env.device)
    for joint_name in joint_group_names:
        joint_ids, joint_names = asset.find_joints(joint_name)
        if len(joint_ids) < 2:
            continue
        front_ids, rear_ids = _split_front_rear_joint_ids(joint_ids, joint_names)
        if len(front_ids) >= 2:
            penalty += torch.var(joint_power[:, front_ids], dim=1, unbiased=False)
        if len(rear_ids) >= 2:
            penalty += torch.var(joint_power[:, rear_ids], dim=1, unbiased=False)

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


def terrain_adaptive_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*TIP"),
    sensor_cfg: SceneEntityCfg | None = SceneEntityCfg("height_scanner"),
    pitch_gain: float = 0.7,
    max_pitch: float = 0.314,
    slope_deadband: float = 0.02,
    min_front_rear_distance: float = 0.25,
) -> torch.Tensor:
    """Penalize body tilt away from the local front/rear terrain slope.

    The target orientation is derived from terrain height under the feet, not the
    foot body z-position, so swing feet do not directly tilt the target.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    base_quat = getattr(asset.data, "root_quat_w", None)
    if base_quat is None:
        base_quat = asset.data.root_link_quat_w
    up_axis_b = torch.zeros((env.num_envs, 3), device=env.device, dtype=base_quat.dtype)
    up_axis_b[:, 2] = 1.0
    body_up_w = math_utils.quat_apply(base_quat, up_axis_b)
    body_up_w = body_up_w / torch.linalg.norm(body_up_w, dim=1, keepdim=True).clamp(min=1.0e-6)

    body_ids = asset_cfg.body_ids
    if asset_cfg.body_names is not None and (body_ids is None or body_ids == slice(None)):
        body_ids, _ = asset.find_bodies(asset_cfg.body_names)
    elif body_ids is None:
        body_ids = list(range(len(asset.body_names)))
    elif isinstance(body_ids, int):
        body_ids = [body_ids]
    elif torch.is_tensor(body_ids):
        body_ids = body_ids.detach().cpu().tolist()
    elif isinstance(body_ids, slice):
        body_ids = list(range(len(asset.body_names)))[body_ids]
    else:
        body_ids = list(body_ids)

    if len(body_ids) < 2:
        reward = torch.sum(torch.square(body_up_w[:, :2]), dim=1)
        return _check_reward_finite(env, "terrain_adaptive_orientation_l2", reward, {"body_up_w": body_up_w})

    feet_positions = asset.data.body_pos_w[:, body_ids, :]
    terrain_z = torch.zeros_like(feet_positions[..., 2])
    foot_ray_hits_w = _get_foot_scanner_hits(env, feet_positions.shape[1])
    if foot_ray_hits_w is not None:
        terrain_z = _nearest_terrain_z_from_foot_hits(feet_positions, foot_ray_hits_w)
    elif sensor_cfg is not None and sensor_cfg.name in env.scene.sensors:
        height_scanner: RayCaster = env.scene.sensors[sensor_cfg.name]
        terrain_z, _ = _nearest_terrain_z_from_body_hits(feet_positions, height_scanner.data.ray_hits_w)

    front_local_ids, rear_local_ids = _split_front_rear_body_local_ids(asset, body_ids)
    if not front_local_ids or not rear_local_ids:
        reward = torch.sum(torch.square(body_up_w[:, :2]), dim=1)
        return _check_reward_finite(
            env,
            "terrain_adaptive_orientation_l2",
            reward,
            {"body_up_w": body_up_w, "terrain_z": terrain_z},
        )

    front_ids = torch.as_tensor(front_local_ids, device=env.device, dtype=torch.long)
    rear_ids = torch.as_tensor(rear_local_ids, device=env.device, dtype=torch.long)
    front_xy = feet_positions.index_select(1, front_ids)[..., :2].mean(dim=1)
    rear_xy = feet_positions.index_select(1, rear_ids)[..., :2].mean(dim=1)
    front_z = terrain_z.index_select(1, front_ids).mean(dim=1)
    rear_z = terrain_z.index_select(1, rear_ids).mean(dim=1)

    front_rear_xy = front_xy - rear_xy
    front_rear_distance = torch.linalg.norm(front_rear_xy, dim=1).clamp(min=min_front_rear_distance)
    forward_dir_xy = front_rear_xy / torch.linalg.norm(front_rear_xy, dim=1, keepdim=True).clamp(min=1.0e-6)
    terrain_slope = (front_z - rear_z) / front_rear_distance
    if slope_deadband > 0.0:
        terrain_slope = torch.sign(terrain_slope) * torch.clamp(torch.abs(terrain_slope) - slope_deadband, min=0.0)

    target_pitch = torch.clamp(pitch_gain * torch.atan(terrain_slope), min=-max_pitch, max=max_pitch)
    target_slope = torch.tan(target_pitch)
    desired_up_w = torch.cat(
        (
            -target_slope.unsqueeze(1) * forward_dir_xy,
            torch.ones_like(target_slope).unsqueeze(1),
        ),
        dim=1,
    )
    desired_up_w = desired_up_w / torch.linalg.norm(desired_up_w, dim=1, keepdim=True).clamp(min=1.0e-6)

    reward = torch.sum(torch.square(body_up_w[:, :2] - desired_up_w[:, :2]), dim=1)
    return _check_reward_finite(
        env,
        "terrain_adaptive_orientation_l2",
        reward,
        {
            "body_up_w": body_up_w,
            "desired_up_w": desired_up_w,
            "terrain_z": terrain_z,
            "terrain_slope": terrain_slope,
            "target_pitch": target_pitch,
        },
    )


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


def standing_foot_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    lin_threshold: float = 0.05,
    ang_threshold: float = 0.05,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize missing foot contacts while the base velocity command is near zero."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    commands = env.command_manager.get_command("base_velocity")

    standing_mask = (
        (torch.norm(commands[:, :2], dim=1) < lin_threshold)
        & (torch.abs(commands[:, 2]) < ang_threshold)
    ).float()

    foot_forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    contacts = foot_forces > force_threshold
    missing_contacts = (~contacts).float().sum(dim=1)
    reward = missing_contacts * standing_mask

    return _check_reward_finite(
        env,
        "standing_foot_contact",
        reward,
        {
            "commands": commands,
            "standing_mask": standing_mask,
            "foot_forces": foot_forces,
            "missing_contacts": missing_contacts,
        },
    )


def standing_foot_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    foot_radius: float,
    lin_threshold: float = 0.05,
    ang_threshold: float = 0.05,
    height_tolerance: float = 0.02,
) -> torch.Tensor:
    """Penalize feet lifted above the local terrain while the base velocity command is near zero."""
    asset: Articulation = env.scene[asset_cfg.name]
    commands = env.command_manager.get_command("base_velocity")

    standing_mask = (
        (torch.norm(commands[:, :2], dim=1) < lin_threshold)
        & (torch.abs(commands[:, 2]) < ang_threshold)
    ).float()

    feet_positions = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_ray_hits_w = _get_foot_scanner_hits(env, feet_positions.shape[1])
    terrain_z = torch.zeros_like(feet_positions[..., 2])

    if foot_ray_hits_w is not None:
        terrain_z = _nearest_terrain_z_from_foot_hits(feet_positions, foot_ray_hits_w)
    elif sensor_cfg.name in env.scene.sensors:
        height_scanner: RayCaster = env.scene.sensors[sensor_cfg.name]
        terrain_z, _ = _nearest_terrain_z_from_body_hits(feet_positions, height_scanner.data.ray_hits_w)

    effective_foot_radius = _effective_foot_radius(env, foot_radius, feet_positions[..., 2])
    feet_height = torch.clamp(feet_positions[..., 2] - terrain_z - effective_foot_radius, min=0.0)
    height_error = torch.clamp(feet_height - height_tolerance, min=0.0)
    reward = torch.sum(torch.square(height_error), dim=1) * standing_mask

    return _check_reward_finite(
        env,
        "standing_foot_height",
        reward,
        {
            "commands": commands,
            "standing_mask": standing_mask,
            "feet_positions": feet_positions,
            "terrain_z": terrain_z,
            "feet_height": feet_height,
            "height_error": height_error,
            "effective_foot_radius": effective_foot_radius,
        },
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


def foot_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    p_des: float,
    foot_radius: float = 0.0,
) -> torch.Tensor:
    """Penalize foot clearance error weighted by horizontal foot speed.

    Computes sum_over_feet((p_des - p)^2 * v_xy), where p is the foot height above terrain.
    """
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

    # p: [num_envs, num_feet], z-axis distance from terrain to the bottom of each foot.
    effective_foot_radius = _effective_foot_radius(env, foot_radius, feet_positions[..., 2])
    p = torch.clamp(feet_positions[..., 2] - terrain_z - effective_foot_radius, min=0.0)
    # v_xy: [num_envs, num_feet], horizontal foot speed magnitude in world frame.
    v_xy = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    reward = torch.sum(torch.square(p_des - p) * v_xy, dim=1)

    return _check_reward_finite(
        env,
        "foot_clearance",
        reward,
        {
            "feet_positions": feet_positions,
            "terrain_z": terrain_z if torch.is_tensor(terrain_z) else torch.tensor(terrain_z, device=reward.device),
            "foot_height": p,
            "foot_vel_xy": v_xy,
            "effective_foot_radius": effective_foot_radius,
        },
    )


def pen_swing_height_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    foot_radius: float,
) -> torch.Tensor:
    """Penalize insufficient swing-foot clearance using a forward-aware terrain reference."""
    _maybe_visualize_foot_height_scanners(env)

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
    cmd_xy = base_velocity_cmd[:, :2]
    cmd_xy_norm = torch.norm(cmd_xy, dim=1, keepdim=True)
    cmd_direction = cmd_xy / torch.clamp(cmd_xy_norm, min=1.0e-6)
    body_forward = math_utils.quat_apply_yaw(
        asset.data.root_quat_w, torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    )[:, :2]
    body_forward = body_forward / torch.clamp(torch.norm(body_forward, dim=1, keepdim=True), min=1.0e-6)
    use_body_forward = cmd_xy_norm.squeeze(-1) <= 0.1
    move_direction = torch.where(use_body_forward.unsqueeze(-1), body_forward, cmd_direction)
    foot_ray_hits_w = _get_foot_scanner_hits(env, feet_positions.shape[1])

    nearest_terrain_z = torch.zeros_like(feet_positions[..., 2])
    forward_terrain_z = torch.zeros_like(feet_positions[..., 2])
    reference_terrain_z = torch.zeros_like(feet_positions[..., 2])
    nearest_hit_xyz = feet_positions.detach().clone()
    highest_hit_xyz = feet_positions.detach().clone()
    if foot_ray_hits_w is not None:
        (
            nearest_terrain_z,
            forward_terrain_z,
            reference_terrain_z,
            nearest_hit_xyz,
            highest_hit_xyz,
        ) = _terrain_refs_from_foot_hits(
            feet_positions,
            foot_ray_hits_w,
            move_direction,
            forward_window,
            lateral_window,
        )
    elif "height_scanner" in env.scene.sensors:
        height_scanner: RayCaster = env.scene.sensors["height_scanner"]
        ray_hits_w = height_scanner.data.ray_hits_w
        valid_hits = torch.isfinite(ray_hits_w[..., 2])
        if torch.any(valid_hits):
            nearest_terrain_z, nearest_hit_idx = _nearest_terrain_z_from_body_hits(feet_positions, ray_hits_w)
            ray_hits_for_feet = ray_hits_w.unsqueeze(1).expand(-1, feet_positions.shape[1], -1, -1)
            nearest_hit_xyz = torch.gather(
                ray_hits_for_feet,
                2,
                nearest_hit_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3),
            ).squeeze(2)

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
            highest_hit_idx = torch.argmax(forward_hit_z, dim=2)
            highest_hit_xyz = torch.gather(
                ray_hits_for_feet,
                2,
                highest_hit_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3),
            ).squeeze(2)
            highest_hit_xyz = torch.where(
                torch.isfinite(forward_terrain_z).unsqueeze(-1),
                highest_hit_xyz,
                nearest_hit_xyz,
            )
            forward_terrain_z = torch.where(
                torch.isfinite(forward_terrain_z), forward_terrain_z, nearest_terrain_z
            )
            reference_terrain_z = torch.maximum(nearest_terrain_z, forward_terrain_z)

    if not torch.any(reference_terrain_z):
        reference_terrain_z = nearest_terrain_z
    effective_foot_radius = _effective_foot_radius(env, foot_radius, feet_positions[..., 2])
    feet_height = torch.clamp(feet_positions[..., 2] - nearest_terrain_z - effective_foot_radius, min=0.0)
    step_rise = torch.clamp(reference_terrain_z - nearest_terrain_z, min=0.0)
    # swing_height_target = gait_params[:, 3:4].expand(-1, feet_height.shape[1]) + clearance_margin + step_rise
    base_clearance = gait_params[:, 3:4].expand(-1, feet_height.shape[1])
    terrain_clearance = step_rise + clearance_margin
    swing_height_target = torch.maximum(base_clearance, terrain_clearance)
    swing_height_target = torch.clamp(swing_height_target, max=0.24)

    swing_mask = _match_quadruped_phase_shape(swing_mask, feet_height.shape[1])

    _maybe_visualize_swing_height_debug(
        env,
        nearest_hit_xyz=nearest_hit_xyz,
        highest_hit_xyz=highest_hit_xyz,
        feet_positions=feet_positions,
        nearest_terrain_z=nearest_terrain_z,
        swing_height_target=swing_height_target,
        foot_radius=effective_foot_radius,
    )

    clearance_error = torch.clamp(swing_height_target - feet_height, min=0.0)
    height_error = torch.square(clearance_error)
    reward = torch.sum(torch.where(swing_mask, height_error, torch.zeros_like(height_error)), dim=1)
    moving_cmd = (torch.norm(base_velocity_cmd[:, :2], dim=1) > 0.1) | (torch.abs(base_velocity_cmd[:, 2]) > 0.1)
    reward *= moving_cmd

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
            "effective_foot_radius": effective_foot_radius,
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


def base_com_height_dreamwaq(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize base CoM height with a DreamWaQ-style L2 term.

    Note:
        Uses an adaptive target height:
        ``adaptive_target = target_height + terrain_height``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # [num_envs, num_rays] terrain hit heights from the height scanner.
        ray_hits_z = sensor.data.ray_hits_w[..., 2]
        finite_hits = torch.isfinite(ray_hits_z)
        sanitized_hits = torch.where(finite_hits, ray_hits_z, torch.zeros_like(ray_hits_z))
        hit_counts = finite_hits.sum(dim=1).clamp(min=1)
        terrain_height = sanitized_hits.sum(dim=1) / hit_counts
    else:
        terrain_height = torch.zeros_like(base_height)
        ray_hits_z = None

    adaptive_target = target_height + terrain_height
    reward = torch.square(base_height - adaptive_target)
    extra = {
        "base_height": base_height,
        "terrain_height": terrain_height,
        "adaptive_target": adaptive_target,
    }
    if ray_hits_z is not None:
        extra["ray_hits_z"] = ray_hits_z
    return _check_reward_finite(env, "base_com_height_dreamwaq", reward, extra)


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
        self.stance_force_scale = float(cfg.params.get("tracking_contacts_stance_force", 0.0))
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
        tracking_contacts_stance_force=None,
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
        standing_cmd = (torch.norm(base_velocity_cmd[:, :2], dim=1) <= 0.1) & (
            torch.abs(base_velocity_cmd[:, 2]) <= 0.1
        )

        # Update contact targets
        desired_contact_states = self.compute_contact_targets(gait_params)

        # Force-based reward
        foot_forces = torch.norm(self.contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids], dim=-1)
        desired_contact_states = self._match_contact_shape(desired_contact_states, foot_forces.shape[1])
        desired_contact_states = torch.where(
            standing_cmd.unsqueeze(1), torch.ones_like(desired_contact_states), desired_contact_states
        )
        force_reward = self._compute_force_reward(foot_forces, desired_contact_states)
        stance_force_reward = self._compute_stance_force_reward(foot_forces, desired_contact_states)

        # Velocity-based reward
        foot_velocities = torch.norm(self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids], dim=-1)
        desired_contact_states = self._match_contact_shape(desired_contact_states, foot_velocities.shape[1])
        velocity_reward = self._compute_velocity_reward(foot_velocities, desired_contact_states)

        # Combine rewards
        total_reward = force_reward + stance_force_reward + velocity_reward
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
                "stance_force_reward": stance_force_reward,
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

        return _match_quadruped_phase_shape(desired_contacts, target_dim)

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

    def _compute_stance_force_reward(self, forces: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """Penalize missing normal force when a foot is commanded to be in stance."""
        if self.stance_force_scale == 0.0:
            return torch.zeros_like(forces[:, 0])

        missing_contact = desired_contacts * torch.exp(-forces**2 / self.force_sigma)
        return torch.mean(missing_contact, dim=1) * self.stance_force_scale

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
