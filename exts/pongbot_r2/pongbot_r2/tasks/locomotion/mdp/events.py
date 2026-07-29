"""Custom event functions for PongBot R2 locomotion environments.

@version 0.0.1
@update 2026-07-10: Add configurable velocity-command deadbands for implicit rough training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

def prepare_quantity_for_tron(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_radius = 0.127,
):
    device = getattr(env, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    env._foot_radius = torch.full((env.num_envs,), float(foot_radius), device=device)


def randomize_foot_radius(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    radius_range: tuple[float, float] = (0.030, 0.037),
):
    """Randomize the effective foot radius used by foot-height rewards."""
    device = getattr(env, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=device)
    else:
        env_ids = env_ids.to(device=device)

    if not hasattr(env, "_foot_radius") or not torch.is_tensor(env._foot_radius):
        env._foot_radius = torch.full(
            (env.num_envs,),
            0.5 * (radius_range[0] + radius_range[1]),
            device=device,
        )
    elif env._foot_radius.shape[0] != env.num_envs:
        env._foot_radius = env._foot_radius.reshape(-1)[:1].expand(env.num_envs).clone().to(device=device)

    env._foot_radius[env_ids] = math_utils.sample_uniform(
        radius_range[0], radius_range[1], (len(env_ids),), device=device
    )

def apply_external_force_torque_stochastic(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: dict[str, tuple[float, float]],
    torque_range: dict[str, tuple[float, float]],
    probability: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the external forces and torques applied to the bodies.

    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
    and torques is equal to the number of bodies times the number of environments. The forces and torques are
    applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called in the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # clear the existing forces and torques
    asset._external_force_b *= 0
    asset._external_torque_b *= 0

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    random_values = torch.rand(env_ids.shape, device=env_ids.device)
    mask = random_values < probability
    masked_env_ids = env_ids[mask]

    if len(masked_env_ids) == 0:
        return

    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    # sample random forces and torques
    size = (len(masked_env_ids), num_bodies, 3)
    force_range_list = [force_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    force_range = torch.tensor(force_range_list, device=asset.device)
    forces = math_utils.sample_uniform(force_range[:, 0], force_range[:, 1], size, asset.device)
    torque_range_list = [torque_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    torque_range = torch.tensor(torque_range_list, device=asset.device)
    torques = math_utils.sample_uniform(torque_range[:, 0], torque_range[:, 1], size, asset.device)
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    asset.set_external_force_and_torque(forces, torques, env_ids=masked_env_ids, body_ids=asset_cfg.body_ids)


def randomize_rigid_body_mass_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the inertia of the bodies by adding, scaling, or setting random values.

    This function allows randomizing the mass of the bodies of the asset. The function samples random values from the
    given distribution parameters and adds, scales, or sets the values into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body masses. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current inertias of the bodies (num_assets, num_bodies)
    inertias = asset.root_physx_view.get_inertias().clone()
    masses = asset.root_physx_view.get_masses().clone()

    masses = _randomize_prop_by_op(
        masses, mass_inertia_distribution_params, env_ids, body_ids, operation=operation, distribution=distribution
    )
    scale = masses / asset.root_physx_view.get_masses()
    inertias *= scale.unsqueeze(-1)

    asset.root_physx_view.set_masses(masses, env_ids)
    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_rigid_body_coms(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_distribution_params: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the center of mass (COM) of the bodies by adding, scaling, or setting random values for each dimension.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    coms = asset.root_physx_view.get_coms().clone()

    # Apply randomization to each dimension separately
    for dim in range(3):  # 0=x, 1=y, 2=z
        coms[..., dim] = _randomize_prop_by_op(
            coms[..., dim],
            com_distribution_params[dim],
            env_ids,
            body_ids,
            operation=operation,
            distribution=distribution,
        )

    asset.root_physx_view.set_coms(coms, env_ids)


def set_zero_command(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    command_name: str = "base_velocity",
    duration_s: float = 2.0,
):
    """Set the selected command term to zero and hold it for the given duration."""
    command_term = env.command_manager.get_term(command_name)

    if env_ids is None:
        command_term.command.zero_()
        command_term.time_left.fill_(duration_s)
        return

    env_ids = env_ids.to(device=command_term.command.device)
    command_term.command[env_ids] = 0.0
    command_term.time_left[env_ids] = duration_s


def zero_small_velocity_commands(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    command_name: str = "base_velocity",
    linear_threshold: float = 0.2,
    angular_threshold: float = 0.2,
) -> None:
    """Apply planar-linear and yaw deadbands to a velocity command."""
    command = env.command_manager.get_term(command_name).command
    if env_ids is None or isinstance(env_ids, slice):
        selected_ids = torch.arange(env.num_envs, device=command.device)
    else:
        selected_ids = torch.as_tensor(env_ids, device=command.device, dtype=torch.long).view(-1)

    selected_command = command[selected_ids]
    linear_mask = torch.linalg.vector_norm(selected_command[:, :2], dim=1) <= linear_threshold
    angular_mask = torch.abs(selected_command[:, 2]) <= angular_threshold
    command[selected_ids[linear_mask], :2] = 0.0
    command[selected_ids[angular_mask], 2] = 0.0


def align_root_yaw_on_terrain_types(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    terrain_type_names: tuple[str, ...] = (),
    yaw_range: tuple[float, float] = (-0.1, 0.1),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Align the root yaw near terrain forward direction for selected terrain types on reset."""
    if not terrain_type_names:
        return

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    device = asset.device

    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=device)
    else:
        env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    terrain = getattr(env.scene, "terrain", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    if terrain_types is None:
        return
    terrain_types = terrain_types.to(device=device, dtype=torch.long).view(-1)
    if terrain_types.numel() < env.num_envs:
        return
    terrain_types = terrain_types[: env.num_envs]

    terrain_generator = getattr(getattr(env.cfg.scene, "terrain", None), "terrain_generator", None)
    selected_type_ids = _terrain_column_ids_for_names(terrain_generator, terrain_type_names)
    if not selected_type_ids:
        return

    selected_ids = torch.tensor(selected_type_ids, device=device, dtype=torch.long)
    terrain_mask = torch.any(terrain_types[env_ids].unsqueeze(1) == selected_ids.unsqueeze(0), dim=1)
    selected_env_ids = env_ids[terrain_mask]
    if selected_env_ids.numel() == 0:
        return

    yaw = math_utils.sample_uniform(
        yaw_range[0],
        yaw_range[1],
        (selected_env_ids.numel(),),
        device=device,
    )
    zeros = torch.zeros_like(yaw)
    yaw_delta = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)

    root_pose = asset.data.root_state_w[selected_env_ids, :7].clone()
    default_root_quat = asset.data.default_root_state[selected_env_ids, 3:7]
    root_pose[:, 3:7] = math_utils.quat_mul(default_root_quat, yaw_delta)
    asset.write_root_pose_to_sim(root_pose, env_ids=selected_env_ids)


def limit_commands_on_terrain_types(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    command_name: str = "base_velocity",
    terrain_type_names: tuple[str, ...] = (),
    initial_lin_vel_x: tuple[float, float] = (-0.8, 0.8),
    initial_lin_vel_y: tuple[float, float] = (-0.3, 0.3),
    initial_ang_vel_z: tuple[float, float] = (-0.5, 0.5),
    start_step: int = 0,
    end_step: int = 80_000,
):
    """Clamp commands for selected terrain types early in training, then release to the configured range."""
    if not terrain_type_names:
        return

    command_term = env.command_manager.get_term(command_name)
    command = command_term.command
    device = command.device

    progress = 1.0
    if end_step > start_step:
        progress = (float(env.common_step_counter) - float(start_step)) / float(end_step - start_step)
        progress = float(torch.clamp(torch.tensor(progress, device=device), 0.0, 1.0).item())
    if progress >= 1.0:
        return

    terrain = getattr(env.scene, "terrain", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    if terrain_types is None:
        return
    terrain_types = terrain_types.to(device=device, dtype=torch.long).view(-1)
    if terrain_types.numel() < env.num_envs:
        return
    terrain_types = terrain_types[: env.num_envs]

    terrain_generator = getattr(getattr(env.cfg.scene, "terrain", None), "terrain_generator", None)
    selected_type_ids = _terrain_column_ids_for_names(terrain_generator, terrain_type_names)
    if not selected_type_ids:
        return

    selected_ids = torch.tensor(selected_type_ids, device=device, dtype=torch.long)
    hard_mask = torch.any(terrain_types.unsqueeze(1) == selected_ids.unsqueeze(0), dim=1)
    if env_ids is not None:
        active_mask = torch.zeros(env.num_envs, device=device, dtype=torch.bool)
        active_mask[env_ids.to(device=device, dtype=torch.long)] = True
        hard_mask &= active_mask
    if not torch.any(hard_mask):
        return

    ranges = command_term.cfg.ranges

    def _interpolate_range(initial_range: tuple[float, float], final_range: tuple[float, float]) -> tuple[float, float]:
        lower = initial_range[0] + (final_range[0] - initial_range[0]) * progress
        upper = initial_range[1] + (final_range[1] - initial_range[1]) * progress
        return lower, upper

    lin_x = _interpolate_range(initial_lin_vel_x, ranges.lin_vel_x)
    lin_y = _interpolate_range(initial_lin_vel_y, ranges.lin_vel_y)
    yaw = _interpolate_range(initial_ang_vel_z, ranges.ang_vel_z)

    command[hard_mask, 0] = torch.clamp(command[hard_mask, 0], lin_x[0], lin_x[1])
    command[hard_mask, 1] = torch.clamp(command[hard_mask, 1], lin_y[0], lin_y[1])
    command[hard_mask, 2] = torch.clamp(command[hard_mask, 2], yaw[0], yaw[1])


def limit_gait_on_terrain_types(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    command_name: str = "gait_command",
    terrain_type_names: tuple[str, ...] = (),
    initial_frequencies: tuple[float, float] = (1.3, 1.7),
    initial_durations: tuple[float, float] = (0.55, 0.65),
    initial_swing_height: tuple[float, float] = (0.10, 0.15),
    start_step: int = 0,
    end_step: int = 80_000,
):
    """Clamp gait commands on selected terrains early in training, then release them."""
    if not terrain_type_names:
        return

    command_term = env.command_manager.get_term(command_name)
    command = command_term.command
    if command.shape[1] < 4:
        return
    device = command.device

    progress = 1.0
    if end_step > start_step:
        progress = (float(env.common_step_counter) - float(start_step)) / float(end_step - start_step)
        progress = float(torch.clamp(torch.tensor(progress, device=device), 0.0, 1.0).item())
    if progress >= 1.0:
        return

    terrain = getattr(env.scene, "terrain", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    if terrain_types is None:
        return
    terrain_types = terrain_types.to(device=device, dtype=torch.long).view(-1)
    if terrain_types.numel() < env.num_envs:
        return
    terrain_types = terrain_types[: env.num_envs]

    terrain_generator = getattr(getattr(env.cfg.scene, "terrain", None), "terrain_generator", None)
    selected_type_ids = _terrain_column_ids_for_names(terrain_generator, terrain_type_names)
    if not selected_type_ids:
        return

    selected_ids = torch.tensor(selected_type_ids, device=device, dtype=torch.long)
    terrain_mask = torch.any(terrain_types.unsqueeze(1) == selected_ids.unsqueeze(0), dim=1)
    if env_ids is not None:
        active_mask = torch.zeros(env.num_envs, device=device, dtype=torch.bool)
        active_mask[env_ids.to(device=device, dtype=torch.long)] = True
        terrain_mask &= active_mask
    if not torch.any(terrain_mask):
        return

    ranges = command_term.cfg.ranges

    def _interpolate_range(initial_range: tuple[float, float], final_range: tuple[float, float]) -> tuple[float, float]:
        lower = initial_range[0] + (final_range[0] - initial_range[0]) * progress
        upper = initial_range[1] + (final_range[1] - initial_range[1]) * progress
        return lower, upper

    frequencies = _interpolate_range(initial_frequencies, ranges.frequencies)
    durations = _interpolate_range(initial_durations, ranges.durations)
    swing_height = _interpolate_range(initial_swing_height, ranges.swing_height)

    command[terrain_mask, 0] = torch.clamp(command[terrain_mask, 0], frequencies[0], frequencies[1])
    command[terrain_mask, 2] = torch.clamp(command[terrain_mask, 2], durations[0], durations[1])
    command[terrain_mask, 3] = torch.clamp(command[terrain_mask, 3], swing_height[0], swing_height[1])


def _terrain_column_ids_for_names(terrain_generator, terrain_type_names: tuple[str, ...]) -> list[int]:
    """Resolve terrain names to TerrainImporter terrain_types ids.

    In curriculum mode, IsaacLab stores env ``terrain_types`` as terrain column ids, not sub-terrain
    dictionary indices. Columns are deterministically assigned from sub-terrain proportions.
    """
    sub_terrains = getattr(terrain_generator, "sub_terrains", None) or {}
    if not sub_terrains:
        return []

    try:
        terrain_names = tuple(str(name) for name in sub_terrains.keys())
        sub_cfgs = list(sub_terrains.values())
    except AttributeError:
        terrain_names = tuple(str(name) for name in sub_terrains)
        sub_cfgs = list(sub_terrains)

    selected_names = {str(name) for name in terrain_type_names}
    if not bool(getattr(terrain_generator, "curriculum", False)):
        return [idx for idx, name in enumerate(terrain_names) if name in selected_names]

    num_cols = int(getattr(terrain_generator, "num_cols", len(terrain_names)))
    if num_cols <= 0:
        return []
    proportions = [max(0.0, float(getattr(sub_cfg, "proportion", 1.0))) for sub_cfg in sub_cfgs]
    total_proportion = sum(proportions)
    if len(proportions) != len(terrain_names) or total_proportion <= 0.0:
        return [idx for idx, name in enumerate(terrain_names) if name in selected_names]

    cumulative = []
    running = 0.0
    for proportion in proportions:
        running += proportion / total_proportion
        cumulative.append(running)

    column_ids = []
    for column_id in range(num_cols):
        threshold = column_id / num_cols + 0.001
        terrain_index = len(cumulative) - 1
        for index, limit in enumerate(cumulative):
            if threshold < limit:
                terrain_index = index
                break
        if terrain_names[terrain_index] in selected_names:
            column_ids.append(column_id)
    return column_ids


def _parallel_axis_matrix(offset: torch.Tensor) -> torch.Tensor:
    """Return (||r||^2 I - r r^T) for each offset vector."""
    eye = torch.eye(3, device=offset.device, dtype=offset.dtype).expand(offset.shape[0], 3, 3)
    outer = offset.unsqueeze(-1) @ offset.unsqueeze(-2)
    norm_sq = torch.sum(offset * offset, dim=-1, keepdim=True).unsqueeze(-1)
    return norm_sq * eye - outer


def _sample_payload_state(
    num_envs: int,
    device: torch.device,
    payload_mass_range: tuple[float, float],
    payload_pos_range: dict[str, tuple[float, float]],
    zero_payload_prob: float = 0.0,
    payload_mass_choices: tuple[float, ...] | list[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample payload mass and position in the target body frame."""
    if payload_mass_choices:
        choices = torch.tensor(payload_mass_choices, dtype=torch.float32, device=device)
        choice_ids = torch.randint(low=0, high=choices.numel(), size=(num_envs,), device=device)
        payload_mass = choices[choice_ids]
    else:
        mass_min = int(torch.ceil(torch.tensor(payload_mass_range[0], dtype=torch.float32)).item())
        mass_max = int(torch.floor(torch.tensor(payload_mass_range[1], dtype=torch.float32)).item())
        if mass_max < mass_min:
            mass_max = mass_min

        payload_mass = torch.randint(
            low=mass_min,
            high=mass_max + 1,
            size=(num_envs,),
            device=device,
        ).to(torch.float32)
    payload_pos = torch.stack(
        (
            math_utils.sample_uniform(payload_pos_range["x"][0], payload_pos_range["x"][1], (num_envs,), device=device),
            math_utils.sample_uniform(payload_pos_range["y"][0], payload_pos_range["y"][1], (num_envs,), device=device),
            math_utils.sample_uniform(payload_pos_range["z"][0], payload_pos_range["z"][1], (num_envs,), device=device),
        ),
        dim=-1,
    )

    if zero_payload_prob > 0.0:
        payload_off = torch.rand(num_envs, device=device) < zero_payload_prob
        payload_mass = torch.where(payload_off, torch.zeros_like(payload_mass), payload_mass)
        payload_pos = torch.where(payload_off.unsqueeze(-1), torch.zeros_like(payload_pos), payload_pos)

    return payload_mass, payload_pos


def _ensure_payload_baseline_cache(
    env: ManagerBasedEnv,
    asset: RigidObject | Articulation,
    body_id: int,
):
    """Cache startup-randomized body properties for payload re-application."""
    if hasattr(env, "_payload_body_id"):
        if env._payload_body_id != body_id:
            raise ValueError(
                f"Payload cache already initialized for body {env._payload_body_id}, got {body_id}."
            )
        return

    masses = asset.root_physx_view.get_masses().clone()
    coms = asset.root_physx_view.get_coms().clone()
    inertias = asset.root_physx_view.get_inertias().clone()

    env._payload_body_id = body_id
    env._payload_baseline_mass = masses[:, body_id].clone()
    env._payload_baseline_com = coms[:, body_id, :3].clone()
    env._payload_baseline_inertia = inertias[:, body_id].clone()

    env._payload_mass = torch.zeros(env.scene.num_envs, device=asset.device, dtype=masses.dtype)
    env._payload_pos_b = torch.zeros(env.scene.num_envs, 3, device=asset.device, dtype=coms.dtype)


def apply_payload_to_body(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    payload_mass_range: tuple[float, float],
    payload_pos_range: dict[str, tuple[float, float]],
    zero_payload_prob: float = 0.0,
    payload_mass_choices: tuple[float, ...] | list[float] | None = None,
):
    """Apply a sampled payload to a single rigid body by updating mass, COM, and inertia."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    if asset_cfg.body_ids == slice(None) or len(asset_cfg.body_ids) != 1:
        raise ValueError("apply_payload_to_body expects exactly one rigid body.")

    body_id = int(asset_cfg.body_ids[0])
    _ensure_payload_baseline_cache(env, asset, body_id)

    if env_ids is None:
        env_ids_dev = torch.arange(env.scene.num_envs, device=asset.device)
        env_ids_cpu = env_ids_dev.cpu()
    else:
        env_ids_dev = env_ids.to(device=asset.device, dtype=torch.long)
        env_ids_cpu = env_ids_dev.cpu()

    masses = asset.root_physx_view.get_masses().clone()
    coms = asset.root_physx_view.get_coms().clone()
    inertias = asset.root_physx_view.get_inertias().clone()

    baseline_mass = env._payload_baseline_mass.to(asset.device)[env_ids_dev]
    baseline_com = env._payload_baseline_com.to(asset.device)[env_ids_dev]

    baseline_inertia_full = env._payload_baseline_inertia.to(asset.device)
    baseline_inertia = baseline_inertia_full.reshape(baseline_inertia_full.shape[0], 3, 3)[env_ids_dev]

    payload_mass, payload_pos_b = _sample_payload_state(
        env_ids_dev.numel(),
        asset.device,
        payload_mass_range,
        payload_pos_range,
        zero_payload_prob=zero_payload_prob,
        payload_mass_choices=payload_mass_choices,
    )

    total_mass = baseline_mass + payload_mass
    total_mass_safe = torch.clamp(total_mass, min=1.0e-6)

    new_com = (
        baseline_mass.unsqueeze(-1) * baseline_com
        + payload_mass.unsqueeze(-1) * payload_pos_b
    ) / total_mass_safe.unsqueeze(-1)

    d_baseline = baseline_com - new_com
    d_payload = payload_pos_b - new_com
    new_inertia = (
        baseline_inertia
        + baseline_mass.unsqueeze(-1).unsqueeze(-1) * _parallel_axis_matrix(d_baseline)
        + payload_mass.unsqueeze(-1).unsqueeze(-1) * _parallel_axis_matrix(d_payload)
    )

    masses[env_ids_cpu, body_id] = total_mass.detach().cpu()
    coms[env_ids_cpu, body_id, :3] = new_com.detach().cpu()
    # Detach the selected body block from the parent tensor before writing back.
    # Otherwise PyTorch detects overlapping memory when the updated view is assigned
    # into the original inertia tensor.
    inertias_body = inertias[:, body_id].clone().reshape(inertias.shape[0], 3, 3)
    inertias_body[env_ids_cpu] = new_inertia.detach().cpu()
    inertias[:, body_id] = inertias_body.reshape(inertias.shape[0], -1)

    asset.root_physx_view.set_masses(masses, env_ids_cpu)
    asset.root_physx_view.set_coms(coms, env_ids_cpu)
    asset.root_physx_view.set_inertias(inertias, env_ids_cpu)

    env._payload_mass[env_ids_dev] = payload_mass
    env._payload_pos_b[env_ids_dev] = payload_pos_b


def apply_payload_to_body_stochastic(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    payload_mass_range: tuple[float, float],
    payload_pos_range: dict[str, tuple[float, float]],
    zero_payload_prob: float = 0.0,
    probability: float = 1.0,
    payload_mass_choices: tuple[float, ...] | list[float] | None = None,
):
    """Stochastically re-sample payload state for a subset of environments."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids_dev = torch.arange(env.scene.num_envs, device=asset.device)
    else:
        env_ids_dev = env_ids.to(device=asset.device, dtype=torch.long)

    if probability < 1.0:
        mask = torch.rand(env_ids_dev.shape[0], device=asset.device) < probability
        env_ids_dev = env_ids_dev[mask]

    if env_ids_dev.numel() == 0:
        return

    apply_payload_to_body(
        env=env,
        env_ids=env_ids_dev,
        asset_cfg=asset_cfg,
        payload_mass_range=payload_mass_range,
        payload_pos_range=payload_pos_range,
        zero_payload_prob=zero_payload_prob,
        payload_mass_choices=payload_mass_choices,
    )


"""
Internal helper functions.
"""


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data
