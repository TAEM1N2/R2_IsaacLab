from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, ContactSensor, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def robot_joint_torque(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint torque of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.applied_torque.to(device)


def robot_joint_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint acc of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.joint_acc.to(device)


def _zero_joint_obs(obs: torch.Tensor, asset: Articulation, joint_names: list[str]) -> torch.Tensor:
    joint_ids = asset.find_joints(joint_names)[0]
    if len(joint_ids) == 0:
        raise ValueError(f"Failed to resolve fault joint names: {joint_names}")
    obs = obs.clone()
    obs[:, joint_ids] = 0.0
    return obs


def joint_pos_rel_zero_joints(
    env: ManagerBasedRLEnv,
    joint_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint positions with selected sensor values forced to zero."""
    asset: Articulation = env.scene[asset_cfg.name]
    obs = asset.data.joint_pos - asset.data.default_joint_pos
    return _zero_joint_obs(obs, asset, joint_names)


def joint_vel_zero_joints(
    env: ManagerBasedRLEnv,
    joint_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint velocities with selected sensor values forced to zero."""
    asset: Articulation = env.scene[asset_cfg.name]
    return _zero_joint_obs(asset.data.joint_vel, asset, joint_names)


def last_action_zero_joints(
    env: ManagerBasedRLEnv,
    joint_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Last action with selected joint commands hidden from observations."""
    asset: Articulation = env.scene[asset_cfg.name]
    fault_joint_ids = asset.find_joints(joint_names)[0]
    if len(fault_joint_ids) == 0:
        raise ValueError(f"Failed to resolve fault joint names: {joint_names}")

    obs = env.action_manager.action.clone()
    action_term = env.action_manager.get_term("joint_pos")
    action_joint_ids = getattr(action_term, "_joint_ids", None)
    if action_joint_ids is None:
        return _zero_joint_obs(obs, asset, joint_names)

    if isinstance(action_joint_ids, slice):
        action_joint_ids = list(range(asset.num_joints))[action_joint_ids]

    fault_joint_ids = {int(joint_id) for joint_id in fault_joint_ids}
    action_columns = [idx for idx, joint_id in enumerate(action_joint_ids) if int(joint_id) in fault_joint_ids]
    if len(action_columns) == 0:
        raise ValueError(f"Fault joints are not part of the joint_pos action term: {joint_names}")

    obs[:, action_columns] = 0.0
    return obs


def joint_pos_rel_zero_flkn(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint positions with FL knee sensor value forced to zero."""
    return joint_pos_rel_zero_joints(env, ["FL_KN_JOINT"], asset_cfg=asset_cfg)


def joint_vel_zero_flkn(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint velocities with FL knee sensor value forced to zero."""
    return joint_vel_zero_joints(env, ["FL_KN_JOINT"], asset_cfg=asset_cfg)


def last_action_zero_flkn(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Last action with the FL knee command hidden from observations."""
    return last_action_zero_joints(env, ["FL_KN_JOINT"], asset_cfg=asset_cfg)
    


def robot_feet_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    """contact force of the robot feet"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    contact_force_tensor = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids].to(device)
    return contact_force_tensor.reshape(contact_force_tensor.shape[0], -1)


def estimated_foot_force_jacobian(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*TIP"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Estimate foot contact forces from joint torques and foot Jacobians.

    The least-squares estimate solves J(q)^T f = tau for the stacked foot force
    vector. If the simulator Jacobian is unavailable, the optional contact
    sensor is used as a bounded fallback so the Phase 2 observation remains
    well-defined during bring-up.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.data.joint_pos.device
    try:
        jacobian = asset.root_physx_view.get_jacobians().to(device)
        body_ids = torch.as_tensor(asset_cfg.body_ids, device=device, dtype=torch.long)
        foot_jacobian = jacobian[:, body_ids, :3, :]
        num_dofs = asset.data.applied_torque.shape[1]
        if foot_jacobian.shape[-1] > num_dofs:
            foot_jacobian = foot_jacobian[..., -num_dofs:]
        stacked_jacobian = foot_jacobian.reshape(env.num_envs, -1, num_dofs)
        torque = asset.data.applied_torque.to(device).unsqueeze(-1)
        force = torch.linalg.lstsq(stacked_jacobian.transpose(1, 2), torque).solution.squeeze(-1)
        return torch.nan_to_num(force, nan=0.0, posinf=1.0e3, neginf=-1.0e3)
    except Exception:
        if sensor_cfg is None:
            return torch.zeros(env.num_envs, len(asset_cfg.body_ids) * 3, device=device)
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids].to(device)
        return force.reshape(force.shape[0], -1)


def robot_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """mass of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_mass.to(device)


def robot_inertia(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """inertia of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    inertia_tensor = asset.data.default_inertia.to(device)
    return inertia_tensor.view(inertia_tensor.shape[0], -1)


def robot_joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint positions of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_pos.to(device)


def robot_joint_stiffness(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint stiffness of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_stiffness.to(device)


def robot_joint_damping(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint damping of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_damping.to(device)


def robot_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """pose of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_pos_w.to(device)


def robot_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """velocity of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_vel_w.to(device)


def robot_material_properties(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """material properties of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    material_tensor = asset.root_physx_view.get_material_properties().to(device)
    return material_tensor.view(material_tensor.shape[0], -1)


def robot_center_of_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """center of mass of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    com_tensor = asset.root_physx_view.get_coms().clone().to(device)
    return com_tensor.view(com_tensor.shape[0], -1)


def rigid_body_mass_moments(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Return per-body mass and first moments [m, m*x, m*y, m*z]."""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    masses = asset.root_physx_view.get_masses().clone().to(device)
    coms = asset.root_physx_view.get_coms().clone().to(device)[..., :3]

    if asset_cfg.body_ids != slice(None):
        body_ids = torch.as_tensor(asset_cfg.body_ids, device=device, dtype=torch.long)
        masses = masses[:, body_ids]
        coms = coms[:, body_ids, :]

    moments = masses.unsqueeze(-1) * coms
    return torch.cat((masses.unsqueeze(-1), moments), dim=-1).reshape(masses.shape[0], -1)


def payload_mass_moments(env: ManagerBasedEnv) -> torch.Tensor:
    """Return payload supervision target [m, m*x, m*y, m*z]."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if not hasattr(env, "_payload_mass") or not hasattr(env, "_payload_pos_b"):
        return torch.zeros(env.num_envs, 4, device=device)

    payload_mass = env._payload_mass.to(device)
    payload_pos_b = env._payload_pos_b.to(device)
    payload_moments = payload_mass.unsqueeze(-1) * payload_pos_b
    return torch.cat((payload_mass.unsqueeze(-1), payload_moments), dim=-1)


def payload_mass(env: ManagerBasedEnv) -> torch.Tensor:
    """Return payload mass supervision target [m]."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if not hasattr(env, "_payload_mass"):
        return torch.zeros(env.num_envs, 1, device=device)

    payload_mass = env._payload_mass.to(device)
    return payload_mass


def terrain_type_id(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return generated terrain type ids as loss-only labels."""
    device = env.device if hasattr(env, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    terrain = getattr(env.scene, "terrain", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    if terrain_types is None:
        return torch.full((env.num_envs, 1), -1.0, device=device)
    return terrain_types.to(device=device, dtype=torch.float32).view(-1, 1)


def phase2_body_height_tracking_reward(
    env: ManagerBasedEnv,
    target_height: float = 0.55,
    std: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    height_error = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    return torch.exp(-height_error / (std * std))


def phase2_grf_reward(
    env: ManagerBasedEnv,
    target_height: float = 0.55,
    gravity: float = 9.81,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*TIP"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    foot_force = estimated_foot_force_jacobian(env, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg).reshape(env.num_envs, -1, 3)
    grf_sum = torch.linalg.norm(foot_force, dim=-1).sum(dim=-1)
    # Current PhysX body masses already include the reset-sampled payload.
    # Adding env._payload_mass again would double-count payload in the GRF target.
    required_grf = asset.root_physx_view.get_masses().to(asset.data.root_pos_w.device).sum(dim=1) * gravity
    height = asset.data.root_pos_w[:, 2]
    reward = 0.75 * (height > target_height).float()
    reward = reward + 0.50 * ((height < target_height) & (grf_sum > required_grf)).float()
    return reward

def phase2_stability_reward(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gravity_std: float = 0.35,
    ang_vel_std: float = 1.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    gravity_xy_error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    ang_vel_xy_error = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    reward = torch.exp(-gravity_xy_error / (gravity_std * gravity_std))
    reward = reward * torch.exp(-ang_vel_xy_error / (ang_vel_std * ang_vel_std))
    return reward


def phase2_base_height_error(
    env: ManagerBasedEnv,
    target_height: float = 0.55,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_pos_w[:, 2] - target_height).unsqueeze(-1)


def phase2_payload_mass_log(env: ManagerBasedEnv) -> torch.Tensor:
    value = payload_mass(env)
    if value.dim() == 1:
        return value.unsqueeze(-1)
    if value.dim() == 2:
        return value
    return value.reshape(value.shape[0], -1)


def phase2_foot_contact_force_sum(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    force_sum = torch.linalg.norm(force, dim=-1).sum(dim=-1)
    return force_sum.unsqueeze(-1)


def payload_body_mass_delta(env: ManagerBasedEnv) -> torch.Tensor:
    """Return BODY mass delta induced by payload relative to cached startup baseline."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if not hasattr(env, "_payload_body_id") or not hasattr(env, "_payload_baseline_mass"):
        return torch.zeros(env.num_envs, 1, device=device)

    asset: Articulation = env.scene["robot"]
    body_id = int(env._payload_body_id)
    current_mass = asset.root_physx_view.get_masses().clone().to(device)[:, body_id]
    baseline_mass = env._payload_baseline_mass.to(device)
    return (current_mass - baseline_mass).unsqueeze(-1)


def payload_body_inertia_delta(env: ManagerBasedEnv) -> torch.Tensor:
    """Return BODY inertia delta induced by payload relative to cached startup baseline."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if not hasattr(env, "_payload_body_id") or not hasattr(env, "_payload_baseline_inertia"):
        return torch.zeros(env.num_envs, 9, device=device)

    asset: Articulation = env.scene["robot"]
    body_id = int(env._payload_body_id)
    current_inertia = asset.root_physx_view.get_inertias().clone().to(device)[:, body_id]
    baseline_inertia = env._payload_baseline_inertia.to(device)
    return current_inertia - baseline_inertia


def robot_contact_force(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """The contact forces of the body."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    body_contact_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]

    return body_contact_force.reshape(body_contact_force.shape[0], -1)


def safe_height_scan(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.55,
    min_height: float = -1.0,
    max_height: float = 1.0,
) -> torch.Tensor:
    """Height scan with explicit sanitization for NaN/Inf and extreme values."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    sensor_z = sensor.data.pos_w[:, 2].unsqueeze(1)
    hit_z = sensor.data.ray_hits_w[..., 2]
    heights = sensor_z - hit_z - offset
    heights = torch.nan_to_num(heights, nan=0.0, posinf=max_height, neginf=min_height)
    heights = torch.clamp(heights, min=min_height, max=max_height)
    return heights


def terrain_height_stats(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.55,
    min_height: float = -1.0,
    max_height: float = 1.0,
) -> torch.Tensor:
    """Return compact local terrain height statistics for loss-only supervision."""
    heights = safe_height_scan(
        env,
        sensor_cfg=sensor_cfg,
        offset=offset,
        min_height=min_height,
        max_height=max_height,
    )
    return torch.stack(
        (
            heights.mean(dim=-1),
            heights.std(dim=-1),
            heights.amin(dim=-1),
            heights.amax(dim=-1),
        ),
        dim=-1,
    )


def get_gait_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get the current gait phase as observation.

    The gait phase is represented by [sin(phase), cos(phase)] to ensure continuity.
    The phase is calculated based on the episode length and gait frequency.

    Returns:
        torch.Tensor: The gait phase observation. Shape: (num_envs, 2).
    """
    # check if episode_length_buf is available
    if not hasattr(env, "episode_length_buf"):
        return torch.zeros(env.num_envs, 2, device=env.device)

    # Get the gait command from command manager
    command_term = env.command_manager.get_term("gait_command")
    # Calculate gait indices based on episode length
    gait_indices = torch.remainder(env.episode_length_buf * env.step_dt * command_term.command[:, 0], 1.0)
    # Reshape gait_indices to (num_envs, 1)
    gait_indices = gait_indices.unsqueeze(-1)
    # Convert to sin/cos representation
    sin_phase = torch.sin(2 * torch.pi * gait_indices)
    cos_phase = torch.cos(2 * torch.pi * gait_indices)

    return torch.cat([sin_phase, cos_phase], dim=-1)


def get_gait_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Get the current gait command parameters as observation.

    Returns:
        torch.Tensor: The gait command parameters [frequency, offset, duration].
                     Shape: (num_envs, 3).
    """
    return env.command_manager.get_command(command_name)

def feet_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.body_lin_vel_w[:, asset_cfg.body_ids].flatten(start_dim=1)

def generated_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)

def joint_pos_rel_exclude_wheel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                                wheel_joints_name: list[str] = ["wheel_[RL]_Joint"] 
                                ) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)

    asset: Articulation = env.scene[asset_cfg.name]
    wheel_joints_idx = asset.find_joints(wheel_joints_name)[0]
    all_joints_idx = range(asset.num_joints)
    pos_idx_exclude_wheel = [i for i in all_joints_idx if i not in wheel_joints_idx]
    return asset.data.joint_pos[:, pos_idx_exclude_wheel] - asset.data.default_joint_pos[:, pos_idx_exclude_wheel]
