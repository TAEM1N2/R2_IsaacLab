# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from pongbot_r2.assets.config.pongbot_r2 import PONGBOT_R2_CFG

@configclass
class RowProgressionCfg:
    enabled: bool = True
    success_threshold: float = 0.30   # 에피소드 평균 리턴 정규화 기준(예시)
    patience: int = 2                 # 몇 에피소드 연속 성공 시 승급
    demote: bool = False              # 실패 시 강등 여부 (원하면 True)
    max_row: int = 9                  # ROW_RU_TERRAINS_CFG.num_rows - 1 과 일치

@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            "static_friction_range": (0.2, 0.8),
            "dynamic_friction_range": (0.2, 0.8),
            "restitution_range": (0., 0.5),
            "num_buckets": 100,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="BODY"),
            "mass_distribution_params": (-2., 3.),
            "operation": "add",
        },
    )

    add_com_offset = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="BODY"),
            "com_range": {
                "x": (-0.0, 0.1),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
            },
        },
    )

    add_actuator_gain = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="on_reset",  # 리셋될 때마다 실행
        params={
            "asset_cfg": SceneEntityCfg(name="robot"), 
            "operation": "scale", # 기존 값에 곱하기
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "distribution": "uniform",
        },
    )

@configclass
class PongbotR2EnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 5
    
    action_scale = [
        0.5, 0.5, 0.5, 0.5, # HR  관절들
        1.0, 1.0, 1.0, 1.0, # HP  관절들
        1.0, 1.0, 1.0, 1.0  # KNP 관절들
    ]

    action_space = 12
    observation_space = 53
    state_space = 0
    recovery_duration = 1.5

    # +++ 노이즈 관련 설정 추가 +++
    add_noise = True      # 노이즈 추가 여부 (True/False)
    noise_level = 1.0     # 전역 노이즈 스케일
    # ++++++++++++++++++++++++++++

    observation_noise_map: dict[str, float] = {
        "root_ang_vel_b": 0.0,
        "projected_gravity_b": 0.0,
        "commands": 0.0,
        "joint_pos": 0.0,
        "joint_vel": 0.0,
        "actions": 0.0,
        "cycles": 0.0, 
        # "root_ang_vel_b": 0.025,
        # "projected_gravity_b": 0.025,
        # "commands": 0.0,
        # "joint_pos": 0.05,
        # "joint_vel": 1.5,
        # "actions": 0.0,
        # "cycles": 0.0, 
    }

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.002,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = PONGBOT_R2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=2, update_period=0.01, track_air_time=True
    )

    reward_lin_vel                  = 1. #1.0
    reward_yaw_rate                 = 0.25
    reward_support_polygon          = 0.0 #0.5 #0.0 #0.5
    penalty_lin_vel_error           = -0.0
    penalty_z_vel                   = -0.5
    penalty_ang_vel                 = -0.0
    penalty_joint_torque            = -0.0000025
    penalty_joint_accel             = -0.000005 
    penalty_joint_vel               = -0.005
    penalty_action_rate             = -0.025 #-0.005
    penalty_undesired_contact       = 0.0 #-0.00075 
    penalty_flat_orientation        = -3. #-3.
    penalty_swing_stance_phase      = -0.025 -0.0
    penalty_base_height             = -1.0 #-2.5
    penalty_num_contact_feet_err    = -0.0
    penalty_slip                    = 0.0  #-0.075
    penalty_foot_height_err         = -0.25 #0.0 #-0.25 #-0.25
    penalty_short_swing             = -3. #-3.
    penalty_default_grf             = 0.0 #-0.0005
    penalty_trot_pitch              = -0.0
    penalty_pre_contact_velocity    = -0.

    penalty_default_pos_standing    = -0.5 #-0.05
    penalty_HR_pos_err              = -0.5
    penalty_HP_pos_err              = -0.25 #-0.15
    penalty_KNP_pos_err             = -0.2

    penalty_joint_limit_proximity   = 0.0 #-10.

    penalty_feet_air_time           = -0.
    

    # commands
    command_plus_x_range = 1.0
    command_plus_y_range = 1.0
    command_plus_yaw_range = 1.57

    command_minus_x_range = -0.5
    command_minus_y_range = -0.5
    command_minus_yaw_range = -1.57

    # +++ freeze 기능 설정값 추가 +++
    freeze_interval: int = 3000  # freeze를 실행할 주기 (단위: 스텝)
    freeze_duration_min: int = 300 # 최소 freeze 지속 시간 (단위: 스텝)
    freeze_duration_max: int = 500 # 최대 freeze 지속 시간 (단위: 스텝)
    # ++++++++++++++++++++++++++++++++

    # +++ '짧은 스윙' 페널티 관련 설정값 추가 +++
    min_swing_time: float = 0.05 # 최소 스윙 시간 (초)
    # +++++++++++++++++++++++++++++++++++++++++


    stand_env_range           = 500
    only_plus_x_envs_range    = 500
    only_minus_x_envs_range   = 500
    only_plus_y_envs_range    = 500
    only_minus_y_envs_range   = 500
    only_plus_yaw_envs_range  = 500
    only_minus_yaw_envs_range = 500
