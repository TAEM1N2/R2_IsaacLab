# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import DelayedPDActuatorCfg


USD_PATH = Path(__file__).resolve().parents[1] / "usd" / "pongbot_r2" / "R2_usd.usd"


RMD_TOR_CTRL = DelayedPDActuatorCfg(
    joint_names_expr = [".*HR_JOINT", ".*HP_JOINT", ".*KN_JOINT"],
    effort_limit={
        ".*HR_JOINT": 120.0,
        ".*HP_JOINT": 120.0,
        ".*KN_JOINT": 150.0,
    },
    velocity_limit=19.,
    stiffness={".*": 100.0},
    damping={".*": 1.},
    armature = {".*": 0.0},
    friction = {".*": 0.0},
    min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
    max_delay=3,  # physics time steps (max: 2.0*3=6.0ms)
)

# Configuration - Articulation.
PONGBOT_R2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=6, solver_velocity_iteration_count=1
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*HR_JOINT": 0.0,   
            ".*HP_JOINT": 0.8,    
            ".*KN_JOINT": -1.5, 
        },
    ),
    actuators={"legs": RMD_TOR_CTRL}, # REALNET_PD_CTRL
    soft_joint_pos_limit_factor=0.95,
)

# Backward-compatible alias for older imports that used the misspelled name.
PONRBOT_R2_CFG = PONGBOT_R2_CFG
