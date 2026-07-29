# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import DelayedPDActuatorCfg


USD_PATH = Path(__file__).resolve().parents[1] / "usd" / "pongbot_r2" / "ktm" / "PONGBOT_R2_V2.usd"

HR_JOINT_STIFFNESS = 50.0
HR_JOINT_DAMPING = 5.0
HP_JOINT_STIFFNESS = 100.0
HP_JOINT_DAMPING = 5.0
KNEE_JOINT_STIFFNESS = 300.0
KNEE_JOINT_DAMPING = 5.0

HR_JOINT_STIFFNESS_RANGE = (50.0, 80.0)
HR_JOINT_DAMPING_RANGE = (4.0, 6.0)
HP_JOINT_STIFFNESS_RANGE = (80.0, 120.0)
HP_JOINT_DAMPING_RANGE = (4.0, 6.0)
KNEE_JOINT_STIFFNESS_RANGE = (240.0, 320.0)
KNEE_JOINT_DAMPING_RANGE = (4.0, 6.0)

# Backward-compatible aliases for configs that still import the old shared names.
HR_HP_JOINT_STIFFNESS = HR_JOINT_STIFFNESS
HR_HP_JOINT_DAMPING = HR_JOINT_DAMPING
HR_HP_JOINT_STIFFNESS_RANGE = HR_JOINT_STIFFNESS_RANGE
HR_HP_JOINT_DAMPING_RANGE = HR_JOINT_DAMPING_RANGE

RMD_TOR_CTRL = DelayedPDActuatorCfg(
    joint_names_expr = [".*HR_JOINT", ".*HP_JOINT", ".*KN_JOINT"],
    effort_limit={
        ".*HR_JOINT": 120.0,
        ".*HP_JOINT": 120.0,
        ".*KN_JOINT": 320.0,
    },
    velocity_limit=19.,
    stiffness={
        ".*HR_JOINT": HR_JOINT_STIFFNESS,
        ".*HP_JOINT": HP_JOINT_STIFFNESS,
        ".*KN_JOINT": KNEE_JOINT_STIFFNESS,
    },
    damping={
        ".*HR_JOINT": HR_JOINT_DAMPING,
        ".*HP_JOINT": HP_JOINT_DAMPING,
        ".*KN_JOINT": KNEE_JOINT_DAMPING,
    },
    armature = {".*": 0.0},
    friction = {".*": 0.0},
    min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
    max_delay=3,  # physics time steps (max: 2.0*3=6.0ms)
)


def _make_rmd_tor_ctrl_with_zero_effort(zero_effort_joint_names: tuple[str, ...]) -> DelayedPDActuatorCfg:
    effort_limit = {
        "FL_HR_JOINT": 120.0,
        "FL_HP_JOINT": 120.0,
        "FL_KN_JOINT": 320.0,
        "FR_HR_JOINT": 120.0,
        "FR_HP_JOINT": 120.0,
        "FR_KN_JOINT": 320.0,
        "RL_HR_JOINT": 120.0,
        "RL_HP_JOINT": 120.0,
        "RL_KN_JOINT": 320.0,
        "RR_HR_JOINT": 120.0,
        "RR_HP_JOINT": 120.0,
        "RR_KN_JOINT": 320.0,
    }
    for joint_name in zero_effort_joint_names:
        effort_limit[joint_name] = 0.0

    return DelayedPDActuatorCfg(
        joint_names_expr = [".*HR_JOINT", ".*HP_JOINT", ".*KN_JOINT"],
        effort_limit=effort_limit,
        velocity_limit=19.,
        stiffness={
            ".*HR_JOINT": HR_JOINT_STIFFNESS,
            ".*HP_JOINT": HP_JOINT_STIFFNESS,
            ".*KN_JOINT": KNEE_JOINT_STIFFNESS,
        },
        damping={
            ".*HR_JOINT": HR_JOINT_DAMPING,
            ".*HP_JOINT": HP_JOINT_DAMPING,
            ".*KN_JOINT": KNEE_JOINT_DAMPING,
        },
        armature = {".*": 0.0},
        friction = {".*": 0.0},
        min_delay=0,
        max_delay=3,
    )


RMD_TOR_CTRL_FLKN_FAULT = _make_rmd_tor_ctrl_with_zero_effort(("FL_KN_JOINT",))
RMD_TOR_CTRL_FL_LEG_FAULT = _make_rmd_tor_ctrl_with_zero_effort(
    ("FL_HR_JOINT", "FL_HP_JOINT", "FL_KN_JOINT")
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
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            ".*HR_JOINT": 0.0,   
            ".*HP_JOINT": 0.64, #0.8,    
            ".*KN_JOINT": -1.25, #-1.5, 
        },
    ),
    actuators={"legs": RMD_TOR_CTRL}, # REALNET_PD_CTRL
    soft_joint_pos_limit_factor=0.95,
)

PONGBOT_R2_FLKN_FAULT_CFG = PONGBOT_R2_CFG.replace(
    actuators={"legs": RMD_TOR_CTRL_FLKN_FAULT}
)

PONGBOT_R2_FL_LEG_FAULT_CFG = PONGBOT_R2_CFG.replace(
    actuators={"legs": RMD_TOR_CTRL_FL_LEG_FAULT}
)

# Backward-compatible alias for older imports that used the misspelled name.
PONRBOT_R2_CFG = PONGBOT_R2_CFG
