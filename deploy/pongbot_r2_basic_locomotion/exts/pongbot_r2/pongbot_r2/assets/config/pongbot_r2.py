"""PongBot R2 articulation used by the basic locomotion tasks.

@version 0.0.1
@update 2026-07-29: Retain only the nominal R2 actuator and articulation configuration.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


USD_PATH = Path(__file__).resolve().parents[1] / "usd" / "pongbot_r2" / "ktm" / "PONGBOT_R2_V2.usd"

HR_JOINT_STIFFNESS = 50.0
HR_JOINT_DAMPING = 5.0
HP_JOINT_STIFFNESS = 100.0
HP_JOINT_DAMPING = 5.0
KNEE_JOINT_STIFFNESS = 300.0
KNEE_JOINT_DAMPING = 5.0

RMD_TOR_CTRL = DelayedPDActuatorCfg(
    joint_names_expr=[".*HR_JOINT", ".*HP_JOINT", ".*KN_JOINT"],
    effort_limit={
        ".*HR_JOINT": 120.0,
        ".*HP_JOINT": 120.0,
        ".*KN_JOINT": 320.0,
    },
    velocity_limit=19.0,
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
    armature={".*": 0.0},
    friction={".*": 0.0},
    min_delay=0,
    max_delay=3,
)

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
            enabled_self_collisions=True,
            solver_position_iteration_count=6,
            solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            ".*HR_JOINT": 0.0,
            ".*HP_JOINT": 0.64,
            ".*KN_JOINT": -1.25,
        },
    ),
    actuators={"legs": RMD_TOR_CTRL},
    soft_joint_pos_limit_factor=0.95,
)
