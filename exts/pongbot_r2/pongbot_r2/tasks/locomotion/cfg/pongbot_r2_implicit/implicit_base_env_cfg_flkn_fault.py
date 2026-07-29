import math
from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import DomeLightCfg, MdlFileCfg, RigidBodyMaterialCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as UniformNoise
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg as BaseCommandsCfg

from pongbot_r2.tasks.locomotion import mdp

##################
# Scene Definition
##################


@configclass
class PFSceneCfg(InteractiveSceneCfg):
    """Configuration for the test scene"""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        visual_material=MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
            + "TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # sky light
    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=DomeLightCfg(
            intensity=750.0,
            color=(0.9, 0.9, 0.9),
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # pongbot robot
    robot: ArticulationCfg = MISSING

    height_scanner: RayCasterCfg = MISSING

    # contact sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=4, track_air_time=True, update_period=0.0
    )


##############
# MDP settings
##############


@configclass
class CommandCfg(BaseCommandsCfg):
    gait_command = mdp.UniformGaitCommandCfg(
        resampling_time_range=(5.0, 5.0),  # Fixed resampling time of 5 seconds
        debug_vis=False,  # No debug visualization needed
        ranges=mdp.UniformGaitCommandCfg.Ranges(
            frequencies=(2.0, 2.0),  # Gait frequency range [Hz]
            offsets=(0.5, 0.5),  # Phase offset range [0-1]
            durations=(0.5, 0.5),  # Contact duration range [0-1]
            swing_height=(0.05, 0.05) # foot swing height
        ),
    )

    def __post_init__(self):
        self.base_velocity.asset_name = "robot"
        self.base_velocity.heading_command = True
        self.base_velocity.debug_vis = True
        self.base_velocity.heading_control_stiffness = 1.0
        self.base_velocity.resampling_time_range = (0.0, 5.0)
        self.base_velocity.rel_standing_envs = 0.2
        self.base_velocity.rel_heading_envs = 0.0
        self.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5), lin_vel_y=(-1.2, 1.2), ang_vel_z=(-1.2, 1.2), heading=(-math.pi, math.pi)
        )


@configclass
class ActionsCfg:
    """Action specifications for the MDP"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*HR_JOINT", ".*HP_JOINT", ".*KN_JOINT"],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class ObservarionsCfg:
    """Observation specifications for the MDP"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observation for policy group"""

        # # robot base measurements
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=GaussianNoise(mean=0.0, std=0.05),clip=(-100.0, 100.0),scale=0.25,)
        proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=GaussianNoise(mean=0.0, std=0.025),clip=(-100.0, 100.0),scale=1.0,)


        # robot joint measurements
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=GaussianNoise(mean=0.0, std=0.01),clip=(-100.0, 100.0),scale=1.0,)
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=GaussianNoise(mean=0.0, std=0.01),clip=(-100.0, 100.0),scale=0.05,)

        # last action
        last_action = ObsTerm(func=mdp.last_action)

        # # gaits
        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class HistoryObsCfg(ObsGroup):
        """History Observation for policy group"""

        # # robot base measurements
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=GaussianNoise(mean=0.0, std=0.05),clip=(-100.0, 100.0),scale=0.25,)
        proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=GaussianNoise(mean=0.0, std=0.025),clip=(-100.0, 100.0),scale=1.0,)

        # robot joint measurements
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=GaussianNoise(mean=0.0, std=0.01),clip=(-100.0, 100.0),scale=1.0,)
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=GaussianNoise(mean=0.0, std=0.01),clip=(-100.0, 100.0),scale=0.05,)
        # last action
        last_action = ObsTerm(func=mdp.last_action)
        # gaits
        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = False

    @configclass
    class CriticCfg(ObsGroup):
        """Observation for critic group"""

        # Policy observation
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        proj_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        last_action = ObsTerm(func=mdp.last_action)
        vel_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})


        # heights = ObsTerm(func=mdp.height_scan,params={"sensor_cfg": SceneEntityCfg("height_scanner")})
        heights = ObsTerm(
           func=mdp.safe_height_scan,
           params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        robot_pos = ObsTerm(func=mdp.robot_pos)
        
        # Privileged observation
        # robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque)
        # robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc)
        robot_feet_contact_force = ObsTerm(
            func=mdp.robot_feet_contact_force,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*TIP"),
            },
        )

        # robot_mass = ObsTerm(func=mdp.robot_mass)
        # robot_inertia = ObsTerm(func=mdp.robot_inertia)
        # robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness)
        # robot_joint_damping = ObsTerm(func=mdp.robot_joint_damping)
        # robot_vel = ObsTerm(func=mdp.robot_vel)
        # robot_material_propertirs = ObsTerm(func=mdp.robot_material_properties)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True


    @configclass
    class CommandsObsCfg(ObsGroup):
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    commands: CommandsObsCfg = CommandsObsCfg()
    obsHistory: HistoryObsCfg = HistoryObsCfg()


@configclass
class Phase2ObservarionsCfg(ObservarionsCfg):
    """Observation groups used only by Phase 2 adaptive training."""

    @configclass
    class AdaptiveCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=GaussianNoise(mean=0.0, std=0.05), clip=(-100.0, 100.0), scale=0.25)
        proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=GaussianNoise(mean=0.0, std=0.025), clip=(-100.0, 100.0), scale=1.0)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=GaussianNoise(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=GaussianNoise(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})
        foot_force = ObsTerm(
            func=mdp.estimated_foot_force_jacobian,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*TIP"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*TIP"),
            },
            clip=(-1000.0, 1000.0),
            scale=0.01,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class AdaptiveCriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        proj_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        last_action = ObsTerm(func=mdp.last_action)
        vel_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})
        
        heights = ObsTerm(func=mdp.safe_height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")})
        robot_pos = ObsTerm(func=mdp.robot_pos)
        robot_feet_contact_force = ObsTerm(
            func=mdp.robot_feet_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*TIP")},
        )
        estimated_foot_force = ObsTerm(
            func=mdp.estimated_foot_force_jacobian,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*TIP"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*TIP"),
            },
            clip=(-1000.0, 1000.0),
            scale=0.01,
        )
        payload_mass = ObsTerm(func=mdp.phase2_payload_mass_log)
        payload_body_mass_delta = ObsTerm(func=mdp.payload_body_mass_delta)
        payload_body_inertia_delta = ObsTerm(func=mdp.payload_body_inertia_delta)
        base_height_error = ObsTerm(func=mdp.phase2_base_height_error, params={"target_height": 0.55})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class Phase2LogCfg(ObsGroup):
        payload_mass = ObsTerm(func=mdp.phase2_payload_mass_log)
        base_height_error = ObsTerm(func=mdp.phase2_base_height_error, params={"target_height": 0.55})
        foot_contact_force = ObsTerm(
            func=mdp.phase2_foot_contact_force_sum,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*TIP")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    adaptive: AdaptiveCfg = AdaptiveCfg()
    adaptiveCritic: AdaptiveCriticCfg = AdaptiveCriticCfg()
    phase2Log: Phase2LogCfg = Phase2LogCfg()


@configclass
class EventsCfg:
    """Configuration for events"""

    # startup
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="BODY"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )
    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*THIGH", ".*CALF", ".*TIP"]),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )
    radomize_rigid_body_mass_inertia = EventTerm(
        func=mdp.randomize_rigid_body_mass_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_inertia_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.2),
            "dynamic_friction_range": (0.7, 0.9),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 48,
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (150, 150),
            "damping_distribution_params": (5.5, 5.5),
            "operation": "abs",
            "distribution": "uniform",
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )
    robot_center_of_mass = EventTerm(
        func=mdp.randomize_rigid_body_coms,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "com_distribution_params": ((-0.1, 0.1), (-0.05, 0.06), (-0.05, 0.05)),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    # reset
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-0.0, 0.0)},
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (0.0, 0.0),
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

    # interval
    push_robot = EventTerm(
        func=mdp.apply_external_force_torque_stochastic,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="BODY"),
            "force_range": {
                "x": (-500.0, 500.0),
                "y": (-500.0, 500.0),
                "z": (-0.0, 0.0),
            },  # force = mass * dv / dt
            "torque_range": {"x": (-50.0, 50.0), "y": (-50.0, 50.0), "z": (-0.0, 0.0)},
            "probability": 0.002,  # Expect step = 1 / probability
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

    set_command_zero = EventTerm(
        func=mdp.set_zero_command,
        mode="interval",
        interval_range_s=(8.0, 15.0),
        params={
            "command_name": "base_velocity",
            "duration_s": 1.0,
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )


@configclass
class Phase2EventsCfg(EventsCfg):
    payload_reset = EventTerm(
        func=mdp.apply_payload_to_body,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="BODY"),
            "payload_mass_range": (0.0, 30.0),
            # "payload_mass_choices": (5.0, 10.0, 15.0, 20.0, 25.0, 30.0),
            "payload_pos_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
            },
            "zero_payload_prob": 0.0,
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

# @configclass
# class RewardsCfg:
#     """Reward terms for the MDP"""

#     rew_lin_vel_xy = RewTerm(
#         func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
#     )
#     rew_ang_vel_z = RewTerm(
#         func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
#     )

#     # ===== Fixed Auxiliary =====
#     pen_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
#     pen_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05) 
#     pen_flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-0.2)
#     pen_joint_accel = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
#     pen_joint_powers = RewTerm(func=mdp.joint_powers_l1, weight=-2.0e-5)
#     pen_base_height = RewTerm( func=mdp.base_com_height_dreamwaq, 
#         params={ "target_height": 0.55 },
#         weight=-1.0, 
#     )
#     pen_foot_clearance = RewTerm(
#         func=mdp.foot_clearance,
#         weight=-0.01,
#         params={"asset_cfg": SceneEntityCfg("robot", body_names=".*TIP"), "p_des": 0.05, "foot_radius": 0.03},
#     )
#     pen_action_rate = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.01, params={"max_value": 400.0})
#     pen_action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.01)
#     pen_joint_powers_var = RewTerm(func=mdp.joint_powers_var, weight=-10.0e-5)


#     # termination related rewards
#     # keep_balance = RewTerm(
#     #     func=mdp.stay_alive,
#     #     weight=0.0
#     # )

#     # pen_standing_vel = RewTerm(
#     #     func=mdp.stand_still,
#     #     weight=-0.1,
#     # )

#     # pen_feet_regulation = RewTerm(
#     #     func=mdp.feet_regulation,
#     #     weight=-0.001,
#     #     params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*TIP"]),
#     #             "base_height_target": 0.6, "foot_radius": 0.03},
#     # )
#     # pen_undesired_contacts = RewTerm(
#     #     func=mdp.undesired_contacts,
#     #     weight= -0.25,
#     #     params={
#     #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*CALF", ".*THIGH", "BODY"]),
#     #         "threshold": 10.0,
#     #     },
#     # )

#     # pen_joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.25)
#     # pen_joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-2.5e-6) 
#     # # pen_joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight= -5e-4)
#     # pen_swing_height_error = RewTerm(
#     #     func=mdp.pen_swing_height_error,
#     #     weight=-1.0,
#     #     params={"command_name": "gait_command", "asset_cfg": SceneEntityCfg("robot", body_names=".*TIP"), "foot_radius": 0.03},
#     # )
#     # pen_joint_default_pos = RewTerm(func=mdp.joint_deviation_l1, weight = -0.1)
#     # pen_standing_joint_default_pos = RewTerm(func=mdp.stand_still_joint_deviation_l1, weight=-0.1)
#     # pen_hip_roll_pos = RewTerm(
#     #     func=mdp.joint_deviation_l1,
#     #     weight=-0.1,
#     #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*HR_JOINT")},
#     # )

#     # pen_standing_vel = RewTerm(func=mdp.stand_still, weight = -1.0)
    
#     # pen_feet_slide = RewTerm(
#     #     func=mdp.feet_slide,
#     #     weight=-0.1,
#     #     params={
#     #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*TIP"),
#     #         "asset_cfg": SceneEntityCfg("robot", body_names=".*TIP"),
#     #     },
#     # )

#     # foot_landing_vel = RewTerm(
#     #     func=mdp.foot_landing_vel,
#     #     weight=-0.1,
#     #     params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*TIP"]),
#     #             "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*TIP"]),
#     #              "foot_radius": 0.03, "about_landing_threshold": 0.08},
#     # )
    


@configclass
class RewardsCfg:
    """Reward terms for the MDP"""

    # termination related rewards
    # keep_balance = RewTerm(
    #     func=mdp.stay_alive,
    #     weight=0.0
    # )

    # tracking related rewards
    rew_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.1)}
    )
    rew_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.1)}
    )
    rew_lin_vel_xy2 = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    rew_ang_vel_z2 = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )


    # =====
    # Gait reward
    test_gait_reward = RewTerm(
        func=mdp.GaitReward,
        weight=0.0,
        params={
            "tracking_contacts_shaped_force": -2.0,
            "tracking_contacts_shaped_vel": -2.0,
            "gait_force_sigma": 25.0,
            "gait_vel_sigma": 0.25,
            "kappa_gait_probs": 0.05,
            "command_name": "gait_command",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*TIP"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*TIP"),
        },
    )

    pen_base_height = RewTerm(
        func=mdp.base_com_height,
        params={
            "target_height": 0.55,
        },
        weight=-0.1, 
    )
    pen_flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-0.1)

    # ===== Fixed Auxiliary =====
    # ===== 
    pen_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    pen_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-5e-3) 
    # pen_feet_regulation = RewTerm(
    #     func=mdp.feet_regulation,
    #     weight=-0.001,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*TIP"]),
    #             "base_height_target": 0.55, "foot_radius": 0.03},
    # )
    pen_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight= -0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*CALF", ".*THIGH"]),
            "threshold": 10.0,
        },
    )

    pen_joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    pen_joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-2.5e-6) 
    pen_combined_torque = RewTerm(
    func=mdp.combined_torque_penalty,
    weight=-0.05,
    params={"torque_safe_ratio": 0.7, "power": 6.0, "high_usage_weight": 0.1},
    )


    # pen_joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight= -5e-4)
    pen_joint_accel = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    # pen_action_rate = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.01, params={"max_value": 400.0})
    pen_action_rate = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.0025, params={"max_value": 400.0})
    pen_swing_height_error = RewTerm(
        func=mdp.pen_swing_height_error,
        weight=-0.1,
        params={"command_name": "gait_command", "asset_cfg": SceneEntityCfg("robot", body_names=".*TIP"), "foot_radius": 0.03},
    )
    pen_action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.0025)
    pen_joint_powers = RewTerm(func=mdp.joint_powers_l1, weight=-5.0e-5)
    # pen_joint_powers_var = RewTerm(func=mdp.joint_powers_var, weight=-2.5e-6)
    # pen_joint_default_pos = RewTerm(func=mdp.joint_deviation_l1, weight = -0.05)
    # pen_standing_joint_default_pos = RewTerm(func=mdp.stand_still_joint_deviation_l1, weight=-0.05)
    # pen_standing_foot_contact = RewTerm(
    #     func=mdp.standing_foot_contact,
    #     weight=-0.5,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*TIP"),
    #         "lin_threshold": 0.05,
    #         "ang_threshold": 0.05,
    #         "force_threshold": 1.0,
    #     },
    # )
    pen_standing_foot_height = RewTerm(
        func=mdp.standing_foot_height,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*TIP"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "foot_radius": 0.03,
            "lin_threshold": 0.05,
            "ang_threshold": 0.05,
            "height_tolerance": 0.02,
        },
    )
    # pen_hip_roll_pos = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*HR_JOINT")},
    # )

    pen_standing_vel = RewTerm(func=mdp.stand_still, weight = -0.25)
    
    pen_feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*TIP"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*TIP"),
        },
    )

    foot_landing_vel = RewTerm(
        func=mdp.foot_landing_vel,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*TIP"]),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*TIP"]),
                 "foot_radius": 0.03, "about_landing_threshold": 0.08},
    )
        
@configclass
class TerminationsCfg:
    """Termination terms for the MDP"""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="BODY"), "threshold": 1.0},
    # )
    base_contact = DoneTerm(
        func=mdp.DelayedIllegalContact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="BODY"), "threshold": 1.0, "delay_s": 0.1},
    )

    # thigh_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP"""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # penalty_ramp = CurrTerm(
    #     func=mdp.ramp_reward_terms_by_weight,
    #     params={
    #         "term_names": (
    #             "pen_joint_torque",
    #             "pen_joint_accel",
    #             "pen_feet_slide",
    #             "pen_action_smoothness",
    #         ),
    #         "start_scale": 0.5,
    #         "end_scale": 1.0,
    #         "start_step": 0,
    #         "end_step": 80_000,
    #     },
    # )



########################
# Environment definition
########################


@configclass
class PFEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the test environment"""

    # Scene settings
    scene: PFSceneCfg = PFSceneCfg(num_envs=4096, env_spacing=2.5)
    # scene: PFSceneCfg = PFSceneCfg(num_envs=2048, env_spacing=2.5)
    debug_action_print: bool = False
    # Basic settings
    observations: ObservarionsCfg = ObservarionsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandCfg = CommandCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization"""
        self.decimation = 5
        self.episode_length_s = 20.0
        # Keep the reset-sampled base velocity command for the whole episode.
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        self.sim.render_interval = 2 * self.decimation
        # simulation settings
        self.sim.dt = 0.002
        self.seed = 42
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
