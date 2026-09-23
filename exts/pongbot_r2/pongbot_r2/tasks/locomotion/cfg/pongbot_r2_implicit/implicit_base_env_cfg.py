"""PongBot R2 implicit locomotion environment configuration.

@version 0.0.10
@update 2026-07-13: Add dense BODY-contact and non-timeout termination penalties for recovery learning.
@update 2026-07-13: Use stationary-baseline-corrected linear tracking in implicit locomotion.
"""

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

from pongbot_r2.assets.config.pongbot_r2 import (
    HP_JOINT_DAMPING_RANGE,
    HP_JOINT_STIFFNESS_RANGE,
    HR_JOINT_DAMPING_RANGE,
    HR_JOINT_STIFFNESS_RANGE,
    KNEE_JOINT_DAMPING_RANGE,
    KNEE_JOINT_STIFFNESS_RANGE,
)
from pongbot_r2.tasks.locomotion import mdp


PONGBOT_R2_USD_TOTAL_MASS_KG = 68.0
PONGBOT_R2_REAL_TOTAL_MASS_KG = 74.0
PONGBOT_R2_REAL_MASS_SCALE = PONGBOT_R2_REAL_TOTAL_MASS_KG / PONGBOT_R2_USD_TOTAL_MASS_KG

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
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
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
    fl_foot_height_scanner: RayCasterCfg = None
    fr_foot_height_scanner: RayCasterCfg = None
    rl_foot_height_scanner: RayCasterCfg = None
    rr_foot_height_scanner: RayCasterCfg = None

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
        resampling_time_range=(20.0, 20.0),  # Overridden to the episode length in the environment config.
        debug_vis=False,  # No debug visualization needed
        ranges=mdp.UniformGaitCommandCfg.Ranges(
            frequencies=(1.2, 1.6),  # Gait frequency range [Hz]
            offsets=(0.5, 0.5),  # Phase offset range [0-1]
            durations=(0.55, 0.65),  # Contact duration range [0-1]
            swing_height=(0.05, 0.1) # foot swing height
        ),
        profiles=(
            (1.20, 0.50, 0.66, 0.07),  # slow trot
            (1.50, 0.50, 0.60, 0.08),  # nominal trot
            (2.00, 0.50, 0.50, 0.08),  # fast trot
            (1.50, 0.50, 0.60, 0.13),  # high-clearance trot
        ),
        profile_probabilities=(0.25, 0.25, 0.25, 0.25),
        jitter=(0.08, 0.0, 0.03, 0.015),
    )

    def __post_init__(self):
        self.base_velocity.asset_name = "robot"
        self.base_velocity.heading_command = True
        self.base_velocity.debug_vis = True
        self.base_velocity.heading_control_stiffness = 1.0
        self.base_velocity.resampling_time_range = (0.0, 5.0)
        self.base_velocity.rel_standing_envs = 0.1
        self.base_velocity.rel_heading_envs = 0.0
        self.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-0.8, 0.8), heading=(-math.pi, math.pi)
        )


@configclass
class LegacyCommandCfg(CommandCfg):
    """Gait-command sampler used by the 2026-07-07 ``사뿐사뿐`` run."""

    gait_command = mdp.UniformGaitCommandCfg(
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformGaitCommandCfg.Ranges(
            frequencies=(1.2, 1.6),
            offsets=(0.5, 0.5),
            durations=(0.55, 0.65),
            swing_height=(0.05, 0.1),
        ),
        # None selects the original independent uniform-range sampler.
        profiles=None,
        profile_probabilities=None,
        jitter=(0.0, 0.0, 0.0, 0.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*HR_JOINT", ".*HP_JOINT", ".*KN_JOINT"],
        scale={
        ".*HR_JOINT": 0.5,     #0.25,
        ".*HP_JOINT": 0.5,      #0.5,
        ".*KN_JOINT": 0.5,     #0.5,
        },
        use_default_offset=True,
    )


@configclass
class LegacyActionsCfg(ActionsCfg):
    """Historical action interface with an explicit [-10, 10] clip."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*HR_JOINT", ".*HP_JOINT", ".*KN_JOINT"],
        scale={".*HR_JOINT": 0.1, ".*HP_JOINT": 0.25, ".*KN_JOINT": 0.25},
        clip={".*": (-10.0, 10.0)},
        use_default_offset=True,
    )


@configclass
class ObservarionsCfg:
    """Observation specifications for the MDP"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observation for policy group"""

        # # robot base measurements
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=GaussianNoise(mean=0.0, std=0.1),clip=(-100.0, 100.0),scale=0.25,)
        proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=GaussianNoise(mean=0.0, std=0.025),clip=(-100.0, 100.0),scale=1.0,)


        # robot joint measurements
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=GaussianNoise(mean=0.0, std=0.05),clip=(-100.0, 100.0),scale=1.0,)
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=GaussianNoise(mean=0.0, std=0.3),clip=(-100.0, 100.0),scale=0.05,)

        # last action
        last_action = ObsTerm(func=mdp.last_action)

        # # # gaits
        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class HistoryObsCfg(ObsGroup):
        """History Observation for policy group"""

        # # robot base measurements
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=GaussianNoise(mean=0.0, std=0.1),clip=(-100.0, 100.0),scale=0.25,)
        proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=GaussianNoise(mean=0.0, std=0.025),clip=(-100.0, 100.0),scale=1.0,)

        # robot joint measurements
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=GaussianNoise(mean=0.0, std=0.05),clip=(-100.0, 100.0),scale=1.0,)
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=GaussianNoise(mean=0.0, std=0.3),clip=(-100.0, 100.0),scale=0.05,)
        # last action
        last_action = ObsTerm(func=mdp.last_action)
        # # gaits
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

    @configclass
    class EncoderTargetObsCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        terrain_height_stats = ObsTerm(
            func=mdp.terrain_height_stats,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    commands: CommandsObsCfg = CommandsObsCfg()
    obsHistory: HistoryObsCfg = HistoryObsCfg()
    encoder_target: EncoderTargetObsCfg = EncoderTargetObsCfg()


@configclass
class LegacyObservarionsCfg(ObservarionsCfg):
    """Observations captured from the 2026-07-07 ``사뿐사뿐`` run."""

    @configclass
    class LegacyCriticCfg(ObsGroup):
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
        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque)
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc)
        robot_feet_contact_force = ObsTerm(
            func=mdp.robot_feet_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*TIP")},
        )
        robot_mass = ObsTerm(func=mdp.robot_mass)
        robot_inertia = ObsTerm(func=mdp.robot_inertia)
        robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness)
        robot_joint_damping = ObsTerm(func=mdp.robot_joint_damping)
        robot_vel = ObsTerm(func=mdp.robot_vel)
        # Preserve the historical misspelling because observation term names
        # determine critic input ordering and checkpoint compatibility.
        robot_material_propertirs = ObsTerm(func=mdp.robot_material_properties)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: LegacyCriticCfg = LegacyCriticCfg()


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
class Phase2LoadAdaptiveObservarionsCfg(Phase2ObservarionsCfg):
    """Phase 2 observations with adaptive history for load-response encoding."""

    @configclass
    class AdaptiveHistoryCfg(Phase2ObservarionsCfg.AdaptiveCfg):
        def __post_init__(self):
            super().__post_init__()
            self.history_length = 10
            self.flatten_history_dim = False

    adaptiveHistory: AdaptiveHistoryCfg = AdaptiveHistoryCfg()


@configclass
class EventsCfg:
    """Configuration for events"""

    # startup
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="BODY"),
            "mass_distribution_params": (-2.0, 2.0),
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
            "mass_inertia_distribution_params": (
                0.9 * PONGBOT_R2_REAL_MASS_SCALE,
                1.1 * PONGBOT_R2_REAL_MASS_SCALE,
            ),
            "operation": "scale",
        },
    )
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.5),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 48,
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )
    tip_radius_randomization = EventTerm(
        func=mdp.randomize_foot_radius,
        mode="startup",
        params={"radius_range": (0.030, 0.037)},
        is_global_time=False,
        min_step_count_between_reset=0,
    )
    robot_hr_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*HR_JOINT"),
            "stiffness_distribution_params": HR_JOINT_STIFFNESS_RANGE,
            "damping_distribution_params": HR_JOINT_DAMPING_RANGE,
            "operation": "abs",
            "distribution": "uniform",
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )
    robot_hp_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*HP_JOINT"),
            "stiffness_distribution_params": HP_JOINT_STIFFNESS_RANGE,
            "damping_distribution_params": HP_JOINT_DAMPING_RANGE,
            "operation": "abs",
            "distribution": "uniform",
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )
    robot_knee_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*KN_JOINT"),
            "stiffness_distribution_params": KNEE_JOINT_STIFFNESS_RANGE,
            "damping_distribution_params": KNEE_JOINT_DAMPING_RANGE,
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
        interval_range_s=(0.01, 0.01),
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
        interval_range_s=(12.0, 20.0),
        params={
            "command_name": "base_velocity",
            "duration_s": 0.5,
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )


@configclass
class ImplicitRoughEventsCfg(EventsCfg):
    limit_pyramid_velocity_commands = EventTerm(
        func=mdp.limit_commands_on_terrain_types,
        mode="interval",
        interval_range_s=(0.01, 0.01),
        params={
            "command_name": "base_velocity",
            "terrain_type_names": (
                "boxes_hard",
                "pyramid_stairs_hard",
                "pyramid_stairs_inv_hard",
            ),
            "initial_lin_vel_x": (0.25, 0.6),
            "initial_lin_vel_y": (-0.05, 0.05),
            "initial_ang_vel_z": (-0.15, 0.15),
            "start_step": 0,
            "end_step": 250_000,
        },
        is_global_time=True,
        min_step_count_between_reset=0,
    )

    zero_small_velocity_commands = EventTerm(
        func=mdp.zero_small_velocity_commands,
        mode="interval",
        interval_range_s=(0.01, 0.01),
        params={
            "command_name": "base_velocity",
            "linear_threshold": 0.2,
            "angular_threshold": 0.2,
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )


@configclass
class LegacyImplicitRoughEventsCfg(EventsCfg):
    """Events captured from the 2026-07-07 implicit rough run."""

    limit_hard_terrain_commands = EventTerm(
        func=mdp.limit_commands_on_terrain_types,
        mode="interval",
        interval_range_s=(0.01, 0.01),
        params={
            "command_name": "base_velocity",
            "terrain_type_names": ("boxes_hard", "pyramid_stairs_hard", "pyramid_stairs_inv_hard"),
            "initial_lin_vel_x": (-0.8, 0.8),
            "initial_lin_vel_y": (-0.3, 0.3),
            "initial_ang_vel_z": (-0.5, 0.5),
            "start_step": 0,
            "end_step": 80_000,
        },
        is_global_time=True,
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


@configclass
class Phase2ComEventsCfg(Phase2EventsCfg):
    payload_reset = EventTerm(
        func=mdp.apply_payload_to_body,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="BODY"),
            "payload_mass_range": (0.0, 30.0),
            "payload_pos_range": {
                "x": (-0.05, 0.05),
                "y": (-0.03, 0.03),
                "z": (0.0, 0.0),
            },
            "zero_payload_prob": 0.0,
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )


@configclass
class Phase2AbruptEventsCfg(Phase2ComEventsCfg):
    payload_interval_change = EventTerm(
        func=mdp.apply_payload_to_body_stochastic,
        mode="interval",
        interval_range_s=(3.0, 8.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="BODY"),
            "payload_mass_range": (0.0, 30.0),
            "payload_pos_range": {
                "x": (-0.05, 0.05),
                "y": (-0.03, 0.03),
                "z": (0.0, 0.0),
            },
            "zero_payload_prob": 0.0,
            "probability": 0.2,
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
    # rew_lin_vel_xy1 = RewTerm(
    #     func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.1)}
    # )
    # rew_ang_vel_z1 = RewTerm(
    #     func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.1)}
    # )
    rew_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_advantage_over_stationary,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "command_threshold": 0.2},
    )
    rew_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    pen_nonzero_lin_vel_xy_error = RewTerm(
        func=mdp.nonzero_lin_vel_tracking_error_l2,
        weight=-0.25,
        params={"command_name": "base_velocity", "lin_threshold": 0.1, "max_error": 4.0},
    )
    pen_moving_stall = RewTerm(
        func=mdp.moving_stall_penalty,
        weight=-0.3,
        params={"command_name": "base_velocity", "command_threshold": 0.2, "velocity_threshold": 0.1},
    )


    # =====
    # Gait reward
    test_gait_reward = RewTerm(
        func=mdp.GaitReward,
        weight=0.1,
        params={
            "tracking_contacts_shaped_force": -2.0,
            "tracking_contacts_shaped_vel": -2.0,
            "tracking_contacts_stance_force": -1.0,
            "gait_force_sigma": 25.0,
            "gait_vel_sigma": 0.25,
            "kappa_gait_probs": 0.05,
            "command_name": "gait_command",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*TIP"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*TIP"),
        },
    )
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time,
    #     weight=1.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*TIP"),
    #         "command_name": "base_velocity",
    #         "threshold": 0.25,
    #     },
    # )

    pen_base_height = RewTerm(
        func=mdp.base_com_height,
        params={
            "target_height": 0.55,
        },
        weight=-0.25 #-0.1,
    )
    pen_flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-0.25) #-0.1)

    # ===== Fixed Auxiliary =====
    # ===== 
    pen_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    pen_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # pen_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-5e-3) 
    # pen_feet_regulation = RewTerm(
    #     func=mdp.feet_regulation,
    #     weight=-0.001,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*TIP"]),
    #             "base_height_target": 0.55, "foot_radius": 0.03},
    # )
    pen_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*CALF", ".*THIGH"]),
            "threshold": 10.0,
        },
    )
    # pen_joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.5)
    # pen_joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-5.5e-6) 
    pen_combined_torque = RewTerm(
    func=mdp.combined_torque_penalty,
    weight=-0.1,
    params={"torque_safe_ratio": 0.6, "power": 6.0, "high_usage_weight": 0.2},
    )

    # pen_joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight= -1e-03) #-5.5e-5)
    pen_joint_accel = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7) 
    # pen_joint_accel = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7) 
    pen_action_rate = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.005, params={"max_value": 400.0}) #-0.03

    # pen_swing_height_error = RewTerm(
    #     func=mdp.pen_swing_height_error,
    #     weight=-2.,
    #     params={"command_name": "gait_command", "asset_cfg": SceneEntityCfg("robot", body_names=".*TIP"), "foot_radius": 0.03},
    # )
    pen_action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.0025) #-0.04
    # pen_hp_action_abs_excess = RewTerm(
    #     func=mdp.joint_action_abs_excess_l2,
    #     weight=-0.15,
    #     params={"joint_name_expr": ".*HP_JOINT", "threshold": 0.9, "max_value": 4.0},
    # )
    # pen_hp_action_delta_excess = RewTerm(
    #     func=mdp.joint_action_delta_excess_l2,
    #     weight=-0.08,
    #     params={"joint_name_expr": ".*HP_JOINT", "threshold": 0.6, "max_value": 20.0},
    # )
    # pen_hp_torque_rate_excess = RewTerm(
    #     func=mdp.JointTorqueRateExcessPenalty,
    #     weight=-0.01,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*HP_JOINT"),
    #         "threshold": 0.30,
    #         "max_value": 20.0,
    #     },
    # )
    pen_joint_powers = RewTerm(func=mdp.joint_powers_l1, weight=-2.5e-5)
    pen_joint_powers_var = RewTerm(func=mdp.joint_powers_var, weight=-1.5e-6)
    # pen_joint_default_pos = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1)
    pen_standing_joint_default_pos = RewTerm(func=mdp.stand_still_joint_deviation_l1, weight=-0.1) #-0.5)
    pen_standing_foot_contact = RewTerm(
        func=mdp.standing_foot_contact,
        weight=-0.01, #-0.05, 
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*TIP"),
            "lin_threshold": 0.05,
            "ang_threshold": 0.05,
            "force_threshold": 1.0,
        },
    )
    pen_standing_foot_height = RewTerm(
        func=mdp.standing_foot_height,
        weight=-0.01, #-0.05,
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
    #     weight=-0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*HR_JOINT")},
    # )

    pen_standing_vel = RewTerm(func=mdp.stand_still, weight=-0.1) #-1.0)
    
    # pen_feet_slide = RewTerm(
    #     func=mdp.feet_slide,
    #     weight=-0.1, #-0.15,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*TIP"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*TIP"),
    #     },
    # )

    # foot_landing_vel = RewTerm(
    #     func=mdp.foot_landing_vel,
    #     weight=-0.025, #-0.5,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*TIP"]),
    #             "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*TIP"]),
    #              "foot_radius": 0.03, "about_landing_threshold": 0.1},
    # )


@configclass
class ImplicitRoughRecoveryRewardsCfg(RewardsCfg):
    """Original implicit-rough rewards with dense BODY-contact recovery pressure."""

    pen_body_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="BODY"),
            "threshold": 10.0,
        },
    )
    pen_termination = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class LegacyRewardsCfg:
    """Reward contract captured from the 2026-07-07 ``사뿐사뿐`` run.

    This intentionally keeps the historical term names and weights instead of
    inheriting the later recovery/speed-tradeoff changes.  It is used only by
    the dedicated ``*-Legacy-v0`` task so the current implicit task remains
    unchanged.
    """

    rew_lin_vel_xy1 = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.1)},
    )
    rew_ang_vel_z1 = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.1)},
    )
    rew_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    rew_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    pen_nonzero_lin_vel_xy_error = RewTerm(
        func=mdp.nonzero_lin_vel_tracking_error_l2,
        weight=-0.25,
        params={"command_name": "base_velocity", "lin_threshold": 0.1, "max_error": 4.0},
    )
    pen_moving_stall = RewTerm(
        func=mdp.moving_stall_penalty,
        weight=-0.3,
        params={"command_name": "base_velocity", "command_threshold": 0.2, "velocity_threshold": 0.1},
    )
    test_gait_reward = RewTerm(
        func=mdp.GaitReward,
        weight=0.3,
        params={
            "tracking_contacts_shaped_force": -2.0,
            "tracking_contacts_shaped_vel": -2.0,
            "tracking_contacts_stance_force": -1.0,
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
        weight=-0.1,
        params={"target_height": 0.55},
    )
    pen_flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    pen_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    pen_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    pen_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*CALF", ".*THIGH"]), "threshold": 10.0},
    )
    pen_joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.5)
    pen_joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-5.5e-6)
    # Keep this term explicit in the legacy contract.  The 2026-07-07
    # ``사뿐사뿐`` run used these values, and inheriting it from RewardsCfg
    # would let later baseline tuning silently change the legacy task.
    pen_combined_torque = RewTerm(
        func=mdp.combined_torque_penalty,
        weight=-0.1,
        params={"torque_safe_ratio": 0.6, "power": 6.0, "high_usage_weight": 0.2},
    )
    pen_joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-3)
    pen_joint_accel = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    pen_action_rate = RewTerm(
        func=mdp.action_rate_l2_clamped, weight=-0.03, params={"max_value": 400.0}
    )
    pen_swing_height_error = RewTerm(
        func=mdp.pen_swing_height_error,
        weight=-2.0,
        params={
            "command_name": "gait_command",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*TIP"),
            "foot_radius": 0.03,
        },
    )
    pen_action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.04)
    # Intentionally omitted for the legacy action-space ablation:
    # ``사뿐사뿐`` used pen_hp_action_abs_excess to pull HP actions inside
    # approximately [-0.9, 0.9].
    pen_hp_action_delta_excess = RewTerm(
        func=mdp.joint_action_delta_excess_l2,
        weight=-0.08,
        params={"joint_name_expr": ".*HP_JOINT", "threshold": 0.6, "max_value": 20.0},
    )
    pen_hp_torque_rate_excess = RewTerm(
        func=mdp.JointTorqueRateExcessPenalty,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*HP_JOINT"), "threshold": 0.3, "max_value": 20.0},
    )
    pen_joint_powers = RewTerm(func=mdp.joint_powers_l1, weight=-2.5e-5)
    pen_joint_powers_var = RewTerm(func=mdp.joint_powers_var, weight=-2.5e-6)
    pen_joint_default_pos = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1)
    pen_standing_joint_default_pos = RewTerm(func=mdp.stand_still_joint_deviation_l1, weight=-0.5)
    pen_standing_foot_contact = RewTerm(
        func=mdp.standing_foot_contact,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*TIP"),
            "lin_threshold": 0.05,
            "ang_threshold": 0.05,
            "force_threshold": 1.0,
        },
    )
    pen_standing_foot_height = RewTerm(
        func=mdp.standing_foot_height,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*TIP"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "foot_radius": 0.03,
            "lin_threshold": 0.05,
            "ang_threshold": 0.05,
            "height_tolerance": 0.02,
        },
    )
    pen_hip_roll_pos = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*HR_JOINT")},
    )
    pen_standing_vel = RewTerm(func=mdp.stand_still, weight=-1.0)
    pen_feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.15,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*TIP"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*TIP"),
        },
    )
    foot_landing_vel = RewTerm(
        func=mdp.foot_landing_vel,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*TIP"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*TIP"]),
            "foot_radius": 0.03,
            "about_landing_threshold": 0.1,
        },
    )

@configclass
class VelocityRewardsCfg:
    """Isaac Lab velocity-locomotion rewards adapted to PongBot foot body names."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*TIP"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP"""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="BODY"), "threshold": 1.0},
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


@configclass
class ImplicitRoughFrontierCurriculumCfg:
    """Guarded obstacle curriculum used by the fifth experiment."""

    terrain_levels = CurrTerm(
        func=mdp.GuardedObstacleCurriculum,
        params={
            "terrain_type_names": ("boxes_hard", "pyramid_stairs_hard", "pyramid_stairs_inv_hard"),
            "command_name": "base_velocity",
            "command_threshold": 0.2,
            "promote_distance": 3.0,
            "demote_progress_ratio": 0.35,
            "minimum_demotion_distance": 0.8,
            "challenge_levels": (1, 2, 3),
            "challenge_probabilities": (0.43, 0.34, 0.23),
            "warmup_steps": 50_000,
            "initial_challenge_ratio": 0.20,
            "final_challenge_ratio": 0.35,
        },
    )


@configclass
class ImplicitRoughFrontierRewardsCfg(RewardsCfg):
    """Fifth-experiment rewards with staged TIP and leg recovery credit."""

    obstacle_contact_recovery = RewTerm(
        func=mdp.ObstacleContactRecoveryReward,
        weight=0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*TIP", ".*CALF", ".*THIGH"]),
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "force_threshold": 10.0,
            "tip_horizontal_ratio": 0.75,
            "tip_stall_speed": 0.12,
            "minimum_contact_steps": 3,
            "recovery_window_steps": 150,
            "short_distance": 0.10,
            "full_distance": 0.25,
            "release_credit": 0.02,
            "short_credit": 0.06,
            "full_credit": 0.14,
            "episode_credit_cap": 0.5,
            "recontact_tolerance_steps": 5,
            "cooldown_steps": 150,
            "command_threshold": 0.2,
        },
    )



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
        # Keep the reset-sampled locomotion commands for the whole episode.
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        self.commands.gait_command.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        self.sim.render_interval = 2 * self.decimation
        # simulation settings
        self.sim.dt = 0.002
        self.seed = 42
        self.scene.fl_foot_height_scanner = None
        self.scene.fr_foot_height_scanner = None
        self.scene.rl_foot_height_scanner = None
        self.scene.rr_foot_height_scanner = None
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
