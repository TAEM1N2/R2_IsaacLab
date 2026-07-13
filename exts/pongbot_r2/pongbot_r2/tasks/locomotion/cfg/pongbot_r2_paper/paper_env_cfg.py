"""Standalone R2 environment for the barrier-based locomotion paper method.

@version 0.0.5
@update 2026-07-13: Add R2-safe action authority and moving-success rough-terrain curricula.
@update 2026-07-13: Use only the paper's rough-trot terrain family and full reported difficulty range.
"""

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import DomeLightCfg, RigidBodyMaterialCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from pongbot_r2.assets.config.pongbot_r2 import (
    HP_JOINT_DAMPING_RANGE,
    HP_JOINT_STIFFNESS_RANGE,
    HR_JOINT_DAMPING_RANGE,
    HR_JOINT_STIFFNESS_RANGE,
    KNEE_JOINT_DAMPING_RANGE,
    KNEE_JOINT_STIFFNESS_RANGE,
    PONGBOT_R2_CFG,
)
from pongbot_r2.tasks.locomotion import mdp

from .terrains_cfg import PAPER_BARRIER_ROUGH_TERRAINS_CFG


def _foot_scanner(foot_name: str) -> RayCasterCfg:
    return RayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{foot_name}",
        attach_yaw_only=True,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        pattern_cfg=patterns.GridPatternCfg(resolution=0.025, size=(0.10, 0.10)),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


@configclass
class PaperBarrierSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=PAPER_BARRIER_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    robot = PONGBOT_R2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    fl_foot_scanner = _foot_scanner("FL_TIP")
    fr_foot_scanner = _foot_scanner("FR_TIP")
    rl_foot_scanner = _foot_scanner("RL_TIP")
    rr_foot_scanner = _foot_scanner("RR_TIP")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=4,
        track_air_time=True,
        update_period=0.0,
    )
    light = AssetBaseCfg(prim_path="/World/skyLight", spawn=DomeLightCfg(intensity=750.0))


@configclass
class PaperCommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        class_type=mdp.PaperVelocityCommand,
        asset_name="robot",
        resampling_time_range=(4.0, 4.0),
        rel_standing_envs=0.10,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.5),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class PaperActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*HR_JOINT", ".*HP_JOINT", ".*KN_JOINT"],
        scale=0.25,
        clip={".*": (-2.0, 2.0)},
        use_default_offset=True,
    )


@configclass
class PaperObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        proprioception = ObsTerm(
            func=mdp.PaperProprioception,
            params={"asset_cfg": SceneEntityCfg("robot"), "action_scale": 0.25, "gait_period": 0.72},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CommandsCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class EstimatorTargetCfg(ObsGroup):
        target = ObsTerm(
            func=mdp.paper_estimator_target,
            params={"asset_cfg": SceneEntityCfg("robot"), "sensor_cfg": SceneEntityCfg("contact_forces")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    obsHistory: PolicyCfg = PolicyCfg()
    commands: CommandsCfg = CommandsCfg()
    estimator_target: EstimatorTargetCfg = EstimatorTargetCfg()
    critic: EstimatorTargetCfg = EstimatorTargetCfg()


@configclass
class PaperEventsCfg:
    robot_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 32,
        },
    )
    hr_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*HR_JOINT"),
            "stiffness_distribution_params": HR_JOINT_STIFFNESS_RANGE,
            "damping_distribution_params": HR_JOINT_DAMPING_RANGE,
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    hp_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*HP_JOINT"),
            "stiffness_distribution_params": HP_JOINT_STIFFNESS_RANGE,
            "damping_distribution_params": HP_JOINT_DAMPING_RANGE,
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    knee_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*KN_JOINT"),
            "stiffness_distribution_params": KNEE_JOINT_STIFFNESS_RANGE,
            "damping_distribution_params": KNEE_JOINT_DAMPING_RANGE,
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    reset_base = EventTerm(
        func=mdp.paper_reset_root_state_curriculum,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reset_joints = EventTerm(
        func=mdp.paper_reset_joints_curriculum,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    zero_small_commands = EventTerm(
        func=mdp.zero_small_velocity_commands,
        mode="interval",
        interval_range_s=(0.01, 0.01),
        params={
            "command_name": "base_velocity",
            "linear_threshold": 0.20,
            "angular_threshold": 0.20,
        },
    )


@configclass
class PaperRewardsCfg:
    paper_standard = RewTerm(
        func=mdp.PaperStandardReward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "action_scale": 0.25,
            "foot_position_weight": 1.0,
            "height_difference_weight": 1.0,
            "torque_normalized_weight": 1.0,
        },
    )
    paper_barrier = RewTerm(
        func=mdp.PaperBarrierReward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "gait_period": 0.72,
            "alpha": 0.10,
        },
    )


@configclass
class PaperTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["BODY", ".*THIGH"]),
            "threshold": 1.0,
        },
    )


@configclass
class PaperCurriculumCfg:
    terrain_competence = CurrTerm(
        func=mdp.PaperTerrainDifficultyCurriculum,
        params={
            "anchor_max_level": 2,
            "probe_min_level": 5,
            "success_progress_ratio": 0.50,
            "failure_progress_ratio": 0.20,
            "success_mean_error": 0.40,
            "failure_mean_error": 0.80,
        },
    )


@configclass
class PaperBarrierRoughEnvCfg(ManagerBasedRLEnvCfg):
    scene: PaperBarrierSceneCfg = PaperBarrierSceneCfg(num_envs=400, env_spacing=2.5)
    observations: PaperObservationsCfg = PaperObservationsCfg()
    actions: PaperActionsCfg = PaperActionsCfg()
    commands: PaperCommandsCfg = PaperCommandsCfg()
    rewards: PaperRewardsCfg = PaperRewardsCfg()
    terminations: PaperTerminationsCfg = PaperTerminationsCfg()
    events: PaperEventsCfg = PaperEventsCfg()
    curriculum: PaperCurriculumCfg = PaperCurriculumCfg()

    def __post_init__(self):
        self.decimation = 5
        self.episode_length_s = 4.0
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.seed = 42
        self.scene.contact_forces.update_period = self.sim.dt
        for scanner_name in (
            "fl_foot_scanner",
            "fr_foot_scanner",
            "rl_foot_scanner",
            "rr_foot_scanner",
        ):
            getattr(self.scene, scanner_name).update_period = self.decimation * self.sim.dt


__all__ = ["PaperBarrierRoughEnvCfg"]
