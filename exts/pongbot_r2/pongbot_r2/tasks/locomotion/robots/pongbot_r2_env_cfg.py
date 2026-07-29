"""Robot-specific PongBot R2 locomotion environment variants.

@version 0.0.11
@update 2026-07-13: Give implicit rough policies one second to recover from BODY contact.
@update 2026-07-13: Connect the Frontier task to guarded fifth-experiment learning.
"""

import math

from isaaclab.utils import configclass

from pongbot_r2.assets.config.pongbot_r2 import PONGBOT_R2_CFG
from pongbot_r2.tasks.locomotion.cfg.pongbot_r2 import terrains_cfg as pongbot_r2_terrains
from pongbot_r2.tasks.locomotion.cfg.pongbot_r2.normal_base_env_cfg import PFEnvCfg as PFR2EnvCfg
from pongbot_r2.tasks.locomotion.cfg.pongbot_r2_implicit import terrains_cfg as pongbot_r2_implicit_terrains
from pongbot_r2.tasks.locomotion.cfg.pongbot_r2_implicit.implicit_base_env_cfg import (
    PFEnvCfg as PFImplicitEnvCfg,
    EventsCfg,
    ImplicitRoughEventsCfg,
    LegacyImplicitRoughEventsCfg,
    LegacyActionsCfg,
    LegacyCommandCfg,
    LegacyObservarionsCfg,
    ImplicitRoughFrontierCurriculumCfg,
    ImplicitRoughFrontierRewardsCfg,
    ImplicitRoughRecoveryRewardsCfg,
    LegacyRewardsCfg,
    VelocityRewardsCfg,
    Phase2ObservarionsCfg,
    Phase2LoadAdaptiveObservarionsCfg,
    Phase2EventsCfg,
    Phase2ComEventsCfg,
    Phase2AbruptEventsCfg,
)
from pongbot_r2.tasks.locomotion.cfg.pongbot_r2_mass import terrains_cfg as pongbot_r2_mass_terrains
from pongbot_r2.tasks.locomotion.cfg.pongbot_r2_mass.mass_base_env_cfg import PFEnvCfg as PFMassEnvCfg
from pongbot_r2.tasks.locomotion.cfg.pongbot_r2_imu import terrains_cfg as pongbot_r2_imu_terrains
from pongbot_r2.tasks.locomotion.cfg.pongbot_r2_imu.without_imu_base_env_cfg import PFEnvCfg as PFIMUEnvCfg

from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from pongbot_r2.tasks.locomotion import mdp
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg


def _make_reward_height_scanner(update_period: float) -> RayCasterCfg:
    """Create a height scanner used only for terrain-aware rewards."""
    return RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/BODY",
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=update_period,
    )


def _make_foot_height_scanner(foot_name: str, update_period: float, debug_vis: bool = False) -> RayCasterCfg:
    """Create a foot-centered scanner for visual inspection of local terrain around a foot."""
    return RayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{foot_name}_TIP",
        attach_yaw_only=True,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.3]),
        debug_vis=debug_vis,
        visualizer_cfg=RAY_CASTER_MARKER_CFG.replace(prim_path=f"/Visuals/FootHeightScanner/{foot_name}"),
        mesh_prim_paths=["/World/ground"],
        update_period=update_period,
    )


######################
# Pointfoot Base Environment
######################


@configclass
class PFBaseEnvCfg(PFR2EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = PONGBOT_R2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            ".*HR_JOINT": 0.0,
            ".*HP_JOINT": 0.8,
            ".*KN_JOINT": -1.5,
        }

        self.events.add_base_mass.params["asset_cfg"].body_names = "BODY"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)

        self.terminations.base_contact.params["sensor_cfg"].body_names = "BODY"
        
        # update viewport camera
        self.viewer.origin_type = "env"
        self.viewer.eye = (100.0, 100.0, 100.0)

    def _blind_rough_terrain_cfg(self):
        return pongbot_r2_terrains.BLIND_ROUGH_TERRAINS_CFG

    def _blind_rough_terrain_play_cfg(self):
        return pongbot_r2_terrains.BLIND_ROUGH_TERRAINS_PLAY_CFG

    def _stairs_terrain_cfg(self):
        return pongbot_r2_terrains.STAIRS_TERRAINS_CFG

    def _stairs_terrain_play_cfg(self):
        return pongbot_r2_terrains.STAIRS_TERRAINS_PLAY_CFG


@configclass
class PFBaseEnvCfg_PLAY(PFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.push_robot = None
        # remove random base mass addition event
        self.events.add_base_mass = None


@configclass
class PFBaseIMUEnvCfg(PFIMUEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = PONGBOT_R2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            ".*HR_JOINT": 0.0,
            ".*HP_JOINT": 0.64, #0.8,
            ".*KN_JOINT": -1.25 #-1.5,
        }

        self.events.add_base_mass.params["asset_cfg"].body_names = "BODY"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)
        self.terminations.base_contact.params["sensor_cfg"].body_names = "BODY"
        self.viewer.origin_type = "world"
        self.viewer.eye = (100.0, 100.0, 100.0)

    def _blind_rough_terrain_cfg(self):
        return pongbot_r2_imu_terrains.BLIND_ROUGH_TERRAINS_CFG

    def _blind_rough_terrain_play_cfg(self):
        return pongbot_r2_imu_terrains.BLIND_ROUGH_TERRAINS_PLAY_CFG

    def _stairs_terrain_cfg(self):
        return pongbot_r2_imu_terrains.STAIRS_TERRAINS_CFG

    def _stairs_terrain_play_cfg(self):
        return pongbot_r2_imu_terrains.STAIRS_TERRAINS_PLAY_CFG


@configclass
class PFBaseIMUEnvCfg_PLAY(PFBaseIMUEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None


@configclass
class PFBaseMassEnvCfg(PFMassEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = PONGBOT_R2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            ".*HR_JOINT": 0.0,
            ".*HP_JOINT": 0.8,
            ".*KN_JOINT": -1.5,
        }

        self.events.add_base_mass.params["asset_cfg"].body_names = "BODY"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)

        self.terminations.base_contact.params["sensor_cfg"].body_names = "BODY"

        self.viewer.origin_type = "env"

    def _blind_rough_terrain_cfg(self):
        return pongbot_r2_mass_terrains.BLIND_ROUGH_TERRAINS_CFG

    def _blind_rough_terrain_play_cfg(self):
        return pongbot_r2_mass_terrains.BLIND_ROUGH_TERRAINS_PLAY_CFG

    def _stairs_terrain_cfg(self):
        return pongbot_r2_mass_terrains.STAIRS_TERRAINS_CFG

    def _stairs_terrain_play_cfg(self):
        return pongbot_r2_mass_terrains.STAIRS_TERRAINS_PLAY_CFG


@configclass
class PFBaseMassEnvCfg_PLAY(PFBaseMassEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None


@configclass
class PFBaseImplicitEnvCfg(PFImplicitEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = PONGBOT_R2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            ".*HR_JOINT": 0.0,
            ".*HP_JOINT": 0.8,
            ".*KN_JOINT": -1.5,
        }

        self.observations.obsHistory.history_length = 5
        self.observations.obsHistory.flatten_history_dim = False
        # self.observations.obsHistory.gait_phase = self.observations.policy.gait_phase
        # self.observations.obsHistory.gait_command = self.observations.policy.gait_command

        self.events.add_base_mass.params["asset_cfg"].body_names = "BODY"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)

        self.terminations.base_contact.params["sensor_cfg"].body_names = "BODY"

        self.viewer.origin_type = "world"
        self.viewer.eye = (100.0, 100.0, 100.0)

    def _blind_rough_terrain_cfg(self):
        return pongbot_r2_implicit_terrains.BLIND_ROUGH_TERRAINS_CFG

    def _implicit_rough_terrain_cfg(self):
        return pongbot_r2_implicit_terrains.IMPLICIT_ROUGH_TERRAINS_CFG

    def _blind_rough_terrain_play_cfg(self):
        return pongbot_r2_implicit_terrains.BLIND_ROUGH_TERRAINS_PLAY_CFG

    def _blind_rough_pca_terrain_play_cfg(self):
        return pongbot_r2_implicit_terrains.BLIND_ROUGH_PCA_TERRAINS_PLAY_CFG

    def _tcp_latent_eval_terrain_play_cfg(self):
        return pongbot_r2_implicit_terrains.TCP_LATENT_EVAL_TERRAINS_PLAY_CFG

    def _stairs_terrain_cfg(self):
        return pongbot_r2_implicit_terrains.STAIRS_TERRAINS_CFG

    def _stairs_terrain_play_cfg(self):
        return pongbot_r2_implicit_terrains.STAIRS_TERRAINS_PLAY_CFG

    def _implicit_stair_terrain_cfg(self):
        return pongbot_r2_implicit_terrains.IMPLICIT_STAIR_TERRAINS_CFG

    def _implicit_stair_terrain_play_cfg(self):
        return pongbot_r2_implicit_terrains.IMPLICIT_STAIR_TERRAINS_PLAY_CFG


@configclass
class PFBaseImplicitEnvCfg_PLAY(PFBaseImplicitEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None


############################
# Pointfoot Blind Flat Environment
############################


@configclass
class PFBlindFlatEnvCfg(PFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.curriculum.terrain_levels = None


@configclass
class PFBlindFlatEnvCfg_PLAY(PFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.curriculum.terrain_levels = None


@configclass
class PFBlindFlatIMUEnvCfg(PFBaseIMUEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.critic.heights = None
        self.curriculum.terrain_levels = None


@configclass
class PFBlindFlatIMUEnvCfg_PLAY(PFBaseIMUEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.critic.heights = None
        self.curriculum.terrain_levels = None


#############################
# Pointfoot Blind Rough Environment
#############################


@configclass
class PFBlindRoughEnvCfg(PFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = _make_reward_height_scanner(self.decimation * self.sim.dt)
        # self.observations.policy.heights = None
        # self.observations.critic.heights = None
        self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = self._blind_rough_terrain_cfg()


@configclass
class PFBlindRoughEnvCfg_PLAY(PFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = _make_reward_height_scanner(self.decimation * self.sim.dt)
        # self.observations.policy.heights = None
        # self.observations.critic.heights = None
        self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = self._blind_rough_terrain_play_cfg()

@configclass
class PFBlindRoughMassEnvCfg(PFBaseMassEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = _make_reward_height_scanner(self.decimation * self.sim.dt)
        # self.observations.policy.heights = None
        # self.observations.critic.heights = None
        self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = self._blind_rough_terrain_cfg()


@configclass
class PFBlindRoughMassEnvCfg_PLAY(PFBaseMassEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = _make_reward_height_scanner(self.decimation * self.sim.dt)
        # self.observations.policy.heights = None
        # self.observations.critic.heights = None
        self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = self._blind_rough_terrain_play_cfg()


@configclass
class PFBlindFlatImplicitEnvCfg(PFBaseImplicitEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = _make_reward_height_scanner(self.decimation * self.sim.dt)
        self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
        self.curriculum.terrain_levels = None


@configclass
class PFBlindFlatImplicitEnvCfg_PLAY(PFBaseImplicitEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = _make_reward_height_scanner(self.decimation * self.sim.dt)
        self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
        self.curriculum.terrain_levels = None


@configclass
class PFBlindRoughImplicitEnvCfg(PFBaseImplicitEnvCfg):
    rewards: ImplicitRoughRecoveryRewardsCfg = ImplicitRoughRecoveryRewardsCfg()
    # Recovery rollback: replace the previous line with ``rewards: LegacyRewardsCfg = LegacyRewardsCfg()``.
    # Velocity baseline: replace the active rewards line with the following one.
    # rewards: VelocityRewardsCfg = VelocityRewardsCfg()
    events: ImplicitRoughEventsCfg = ImplicitRoughEventsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.terminations.base_contact.func = mdp.DelayedIllegalContact
        self.terminations.base_contact.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="BODY"),
            "threshold": 10.0,
            "delay_s": 1.0,  # Recovery rollback: 0.15.
        }

        self.scene.height_scanner = _make_reward_height_scanner(self.decimation * self.sim.dt)
        self.scene.fl_foot_height_scanner = _make_foot_height_scanner("FL", self.decimation * self.sim.dt)
        self.scene.fr_foot_height_scanner = _make_foot_height_scanner("FR", self.decimation * self.sim.dt)
        self.scene.rl_foot_height_scanner = _make_foot_height_scanner("RL", self.decimation * self.sim.dt)
        self.scene.rr_foot_height_scanner = _make_foot_height_scanner("RR", self.decimation * self.sim.dt)
        if hasattr(self.rewards, "pen_base_height"):
            self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = self._implicit_rough_terrain_cfg()


@configclass
class PFBlindRoughImplicitLegacyEnvCfg(PFBlindRoughImplicitEnvCfg):
    """Exact reward/event/reset variant for the 2026-07-07 run.

    Kept as a separate task so the current recovery implicit task is not
    changed when reproducing the historical checkpoint experiment.
    """

    rewards: LegacyRewardsCfg = LegacyRewardsCfg()
    events: LegacyImplicitRoughEventsCfg = LegacyImplicitRoughEventsCfg()
    actions: LegacyActionsCfg = LegacyActionsCfg()
    commands: LegacyCommandCfg = LegacyCommandCfg()
    observations: LegacyObservarionsCfg = LegacyObservarionsCfg()

    def _implicit_rough_terrain_cfg(self):
        return pongbot_r2_implicit_terrains.LEGACY_IMPLICIT_ROUGH_TERRAINS_CFG

    def __post_init__(self):
        super().__post_init__()
        # Historical run used immediate IsaacLab illegal-contact termination,
        # not the later DelayedIllegalContact recovery termination.
        self.terminations.base_contact.func = mdp.illegal_contact
        self.terminations.base_contact.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="BODY"),
            "threshold": 1.0,
        }
        self.scene.terrain.max_init_terrain_level = 4
        self.commands.gait_command.resampling_time_range = (5.0, 5.0)
        self.commands.base_velocity.rel_standing_envs = 0.15
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 1.0
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.2, 1.2),
            lin_vel_y=(-1.2, 1.2),
            ang_vel_z=(-1.2, 1.2),
            heading=(-math.pi, math.pi),
        )


@configclass
class PFBlindRoughImplicitFrontierEnvCfg(PFBlindRoughImplicitEnvCfg):
    """Fifth obstacle experiment with guarded challenge replay and recovery."""

    curriculum: ImplicitRoughFrontierCurriculumCfg = ImplicitRoughFrontierCurriculumCfg()
    rewards: ImplicitRoughFrontierRewardsCfg = ImplicitRoughFrontierRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.terminations.base_contact.params["delay_s"] = 0.15
        self.scene.terrain.max_init_terrain_level = 2
        self.observations.obsHistory.history_length = 5
        self.commands.base_velocity.ranges.lin_vel_x = (0.25, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.15, 0.15)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)
        self.rewards.test_gait_reward.weight = 0.0
        self.rewards.pen_swing_height_error.weight = -0.5
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.pen_joint_torque.weight = -4.5e-6
        self.rewards.pen_joint_vel_l2.weight = -7.5e-4
        self.rewards.pen_joint_powers_var.weight = -1.25e-6
        self.rewards.pen_joint_default_pos.weight = -0.015
        self.rewards.pen_hip_roll_pos.weight = -0.05
        self.rewards.foot_landing_vel.weight = -0.2


@configclass
class PFBlindRoughImplicitTCPEnvCfg(PFBlindRoughImplicitEnvCfg):
    rewards: LegacyRewardsCfg = LegacyRewardsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.terminations.base_contact.params["delay_s"] = 0.15
        self.scene.terrain.terrain_generator = self._blind_rough_terrain_cfg()
        self.observations.obsHistory.history_length = 10


@configclass
class PFBlindRoughImplicitEnvCfg_PLAY(PFBaseImplicitEnvCfg_PLAY):
    rewards: LegacyRewardsCfg = LegacyRewardsCfg()
    # Velocity baseline: comment the previous line and uncomment this one.
    # rewards: VelocityRewardsCfg = VelocityRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = _make_reward_height_scanner(self.decimation * self.sim.dt)
        self.scene.fl_foot_height_scanner = _make_foot_height_scanner("FL", self.decimation * self.sim.dt, debug_vis=True)
        self.scene.fr_foot_height_scanner = _make_foot_height_scanner("FR", self.decimation * self.sim.dt, debug_vis=True)
        self.scene.rl_foot_height_scanner = _make_foot_height_scanner("RL", self.decimation * self.sim.dt, debug_vis=True)
        self.scene.rr_foot_height_scanner = _make_foot_height_scanner("RR", self.decimation * self.sim.dt, debug_vis=True)
        if hasattr(self.rewards, "pen_base_height"):
            self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = self._blind_rough_terrain_play_cfg()


@configclass
class PFBlindRoughImplicitFrontierEnvCfg_PLAY(PFBlindRoughImplicitEnvCfg_PLAY):
    """Play configuration matching the third experiment's history contract."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.obsHistory.history_length = 5
        self.rewards.test_gait_reward.weight = 0.0
        self.rewards.pen_swing_height_error.weight = -0.5
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.pen_joint_torque.weight = -4.5e-6
        self.rewards.pen_joint_vel_l2.weight = -7.5e-4
        self.rewards.pen_joint_powers_var.weight = -1.25e-6
        self.rewards.pen_joint_default_pos.weight = -0.015
        self.rewards.pen_hip_roll_pos.weight = -0.05
        self.rewards.foot_landing_vel.weight = -0.2


@configclass
class PFBlindRoughImplicitPcaEnvCfg_PLAY(PFBlindRoughImplicitEnvCfg_PLAY):
    rewards: LegacyRewardsCfg = LegacyRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = self._blind_rough_pca_terrain_play_cfg()


@configclass
class PFBlindRoughImplicitLatentEvalEnvCfg_PLAY(PFBlindRoughImplicitEnvCfg_PLAY):
    rewards: LegacyRewardsCfg = LegacyRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = self._tcp_latent_eval_terrain_play_cfg()


@configclass
class PFBlindRoughImplicitTCPLatentEnvCfg_PLAY(PFBlindRoughImplicitEnvCfg_PLAY):
    rewards: LegacyRewardsCfg = LegacyRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.observations.obsHistory.history_length = 10
        self.scene.terrain.terrain_generator = self._tcp_latent_eval_terrain_play_cfg()


@configclass
class PFBlindImplicitStairEnvCfg(PFBaseImplicitEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = _make_reward_height_scanner(self.decimation * self.sim.dt)
        self.scene.fl_foot_height_scanner = _make_foot_height_scanner("FL", self.decimation * self.sim.dt)
        self.scene.fr_foot_height_scanner = _make_foot_height_scanner("FR", self.decimation * self.sim.dt)
        self.scene.rl_foot_height_scanner = _make_foot_height_scanner("RL", self.decimation * self.sim.dt)
        self.scene.rr_foot_height_scanner = _make_foot_height_scanner("RR", self.decimation * self.sim.dt)
        self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.25, 0.75)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.05, 0.05)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.25, 0.25)
        self.commands.gait_command.ranges.frequencies = (1.3, 1.7)
        self.commands.gait_command.ranges.durations = (0.55, 0.65)
        self.commands.gait_command.ranges.swing_height = (0.08, 0.12)
        self.events.reset_robot_base.params["pose_range"]["yaw"] = (-0.1, 0.1)
        self.rewards.foot_landing_vel.weight = -0.3

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = 4
        self.scene.terrain.terrain_generator = self._implicit_stair_terrain_cfg()


@configclass
class PFBlindImplicitStairEnvCfg_PLAY(PFBaseImplicitEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = _make_reward_height_scanner(self.decimation * self.sim.dt)
        self.scene.fl_foot_height_scanner = _make_foot_height_scanner("FL", self.decimation * self.sim.dt, debug_vis=True)
        self.scene.fr_foot_height_scanner = _make_foot_height_scanner("FR", self.decimation * self.sim.dt, debug_vis=True)
        self.scene.rl_foot_height_scanner = _make_foot_height_scanner("RL", self.decimation * self.sim.dt, debug_vis=True)
        self.scene.rr_foot_height_scanner = _make_foot_height_scanner("RR", self.decimation * self.sim.dt, debug_vis=True)
        self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.events.reset_robot_base.params["pose_range"]["yaw"] = (0.0, 0.0)

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = self._implicit_stair_terrain_play_cfg()


@configclass
class PFBlindRoughPhase2AdaptiveEnvCfg(PFBaseImplicitEnvCfg):
    observations: Phase2ObservarionsCfg = Phase2ObservarionsCfg()
    events: Phase2EventsCfg = Phase2EventsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = _make_reward_height_scanner(self.decimation * self.sim.dt)
        self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = self._blind_rough_terrain_cfg()


@configclass
class PFBlindRoughPhase2AdaptiveEnvCfg_PLAY(PFBaseImplicitEnvCfg_PLAY):
    observations: Phase2ObservarionsCfg = Phase2ObservarionsCfg()
    events: Phase2EventsCfg = Phase2EventsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = _make_reward_height_scanner(self.decimation * self.sim.dt)
        self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = self._blind_rough_terrain_play_cfg()


@configclass
class PFBlindRoughPhase2LoadAdaptiveEnvCfg(PFBlindRoughPhase2AdaptiveEnvCfg):
    observations: Phase2LoadAdaptiveObservarionsCfg = Phase2LoadAdaptiveObservarionsCfg()


@configclass
class PFBlindRoughPhase2LoadAdaptiveEnvCfg_PLAY(PFBlindRoughPhase2AdaptiveEnvCfg_PLAY):
    observations: Phase2LoadAdaptiveObservarionsCfg = Phase2LoadAdaptiveObservarionsCfg()


@configclass
class PFBlindRoughPhase2LoadAdaptiveComEnvCfg(PFBlindRoughPhase2LoadAdaptiveEnvCfg):
    events: Phase2ComEventsCfg = Phase2ComEventsCfg()


@configclass
class PFBlindRoughPhase2LoadAdaptiveComEnvCfg_PLAY(PFBlindRoughPhase2LoadAdaptiveEnvCfg_PLAY):
    events: Phase2ComEventsCfg = Phase2ComEventsCfg()


@configclass
class PFBlindRoughPhase2LoadAdaptiveAbruptEnvCfg(PFBlindRoughPhase2LoadAdaptiveEnvCfg):
    events: Phase2AbruptEventsCfg = Phase2AbruptEventsCfg()


@configclass
class PFBlindRoughPhase2LoadAdaptiveAbruptEnvCfg_PLAY(PFBlindRoughPhase2LoadAdaptiveEnvCfg_PLAY):
    events: Phase2AbruptEventsCfg = Phase2AbruptEventsCfg()


@configclass
class PFBlindRoughIMUEnvCfg(PFBaseIMUEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = _make_reward_height_scanner(self.decimation * self.sim.dt)
        self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = self._blind_rough_terrain_cfg()


@configclass
class PFBlindRoughIMUEnvCfg_PLAY(PFBaseIMUEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = _make_reward_height_scanner(self.decimation * self.sim.dt)
        self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = self._blind_rough_terrain_play_cfg()

##############################
# Pointfoot Blind Stairs Environment
##############################


@configclass
class PFBlindStairEnvCfg(PFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = _make_reward_height_scanner(self.decimation * self.sim.dt)
        self.observations.policy.heights = None
        self.observations.critic.heights = None
        self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-math.pi / 6, math.pi / 6)

        self.rewards.rew_lin_vel_xy.weight = 2.0
        self.rewards.rew_ang_vel_z.weight = 1.0
        # self.rewards.pen_lin_vel_z.weight = -0.5
        # self.rewards.pen_ang_vel_xy.weight = -0.001
        # self.rewards.pen_action_rate.weight = -0.0025
        # self.rewards.pen_flat_orientation.weight = -2.0
        # self.rewards.pen_undesired_contacts.weight = -1.0

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = self._stairs_terrain_cfg()


@configclass
class PFBlindStairEnvCfg_PLAY(PFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = _make_reward_height_scanner(self.decimation * self.sim.dt)
        self.observations.policy.heights = None
        self.observations.critic.heights = None
        self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)

        self.events.reset_robot_base.params["pose_range"]["yaw"] = (-0.0, 0.0)

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = self._stairs_terrain_play_cfg().replace(difficulty_range=(0.5, 0.5))


@configclass
class PFBlindStairIMUEnvCfg(PFBaseIMUEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = _make_reward_height_scanner(self.decimation * self.sim.dt)
        self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-math.pi / 6, math.pi / 6)

        self.rewards.rew_lin_vel_xy2.weight = 2.0
        self.rewards.rew_ang_vel_z2.weight = 1.0

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = self._stairs_terrain_cfg()


@configclass
class PFBlindStairIMUEnvCfg_PLAY(PFBaseIMUEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = _make_reward_height_scanner(self.decimation * self.sim.dt)
        self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)

        self.events.reset_robot_base.params["pose_range"]["yaw"] = (-0.0, 0.0)

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = self._stairs_terrain_play_cfg().replace(difficulty_range=(0.5, 0.5))


#############################
# Pointfoot Stair Environment with height scan
#############################

@configclass
class PFStairEnvCfgv1(PFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/BODY",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
                    noise=GaussianNoise(mean=0.0, std=0.01),
                    clip = (0.0, 10.0),
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip = (0.0, 10.0),
        )
        
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = self._stairs_terrain_cfg()


@configclass
class PFStairEnvCfgv1_PLAY(PFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/BODY",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip = (0.0, 10.0),
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip = (0.0, 10.0),
        )
        
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = self._stairs_terrain_play_cfg().replace(difficulty_range=(0.5, 0.5))
