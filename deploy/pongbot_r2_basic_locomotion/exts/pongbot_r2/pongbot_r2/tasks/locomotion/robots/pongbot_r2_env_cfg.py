"""Robot-specific Flat/Rough environment variants for source distribution.

@version 0.0.1
@update 2026-07-29: Isolate nominal point-foot Flat/Rough train and play environments.
"""

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from pongbot_r2.assets.config.pongbot_r2 import PONGBOT_R2_CFG
from pongbot_r2.tasks.locomotion.cfg.pongbot_r2 import terrains_cfg
from pongbot_r2.tasks.locomotion.cfg.pongbot_r2.normal_base_env_cfg import PFEnvCfg


def _make_reward_height_scanner(update_period: float) -> RayCasterCfg:
    return RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/BODY",
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=update_period,
    )


@configclass
class PFBaseEnvCfg(PFEnvCfg):
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
        self.viewer.eye = (100.0, 100.0, 100.0)


@configclass
class PFBaseEnvCfg_PLAY(PFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None


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
class PFBlindRoughEnvCfg(PFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = _make_reward_height_scanner(self.decimation * self.sim.dt)
        self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = terrains_cfg.BLIND_ROUGH_TERRAINS_CFG


@configclass
class PFBlindRoughEnvCfg_PLAY(PFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = _make_reward_height_scanner(self.decimation * self.sim.dt)
        self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = terrains_cfg.BLIND_ROUGH_TERRAINS_PLAY_CFG
