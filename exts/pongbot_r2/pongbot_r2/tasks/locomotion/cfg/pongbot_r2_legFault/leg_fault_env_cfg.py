from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from pongbot_r2.assets.config.pongbot_r2 import PONGBOT_R2_FLKN_FAULT_CFG, PONGBOT_R2_FL_LEG_FAULT_CFG
from pongbot_r2.tasks.locomotion import mdp
from pongbot_r2.tasks.locomotion.cfg.pongbot_r2_implicit.implicit_base_env_cfg import PFEnvCfg as PFImplicitEnvCfg
from pongbot_r2.tasks.locomotion.cfg.pongbot_r2_legFault.terrains_cfg import (
    BLIND_ROUGH_TERRAINS_CFG,
    BLIND_ROUGH_TERRAINS_PLAY_CFG,
)


FL_KN_FAULT_JOINT_NAMES = ["FL_KN_JOINT"]
FL_LEG_FAULT_JOINT_NAMES = ["FL_HR_JOINT", "FL_HP_JOINT", "FL_KN_JOINT"]


def _make_reward_height_scanner(update_period: float) -> RayCasterCfg:
    return RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/BODY",
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=update_period,
    )


def _set_obs_term_joint_fault(term, joint_names: list[str]) -> None:
    term.params = dict(term.params or {})
    term.params["joint_names"] = joint_names


def _apply_joint_fault_observations(observations, joint_names: list[str]) -> None:
    for group_name in ("policy", "critic", "obsHistory"):
        group = getattr(observations, group_name, None)
        if group is None:
            continue
        group.joint_pos.func = mdp.joint_pos_rel_zero_joints
        _set_obs_term_joint_fault(group.joint_pos, joint_names)
        group.joint_vel.func = mdp.joint_vel_zero_joints
        _set_obs_term_joint_fault(group.joint_vel, joint_names)
        group.last_action.func = mdp.last_action_zero_joints
        _set_obs_term_joint_fault(group.last_action, joint_names)


@configclass
class PFBaseImplicitJointFaultEnvCfg(PFImplicitEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = self._fault_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            ".*HR_JOINT": 0.0,
            ".*HP_JOINT": 0.8,
            ".*KN_JOINT": -1.5,
        }

        self.observations.obsHistory.history_length = 5
        self.observations.obsHistory.flatten_history_dim = False
        self.events.add_base_mass.params["asset_cfg"].body_names = "BODY"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)
        self.terminations.base_contact.params["sensor_cfg"].body_names = "BODY"
        self.viewer.origin_type = "world"
        self.viewer.eye = (100.0, 100.0, 100.0)

        _apply_joint_fault_observations(self.observations, self._fault_joint_names())

    def _fault_joint_names(self) -> list[str]:
        return FL_KN_FAULT_JOINT_NAMES

    def _fault_robot_cfg(self):
        return PONGBOT_R2_FLKN_FAULT_CFG

    def _blind_rough_terrain_cfg(self):
        return BLIND_ROUGH_TERRAINS_CFG

    def _blind_rough_terrain_play_cfg(self):
        return BLIND_ROUGH_TERRAINS_PLAY_CFG


@configclass
class PFBaseImplicitJointFaultEnvCfg_PLAY(PFBaseImplicitJointFaultEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None


@configclass
class PFBlindRoughImplicitFLKNFaultEnvCfg(PFBaseImplicitJointFaultEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = _make_reward_height_scanner(self.decimation * self.sim.dt)
        self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = self._blind_rough_terrain_cfg()


@configclass
class PFBlindRoughImplicitFLKNFaultEnvCfg_PLAY(PFBaseImplicitJointFaultEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = _make_reward_height_scanner(self.decimation * self.sim.dt)
        self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = self._blind_rough_terrain_play_cfg()


@configclass
class PFBlindRoughImplicitFLLegFaultEnvCfg(PFBaseImplicitJointFaultEnvCfg):
    def _fault_joint_names(self) -> list[str]:
        return FL_LEG_FAULT_JOINT_NAMES

    def _fault_robot_cfg(self):
        return PONGBOT_R2_FL_LEG_FAULT_CFG

    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = _make_reward_height_scanner(self.decimation * self.sim.dt)
        self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = self._blind_rough_terrain_cfg()


@configclass
class PFBlindRoughImplicitFLLegFaultEnvCfg_PLAY(PFBaseImplicitJointFaultEnvCfg_PLAY):
    def _fault_joint_names(self) -> list[str]:
        return FL_LEG_FAULT_JOINT_NAMES

    def _fault_robot_cfg(self):
        return PONGBOT_R2_FL_LEG_FAULT_CFG

    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = _make_reward_height_scanner(self.decimation * self.sim.dt)
        self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = self._blind_rough_terrain_play_cfg()
