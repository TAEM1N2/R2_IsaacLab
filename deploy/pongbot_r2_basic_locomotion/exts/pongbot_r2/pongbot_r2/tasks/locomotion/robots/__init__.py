"""Gym registrations for the basic PongBot R2 locomotion tasks.

@version 0.0.1
@update 2026-07-29: Isolate Flat/Rough train and play registrations for source distribution.
"""

import gymnasium as gym

from pongbot_r2.tasks.locomotion.agents.rsl_rl_ppo_cfg import (
    PongBot_R2FlatPPORunnerCfg,
    PongBot_R2RoughPPORunnerCfg,
)

from . import pongbot_r2_env_cfg


_FLAT_RUNNER_CFG = PongBot_R2FlatPPORunnerCfg()
_ROUGH_RUNNER_CFG = PongBot_R2RoughPPORunnerCfg()


def _register(task_id: str, env_cfg, runner_cfg) -> None:
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "rsl_rl_cfg_entry_point": runner_cfg,
        },
    )


_register("PongBot-R2-Blind-Flat-v0", pongbot_r2_env_cfg.PFBlindFlatEnvCfg, _FLAT_RUNNER_CFG)
_register("PongBot-R2-Blind-Flat-Play-v0", pongbot_r2_env_cfg.PFBlindFlatEnvCfg_PLAY, _FLAT_RUNNER_CFG)
_register("PongBot-R2-Blind-Rough-v0", pongbot_r2_env_cfg.PFBlindRoughEnvCfg, _ROUGH_RUNNER_CFG)
_register("PongBot-R2-Blind-Rough-Play-v0", pongbot_r2_env_cfg.PFBlindRoughEnvCfg_PLAY, _ROUGH_RUNNER_CFG)
