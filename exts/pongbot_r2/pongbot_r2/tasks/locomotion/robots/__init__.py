import gymnasium as gym

from pongbot_r2.tasks.locomotion.agents.rsl_rl_ppo_cfg import PongBot_R2FlatPPORunnerCfg, PongBot_R2RoughPPORunnerCfg, PongBot_R2StairPPORunnerCfg

from . import pongbot_r2_env_cfg

##
# Create PPO runners for RSL-RL
##

pongbot_r2_blind_flat_runner_cfg = PongBot_R2FlatPPORunnerCfg()
pongbot_r2_blind_rough_runner_cfg= PongBot_R2RoughPPORunnerCfg()
pongbot_r2_stair_runner_cfg= PongBot_R2StairPPORunnerCfg()

##
# Register Gym environments
##

############################
# PF Blind Flat Environment
############################
gym.register(
    id="PongBot-R2-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": pongbot_r2_blind_flat_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pongbot_r2_blind_flat_runner_cfg,
    },
)

############################
# PF Blind Rough Environment
############################
gym.register(
    id="PongBot-R2-Blind-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": pongbot_r2_blind_rough_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Blind-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pongbot_r2_blind_rough_runner_cfg,
    },
)

############################
# PF Blind Stair Environment
############################
gym.register(
    id="PongBot-R2-Blind-Stair-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindStairEnvCfg,
        "rsl_rl_cfg_entry_point": pongbot_r2_stair_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Blind-Stair-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindStairEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pongbot_r2_stair_runner_cfg,
    },
)

############################
# PF Stair Environment (Height Scan)
############################
gym.register(
    id="PongBot-R2-Stair-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFStairEnvCfgv1,
        "rsl_rl_cfg_entry_point": pongbot_r2_stair_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Stair-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFStairEnvCfgv1_PLAY,
        "rsl_rl_cfg_entry_point": pongbot_r2_stair_runner_cfg,
    },
)