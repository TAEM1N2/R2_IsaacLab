"""Gym registrations for PongBot R2 locomotion environments.

@version 0.0.2
@update 2026-07-12: Register the isolated third-experiment frontier train and play tasks.
@update 2026-07-10: Register the isolated long-rollout runner for implicit rough tasks.
"""

import gymnasium as gym

from pongbot_r2.tasks.locomotion.agents.rsl_rl_ppo_cfg import PongBot_R2FlatPPORunnerCfg, PongBot_R2RoughPPORunnerCfg, PongBot_R2StairPPORunnerCfg, \
PongBot_R2FlatIMUPPORunnerCfg, PongBot_R2RoughIMUPPORunnerCfg, PongBot_R2StairIMUPPORunnerCfg, PongBot_R2RoughMassPPORunnerCfg, PongBot_R2ImplicitFlatPPORunnerCfg, PongBot_R2ImplicitRoughPPORunnerCfg, PongBot_R2ImplicitRoughPyramidPPORunnerCfg, PongBot_R2ImplicitRoughLegacyPPORunnerCfg, PongBot_R2ImplicitRoughFrontierPPORunnerCfg, PongBot_R2ImplicitRoughTCPPPORunnerCfg, PongBot_R2ImplicitStairPPORunnerCfg, PongBot_R2VelOnlyImplicitFlatPPORunnerCfg, PongBot_R2VelOnlyImplicitRoughPPORunnerCfg, PongBot_R2Phase2AdaptiveRoughPPORunnerCfg, PongBot_R2Phase2LoadAdaptiveRoughPPORunnerCfg, PongBot_R2Phase2LoadAdaptiveComRoughPPORunnerCfg, PongBot_R2Phase2LoadAdaptiveAbruptRoughPPORunnerCfg

from . import pongbot_r2_env_cfg
from pongbot_r2.tasks.locomotion.cfg.pongbot_r2_legFault import leg_fault_env_cfg

##
# Create PPO runners for RSL-RL
##

pongbot_r2_blind_flat_runner_cfg = PongBot_R2FlatPPORunnerCfg()
pongbot_r2_blind_rough_runner_cfg= PongBot_R2RoughPPORunnerCfg()
pongbot_r2_blind_rough_mass_runner_cfg = PongBot_R2RoughMassPPORunnerCfg()
pongbot_r2_implicit_flat_runner_cfg = PongBot_R2ImplicitFlatPPORunnerCfg()
pongbot_r2_implicit_rough_runner_cfg = PongBot_R2ImplicitRoughPyramidPPORunnerCfg()
pongbot_r2_implicit_rough_legacy_runner_cfg = PongBot_R2ImplicitRoughLegacyPPORunnerCfg()
pongbot_r2_implicit_rough_frontier_runner_cfg = PongBot_R2ImplicitRoughFrontierPPORunnerCfg()
pongbot_r2_implicit_rough_tcp_runner_cfg = PongBot_R2ImplicitRoughTCPPPORunnerCfg()
pongbot_r2_implicit_stair_runner_cfg = PongBot_R2ImplicitStairPPORunnerCfg()
pongbot_r2_velonly_implicit_flat_runner_cfg = PongBot_R2VelOnlyImplicitFlatPPORunnerCfg()
pongbot_r2_velonly_implicit_rough_runner_cfg = PongBot_R2VelOnlyImplicitRoughPPORunnerCfg()
pongbot_r2_phase2_adaptive_rough_runner_cfg = PongBot_R2Phase2AdaptiveRoughPPORunnerCfg()
pongbot_r2_phase2_load_adaptive_rough_runner_cfg = PongBot_R2Phase2LoadAdaptiveRoughPPORunnerCfg()
pongbot_r2_phase2_load_adaptive_com_rough_runner_cfg = PongBot_R2Phase2LoadAdaptiveComRoughPPORunnerCfg()
pongbot_r2_phase2_load_adaptive_abrupt_rough_runner_cfg = PongBot_R2Phase2LoadAdaptiveAbruptRoughPPORunnerCfg()
pongbot_r2_stair_runner_cfg= PongBot_R2StairPPORunnerCfg()
pongbot_r2_blind_flat_imu_runner_cfg = PongBot_R2FlatIMUPPORunnerCfg()
pongbot_r2_blind_rough_imu_runner_cfg = PongBot_R2RoughIMUPPORunnerCfg()
pongbot_r2_blind_stair_imu_runner_cfg = PongBot_R2StairIMUPPORunnerCfg()


##
# Register Gym environments
##

################################################################## normal #################################################################

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

################################################################## w/o IMU #################################################################

############################
# imu encoder Flat (Plane)
############################
gym.register(
    id="PongBot-R2-Imu-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindFlatIMUEnvCfg,
        "rsl_rl_cfg_entry_point": pongbot_r2_blind_flat_imu_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Imu-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindFlatIMUEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pongbot_r2_blind_flat_imu_runner_cfg,
    },
)

############################
# imu encoder Rough
############################
gym.register(
    id="PongBot-R2-Imu-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindRoughIMUEnvCfg,
        "rsl_rl_cfg_entry_point": pongbot_r2_blind_rough_imu_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Imu-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindRoughIMUEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pongbot_r2_blind_rough_imu_runner_cfg,
    },
)

############################
# imu encoder Blind Stair
############################
gym.register(
    id="PongBot-R2-Imu-Stair-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindStairIMUEnvCfg,
        "rsl_rl_cfg_entry_point": pongbot_r2_blind_stair_imu_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Imu-Stair-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindStairIMUEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pongbot_r2_blind_stair_imu_runner_cfg,
    },
)




################################################################## mass estimator #################################################################
############################
# PF Blind Rough Environment
############################
gym.register(
    id="PongBot-R2-Mass-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindRoughMassEnvCfg,
        "rsl_rl_cfg_entry_point": pongbot_r2_blind_rough_mass_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Mass-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindRoughMassEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pongbot_r2_blind_rough_mass_runner_cfg,
    },
)



################################################################## implicit estimator #################################################################
############################
# PF Blind Flat Environment
############################
gym.register(
    id="PongBot-R2-Implicit-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindFlatImplicitEnvCfg,
        "rsl_rl_cfg_entry_point": pongbot_r2_implicit_flat_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Implicit-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindFlatImplicitEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pongbot_r2_implicit_flat_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-VelOnly-Implicit-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindFlatImplicitEnvCfg,
        "rsl_rl_cfg_entry_point": pongbot_r2_velonly_implicit_flat_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-VelOnly-Implicit-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindFlatImplicitEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pongbot_r2_velonly_implicit_flat_runner_cfg,
    },
)

############################
# PF Blind Rough Environment
############################
gym.register(
    id="PongBot-R2-Implicit-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindRoughImplicitEnvCfg,
        "rsl_rl_cfg_entry_point": pongbot_r2_implicit_rough_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Implicit-Rough-Legacy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindRoughImplicitLegacyEnvCfg,
        "rsl_rl_cfg_entry_point": pongbot_r2_implicit_rough_legacy_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Implicit-Rough-Frontier-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindRoughImplicitFrontierEnvCfg,
        "rsl_rl_cfg_entry_point": pongbot_r2_implicit_rough_frontier_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Implicit-Rough-Frontier-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindRoughImplicitFrontierEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pongbot_r2_implicit_rough_frontier_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-VelOnly-Implicit-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindRoughImplicitEnvCfg,
        "rsl_rl_cfg_entry_point": pongbot_r2_velonly_implicit_rough_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-VelOnly-Implicit-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindRoughImplicitEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pongbot_r2_velonly_implicit_rough_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Implicit-Rough-TCP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindRoughImplicitTCPEnvCfg,
        "rsl_rl_cfg_entry_point": pongbot_r2_implicit_rough_tcp_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Implicit-Rough-TCP-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindRoughImplicitTCPLatentEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pongbot_r2_implicit_rough_tcp_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Implicit-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindRoughImplicitEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pongbot_r2_implicit_rough_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Implicit-Rough-PCA-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindRoughImplicitPcaEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pongbot_r2_implicit_rough_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Implicit-Rough-LatentEval-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindRoughImplicitLatentEvalEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pongbot_r2_implicit_rough_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Implicit-Stair-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindImplicitStairEnvCfg,
        "rsl_rl_cfg_entry_point": pongbot_r2_implicit_stair_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Implicit-Stair-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindImplicitStairEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pongbot_r2_implicit_stair_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-IMPLICIT_STAIR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindImplicitStairEnvCfg,
        "rsl_rl_cfg_entry_point": pongbot_r2_implicit_stair_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-IMPLICIT_STAIR-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindImplicitStairEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pongbot_r2_implicit_stair_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Implicit-Rough-FLKN-Fault-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": leg_fault_env_cfg.PFBlindRoughImplicitFLKNFaultEnvCfg,
        "rsl_rl_cfg_entry_point": pongbot_r2_implicit_rough_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Implicit-Rough-FLKN-Fault-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": leg_fault_env_cfg.PFBlindRoughImplicitFLKNFaultEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pongbot_r2_implicit_rough_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Implicit-Rough-FL-Leg-Fault-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": leg_fault_env_cfg.PFBlindRoughImplicitFLLegFaultEnvCfg,
        "rsl_rl_cfg_entry_point": pongbot_r2_implicit_rough_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Implicit-Rough-FL-Leg-Fault-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": leg_fault_env_cfg.PFBlindRoughImplicitFLLegFaultEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pongbot_r2_implicit_rough_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Phase2-Adaptive-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindRoughPhase2AdaptiveEnvCfg,
        "rsl_rl_cfg_entry_point": pongbot_r2_phase2_adaptive_rough_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Phase2-Adaptive-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindRoughPhase2AdaptiveEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pongbot_r2_phase2_adaptive_rough_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Phase2-LoadAdaptive-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindRoughPhase2LoadAdaptiveEnvCfg,
        "rsl_rl_cfg_entry_point": pongbot_r2_phase2_load_adaptive_rough_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Phase2-LoadAdaptive-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindRoughPhase2LoadAdaptiveEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pongbot_r2_phase2_load_adaptive_rough_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Phase2-LoadAdaptive-Com-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindRoughPhase2LoadAdaptiveComEnvCfg,
        "rsl_rl_cfg_entry_point": pongbot_r2_phase2_load_adaptive_com_rough_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Phase2-LoadAdaptive-Com-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindRoughPhase2LoadAdaptiveComEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pongbot_r2_phase2_load_adaptive_com_rough_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Phase2-LoadAdaptive-Abrupt-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindRoughPhase2LoadAdaptiveAbruptEnvCfg,
        "rsl_rl_cfg_entry_point": pongbot_r2_phase2_load_adaptive_abrupt_rough_runner_cfg,
    },
)

gym.register(
    id="PongBot-R2-Phase2-LoadAdaptive-Abrupt-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pongbot_r2_env_cfg.PFBlindRoughPhase2LoadAdaptiveAbruptEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pongbot_r2_phase2_load_adaptive_abrupt_rough_runner_cfg,
    },
)
