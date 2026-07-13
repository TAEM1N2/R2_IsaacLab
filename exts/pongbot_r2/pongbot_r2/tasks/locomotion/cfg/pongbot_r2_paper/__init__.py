"""Paper-faithful barrier locomotion task for PongBot R2.

@version 0.0.2
@update 2026-07-13: Register the standalone training task and runner.
@update 2026-07-13: Add the isolated R2 paper-barrier rough-terrain task.
"""

import gymnasium as gym

from .paper_env_cfg import PaperBarrierRoughEnvCfg
from .paper_rsl_rl_cfg import PongBotR2PaperBarrierRunnerCfg


gym.register(
    id="PongBot-R2-PaperBarrier-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PaperBarrierRoughEnvCfg,
        "rsl_rl_cfg_entry_point": PongBotR2PaperBarrierRunnerCfg(),
    },
)

__all__ = ["PaperBarrierRoughEnvCfg", "PongBotR2PaperBarrierRunnerCfg"]
