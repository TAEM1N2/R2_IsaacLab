"""Training configuration for the R2 barrier-paper reproduction.

@version 0.0.1
@update 2026-07-13: Add the paper-scale 160k-sample dual-PPO configuration.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


@configclass
class PaperBarrierPolicyCfg:
    class_name: str = "PaperBarrierActorCritic"
    init_noise_std: float = 1.0
    actor_hidden_dims: list[int] = [256, 128, 64]
    critic_hidden_dims: list[int] = [256, 128, 64]
    estimator_hidden_dims: list[int] = [256, 128]
    activation: str = "elu"
    logstd_min: float = -5.0
    logstd_max: float = 0.0


@configclass
class PaperBarrierAlgorithmCfg:
    class_name: str = "PaperBarrierPPO"
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    estimator_loss_coef: float = 1.0
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0


@configclass
class PongBotR2PaperBarrierRunnerCfg(RslRlOnPolicyRunnerCfg):
    runner_type: str = "PaperBarrierRunner"
    num_steps_per_env: int = 400
    max_iterations: int = 10_000
    save_interval: int = 100
    experiment_name: str = "pongbot_r2_paper_barrier_rough"
    empirical_normalization: bool = False
    calibration_steps: int = 100
    policy: PaperBarrierPolicyCfg = PaperBarrierPolicyCfg()
    algorithm: PaperBarrierAlgorithmCfg = PaperBarrierAlgorithmCfg()


__all__ = ["PongBotR2PaperBarrierRunnerCfg"]
