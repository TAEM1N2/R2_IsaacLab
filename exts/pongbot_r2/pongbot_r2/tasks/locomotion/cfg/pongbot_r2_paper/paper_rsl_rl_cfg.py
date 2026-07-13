"""Training configuration for the R2 barrier-paper reproduction.

@version 0.0.3
@update 2026-07-13: Preserve physical exploration while bounding R2 actor KL and action noise.
@update 2026-07-13: Configure independent optimization and return-scale-normalized dual critics.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


@configclass
class PaperBarrierPolicyCfg:
    class_name: str = "PaperBarrierActorCritic"
    init_noise_std: float = 0.4
    actor_hidden_dims: list[int] = [256, 128, 64]
    critic_hidden_dims: list[int] = [256, 128, 64]
    estimator_hidden_dims: list[int] = [256, 128]
    activation: str = "elu"
    logstd_min: float = -5.0
    logstd_max: float = -0.5108256237659907  # log(0.6)


@configclass
class PaperBarrierAlgorithmCfg:
    class_name: str = "PaperBarrierPPO"
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.005
    estimator_loss_coef: float = 1.0
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    learning_rate_min: float = 1.0e-5
    learning_rate_max: float = 1.0e-3
    standard_critic_learning_rate: float = 1.0e-3
    barrier_critic_learning_rate: float = 1.0e-3
    estimator_learning_rate: float = 1.0e-3
    value_scale_min: float = 1.0
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    kl_upper_factor: float = 1.5
    kl_lower_factor: float = 0.5
    initial_standard_advantage_weight: float = 0.60
    final_standard_advantage_weight: float = 0.50
    advantage_transition_success: float = 0.30
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
