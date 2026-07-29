"""RSL-RL runner configurations for the basic Flat/Rough tasks.

@version 0.0.1
@update 2026-07-29: Isolate the nominal 3D MLP-encoder PPO configurations.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg

from pongbot_r2.utils.wrappers.rsl_rl.rl_mlp_cfg import EncoderCfg, RslRlPpoAlgorithmMlpCfg


def _policy_cfg() -> RslRlPpoActorCriticCfg:
    return RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )


def _algorithm_cfg() -> RslRlPpoAlgorithmMlpCfg:
    return RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,
    )


def _encoder_cfg() -> EncoderCfg:
    return EncoderCfg(
        output_detach=True,
        num_output_dim=3,
        hidden_dims=[256, 128],
        activation="elu",
        orthogonal_init=False,
    )


@configclass
class PongBot_R2FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 200
    experiment_name = "pongbot_r2_flat"
    empirical_normalization = False
    policy = _policy_cfg()
    algorithm = _algorithm_cfg()
    encoder = _encoder_cfg()


@configclass
class PongBot_R2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 200
    experiment_name = "pongbot_r2_rough"
    empirical_normalization = False
    policy = _policy_cfg()
    algorithm = _algorithm_cfg()
    encoder = _encoder_cfg()
