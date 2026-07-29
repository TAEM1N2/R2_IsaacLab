"""RSL-RL runner configurations for PongBot R2 locomotion tasks.

@version 0.0.6
@update 2026-07-14: Reduce Legacy initial exploration noise and entropy regularization.
@update 2026-07-12: Restore the frontier runner to the original consecutive five-frame history.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from pongbot_r2.utils.wrappers.rsl_rl.rl_mlp_cfg import (
    EncoderCfg,
    ContextEstimatorCfg,
    IMUEncoderCfg,
    MassEncoderCfg,
    AdaptivePolicyCfg,
    RslRlPpoAlgorithmMlpCfg,
)

# Isaac Lab original RSL-RL configuration
@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations =100000
    save_interval = 200
    experiment_name = "pongbot_r2_direct"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
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
    )

@configclass
class PongBot_R2FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations =100000
    save_interval = 200
    experiment_name = "pongbot_r2_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
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
    encoder = EncoderCfg(
        output_detach=True,
        num_output_dim=3,
        hidden_dims=[256, 128],
        activation="elu",
        orthogonal_init=False,
    )

@configclass
class PongBot_R2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations =100000
    save_interval = 200
    experiment_name = "pongbot_r2_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
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
    encoder = EncoderCfg(
        output_detach=True,
        num_output_dim=3,
        hidden_dims=[256, 128],
        activation="elu",
        orthogonal_init=False,
    )

@configclass
class PongBot_R2StairPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations =100000
    save_interval = 200
    experiment_name = "pongbot_r2_stair"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
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
    encoder = EncoderCfg(
        output_detach=True,
        num_output_dim=3,
        hidden_dims=[256, 128],
        activation="elu",
        orthogonal_init=False,
    )




############################################################ w/o IMU ###########################################################
@configclass
class PongBot_R2FlatIMUPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations =100000
    save_interval = 200
    experiment_name = "pongbot_r2_flat_imu"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="IMU_PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,
        bootstrap_mode="adaptive",
        bootstrap_min_iters=1000,
        bootstrap_loss_threshold=0.18,
        bootstrap_loss_hysteresis=0.01,
        bootstrap_ema_alpha=0.1,
        est_learning_rate = 3.4e-4,
    )
    encoder = IMUEncoderCfg(
        output_detach=True,
        num_output_dim=9,
        hidden_dims=[256, 128],
        activation="elu",
        orthogonal_init=False,
    )

@configclass
class PongBot_R2RoughIMUPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations =100000
    save_interval = 200
    experiment_name = "pongbot_r2_rough_imu"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="IMU_PPO",
        
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,
        bootstrap_mode="adaptive",
        bootstrap_min_iters=1000,
        bootstrap_loss_threshold=0.18,
        bootstrap_loss_hysteresis=0.01,
        bootstrap_ema_alpha=0.1,
        est_learning_rate = 1.0e-4,
    )
    encoder = IMUEncoderCfg(
        output_detach=True,
        num_output_dim=9,
        hidden_dims=[256, 128],
        activation="elu",
        orthogonal_init=False,
    )


@configclass
class PongBot_R2StairIMUPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations =100000
    save_interval = 200
    experiment_name = "pongbot_r2_stair_imu"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="IMU_PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,
        bootstrap_mode="adaptive",
        bootstrap_min_iters=1000,
        bootstrap_loss_threshold=0.18,
        bootstrap_loss_hysteresis=0.01,
        bootstrap_ema_alpha=0.1,
        est_learning_rate = 1.0e-4,
    )
    encoder = IMUEncoderCfg(
        output_detach=True,
        num_output_dim=9,
        hidden_dims=[256, 128],
        activation="elu",
        orthogonal_init=False,
    )





###################################################### Mass #############################################################################
@configclass
class PongBot_R2RoughMassPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations =100000
    save_interval = 200
    experiment_name = "pongbot_r2_mass"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,
        est_learning_rate = 1.0e-4,
    )
    encoder = MassEncoderCfg(
        output_detach=True,
        hidden_dims=[256, 128],
        activation="elu",
        orthogonal_init=False,
    )


###################################################### Implicit CE Net ####################################################################
@configclass
class PongBot_R2ImplicitRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 200
    experiment_name = "pongbot_r2_implicit"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="Implicit_PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        action_bound_loss_coef=0.02,
        action_bound_threshold=1.0,
        action_bound_max_excess=4.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=5,
        est_learning_rate=1.0e-4,
        vae_beta=1.0,
        policy_include_history=False,
        reward_weight_curriculum_terms=(
            # "pen_joint_vel",
            # "pen_joit_accel",
            # "pen_action_rate",
            # "pen_action_smoothness",
            # "pen_joint_powers_var",
        ),
        reward_weight_curriculum_annealing_rate=0.998,
        reward_weight_curriculum_log_interval=1,
    )
    encoder = ContextEstimatorCfg(
        history_len=5,
        latent_dim=16,
        encoder_hidden_dims=[128, 64],
        decoder_hidden_dims=[64, 128],
        beta=1.0,
        output_detach=True,
    )


@configclass
class PongBot_R2ImplicitRoughPyramidPPORunnerCfg(PongBot_R2ImplicitRoughPPORunnerCfg):
    """Implicit rough runner reserved for pyramid-terrain experiments."""

    num_steps_per_env = 24


@configclass
class PongBot_R2ImplicitRoughLegacyPPORunnerCfg(PongBot_R2ImplicitRoughPPORunnerCfg):
    """Runner matching the 2026-07-07 implicit rough experiment contract."""

    experiment_name = "pongbot_r2_implicit_legacy"

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.3
        self.algorithm.entropy_coef = 0.001
        # This experiment intentionally removes the policy-mean action-bound
        # regularizer.  Environment-side action clipping is handled separately
        # by train.py and is widened to [-10, 10] only for this task.
        self.algorithm.action_bound_loss_coef = 0.0
        self.algorithm.action_bound_threshold = 10.0
        self.algorithm.action_bound_max_excess = None


@configclass
class PongBot_R2ImplicitRoughFrontierPPORunnerCfg(PongBot_R2ImplicitRoughPyramidPPORunnerCfg):
    """Third-experiment runner using the original consecutive five-frame history."""

    experiment_name = "pongbot_r2_implicit_frontier"

    def __post_init__(self):
        self.algorithm.obs_history_offsets = ()


@configclass
class PongBot_R2ImplicitFlatPPORunnerCfg(PongBot_R2ImplicitRoughPPORunnerCfg):
    experiment_name = "pongbot_r2_implicit_flat"


@configclass
class PongBot_R2VelOnlyImplicitRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 200
    experiment_name = "pongbot_r2_velonly_implicit"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        action_bound_loss_coef=0.05,
        action_bound_threshold=0.9,
        action_bound_max_excess=4.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=5,
        est_learning_rate=1.0e-4,
        policy_include_history=False,
        reward_weight_curriculum_terms=(),
        reward_weight_curriculum_annealing_rate=0.998,
        reward_weight_curriculum_log_interval=1,
    )
    encoder = EncoderCfg(
        output_detach=True,
        num_output_dim=3,
        hidden_dims=[128, 64],
        activation="elu",
        orthogonal_init=False,
    )


@configclass
class PongBot_R2VelOnlyImplicitFlatPPORunnerCfg(PongBot_R2VelOnlyImplicitRoughPPORunnerCfg):
    experiment_name = "pongbot_r2_velonly_implicit_flat"


@configclass
class PongBot_R2ImplicitRoughTCPPPORunnerCfg(PongBot_R2ImplicitRoughPPORunnerCfg):
    experiment_name = "pongbot_r2_implicit_rough_tcp_split"
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="Implicit_PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,
        est_learning_rate=1.0e-4,
        vae_beta=1.0,
        policy_include_history=False,
        terrain_prior_warmup_iters=1000,
        terrain_property_target_dim=4,
        reward_weight_curriculum_terms=(),
        reward_weight_curriculum_annealing_rate=0.998,
        reward_weight_curriculum_log_interval=1,
    )
    encoder = ContextEstimatorCfg(
        history_len=10,
        latent_dim=16,
        encoder_hidden_dims=[128, 64],
        decoder_hidden_dims=[64, 128],
        beta=1.0,
        output_detach=True,
        terrain_conditioned_prior=True,
        z_terrain_dim=6,
        residual_kl_coef=0.0,
        num_terrain_classes=6,
        terrain_prior_radius=3.0,
        terrain_prior_std=1.0,
        terrain_prior_coef=1.0,
        terrain_classification_coef=0.3,
        terrain_property_dim=4,
        terrain_property_coef=0.1,
        target_source="observation_group",
        target_observation_group="encoder_target",
    )


@configclass
class PongBot_R2ImplicitStairPPORunnerCfg(PongBot_R2ImplicitRoughPPORunnerCfg):
    experiment_name = "pongbot_r2_implicit_stair"


@configclass
class PongBot_R2Phase2AdaptiveRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 200
    experiment_name = "pongbot_r2_phase2_adaptive"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    adaptive_policy = AdaptivePolicyCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
        orthogonal_init=False,
        action_scale=0.10,
        logstd_min=-5.0,
        logstd_max=1.0,
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="Phase2_Adaptive_PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        adaptive_learning_rate=1.0e-4,
        adaptive_value_loss_coef=1.0,
        adaptive_entropy_coef=0.0,
        adaptive_reward_scale=0.1,
        adaptive_shared_reward_scales={
            "rew_lin_vel_xy": 0.0,
            "rew_ang_vel_z": 0.0,
            "test_gait_reward": 0.25,
            "pen_base_height": -1.0,
            "pen_flat_orientation": -0.1,
            "pen_lin_vel_z": -0.5,
            "pen_ang_vel_xy": -5.0e-3,
            "pen_undesired_contacts": -0.25,
            "pen_joint_pos_limits": -2.5,
            "pen_joint_torque": -2.5e-6,
            "pen_joint_accel": -2.5e-7,
            "pen_action_rate": -0.01,
            "pen_swing_height_error": -1.0,
            "pen_action_smoothness": -0.01,
            "pen_joint_powers_var": -2.5e-6,
            "pen_joint_default_pos": -0.1,
            "pen_standing_joint_default_pos": -0.5,
            "pen_hip_roll_pos": -0.1,
            "pen_standing_vel": -0.25,
            "pen_feet_slide": -0.1,
            "foot_landing_vel": -0.1,
        },
        adaptive_extra_reward_scales={
            "body_height": 1.0,
            "grf": 1.0,
            "stability": 1.0,
        },
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=5,
        est_learning_rate=1.0e-4,
        vae_beta=1.0,
        policy_include_history=False,
    )
    encoder = ContextEstimatorCfg(
        history_len=5,
        latent_dim=16,
        encoder_hidden_dims=[128, 64],
        decoder_hidden_dims=[64, 128],
        beta=1.0,
        output_detach=True,
    )


@configclass
class PongBot_R2Phase2LoadAdaptiveRoughPPORunnerCfg(PongBot_R2Phase2AdaptiveRoughPPORunnerCfg):
    experiment_name = "pongbot_r2_phase2_load_adaptive"
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="Phase2_LoadAdaptive_PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        adaptive_learning_rate=1.0e-4,
        adaptive_value_loss_coef=1.0,
        adaptive_entropy_coef=0.0,
        adaptive_reward_scale=0.1,
        load_latent_dim=16,
        load_encoder_hidden_dims=[128, 64],
        load_transition_hidden_dims=[128, 64],
        load_activation="elu",
        load_orthogonal_init=False,
        load_learning_rate=1.0e-4,
        load_triplet_margin=0.5, #1.0,
        load_triplet_loss_coef=0.1, #1.0,
        load_hard_negative_mass_diff_min=5.0,
        load_hard_negative_fallback_mass_diff_min=3.0,
        load_hard_negative_command_dist_max=0.2,
        load_hard_negative_num_candidates=32,
        load_probe_learning_rate=1.0e-4,
        adaptive_shared_reward_scales={
            "rew_lin_vel_xy": 0.0,
            "rew_ang_vel_z": 0.0,
            "test_gait_reward": 0.25,
            "pen_base_height": -1.0,
            "pen_flat_orientation": -0.1,
            "pen_lin_vel_z": -0.5,
            "pen_ang_vel_xy": -5.0e-3,
            "pen_undesired_contacts": -0.25,
            "pen_joint_pos_limits": -2.5,
            "pen_joint_torque": -2.5e-6,
            "pen_joint_accel": -2.5e-7,
            "pen_action_rate": -0.01,
            "pen_swing_height_error": -1.0,
            "pen_action_smoothness": -0.01,
            "pen_joint_powers_var": -2.5e-6,
            "pen_joint_default_pos": -0.1,
            "pen_standing_joint_default_pos": -0.5,
            "pen_hip_roll_pos": -0.1,
            "pen_standing_vel": -0.25,
            "pen_feet_slide": -0.1,
            "foot_landing_vel": -0.1,
        },
        adaptive_extra_reward_scales={
            "body_height": 1.0,
            "grf": 1.0,
            "stability": 1.0,
        },
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=5,
        est_learning_rate=1.0e-4,
        vae_beta=1.0,
        policy_include_history=False,
    )


@configclass
class PongBot_R2Phase2LoadAdaptiveComRoughPPORunnerCfg(PongBot_R2Phase2LoadAdaptiveRoughPPORunnerCfg):
    experiment_name = "pongbot_r2_phase2_load_adaptive_com"


@configclass
class PongBot_R2Phase2LoadAdaptiveAbruptRoughPPORunnerCfg(PongBot_R2Phase2LoadAdaptiveRoughPPORunnerCfg):
    experiment_name = "pongbot_r2_phase2_load_adaptive_abrupt"
