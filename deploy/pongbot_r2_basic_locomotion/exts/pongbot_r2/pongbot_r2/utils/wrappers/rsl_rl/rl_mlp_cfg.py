# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extended RSL-RL configuration schemas used by PongBot training.

@version 0.0.1
@update 2026-07-12: Add optional lag-based observation-history subsampling.
"""

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg


@configclass
class RslRlPpoAlgorithmMlpCfg(RslRlPpoAlgorithmCfg):
    """Configuration of the runner for on-policy algorithms."""

    # runner_type: str = "OnPolicyRunner"

    obs_history_len: int = 1
    obs_history_offsets: tuple[int, ...] = ()
    encoder_warmup_iters: int = 0
    bootstrap_mode: Literal["warmup", "adaptive"] = "adaptive"
    bootstrap_min_iters: int = 0
    bootstrap_loss_threshold: float = 0.15
    bootstrap_loss_hysteresis: float = 0.01
    bootstrap_ema_alpha: float = 0.1
    est_learning_rate: float = 1.0e-4
    vae_beta: float = 1.0
    policy_include_history: bool = True
    terrain_prior_warmup_iters: int = 0
    terrain_property_target_dim: int = 0
    adaptive_learning_rate: float = 1.0e-3
    adaptive_value_loss_coef: float = 1.0
    adaptive_entropy_coef: float = 0.01
    adaptive_reward_scale: float = 0.1
    load_latent_dim: int = 16
    load_encoder_hidden_dims: list[int] = [128, 64]
    load_transition_hidden_dims: list[int] = [128, 64]
    load_activation: str = "elu"
    load_orthogonal_init: bool = False
    load_learning_rate: float = 1.0e-4
    load_triplet_margin: float = 0.5 #1.0
    load_triplet_loss_coef: float = 0.1 #1.0
    load_hard_negative_mass_diff_min: float = 5.0
    load_hard_negative_fallback_mass_diff_min: float = 3.0
    load_hard_negative_command_dist_max: float = 0.5
    load_hard_negative_num_candidates: int = 32
    load_probe_learning_rate: float = 1.0e-4
    adaptive_shared_reward_scales: dict[str, float] | None = None
    adaptive_extra_reward_scales: dict[str, float] | None = None
    reward_weight_curriculum_terms: tuple[str, ...] = ()
    reward_weight_curriculum_annealing_rate: float = 0.998
    reward_weight_curriculum_log_interval: int = 1
    action_bound_loss_coef: float = 0.0
    action_bound_threshold: float = 1.0
    action_bound_max_excess: float = 4.0


@configclass
class EncoderCfg:
    class_name: str = "MLP_Encoder"
    output_detach : bool = True
    num_input_dim : int = MISSING
    num_output_dim : int = 3
    hidden_dims : list[int] = [256, 128]
    activation : str = "elu"
    orthogonal_init : bool = False
    policy_output_scales: list[float] | None = None
    target_source: Literal["critic_prefix", "observation_group"] = "critic_prefix"
    target_observation_group: str | None = None


@configclass
class MassEncoderCfg(EncoderCfg):
    num_output_dim: int = 1
    policy_output_scales: list[float] | None = [0.02]
    target_source: Literal["critic_prefix", "observation_group"] = "observation_group"
    target_observation_group: str | None = "encoder_target"

@configclass
class IMUEncoderCfg:
    class_name: str = "IMU_Encoder"
    output_detach : bool = True
    num_input_dim : int = MISSING
    num_output_dim : int = 9
    hidden_dims : list[int] = [256, 128]
    activation : str = "elu"
    orthogonal_init : bool = False
    target_source: Literal["critic_prefix", "observation_group"] = "critic_prefix"
    target_observation_group: str | None = None


@configclass
class ContextEstimatorCfg:
    class_name: str = "ContextEstimatorNet"
    history_len: int = 5
    obs_dim: int | None = None
    latent_dim: int = 16
    encoder_hidden_dims: list[int] = [128, 64]
    decoder_hidden_dims: list[int] = [64, 128]
    beta: float = 1.0
    output_detach: bool = True
    target_source: Literal["critic_prefix", "observation_group"] = "critic_prefix"
    target_observation_group: str | None = None
    terrain_conditioned_prior: bool = False
    z_terrain_dim: int = 0
    residual_kl_coef: float = 0.0
    num_terrain_classes: int = 0
    terrain_prior_radius: float = 1.0
    terrain_prior_std: float = 2.0
    terrain_prior_coef: float = 1.0
    terrain_classification_coef: float = 0.0
    terrain_property_dim: int = 0
    terrain_property_coef: float = 0.0


@configclass
class AdaptivePolicyCfg:
    init_noise_std: float = 0.5
    actor_hidden_dims: list[int] = [256, 128]
    critic_hidden_dims: list[int] = [256, 128]
    activation: str = "elu"
    orthogonal_init: bool = False
    action_scale: float = 0.15
    logstd_min: float = -5.0
    logstd_max: float = 1.0


import os
import copy
import torch


class _EncoderEncodeWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = copy.deepcopy(encoder).to("cpu")
        self.encoder.eval()

    def forward(self, encoder_input):
        return self.encoder.encode(encoder_input)


def export_mlp_as_onnx(mlp, path, name, input_dim):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, name + ".onnx")
    model = copy.deepcopy(mlp).to("cpu")
    model.eval()

    dummy_input = torch.randn(input_dim)
    input_names = ["mlp_input"]
    output_names = ["mlp_output"]

    torch.onnx.export(
        model,
        dummy_input,
        path,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=13,
    )
    print("Exported MLP as onnx script to: ", path)


def export_encoder_as_onnx(encoder, path, name, input_shape):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, name + ".onnx")

    if isinstance(input_shape, int):
        dummy_input = torch.randn(input_shape)
    elif isinstance(input_shape, (tuple, list)):
        dummy_input = torch.randn((1, *tuple(input_shape)))
    else:
        raise TypeError(f"Unsupported encoder input shape: {input_shape!r}")

    model = _EncoderEncodeWrapper(encoder)

    torch.onnx.export(
        model,
        dummy_input,
        path,
        verbose=True,
        input_names=["encoder_input"],
        output_names=["encoder_output"],
        export_params=True,
        opset_version=13,
    )
    print("Exported encoder as onnx script to: ", path)

def export_policy_as_jit(actor_critic, path):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "policy.pt")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)
