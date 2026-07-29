# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path

from rsl_rl.modules import (
    ActorCritic,
    AdaptiveActorCritic,
    MLP_Encoder,
    IMU_Encoder,
    ContextEstimatorNet,
    LoadEncoder,
    LoadTransitionModel,
    build_context_policy_input,
)
from rsl_rl.storage import RolloutStorage


def _compute_explained_variance_from_sums(
    num_samples: int,
    returns_sum: float,
    returns_sq_sum: float,
    residual_sum: float,
    residual_sq_sum: float,
) -> float:
    """Compute explained variance from accumulated first and second moments."""
    if num_samples <= 1:
        return 0.0

    returns_mean = returns_sum / num_samples
    residual_mean = residual_sum / num_samples
    returns_var = max(returns_sq_sum / num_samples - returns_mean * returns_mean, 0.0)
    residual_var = max(residual_sq_sum / num_samples - residual_mean * residual_mean, 0.0)
    if returns_var <= 1.0e-12:
        return 0.0
    return 1.0 - residual_var / returns_var


class PPO:
    actor_critic: ActorCritic
    encoder: MLP_Encoder

    def __init__(
        self,
        num_group,
        encoder,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        vae_beta=1.0,
        est_learning_rate=1.0e-3,
        policy_include_history=True,
        critic_take_latent=False,
        early_stop=False,
        anneal_lr=False,
        device="cpu",
        **kwargs,
    ):
        self.device = device
        self.num_group = num_group

        self.desired_kl = desired_kl
        self.early_stop = early_stop
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.anneal_lr = anneal_lr
        self.vae_beta = vae_beta
        self.policy_include_history = policy_include_history
        self.critic_take_latent = critic_take_latent

        self.encoder = encoder

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam([{"params": self.actor_critic.parameters()}], lr=learning_rate)

        if self.encoder.num_output_dim != 0:
            self.extra_optimizer = optim.Adam(
                self.encoder.parameters(), lr=est_learning_rate
            )
        else:
            self.extra_optimizer = None
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.action_bound_loss_coef = float(kwargs.pop("action_bound_loss_coef", 0.0))
        self.action_bound_threshold = float(kwargs.pop("action_bound_threshold", 1.0))
        self.action_bound_max_excess = kwargs.pop("action_bound_max_excess", 4.0)
        self.action_bound_max_excess = (
            None if self.action_bound_max_excess is None else float(self.action_bound_max_excess)
        )
        self.last_action_bound_loss = 0.0
        self.last_action_bound_loss_weighted = 0.0

    def _build_actor_obs(self, encoder_out, obs, commands, obs_history):
        assert encoder_out.ndim == 2, f"encoder_out must be [batch, dim], got {tuple(encoder_out.shape)}"
        assert obs.ndim == 2, f"obs must be [batch, dim], got {tuple(obs.shape)}"
        assert commands.ndim == 2, f"commands must be [batch, dim], got {tuple(commands.shape)}"
        assert obs_history.ndim == 2, f"obs_history must be flattened [batch, dim], got {tuple(obs_history.shape)}"
        assert encoder_out.shape[0] == obs.shape[0] == commands.shape[0] == obs_history.shape[0], (
            "batch size mismatch for actor obs concat: "
            f"encoder_out={tuple(encoder_out.shape)}, obs={tuple(obs.shape)}, "
            f"commands={tuple(commands.shape)}, obs_history={tuple(obs_history.shape)}"
        )
        if self.policy_include_history:
            return torch.cat((encoder_out, obs, commands, obs_history), dim=-1)
        return torch.cat((encoder_out, obs, commands), dim=-1)

    def _select_encoder_target(self, critic_obs_batch, encoder_target_batch):
        target_dim = self.encoder.num_output_dim
        if encoder_target_batch is not None:
            return encoder_target_batch[:, :target_dim]
        return critic_obs_batch[:, :target_dim]

    def _action_bound_loss(self, action_mean: torch.Tensor) -> torch.Tensor:
        if self.action_bound_loss_coef <= 0.0:
            return torch.zeros((), device=action_mean.device, dtype=action_mean.dtype)
        excess = torch.relu(torch.abs(action_mean) - self.action_bound_threshold)
        if self.action_bound_max_excess is not None:
            excess = torch.clamp(excess, max=self.action_bound_max_excess)
        return torch.mean(torch.square(excess))

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        encoder_target_shape,
        obs_history_shape,
        commands_shape,
        action_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            encoder_target_shape,
            obs_history_shape,
            commands_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, obs_history, commands, critic_obs):
        critic_obs = torch.cat((critic_obs, commands), dim=-1)
        # act
        encoder_out = self.encoder.encode(obs_history)

        actor_obs = self._build_actor_obs(encoder_out, obs, commands, obs_history)
        self.transition.actions = self.actor_critic.act(actor_obs).detach()


        # evaluate
        if self.critic_take_latent:
            critic_obs = torch.cat((critic_obs, encoder_out), dim=-1)
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()

        # storage
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_obs = critic_obs
        self.transition.encoder_targets = None
        self.transition.observation_history = obs_history
        self.transition.commands = commands
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_obs=None, encoder_targets=None):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # Record the transition
        self.transition.next_observations = next_obs
        self.transition.encoder_targets = encoder_targets
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self, clear_storage=True):
        num_updates = 0
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_action_bound_loss = 0
        mean_kl = 0
        num_value_samples = 0
        returns_sum = 0.0
        returns_sq_sum = 0.0
        residual_sum = 0.0
        residual_sq_sum = 0.0
        generator = self.storage.mini_batch_generator(
            self.num_group,
            self.num_mini_batches,
            self.num_learning_epochs,
        )
        for (
            obs_batch,
            critic_obs_batch,
            obs_history_batch, _,
            group_commands_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            encoder_out_batch = self.encoder.encode(obs_history_batch)
            commands_batch = group_commands_batch
            self.actor_critic.act(self._build_actor_obs(encoder_out_batch, obs_batch, commands_batch, obs_history_batch))

            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )

            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            kl_mean = torch.tensor(0, device=self.device, requires_grad=False)
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (
                        torch.square(old_sigma_batch)
                        + torch.square(old_mu_batch - mu_batch)
                    )
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)

            # KL
            if self.desired_kl != None and self.schedule == "adaptive":
                with torch.inference_mode():
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            if self.desired_kl != None and self.early_stop:
                if kl_mean > self.desired_kl * 1.5:
                    print("early stop, num_updates =", num_updates)
                    break

            # Surrogate loss
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            # print(ratio)
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            entropy_batch_mean = entropy_batch.mean()
            action_bound_loss = self._action_bound_loss(mu_batch)
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch_mean
                + self.action_bound_loss_coef * action_bound_loss
            )

            if self.anneal_lr:
                frac = 1.0 - num_updates / (
                    self.num_learning_epochs * self.num_mini_batches
                )
                self.optimizer.param_groups[0]["lr"] = frac * self.learning_rate

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            returns_detached = returns_batch.detach().double().reshape(-1)
            residual_detached = (returns_batch - value_batch).detach().double().reshape(-1)
            num_value_samples += returns_detached.numel()
            returns_sum += returns_detached.sum().item()
            returns_sq_sum += torch.square(returns_detached).sum().item()
            residual_sum += residual_detached.sum().item()
            residual_sq_sum += torch.square(residual_detached).sum().item()

            num_updates += 1
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_action_bound_loss += action_bound_loss.item()
            mean_kl += kl_mean.item()

        num_updates_extra = 0
        mean_extra_loss = 0
        if self.extra_optimizer is not None:
            generator = self.storage.encoder_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
            for (
                next_obs_batch,
                critic_obs_batch,
                encoder_target_batch,
                obs_history_batch,
            ) in generator:
                if hasattr(self.encoder, "get_encoder_out"):
                    self.encoder.encode(obs_history_batch)
                    encode_batch = self.encoder.get_encoder_out()

                if hasattr(self.encoder, "get_encoder_out"):
                    target_batch = self._select_encoder_target(critic_obs_batch, encoder_target_batch)
                    extra_loss = (
                        (encode_batch[:, : target_batch.shape[1]] - target_batch).pow(2).mean()
                    )
                else:
                    extra_loss = torch.zeros_like(value_loss)

                self.extra_optimizer.zero_grad()
                extra_loss.backward()
                self.extra_optimizer.step()

                num_updates_extra += 1
                mean_extra_loss += extra_loss.item()

        mean_value_loss /= num_updates
        if num_updates_extra > 0:
            mean_extra_loss /= num_updates_extra
        mean_surrogate_loss /= num_updates
        mean_action_bound_loss /= num_updates
        self.last_action_bound_loss = mean_action_bound_loss
        self.last_action_bound_loss_weighted = self.action_bound_loss_coef * mean_action_bound_loss
        mean_kl /= num_updates
        mean_explained_variance = _compute_explained_variance_from_sums(
            num_value_samples, returns_sum, returns_sq_sum, residual_sum, residual_sq_sum
        )
        if clear_storage:
            self.storage.clear()

        return (mean_value_loss, mean_extra_loss, mean_surrogate_loss, mean_kl, mean_explained_variance)

class IMU_PPO:
    actor_critic: ActorCritic
    encoder: IMU_Encoder

    def __init__(
        self,
        num_group,
        encoder,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        vae_beta=1.0,
        est_learning_rate=1.0e-3,
        policy_include_history=True,
        critic_take_latent=False,
        early_stop=False,
        anneal_lr=False,
        device="cpu",
        **kwargs,
    ):
        self.device = device
        self.num_group = num_group

        self.desired_kl = desired_kl
        self.early_stop = early_stop
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.anneal_lr = anneal_lr
        self.vae_beta = vae_beta
        self.policy_include_history = policy_include_history
        self.critic_take_latent = critic_take_latent

        self.encoder = encoder

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam([{"params": self.actor_critic.parameters()}], lr=learning_rate)

        if self.encoder.num_output_dim != 0:
            self.extra_optimizer = optim.Adam(
                self.encoder.parameters(), lr=est_learning_rate
            )
        else:
            self.extra_optimizer = None
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.action_bound_loss_coef = float(kwargs.pop("action_bound_loss_coef", 0.0))
        self.action_bound_threshold = float(kwargs.pop("action_bound_threshold", 1.0))
        self.action_bound_max_excess = kwargs.pop("action_bound_max_excess", 4.0)
        self.action_bound_max_excess = (
            None if self.action_bound_max_excess is None else float(self.action_bound_max_excess)
        )
        self.last_action_bound_loss = 0.0
        self.last_action_bound_loss_weighted = 0.0
        self.encoder_warmup_iters = kwargs.pop("encoder_warmup_iters", 0)
        self.bootstrap_mode = kwargs.pop("bootstrap_mode", "warmup")
        self.bootstrap_min_iters = kwargs.pop("bootstrap_min_iters", 0)
        self.bootstrap_loss_threshold = kwargs.pop("bootstrap_loss_threshold", 0.05)
        self.bootstrap_loss_hysteresis = kwargs.pop("bootstrap_loss_hysteresis", 0.01)
        self.bootstrap_ema_alpha = kwargs.pop("bootstrap_ema_alpha", 0.1)
        self.bootstrap_metric_epsilon = kwargs.pop("bootstrap_metric_epsilon", 1.0e-6)
        self.learning_iteration = 0
        self.bootstrap_metric = None
        self.bootstrap_encoder_active = False
        self.use_raw_imu_bootstrap = True
        self.nan_debug_dir = Path.cwd() / "nan_debug"
        self.nan_debug_context = {}

    def _select_encoder_target(self, critic_obs_batch, encoder_target_batch):
        target_dim = self.encoder.num_output_dim
        if encoder_target_batch is not None:
            return encoder_target_batch[:, :target_dim]
        return critic_obs_batch[:, :target_dim]

    def _action_bound_loss(self, action_mean: torch.Tensor) -> torch.Tensor:
        if self.action_bound_loss_coef <= 0.0:
            return torch.zeros((), device=action_mean.device, dtype=action_mean.dtype)
        excess = torch.relu(torch.abs(action_mean) - self.action_bound_threshold)
        if self.action_bound_max_excess is not None:
            excess = torch.clamp(excess, max=self.action_bound_max_excess)
        return torch.mean(torch.square(excess))

    def set_learning_iteration(self, iteration: int):
        self.learning_iteration = iteration

    def should_use_raw_imu(self) -> bool:
        if self.bootstrap_mode == "adaptive":
            return self.use_raw_imu_bootstrap
        return self.learning_iteration < self.encoder_warmup_iters

    def get_bootstrap_state(self) -> dict:
        return {
            "mode": self.bootstrap_mode,
            "use_raw_imu": self.should_use_raw_imu(),
            "bootstrap_metric": self.bootstrap_metric,
            "bootstrap_encoder_active": self.bootstrap_encoder_active,
            "bootstrap_min_iters": self.bootstrap_min_iters,
            "bootstrap_loss_threshold": self.bootstrap_loss_threshold,
        }

    def update_bootstrap_state(self, bootstrap_metric: float | None) -> None:
        if self.bootstrap_mode != "adaptive":
            return
        if bootstrap_metric is None:
            return

        metric_value = float(bootstrap_metric)
        self.bootstrap_metric = metric_value
        if self.bootstrap_encoder_active:
            self.use_raw_imu_bootstrap = False
            return

        if self.learning_iteration >= self.bootstrap_min_iters and metric_value <= self.bootstrap_loss_threshold:
            self.bootstrap_encoder_active = True
            self.use_raw_imu_bootstrap = False
            return

        self.use_raw_imu_bootstrap = True

    def set_nan_debug_dir(self, path: str):
        self.nan_debug_dir = Path(path)
        self.nan_debug_dir.mkdir(parents=True, exist_ok=True)

    def set_nan_debug_context(self, **context):
        self.nan_debug_context = context

    @staticmethod
    def _tensor_stats(tensor: torch.Tensor) -> dict:
        finite = torch.isfinite(tensor)
        stats = {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "num_bad": int((~finite).sum().item()),
        }
        if finite.any():
            finite_values = tensor[finite]
            stats.update(
                {
                    "min": float(finite_values.min().item()),
                    "max": float(finite_values.max().item()),
                    "mean": float(finite_values.mean().item()),
                    "std": float(finite_values.std().item()) if finite_values.numel() > 1 else 0.0,
                }
            )
        return stats

    def _dump_invalid_tensor(self, name: str, tensor: torch.Tensor, extra: dict | None = None) -> None:
        context = {"learning_iteration": self.learning_iteration, **self.nan_debug_context}
        if extra:
            context.update(extra)
        self.nan_debug_dir.mkdir(parents=True, exist_ok=True)
        dump_path = self.nan_debug_dir / (
            f"algo_nan_it{context.get('iteration', context.get('learning_iteration', -1))}"
            f"_step{context.get('rollout_step', -1)}_mb{context.get('mini_batch', -1)}_{name}.pt"
        )
        payload = {
            "context": context,
            "name": name,
            "stats": self._tensor_stats(tensor),
            "tensor": tensor.detach().cpu(),
        }
        torch.save(payload, dump_path)
        print(f"[NAN_DEBUG] Invalid values detected in {name}. dump={dump_path}", flush=True)
        print(f"[NAN_DEBUG] Context: {context}", flush=True)
        print(f"[NAN_DEBUG] Stats: {payload['stats']}", flush=True)
        raise RuntimeError(f"{name} contains NaN/Inf")

    def _check_finite(self, name: str, tensor: torch.Tensor, extra: dict | None = None) -> None:
        if torch.isfinite(tensor).all():
            return
        self._dump_invalid_tensor(name, tensor, extra)

    def _dump_debug_bundle(self, name: str, tensors: dict[str, torch.Tensor], extra: dict | None = None) -> None:
        context = {"learning_iteration": self.learning_iteration, **self.nan_debug_context}
        if extra:
            context.update(extra)
        self.nan_debug_dir.mkdir(parents=True, exist_ok=True)
        dump_path = self.nan_debug_dir / (
            f"algo_nan_it{context.get('iteration', context.get('learning_iteration', -1))}"
            f"_step{context.get('rollout_step', -1)}_mb{context.get('mini_batch', -1)}_{name}_bundle.pt"
        )
        payload = {
            "context": context,
            "name": name,
            "stats": {tensor_name: self._tensor_stats(tensor) for tensor_name, tensor in tensors.items()},
            "tensors": {tensor_name: tensor.detach().cpu() for tensor_name, tensor in tensors.items()},
        }
        torch.save(payload, dump_path)
        print(f"[NAN_DEBUG] Debug bundle saved for {name}. dump={dump_path}", flush=True)

    def _check_module_params(self, module: nn.Module, module_name: str, extra: dict | None = None) -> None:
        for name, param in module.named_parameters():
            if not torch.isfinite(param).all():
                self._dump_invalid_tensor(f"{module_name}.{name}", param, extra)

    def _build_actor_obs(self, obs, obs_history, encoder_out, commands):
        assert encoder_out.ndim == 2, f"encoder_out must be [batch, dim], got {tuple(encoder_out.shape)}"
        assert obs.ndim == 2, f"obs must be [batch, dim], got {tuple(obs.shape)}"
        assert commands.ndim == 2, f"commands must be [batch, dim], got {tuple(commands.shape)}"
        assert obs_history.ndim == 2, f"obs_history must be flattened [batch, dim], got {tuple(obs_history.shape)}"
        assert encoder_out.shape[0] == obs.shape[0] == commands.shape[0] == obs_history.shape[0], (
            "batch size mismatch for actor obs concat: "
            f"encoder_out={tuple(encoder_out.shape)}, obs={tuple(obs.shape)}, "
            f"commands={tuple(commands.shape)}, obs_history={tuple(obs_history.shape)}"
        )
        if self.policy_include_history:
            return torch.cat((encoder_out, obs, commands, obs_history), dim=-1)
        return torch.cat((encoder_out, obs, commands), dim=-1)

    @staticmethod
    def _split_imu_targets(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tensor[:, 0:3], tensor[:, 3:6], tensor[:, 6:9]

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        encoder_target_shape,
        obs_history_shape,
        commands_shape,
        action_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            encoder_target_shape,
            obs_history_shape,
            commands_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, obs_history, commands, critic_obs):
        self._check_finite("act.obs", obs)
        self._check_finite("act.obs_history", obs_history)
        self._check_finite("act.commands", commands)
        self._check_finite("act.critic_obs", critic_obs)
        critic_obs = torch.cat((critic_obs, commands), dim=-1)
        # act
        encoder_out = self.encoder.encode(obs_history)
        self._check_finite("act.encoder_out", encoder_out)
        #obs_history
        # actor_obs = self._build_actor_obs(obs, obs_history, commands)
        actor_obs = self._build_actor_obs(obs, obs_history, encoder_out, commands)
        self._check_finite("act.actor_obs", actor_obs)
        self.transition.actions = self.actor_critic.act(
            actor_obs
        ).detach()
        self._check_finite("act.action_mean", self.actor_critic.action_mean)
        self._check_finite("act.actions", self.transition.actions)

        # evaluate
        if self.critic_take_latent:
            critic_obs = torch.cat((critic_obs, encoder_out), dim=-1)
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self._check_finite("act.values", self.transition.values)

        # storage
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_obs = critic_obs
        self.transition.encoder_targets = None
        self.transition.observation_history = obs_history
        self.transition.commands = commands
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_obs=None, encoder_targets=None):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # Record the transition
        self.transition.next_observations = next_obs
        self.transition.encoder_targets = encoder_targets
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self, clear_storage=True):
        num_updates = 0
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_action_bound_loss = 0
        mean_kl = 0
        num_value_samples = 0
        returns_sum = 0.0
        returns_sq_sum = 0.0
        residual_sum = 0.0
        residual_sq_sum = 0.0
        generator = self.storage.mini_batch_generator(
            self.num_group,
            self.num_mini_batches,
            self.num_learning_epochs,
        )
        for mini_batch_idx, (
            obs_batch,
            critic_obs_batch,
            obs_history_batch, _,
            group_commands_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in enumerate(generator):
            batch_context = {"mini_batch": mini_batch_idx}
            self._check_finite("update.obs_batch", obs_batch, batch_context)
            self._check_finite("update.critic_obs_batch", critic_obs_batch, batch_context)
            self._check_finite("update.obs_history_batch", obs_history_batch, batch_context)
            self._check_finite("update.commands_batch", group_commands_batch, batch_context)
            self._check_finite("update.actions_batch", actions_batch, batch_context)
            self._check_finite("update.advantages_batch", advantages_batch, batch_context)
            self._check_finite("update.returns_batch", returns_batch, batch_context)
            encoder_out_batch = self.encoder.encode(obs_history_batch)
            self._check_finite("update.encoder_out_batch", encoder_out_batch, batch_context)
            commands_batch = group_commands_batch
            actor_obs_batch = self._build_actor_obs(obs_batch, obs_history_batch, encoder_out_batch, commands_batch)
            self._check_finite("update.actor_obs_batch", actor_obs_batch, batch_context)
            self.actor_critic.act(actor_obs_batch)
            self._check_finite("update.action_mean", self.actor_critic.action_mean, batch_context)

            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )
            self._check_finite("update.actions_log_prob_batch", actions_log_prob_batch, batch_context)

            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy
            self._check_finite("update.value_batch", value_batch, batch_context)
            self._check_finite("update.mu_batch", mu_batch, batch_context)
            self._check_finite("update.sigma_batch", sigma_batch, batch_context)
            self._check_finite("update.entropy_batch", entropy_batch, batch_context)

            kl_mean = torch.tensor(0, device=self.device, requires_grad=False)
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (
                        torch.square(old_sigma_batch)
                        + torch.square(old_mu_batch - mu_batch)
                    )
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)
            self._check_finite("update.kl_mean", kl_mean, batch_context)

            # KL
            if self.desired_kl != None and self.schedule == "adaptive":
                with torch.inference_mode():
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            if self.desired_kl != None and self.early_stop:
                if kl_mean > self.desired_kl * 1.5:
                    print("early stop, num_updates =", num_updates)
                    break

            # Surrogate loss
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            # print(ratio)
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            entropy_batch_mean = entropy_batch.mean()
            action_bound_loss = self._action_bound_loss(mu_batch)
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch_mean
                + self.action_bound_loss_coef * action_bound_loss
            )
            self._check_finite("update.surrogate_loss", surrogate_loss, batch_context)
            self._check_finite("update.action_bound_loss", action_bound_loss, batch_context)
            if not torch.isfinite(value_loss).all():
                self._dump_debug_bundle(
                    "update.value_loss",
                    {
                        "value_loss": value_loss,
                        "value_batch": value_batch,
                        "returns_batch": returns_batch,
                        "critic_obs_batch": critic_obs_batch,
                        "target_values_batch": target_values_batch,
                        "value_diff": value_batch - returns_batch,
                    },
                    batch_context,
                )
            self._check_finite("update.value_loss", value_loss, batch_context)
            self._check_finite("update.loss", loss, batch_context)

            if self.anneal_lr:
                frac = 1.0 - num_updates / (
                    self.num_learning_epochs * self.num_mini_batches
                )
                self.optimizer.param_groups[0]["lr"] = frac * self.learning_rate

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self._check_module_params(self.actor_critic, "actor_critic_pre_step", batch_context)
            self.optimizer.step()
            self._check_module_params(self.actor_critic, "actor_critic_post_step", batch_context)

            returns_detached = returns_batch.detach().double().reshape(-1)
            residual_detached = (returns_batch - value_batch).detach().double().reshape(-1)
            num_value_samples += returns_detached.numel()
            returns_sum += returns_detached.sum().item()
            returns_sq_sum += torch.square(returns_detached).sum().item()
            residual_sum += residual_detached.sum().item()
            residual_sq_sum += torch.square(residual_detached).sum().item()

            num_updates += 1
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_action_bound_loss += action_bound_loss.item()
            mean_kl += kl_mean.item()

        num_updates_extra = 0
        mean_extra_loss = 0
        mean_lin_vel_loss = 0
        mean_ang_vel_loss = 0
        mean_ang_vel_x_loss = 0
        mean_ang_vel_y_loss = 0
        mean_ang_vel_z_loss = 0
        mean_ang_vel_x_smooth_l1 = 0
        mean_proj_grav_loss = 0
        bootstrap_dim_mse_sum = None
        bootstrap_target_sum = None
        bootstrap_target_sq_sum = None
        bootstrap_target_count = 0
        ang_vel_target_count = 0
        ang_vel_target_sum = torch.zeros(3, dtype=torch.double)
        ang_vel_target_sq_sum = torch.zeros(3, dtype=torch.double)
        ang_vel_target_max = torch.full((3,), float("-inf"), dtype=torch.double)
        if self.extra_optimizer is not None:
            generator = self.storage.encoder_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
            for mini_batch_idx, (
                next_obs_batch,
                critic_obs_batch,
                encoder_target_batch,
                obs_history_batch,
            ) in enumerate(generator):
                batch_context = {"encoder_mini_batch": mini_batch_idx}
                self._check_finite("encoder.next_obs_batch", next_obs_batch, batch_context)
                self._check_finite("encoder.critic_obs_batch", critic_obs_batch, batch_context)
                if encoder_target_batch is not None:
                    self._check_finite("encoder.encoder_target_batch", encoder_target_batch, batch_context)
                self._check_finite("encoder.obs_history_batch", obs_history_batch, batch_context)
                if hasattr(self.encoder, "get_encoder_out"):
                    self.encoder.encode(obs_history_batch)
                    encode_batch = self.encoder.get_encoder_out()
                    self._check_finite("encoder.encode_batch", encode_batch, batch_context)

                if hasattr(self.encoder, "get_encoder_out"):
                    target_batch = self._select_encoder_target(critic_obs_batch, encoder_target_batch)
                    target_dim = target_batch.shape[1]
                    pred_batch = encode_batch[:, :target_dim]
                    pred_batch_detached = pred_batch.detach().double()
                    target_batch_detached = target_batch.detach().double()
                    diff_batch_detached = pred_batch_detached - target_batch_detached
                    dim_mse = torch.square(diff_batch_detached).mean(dim=0).cpu()
                    if bootstrap_dim_mse_sum is None:
                        bootstrap_dim_mse_sum = torch.zeros(target_dim, dtype=torch.double)
                        bootstrap_target_sum = torch.zeros(target_dim, dtype=torch.double)
                        bootstrap_target_sq_sum = torch.zeros(target_dim, dtype=torch.double)
                    bootstrap_dim_mse_sum += dim_mse
                    bootstrap_target_sum += target_batch_detached.sum(dim=0).cpu()
                    bootstrap_target_sq_sum += torch.square(target_batch_detached).sum(dim=0).cpu()
                    bootstrap_target_count += target_batch.shape[0]
                    if target_dim >= 9:
                        pred_lin_vel, pred_ang_vel, pred_proj_grav = self._split_imu_targets(pred_batch)
                        target_lin_vel, target_ang_vel, target_proj_grav = self._split_imu_targets(target_batch)
                        lin_vel_loss = (pred_lin_vel - target_lin_vel).pow(2).mean()
                        ang_vel_x_loss = (pred_ang_vel[:, 0] - target_ang_vel[:, 0]).pow(2).mean()
                        ang_vel_y_loss = (pred_ang_vel[:, 1] - target_ang_vel[:, 1]).pow(2).mean()
                        ang_vel_z_loss = (pred_ang_vel[:, 2] - target_ang_vel[:, 2]).pow(2).mean()
                        ang_vel_x_smooth_l1 = F.smooth_l1_loss(pred_ang_vel[:, 0], target_ang_vel[:, 0])
                        ang_vel_loss = (pred_ang_vel - target_ang_vel).pow(2).mean()
                        proj_grav_loss = (pred_proj_grav - target_proj_grav).pow(2).mean()
                        extra_loss = (lin_vel_loss + ang_vel_loss + proj_grav_loss) / 3.0

                        target_ang_vel_detached = target_ang_vel.detach().double()
                        ang_vel_target_count += target_ang_vel_detached.shape[0]
                        ang_vel_target_sum += target_ang_vel_detached.sum(dim=0).cpu()
                        ang_vel_target_sq_sum += torch.square(target_ang_vel_detached).sum(dim=0).cpu()
                        ang_vel_target_max = torch.maximum(
                            ang_vel_target_max,
                            target_ang_vel_detached.max(dim=0).values.cpu(),
                        )
                    else:
                        extra_loss = (pred_batch - target_batch).pow(2).mean()
                        lin_vel_loss = extra_loss
                        ang_vel_loss = torch.zeros_like(extra_loss)
                        ang_vel_x_loss = torch.zeros_like(extra_loss)
                        ang_vel_y_loss = torch.zeros_like(extra_loss)
                        ang_vel_z_loss = torch.zeros_like(extra_loss)
                        ang_vel_x_smooth_l1 = torch.zeros_like(extra_loss)
                        proj_grav_loss = torch.zeros_like(extra_loss)
                else:
                    extra_loss = torch.zeros_like(value_loss)
                    lin_vel_loss = torch.zeros_like(extra_loss)
                    ang_vel_loss = torch.zeros_like(extra_loss)
                    ang_vel_x_loss = torch.zeros_like(extra_loss)
                    ang_vel_y_loss = torch.zeros_like(extra_loss)
                    ang_vel_z_loss = torch.zeros_like(extra_loss)
                    ang_vel_x_smooth_l1 = torch.zeros_like(extra_loss)
                    proj_grav_loss = torch.zeros_like(extra_loss)
                self._check_finite("encoder.extra_loss", extra_loss, batch_context)
                self._check_finite("encoder.lin_vel_loss", lin_vel_loss, batch_context)
                self._check_finite("encoder.ang_vel_loss", ang_vel_loss, batch_context)
                self._check_finite("encoder.ang_vel_x_loss", ang_vel_x_loss, batch_context)
                self._check_finite("encoder.ang_vel_y_loss", ang_vel_y_loss, batch_context)
                self._check_finite("encoder.ang_vel_z_loss", ang_vel_z_loss, batch_context)
                self._check_finite("encoder.ang_vel_x_smooth_l1", ang_vel_x_smooth_l1, batch_context)
                self._check_finite("encoder.proj_grav_loss", proj_grav_loss, batch_context)

                self.extra_optimizer.zero_grad()
                extra_loss.backward()
                self.extra_optimizer.step()
                self._check_module_params(self.encoder, "encoder_post_step", batch_context)

                num_updates_extra += 1
                mean_extra_loss += extra_loss.item()
                mean_lin_vel_loss += lin_vel_loss.item()
                mean_ang_vel_loss += ang_vel_loss.item()
                mean_ang_vel_x_loss += ang_vel_x_loss.item()
                mean_ang_vel_y_loss += ang_vel_y_loss.item()
                mean_ang_vel_z_loss += ang_vel_z_loss.item()
                mean_ang_vel_x_smooth_l1 += ang_vel_x_smooth_l1.item()
                mean_proj_grav_loss += proj_grav_loss.item()

        mean_value_loss /= num_updates
        if num_updates_extra > 0:
            mean_extra_loss /= num_updates_extra
            mean_lin_vel_loss /= num_updates_extra
            mean_ang_vel_loss /= num_updates_extra
            mean_ang_vel_x_loss /= num_updates_extra
            mean_ang_vel_y_loss /= num_updates_extra
            mean_ang_vel_z_loss /= num_updates_extra
            mean_ang_vel_x_smooth_l1 /= num_updates_extra
            mean_proj_grav_loss /= num_updates_extra
        if ang_vel_target_count > 0:
            ang_vel_target_mean = ang_vel_target_sum / ang_vel_target_count
            ang_vel_target_second_moment = ang_vel_target_sq_sum / ang_vel_target_count
            ang_vel_target_std = torch.sqrt(
                torch.clamp(ang_vel_target_second_moment - torch.square(ang_vel_target_mean), min=0.0)
            )
            ang_vel_target_rmse = torch.sqrt(torch.clamp(ang_vel_target_second_moment, min=0.0))
        else:
            ang_vel_target_mean = torch.zeros(3, dtype=torch.double)
            ang_vel_target_std = torch.zeros(3, dtype=torch.double)
            ang_vel_target_max = torch.zeros(3, dtype=torch.double)
            ang_vel_target_rmse = torch.zeros(3, dtype=torch.double)
        bootstrap_metric = None
        bootstrap_metric_ang_vel = None
        if num_updates_extra > 0 and bootstrap_dim_mse_sum is not None and bootstrap_target_count > 0:
            mean_dim_mse = bootstrap_dim_mse_sum / num_updates_extra
            bootstrap_target_mean = bootstrap_target_sum / bootstrap_target_count
            bootstrap_target_second_moment = bootstrap_target_sq_sum / bootstrap_target_count
            bootstrap_target_var = torch.clamp(
                bootstrap_target_second_moment - torch.square(bootstrap_target_mean), min=0.0
            )
            normalized_dim_mse = mean_dim_mse / (bootstrap_target_var + self.bootstrap_metric_epsilon)
            bootstrap_metric = float(normalized_dim_mse.mean().item())
            if normalized_dim_mse.numel() >= 6:
                bootstrap_metric_ang_vel = float(normalized_dim_mse[3:6].mean().item())
        self.update_bootstrap_state(bootstrap_metric)
        mean_surrogate_loss /= num_updates
        mean_action_bound_loss /= num_updates
        self.last_action_bound_loss = mean_action_bound_loss
        self.last_action_bound_loss_weighted = self.action_bound_loss_coef * mean_action_bound_loss
        mean_kl /= num_updates
        mean_explained_variance = _compute_explained_variance_from_sums(
            num_value_samples, returns_sum, returns_sq_sum, residual_sum, residual_sq_sum
        )
        if clear_storage:
            self.storage.clear()

        return (
            mean_value_loss,
            mean_extra_loss,
            mean_lin_vel_loss,
            mean_ang_vel_loss,
            mean_ang_vel_x_loss,
            mean_ang_vel_y_loss,
            mean_ang_vel_z_loss,
            mean_ang_vel_x_smooth_l1,
            bootstrap_metric,
            bootstrap_metric_ang_vel,
            float(ang_vel_target_mean[0].item()),
            float(ang_vel_target_mean[1].item()),
            float(ang_vel_target_mean[2].item()),
            float(ang_vel_target_std[0].item()),
            float(ang_vel_target_std[1].item()),
            float(ang_vel_target_std[2].item()),
            float(ang_vel_target_max[0].item()),
            float(ang_vel_target_max[1].item()),
            float(ang_vel_target_max[2].item()),
            float(ang_vel_target_rmse[0].item()),
            float(ang_vel_target_rmse[1].item()),
            float(ang_vel_target_rmse[2].item()),
            mean_proj_grav_loss,
            mean_surrogate_loss,
            mean_kl,
            mean_explained_variance,
        )


class Implicit_PPO(IMU_PPO):
    actor_critic: ActorCritic
    encoder: ContextEstimatorNet

    def __init__(
        self,
        *args,
        terrain_prior_warmup_iters=0,
        terrain_property_target_dim=0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.terrain_prior_warmup_iters = int(terrain_prior_warmup_iters)
        self.terrain_property_target_dim = int(terrain_property_target_dim)

    @property
    def uses_terrain_labels(self) -> bool:
        return bool(
            getattr(self.encoder, "terrain_conditioned_prior", False)
            or getattr(self.encoder, "terrain_classifier", None) is not None
        )

    def _terrain_warmup_scale(self) -> float:
        if self.terrain_prior_warmup_iters <= 0:
            return 1.0
        return min(1.0, float(self.learning_iteration + 1) / float(self.terrain_prior_warmup_iters))

    def _ce_forward_pass(self, obs_history: torch.Tensor):
        """Run CE Net only; policy input construction is handled separately."""
        return self.encoder.forward_train(obs_history)

    def _build_context_policy_input(
        self,
        encoder_output: torch.Tensor,
        current_obs: torch.Tensor,
        commands: torch.Tensor,
        obs_history: torch.Tensor,
    ) -> torch.Tensor:
        """Build concat([encoder_output, current_obs, commands], dim=-1) with shape assertions."""
        if self.policy_include_history:
            obs_history = obs_history.flatten(start_dim=1) if obs_history.dim() > 2 else obs_history
        else:
            obs_history = None
        return build_context_policy_input(
            encoder_output,
            current_obs,
            commands,
            obs_history=obs_history,
            include_history=self.policy_include_history,
        )

    def _actor_forward(self, policy_input: torch.Tensor) -> torch.Tensor:
        """Run the nominal actor only."""
        return self.actor_critic.act(policy_input)

    def _select_velocity_target(self, critic_obs_batch: torch.Tensor, encoder_target_batch: torch.Tensor | None):
        if encoder_target_batch is not None:
            assert encoder_target_batch.ndim == 2 and encoder_target_batch.shape[1] >= 3, (
                f"encoder_target_batch must contain velocity target [batch, >=3], got {tuple(encoder_target_batch.shape)}"
            )
            return encoder_target_batch[:, :3]
        assert critic_obs_batch.ndim == 2 and critic_obs_batch.shape[1] >= 3, (
            f"critic_obs_batch must contain privileged base velocity [batch, >=3], got {tuple(critic_obs_batch.shape)}"
        )
        return critic_obs_batch[:, :3]

    def _prepare_policy_latent(self, v_t: torch.Tensor, z_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.encoder.output_detach:
            return v_t, z_t
        return v_t.detach(), z_t.detach()

    def act(self, obs, obs_history, commands, critic_obs):
        self._check_finite("act.obs", obs)
        self._check_finite("act.obs_history", obs_history)
        self._check_finite("act.commands", commands)
        self._check_finite("act.critic_obs", critic_obs)
        critic_obs = torch.cat((critic_obs, commands), dim=-1)

        # CE Net forward pass: obs_history [batch, H, obs_dim] -> v_t [batch, 3], z_t [batch, latent_dim].
        v_t, z_t, _mu, _logvar, _o_next_recon = self._ce_forward_pass(obs_history)
        self._check_finite("act.ce_v_t", v_t)
        self._check_finite("act.ce_z_t", z_t)
        v_policy, z_policy = self._prepare_policy_latent(v_t, z_t)
        encoder_output = torch.cat((v_policy, z_policy), dim=-1)

        # Policy input construction: concat([encoder_output, current_obs, commands], dim=-1).
        actor_obs = self._build_context_policy_input(encoder_output, obs, commands, obs_history)
        self._check_finite("act.actor_obs", actor_obs)

        # Actor forward pass.
        self.transition.actions = self._actor_forward(actor_obs).detach()
        self._check_finite("act.action_mean", self.actor_critic.action_mean)
        self._check_finite("act.actions", self.transition.actions)

        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self._check_finite("act.values", self.transition.values)

        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_obs = critic_obs
        self.transition.encoder_targets = None
        self.transition.observation_history = obs_history
        self.transition.commands = commands
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_obs=None, encoder_targets=None, transition_metadata=None):
        if transition_metadata is not None:
            self.transition.terrain_id = transition_metadata.get("terrain_id")
            self.transition.timestep = transition_metadata.get("timestep")
        super().process_env_step(
            rewards,
            dones,
            infos,
            next_obs=next_obs,
            encoder_targets=encoder_targets,
        )

    def update(self, clear_storage=True):
        num_updates = 0
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_action_bound_loss = 0
        mean_kl = 0
        num_value_samples = 0
        returns_sum = 0.0
        returns_sq_sum = 0.0
        residual_sum = 0.0
        residual_sq_sum = 0.0
        generator = self.storage.mini_batch_generator(
            self.num_group,
            self.num_mini_batches,
            self.num_learning_epochs,
        )
        for mini_batch_idx, (
            obs_batch,
            critic_obs_batch,
            obs_history_batch, _,
            _group_commands_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in enumerate(generator):
            batch_context = {"mini_batch": mini_batch_idx}
            self._check_finite("update.obs_batch", obs_batch, batch_context)
            self._check_finite("update.critic_obs_batch", critic_obs_batch, batch_context)
            self._check_finite("update.obs_history_batch", obs_history_batch, batch_context)

            # CE Net forward pass.
            v_t, z_t, _mu, _logvar, _o_next_recon = self._ce_forward_pass(obs_history_batch)
            self._check_finite("update.ce_v_t", v_t, batch_context)
            self._check_finite("update.ce_z_t", z_t, batch_context)
            v_policy, z_policy = self._prepare_policy_latent(v_t, z_t)
            encoder_output_batch = torch.cat((v_policy, z_policy), dim=-1)

            # Policy input construction.
            actor_obs_batch = self._build_context_policy_input(
                encoder_output_batch,
                obs_batch,
                _group_commands_batch,
                obs_history_batch,
            )
            self._check_finite("update.actor_obs_batch", actor_obs_batch, batch_context)

            # Actor forward pass.
            self._actor_forward(actor_obs_batch)
            self._check_finite("update.action_mean", self.actor_critic.action_mean, batch_context)

            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            kl_mean = torch.tensor(0, device=self.device, requires_grad=False)
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            entropy_batch_mean = entropy_batch.mean()
            action_bound_loss = self._action_bound_loss(mu_batch)
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch_mean
                + self.action_bound_loss_coef * action_bound_loss
            )
            self._check_finite("update.action_bound_loss", action_bound_loss, batch_context)
            self._check_finite("update.loss", loss, batch_context)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            returns_detached = returns_batch.detach().double().reshape(-1)
            residual_detached = (returns_batch - value_batch).detach().double().reshape(-1)
            num_value_samples += returns_detached.numel()
            returns_sum += returns_detached.sum().item()
            returns_sq_sum += torch.square(returns_detached).sum().item()
            residual_sum += residual_detached.sum().item()
            residual_sq_sum += torch.square(residual_detached).sum().item()

            num_updates += 1
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_action_bound_loss += action_bound_loss.item()
            mean_kl += kl_mean.item()

        num_updates_extra = 0
        mean_extra_loss = 0
        mean_ce_terms = {
            "loss": 0.0,
            "est_loss": 0.0,
            "vae_loss": 0.0,
            "recon_loss": 0.0,
            "kl_loss": 0.0,
        }
        if self.extra_optimizer is not None:
            generator = self.storage.encoder_mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
                include_terrain_id=self.uses_terrain_labels,
            )
            for mini_batch_idx, batch in enumerate(generator):
                if self.uses_terrain_labels:
                    (
                        next_obs_batch,
                        critic_obs_batch,
                        encoder_target_batch,
                        obs_history_batch,
                        terrain_id_batch,
                    ) = batch
                else:
                    next_obs_batch, critic_obs_batch, encoder_target_batch, obs_history_batch = batch
                    terrain_id_batch = None
                batch_context = {"encoder_mini_batch": mini_batch_idx}
                self._check_finite("encoder.next_obs_batch", next_obs_batch, batch_context)
                self._check_finite("encoder.critic_obs_batch", critic_obs_batch, batch_context)
                self._check_finite("encoder.obs_history_batch", obs_history_batch, batch_context)
                v_target_batch = self._select_velocity_target(critic_obs_batch, encoder_target_batch)
                terrain_property_target = None
                if (
                    encoder_target_batch is not None
                    and self.terrain_property_target_dim > 0
                    and encoder_target_batch.shape[1] >= 3 + self.terrain_property_target_dim
                ):
                    terrain_property_target = encoder_target_batch[:, 3 : 3 + self.terrain_property_target_dim]
                ce_loss, ce_terms = self.encoder.compute_loss(
                    obs_history_batch,
                    v_target_batch,
                    next_obs_batch,
                    beta=self.vae_beta,
                    terrain_id=terrain_id_batch,
                    terrain_property_target=terrain_property_target,
                    terrain_warmup_scale=self._terrain_warmup_scale(),
                )
                for name, value in ce_terms.items():
                    self._check_finite(f"encoder.{name}", value, batch_context)

                self.extra_optimizer.zero_grad()
                ce_loss.backward()
                nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
                self.extra_optimizer.step()
                self._check_module_params(self.encoder, "context_estimator_post_step", batch_context)

                num_updates_extra += 1
                mean_extra_loss += ce_loss.item()
                mean_ce_terms["loss"] += ce_terms["ce_loss"].item()
                mean_ce_terms["est_loss"] += ce_terms["estimation_loss"].item()
                mean_ce_terms["vae_loss"] += ce_terms["vae_loss"].item()
                mean_ce_terms["recon_loss"] += ce_terms["reconstruction_loss"].item()
                mean_ce_terms["kl_loss"] += ce_terms["kl_loss"].item()
                if "terrain_classification_loss" in ce_terms:
                    mean_ce_terms.setdefault("terrain_cls_loss", 0.0)
                    mean_ce_terms["terrain_cls_loss"] += ce_terms["terrain_classification_loss"].item()
                if "terrain_property_loss" in ce_terms:
                    mean_ce_terms.setdefault("terrain_prop_loss", 0.0)
                    mean_ce_terms["terrain_prop_loss"] += ce_terms["terrain_property_loss"].item()
                if "terrain_warmup_scale" in ce_terms:
                    mean_ce_terms.setdefault("terrain_warmup_scale", 0.0)
                    mean_ce_terms["terrain_warmup_scale"] += ce_terms["terrain_warmup_scale"].item()

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_action_bound_loss /= num_updates
        self.last_action_bound_loss = mean_action_bound_loss
        self.last_action_bound_loss_weighted = self.action_bound_loss_coef * mean_action_bound_loss
        mean_kl /= num_updates
        if num_updates_extra > 0:
            mean_extra_loss /= num_updates_extra
            mean_ce_terms = {name: value / num_updates_extra for name, value in mean_ce_terms.items()}
            mean_ce_terms["beta"] = float(self.vae_beta)
        else:
            mean_ce_terms = None
        mean_explained_variance = _compute_explained_variance_from_sums(
            num_value_samples, returns_sum, returns_sq_sum, residual_sum, residual_sq_sum
        )
        if clear_storage:
            self.storage.clear()

        return (
            mean_value_loss,
            mean_extra_loss,
            mean_surrogate_loss,
            mean_kl,
            mean_explained_variance,
            mean_ce_terms,
        )


class Phase2_Adaptive_PPO(Implicit_PPO):
    """MULE-style Phase 2 PPO with separate nominal and adaptive policies."""

    adaptive_actor_critic: AdaptiveActorCritic

    def __init__(
        self,
        num_group,
        encoder,
        actor_critic,
        adaptive_obs_dim,
        adaptive_critic_obs_dim=None,
        adaptive_policy_cfg=None,
        adaptive_learning_rate=1.0e-3,
        adaptive_value_loss_coef=None,
        adaptive_entropy_coef=None,
        adaptive_reward_scale=1.0,
        **kwargs,
    ):
        super().__init__(num_group, encoder, actor_critic, **kwargs)
        if adaptive_obs_dim is None or adaptive_obs_dim <= 0:
            raise ValueError(f"adaptive_obs_dim must be positive for Phase 2, got {adaptive_obs_dim}")
        if adaptive_critic_obs_dim is None:
            adaptive_critic_obs_dim = adaptive_obs_dim
        if adaptive_critic_obs_dim <= 0:
            raise ValueError(
                f"adaptive_critic_obs_dim must be positive for Phase 2, got {adaptive_critic_obs_dim}"
            )

        adaptive_policy_cfg = dict(adaptive_policy_cfg or {})
        self.adaptive_value_loss_coef = (
            self.value_loss_coef if adaptive_value_loss_coef is None else adaptive_value_loss_coef
        )
        self.adaptive_entropy_coef = self.entropy_coef if adaptive_entropy_coef is None else adaptive_entropy_coef
        self.adaptive_reward_scale = float(adaptive_reward_scale)
        self.adaptive_actor_critic = AdaptiveActorCritic(
            adaptive_obs_dim,
            adaptive_critic_obs_dim,
            self.actor_critic.actor[-1].out_features,
            **adaptive_policy_cfg,
        ).to(self.device)
        self.adaptive_optimizer = optim.Adam(
            [{"params": self.adaptive_actor_critic.parameters()}],
            lr=adaptive_learning_rate,
        )
        self.adaptive_learning_rate = adaptive_learning_rate
        self.last_loss_terms = {}

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        encoder_target_shape,
        obs_history_shape,
        commands_shape,
        action_shape,
        adaptive_obs_shape=None,
        adaptive_critic_obs_shape=None,
        adaptive_history_shape=None,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            encoder_target_shape,
            obs_history_shape,
            commands_shape,
            action_shape,
            self.device,
            adaptive_obs_shape=adaptive_obs_shape,
            adaptive_critic_obs_shape=adaptive_critic_obs_shape,
            adaptive_history_shape=adaptive_history_shape,
        )

    def act(self, obs, obs_history, commands, critic_obs, adaptive_obs=None, adaptive_critic_obs=None):
        if adaptive_obs is None:
            raise ValueError("Phase2_Adaptive_PPO.act requires adaptive_obs")
        if adaptive_critic_obs is None:
            adaptive_critic_obs = adaptive_obs
        self._check_finite("act.adaptive_obs", adaptive_obs)
        self._check_finite("act.adaptive_critic_obs", adaptive_critic_obs)

        critic_obs = torch.cat((critic_obs, commands), dim=-1)
        v_t, z_t, _mu, _logvar, _o_next_recon = self._ce_forward_pass(obs_history)
        self._check_finite("act.ce_v_t", v_t)
        self._check_finite("act.ce_z_t", z_t)
        v_policy, z_policy = self._prepare_policy_latent(v_t, z_t)
        encoder_output = torch.cat((v_policy, z_policy), dim=-1)

        actor_obs = self._build_context_policy_input(encoder_output, obs, commands, obs_history)
        self._check_finite("act.actor_obs", actor_obs)
        nominal_action = self._actor_forward(actor_obs)
        self._check_finite("act.nominal_action", nominal_action)

        raw_adaptive_action = self.adaptive_actor_critic.act(adaptive_obs)
        delta_action = self.adaptive_actor_critic.raw_to_delta(raw_adaptive_action)
        final_action = nominal_action + delta_action
        self._check_finite("act.raw_adaptive_action", raw_adaptive_action)
        self._check_finite("act.delta_action", delta_action)
        self._check_finite("act.final_action", final_action)

        self.transition.actions = nominal_action.detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(nominal_action).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_obs = critic_obs
        self.transition.encoder_targets = None
        self.transition.observation_history = obs_history
        self.transition.commands = commands

        self.transition.adaptive_observations = adaptive_obs
        self.transition.adaptive_critic_observations = adaptive_critic_obs
        self.transition.adaptive_values = self.adaptive_actor_critic.evaluate(adaptive_critic_obs).detach()
        self.transition.adaptive_actions = raw_adaptive_action.detach()
        self.transition.adaptive_actions_log_prob = self.adaptive_actor_critic.get_actions_log_prob(
            raw_adaptive_action
        ).detach()
        self.transition.adaptive_action_mean = self.adaptive_actor_critic.action_mean.detach()
        self.transition.adaptive_action_sigma = self.adaptive_actor_critic.action_std.detach()
        return final_action.detach()

    def process_env_step(
        self,
        rewards,
        dones,
        infos,
        next_obs=None,
        encoder_targets=None,
        adaptive_rewards=None,
        transition_metadata=None,
    ):
        if adaptive_rewards is None:
            raise ValueError("Phase2_Adaptive_PPO.process_env_step requires adaptive_rewards")
        self.transition.adaptive_rewards = adaptive_rewards.clone() * self.adaptive_reward_scale
        if transition_metadata is not None:
            self.transition.payload_mass = transition_metadata.get("payload_mass")
            self.transition.payload_pos_b = transition_metadata.get("payload_pos_b")
            self.transition.terrain_id = transition_metadata.get("terrain_id")
            self.transition.timestep = transition_metadata.get("timestep")
        super().process_env_step(
            rewards,
            dones,
            infos,
            next_obs=next_obs,
            encoder_targets=encoder_targets,
        )
        self.adaptive_actor_critic.reset(dones)

    def compute_adaptive_returns(self, last_adaptive_critic_obs):
        last_values = self.adaptive_actor_critic.evaluate(last_adaptive_critic_obs).detach()
        self.storage.compute_adaptive_returns(last_values, self.gamma, self.lam)

    def _update_adaptive_policy(self):
        num_updates = 0
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_kl = 0.0
        generator = self.storage.adaptive_mini_batch_generator(
            self.num_group,
            self.num_mini_batches,
            self.num_learning_epochs,
        )
        for mini_batch_idx, (
            adaptive_obs_batch,
            adaptive_critic_obs_batch,
            raw_actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in enumerate(generator):
            batch_context = {"adaptive_mini_batch": mini_batch_idx}
            self._check_finite("adaptive.obs_batch", adaptive_obs_batch, batch_context)
            self._check_finite("adaptive.critic_obs_batch", adaptive_critic_obs_batch, batch_context)
            self.adaptive_actor_critic.act(adaptive_obs_batch)
            actions_log_prob_batch = self.adaptive_actor_critic.get_actions_log_prob(raw_actions_batch)
            value_batch = self.adaptive_actor_critic.evaluate(adaptive_critic_obs_batch)
            mu_batch = self.adaptive_actor_critic.action_mean
            sigma_batch = self.adaptive_actor_critic.action_std
            entropy_batch = self.adaptive_actor_critic.entropy

            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.adaptive_value_loss_coef * value_loss
                - self.adaptive_entropy_coef * entropy_batch.mean()
            )
            self._check_finite("adaptive.loss", loss, batch_context)

            self.adaptive_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.adaptive_actor_critic.parameters(), self.max_grad_norm)
            self.adaptive_optimizer.step()
            self.adaptive_actor_critic.clamp_logstd_()

            num_updates += 1
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_kl += kl_mean.item()

        if num_updates == 0:
            return 0.0, 0.0, 0.0
        return mean_value_loss / num_updates, mean_surrogate_loss / num_updates, mean_kl / num_updates

    def update(self):
        nominal_result = super().update(clear_storage=False)
        adaptive_value_loss, adaptive_surrogate_loss, adaptive_kl = self._update_adaptive_policy()
        self.storage.clear()

        (
            nominal_value_loss,
            ce_loss,
            nominal_surrogate_loss,
            nominal_kl,
            explained_variance,
            ce_loss_terms,
        ) = nominal_result
        nominal_loss = nominal_surrogate_loss + self.value_loss_coef * nominal_value_loss
        adaptive_loss = adaptive_surrogate_loss + self.adaptive_value_loss_coef * adaptive_value_loss
        total_loss = nominal_loss + adaptive_loss + ce_loss
        self.last_loss_terms = {
            "nominal_loss": float(nominal_loss),
            "adaptive_loss": float(adaptive_loss),
            "ce_loss": float(ce_loss),
            "total_loss": float(total_loss),
            "adaptive_value_loss": float(adaptive_value_loss),
            "adaptive_surrogate_loss": float(adaptive_surrogate_loss),
            "adaptive_kl": float(adaptive_kl),
            "adaptive_reward_scale": float(self.adaptive_reward_scale),
        }
        return (
            nominal_value_loss,
            ce_loss,
            nominal_surrogate_loss,
            nominal_kl,
            explained_variance,
            ce_loss_terms,
            self.last_loss_terms,
        )


class Phase2_LoadAdaptive_PPO(Phase2_Adaptive_PPO):
    """Phase 2 residual policy conditioned on a load-response latent."""

    def __init__(
        self,
        num_group,
        encoder,
        actor_critic,
        adaptive_obs_dim,
        adaptive_critic_obs_dim=None,
        adaptive_policy_cfg=None,
        load_history_shape=None,
        load_latent_dim=16,
        load_encoder_hidden_dims=(128, 64),
        load_transition_hidden_dims=(128, 64),
        load_activation="elu",
        load_orthogonal_init=False,
        load_learning_rate=1.0e-4,
        load_triplet_margin=1.0,
        load_triplet_loss_coef=1.0,
        load_hard_negative_mass_diff_min=5.0,
        load_hard_negative_fallback_mass_diff_min=3.0,
        load_hard_negative_command_dist_max=0.5,
        load_hard_negative_num_candidates=32,
        load_probe_learning_rate=1.0e-4,
        **kwargs,
    ):
        if load_history_shape is None:
            raise ValueError("Phase2_LoadAdaptive_PPO requires load_history_shape")
        self.raw_adaptive_obs_dim = int(adaptive_obs_dim)
        self.load_history_shape = tuple(load_history_shape)
        self.load_history_dim = int(torch.tensor(self.load_history_shape).prod().item())
        self.load_latent_dim = int(load_latent_dim)
        adaptive_actor_obs_dim = self.raw_adaptive_obs_dim + self.load_latent_dim

        super().__init__(
            num_group,
            encoder,
            actor_critic,
            adaptive_obs_dim=adaptive_actor_obs_dim,
            adaptive_critic_obs_dim=adaptive_critic_obs_dim,
            adaptive_policy_cfg=adaptive_policy_cfg,
            **kwargs,
        )

        self.load_encoder = LoadEncoder(
            self.load_history_dim,
            latent_dim=self.load_latent_dim,
            hidden_dims=load_encoder_hidden_dims,
            activation=load_activation,
            orthogonal_init=load_orthogonal_init,
        ).to(self.device)
        self.load_transition_model = LoadTransitionModel(
            self.load_latent_dim,
            self.actor_critic.actor[-1].out_features,
            hidden_dims=load_transition_hidden_dims,
            activation=load_activation,
            orthogonal_init=load_orthogonal_init,
        ).to(self.device)
        self.load_triplet_margin = float(load_triplet_margin)
        self.load_triplet_loss_coef = float(load_triplet_loss_coef)
        self.load_learning_rate = float(load_learning_rate)
        self.load_hard_negative_mass_diff_min = float(load_hard_negative_mass_diff_min)
        self.load_hard_negative_fallback_mass_diff_min = float(load_hard_negative_fallback_mass_diff_min)
        self.load_hard_negative_command_dist_max = float(load_hard_negative_command_dist_max)
        self.load_hard_negative_num_candidates = int(load_hard_negative_num_candidates)
        action_dim = self.actor_critic.actor[-1].out_features
        self.load_payload_mass_probe = nn.Linear(self.load_latent_dim, 1).to(self.device)
        self.load_payload_pos_probe = nn.Linear(self.load_latent_dim, 3).to(self.device)
        self.load_force_probe = nn.Linear(self.load_latent_dim, action_dim).to(self.device)
        self.load_delta_action_probe = nn.Linear(self.load_latent_dim, action_dim).to(self.device)
        self.load_probe_optimizer = optim.Adam(
            list(self.load_payload_mass_probe.parameters())
            + list(self.load_payload_pos_probe.parameters())
            + list(self.load_force_probe.parameters())
            + list(self.load_delta_action_probe.parameters()),
            lr=float(load_probe_learning_rate),
        )
        self.last_load_diagnostics = {}
        self.last_probe_terms = {}
        self.adaptive_optimizer = optim.Adam(
            [
                {"params": self.adaptive_actor_critic.parameters(), "lr": self.adaptive_learning_rate},
                {"params": self.load_encoder.parameters(), "lr": self.load_learning_rate},
                {"params": self.load_transition_model.parameters(), "lr": self.load_learning_rate},
            ]
        )

    def _build_load_adaptive_input(self, adaptive_obs, adaptive_history):
        z_load = self.load_encoder(adaptive_history)
        self._check_finite("load.z_load", z_load)
        return torch.cat((adaptive_obs, z_load), dim=-1), z_load

    @staticmethod
    def _r2_score(pred, target):
        target_mean = target.mean(dim=0, keepdim=True)
        ss_res = torch.sum(torch.square(pred - target))
        ss_tot = torch.sum(torch.square(target - target_mean))
        return 1.0 - ss_res / (ss_tot + 1.0e-8)

    def _update_load_probes(self, z_load, payload_mass, payload_pos_b, adaptive_obs, delta_action):
        z_detached = z_load.detach()
        payload_mass = payload_mass.view(-1, 1).detach()
        payload_pos_b = payload_pos_b.detach()
        force_dim = delta_action.shape[-1]
        force_target = adaptive_obs[:, -force_dim:].detach()
        delta_target = delta_action.detach()

        mass_pred = self.load_payload_mass_probe(z_detached)
        pos_pred = self.load_payload_pos_probe(z_detached)
        force_pred = self.load_force_probe(z_detached)
        delta_pred = self.load_delta_action_probe(z_detached)

        mass_loss = F.mse_loss(mass_pred, payload_mass)
        pos_loss = F.mse_loss(pos_pred, payload_pos_b)
        force_loss = F.mse_loss(force_pred, force_target)
        delta_loss = F.mse_loss(delta_pred, delta_target)
        loss = mass_loss + pos_loss + force_loss + delta_loss

        self.load_probe_optimizer.zero_grad()
        loss.backward()
        self.load_probe_optimizer.step()

        return {
            "payload_mass_mse": float(mass_loss.detach()),
            "payload_mass_r2": float(self._r2_score(mass_pred.detach(), payload_mass)),
            "payload_pos_mse": float(pos_loss.detach()),
            "payload_pos_r2": float(self._r2_score(pos_pred.detach(), payload_pos_b)),
            "force_mse": float(force_loss.detach()),
            "delta_action_mse": float(delta_loss.detach()),
        }

    def _select_hard_negative_indices(self, payload_mass, commands):
        batch_size = payload_mass.shape[0]
        if batch_size <= 1:
            return None, {}
        num_candidates = max(1, self.load_hard_negative_num_candidates)
        candidates = torch.randint(batch_size, (batch_size, num_candidates), device=self.device)
        row_idx = torch.arange(batch_size, device=self.device).unsqueeze(1)
        candidates = torch.where(candidates == row_idx, (candidates + 1) % batch_size, candidates)

        mass = payload_mass.view(-1)
        mass_diff = torch.abs(mass.unsqueeze(1) - mass[candidates])
        command_dist = torch.linalg.norm(commands.unsqueeze(1) - commands[candidates], dim=-1)
        hard_mask = (
            (mass_diff >= self.load_hard_negative_mass_diff_min)
            & (command_dist <= self.load_hard_negative_command_dist_max)
        )
        fallback_mask = mass_diff >= self.load_hard_negative_fallback_mass_diff_min

        hard_any = hard_mask.any(dim=1)
        fallback_any = fallback_mask.any(dim=1)
        random_choice = torch.randint(num_candidates, (batch_size,), device=self.device)
        hard_choice = hard_mask.float().argmax(dim=1)
        fallback_choice = fallback_mask.float().argmax(dim=1)
        choice = torch.where(hard_any, hard_choice, torch.where(fallback_any, fallback_choice, random_choice))
        neg_idx = candidates[torch.arange(batch_size, device=self.device), choice]

        selected_mass_diff = torch.abs(mass - mass[neg_idx])
        selected_command_dist = torch.linalg.norm(commands - commands[neg_idx], dim=-1)
        diagnostics = {
            "hard_negative_success_rate": float(hard_any.float().mean()),
            "hard_negative_fallback_rate": float((~hard_any & fallback_any).float().mean()),
            "hard_negative_mass_diff": float(selected_mass_diff.mean()),
            "hard_negative_command_dist": float(selected_command_dist.mean()),
        }
        return neg_idx, diagnostics

    def act(self, obs, obs_history, commands, critic_obs, adaptive_obs=None, adaptive_critic_obs=None, adaptive_history=None):
        if adaptive_obs is None:
            raise ValueError("Phase2_LoadAdaptive_PPO.act requires adaptive_obs")
        if adaptive_history is None:
            raise ValueError("Phase2_LoadAdaptive_PPO.act requires adaptive_history")
        if adaptive_critic_obs is None:
            adaptive_critic_obs = adaptive_obs
        self._check_finite("act.adaptive_obs", adaptive_obs)
        self._check_finite("act.adaptive_history", adaptive_history)
        self._check_finite("act.adaptive_critic_obs", adaptive_critic_obs)

        critic_obs = torch.cat((critic_obs, commands), dim=-1)
        v_t, z_t, _mu, _logvar, _o_next_recon = self._ce_forward_pass(obs_history)
        self._check_finite("act.ce_v_t", v_t)
        self._check_finite("act.ce_z_t", z_t)
        v_policy, z_policy = self._prepare_policy_latent(v_t, z_t)
        encoder_output = torch.cat((v_policy, z_policy), dim=-1)

        actor_obs = self._build_context_policy_input(encoder_output, obs, commands, obs_history)
        self._check_finite("act.actor_obs", actor_obs)
        nominal_action = self._actor_forward(actor_obs)
        self._check_finite("act.nominal_action", nominal_action)

        adaptive_policy_input, _z_load = self._build_load_adaptive_input(adaptive_obs, adaptive_history)
        raw_adaptive_action = self.adaptive_actor_critic.act(adaptive_policy_input)
        delta_action = self.adaptive_actor_critic.raw_to_delta(raw_adaptive_action)
        final_action = nominal_action + delta_action
        self._check_finite("act.raw_adaptive_action", raw_adaptive_action)
        self._check_finite("act.delta_action", delta_action)
        self._check_finite("act.final_action", final_action)

        self.transition.actions = nominal_action.detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(nominal_action).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_obs = critic_obs
        self.transition.encoder_targets = None
        self.transition.observation_history = obs_history
        self.transition.commands = commands

        self.transition.adaptive_observations = adaptive_obs
        self.transition.adaptive_critic_observations = adaptive_critic_obs
        self.transition.adaptive_history = adaptive_history
        self.transition.adaptive_values = self.adaptive_actor_critic.evaluate(adaptive_critic_obs).detach()
        self.transition.adaptive_actions = raw_adaptive_action.detach()
        self.transition.adaptive_actions_log_prob = self.adaptive_actor_critic.get_actions_log_prob(
            raw_adaptive_action
        ).detach()
        self.transition.adaptive_action_mean = self.adaptive_actor_critic.action_mean.detach()
        self.transition.adaptive_action_sigma = self.adaptive_actor_critic.action_std.detach()
        return final_action.detach()

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        encoder_target_shape,
        obs_history_shape,
        commands_shape,
        action_shape,
        adaptive_obs_shape=None,
        adaptive_critic_obs_shape=None,
        adaptive_history_shape=None,
    ):
        if adaptive_history_shape is None:
            adaptive_history_shape = list(self.load_history_shape)
        super().init_storage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            encoder_target_shape,
            obs_history_shape,
            commands_shape,
            action_shape,
            adaptive_obs_shape=adaptive_obs_shape,
            adaptive_critic_obs_shape=adaptive_critic_obs_shape,
            adaptive_history_shape=adaptive_history_shape,
        )

    def _update_adaptive_policy(self):
        num_updates = 0
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_kl = 0.0
        probe_sums = {}
        generator = self.storage.adaptive_load_mini_batch_generator(
            self.num_group,
            self.num_mini_batches,
            self.num_learning_epochs,
        )
        for mini_batch_idx, (
            adaptive_obs_batch,
            adaptive_history_batch,
            adaptive_critic_obs_batch,
            raw_actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            payload_mass_batch,
            payload_pos_b_batch,
        ) in enumerate(generator):
            batch_context = {"adaptive_load_mini_batch": mini_batch_idx}
            self._check_finite("adaptive.obs_batch", adaptive_obs_batch, batch_context)
            self._check_finite("adaptive.history_batch", adaptive_history_batch, batch_context)
            adaptive_policy_input, _z_load = self._build_load_adaptive_input(
                adaptive_obs_batch, adaptive_history_batch
            )
            self.adaptive_actor_critic.act(adaptive_policy_input)
            actions_log_prob_batch = self.adaptive_actor_critic.get_actions_log_prob(raw_actions_batch)
            value_batch = self.adaptive_actor_critic.evaluate(adaptive_critic_obs_batch)
            mu_batch = self.adaptive_actor_critic.action_mean
            sigma_batch = self.adaptive_actor_critic.action_std
            entropy_batch = self.adaptive_actor_critic.entropy
            delta_action_batch = self.adaptive_actor_critic.raw_to_delta(raw_actions_batch)
            probe_terms = self._update_load_probes(
                _z_load,
                payload_mass_batch,
                payload_pos_b_batch,
                adaptive_obs_batch,
                delta_action_batch,
            )
            for key, value in probe_terms.items():
                probe_sums[key] = probe_sums.get(key, 0.0) + value

            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.adaptive_value_loss_coef * value_loss
                - self.adaptive_entropy_coef * entropy_batch.mean()
            )
            self._check_finite("adaptive.loss", loss, batch_context)

            self.adaptive_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.adaptive_actor_critic.parameters())
                + list(self.load_encoder.parameters())
                + list(self.load_transition_model.parameters()),
                self.max_grad_norm,
            )
            self.adaptive_optimizer.step()
            self.adaptive_actor_critic.clamp_logstd_()

            num_updates += 1
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_kl += kl_mean.item()

        if num_updates == 0:
            return 0.0, 0.0, 0.0
        self.last_probe_terms = {key: value / num_updates for key, value in probe_sums.items()}
        return mean_value_loss / num_updates, mean_surrogate_loss / num_updates, mean_kl / num_updates

    def _update_load_transition_model(self):
        num_updates = 0
        mean_triplet_loss = 0.0
        diag_sums = {}
        generator = self.storage.load_transition_mini_batch_generator(
            self.num_group,
            self.num_mini_batches,
            self.num_learning_epochs,
        )
        for mini_batch_idx, (
            _adaptive_obs_batch,
            adaptive_history_batch,
            next_adaptive_history_batch,
            nominal_actions_batch,
            raw_adaptive_actions_batch,
            commands_batch,
            payload_mass_batch,
            next_payload_mass_batch,
            payload_pos_b_batch,
            next_payload_pos_b_batch,
            _terrain_id_batch,
            _env_id_batch,
            _timestep_batch,
        ) in enumerate(generator):
            if adaptive_history_batch.shape[0] < 2:
                continue
            batch_context = {"load_transition_mini_batch": mini_batch_idx}
            z_load = self.load_encoder(adaptive_history_batch)
            z_load_next = self.load_encoder(next_adaptive_history_batch)

            delta_action = self.adaptive_actor_critic.raw_to_delta(raw_adaptive_actions_batch)
            z_load_hat_next = self.load_transition_model(z_load, nominal_actions_batch, delta_action)

            neg_idx, hard_diag = self._select_hard_negative_indices(payload_mass_batch, commands_batch)
            if neg_idx is None:
                continue

            z_load_neg = z_load_next[neg_idx].detach()
            z_load_next_target = z_load_next.detach()

            # ------------------------------------------------------------
            # Normalize only for triplet distance computation.
            # This prevents distance explosion caused by growing latent norms.
            # ------------------------------------------------------------
            z_hat_loss = F.normalize(z_load_hat_next, p=2, dim=-1, eps=1.0e-6)
            z_next_loss = F.normalize(z_load_next_target, p=2, dim=-1, eps=1.0e-6)
            z_neg_loss = F.normalize(z_load_neg, p=2, dim=-1, eps=1.0e-6)

            positive_dist = torch.sum(torch.square(z_hat_loss - z_next_loss), dim=-1)
            negative_dist = torch.sum(torch.square(z_hat_loss - z_neg_loss), dim=-1)

            triplet_raw = positive_dist - negative_dist + self.load_triplet_margin
            triplet_loss = torch.relu(triplet_raw).mean()
            loss = self.load_triplet_loss_coef * triplet_loss
            self._check_finite("load.triplet_loss", loss, batch_context)

            self.adaptive_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.load_encoder.parameters()) + list(self.load_transition_model.parameters()),
                self.max_grad_norm,
            )
            self.adaptive_optimizer.step()

            num_updates += 1
            mean_triplet_loss += triplet_loss.item()
            payload_change = (
                (torch.abs(next_payload_mass_batch.view(-1) - payload_mass_batch.view(-1)) > 1.0e-5)
                | (torch.linalg.norm(next_payload_pos_b_batch - payload_pos_b_batch, dim=-1) > 1.0e-5)
            )
            z_shift = torch.linalg.norm(z_load_next.detach() - z_load.detach(), dim=-1)
            delta_norm = torch.linalg.norm(delta_action.detach(), dim=-1)
            raw_delta = raw_adaptive_actions_batch.detach()
            tanh_delta = torch.tanh(raw_delta)
            logstd = self.adaptive_actor_critic.clamped_logstd.detach()
            diag_terms = {
                "load_pos_dist": float(positive_dist.detach().mean()),
                "load_neg_dist": float(negative_dist.detach().mean()),
                "load_margin": self.load_triplet_margin,
                "load_active_ratio": float((triplet_raw.detach() > 0.0).float().mean()),
                "z_load_norm": float(torch.linalg.norm(z_load.detach(), dim=-1).mean()),
                "z_load_std": float(z_load.detach().std(dim=0).mean()),
                "z_load_step_delta_norm": float(z_shift.mean()),
                "delta_action_norm": float(delta_norm.mean()),
                "delta_action_saturation_ratio": float((torch.abs(tanh_delta) > 0.95).float().mean()),
                "raw_delta_action_abs_mean": float(torch.abs(raw_delta).mean()),
                "adaptive_logstd_mean": float(logstd.mean()),
                "adaptive_logstd_max": float(logstd.max()),
                "adaptive_action_std_mean": float(torch.exp(logstd).mean()),
                "nominal_action_norm": float(torch.linalg.norm(nominal_actions_batch.detach(), dim=-1).mean()),
                "payload_change_rate": float(payload_change.float().mean()),
                "z_load_shift_after_payload_change": float(z_shift[payload_change].mean()) if payload_change.any() else 0.0,
                "delta_action_response_after_payload_change": float(delta_norm[payload_change].mean())
                if payload_change.any()
                else 0.0,
            }
            diag_terms.update(hard_diag)
            for key, value in diag_terms.items():
                diag_sums[key] = diag_sums.get(key, 0.0) + value

        if num_updates == 0:
            self.last_load_diagnostics = {}
            return 0.0
        self.last_load_diagnostics = {key: value / num_updates for key, value in diag_sums.items()}
        return mean_triplet_loss / num_updates

    def update(self):
        nominal_result = Implicit_PPO.update(self, clear_storage=False)
        adaptive_value_loss, adaptive_surrogate_loss, adaptive_kl = self._update_adaptive_policy()
        load_triplet_loss = self._update_load_transition_model()
        self.storage.clear()

        (
            nominal_value_loss,
            ce_loss,
            nominal_surrogate_loss,
            nominal_kl,
            explained_variance,
            ce_loss_terms,
        ) = nominal_result
        nominal_loss = nominal_surrogate_loss + self.value_loss_coef * nominal_value_loss
        adaptive_loss = adaptive_surrogate_loss + self.adaptive_value_loss_coef * adaptive_value_loss
        load_loss = self.load_triplet_loss_coef * load_triplet_loss
        total_loss = nominal_loss + adaptive_loss + ce_loss + load_loss
        self.last_loss_terms = {
            "nominal_loss": float(nominal_loss),
            "adaptive_loss": float(adaptive_loss),
            "ce_loss": float(ce_loss),
            "load_triplet_loss": float(load_triplet_loss),
            "load_loss": float(load_loss),
            "total_loss": float(total_loss),
            "adaptive_value_loss": float(adaptive_value_loss),
            "adaptive_surrogate_loss": float(adaptive_surrogate_loss),
            "adaptive_kl": float(adaptive_kl),
            "adaptive_reward_scale": float(self.adaptive_reward_scale),
        }
        self.last_loss_terms.update(self.last_load_diagnostics)
        self.last_loss_terms.update({f"probe_{key}": value for key, value in self.last_probe_terms.items()})
        return (
            nominal_value_loss,
            ce_loss,
            nominal_surrogate_loss,
            nominal_kl,
            explained_variance,
            ce_loss_terms,
            self.last_loss_terms,
        )

    def act_inference(self, obs, obs_history, commands, adaptive_obs, adaptive_history):
        v_t, z_t, _mu, _logvar, _o_next_recon = self._ce_forward_pass(obs_history)
        v_policy, z_policy = self._prepare_policy_latent(v_t, z_t)
        encoder_output = torch.cat((v_policy, z_policy), dim=-1)
        actor_obs = self._build_context_policy_input(encoder_output, obs, commands, obs_history)
        nominal_action = self.actor_critic.act_inference(actor_obs)
        adaptive_policy_input, _z_load = self._build_load_adaptive_input(adaptive_obs, adaptive_history)
        delta_action = self.adaptive_actor_critic.act_inference(adaptive_policy_input)
        return nominal_action + delta_action
