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

from rsl_rl.modules import ActorCritic, MLP_Encoder, IMU_Encoder
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

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        obs_history_shape,
        commands_shape,
        action_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
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
        self.transition.actions = self.actor_critic.act(
            torch.cat((encoder_out, obs, commands, obs_history), dim=-1)
        ).detach()

        #obs_history
        # self.transition.actions = self.actor_critic.act(
        #     torch.cat((encoder_out, obs, commands), dim=-1)
        # ).detach()


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
        self.transition.observation_history = obs_history
        self.transition.commands = commands
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_obs=None):
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
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        num_updates = 0
        mean_value_loss = 0
        mean_surrogate_loss = 0
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
            self.actor_critic.act(
                torch.cat(
                    (encoder_out_batch, obs_batch, commands_batch, obs_history_batch),
                    dim=-1,
                )
            )
            #obs_history
            # self.actor_critic.act(
            #     torch.cat(
            #         (encoder_out_batch, obs_batch, commands_batch),
            #         dim=-1,
            #     )
            # )

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
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch_mean
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
                obs_history_batch,
            ) in generator:
                if hasattr(self.encoder, "get_encoder_out"):
                    self.encoder.encode(obs_history_batch)
                    encode_batch = self.encoder.get_encoder_out()

                if hasattr(self.encoder, "get_encoder_out"):
                    target_dim = self.encoder.num_output_dim
                    extra_loss = (
                        (encode_batch[:, :target_dim] - critic_obs_batch[:, :target_dim]).pow(2).mean()
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
        mean_kl /= num_updates
        mean_explained_variance = _compute_explained_variance_from_sums(
            num_value_samples, returns_sum, returns_sq_sum, residual_sum, residual_sq_sum
        )
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
        # del obs_history
        # return torch.cat((encoder_out, obs, commands), dim=-1)
        #obs_history 
        return torch.cat((encoder_out, obs, commands, obs_history), dim=-1)

    @staticmethod
    def _split_imu_targets(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tensor[:, 0:3], tensor[:, 3:6], tensor[:, 6:9]

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        obs_history_shape,
        commands_shape,
        action_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
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
        self.transition.observation_history = obs_history
        self.transition.commands = commands
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_obs=None):
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
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        num_updates = 0
        mean_value_loss = 0
        mean_surrogate_loss = 0
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
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch_mean
            )
            self._check_finite("update.surrogate_loss", surrogate_loss, batch_context)
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
                obs_history_batch,
            ) in enumerate(generator):
                batch_context = {"encoder_mini_batch": mini_batch_idx}
                self._check_finite("encoder.next_obs_batch", next_obs_batch, batch_context)
                self._check_finite("encoder.critic_obs_batch", critic_obs_batch, batch_context)
                self._check_finite("encoder.obs_history_batch", obs_history_batch, batch_context)
                if hasattr(self.encoder, "get_encoder_out"):
                    self.encoder.encode(obs_history_batch)
                    encode_batch = self.encoder.get_encoder_out()
                    self._check_finite("encoder.encode_batch", encode_batch, batch_context)

                if hasattr(self.encoder, "get_encoder_out"):
                    target_dim = self.encoder.num_output_dim
                    pred_batch = encode_batch[:, :target_dim]
                    target_batch = critic_obs_batch[:, :target_dim]
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
        mean_kl /= num_updates
        mean_explained_variance = _compute_explained_variance_from_sums(
            num_value_samples, returns_sum, returns_sq_sum, residual_sum, residual_sq_sum
        )
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
