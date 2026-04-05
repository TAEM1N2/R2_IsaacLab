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
from pathlib import Path

from rsl_rl.modules import ActorCritic, MLP_Encoder, IMU_Encoder
from rsl_rl.storage import RolloutStorage


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
            torch.cat((encoder_out, obs, commands), dim=-1)
        ).detach()

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
                    (encoder_out_batch, obs_batch, commands_batch),
                    dim=-1,
                )
            )

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
            mean_extra_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_kl /= num_updates
        self.storage.clear()

        return (mean_value_loss, mean_extra_loss, mean_surrogate_loss, mean_kl)

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
        self.learning_iteration = 0
        self.nan_debug_dir = Path.cwd() / "nan_debug"
        self.nan_debug_context = {}

    def set_learning_iteration(self, iteration: int):
        self.learning_iteration = iteration

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

    def _check_module_params(self, module: nn.Module, module_name: str, extra: dict | None = None) -> None:
        for name, param in module.named_parameters():
            if not torch.isfinite(param).all():
                self._dump_invalid_tensor(f"{module_name}.{name}", param, extra)

    def _build_actor_obs(self, obs, encoder_out, commands):
        actor_obs = obs
        if encoder_out.shape[1] >= 9 and obs.shape[1] >= 6 and self.learning_iteration >= self.encoder_warmup_iters:
            actor_obs = obs.clone()
            actor_obs[:, 0:6] = encoder_out[:, 3:9]
        return torch.cat((encoder_out, actor_obs, commands), dim=-1)

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
        actor_obs = self._build_actor_obs(obs, encoder_out, commands)
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
            actor_obs_batch = self._build_actor_obs(obs_batch, encoder_out_batch, commands_batch)
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
                    extra_loss = (
                        (encode_batch[:, :target_dim] - critic_obs_batch[:, :target_dim]).pow(2).mean()
                    )
                else:
                    extra_loss = torch.zeros_like(value_loss)
                self._check_finite("encoder.extra_loss", extra_loss, batch_context)

                self.extra_optimizer.zero_grad()
                extra_loss.backward()
                self.extra_optimizer.step()
                self._check_module_params(self.encoder, "encoder_post_step", batch_context)

                num_updates_extra += 1
                mean_extra_loss += extra_loss.item()

        mean_value_loss /= num_updates
        if num_updates_extra > 0:
            mean_extra_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_kl /= num_updates
        self.storage.clear()

        return (mean_value_loss, mean_extra_loss, mean_surrogate_loss, mean_kl)
