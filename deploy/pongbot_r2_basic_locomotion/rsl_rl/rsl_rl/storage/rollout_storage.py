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
import numpy as np

class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.next_observations = None
            self.critic_obs = None
            self.encoder_targets = None
            self.observation_history = None
            self.commands = None
            self.adaptive_observations = None
            self.adaptive_critic_observations = None
            self.adaptive_rewards = None
            self.adaptive_history = None
            self.adaptive_values = None
            self.adaptive_actions = None
            self.adaptive_actions_log_prob = None
            self.adaptive_action_mean = None
            self.adaptive_action_sigma = None
            self.payload_mass = None
            self.payload_pos_b = None
            self.terrain_id = None
            self.env_id = None
            self.timestep = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        all_obs_shape,
        encoder_target_shape,
        obs_history_shape,
        commands_shape,
        actions_shape,
        device="cpu",
        adaptive_obs_shape=None,
        adaptive_critic_obs_shape=None,
        adaptive_history_shape=None,
    ):
        self.device = device

        self.obs_shape = obs_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(
            num_transitions_per_env, num_envs, *obs_shape, device=self.device
        )
        self.next_observations = torch.zeros(
            num_transitions_per_env, num_envs, *obs_shape, device=self.device
        )
        if all_obs_shape[0] is not None:
            self.critic_obs = torch.zeros(
                num_transitions_per_env,
                num_envs,
                *all_obs_shape,
                device=self.device
            )
        else:
            self.critic_obs = None
        if encoder_target_shape[0] is not None:
            self.encoder_targets = torch.zeros(
                num_transitions_per_env,
                num_envs,
                *encoder_target_shape,
                device=self.device,
            )
        else:
            self.encoder_targets = None
        self.observation_history = torch.zeros(
            num_transitions_per_env, num_envs, *obs_history_shape, device=self.device
        )
        self.commands = torch.zeros(
            num_transitions_per_env, num_envs, *commands_shape, device=self.device
        )
        self.payload_mass = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.payload_pos_b = torch.zeros(num_transitions_per_env, num_envs, 3, device=self.device)
        self.terrain_id = torch.full(
            (num_transitions_per_env, num_envs, 1), -1.0, device=self.device
        )
        self.env_id = torch.arange(num_envs, device=self.device, dtype=torch.float32).view(1, num_envs, 1).repeat(
            num_transitions_per_env, 1, 1
        )
        self.timestep = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        if adaptive_obs_shape is not None and adaptive_obs_shape[0] is not None:
            if adaptive_critic_obs_shape is None:
                adaptive_critic_obs_shape = adaptive_obs_shape
            self.adaptive_observations = torch.zeros(
                num_transitions_per_env,
                num_envs,
                *adaptive_obs_shape,
                device=self.device,
            )
            self.adaptive_critic_observations = torch.zeros(
                num_transitions_per_env,
                num_envs,
                *adaptive_critic_obs_shape,
                device=self.device,
            )
            self.adaptive_rewards = torch.zeros(
                num_transitions_per_env, num_envs, 1, device=self.device
            )
            if adaptive_history_shape is not None and adaptive_history_shape[0] is not None:
                self.adaptive_history = torch.zeros(
                    num_transitions_per_env,
                    num_envs,
                    *adaptive_history_shape,
                    device=self.device,
                )
            else:
                self.adaptive_history = None
            self.adaptive_values = torch.zeros(
                num_transitions_per_env, num_envs, 1, device=self.device
            )
            self.adaptive_returns = torch.zeros(
                num_transitions_per_env, num_envs, 1, device=self.device
            )
            self.adaptive_advantages = torch.zeros(
                num_transitions_per_env, num_envs, 1, device=self.device
            )
            self.adaptive_actions = torch.zeros(
                num_transitions_per_env, num_envs, *actions_shape, device=self.device
            )
            self.adaptive_actions_log_prob = torch.zeros(
                num_transitions_per_env, num_envs, 1, device=self.device
            )
            self.adaptive_mu = torch.zeros(
                num_transitions_per_env, num_envs, *actions_shape, device=self.device
            )
            self.adaptive_sigma = torch.zeros(
                num_transitions_per_env, num_envs, *actions_shape, device=self.device
            )
        else:
            self.adaptive_observations = None
            self.adaptive_critic_observations = None
            self.adaptive_rewards = None
            self.adaptive_history = None
            self.adaptive_values = None
            self.adaptive_returns = None
            self.adaptive_advantages = None
            self.adaptive_actions = None
            self.adaptive_actions_log_prob = None
            self.adaptive_mu = None
            self.adaptive_sigma = None
        self.rewards = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.actions = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.dones = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        ).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.values = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.returns = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.advantages = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.mu = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.sigma = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        self.next_observations[self.step].copy_(transition.next_observations)
        if self.critic_obs is not None:
            self.critic_obs[self.step].copy_(transition.critic_obs)
        if self.encoder_targets is not None and transition.encoder_targets is not None:
            self.encoder_targets[self.step].copy_(transition.encoder_targets)
        self.observation_history[self.step].copy_(transition.observation_history)
        self.commands[self.step].copy_(transition.commands)
        if transition.payload_mass is not None:
            self.payload_mass[self.step].copy_(transition.payload_mass.view(-1, 1))
        if transition.payload_pos_b is not None:
            self.payload_pos_b[self.step].copy_(transition.payload_pos_b)
        if transition.terrain_id is not None:
            self.terrain_id[self.step].copy_(transition.terrain_id.view(-1, 1).to(self.device))
        if transition.timestep is not None:
            self.timestep[self.step].copy_(transition.timestep.view(-1, 1).to(self.device))
        if self.adaptive_observations is not None and transition.adaptive_observations is not None:
            self.adaptive_observations[self.step].copy_(transition.adaptive_observations)
            self.adaptive_critic_observations[self.step].copy_(transition.adaptive_critic_observations)
            self.adaptive_rewards[self.step].copy_(transition.adaptive_rewards.view(-1, 1))
            if self.adaptive_history is not None and transition.adaptive_history is not None:
                self.adaptive_history[self.step].copy_(transition.adaptive_history)
            self.adaptive_values[self.step].copy_(transition.adaptive_values)
            self.adaptive_actions[self.step].copy_(transition.adaptive_actions)
            self.adaptive_actions_log_prob[self.step].copy_(transition.adaptive_actions_log_prob.view(-1, 1))
            self.adaptive_mu[self.step].copy_(transition.adaptive_action_mean)
            self.adaptive_sigma[self.step].copy_(transition.adaptive_action_sigma)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = (
            hidden_states[0]
            if isinstance(hidden_states[0], tuple)
            else (hidden_states[0],)
        )
        hid_c = (
            hidden_states[1]
            if isinstance(hidden_states[1], tuple)
            else (hidden_states[1],)
        )

        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(
                    self.observations.shape[0], *hid_a[i].shape, device=self.device
                )
                for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(
                    self.observations.shape[0], *hid_c[i].shape, device=self.device
                )
                for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.rewards[step]
                + next_is_not_terminal * gamma * next_values
                - self.values[step]
            )
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )

    def compute_adaptive_returns(self, last_values, gamma, lam):
        if self.adaptive_returns is None:
            return
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.adaptive_values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.adaptive_rewards[step]
                + next_is_not_terminal * gamma * next_values
                - self.adaptive_values[step]
            )
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.adaptive_returns[step] = advantage + self.adaptive_values[step]

        self.adaptive_advantages = self.adaptive_returns - self.adaptive_values
        self.adaptive_advantages = (self.adaptive_advantages - self.adaptive_advantages.mean()) / (
            self.adaptive_advantages.std() + 1e-8
        )

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (
                flat_dones.new_tensor([-1], dtype=torch.int64),
                flat_dones.nonzero(as_tuple=False)[:, 0],
            )
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(
        self,
        num_group,
        num_mini_batches,
        num_epochs=8,
    ):
        group_batch_size = num_group * self.num_transitions_per_env
        group_mini_batch_size = group_batch_size // num_mini_batches
        group_indices = torch.randperm(
            num_mini_batches * group_mini_batch_size,
            requires_grad=False,
            device=self.device,
        )
        group_group_idx = torch.arange(0, num_group)
        group_observations = self.observations[:, group_group_idx, :].flatten(0, 1)

        group_critic_obs = self.critic_obs[:, group_group_idx, :].flatten(0, 1)
        group_obs_history = self.observation_history[:, group_group_idx, :].flatten(0, 1)

        group_commands = self.commands[:, group_group_idx, :].flatten(0, 1)
        group_actions = self.actions[:, group_group_idx, :].flatten(0, 1)
        group_values = self.values[:, group_group_idx, :].flatten(0, 1)
        group_returns = self.returns[:, group_group_idx, :].flatten(0, 1)

        group_old_actions_log_prob = self.actions_log_prob[:, group_group_idx, :].flatten(0, 1)
        group_advantages = self.advantages[:, group_group_idx, :].flatten(0, 1)
        group_old_mu = self.mu[:, group_group_idx, :].flatten(0, 1)
        group_old_sigma = self.sigma[:, group_group_idx, :].flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                group_start = i * group_mini_batch_size
                group_end = (i + 1) * group_mini_batch_size
                group_batch_idx = group_indices[group_start:group_end]

                group_obs_batch = group_observations[group_batch_idx]
                obs_batch = group_obs_batch
                group_critic_obs_batch = group_critic_obs[group_batch_idx]
                critic_obs_batch = group_critic_obs_batch

                group_obs_history_batch = group_obs_history[group_batch_idx]
                obs_history_batch = group_obs_history_batch

                group_commands_batch = group_commands[group_batch_idx]
                group_actions_batch = group_actions[group_batch_idx]
                actions_batch = group_actions_batch

                group_target_values_batch = group_values[group_batch_idx]
                target_values_batch = group_target_values_batch

                group_returns_batch = group_returns[group_batch_idx]
                returns_batch = group_returns_batch

                group_old_actions_log_prob_batch = group_old_actions_log_prob[group_batch_idx]
                old_actions_log_prob_batch = group_old_actions_log_prob_batch

                group_advantages_batch = group_advantages[group_batch_idx]
                advantages_batch = group_advantages_batch

                group_old_mu_batch = group_old_mu[group_batch_idx]
                old_mu_batch = group_old_mu_batch

                group_old_sigma_batch = group_old_sigma[group_batch_idx]
                old_sigma_batch = group_old_sigma_batch

                yield obs_batch, critic_obs_batch, obs_history_batch, group_obs_history_batch, group_commands_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,

    def adaptive_mini_batch_generator(self, num_group, num_mini_batches, num_epochs=8):
        if self.adaptive_observations is None:
            return
        group_batch_size = num_group * self.num_transitions_per_env
        group_mini_batch_size = group_batch_size // num_mini_batches
        group_indices = torch.randperm(
            num_mini_batches * group_mini_batch_size,
            requires_grad=False,
            device=self.device,
        )
        group_group_idx = torch.arange(0, num_group)
        adaptive_observations = self.adaptive_observations[:, group_group_idx, :].flatten(0, 1)
        adaptive_critic_observations = self.adaptive_critic_observations[:, group_group_idx, :].flatten(0, 1)
        adaptive_actions = self.adaptive_actions[:, group_group_idx, :].flatten(0, 1)
        adaptive_values = self.adaptive_values[:, group_group_idx, :].flatten(0, 1)
        adaptive_returns = self.adaptive_returns[:, group_group_idx, :].flatten(0, 1)
        adaptive_old_actions_log_prob = self.adaptive_actions_log_prob[:, group_group_idx, :].flatten(0, 1)
        adaptive_advantages = self.adaptive_advantages[:, group_group_idx, :].flatten(0, 1)
        adaptive_old_mu = self.adaptive_mu[:, group_group_idx, :].flatten(0, 1)
        adaptive_old_sigma = self.adaptive_sigma[:, group_group_idx, :].flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * group_mini_batch_size
                end = (i + 1) * group_mini_batch_size
                batch_idx = group_indices[start:end]
                yield (
                    adaptive_observations[batch_idx],
                    adaptive_critic_observations[batch_idx],
                    adaptive_actions[batch_idx],
                    adaptive_values[batch_idx],
                    adaptive_advantages[batch_idx],
                    adaptive_returns[batch_idx],
                    adaptive_old_actions_log_prob[batch_idx],
                    adaptive_old_mu[batch_idx],
                    adaptive_old_sigma[batch_idx],
                )

    def adaptive_load_mini_batch_generator(self, num_group, num_mini_batches, num_epochs=8):
        if self.adaptive_observations is None or self.adaptive_history is None:
            return
        group_batch_size = num_group * self.num_transitions_per_env
        group_mini_batch_size = group_batch_size // num_mini_batches
        group_indices = torch.randperm(
            num_mini_batches * group_mini_batch_size,
            requires_grad=False,
            device=self.device,
        )
        group_group_idx = torch.arange(0, num_group, device=self.device)
        adaptive_observations = self.adaptive_observations[:, group_group_idx, :].flatten(0, 1)
        adaptive_history = self.adaptive_history[:, group_group_idx].flatten(0, 1)
        adaptive_critic_observations = self.adaptive_critic_observations[:, group_group_idx, :].flatten(0, 1)
        adaptive_actions = self.adaptive_actions[:, group_group_idx, :].flatten(0, 1)
        adaptive_values = self.adaptive_values[:, group_group_idx, :].flatten(0, 1)
        adaptive_returns = self.adaptive_returns[:, group_group_idx, :].flatten(0, 1)
        adaptive_old_actions_log_prob = self.adaptive_actions_log_prob[:, group_group_idx, :].flatten(0, 1)
        adaptive_advantages = self.adaptive_advantages[:, group_group_idx, :].flatten(0, 1)
        adaptive_old_mu = self.adaptive_mu[:, group_group_idx, :].flatten(0, 1)
        adaptive_old_sigma = self.adaptive_sigma[:, group_group_idx, :].flatten(0, 1)
        payload_mass = self.payload_mass[:, group_group_idx, :].flatten(0, 1)
        payload_pos_b = self.payload_pos_b[:, group_group_idx, :].flatten(0, 1)

        for _epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * group_mini_batch_size
                end = (i + 1) * group_mini_batch_size
                batch_idx = group_indices[start:end]
                yield (
                    adaptive_observations[batch_idx],
                    adaptive_history[batch_idx],
                    adaptive_critic_observations[batch_idx],
                    adaptive_actions[batch_idx],
                    adaptive_values[batch_idx],
                    adaptive_advantages[batch_idx],
                    adaptive_returns[batch_idx],
                    adaptive_old_actions_log_prob[batch_idx],
                    adaptive_old_mu[batch_idx],
                    adaptive_old_sigma[batch_idx],
                    payload_mass[batch_idx],
                    payload_pos_b[batch_idx],
                )

    def load_transition_mini_batch_generator(self, num_group, num_mini_batches, num_epochs=8):
        if self.adaptive_history is None:
            return
        if self.num_transitions_per_env < 2:
            return

        group_group_idx = torch.arange(0, num_group, device=self.device)
        valid_mask = ~self.dones[:-1, group_group_idx, 0].bool()
        valid_time, valid_env = torch.nonzero(valid_mask, as_tuple=True)
        num_valid = valid_time.numel()
        if num_valid == 0:
            return

        mini_batch_size = max(1, num_valid // num_mini_batches)
        for _epoch in range(num_epochs):
            indices = torch.randperm(num_valid, requires_grad=False, device=self.device)
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = num_valid if i == num_mini_batches - 1 else min((i + 1) * mini_batch_size, num_valid)
                if start >= end:
                    continue
                batch_select = indices[start:end]
                t_idx = valid_time[batch_select]
                env_idx = valid_env[batch_select]
                yield (
                    self.adaptive_observations[t_idx, env_idx],
                    self.adaptive_history[t_idx, env_idx],
                    self.adaptive_history[t_idx + 1, env_idx],
                    self.actions[t_idx, env_idx],
                    self.adaptive_actions[t_idx, env_idx],
                    self.commands[t_idx, env_idx],
                    self.payload_mass[t_idx, env_idx],
                    self.payload_mass[t_idx + 1, env_idx],
                    self.payload_pos_b[t_idx, env_idx],
                    self.payload_pos_b[t_idx + 1, env_idx],
                    self.terrain_id[t_idx, env_idx],
                    self.env_id[t_idx, env_idx],
                    self.timestep[t_idx, env_idx],
                )

    def encoder_mini_batch_generator(self, num_mini_batches, num_epochs=8, include_terrain_id=False):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )

        observations = self.observations.flatten(0, 1)
        next_observations = self.next_observations.flatten(0, 1)
        if self.critic_obs is not None:
            critic_obs = self.critic_obs.flatten(0, 1)
        else:
            critic_obs = observations
        if self.encoder_targets is not None:
            encoder_targets = self.encoder_targets.flatten(0, 1)
        else:
            encoder_targets = None
        obs_history = self.observation_history.flatten(0, 1)
        terrain_id = self.terrain_id.flatten(0, 1) if include_terrain_id else None

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                next_obs_batch = next_observations[batch_idx]
                critic_obs_batch = critic_obs[batch_idx]
                encoder_target_batch = encoder_targets[batch_idx] if encoder_targets is not None else None
                obs_history_batch = obs_history[batch_idx]
                if include_terrain_id:
                    terrain_id_batch = terrain_id[batch_idx]
                    yield next_obs_batch, critic_obs_batch, encoder_target_batch, obs_history_batch, terrain_id_batch
                    continue
                yield next_obs_batch, critic_obs_batch, encoder_target_batch, obs_history_batch
