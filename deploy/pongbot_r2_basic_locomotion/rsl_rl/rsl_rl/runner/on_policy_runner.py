# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""On-policy runner with PongBot context-estimator and diagnostic extensions.

@version 0.0.2
@update 2026-07-12: Fix frontier metric logging and support dilated observation histories.
"""
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

import math
import os
import statistics
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from isaaclab.managers import SceneEntityCfg

from rsl_rl.algorithm import PPO, IMU_PPO, Implicit_PPO, Phase2_Adaptive_PPO, Phase2_LoadAdaptive_PPO
from rsl_rl.modules import MLP_Encoder, IMU_Encoder, ContextEstimatorNet, ActorCritic
from rsl_rl.env import VecEnv


class OnPolicyRunner:
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        print(f"encoder cfg: {train_cfg.keys()}")
        self.ecd_cfg = train_cfg["encoder"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.adaptive_policy_cfg = train_cfg.get("adaptive_policy", None)
        self.adaptive_shared_reward_scales = self.alg_cfg.pop("adaptive_shared_reward_scales", None)
        self.adaptive_extra_reward_scales = self.alg_cfg.pop(
            "adaptive_extra_reward_scales",
            {"body_height": 1.0, "grf": 1.0, "stability": 1.0},
        )
        self.reward_weight_curriculum_terms = tuple(self.alg_cfg.pop("reward_weight_curriculum_terms", ()))
        self.reward_weight_curriculum_annealing_rate = float(
            self.alg_cfg.pop("reward_weight_curriculum_annealing_rate", 0.998)
        )
        self.reward_weight_curriculum_log_interval = int(
            self.alg_cfg.pop("reward_weight_curriculum_log_interval", 1)
        )
        self.reward_weight_curriculum_base_weights = {}
        self.reward_weight_curriculum_missing_terms = set()
        self.device = device
        self.env = env
        obs, extras = self.env.get_observations()
        self.num_obs = obs.shape[1]
        self.obs_history_len = self.alg_cfg.pop("obs_history_len")
        self.obs_history_offsets = tuple(int(offset) for offset in self.alg_cfg.pop("obs_history_offsets", ()))
        self.policy_include_history = self.alg_cfg.get("policy_include_history", True)
        self.encoder_target_source = self.ecd_cfg.pop("target_source", "critic_prefix")
        self.encoder_target_group = self.ecd_cfg.pop("target_observation_group", None)
        encoder_class_name = self.ecd_cfg.pop("class_name", "MLP_Encoder")
        self.uses_context_estimator = encoder_class_name == "ContextEstimatorNet"
        obs_history_raw = extras["observations"].get("obsHistory")
        assert obs_history_raw is not None, "obsHistory not found in observations"
        obs_history = self._prepare_obs_history(obs_history_raw)
        if self.uses_context_estimator:
            assert obs_history.dim() == 3, f"CE Net requires obsHistory [batch, H, obs_dim], got {tuple(obs_history.shape)}"
            self.obs_history_shape = tuple(obs_history.shape[1:])
            self.obs_history_dim = int(np.prod(self.obs_history_shape))
            configured_history_len = self.ecd_cfg.get("history_len")
            if configured_history_len is None:
                self.ecd_cfg["history_len"] = self.obs_history_shape[0]
            self.ecd_cfg["obs_dim"] = self.obs_history_shape[1]
            if self.obs_history_shape[0] != self.ecd_cfg["history_len"]:
                raise ValueError(
                    f"CE Net history length mismatch: obsHistory={self.obs_history_shape[0]} "
                    f"cfg={self.ecd_cfg['history_len']}"
                )
        else:
            self.obs_history_shape = (obs_history.reshape(obs_history.shape[0], -1).shape[1],)
            self.obs_history_dim = self.obs_history_shape[0]
        assert "commands" in extras["observations"], f"Commands not found in observations"
        self.num_commands = extras["observations"]["commands"].shape[1]
        assert "critic" in extras["observations"], f"Critic observations not found in observations"
        num_critic_obs = extras["observations"]["critic"].shape[1] + self.num_commands
        self.alg_class_name = self.alg_cfg.get("class_name")
        self.uses_phase2_load_adaptive = self.alg_class_name == "Phase2_LoadAdaptive_PPO"
        self.uses_phase2_adaptive = self.alg_class_name in ("Phase2_Adaptive_PPO", "Phase2_LoadAdaptive_PPO")
        self.adaptive_obs_dim = None
        self.adaptive_critic_obs_dim = None
        self.adaptive_history_shape = None
        if self.uses_phase2_adaptive:
            adaptive_obs = extras["observations"].get("adaptive")
            assert adaptive_obs is not None, "Phase 2 requires 'adaptive' observation group"
            adaptive_critic_obs = extras["observations"].get("adaptiveCritic", adaptive_obs)
            self.adaptive_obs_dim = adaptive_obs.shape[1]
            self.adaptive_critic_obs_dim = adaptive_critic_obs.shape[1]
            if self.uses_phase2_load_adaptive:
                adaptive_history = extras["observations"].get("adaptiveHistory")
                assert adaptive_history is not None, "Phase 2 load-adaptive requires 'adaptiveHistory' observation group"
                self.adaptive_history_shape = tuple(adaptive_history.shape[1:])
        self.phase2_log_names = ("payload_kg", "base_height_error", "foot_contact_force")
        privileged_input_size = num_critic_obs
        if not self.uses_context_estimator:
            self.ecd_cfg["num_input_dim"] = self.obs_history_dim

        encoder_class = {
            "MLP_Encoder": MLP_Encoder,
            "IMU_Encoder": IMU_Encoder,
            "ContextEstimatorNet": ContextEstimatorNet,
        }[encoder_class_name]
        encoder = encoder_class(
            **self.ecd_cfg,
        ).to(self.device)
        encoder_target = self._resolve_encoder_target(extras["observations"])
        if self.uses_context_estimator:
            self.encoder_target_dim = 3 + int(self.alg_cfg.get("terrain_property_target_dim", 0))
        else:
            self.encoder_target_dim = encoder.num_output_dim
        if encoder_target is not None and encoder_target.shape[1] != self.encoder_target_dim:
            raise ValueError(
                f"encoder target dim mismatch: obs={encoder_target.shape[1]} encoder={self.encoder_target_dim}"
            )

        actor_critic_class = ActorCritic
        #obs_history
        # actor_critic: ActorCritic = actor_critic_class(
        #     self.num_obs + encoder.num_output_dim + self.num_commands,
        #     num_critic_obs,
        #     self.env.num_actions,
        #     **self.policy_cfg,
        # ).to(self.device)

        if self.uses_context_estimator:
            num_actor_obs = self.num_obs + encoder.num_output_dim + self.num_commands
            if self.policy_include_history:
                num_actor_obs += self.obs_history_dim
        elif self.policy_include_history:
            num_actor_obs = self.num_obs + encoder.num_output_dim + self.num_commands + self.obs_history_dim
        else:
            num_actor_obs = self.num_obs + encoder.num_output_dim + self.num_commands

        actor_critic: ActorCritic = actor_critic_class(
            num_actor_obs,
            num_critic_obs,
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        # actor_critic: ActorCritic = actor_critic_class(
        #     self.num_obs
        #     + self.num_commands
        #     + self.obs_history_dim,
        #     num_critic_obs,
        #     self.env.num_actions,
        #     **self.policy_cfg,
        # ).to(self.device)

        alg_class = {
            "PPO": PPO,
            "IMU_PPO": IMU_PPO,
            "Implicit_PPO": Implicit_PPO,
            "Phase2_Adaptive_PPO": Phase2_Adaptive_PPO,
            "Phase2_LoadAdaptive_PPO": Phase2_LoadAdaptive_PPO,
        }[self.alg_cfg.pop("class_name")]
        if self.uses_phase2_adaptive:
            load_kwargs = {"load_history_shape": self.adaptive_history_shape} if self.uses_phase2_load_adaptive else {}
            self.alg = alg_class(
                self.env.num_envs,
                encoder,
                actor_critic,
                adaptive_obs_dim=self.adaptive_obs_dim,
                adaptive_critic_obs_dim=self.adaptive_critic_obs_dim,
                adaptive_policy_cfg=self.adaptive_policy_cfg,
                device=self.device,
                **load_kwargs,
                **self.alg_cfg,
            )
        else:
            self.alg = alg_class(
                self.env.num_envs,
                encoder,
                actor_critic,
                device = self.device,
                **self.alg_cfg,
            )

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        storage_args = (
            self.env.num_envs,
            self.num_steps_per_env,
            [self.num_obs],
            [num_critic_obs],
            [self.encoder_target_dim] if encoder_target is not None else [None],
            list(self.obs_history_shape),
            [self.num_commands],
            [self.env.num_actions],
        )
        if self.uses_phase2_adaptive:
            if self.uses_phase2_load_adaptive:
                self.alg.init_storage(
                    *storage_args,
                    [self.adaptive_obs_dim],
                    [self.adaptive_critic_obs_dim],
                    list(self.adaptive_history_shape),
                )
            else:
                self.alg.init_storage(*storage_args, [self.adaptive_obs_dim], [self.adaptive_critic_obs_dim])
        else:
            self.alg.init_storage(*storage_args)

        self.obs_mean = torch.tensor(
            0, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.obs_std = torch.tensor(
            1, dtype=torch.float, device=self.device, requires_grad=False
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.nan_debug_enabled = True
        self.nan_debug_dir = Path(self.log_dir or os.getcwd()) / "nan_debug"
        self.nan_debug_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.alg, "set_nan_debug_dir"):
            self.alg.set_nan_debug_dir(str(self.nan_debug_dir))

        # _, _ = self.env.reset()
        _ = self.env.reset()
        self.action_group_indices = self._resolve_action_group_indices()
        self._print_actuator_gains_once()
        self._print_spawn_debug_once()

    @staticmethod
    def _flatten_obs_history(obs_history: torch.Tensor) -> torch.Tensor:
        return obs_history.flatten(start_dim=1) if obs_history.dim() > 2 else obs_history

    def _prepare_obs_history(self, obs_history: torch.Tensor) -> torch.Tensor:
        if self.uses_context_estimator:
            if obs_history.dim() != 3:
                raise ValueError(f"CE Net requires obsHistory [batch, H, obs_dim], got {tuple(obs_history.shape)}")
            if self.obs_history_offsets:
                raw_history_len = obs_history.shape[1]
                max_offset = max(self.obs_history_offsets)
                if min(self.obs_history_offsets) < 0 or max_offset >= raw_history_len:
                    raise ValueError(
                        f"obs_history_offsets={self.obs_history_offsets} are invalid for raw history length "
                        f"{raw_history_len}"
                    )
                indices = torch.as_tensor(
                    [raw_history_len - 1 - offset for offset in self.obs_history_offsets],
                    device=obs_history.device,
                    dtype=torch.long,
                )
                obs_history = torch.index_select(obs_history, dim=1, index=indices)
            if obs_history.shape[1] != self.obs_history_len:
                raise ValueError(
                    f"Prepared obsHistory length mismatch: got {obs_history.shape[1]}, "
                    f"expected {self.obs_history_len}"
                )
            return obs_history
        return self._flatten_obs_history(obs_history)

    def _resolve_encoder_target(self, observations_dict: dict) -> torch.Tensor | None:
        if self.encoder_target_source == "critic_prefix":
            return None
        if self.encoder_target_source != "observation_group":
            raise ValueError(f"Unsupported encoder target source: {self.encoder_target_source}")
        if not self.encoder_target_group:
            raise ValueError("target_observation_group must be set when using observation_group supervision")
        encoder_target = observations_dict.get(self.encoder_target_group)
        if encoder_target is None:
            raise ValueError(f"Observation group '{self.encoder_target_group}' not found for encoder supervision")
        return encoder_target

    def _compute_shared_adaptive_reward(self) -> torch.Tensor:
        base_env = getattr(self.env, "unwrapped", None)
        if base_env is None or not hasattr(base_env, "reward_manager"):
            raise RuntimeError("Phase 2 adaptive reward requires env.reward_manager")

        reward_manager = base_env.reward_manager
        step_reward = reward_manager._step_reward
        term_names = reward_manager.active_terms
        if self.adaptive_shared_reward_scales is None:
            return step_reward.sum(dim=1)

        reward = torch.zeros(step_reward.shape[0], device=step_reward.device)
        term_index = {name: idx for idx, name in enumerate(term_names)}
        for name, scale in self.adaptive_shared_reward_scales.items():
            if float(scale) == 0.0:
                continue
            if name not in term_index:
                raise KeyError(f"Adaptive shared reward term '{name}' not found in RewardManager")
            reward = reward + float(scale) * step_reward[:, term_index[name]]
        return reward

    def _compute_phase2_extra_reward(self) -> torch.Tensor:
        from pongbot_r2.tasks.locomotion import mdp

        base_env = getattr(self.env, "unwrapped", None)
        if base_env is None:
            raise RuntimeError("Phase 2 extra reward requires unwrapped env")

        reward = torch.zeros(base_env.num_envs, device=base_env.device)
        scales = self.adaptive_extra_reward_scales or {}
        if scales.get("body_height", 0.0) != 0.0:
            reward = reward + float(scales["body_height"]) * mdp.phase2_body_height_tracking_reward(
                base_env, target_height=0.55
            )
        if scales.get("grf", 0.0) != 0.0:
            reward = reward + float(scales["grf"]) * mdp.phase2_grf_reward(
                base_env,
                target_height=0.55,
                asset_cfg=SceneEntityCfg("robot", body_names=".*TIP"),
                sensor_cfg=SceneEntityCfg("contact_forces", body_names=".*TIP"),
            )
        if scales.get("stability", 0.0) != 0.0:
            reward = reward + float(scales["stability"]) * mdp.phase2_stability_reward(base_env)
        return reward

    def _collect_terrain_metadata(self, timestep: int) -> dict[str, torch.Tensor]:
        base_env = getattr(self.env, "unwrapped", None)
        if base_env is None:
            return {}
        device = self.device
        num_envs = self.env.num_envs
        terrain_id = torch.full((num_envs,), -1.0, device=device)
        terrain = getattr(getattr(base_env, "scene", None), "terrain", None)
        terrain_types = getattr(terrain, "terrain_types", None)
        if terrain_types is not None:
            terrain_values = terrain_types.to(device=device, dtype=torch.float32).view(-1)
            if terrain_values.numel() >= num_envs:
                terrain_id = terrain_values[:num_envs]
        return {
            "terrain_id": terrain_id,
            "timestep": torch.full((num_envs,), float(timestep), device=device),
        }

    def _resolve_terrain_log_names(self) -> tuple[str, ...]:
        base_env = getattr(self.env, "unwrapped", None)
        cfg = getattr(base_env, "cfg", None)
        scene_cfg = getattr(cfg, "scene", None)
        terrain_cfg = getattr(scene_cfg, "terrain", None)
        terrain_generator = getattr(terrain_cfg, "terrain_generator", None)
        sub_terrains = getattr(terrain_generator, "sub_terrains", None)
        if not sub_terrains:
            return ()
        try:
            terrain_names = tuple(str(name) for name in sub_terrains.keys())
        except AttributeError:
            terrain_names = tuple(str(name) for name in sub_terrains)
        if not bool(getattr(terrain_generator, "curriculum", False)):
            return terrain_names

        num_cols = int(getattr(terrain_generator, "num_cols", len(terrain_names)))
        if num_cols <= 0:
            return terrain_names
        try:
            sub_cfgs = list(sub_terrains.values())
        except AttributeError:
            sub_cfgs = list(sub_terrains)
        proportions = [max(0.0, float(getattr(sub_cfg, "proportion", 1.0))) for sub_cfg in sub_cfgs]
        total_proportion = sum(proportions)
        if len(proportions) != len(terrain_names) or total_proportion <= 0.0:
            return terrain_names

        cumulative = []
        running = 0.0
        for proportion in proportions:
            running += proportion / total_proportion
            cumulative.append(running)

        column_names = []
        for column_id in range(num_cols):
            threshold = column_id / num_cols + 0.001
            terrain_index = len(cumulative) - 1
            for index, limit in enumerate(cumulative):
                if threshold < limit:
                    terrain_index = index
                    break
            column_names.append(terrain_names[terrain_index])
        return tuple(column_names)

    @staticmethod
    def _sanitize_terrain_log_label(name: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in name)

    def _terrain_log_label(self, terrain_id: int) -> str:
        terrain_names = getattr(self, "terrain_log_names", None)
        if terrain_names is None:
            terrain_names = self._resolve_terrain_log_names()
            self.terrain_log_names = terrain_names
        if 0 <= terrain_id < len(terrain_names):
            return self._sanitize_terrain_log_label(terrain_names[terrain_id])
        return f"unknown_{terrain_id}"

    def _env_vector(self, values, dtype: torch.dtype, default: float | int) -> torch.Tensor:
        result = torch.full((self.env.num_envs,), default, device=self.device, dtype=dtype)
        if values is None:
            return result
        values = torch.as_tensor(values, device=self.device).view(-1)
        count = min(values.numel(), self.env.num_envs)
        if count > 0:
            result[:count] = values[:count].to(dtype=dtype)
        return result

    def _select_done_values(self, values, done_env_ids: torch.Tensor, default: float = 0.0) -> torch.Tensor:
        result = torch.full((done_env_ids.numel(),), default, device=self.device, dtype=torch.float32)
        if values is None or done_env_ids.numel() == 0:
            return result
        values = torch.as_tensor(values, device=self.device).view(-1)
        if values.numel() == self.env.num_envs:
            return values[done_env_ids].to(dtype=torch.float32)
        if values.numel() == done_env_ids.numel():
            return values.to(dtype=torch.float32)
        if values.numel() == 1:
            result.fill_(float(values.item()))
            return result
        count = min(values.numel(), done_env_ids.numel())
        if count > 0:
            result[:count] = values[:count].to(dtype=torch.float32)
        return result

    def _collect_terrain_episode_snapshot(self) -> dict[str, torch.Tensor]:
        base_env = getattr(self.env, "unwrapped", None)
        terrain = getattr(getattr(base_env, "scene", None), "terrain", None)
        terrain_type = self._env_vector(getattr(terrain, "terrain_types", None), torch.long, -1)
        terrain_level = self._env_vector(getattr(terrain, "terrain_levels", None), torch.float32, float("nan"))

        error_vel_xy = None
        command_manager = getattr(base_env, "command_manager", None)
        if command_manager is not None:
            try:
                base_velocity_command = command_manager.get_term("base_velocity")
            except (AttributeError, KeyError):
                base_velocity_command = None
            if base_velocity_command is not None:
                error_vel_xy = getattr(base_velocity_command, "metrics", {}).get("error_vel_xy")

        return {
            "terrain_type": terrain_type,
            "terrain_level": terrain_level,
            "error_vel_xy": self._env_vector(error_vel_xy, torch.float32, float("nan")),
        }

    def _update_terrain_episode_stats(
        self,
        terrain_episode_stats: dict[str, dict[str, float]],
        snapshot: dict[str, torch.Tensor] | None,
        done_env_ids: torch.Tensor,
        infos: dict,
    ) -> None:
        if snapshot is None or done_env_ids.numel() == 0:
            return

        done_env_ids = done_env_ids.to(device=self.device, dtype=torch.long).view(-1)
        terrain_types = snapshot["terrain_type"][done_env_ids]
        terrain_levels = snapshot["terrain_level"][done_env_ids]
        error_vel_xy = snapshot["error_vel_xy"][done_env_ids]

        base_env = getattr(self.env, "unwrapped", None)
        termination_manager = getattr(base_env, "termination_manager", None)
        base_contact_values = None
        if termination_manager is not None:
            try:
                base_contact_values = termination_manager.get_term("base_contact")
            except KeyError:
                base_contact_values = None
        if base_contact_values is None:
            base_contact_values = getattr(base_env, "reset_terminated", None)

        timeout_values = infos.get("time_outs") if isinstance(infos, dict) else None
        if timeout_values is None:
            timeout_values = getattr(base_env, "reset_time_outs", None)

        base_contact = self._select_done_values(base_contact_values, done_env_ids)
        time_out = self._select_done_values(timeout_values, done_env_ids)

        for terrain_id_tensor in torch.unique(terrain_types):
            terrain_id = int(terrain_id_tensor.item())
            mask = terrain_types == terrain_id
            count = int(mask.sum().item())
            if count <= 0:
                continue

            entry = terrain_episode_stats.setdefault(
                self._terrain_log_label(terrain_id),
                {
                    "count": 0.0,
                    "base_contact": 0.0,
                    "time_out": 0.0,
                    "error_vel_xy_sum": 0.0,
                    "error_vel_xy_count": 0.0,
                    "terrain_level_sum": 0.0,
                    "terrain_level_count": 0.0,
                    "terrain_level_max": float("-inf"),
                    "terrain_level_ge_10": 0.0,
                    "terrain_level_ge_14": 0.0,
                    "terrain_level_ge_20": 0.0,
                    "terrain_level_ge_24": 0.0,
                },
            )
            entry["count"] += float(count)
            entry["base_contact"] += float(base_contact[mask].sum().item())
            entry["time_out"] += float(time_out[mask].sum().item())

            terrain_level_values = terrain_levels[mask]
            terrain_level_values = terrain_level_values[torch.isfinite(terrain_level_values)]
            if terrain_level_values.numel() > 0:
                entry["terrain_level_sum"] += float(terrain_level_values.sum().item())
                entry["terrain_level_count"] += float(terrain_level_values.numel())
                entry["terrain_level_max"] = max(
                    entry["terrain_level_max"], float(terrain_level_values.max().item())
                )
                entry["terrain_level_ge_10"] += float((terrain_level_values >= 10.0).sum().item())
                entry["terrain_level_ge_14"] += float((terrain_level_values >= 14.0).sum().item())
                entry["terrain_level_ge_20"] += float((terrain_level_values >= 20.0).sum().item())
                entry["terrain_level_ge_24"] += float((terrain_level_values >= 24.0).sum().item())

            error_values = error_vel_xy[mask]
            error_values = error_values[torch.isfinite(error_values)]
            if error_values.numel() > 0:
                entry["error_vel_xy_sum"] += float(error_values.sum().item())
                entry["error_vel_xy_count"] += float(error_values.numel())

    def _compute_adaptive_reward(self) -> torch.Tensor:
        return self._compute_shared_adaptive_reward() + self._compute_phase2_extra_reward()

    def _collect_phase2_metadata(self, timestep: int) -> dict[str, torch.Tensor]:
        base_env = getattr(self.env, "unwrapped", None)
        if base_env is None:
            return {}
        device = self.device
        num_envs = self.env.num_envs
        payload_mass = getattr(base_env, "_payload_mass", None)
        if payload_mass is None:
            payload_mass = torch.zeros(num_envs, device=device)
        else:
            payload_mass = payload_mass.to(device)
        payload_pos_b = getattr(base_env, "_payload_pos_b", None)
        if payload_pos_b is None:
            payload_pos_b = torch.zeros(num_envs, 3, device=device)
        else:
            payload_pos_b = payload_pos_b.to(device)
        terrain_metadata = self._collect_terrain_metadata(timestep)
        return {
            "payload_mass": payload_mass,
            "payload_pos_b": payload_pos_b,
            "terrain_id": terrain_metadata.get("terrain_id"),
            "timestep": terrain_metadata.get("timestep"),
        }

    def _resolve_phase2_log_obs(self, observations_dict: dict) -> torch.Tensor | None:
        return observations_dict.get("phase2Log")

    def _resolve_action_group_indices(self) -> dict[str, torch.Tensor]:
        groups: dict[str, list[int]] = {"hr": [], "hp": [], "kn": []}
        base_env = getattr(self.env, "unwrapped", None)
        try:
            asset = base_env.scene["robot"]
            action_term = base_env.action_manager.get_term("joint_pos")
            action_joint_ids = action_term._joint_ids
        except (AttributeError, KeyError):
            action_joint_ids = None

        if action_joint_ids is not None:
            if isinstance(action_joint_ids, slice):
                action_joint_ids = list(range(len(asset.joint_names)))[action_joint_ids]
            for action_idx, joint_id in enumerate(action_joint_ids):
                joint_name = asset.joint_names[int(joint_id)].lower()
                if "_hr_joint" in joint_name:
                    groups["hr"].append(action_idx)
                elif "_hp_joint" in joint_name:
                    groups["hp"].append(action_idx)
                elif "_kn_joint" in joint_name:
                    groups["kn"].append(action_idx)

        if not any(groups.values()) and self.env.num_actions == 12:
            groups = {
                "hr": [0, 3, 6, 9],
                "hp": [1, 4, 7, 10],
                "kn": [2, 5, 8, 11],
            }

        return {
            name: torch.tensor(indices, dtype=torch.long, device=self.device)
            for name, indices in groups.items()
            if indices
        }

    def _empty_action_group_accumulators(self) -> dict[str, dict[str, float]]:
        return {
            name: {"saturation_count": 0.0, "element_count": 0, "abs_sum": 0.0, "abs_max": 0.0}
            for name in self.action_group_indices
        }

    def _apply_reward_weight_curriculum(self, iteration: int) -> dict | None:
        if not self.reward_weight_curriculum_terms:
            return None

        base_env = getattr(self.env, "unwrapped", self.env)
        if not hasattr(base_env, "reward_manager"):
            raise RuntimeError("Reward weight curriculum requires env.reward_manager")

        if not 0.0 < self.reward_weight_curriculum_annealing_rate <= 1.0:
            raise ValueError(
                "reward_weight_curriculum_annealing_rate must be in (0, 1], "
                f"got {self.reward_weight_curriculum_annealing_rate}"
            )

        scale = self.reward_weight_curriculum_annealing_rate ** iteration
        current_weights = {}
        for term_name in self.reward_weight_curriculum_terms:
            try:
                term_cfg = base_env.reward_manager.get_term_cfg(term_name)
            except ValueError:
                if term_name not in self.reward_weight_curriculum_missing_terms:
                    print(
                        f"[RewardWeightCurriculum] reward term '{term_name}' not found; skipping.",
                        flush=True,
                    )
                    self.reward_weight_curriculum_missing_terms.add(term_name)
                continue

            if term_name not in self.reward_weight_curriculum_base_weights:
                self.reward_weight_curriculum_base_weights[term_name] = float(term_cfg.weight)

            term_cfg.weight = self.reward_weight_curriculum_base_weights[term_name] * scale
            base_env.reward_manager.set_term_cfg(term_name, term_cfg)
            current_weights[term_name] = float(term_cfg.weight)

        return {
            "scale": float(scale),
            "annealing_rate": self.reward_weight_curriculum_annealing_rate,
            "weights": current_weights,
        }

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

    def _check_tensor_finite(self, name: str, tensor: torch.Tensor, context: dict) -> None:
        if not self.nan_debug_enabled or torch.isfinite(tensor).all():
            return
        dump_path = self.nan_debug_dir / (
            f"runner_nan_it{context.get('iteration', -1)}_step{context.get('rollout_step', -1)}_{name}.pt"
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

    def _print_actuator_gains_once(self) -> None:
        """Print actuator kp/kd gains for env 0 once at startup."""
        base_env = getattr(self.env, "unwrapped", None)
        if base_env is None:
            return

        try:
            asset = base_env.scene["robot"]
        except KeyError:
            return

        if not hasattr(asset, "actuators"):
            return

        for actuator_name, actuator in asset.actuators.items():
            kp = [round(float(v), 2) for v in actuator.stiffness[0].detach().cpu().tolist()]
            kd = [round(float(v), 2) for v in actuator.damping[0].detach().cpu().tolist()]
            print(f"[ACTUATOR_DEBUG] {actuator_name} kp {kp}", flush=True)
            print(f"[ACTUATOR_DEBUG] {actuator_name} kd {kd}", flush=True)

    def _print_spawn_debug_once(self) -> None:
        """Print spawn/terrain height debug values for env 0 once at startup."""
        base_env = getattr(self.env, "unwrapped", None)
        if base_env is None or not hasattr(base_env, "scene"):
            return

        scene = base_env.scene
        try:
            asset = scene["robot"]
        except KeyError:
            return

        env_idx = 0
        try:
            env_origin_z = float(scene.env_origins[env_idx, 2].item())
        except Exception:
            env_origin_z = float("nan")

        try:
            root_world_z = float(asset.data.root_pos_w[env_idx, 2].item())
        except Exception:
            root_world_z = float("nan")

        try:
            foot_ids = asset.find_bodies(".*TIP")[0]
        except Exception:
            foot_ids = []

        if len(foot_ids) == 0:
            print(
                "[SPAWN_DEBUG]",
                f"env={env_idx}",
                f"env_origin_z={env_origin_z:.4f}",
                f"root_world_z={root_world_z:.4f}",
                "foot_world_z=[]",
                "terrain_z=[]",
                flush=True,
            )
            return

        foot_positions = asset.data.body_pos_w[env_idx, foot_ids, :]
        foot_world_z = [round(float(v), 4) for v in foot_positions[:, 2].detach().cpu().tolist()]

        terrain_z_values = [float("nan")] * len(foot_ids)
        if "height_scanner" in scene.sensors:
            height_scanner = scene.sensors["height_scanner"]
            ray_hits_w = height_scanner.data.ray_hits_w[env_idx]
            valid_hits = torch.isfinite(ray_hits_w[:, 2])
            if torch.any(valid_hits):
                foot_xy = foot_positions[:, :2]
                ray_xy = ray_hits_w[:, :2]
                dist_sq = torch.sum(torch.square(foot_xy.unsqueeze(1) - ray_xy.unsqueeze(0)), dim=-1)
                dist_sq = torch.where(valid_hits.unsqueeze(0), dist_sq, torch.inf)
                nearest_hit_idx = torch.argmin(dist_sq, dim=1)
                terrain_z = ray_hits_w[nearest_hit_idx, 2]
                terrain_z = torch.where(torch.isfinite(terrain_z), terrain_z, torch.full_like(terrain_z, torch.nan))
                terrain_z_values = [round(float(v), 4) for v in terrain_z.detach().cpu().tolist()]

        print(
            "[SPAWN_DEBUG]",
            f"env={env_idx}",
            f"env_origin_z={env_origin_z:.4f}",
            f"root_world_z={root_world_z:.4f}",
            f"foot_world_z={foot_world_z}",
            f"terrain_z={terrain_z_values}",
            flush=True,
        )

    def _print_action_debug(self, step_idx: int) -> None:
        """Print action processing and actuator outputs for env 0."""
        def _round_tensor(values: torch.Tensor) -> list[float]:
            return [round(float(v), 2) for v in values.detach().cpu().tolist()]

        base_env = getattr(self.env, "unwrapped", None)
        if base_env is None or not hasattr(base_env, "action_manager"):
            return
        if not getattr(base_env.cfg, "debug_action_print", False):
            return

        try:
            asset = base_env.scene["robot"]
            action_term = base_env.action_manager.get_term("joint_pos")
        except KeyError:
            return

        env_idx = 0
        joint_ids = action_term._joint_ids
        print(
            "[ACTION_DEBUG]",
            f"step={step_idx}",
            f"raw={_round_tensor(action_term.raw_actions[env_idx])}",
            f"processed={_round_tensor(action_term.processed_actions[env_idx])}",
            f"default_joint_pos={_round_tensor(asset.data.default_joint_pos[env_idx, joint_ids])}",
            f"joint_pos_target={_round_tensor(asset.data.joint_pos_target[env_idx, joint_ids])}",
            f"current_joint_pos={_round_tensor(asset.data.joint_pos[env_idx, joint_ids])}",
            f"computed_torque={_round_tensor(asset.data.computed_torque[env_idx, joint_ids])}",
            f"applied_torque={_round_tensor(asset.data.applied_torque[env_idx, joint_ids])}",
            flush=True,
        )

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Wandb & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "wandb":
                from ..utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
                self.writer.log_config(
                    self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg
                )
            elif self.logger_type == "tensorboard":
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, extras = self.env.get_observations()
        obs_history_raw = extras["observations"].get("obsHistory")
        assert obs_history_raw is not None, "obsHistory not found in observations"
        if (
            obs_history_raw.dim() >= 3
            and not self.obs_history_offsets
            and obs_history_raw.shape[1] != self.obs_history_len
        ):
            raise ValueError(
                f"obsHistory length mismatch: env={obs_history_raw.shape[1]} cfg={self.obs_history_len}"
            )
        obs_history = self._prepare_obs_history(obs_history_raw)
        critic_obs = extras["observations"].get("critic")
        commands = extras["observations"].get("commands")
        encoder_target = self._resolve_encoder_target(extras["observations"])
        adaptive_obs = extras["observations"].get("adaptive") if self.uses_phase2_adaptive else None
        adaptive_critic_obs = extras["observations"].get("adaptiveCritic", adaptive_obs) if self.uses_phase2_adaptive else None
        adaptive_history = extras["observations"].get("adaptiveHistory") if self.uses_phase2_load_adaptive else None
        phase2_log_obs = self._resolve_phase2_log_obs(extras["observations"]) if self.uses_phase2_adaptive else None

        obs, obs_history, commands, critic_obs = (
            obs.to(self.device),
            obs_history.to(self.device),
            commands.to(self.device),
            critic_obs.to(self.device),
        )
        encoder_target = encoder_target.to(self.device) if encoder_target is not None else None
        adaptive_obs = adaptive_obs.to(self.device) if adaptive_obs is not None else None
        adaptive_critic_obs = adaptive_critic_obs.to(self.device) if adaptive_critic_obs is not None else None
        adaptive_history = adaptive_history.to(self.device) if adaptive_history is not None else None
        phase2_log_obs = phase2_log_obs.to(self.device) if phase2_log_obs is not None else None
        # ???
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            if hasattr(self.alg, "set_learning_iteration"):
                self.alg.set_learning_iteration(it)
            reward_curriculum_state = self._apply_reward_weight_curriculum(it)
            start = time.time()
            phase2_log_sum = None
            phase2_log_count = 0
            action_saturation_count = 0.0
            action_element_count = 0
            action_abs_sum = 0.0
            action_abs_max = 0.0
            action_group_accumulators = self._empty_action_group_accumulators()
            terrain_episode_stats = {}
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    debug_context = {"iteration": it, "rollout_step": i}
                    transition_metadata = None
                    if self.uses_phase2_load_adaptive:
                        transition_metadata = self._collect_phase2_metadata(it * self.num_steps_per_env + i)
                    elif self.uses_context_estimator and getattr(self.alg, "uses_terrain_labels", False):
                        transition_metadata = self._collect_terrain_metadata(it * self.num_steps_per_env + i)
                    if hasattr(self.alg, "set_nan_debug_context"):
                        self.alg.set_nan_debug_context(**debug_context)
                    self._check_tensor_finite("obs_pre_act", obs, debug_context)
                    self._check_tensor_finite("obs_history_pre_act", obs_history, debug_context)
                    self._check_tensor_finite("commands_pre_act", commands, debug_context)
                    self._check_tensor_finite("critic_obs_pre_act", critic_obs, debug_context)
                    if encoder_target is not None:
                        self._check_tensor_finite("encoder_target_pre_act", encoder_target, debug_context)
                    if self.uses_phase2_adaptive:
                        self._check_tensor_finite("adaptive_obs_pre_act", adaptive_obs, debug_context)
                        self._check_tensor_finite("adaptive_critic_obs_pre_act", adaptive_critic_obs, debug_context)
                        if self.uses_phase2_load_adaptive:
                            self._check_tensor_finite("adaptive_history_pre_act", adaptive_history, debug_context)
                            actions = self.alg.act(
                                obs,
                                obs_history,
                                commands,
                                critic_obs,
                                adaptive_obs,
                                adaptive_critic_obs,
                                adaptive_history,
                            )
                        else:
                            actions = self.alg.act(
                                obs, obs_history, commands, critic_obs, adaptive_obs, adaptive_critic_obs
                            )
                    else:
                        actions = self.alg.act(obs, obs_history, commands, critic_obs)
                    action_abs = actions.detach().abs()
                    action_saturation_count += (action_abs > 0.95).sum().item()
                    action_element_count += action_abs.numel()
                    action_abs_sum += action_abs.sum().item()
                    action_abs_max = max(action_abs_max, action_abs.max().item())
                    for group_name, action_indices in self.action_group_indices.items():
                        group_action_abs = action_abs[:, action_indices]
                        group_acc = action_group_accumulators[group_name]
                        group_acc["saturation_count"] += (group_action_abs > 0.95).sum().item()
                        group_acc["element_count"] += group_action_abs.numel()
                        group_acc["abs_sum"] += group_action_abs.sum().item()
                        group_acc["abs_max"] = max(group_acc["abs_max"], group_action_abs.max().item())
                    terrain_log_snapshot = (
                        self._collect_terrain_episode_snapshot() if self.log_dir is not None else None
                    )
                    (obs, rewards, dones, infos) = self.env.step(actions)
                    if self.env.num_envs == 1:
                        self._print_action_debug(step_idx=i)

                    critic_obs = infos["observations"]["critic"]
                    commands = infos["observations"]["commands"]
                    next_encoder_target = self._resolve_encoder_target(infos["observations"])
                    next_adaptive_obs = infos["observations"].get("adaptive") if self.uses_phase2_adaptive else None
                    next_adaptive_critic_obs = (
                        infos["observations"].get("adaptiveCritic", next_adaptive_obs)
                        if self.uses_phase2_adaptive
                        else None
                    )
                    next_adaptive_history = (
                        infos["observations"].get("adaptiveHistory") if self.uses_phase2_load_adaptive else None
                    )
                    next_adaptive_reward = self._compute_adaptive_reward() if self.uses_phase2_adaptive else None
                    next_phase2_log_obs = (
                        self._resolve_phase2_log_obs(infos["observations"]) if self.uses_phase2_adaptive else None
                    )
                    next_obs_history = infos["observations"].get("obsHistory")
                    assert next_obs_history is not None, "obsHistory not found in step observations"
                    done_env_ids = (dones > 0).nonzero(as_tuple=False).flatten()

                    obs, obs_history, commands, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        self._prepare_obs_history(next_obs_history).to(self.device),
                        commands.to(self.device),
                        critic_obs.to(self.device), # critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    next_encoder_target = next_encoder_target.to(self.device) if next_encoder_target is not None else None
                    next_adaptive_obs = next_adaptive_obs.to(self.device) if next_adaptive_obs is not None else None
                    next_adaptive_critic_obs = (
                        next_adaptive_critic_obs.to(self.device) if next_adaptive_critic_obs is not None else None
                    )
                    next_adaptive_history = (
                        next_adaptive_history.to(self.device) if next_adaptive_history is not None else None
                    )
                    next_adaptive_reward = (
                        next_adaptive_reward.to(self.device) if next_adaptive_reward is not None else None
                    )
                    next_phase2_log_obs = (
                        next_phase2_log_obs.to(self.device) if next_phase2_log_obs is not None else None
                    )
                    self._check_tensor_finite("obs_post_step", obs, debug_context)
                    self._check_tensor_finite("obs_history_post_step", obs_history, debug_context)
                    self._check_tensor_finite("commands_post_step", commands, debug_context)
                    self._check_tensor_finite("critic_obs_post_step", critic_obs, debug_context)
                    if next_encoder_target is not None:
                        self._check_tensor_finite("encoder_target_post_step", next_encoder_target, debug_context)
                    if self.uses_phase2_adaptive:
                        self._check_tensor_finite("adaptive_obs_post_step", next_adaptive_obs, debug_context)
                        self._check_tensor_finite(
                            "adaptive_critic_obs_post_step", next_adaptive_critic_obs, debug_context
                        )
                        if self.uses_phase2_load_adaptive:
                            self._check_tensor_finite(
                                "adaptive_history_post_step", next_adaptive_history, debug_context
                            )
                        self._check_tensor_finite("adaptive_reward_post_step", next_adaptive_reward, debug_context)
                        if next_phase2_log_obs is not None:
                            self._check_tensor_finite("phase2_log_post_step", next_phase2_log_obs, debug_context)
                            log_sum = next_phase2_log_obs.detach().sum(dim=0)
                            phase2_log_sum = log_sum if phase2_log_sum is None else phase2_log_sum + log_sum
                            phase2_log_count += next_phase2_log_obs.shape[0]
                    self._check_tensor_finite("rewards_post_step", rewards, debug_context)
                    if self.uses_phase2_adaptive:
                        self.alg.process_env_step(
                            rewards,
                            dones,
                            infos,
                            obs,
                            encoder_target,
                            next_adaptive_reward,
                            transition_metadata,
                        )
                    elif transition_metadata is not None:
                        self.alg.process_env_step(
                            rewards,
                            dones,
                            infos,
                            obs,
                            encoder_target,
                            transition_metadata=transition_metadata,
                        )
                    else:
                        self.alg.process_env_step(rewards, dones, infos, obs, encoder_target)
                    encoder_target = next_encoder_target
                    adaptive_obs = next_adaptive_obs
                    adaptive_critic_obs = next_adaptive_critic_obs
                    adaptive_history = next_adaptive_history
                    phase2_log_obs = next_phase2_log_obs

                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        self._update_terrain_episode_stats(
                            terrain_episode_stats, terrain_log_snapshot, new_ids.flatten(), infos
                        )
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                action_saturation_ratio = (
                    action_saturation_count / action_element_count if action_element_count > 0 else 0.0
                )
                action_abs_mean = action_abs_sum / action_element_count if action_element_count > 0 else 0.0
                action_group_stats = {}
                for group_name, group_acc in action_group_accumulators.items():
                    group_count = group_acc["element_count"]
                    action_group_stats[group_name] = {
                        "action_saturation_ratio": (
                            group_acc["saturation_count"] / group_count if group_count > 0 else 0.0
                        ),
                        "action_abs_mean": group_acc["abs_sum"] / group_count if group_count > 0 else 0.0,
                        "action_abs_max": group_acc["abs_max"],
                    }

                # Learning step
                start = stop

                critic_obs_ = torch.cat((critic_obs, commands), dim=-1)
                if self.alg.critic_take_latent:
                    encoder_out = self.alg.encoder.encode(obs_history)
                    self.alg.compute_returns(
                        torch.cat((critic_obs_, encoder_out), dim=-1)
                    )
                else:
                    self.alg.compute_returns(critic_obs_)
                if self.uses_phase2_adaptive:
                    self.alg.compute_adaptive_returns(adaptive_critic_obs)

            update_result = self.alg.update()
            ce_loss_terms = None
            phase2_loss_terms = None
            phase2_log_terms = None
            if phase2_log_sum is not None and phase2_log_count > 0:
                phase2_log_mean = phase2_log_sum / phase2_log_count
                phase2_log_terms = {
                    name: float(phase2_log_mean[idx].item())
                    for idx, name in enumerate(self.phase2_log_names[: phase2_log_mean.numel()])
                }
            if len(update_result) == 26:
                (
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
                    ang_vel_target_mean_x,
                    ang_vel_target_mean_y,
                    ang_vel_target_mean_z,
                    ang_vel_target_std_x,
                    ang_vel_target_std_y,
                    ang_vel_target_std_z,
                    ang_vel_target_max_x,
                    ang_vel_target_max_y,
                    ang_vel_target_max_z,
                    ang_vel_target_rmse_x,
                    ang_vel_target_rmse_y,
                    ang_vel_target_rmse_z,
                    mean_proj_grav_loss,
                    mean_surrogate_loss,
                    mean_kl,
                    mean_explained_variance,
                ) = update_result
            elif len(update_result) == 6:
                (
                    mean_value_loss,
                    mean_extra_loss,
                    mean_surrogate_loss,
                    mean_kl,
                    mean_explained_variance,
                    ce_loss_terms,
                ) = update_result
                mean_lin_vel_loss = 0.0
                mean_ang_vel_loss = 0.0
                mean_ang_vel_x_loss = 0.0
                mean_ang_vel_y_loss = 0.0
                mean_ang_vel_z_loss = 0.0
                mean_ang_vel_x_smooth_l1 = 0.0
                bootstrap_metric = None
                bootstrap_metric_ang_vel = None
                ang_vel_target_mean_x = 0.0
                ang_vel_target_mean_y = 0.0
                ang_vel_target_mean_z = 0.0
                ang_vel_target_std_x = 0.0
                ang_vel_target_std_y = 0.0
                ang_vel_target_std_z = 0.0
                ang_vel_target_max_x = 0.0
                ang_vel_target_max_y = 0.0
                ang_vel_target_max_z = 0.0
                ang_vel_target_rmse_x = 0.0
                ang_vel_target_rmse_y = 0.0
                ang_vel_target_rmse_z = 0.0
                mean_proj_grav_loss = 0.0
            elif len(update_result) == 7:
                (
                    mean_value_loss,
                    mean_extra_loss,
                    mean_surrogate_loss,
                    mean_kl,
                    mean_explained_variance,
                    ce_loss_terms,
                    phase2_loss_terms,
                ) = update_result
                mean_lin_vel_loss = 0.0
                mean_ang_vel_loss = 0.0
                mean_ang_vel_x_loss = 0.0
                mean_ang_vel_y_loss = 0.0
                mean_ang_vel_z_loss = 0.0
                mean_ang_vel_x_smooth_l1 = 0.0
                bootstrap_metric = None
                bootstrap_metric_ang_vel = None
                ang_vel_target_mean_x = 0.0
                ang_vel_target_mean_y = 0.0
                ang_vel_target_mean_z = 0.0
                ang_vel_target_std_x = 0.0
                ang_vel_target_std_y = 0.0
                ang_vel_target_std_z = 0.0
                ang_vel_target_max_x = 0.0
                ang_vel_target_max_y = 0.0
                ang_vel_target_max_z = 0.0
                ang_vel_target_rmse_x = 0.0
                ang_vel_target_rmse_y = 0.0
                ang_vel_target_rmse_z = 0.0
                mean_proj_grav_loss = 0.0
            elif len(update_result) == 5:
                (
                    mean_value_loss,
                    mean_extra_loss,
                    mean_surrogate_loss,
                    mean_kl,
                    mean_explained_variance,
                ) = update_result
                mean_lin_vel_loss = 0.0
                mean_ang_vel_loss = 0.0
                mean_ang_vel_x_loss = 0.0
                mean_ang_vel_y_loss = 0.0
                mean_ang_vel_z_loss = 0.0
                mean_ang_vel_x_smooth_l1 = 0.0
                bootstrap_metric = None
                bootstrap_metric_ang_vel = None
                ang_vel_target_mean_x = 0.0
                ang_vel_target_mean_y = 0.0
                ang_vel_target_mean_z = 0.0
                ang_vel_target_std_x = 0.0
                ang_vel_target_std_y = 0.0
                ang_vel_target_std_z = 0.0
                ang_vel_target_max_x = 0.0
                ang_vel_target_max_y = 0.0
                ang_vel_target_max_z = 0.0
                ang_vel_target_rmse_x = 0.0
                ang_vel_target_rmse_y = 0.0
                ang_vel_target_rmse_z = 0.0
                mean_proj_grav_loss = 0.0
            else:
                raise ValueError(f"Unexpected update() result length: {len(update_result)}")
            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(
            os.path.join(
                self.log_dir, "model_{}.pt".format(self.current_learning_iteration)
            )
        )

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
        # mean_std = self.alg.actor_critic.std.mean()
        mean_std = torch.exp(self.alg.actor_critic.logstd).mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar(
            "Loss/value_function", locs["mean_value_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/encoder", locs["mean_extra_loss"], locs["it"])
        self.writer.add_scalar("Loss/encoder_lin_vel", locs["mean_lin_vel_loss"], locs["it"])
        self.writer.add_scalar("Loss/encoder_ang_vel", locs["mean_ang_vel_loss"], locs["it"])
        self.writer.add_scalar("Loss/encoder_ang_vel_x", locs["mean_ang_vel_x_loss"], locs["it"])
        self.writer.add_scalar("Loss/encoder_ang_vel_y", locs["mean_ang_vel_y_loss"], locs["it"])
        self.writer.add_scalar("Loss/encoder_ang_vel_z", locs["mean_ang_vel_z_loss"], locs["it"])
        self.writer.add_scalar("Loss/encoder_proj_grav", locs["mean_proj_grav_loss"], locs["it"])
        if locs["ce_loss_terms"] is not None:
            self.writer.add_scalar("ce/loss", locs["ce_loss_terms"]["loss"], locs["it"])
            self.writer.add_scalar("ce/est_loss", locs["ce_loss_terms"]["est_loss"], locs["it"])
            self.writer.add_scalar("ce/vae_loss", locs["ce_loss_terms"]["vae_loss"], locs["it"])
            self.writer.add_scalar("ce/recon_loss", locs["ce_loss_terms"]["recon_loss"], locs["it"])
            self.writer.add_scalar("ce/beta", locs["ce_loss_terms"]["beta"], locs["it"])
            self.writer.add_scalar("ce/kl_loss", locs["ce_loss_terms"]["kl_loss"], locs["it"])
            for name in ("terrain_cls_loss", "terrain_prop_loss", "terrain_warmup_scale"):
                if name in locs["ce_loss_terms"]:
                    self.writer.add_scalar(f"ce/{name}", locs["ce_loss_terms"][name], locs["it"])
        if locs.get("phase2_loss_terms") is not None:
            self.writer.add_scalar("phase2/nominal_loss", locs["phase2_loss_terms"]["nominal_loss"], locs["it"])
            self.writer.add_scalar("phase2/adaptive_loss", locs["phase2_loss_terms"]["adaptive_loss"], locs["it"])
            self.writer.add_scalar(
                "phase2/adaptive_value_loss", locs["phase2_loss_terms"]["adaptive_value_loss"], locs["it"]
            )
            self.writer.add_scalar(
                "phase2/adaptive_surrogate_loss", locs["phase2_loss_terms"]["adaptive_surrogate_loss"], locs["it"]
            )
            self.writer.add_scalar("phase2/ce_loss", locs["phase2_loss_terms"]["ce_loss"], locs["it"])
            self.writer.add_scalar("phase2/total_loss", locs["phase2_loss_terms"]["total_loss"], locs["it"])
            self.writer.add_scalar("phase2/adaptive_kl", locs["phase2_loss_terms"]["adaptive_kl"], locs["it"])
            self.writer.add_scalar(
                "phase2/adaptive_reward_scale", locs["phase2_loss_terms"]["adaptive_reward_scale"], locs["it"]
            )
            if "load_triplet_loss" in locs["phase2_loss_terms"]:
                self.writer.add_scalar(
                    "phase2/load_triplet_loss", locs["phase2_loss_terms"]["load_triplet_loss"], locs["it"]
                )
            if "load_loss" in locs["phase2_loss_terms"]:
                self.writer.add_scalar("phase2/load_loss", locs["phase2_loss_terms"]["load_loss"], locs["it"])
            logged_phase2_keys = {
                "nominal_loss",
                "adaptive_loss",
                "adaptive_value_loss",
                "adaptive_surrogate_loss",
                "ce_loss",
                "total_loss",
                "adaptive_kl",
                "adaptive_reward_scale",
                "load_triplet_loss",
                "load_loss",
            }
            for key, value in locs["phase2_loss_terms"].items():
                if key in logged_phase2_keys:
                    continue
                self.writer.add_scalar(f"phase2/{key}", value, locs["it"])
        if locs.get("phase2_log_terms") is not None:
            for name, value in locs["phase2_log_terms"].items():
                self.writer.add_scalar(f"phase2/{name}", value, locs["it"])
        for terrain_name, stats in locs.get("terrain_episode_stats", {}).items():
            count = stats.get("count", 0.0)
            if count <= 0.0:
                continue
            self.writer.add_scalar(f"Terrain/{terrain_name}/count", count, locs["it"])
            self.writer.add_scalar(
                f"Terrain/{terrain_name}/base_contact_rate", stats.get("base_contact", 0.0) / count, locs["it"]
            )
            self.writer.add_scalar(
                f"Terrain/{terrain_name}/timeout_rate", stats.get("time_out", 0.0) / count, locs["it"]
            )
            error_count = stats.get("error_vel_xy_count", 0.0)
            if error_count > 0.0:
                self.writer.add_scalar(
                    f"Terrain/{terrain_name}/error_vel_xy", stats.get("error_vel_xy_sum", 0.0) / error_count, locs["it"]
                )
            terrain_level_count = stats.get("terrain_level_count", 0.0)
            if terrain_level_count > 0.0:
                self.writer.add_scalar(
                    f"Terrain/{terrain_name}/mean_terrain_level",
                    stats.get("terrain_level_sum", 0.0) / terrain_level_count,
                    locs["it"],
                )
                terrain_level_max = stats.get("terrain_level_max", float("-inf"))
                if terrain_level_max > float("-inf"):
                    self.writer.add_scalar(
                        f"Terrain/{terrain_name}/max_terrain_level",
                        terrain_level_max,
                        locs["it"],
                    )
                self.writer.add_scalar(
                    f"Terrain/{terrain_name}/level_ge_10_rate",
                    stats.get("terrain_level_ge_10", 0.0) / terrain_level_count,
                    locs["it"],
                )
                self.writer.add_scalar(
                    f"Terrain/{terrain_name}/level_ge_14_rate",
                    stats.get("terrain_level_ge_14", 0.0) / terrain_level_count,
                    locs["it"],
                )
                self.writer.add_scalar(
                    f"Terrain/{terrain_name}/level_ge_20_rate",
                    stats.get("terrain_level_ge_20", 0.0) / terrain_level_count,
                    locs["it"],
                )
                self.writer.add_scalar(
                    f"Terrain/{terrain_name}/level_ge_24_rate",
                    stats.get("terrain_level_ge_24", 0.0) / terrain_level_count,
                    locs["it"],
                )
        base_env = getattr(self.env, "unwrapped", None)
        frontier_metrics = getattr(base_env, "_pongbot_frontier_metrics", {}) if base_env is not None else {}
        for terrain_name, metrics in frontier_metrics.items():
            for metric_name, value in metrics.items():
                if math.isfinite(float(value)):
                    self.writer.add_scalar(f"Frontier/{terrain_name}/{metric_name}", value, locs["it"])
        if locs["bootstrap_metric"] is not None:
            self.writer.add_scalar("Bootstrap/metric_normalized_mse", locs["bootstrap_metric"], locs["it"])
        if locs["bootstrap_metric_ang_vel"] is not None:
            self.writer.add_scalar(
                "Bootstrap/metric_ang_vel_normalized_mse", locs["bootstrap_metric_ang_vel"], locs["it"]
            )
        self.writer.add_scalar("Target/ang_vel_x_mean", locs["ang_vel_target_mean_x"], locs["it"])
        self.writer.add_scalar("Target/ang_vel_y_mean", locs["ang_vel_target_mean_y"], locs["it"])
        self.writer.add_scalar("Target/ang_vel_z_mean", locs["ang_vel_target_mean_z"], locs["it"])
        self.writer.add_scalar("Target/ang_vel_x_std", locs["ang_vel_target_std_x"], locs["it"])
        self.writer.add_scalar("Target/ang_vel_y_std", locs["ang_vel_target_std_y"], locs["it"])
        self.writer.add_scalar("Target/ang_vel_z_std", locs["ang_vel_target_std_z"], locs["it"])
        self.writer.add_scalar("Target/ang_vel_x_max", locs["ang_vel_target_max_x"], locs["it"])
        self.writer.add_scalar("Target/ang_vel_y_max", locs["ang_vel_target_max_y"], locs["it"])
        self.writer.add_scalar("Target/ang_vel_z_max", locs["ang_vel_target_max_z"], locs["it"])
        self.writer.add_scalar("Target/ang_vel_x_rmse", locs["ang_vel_target_rmse_x"], locs["it"])
        self.writer.add_scalar("Target/ang_vel_y_rmse", locs["ang_vel_target_rmse_y"], locs["it"])
        self.writer.add_scalar("Target/ang_vel_z_rmse", locs["ang_vel_target_rmse_z"], locs["it"])
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/action_bound",
            float(getattr(self.alg, "last_action_bound_loss", 0.0)),
            locs["it"],
        )
        self.writer.add_scalar(
            "Loss/action_bound_weighted",
            float(getattr(self.alg, "last_action_bound_loss_weighted", 0.0)),
            locs["it"],
        )
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        if hasattr(self.alg, "get_bootstrap_state"):
            bootstrap_state = self.alg.get_bootstrap_state()
            if bootstrap_state.get("bootstrap_metric") is not None:
                self.writer.add_scalar(
                    "Bootstrap/metric", bootstrap_state["bootstrap_metric"], locs["it"]
                )
            self.writer.add_scalar(
                "Bootstrap/use_raw_imu", float(bootstrap_state["use_raw_imu"]), locs["it"]
            )
            self.writer.add_scalar(
                "Bootstrap/encoder_active", float(bootstrap_state.get("bootstrap_encoder_active", False)), locs["it"]
            )
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Policy/mean_kl", locs["mean_kl"], locs["it"])
        self.writer.add_scalar("Policy/action_saturation_ratio", locs["action_saturation_ratio"], locs["it"])
        self.writer.add_scalar("Policy/action_abs_mean", locs["action_abs_mean"], locs["it"])
        self.writer.add_scalar("Policy/action_abs_max", locs["action_abs_max"], locs["it"])
        for group_name, group_stats in locs.get("action_group_stats", {}).items():
            self.writer.add_scalar(
                f"Policy/action_saturation_ratio/{group_name}",
                group_stats["action_saturation_ratio"],
                locs["it"],
            )
            self.writer.add_scalar(
                f"Policy/action_abs_mean/{group_name}",
                group_stats["action_abs_mean"],
                locs["it"],
            )
            self.writer.add_scalar(
                f"Policy/action_abs_max/{group_name}",
                group_stats["action_abs_max"],
                locs["it"],
            )
        self.writer.add_scalar("Value/explained_variance", locs["mean_explained_variance"], locs["it"])
        if locs.get("reward_curriculum_state") is not None:
            curriculum_state = locs["reward_curriculum_state"]
            self.writer.add_scalar("RewardCurriculum/anneal_scale", curriculum_state["scale"], locs["it"])
            self.writer.add_scalar(
                "RewardCurriculum/annealing_rate", curriculum_state["annealing_rate"], locs["it"]
            )
            for name, weight in curriculum_state["weights"].items():
                self.writer.add_scalar(f"RewardCurriculum/{name}_weight", weight, locs["it"])
                self.writer.add_scalar(f"RewardCurriculum/{name}_abs_weight", abs(weight), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            if (
                self.logger_type != "wandb"
            ):  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar(
                    "Train/mean_reward/time",
                    statistics.mean(locs["rewbuffer"]),
                    self.tot_time,
                )
                self.writer.add_scalar(
                    "Train/mean_episode_length/time",
                    statistics.mean(locs["lenbuffer"]),
                    self.tot_time,
                )

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "
        reward_curriculum_string = ""
        if locs.get("reward_curriculum_state") is not None:
            curriculum_state = locs["reward_curriculum_state"]
            interval = max(1, self.reward_weight_curriculum_log_interval)
            if locs["it"] % interval == 0:
                weights = ", ".join(
                    f"{name}={weight:.3e}" for name, weight in curriculum_state["weights"].items()
                )
                reward_curriculum_string = (
                    f"""{'Reward anneal scale:':>{pad}} {curriculum_state['scale']:.6f} """
                    f"""(lambda={curriculum_state['annealing_rate']:.6f})\n"""
                    f"""{'Reward annealed weights:':>{pad}} {weights}\n"""
                )

        if len(locs["rewbuffer"]) > 0:
            ce_loss_string = ""
            if locs["ce_loss_terms"] is not None:
                ce_loss_string = (
                    f"""{'CE loss:':>{pad}} {locs['ce_loss_terms']['loss']:.6f}\n"""
                    f"""{'CE est loss:':>{pad}} {locs['ce_loss_terms']['est_loss']:.6f}\n"""
                    f"""{'CE vae loss:':>{pad}} {locs['ce_loss_terms']['vae_loss']:.6f}\n"""
                    f"""{'CE recon loss:':>{pad}} {locs['ce_loss_terms']['recon_loss']:.6f}\n"""
                    f"""{'CE beta:':>{pad}} {locs['ce_loss_terms']['beta']:.6f}\n"""
                    f"""{'CE kl loss:':>{pad}} {locs['ce_loss_terms']['kl_loss']:.6f}\n"""
                )
            phase2_loss_string = ""
            if locs.get("phase2_loss_terms") is not None:
                phase2_loss_string = (
                    f"""{'Phase2 nominal loss:':>{pad}} {locs['phase2_loss_terms']['nominal_loss']:.6f}\n"""
                    f"""{'Phase2 adaptive loss:':>{pad}} {locs['phase2_loss_terms']['adaptive_loss']:.6f}\n"""
                    f"""{'Phase2 adaptive value loss:':>{pad}} {locs['phase2_loss_terms']['adaptive_value_loss']:.6f}\n"""
                    f"""{'Phase2 adaptive sur loss:':>{pad}} {locs['phase2_loss_terms']['adaptive_surrogate_loss']:.6f}\n"""
                    f"""{'Phase2 CE loss:':>{pad}} {locs['phase2_loss_terms']['ce_loss']:.6f}\n"""
                    f"""{'Phase2 total loss:':>{pad}} {locs['phase2_loss_terms']['total_loss']:.6f}\n"""
                )
                if "load_triplet_loss" in locs["phase2_loss_terms"]:
                    phase2_loss_string += (
                        f"""{'Phase2 load triplet loss:':>{pad}} """
                        f"""{locs['phase2_loss_terms']['load_triplet_loss']:.6f}\n"""
                    )
            phase2_metric_string = ""
            if locs.get("phase2_log_terms") is not None:
                phase2_metric_string = "".join(
                    f"{('Phase2 ' + name + ':'):>{pad}} {value:.4f}\n"
                    for name, value in locs["phase2_log_terms"].items()
                )
            bootstrap_string = ""
            if hasattr(self.alg, "get_bootstrap_state"):
                bootstrap_state = self.alg.get_bootstrap_state()
                metric_value = bootstrap_state.get("bootstrap_metric")
                metric_string = f"{metric_value:.6f}" if metric_value is not None else "n/a"
                bootstrap_string = (
                    f"""{'Bootstrap mode:':>{pad}} {bootstrap_state['mode']}\n"""
                    f"""{'Use raw imu:':>{pad}} {bootstrap_state['use_raw_imu']}\n"""
                    f"""{'Encoder active:':>{pad}} {bootstrap_state.get('bootstrap_encoder_active', False)}\n"""
                    f"""{'Bootstrap metric:':>{pad}} {metric_string}\n"""
                )
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Explained variance:':>{pad}} {locs['mean_explained_variance']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean KL divergence:':>{pad}} {locs['mean_kl']:.6f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.4f}\n"""
                f"""{'Learning rate:':>{pad}} {self.alg.learning_rate:.4f}\n"""
                f"""{reward_curriculum_string}"""
                f"""{'Encoder loss:':>{pad}} {locs['mean_extra_loss']:.4f}\n"""
                f"""{ce_loss_string}"""
                f"""{phase2_loss_string}"""
                f"""{phase2_metric_string}"""
                # f"""{'Encoder lin vel loss:':>{pad}} {locs['mean_lin_vel_loss']:.4f}\n"""
                # f"""{'Encoder ang vel loss:':>{pad}} {locs['mean_ang_vel_loss']:.4f}\n"""
                # f"""{'Encoder ang vel x loss:':>{pad}} {locs['mean_ang_vel_x_loss']:.4f}\n"""
                # f"""{'Encoder ang vel y loss:':>{pad}} {locs['mean_ang_vel_y_loss']:.4f}\n"""
                # f"""{'Encoder ang vel z loss:':>{pad}} {locs['mean_ang_vel_z_loss']:.4f}\n"""
                # f"""{'Encoder ang vel x smooth l1:':>{pad}} {locs['mean_ang_vel_x_smooth_l1']:.4f}\n"""
                # f"""{'Encoder proj grav loss:':>{pad}} {locs['mean_proj_grav_loss']:.4f}\n"""
                # f"""{'Bootstrap normalized mse:':>{pad}} {locs['bootstrap_metric']:.4f}\n"""
                # f"""{'Bootstrap ang vel norm mse:':>{pad}} {locs['bootstrap_metric_ang_vel']:.4f}\n"""
                # f"""{'Target ang vel x mean/std/max/rmse:':>{pad}} {locs['ang_vel_target_mean_x']:.4f} / {locs['ang_vel_target_std_x']:.4f} / {locs['ang_vel_target_max_x']:.4f} / {locs['ang_vel_target_rmse_x']:.4f}\n"""
                # f"""{'Target ang vel y mean/std/max/rmse:':>{pad}} {locs['ang_vel_target_mean_y']:.4f} / {locs['ang_vel_target_std_y']:.4f} / {locs['ang_vel_target_max_y']:.4f} / {locs['ang_vel_target_rmse_y']:.4f}\n"""
                # f"""{'Target ang vel z mean/std/max/rmse:':>{pad}} {locs['ang_vel_target_mean_z']:.4f} / {locs['ang_vel_target_std_z']:.4f} / {locs['ang_vel_target_max_z']:.4f} / {locs['ang_vel_target_rmse_z']:.4f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )

            # log_string += bootstrap_string
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            ce_loss_string = ""
            if locs["ce_loss_terms"] is not None:
                ce_loss_string = (
                    f"""{'CE loss:':>{pad}} {locs['ce_loss_terms']['loss']:.6f}\n"""
                    f"""{'CE est loss:':>{pad}} {locs['ce_loss_terms']['est_loss']:.6f}\n"""
                    f"""{'CE vae loss:':>{pad}} {locs['ce_loss_terms']['vae_loss']:.6f}\n"""
                    f"""{'CE recon loss:':>{pad}} {locs['ce_loss_terms']['recon_loss']:.6f}\n"""
                    f"""{'CE beta:':>{pad}} {locs['ce_loss_terms']['beta']:.6f}\n"""
                    f"""{'CE kl loss:':>{pad}} {locs['ce_loss_terms']['kl_loss']:.6f}\n"""
                )
            phase2_loss_string = ""
            if locs.get("phase2_loss_terms") is not None:
                phase2_loss_string = (
                    f"""{'Phase2 nominal loss:':>{pad}} {locs['phase2_loss_terms']['nominal_loss']:.6f}\n"""
                    f"""{'Phase2 adaptive loss:':>{pad}} {locs['phase2_loss_terms']['adaptive_loss']:.6f}\n"""
                    f"""{'Phase2 adaptive value loss:':>{pad}} {locs['phase2_loss_terms']['adaptive_value_loss']:.6f}\n"""
                    f"""{'Phase2 adaptive sur loss:':>{pad}} {locs['phase2_loss_terms']['adaptive_surrogate_loss']:.6f}\n"""
                    f"""{'Phase2 CE loss:':>{pad}} {locs['phase2_loss_terms']['ce_loss']:.6f}\n"""
                    f"""{'Phase2 total loss:':>{pad}} {locs['phase2_loss_terms']['total_loss']:.6f}\n"""
                )
                if "load_triplet_loss" in locs["phase2_loss_terms"]:
                    phase2_loss_string += (
                        f"""{'Phase2 load triplet loss:':>{pad}} """
                        f"""{locs['phase2_loss_terms']['load_triplet_loss']:.6f}\n"""
                    )
            phase2_metric_string = ""
            if locs.get("phase2_log_terms") is not None:
                phase2_metric_string = "".join(
                    f"{('Phase2 ' + name + ':'):>{pad}} {value:.4f}\n"
                    for name, value in locs["phase2_log_terms"].items()
                )
            bootstrap_string = ""
            if hasattr(self.alg, "get_bootstrap_state"):
                bootstrap_state = self.alg.get_bootstrap_state()
                metric_value = bootstrap_state.get("bootstrap_metric")
                metric_string = f"{metric_value:.6f}" if metric_value is not None else "n/a"
                bootstrap_string = (
                    f"""{'Bootstrap mode:':>{pad}} {bootstrap_state['mode']}\n"""
                    f"""{'Use raw imu:':>{pad}} {bootstrap_state['use_raw_imu']}\n"""
                    f"""{'Encoder active:':>{pad}} {bootstrap_state.get('bootstrap_encoder_active', False)}\n"""
                    f"""{'Bootstrap metric:':>{pad}} {metric_string}\n"""
                )
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Explained variance:':>{pad}} {locs['mean_explained_variance']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean KL divergence:':>{pad}} {locs['mean_kl']:.6f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{reward_curriculum_string}"""
                f"""{'Encoder loss:':>{pad}} {locs['mean_extra_loss']:.4f}\n"""
                f"""{ce_loss_string}"""
                f"""{phase2_loss_string}"""
                f"""{phase2_metric_string}"""
                f"""{'Encoder lin vel loss:':>{pad}} {locs['mean_lin_vel_loss']:.4f}\n"""
                f"""{'Encoder ang vel loss:':>{pad}} {locs['mean_ang_vel_loss']:.4f}\n"""
                f"""{'Encoder ang vel x loss:':>{pad}} {locs['mean_ang_vel_x_loss']:.4f}\n"""
                f"""{'Encoder ang vel y loss:':>{pad}} {locs['mean_ang_vel_y_loss']:.4f}\n"""
                f"""{'Encoder ang vel z loss:':>{pad}} {locs['mean_ang_vel_z_loss']:.4f}\n"""
                f"""{'Encoder ang vel x smooth l1:':>{pad}} {locs['mean_ang_vel_x_smooth_l1']:.4f}\n"""
                f"""{'Encoder proj grav loss:':>{pad}} {locs['mean_proj_grav_loss']:.4f}\n"""
                f"""{'Bootstrap normalized mse:':>{pad}} {locs['bootstrap_metric']:.4f}\n"""
                f"""{'Bootstrap ang vel norm mse:':>{pad}} {locs['bootstrap_metric_ang_vel']:.4f}\n"""
                f"""{'Target ang vel x mean/std/max/rmse:':>{pad}} {locs['ang_vel_target_mean_x']:.4f} / {locs['ang_vel_target_std_x']:.4f} / {locs['ang_vel_target_max_x']:.4f} / {locs['ang_vel_target_rmse_x']:.4f}\n"""
                f"""{'Target ang vel y mean/std/max/rmse:':>{pad}} {locs['ang_vel_target_mean_y']:.4f} / {locs['ang_vel_target_std_y']:.4f} / {locs['ang_vel_target_max_y']:.4f} / {locs['ang_vel_target_rmse_y']:.4f}\n"""
                f"""{'Target ang vel z mean/std/max/rmse:':>{pad}} {locs['ang_vel_target_mean_z']:.4f} / {locs['ang_vel_target_std_z']:.4f} / {locs['ang_vel_target_max_z']:.4f} / {locs['ang_vel_target_rmse_z']:.4f}\n"""
            )
            # log_string += bootstrap_string
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "encoder_state_dict": self.alg.encoder.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "extra_optimizer_state_dict": self.alg.extra_optimizer.state_dict() if self.alg.extra_optimizer is not None else None,
                "adaptive_model_state_dict": self.alg.adaptive_actor_critic.state_dict()
                if hasattr(self.alg, "adaptive_actor_critic")
                else None,
                "adaptive_optimizer_state_dict": self.alg.adaptive_optimizer.state_dict()
                if hasattr(self.alg, "adaptive_optimizer")
                else None,
                "load_encoder_state_dict": self.alg.load_encoder.state_dict()
                if hasattr(self.alg, "load_encoder")
                else None,
                "load_transition_model_state_dict": self.alg.load_transition_model.state_dict()
                if hasattr(self.alg, "load_transition_model")
                else None,
                "load_probe_state_dict": {
                    "payload_mass": self.alg.load_payload_mass_probe.state_dict(),
                    "payload_pos": self.alg.load_payload_pos_probe.state_dict(),
                    "force": self.alg.load_force_probe.state_dict(),
                    "delta_action": self.alg.load_delta_action_probe.state_dict(),
                }
                if hasattr(self.alg, "load_payload_mass_probe")
                else None,
                "load_probe_optimizer_state_dict": self.alg.load_probe_optimizer.state_dict()
                if hasattr(self.alg, "load_probe_optimizer")
                else None,
                "bootstrap_state": self.alg.get_bootstrap_state() if hasattr(self.alg, "get_bootstrap_state") else None,
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=False):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.encoder.load_state_dict(loaded_dict["encoder_state_dict"])
        if hasattr(self.alg, "adaptive_actor_critic") and loaded_dict.get("adaptive_model_state_dict") is not None:
            self.alg.adaptive_actor_critic.load_state_dict(loaded_dict["adaptive_model_state_dict"])
            if hasattr(self.alg.adaptive_actor_critic, "clamp_logstd_"):
                self.alg.adaptive_actor_critic.clamp_logstd_()
        if hasattr(self.alg, "load_encoder") and loaded_dict.get("load_encoder_state_dict") is not None:
            self.alg.load_encoder.load_state_dict(loaded_dict["load_encoder_state_dict"])
        if (
            hasattr(self.alg, "load_transition_model")
            and loaded_dict.get("load_transition_model_state_dict") is not None
        ):
            self.alg.load_transition_model.load_state_dict(loaded_dict["load_transition_model_state_dict"])
        load_probe_state = loaded_dict.get("load_probe_state_dict")
        if hasattr(self.alg, "load_payload_mass_probe") and load_probe_state is not None:
            self.alg.load_payload_mass_probe.load_state_dict(load_probe_state["payload_mass"])
            self.alg.load_payload_pos_probe.load_state_dict(load_probe_state["payload_pos"])
            self.alg.load_force_probe.load_state_dict(load_probe_state["force"])
            self.alg.load_delta_action_probe.load_state_dict(load_probe_state["delta_action"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if self.alg.extra_optimizer is not None and loaded_dict.get("extra_optimizer_state_dict") is not None:
                self.alg.extra_optimizer.load_state_dict(loaded_dict["extra_optimizer_state_dict"])
            if hasattr(self.alg, "adaptive_optimizer") and loaded_dict.get("adaptive_optimizer_state_dict") is not None:
                self.alg.adaptive_optimizer.load_state_dict(loaded_dict["adaptive_optimizer_state_dict"])
            if hasattr(self.alg, "load_probe_optimizer") and loaded_dict.get("load_probe_optimizer_state_dict") is not None:
                self.alg.load_probe_optimizer.load_state_dict(loaded_dict["load_probe_optimizer_state_dict"])
        bootstrap_state = loaded_dict.get("bootstrap_state")
        if bootstrap_state and hasattr(self.alg, "bootstrap_mode"):
            self.alg.bootstrap_mode = bootstrap_state.get("mode", self.alg.bootstrap_mode)
            self.alg.bootstrap_metric = bootstrap_state.get("bootstrap_metric")
            self.alg.bootstrap_encoder_active = bootstrap_state.get("bootstrap_encoder_active", False)
            self.alg.use_raw_imu_bootstrap = bootstrap_state.get("use_raw_imu", self.alg.use_raw_imu_bootstrap)
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_inference_encoder(self, device=None):
        self.alg.encoder.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.encoder.to(device)
        return self.alg.encoder.encode

    def build_inference_actor_obs(self, obs, commands, obs_history, encoder_output):
        if self.uses_context_estimator:
            history_input = obs_history.flatten(start_dim=1) if self.policy_include_history else None
            return self.alg._build_context_policy_input(encoder_output, obs, commands, history_input)

        history_input = obs_history.flatten(start_dim=1) if obs_history.dim() > 2 else obs_history
        return self.alg._build_actor_obs(encoder_output, obs, commands, history_input)

    def get_encoder_export_shape(self):
        if self.uses_context_estimator:
            return self.obs_history_shape
        return self.obs_history_dim

    def get_actor_critic(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic
