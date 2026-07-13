"""Runner for the isolated R2 barrier-reward paper task.

@version 0.0.10
@update 2026-07-13: Save independent optimizer states and report fixed full-range rough terrain.
@update 2026-07-13: Print each PaperBarrier terminal metric on its own labeled line.
"""

import os
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.algorithm.paper_barrier_ppo import PaperBarrierPPO
from rsl_rl.modules.paper_barrier_actor_critic import PaperBarrierActorCritic


class PaperBarrierRunner:
    """Minimal runner that implements the paper's two-critic actor update."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.env = env
        self.cfg = train_cfg
        self.device = device
        self.log_dir = log_dir
        self.writer = None
        self.current_learning_iteration = 0
        self.num_steps_per_env = train_cfg["num_steps_per_env"]
        self.save_interval = train_cfg["save_interval"]
        self.obs_history_len = 1
        self.obs_history_offsets = ()
        self.uses_context_estimator = False
        self.base_env = self._find_base_env(env)
        self.calibration = self._calibrate_robot(train_cfg.get("calibration_steps", 100))
        self.terrain_column_names = self._terrain_column_names()

        obs, extras = env.get_observations()
        observations = extras["observations"]
        commands = observations["commands"]
        targets = observations["estimator_target"]
        policy_cfg = dict(train_cfg["policy"])
        policy_cfg.pop("class_name", None)
        actor_critic = PaperBarrierActorCritic(
            proprio_dim=obs.shape[1],
            command_dim=commands.shape[1],
            target_dim=targets.shape[1],
            num_actions=env.num_actions,
            **policy_cfg,
        ).to(device)
        self.alg = PaperBarrierPPO(actor_critic, dict(train_cfg["algorithm"]), device)
        self.alg.encoder = actor_critic.estimator
        self.alg.init_storage(
            env.num_envs,
            self.num_steps_per_env,
            obs.shape[1],
            commands.shape[1],
            targets.shape[1],
            env.num_actions,
        )
        print(
            "[PAPER_BARRIER] dimensions: "
            f"proprio={obs.shape[1]}, commands={commands.shape[1]}, target={targets.shape[1]}, "
            f"actions={env.num_actions}, rollout_batch={env.num_envs * self.num_steps_per_env}"
        )
        print(f"[PAPER_BARRIER] calibration: {self.calibration}")

    @staticmethod
    def _find_base_env(env):
        current = env
        visited = set()
        while id(current) not in visited:
            visited.add(id(current))
            if hasattr(current, "scene") and hasattr(current, "reward_manager"):
                return current
            unwrapped = getattr(current, "unwrapped", None)
            if unwrapped is not None and unwrapped is not current:
                current = unwrapped
                continue
            nested = getattr(current, "env", None)
            if nested is not None:
                current = nested
                continue
            break
        raise RuntimeError("PaperBarrierRunner could not locate the ManagerBasedRLEnv")

    def _calibrate_robot(self, steps: int) -> dict:
        """Settle at zero action, then derive morphology-dependent reward constants."""
        if steps <= 0:
            return {}
        with torch.inference_mode():
            asset = self.base_env.scene["robot"]
            terrain = self.base_env.scene.terrain
            env_ids = torch.arange(self.base_env.num_envs, device=self.base_env.device)
            if terrain.terrain_origins is not None:
                terrain.terrain_levels[env_ids] = 0
                terrain.env_origins[env_ids] = terrain.terrain_origins[
                    terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
                ]
                self.base_env.scene.env_origins[env_ids] = terrain.env_origins[env_ids]

            root_state = asset.data.default_root_state.clone()
            root_state[:, :3] += self.base_env.scene.env_origins
            root_state[:, 7:] = 0.0
            asset.write_root_state_to_sim(root_state, env_ids=env_ids)
            asset.write_joint_state_to_sim(
                asset.data.default_joint_pos.clone(),
                torch.zeros_like(asset.data.default_joint_vel),
                env_ids=env_ids,
            )

            contact_sensor = self.base_env.scene.sensors["contact_forces"]
            foot_names = ("FL_TIP", "FR_TIP", "RL_TIP", "RR_TIP")
            thigh_names = ("FL_THIGH", "FR_THIGH", "RL_THIGH", "RR_THIGH")
            foot_ids = [int(asset.find_bodies(name, preserve_order=True)[0][0]) for name in foot_names]
            thigh_ids = [int(asset.find_bodies(name, preserve_order=True)[0][0]) for name in thigh_names]
            sensor_foot_ids = [contact_sensor.body_names.index(name) for name in foot_names]

            zero_actions = torch.zeros(self.env.num_envs, self.env.num_actions, device=self.device)
            geometry_step = min(20, steps)
            force_samples = []
            nominal_foot_pos_b = None
            captured_thigh_height = None
            for step in range(1, steps + 1):
                self.env.step(zero_actions)
                if step >= max(1, geometry_step // 2):
                    force_samples.append(
                        torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_foot_ids], dim=-1).clone()
                    )
                if step == geometry_step:
                    relative_w = asset.data.body_pos_w[:, foot_ids] - asset.data.root_pos_w.unsqueeze(1)
                    root_quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, 4, -1)
                    foot_pos_b = self._quat_apply_inverse(root_quat, relative_w)
                    nominal_foot_pos_b = torch.median(foot_pos_b, dim=0).values

                    terrain_heights = []
                    for name in ("fl_foot_scanner", "fr_foot_scanner", "rl_foot_scanner", "rr_foot_scanner"):
                        hit_z = self.base_env.scene.sensors[name].data.ray_hits_w[..., 2]
                        hit_z = torch.where(torch.isfinite(hit_z), hit_z, -torch.inf)
                        local_height = torch.max(hit_z, dim=1).values
                        local_height = torch.where(
                            torch.isfinite(local_height), local_height, torch.zeros_like(local_height)
                        )
                        terrain_heights.append(local_height)
                    terrain_z = torch.stack(terrain_heights, dim=1)
                    captured_thigh_height = asset.data.body_pos_w[:, thigh_ids, 2] - terrain_z

            if nominal_foot_pos_b is None or captured_thigh_height is None:
                raise RuntimeError("Paper calibration did not capture canonical geometry")
            thigh_height = captured_thigh_height
            front_center = torch.median(torch.mean(thigh_height[:, :2], dim=1))
            hind_center = torch.median(torch.mean(thigh_height[:, 2:], dim=1))
            mean_center = 0.5 * (front_center + hind_center)
            length_scale = torch.clamp(mean_center / 0.59, 0.65, 1.50)

            forces = torch.cat(force_samples, dim=0)
            stance_forces = forces[forces > 1.0]
            stance_force = torch.median(stance_forces) if stance_forces.numel() else torch.tensor(10.0, device=self.device)
            contact_force_on = torch.clamp(0.15 * stance_force, 1.0, 50.0)

            self.base_env._paper_nominal_foot_pos_b = nominal_foot_pos_b.detach()
            self.base_env._paper_body_height_centers = (float(front_center), float(hind_center))
            self.base_env._paper_nominal_height_difference = float(front_center - hind_center)
            self.base_env._paper_length_scale = float(length_scale)
            self.base_env._paper_contact_force_on = float(contact_force_on)
            self.base_env._paper_foot_radius = 0.03

            calibration = {
                "nominal_foot_pos_b": nominal_foot_pos_b.cpu().tolist(),
                "body_height_centers": [float(front_center), float(hind_center)],
                "nominal_height_difference": float(front_center - hind_center),
                "length_scale": float(length_scale),
                "contact_force_on": float(contact_force_on),
                "foot_radius": 0.03,
                "settling_steps": steps,
            }
            self.env.reset()
            return calibration

    @staticmethod
    def _quat_apply_inverse(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        # Isaac Lab quaternions use (w, x, y, z).
        xyz = quaternion[..., 1:]
        w = quaternion[..., :1]
        return vector - 2.0 * w * torch.cross(xyz, vector, dim=-1) + 2.0 * torch.cross(
            xyz, torch.cross(xyz, vector, dim=-1), dim=-1
        )

    def _raw_reward_streams(self) -> tuple[torch.Tensor, torch.Tensor]:
        manager = self.base_env.reward_manager
        names = list(manager.active_terms)
        try:
            standard_idx = names.index("paper_standard")
            barrier_idx = names.index("paper_barrier")
        except ValueError as exc:
            raise RuntimeError(f"Expected paper reward terms, got {names}") from exc
        # RewardManager._step_reward stores raw weighted terms (before the environment dt sum).
        return (
            manager._step_reward[:, standard_idx].to(self.device),
            manager._step_reward[:, barrier_idx].to(self.device),
        )

    def _terrain_column_names(self) -> list[str]:
        """Map curriculum terrain column indices to their configured terrain names."""
        generator = getattr(self.base_env.cfg.scene.terrain, "terrain_generator", None)
        if generator is None or not getattr(generator, "sub_terrains", None):
            return ["unknown"]

        items = list(generator.sub_terrains.items())
        total = sum(float(cfg.proportion) for _, cfg in items)
        names = []
        for column in range(int(generator.num_cols)):
            point = column / float(generator.num_cols) + 0.001
            cumulative = 0.0
            selected_name = items[-1][0]
            for name, cfg in items:
                cumulative += float(cfg.proportion) / total
                if point < cumulative:
                    selected_name = name
                    break
            names.append(selected_name)
        return names

    @staticmethod
    def _accumulate_metric(sums: dict, counts: dict, name: str, value: torch.Tensor) -> None:
        """Accumulate finite tensor values without synchronizing the GPU each step."""
        value = value.float()
        finite = torch.isfinite(value)
        finite_sum = torch.where(finite, value, 0.0).sum()
        finite_count = finite.sum()
        if name not in sums:
            sums[name] = finite_sum
            counts[name] = finite_count
        else:
            sums[name] += finite_sum
            counts[name] += finite_count

    def _accumulate_rollout_diagnostics(self, sums: dict, counts: dict) -> None:
        """Collect reward diagnostics as true rollout averages instead of final-step snapshots."""
        for group_name in ("_paper_standard_metrics", "_paper_barrier_metrics"):
            for name, value in getattr(self.base_env, group_name, {}).items():
                self._accumulate_metric(sums, counts, name, value)

        barrier_metrics = getattr(self.base_env, "_paper_barrier_metrics", {})
        moving = barrier_metrics.get("Tracking/is_moving_command")
        lin_error = barrier_metrics.get("Tracking/lin_vel_xy_error")
        actual_speed = barrier_metrics.get("Tracking/actual_xy_speed")
        if moving is not None and lin_error is not None and actual_speed is not None:
            moving_mask = moving > 0.5
            self._accumulate_metric(
                sums,
                counts,
                "Tracking/moving_lin_vel_xy_error",
                lin_error[moving_mask],
            )
            self._accumulate_metric(
                sums,
                counts,
                "Tracking/stationary_xy_speed",
                actual_speed[~moving_mask],
            )

    @staticmethod
    def _finalize_rollout_diagnostics(sums: dict, counts: dict) -> dict[str, float]:
        """Convert accumulated rollout tensors to scalar means once per iteration."""
        result = {}
        for name, value_sum in sums.items():
            denominator = torch.clamp(counts[name], min=1)
            result[name] = float((value_sum / denominator).detach())
        return result

    def _new_episode_statistics(self) -> dict[str, torch.Tensor]:
        """Allocate iteration-local episode outcome accumulators."""
        scalar_names = (
            "count",
            "timeout",
            "tracking_success",
            "length_sum",
            "reward_sum",
            "xy_error_sum",
            "yaw_error_sum",
            "moving_count",
            "moving_success",
            "stationary_count",
            "stationary_success",
            "level_sum",
        )
        statistics = {name: torch.zeros((), device=self.device) for name in scalar_names}
        num_columns = len(self.terrain_column_names)
        for name in ("terrain_count", "terrain_timeout", "terrain_tracking_success", "terrain_length_sum"):
            statistics[name] = torch.zeros(num_columns, device=self.device)
        return statistics

    def _record_episode_outcomes(
        self,
        statistics: dict[str, torch.Tensor],
        done_ids: torch.Tensor,
        time_outs: torch.Tensor,
        episode_reward: torch.Tensor,
        episode_length: torch.Tensor,
        episode_xy_error: torch.Tensor,
        episode_yaw_error: torch.Tensor,
        episode_moving_steps: torch.Tensor,
        episode_terrain_type: torch.Tensor,
        episode_terrain_level: torch.Tensor,
    ) -> None:
        """Record survival, tracking, and terrain outcomes for completed episodes."""
        if done_ids.numel() == 0:
            return

        lengths = torch.clamp(episode_length[done_ids], min=1.0)
        xy_error = episode_xy_error[done_ids] / lengths
        yaw_error = episode_yaw_error[done_ids] / lengths
        timeout = time_outs[done_ids] > 0.5
        tracking_success = timeout & (xy_error <= 0.4) & (yaw_error <= 0.4)
        moving_episode = episode_moving_steps[done_ids] >= 0.5 * lengths
        terrain_types = episode_terrain_type[done_ids].long().clamp(0, len(self.terrain_column_names) - 1)
        ones = torch.ones_like(lengths)

        statistics["count"] += ones.sum()
        statistics["timeout"] += timeout.float().sum()
        statistics["tracking_success"] += tracking_success.float().sum()
        statistics["length_sum"] += lengths.sum()
        statistics["reward_sum"] += episode_reward[done_ids].sum()
        statistics["xy_error_sum"] += xy_error.sum()
        statistics["yaw_error_sum"] += yaw_error.sum()
        statistics["moving_count"] += moving_episode.float().sum()
        statistics["moving_success"] += (tracking_success & moving_episode).float().sum()
        statistics["stationary_count"] += (~moving_episode).float().sum()
        statistics["stationary_success"] += (tracking_success & ~moving_episode).float().sum()
        statistics["level_sum"] += episode_terrain_level[done_ids].float().sum()
        statistics["terrain_count"] += torch.bincount(
            terrain_types,
            weights=ones,
            minlength=len(self.terrain_column_names),
        )
        statistics["terrain_timeout"] += torch.bincount(
            terrain_types,
            weights=timeout.float(),
            minlength=len(self.terrain_column_names),
        )
        statistics["terrain_tracking_success"] += torch.bincount(
            terrain_types,
            weights=tracking_success.float(),
            minlength=len(self.terrain_column_names),
        )
        statistics["terrain_length_sum"] += torch.bincount(
            terrain_types,
            weights=lengths,
            minlength=len(self.terrain_column_names),
        )

    def _finalize_episode_statistics(self, statistics: dict[str, torch.Tensor]) -> dict[str, float]:
        """Create interpretable episode and per-terrain success metrics."""
        count = torch.clamp(statistics["count"], min=1.0)
        moving_count = torch.clamp(statistics["moving_count"], min=1.0)
        stationary_count = torch.clamp(statistics["stationary_count"], min=1.0)
        metrics = {
            "Episode/completed_count": float(statistics["count"]),
            "Episode/timeout_rate": float(statistics["timeout"] / count),
            "Episode/early_termination_rate": float(1.0 - statistics["timeout"] / count),
            "Episode/tracking_success_rate": float(statistics["tracking_success"] / count),
            "Episode/moving_tracking_success_rate": float(statistics["moving_success"] / moving_count),
            "Episode/stationary_tracking_success_rate": float(
                statistics["stationary_success"] / stationary_count
            ),
            "Episode/mean_length_current": float(statistics["length_sum"] / count),
            "Episode/survival_fraction": float(
                statistics["length_sum"] / count / float(self.env.max_episode_length)
            ),
            "Episode/mean_reward_current": float(statistics["reward_sum"] / count),
            "Episode/mean_lin_vel_xy_max_abs_error": float(statistics["xy_error_sum"] / count),
            "Episode/mean_yaw_rate_error": float(statistics["yaw_error_sum"] / count),
            "Terrain/episode_mean_level": float(statistics["level_sum"] / count),
        }

        unique_names = list(dict.fromkeys(self.terrain_column_names))
        for terrain_name in unique_names:
            indices = [index for index, name in enumerate(self.terrain_column_names) if name == terrain_name]
            terrain_count = torch.clamp(statistics["terrain_count"][indices].sum(), min=1.0)
            metrics[f"TerrainSuccess/{terrain_name}_timeout_rate"] = float(
                statistics["terrain_timeout"][indices].sum() / terrain_count
            )
            metrics[f"TerrainSuccess/{terrain_name}_tracking_rate"] = float(
                statistics["terrain_tracking_success"][indices].sum() / terrain_count
            )
            metrics[f"TerrainEpisode/{terrain_name}_mean_length"] = float(
                statistics["terrain_length_sum"][indices].sum() / terrain_count
            )
        return metrics

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, extras = self.env.get_observations()
        obs = obs.to(self.device)
        commands = extras["observations"]["commands"].to(self.device)
        targets = extras["observations"]["estimator_target"].to(self.device)
        reward_buffer = deque(maxlen=100)
        length_buffer = deque(maxlen=100)
        episode_reward = torch.zeros(self.env.num_envs, device=self.device)
        episode_length = torch.zeros(self.env.num_envs, device=self.device)
        episode_xy_error = torch.zeros(self.env.num_envs, device=self.device)
        episode_yaw_error = torch.zeros(self.env.num_envs, device=self.device)
        episode_moving_steps = torch.zeros(self.env.num_envs, device=self.device)
        terrain = self.base_env.scene.terrain
        episode_terrain_type = terrain.terrain_types.to(self.device).clone()
        episode_terrain_level = terrain.terrain_levels.to(self.device).clone()
        total_steps = 0

        for iteration in range(self.current_learning_iteration, num_learning_iterations):
            start = time.time()
            self.base_env._paper_learning_iteration = iteration
            standard_sum = 0.0
            barrier_sum = 0.0
            rollout_metric_sums = {}
            rollout_metric_counts = {}
            episode_statistics = self._new_episode_statistics()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    transition = self.alg.act(obs, commands, targets)
                    next_obs, _, dones, infos = self.env.step(transition["actions"])
                    standard_rewards, barrier_rewards = self._raw_reward_streams()
                    dones = dones.to(self.device)
                    dones_float = dones.float()
                    time_outs = infos.get("time_outs")
                    if time_outs is None:
                        time_outs = torch.zeros_like(dones_float)
                    else:
                        time_outs = time_outs.to(self.device).float()
                    standard_rewards = standard_rewards + self.alg.gamma * transition["standard_values"] * time_outs
                    barrier_rewards = barrier_rewards + self.alg.gamma * transition["barrier_values"] * time_outs
                    self.alg.storage.add(
                        {
                            "proprio": obs,
                            "commands": commands,
                            "targets": targets,
                            "estimated_targets": transition["estimated_targets"],
                            "actions": transition["actions"],
                            "log_probs": transition["log_probs"],
                            "action_mean": transition["action_mean"],
                            "action_std": transition["action_std"],
                            "standard_rewards": standard_rewards,
                            "barrier_rewards": barrier_rewards,
                            "dones": dones_float,
                            "standard_values": transition["standard_values"],
                            "barrier_values": transition["barrier_values"],
                        }
                    )
                    standard_sum += float(standard_rewards.mean())
                    barrier_sum += float(barrier_rewards.mean())
                    self._accumulate_rollout_diagnostics(rollout_metric_sums, rollout_metric_counts)
                    barrier_metrics = getattr(self.base_env, "_paper_barrier_metrics", {})
                    episode_xy_error += barrier_metrics["Tracking/lin_vel_xy_max_abs_error"].to(self.device)
                    episode_yaw_error += barrier_metrics["Tracking/yaw_rate_error"].to(self.device)
                    episode_moving_steps += barrier_metrics["Tracking/is_moving_command"].to(self.device)
                    episode_reward += standard_rewards + barrier_rewards
                    episode_length += 1
                    done_ids = torch.nonzero(dones, as_tuple=False).flatten()
                    if done_ids.numel():
                        self._record_episode_outcomes(
                            episode_statistics,
                            done_ids,
                            time_outs,
                            episode_reward,
                            episode_length,
                            episode_xy_error,
                            episode_yaw_error,
                            episode_moving_steps,
                            episode_terrain_type,
                            episode_terrain_level,
                        )
                        reward_buffer.extend(episode_reward[done_ids].cpu().tolist())
                        length_buffer.extend(episode_length[done_ids].cpu().tolist())
                        episode_reward[done_ids] = 0.0
                        episode_length[done_ids] = 0.0
                        episode_xy_error[done_ids] = 0.0
                        episode_yaw_error[done_ids] = 0.0
                        episode_moving_steps[done_ids] = 0.0
                        episode_terrain_type[done_ids] = terrain.terrain_types[done_ids].to(self.device)
                        episode_terrain_level[done_ids] = terrain.terrain_levels[done_ids].to(self.device)
                    obs = next_obs.to(self.device)
                    commands = infos["observations"]["commands"].to(self.device)
                    targets = infos["observations"]["estimator_target"].to(self.device)

                self.alg.compute_returns(obs, commands, targets)
            metrics = self.alg.update()
            total_steps += self.env.num_envs * self.num_steps_per_env
            elapsed = time.time() - start
            mean_standard = standard_sum / self.num_steps_per_env
            mean_barrier = barrier_sum / self.num_steps_per_env
            terrain_difficulty = 1.0
            rollout_metrics = self._finalize_rollout_diagnostics(rollout_metric_sums, rollout_metric_counts)
            rollout_metrics["Terrain/actual_mean_level"] = float(terrain.terrain_levels.float().mean())
            rollout_metrics["Terrain/actual_max_level"] = float(terrain.terrain_levels.max())
            episode_metrics = self._finalize_episode_statistics(episode_statistics)
            self._log(
                iteration,
                total_steps,
                elapsed,
                mean_standard,
                mean_barrier,
                terrain_difficulty,
                metrics,
                rollout_metrics,
                episode_metrics,
                reward_buffer,
                length_buffer,
            )
            if self.log_dir is not None and iteration % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{iteration}.pt"))
            self.current_learning_iteration = iteration + 1
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _log(
        self,
        iteration,
        total_steps,
        elapsed,
        standard,
        barrier,
        difficulty,
        optimization_metrics,
        rollout_metrics,
        episode_metrics,
        rewards,
        lengths,
    ):
        mean_episode_reward = sum(rewards) / len(rewards) if rewards else 0.0
        mean_episode_length = sum(lengths) / len(lengths) if lengths else 0.0
        print(
            "\n[PAPER_BARRIER]\n"
            f"  iteration                 : {iteration:05d}\n"
            f"  total_steps               : {total_steps}\n"
            f"  iteration_time_s          : {elapsed:.2f}\n"
            f"  standard_reward_per_step  : {standard:.4f}\n"
            f"  barrier_reward_per_step   : {barrier:.4f}\n"
            f"  mean_episode_reward       : {mean_episode_reward:.3f}\n"
            f"  mean_episode_length       : {mean_episode_length:.1f}\n"
            f"  terrain_difficulty        : {difficulty:.3f}\n"
            f"  estimator_loss            : {optimization_metrics['estimator_loss']:.4f}\n"
            f"  policy_loss               : {optimization_metrics['policy_loss']:.4f}"
        )
        print(
            "[PAPER_BARRIER_DIAG]\n"
            f"  timeout_rate              : {episode_metrics['Episode/timeout_rate']:.3f}\n"
            f"  tracking_success_rate     : {episode_metrics['Episode/tracking_success_rate']:.3f}\n"
            f"  lin_vel_xy_error          : {rollout_metrics.get('Tracking/lin_vel_xy_error', 0.0):.3f}\n"
            f"  body_contact_rate         : {rollout_metrics.get('Contact/body_rate', 0.0):.4f}\n"
            f"  gait_violation_rate       : {rollout_metrics.get('Diagnostics/gait_violation', 0.0):.3f}\n"
            f"  clearance_violation_rate  : {rollout_metrics.get('Diagnostics/clearance_violation', 0.0):.3f}\n"
            f"  actual_mean_terrain_level : {rollout_metrics.get('Terrain/actual_mean_level', 0.0):.2f}\n"
            f"  mean_kl                   : {optimization_metrics.get('mean_kl', 0.0):.5f}\n"
            f"  learning_rate             : {optimization_metrics.get('learning_rate', 0.0):.6f}"
        )
        if self.writer is None:
            return
        self.writer.add_scalar("Reward/paper_standard_per_step", standard, iteration)
        self.writer.add_scalar("Reward/paper_barrier_per_step", barrier, iteration)
        self.writer.add_scalar("Train/mean_episode_reward", mean_episode_reward, iteration)
        self.writer.add_scalar("Train/mean_episode_length", mean_episode_length, iteration)
        self.writer.add_scalar("Curriculum/terrain_difficulty", difficulty, iteration)
        self.writer.add_scalar("Performance/iteration_seconds", elapsed, iteration)
        for name, value in optimization_metrics.items():
            self.writer.add_scalar(f"Loss/{name}", value, iteration)
        for name, value in rollout_metrics.items():
            self.writer.add_scalar(name, value, iteration)
        for name, value in episode_metrics.items():
            self.writer.add_scalar(name, value, iteration)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "optimizer_state_dicts": self.alg.optimizer_state_dict(),
                "iter": self.current_learning_iteration,
                "calibration": self.calibration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=False):
        checkpoint = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        self.alg.actor_critic.clamp_logstd_()
        if load_optimizer:
            if "optimizer_state_dicts" in checkpoint:
                self.alg.load_optimizer_state_dict(checkpoint["optimizer_state_dicts"])
            elif "optimizer_state_dict" in checkpoint:
                print(
                    "[WARN] Legacy PaperBarrier checkpoint has one coupled optimizer; "
                    "model weights were restored but optimizer state was not."
                )
        self.current_learning_iteration = checkpoint.get("iter", 0)
        self.calibration = checkpoint.get("calibration", self.calibration)
        self._restore_calibration(self.calibration)
        return checkpoint.get("infos")

    def _restore_calibration(self, calibration: dict) -> None:
        if not calibration:
            return
        self.base_env._paper_nominal_foot_pos_b = torch.tensor(
            calibration["nominal_foot_pos_b"], device=self.device
        )
        self.base_env._paper_body_height_centers = tuple(calibration["body_height_centers"])
        self.base_env._paper_nominal_height_difference = calibration["nominal_height_difference"]
        self.base_env._paper_length_scale = calibration["length_scale"]
        self.base_env._paper_contact_force_on = calibration["contact_force_on"]
        self.base_env._paper_foot_radius = calibration["foot_radius"]

    def get_inference_policy(self, device=None):
        model = self.alg.actor_critic
        model.eval()
        if device is not None:
            model.to(device)
        return model.act_inference

    def get_inference_encoder(self, device=None):
        estimator = self.alg.actor_critic.estimator
        estimator.eval()
        if device is not None:
            estimator.to(device)
        return estimator.encode

    @staticmethod
    def build_inference_actor_obs(obs, commands, obs_history, encoder_output):
        del obs_history
        return torch.cat((obs, commands, encoder_output), dim=1)

    def get_encoder_export_shape(self):
        return (self.alg.actor_critic.proprio_dim,)

    def get_actor_critic(self, device=None):
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic
