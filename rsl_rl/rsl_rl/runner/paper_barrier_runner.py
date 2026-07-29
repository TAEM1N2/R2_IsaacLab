"""Runner for the isolated R2 barrier-reward paper task.

@version 0.0.12
@update 2026-07-13: Log moving traversal competence and checkpoint the R2 terrain curriculum.
@update 2026-07-13: Restore calibrated terrain assignments and enforce the R2 rough-trot training contract.
"""

import math
import os
import time
from collections import Counter, deque

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
        # Diagnostics are for observability, not the PPO objective.  Sampling
        # them every few control steps avoids repeatedly traversing dozens of
        # detached metric tensors in the Python rollout loop.
        self.diagnostics_interval = max(int(train_cfg.get("diagnostics_interval", 1)), 1)
        self.save_interval = train_cfg["save_interval"]
        self.obs_history_len = 1
        self.obs_history_offsets = ()
        self.uses_context_estimator = False
        self.base_env = self._find_base_env(env)
        self.calibration = self._calibrate_robot(train_cfg.get("calibration_steps", 100))
        self.terrain_column_names = self._terrain_column_names()
        # Play can intentionally use a single visual environment, where the
        # training-only full-range terrain/role distribution contract cannot
        # hold. Training keeps strict validation by default.
        if train_cfg.get("validate_training_contract", True):
            self._validate_training_contract()

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
            self.base_env._paper_calibrating = True
            env_ids = torch.arange(self.base_env.num_envs, device=self.base_env.device)
            original_terrain_levels = None
            if terrain.terrain_origins is not None:
                original_terrain_levels = terrain.terrain_levels.clone()
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
            stance_force = (
                torch.median(stance_forces)
                if stance_forces.numel()
                else torch.tensor(10.0, device=self.device)
            )
            # Simulation contact labels should retain light TIP touches. A small
            # morphology-calibrated threshold avoids the former ~29-N dead zone.
            contact_force_on = torch.clamp(0.03 * stance_force, 1.0, 5.0)

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
            self.base_env._paper_calibrating = False
            if original_terrain_levels is not None:
                terrain.terrain_levels.copy_(original_terrain_levels)
                terrain.env_origins[:] = terrain.terrain_origins[
                    terrain.terrain_levels, terrain.terrain_types
                ]
                self.base_env.scene.env_origins[:] = terrain.env_origins
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
        standard = manager._step_reward[:, standard_idx].to(self.device)
        # PaperBarrierReward computes the command-direction progress after the
        # physics step.  Add the bounded moving-only shaping to the standard
        # locomotion stream (never to the constraint/barrier stream).
        progress = getattr(self.base_env, "_paper_progress_reward", None)
        if progress is not None:
            standard = standard + progress.to(self.device)
        return standard, manager._step_reward[:, barrier_idx].to(self.device)

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

    def _validate_training_contract(self) -> None:
        """Fail before learning if the R2 rough-trot task silently violates its contract."""
        terrain = self.base_env.scene.terrain
        levels = terrain.terrain_levels
        if levels.numel() == 0 or torch.unique(levels).numel() < 2:
            raise RuntimeError(
                "PaperBarrier terrain levels are not distributed after calibration; "
                "expected full-range rough-trot assignments."
            )
        if torch.any(levels < 0) or torch.any(levels >= terrain.max_terrain_level):
            raise RuntimeError("PaperBarrier terrain levels fall outside the generated row range.")

        expected_columns = {
            "flat": 4,
            "bumpy": 4,
            "slope_up": 2,
            "slope_down": 2,
            "stairs_up": 2,
            "stairs_down": 2,
        }
        actual_columns = dict(Counter(self.terrain_column_names))
        if actual_columns != expected_columns:
            raise RuntimeError(
                f"PaperBarrier terrain-column contract mismatch: {actual_columns} != {expected_columns}."
            )

        asset = self.base_env.scene["robot"]
        legs_actuator = asset.actuators.get("legs")
        if legs_actuator is None:
            raise RuntimeError("PaperBarrier requires the R2 'legs' actuator group.")
        effort_limits = legs_actuator.effort_limit
        if not torch.isfinite(effort_limits).all() or not torch.all(effort_limits > 0.0):
            raise RuntimeError("PaperBarrier received non-positive or non-finite R2 effort limits.")
        unique_effort_limits = torch.unique(effort_limits).sort().values
        expected_effort_limits = torch.tensor((120.0, 320.0), device=effort_limits.device)
        if unique_effort_limits.shape != expected_effort_limits.shape or not torch.allclose(
            unique_effort_limits, expected_effort_limits
        ):
            raise RuntimeError(
                "PaperBarrier expected R2 effort limits {120, 320} Nm, got "
                f"{unique_effort_limits.detach().cpu().tolist()}."
            )

        contact_force_on = float(self.calibration["contact_force_on"])
        if not 1.0 <= contact_force_on <= 5.0:
            raise RuntimeError(f"PaperBarrier TIP contact threshold is invalid: {contact_force_on} N.")
        action_cfg = self.base_env.cfg.actions.joint_pos
        if not isinstance(action_cfg.scale, (int, float)) or float(action_cfg.scale) != 0.25:
            raise RuntimeError(f"PaperBarrier expected an R2 action scale of 0.25, got {action_cfg.scale}.")
        if action_cfg.clip != {".*": (-2.0, 2.0)}:
            raise RuntimeError(f"PaperBarrier expected raw action clipping [-2, 2], got {action_cfg.clip}.")
        curriculum_term = getattr(self.base_env, "_paper_curriculum_term", None)
        if curriculum_term is None:
            raise RuntimeError("PaperBarrier competence curriculum was not initialized.")
        roles = curriculum_term.roles
        if roles.shape != (self.base_env.num_envs,) or set(torch.unique(roles).cpu().tolist()) != {
            0,
            1,
            2,
        }:
            raise RuntimeError("PaperBarrier expected non-empty anchor/frontier/probe curriculum roles.")
        expected_roles = torch.remainder(
            torch.arange(self.base_env.num_envs, device=roles.device), 10
        )
        expected_roles = torch.where(
            expected_roles < 3,
            torch.zeros_like(expected_roles),
            torch.where(expected_roles < 8, torch.ones_like(expected_roles), 2 * torch.ones_like(expected_roles)),
        )
        if not torch.equal(roles, expected_roles):
            raise RuntimeError("PaperBarrier curriculum role assignment does not match the 30/50/20 contract.")
        policy_cfg = self.cfg["policy"]
        algorithm_cfg = self.cfg["algorithm"]
        if float(policy_cfg["init_noise_std"]) != 0.4 or not math.isclose(
            float(policy_cfg["logstd_max"]), math.log(0.6), rel_tol=0.0, abs_tol=1.0e-8
        ):
            raise RuntimeError("PaperBarrier expected initial/max action std 0.4/0.6.")
        if float(algorithm_cfg["learning_rate_max"]) != 1.0e-3:
            raise RuntimeError("PaperBarrier actor learning-rate cap must remain 1e-3.")
        clearance_lower_bound = 0.07 * float(self.calibration["length_scale"])
        role_fractions = [float((roles == role).float().mean()) for role in range(3)]
        print(
            "[PAPER_BARRIER] contract: "
            f"terrain_level_mean={float(levels.float().mean()):.2f}, "
            f"terrain_level_max={int(levels.max())}, columns={actual_columns}, "
            f"effort_limits_nm={unique_effort_limits.detach().cpu().tolist()}, "
            f"tip_contact_threshold_n={contact_force_on:.2f}, "
            f"clearance_lower_bound_m={clearance_lower_bound:.3f}, "
            f"curriculum_role_fractions={role_fractions}, "
            "action_scale_rad=0.25, action_clip=[-2, 2], action_std=0.4(max=0.6)"
        )

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
            "traversal_count",
            "traversal_success",
            "stationary_count",
            "stationary_success",
            "level_sum",
        )
        statistics = {name: torch.zeros((), device=self.device) for name in scalar_names}
        num_columns = len(self.terrain_column_names)
        for name in (
            "terrain_count",
            "terrain_timeout",
            "terrain_tracking_success",
            "terrain_traversal_count",
            "terrain_traversal_success",
            "terrain_length_sum",
        ):
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
        episode_traversal_steps: torch.Tensor,
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
        traversal_episode = episode_traversal_steps[done_ids] >= 0.5 * lengths
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
        statistics["traversal_count"] += traversal_episode.float().sum()
        statistics["traversal_success"] += (tracking_success & traversal_episode).float().sum()
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
        statistics["terrain_traversal_count"] += torch.bincount(
            terrain_types,
            weights=traversal_episode.float(),
            minlength=len(self.terrain_column_names),
        )
        statistics["terrain_traversal_success"] += torch.bincount(
            terrain_types,
            weights=(tracking_success & traversal_episode).float(),
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
        traversal_count = torch.clamp(statistics["traversal_count"], min=1.0)
        metrics = {
            "Episode/completed_count": float(statistics["count"]),
            "Episode/timeout_rate": float(statistics["timeout"] / count),
            "Episode/early_termination_rate": float(1.0 - statistics["timeout"] / count),
            "Episode/tracking_success_rate": float(statistics["tracking_success"] / count),
            "Episode/moving_tracking_success_rate": float(statistics["moving_success"] / moving_count),
            "Episode/traversal_tracking_success_rate": float(
                statistics["traversal_success"] / traversal_count
            ),
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
            terrain_traversal_count = torch.clamp(
                statistics["terrain_traversal_count"][indices].sum(), min=1.0
            )
            metrics[f"TerrainSuccess/{terrain_name}_moving_tracking_rate"] = float(
                statistics["terrain_traversal_success"][indices].sum()
                / terrain_traversal_count
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
        episode_traversal_steps = torch.zeros(self.env.num_envs, device=self.device)
        terrain = self.base_env.scene.terrain
        episode_terrain_type = terrain.terrain_types.to(self.device).clone()
        episode_terrain_level = terrain.terrain_levels.to(self.device).clone()
        total_steps = 0

        for iteration in range(self.current_learning_iteration, num_learning_iterations):
            start = time.time()
            self.base_env._paper_learning_iteration = iteration
            # Keep rollout aggregates on-device.  Calling ``float(tensor)`` in
            # the inner loop forces a CUDA synchronisation four times per
            # environment step, which is especially expensive for the long
            # PaperBarrier horizon.  Convert to Python scalars only once after
            # the rollout has completed.
            standard_sum = torch.zeros((), device=self.device)
            barrier_sum = torch.zeros((), device=self.device)
            training_standard_sum = torch.zeros((), device=self.device)
            training_barrier_sum = torch.zeros((), device=self.device)
            rollout_metric_sums = {}
            rollout_metric_counts = {}
            episode_statistics = self._new_episode_statistics()
            with torch.inference_mode():
                for step_idx in range(self.num_steps_per_env):
                    transition = self.alg.act(obs, commands, targets)
                    next_obs, _, dones, infos = self.env.step(transition["actions"])
                    raw_standard_rewards, raw_barrier_rewards = self._raw_reward_streams()
                    dones = dones.to(self.device)
                    dones_float = dones.float()
                    time_outs = infos.get("time_outs")
                    if time_outs is None:
                        time_outs = torch.zeros_like(dones_float)
                    else:
                        time_outs = time_outs.to(self.device).float()
                    training_standard_rewards = (
                        raw_standard_rewards + self.alg.gamma * transition["standard_values"] * time_outs
                    )
                    training_barrier_rewards = (
                        raw_barrier_rewards + self.alg.gamma * transition["barrier_values"] * time_outs
                    )
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
                            "standard_rewards": training_standard_rewards,
                            "barrier_rewards": training_barrier_rewards,
                            "dones": dones_float,
                            "standard_values": transition["standard_values"],
                            "barrier_values": transition["barrier_values"],
                        }
                    )
                    standard_sum += raw_standard_rewards.mean()
                    barrier_sum += raw_barrier_rewards.mean()
                    training_standard_sum += training_standard_rewards.mean()
                    training_barrier_sum += training_barrier_rewards.mean()
                    if step_idx % self.diagnostics_interval == 0:
                        self._accumulate_rollout_diagnostics(rollout_metric_sums, rollout_metric_counts)
                    barrier_metrics = getattr(self.base_env, "_paper_barrier_metrics", {})
                    episode_xy_error += barrier_metrics["Tracking/lin_vel_xy_max_abs_error"].to(self.device)
                    episode_yaw_error += barrier_metrics["Tracking/yaw_rate_error"].to(self.device)
                    episode_moving_steps += barrier_metrics["Tracking/is_moving_command"].to(self.device)
                    episode_traversal_steps += barrier_metrics["Tracking/is_traversal_command"].to(
                        self.device
                    )
                    episode_reward += raw_standard_rewards + raw_barrier_rewards
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
                            episode_traversal_steps,
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
                        episode_traversal_steps[done_ids] = 0.0
                        episode_terrain_type[done_ids] = terrain.terrain_types[done_ids].to(self.device)
                        episode_terrain_level[done_ids] = terrain.terrain_levels[done_ids].to(self.device)
                    obs = next_obs.to(self.device)
                    commands = infos["observations"]["commands"].to(self.device)
                    targets = infos["observations"]["estimator_target"].to(self.device)

                self.alg.compute_returns(obs, commands, targets)
            episode_metrics = self._finalize_episode_statistics(episode_statistics)
            self.alg.set_locomotion_success(
                episode_metrics.get("Episode/traversal_tracking_success_rate", 0.0)
            )
            metrics = self.alg.update()
            total_steps += self.env.num_envs * self.num_steps_per_env
            elapsed = time.time() - start
            mean_standard = float((standard_sum / self.num_steps_per_env).detach())
            mean_barrier = float((barrier_sum / self.num_steps_per_env).detach())
            mean_training_standard = float((training_standard_sum / self.num_steps_per_env).detach())
            mean_training_barrier = float((training_barrier_sum / self.num_steps_per_env).detach())
            configured_max_difficulty = 1.0
            rollout_metrics = self._finalize_rollout_diagnostics(rollout_metric_sums, rollout_metric_counts)
            rollout_metrics["Terrain/actual_mean_level"] = float(terrain.terrain_levels.float().mean())
            rollout_metrics["Terrain/actual_max_level"] = float(terrain.terrain_levels.max())
            curriculum_term = getattr(self.base_env, "_paper_curriculum_term", None)
            if curriculum_term is not None:
                frontier = curriculum_term.roles == 1
                rollout_metrics["Curriculum/frontier_mean_competence"] = float(
                    curriculum_term.competence[frontier].float().mean()
                )
                rollout_metrics["Curriculum/frontier_last_success_rate"] = float(
                    curriculum_term.last_success[frontier].float().mean()
                )
                rollout_metrics["Curriculum/anchor_fraction"] = float(
                    (curriculum_term.roles == 0).float().mean()
                )
                rollout_metrics["Curriculum/probe_fraction"] = float(
                    (curriculum_term.roles == 2).float().mean()
                )
            self._log(
                iteration,
                total_steps,
                elapsed,
                mean_standard,
                mean_barrier,
                mean_training_standard,
                mean_training_barrier,
                configured_max_difficulty,
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
        training_standard,
        training_barrier,
        configured_max_difficulty,
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
            f"  training_reward_per_step  : {training_standard + training_barrier:.4f}\n"
            f"  mean_episode_reward       : {mean_episode_reward:.3f}\n"
            f"  mean_episode_length       : {mean_episode_length:.1f}\n"
            f"  terrain_max_difficulty    : {configured_max_difficulty:.3f}\n"
            f"  estimator_loss            : {optimization_metrics['estimator_loss']:.4f}\n"
            f"  policy_loss               : {optimization_metrics['policy_loss']:.4f}"
        )
        print(
            "[PAPER_BARRIER_DIAG]\n"
            f"  timeout_rate              : {episode_metrics['Episode/timeout_rate']:.3f}\n"
            f"  tracking_success_rate     : {episode_metrics['Episode/tracking_success_rate']:.3f}\n"
            f"  moving_tracking_success   : {episode_metrics['Episode/moving_tracking_success_rate']:.3f}\n"
            f"  traversal_success         : {episode_metrics['Episode/traversal_tracking_success_rate']:.3f}\n"
            f"  lin_vel_xy_error          : {rollout_metrics.get('Tracking/lin_vel_xy_error', 0.0):.3f}\n"
            f"  actual_xy_speed           : {rollout_metrics.get('Tracking/actual_xy_speed', 0.0):.3f}\n"
            f"  projected_speed           : {rollout_metrics.get('Tracking/projected_speed', 0.0):.3f}\n"
            f"  progress_reward           : {rollout_metrics.get('Tracking/progress_reward', 0.0):.3f}\n"
            f"  underspeed_penalty        : {rollout_metrics.get('Tracking/underspeed_penalty', 0.0):.3f}\n"
            f"  swing_clearance_m         : {rollout_metrics.get('Foot/mean_swing_clearance', 0.0):.3f}\n"
            f"  body_contact_rate         : {rollout_metrics.get('Contact/body_rate', 0.0):.4f}\n"
            f"  persistent_calf_rate      : {rollout_metrics.get('Contact/calf_persistent_rate', 0.0):.3f}\n"
            f"  gait_violation_rate       : {rollout_metrics.get('Diagnostics/gait_violation', 0.0):.3f}\n"
            f"  clearance_violation_rate  : {rollout_metrics.get('Diagnostics/clearance_violation', 0.0):.3f}\n"
            f"  actual_mean_terrain_level : {rollout_metrics.get('Terrain/actual_mean_level', 0.0):.2f}\n"
            f"  frontier_competence       : {rollout_metrics.get('Curriculum/frontier_mean_competence', 0.0):.2f}\n"
            f"  mean_kl                   : {optimization_metrics.get('mean_kl', 0.0):.5f}\n"
            f"  learning_rate             : {optimization_metrics.get('learning_rate', 0.0):.6f}"
        )
        if self.writer is None:
            return
        self.writer.add_scalar("Reward/paper_standard_per_step", standard, iteration)
        self.writer.add_scalar("Reward/paper_barrier_per_step", barrier, iteration)
        self.writer.add_scalar("TrainingReward/paper_standard_per_step", training_standard, iteration)
        self.writer.add_scalar("TrainingReward/paper_barrier_per_step", training_barrier, iteration)
        self.writer.add_scalar("Train/mean_episode_reward", mean_episode_reward, iteration)
        self.writer.add_scalar("Train/mean_episode_length", mean_episode_length, iteration)
        self.writer.add_scalar(
            "Terrain/configured_max_difficulty", configured_max_difficulty, iteration
        )
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
                "curriculum_state_dict": (
                    self.base_env._paper_curriculum_term.state_dict()
                    if hasattr(self.base_env, "_paper_curriculum_term")
                    else None
                ),
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
        curriculum_state = checkpoint.get("curriculum_state_dict")
        curriculum_term = getattr(self.base_env, "_paper_curriculum_term", None)
        if curriculum_state is not None and curriculum_term is not None:
            curriculum_term.load_state_dict(curriculum_state)
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
