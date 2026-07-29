"""Curriculum terms for PongBot locomotion training.

@version 0.0.3
@update 2026-07-13: Add guarded competence/challenge replay for the fifth experiment.
@update 2026-07-13: Add robust per-environment obstacle progression for the fourth experiment.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import CurriculumTermCfg


class TerrainFrontierCurriculum(ManagerTermBase):
    """Sample obstacle levels around a terrain-specific competence frontier."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        names = tuple(cfg.params["terrain_type_names"])
        initial_frontiers = tuple(cfg.params["initial_frontiers"])
        if len(names) != len(initial_frontiers):
            raise ValueError("terrain_type_names and initial_frontiers must have the same length")

        probabilities = torch.as_tensor(
            cfg.params["level_probabilities"], device=env.device, dtype=torch.float32
        )
        if probabilities.numel() != len(cfg.params["level_offsets"]):
            raise ValueError("level_offsets and level_probabilities must have the same length")
        if torch.any(probabilities < 0.0) or float(probabilities.sum().item()) <= 0.0:
            raise ValueError("level_probabilities must be non-negative with a positive sum")

        self.terrain_type_names = names
        self.frontiers = {name: int(level) for name, level in zip(names, initial_frontiers)}
        self.success_counts = {name: 0 for name in names}
        self.failure_counts = {name: 0 for name in names}
        self.promotion_counts = {name: 0 for name in names}
        self.demotion_counts = {name: 0 for name in names}
        self.last_update_steps = {name: -int(cfg.params["update_cooldown_steps"]) for name in names}
        self.level_probabilities = probabilities / probabilities.sum()
        self._column_ids: dict[str, tuple[int, ...]] | None = None

    def _resolve_column_ids(self, env: ManagerBasedRLEnv) -> dict[str, tuple[int, ...]]:
        if self._column_ids is None:
            from .events import _terrain_column_ids_for_names

            generator = env.cfg.scene.terrain.terrain_generator
            self._column_ids = {
                name: tuple(_terrain_column_ids_for_names(generator, (name,))) for name in self.terrain_type_names
            }
            missing = [name for name, column_ids in self._column_ids.items() if not column_ids]
            if missing:
                raise ValueError(f"Frontier terrain names did not resolve to terrain columns: {missing}")
        return self._column_ids

    @staticmethod
    def _base_contact_mask(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
        try:
            values = env.termination_manager.get_term("base_contact")
        except (AttributeError, KeyError):
            return torch.zeros(env_ids.numel(), device=env.device, dtype=torch.bool)
        return values[env_ids].to(device=env.device, dtype=torch.bool)

    def _publish_metrics(self, env: ManagerBasedRLEnv) -> None:
        metrics = {}
        for name in self.terrain_type_names:
            trials = self.success_counts[name] + self.failure_counts[name]
            success_rate = self.success_counts[name] / trials if trials > 0 else float("nan")
            metrics[name] = {
                "frontier_level": float(self.frontiers[name]),
                "success_rate": float(success_rate),
                "eligible_episodes": float(trials),
                "promotion_count": float(self.promotion_counts[name]),
                "demotion_count": float(self.demotion_counts[name]),
            }
        env._pongbot_frontier_metrics = metrics

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        terrain_type_names: tuple[str, ...],
        initial_frontiers: tuple[int, ...],
        progress_success: float,
        progress_failure: float,
        command_threshold: float,
        velocity_error_threshold: float,
        minimum_samples: int,
        promotion_rate: float,
        demotion_rate: float,
        update_cooldown_steps: int,
        level_offsets: tuple[int, ...],
        level_probabilities: tuple[float, ...],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        del terrain_type_names, initial_frontiers, level_probabilities
        terrain = env.scene.terrain
        if terrain.terrain_origins is None:
            return torch.tensor(0.0, device=env.device)

        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).view(-1)
        if env_ids.numel() == 0:
            return torch.mean(terrain.terrain_levels.float())

        asset = env.scene[asset_cfg.name]
        command = env.command_manager.get_command("base_velocity")[env_ids, :2]
        command_norm = torch.linalg.vector_norm(command, dim=1)
        command_direction = command / command_norm.unsqueeze(1).clamp(min=1.0e-6)
        displacement = asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
        forward_progress = torch.sum(displacement * command_direction, dim=1)
        base_contact = self._base_contact_mask(env, env_ids)

        error_vel_xy = None
        try:
            error_vel_xy = env.command_manager.get_term("base_velocity").metrics.get("error_vel_xy")
        except (AttributeError, KeyError):
            pass
        if error_vel_xy is None:
            velocity_ok = torch.ones(env_ids.numel(), device=env.device, dtype=torch.bool)
        else:
            velocity_ok = error_vel_xy[env_ids] <= velocity_error_threshold

        terrain_types = terrain.terrain_types[env_ids]
        column_ids_by_name = self._resolve_column_ids(env)
        obstacle_mask = torch.zeros(env_ids.numel(), device=env.device, dtype=torch.bool)
        common_step = int(env.common_step_counter)

        for name in self.terrain_type_names:
            column_ids = torch.as_tensor(column_ids_by_name[name], device=env.device, dtype=torch.long)
            type_mask = torch.any(terrain_types.unsqueeze(1) == column_ids.unsqueeze(0), dim=1)
            obstacle_mask |= type_mask
            if not torch.any(type_mask):
                continue

            moving = command_norm > command_threshold
            success = type_mask & moving & (forward_progress > progress_success) & velocity_ok & ~base_contact
            failure = type_mask & moving & ((forward_progress < progress_failure) | base_contact)
            self.success_counts[name] += int(success.sum().item())
            self.failure_counts[name] += int(failure.sum().item())

            trials = self.success_counts[name] + self.failure_counts[name]
            cooldown_ready = common_step - self.last_update_steps[name] >= update_cooldown_steps
            if trials >= minimum_samples and cooldown_ready:
                success_rate = self.success_counts[name] / trials
                if success_rate >= promotion_rate:
                    self.frontiers[name] += 1
                    self.promotion_counts[name] += 1
                elif success_rate <= demotion_rate:
                    self.frontiers[name] -= 1
                    self.demotion_counts[name] += 1
                self.frontiers[name] = int(min(max(self.frontiers[name], 0), terrain.max_terrain_level - 1))
                self.success_counts[name] = 0
                self.failure_counts[name] = 0
                self.last_update_steps[name] = common_step

            local_ids = torch.nonzero(type_mask, as_tuple=False).flatten()
            sampled_choice = torch.multinomial(
                self.level_probabilities,
                num_samples=local_ids.numel(),
                replacement=True,
            )
            offsets = torch.as_tensor(level_offsets, device=env.device, dtype=torch.long)
            sampled_levels = self.frontiers[name] + offsets[sampled_choice]
            sampled_levels = torch.clamp(sampled_levels, 0, terrain.max_terrain_level - 1)
            selected_env_ids = env_ids[local_ids]
            terrain.terrain_levels[selected_env_ids] = sampled_levels
            terrain.env_origins[selected_env_ids] = terrain.terrain_origins[
                sampled_levels, terrain.terrain_types[selected_env_ids]
            ]

        regular_local_ids = torch.nonzero(~obstacle_mask, as_tuple=False).flatten()
        if regular_local_ids.numel() > 0:
            regular_env_ids = env_ids[regular_local_ids]
            distance = torch.linalg.vector_norm(displacement[regular_local_ids], dim=1)
            terrain_size_x = float(env.cfg.scene.terrain.terrain_generator.size[0])
            move_up = distance > terrain_size_x / 2.0
            move_down = distance < command_norm[regular_local_ids] * env.max_episode_length_s * 0.5
            move_down &= ~move_up
            terrain.update_env_origins(regular_env_ids, move_up, move_down)

        self._publish_metrics(env)
        return torch.mean(terrain.terrain_levels.float())


def obstacle_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    command_threshold: float = 0.2,
    promote_distance: float = 3.0,
    demote_progress_ratio: float = 0.35,
    minimum_demotion_distance: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Independently update terrain levels from traversal distance."""
    terrain = env.scene.terrain
    if terrain.terrain_origins is None:
        return torch.tensor(0.0, device=env.device)
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).view(-1)
    if env_ids.numel() == 0:
        return torch.mean(terrain.terrain_levels.float())
    asset = env.scene[asset_cfg.name]
    command_speed = torch.linalg.vector_norm(
        env.command_manager.get_command(command_name)[env_ids, :2], dim=1
    )
    distance = torch.linalg.vector_norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    moving = command_speed > command_threshold
    demotion_limit = torch.clamp(
        command_speed * env.max_episode_length_s * demote_progress_ratio,
        min=minimum_demotion_distance,
    )
    move_up = moving & (distance > promote_distance)
    move_down = moving & ~move_up & (distance < demotion_limit)
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


class GuardedObstacleCurriculum(ManagerTermBase):
    """Keep obstacle challenges without letting their failures demote competence."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.is_challenge = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.competence_levels = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self._column_ids = None

    def reset(self, env_ids=None) -> None:
        # ``compute()`` assigns the role for the next episode immediately
        # before CurriculumManager.reset().  Preserve that assignment here.
        del env_ids

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        terrain_type_names: tuple[str, ...],
        command_threshold: float,
        promote_distance: float,
        demote_progress_ratio: float,
        minimum_demotion_distance: float,
        challenge_levels: tuple[int, ...],
        challenge_probabilities: tuple[float, ...],
        warmup_steps: int,
        initial_challenge_ratio: float,
        final_challenge_ratio: float,
        command_name: str = "base_velocity",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        terrain = env.scene.terrain
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).view(-1)
        if env_ids.numel() == 0 or terrain.terrain_origins is None:
            return torch.mean(terrain.terrain_levels.float())
        if self._column_ids is None:
            from .events import _terrain_column_ids_for_names
            generator = env.cfg.scene.terrain.terrain_generator
            self._column_ids = tuple(_terrain_column_ids_for_names(generator, terrain_type_names))

        asset = env.scene[asset_cfg.name]
        speed = torch.linalg.vector_norm(
            env.command_manager.get_command(command_name)[env_ids, :2], dim=1
        )
        distance = torch.linalg.vector_norm(
            asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
        )
        terrain_types = terrain.terrain_types[env_ids]
        obstacle = torch.zeros(env_ids.numel(), dtype=torch.bool, device=env.device)
        for column_id in self._column_ids:
            obstacle |= terrain_types == column_id

        previous_challenge = self.is_challenge[env_ids]
        uninitialized = self.competence_levels[env_ids] < 0
        self.competence_levels[env_ids[uninitialized]] = terrain.terrain_levels[env_ids[uninitialized]]
        competence = obstacle & ~previous_challenge & (speed > command_threshold)
        demotion_limit = torch.clamp(
            speed * env.max_episode_length_s * demote_progress_ratio,
            min=minimum_demotion_distance,
        )
        move_up = competence & (distance > promote_distance)
        move_down = competence & ~move_up & (distance < demotion_limit)
        regular = ~obstacle
        move_up |= regular & (distance > promote_distance)
        move_down |= regular & ~move_up & (distance < demotion_limit) & (speed > command_threshold)
        terrain.update_env_origins(env_ids, move_up, move_down)
        evaluated = ~previous_challenge
        self.competence_levels[env_ids[evaluated]] = terrain.terrain_levels[env_ids[evaluated]]
        returning = previous_challenge
        if torch.any(returning):
            selected = env_ids[returning]
            restored = self.competence_levels[selected].clamp(0, terrain.max_terrain_level - 1)
            terrain.terrain_levels[selected] = restored
            terrain.env_origins[selected] = terrain.terrain_origins[restored, terrain.terrain_types[selected]]

        progress = min(float(env.common_step_counter) / max(warmup_steps, 1), 1.0)
        ratio = initial_challenge_ratio + progress * (final_challenge_ratio - initial_challenge_ratio)
        next_challenge = obstacle & (torch.rand(env_ids.numel(), device=env.device) < ratio)
        self.is_challenge[env_ids] = next_challenge
        challenge_local = torch.nonzero(next_challenge, as_tuple=False).flatten()
        if challenge_local.numel() > 0:
            probabilities = torch.as_tensor(challenge_probabilities, device=env.device, dtype=torch.float32)
            probabilities /= probabilities.sum()
            choices = torch.multinomial(probabilities, challenge_local.numel(), replacement=True)
            levels = torch.as_tensor(challenge_levels, device=env.device, dtype=torch.long)[choices]
            levels.clamp_(0, terrain.max_terrain_level - 1)
            selected = env_ids[challenge_local]
            terrain.terrain_levels[selected] = levels
            terrain.env_origins[selected] = terrain.terrain_origins[levels, terrain.terrain_types[selected]]

        env._pongbot_challenge_ratio = float(next_challenge.float().mean().item())
        return torch.mean(terrain.terrain_levels.float())


def modify_event_parameter(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    param_name: str,
    value: Any | SceneEntityCfg,
    num_steps: int,
) -> torch.Tensor:
    """Curriculum that modifies a parameter of an event at a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the event term.
        param_name: The name of the event term parameter.
        value: The new value for the event term parameter.
        num_steps: The number of steps after which the change should be applied.

    Returns:
        torch.Tensor: Whether the parameter has already been modified or not.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.event_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.params[param_name] = value
        env.event_manager.set_term_cfg(term_name, term_cfg)
        return torch.ones(1)
    return torch.zeros(1)


def disable_termination(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    num_steps: int,
) -> torch.Tensor:
    """Curriculum that modifies the push velocity range at a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the termination term.
        num_steps: The number of steps after which the change should be applied.

    Returns:
        torch.Tensor: Whether the parameter has already been modified or not.
    """
    env.command_manager.num_envs
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.termination_manager.get_term_cfg(term_name)
        # Remove term settings
        term_cfg.params = dict()
        term_cfg.func = lambda env: torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env.termination_manager.set_term_cfg(term_name, term_cfg)
        return torch.ones(1)
    return torch.zeros(1)


def ramp_reward_terms_by_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_names: Sequence[str],
    start_scale: float,
    end_scale: float,
    start_step: int,
    end_step: int,
) -> dict[str, float]:
    """Linearly ramp reward term weights by a shared scale factor.

    This is mainly useful for gradually enabling penalty terms so that early exploration
    is not dominated by strong negative shaping rewards.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_names: Reward term names to scale.
        start_scale: Scale applied at ``start_step``.
        end_scale: Scale applied at ``end_step``.
        start_step: Global environment step at which the ramp starts.
        end_step: Global environment step at which the ramp finishes.

    Returns:
        Dictionary with the current scale for curriculum logging.
    """
    del env_ids

    if end_step <= start_step:
        raise ValueError(f"'end_step' ({end_step}) must be greater than 'start_step' ({start_step}).")

    progress = (env.common_step_counter - start_step) / float(end_step - start_step)
    progress = float(torch.clamp(torch.tensor(progress), 0.0, 1.0).item())
    current_scale = start_scale + (end_scale - start_scale) * progress

    if not hasattr(env, "_reward_term_base_weights"):
        env._reward_term_base_weights = {}

    for term_name in term_names:
        try:
            term_cfg = env.reward_manager.get_term_cfg(term_name)
        except ValueError:
            # Allow play / config drift to continue even if an old reward term name
            # remains in the curriculum list.
            continue

        if term_name not in env._reward_term_base_weights:
            env._reward_term_base_weights[term_name] = term_cfg.weight

        term_cfg.weight = env._reward_term_base_weights[term_name] * current_scale
        env.reward_manager.set_term_cfg(term_name, term_cfg)

    return {"penalty_scale": current_scale}


def terrain_levels_stair_progress(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stair_terrain_type_names: tuple[str, ...] = ("one_way_stairs", "one_way_wide_step"),
    stair_progress_up: float = 2.5,
    stair_progress_down: float = 0.6,
) -> torch.Tensor:
    """Use forward stair progress for stair terrains and velocity distance elsewhere."""
    from .events import _terrain_column_ids_for_names

    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    device = env.device
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
    command = env.command_manager.get_command("base_velocity")

    env_origins = env.scene.env_origins[env_ids]
    root_pos_w = asset.data.root_pos_w[env_ids]
    distance = torch.norm(root_pos_w[:, :2] - env_origins[:, :2], dim=1)

    terrain_generator = getattr(getattr(env.cfg.scene, "terrain", None), "terrain_generator", None)
    terrain_size_x = float(getattr(terrain_generator, "size", (8.0, 8.0))[0])
    move_up = distance > terrain_size_x / 2.0
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5

    terrain_types = getattr(terrain, "terrain_types", None)
    stair_type_ids = _terrain_column_ids_for_names(terrain_generator, stair_terrain_type_names)
    if terrain_types is not None and stair_type_ids:
        terrain_types = terrain_types.to(device=device, dtype=torch.long).view(-1)
        if terrain_types.numel() >= env.num_envs:
            terrain_type_ids = terrain_types[env_ids]
            stair_ids = torch.tensor(stair_type_ids, device=device, dtype=torch.long)
            stair_mask = torch.any(terrain_type_ids.unsqueeze(1) == stair_ids.unsqueeze(0), dim=1)
            if torch.any(stair_mask):
                stair_progress = root_pos_w[:, 0] - env_origins[:, 0]
                move_up = torch.where(stair_mask, stair_progress > stair_progress_up, move_up)
                move_down = torch.where(stair_mask, stair_progress < stair_progress_down, move_down)

    move_down *= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def ramp_payload_mass_range(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_names: Sequence[str],
    start_mass_range: tuple[float, float],
    end_mass_range: tuple[float, float],
    start_step: int,
    end_step: int,
) -> dict[str, float]:
    """Linearly ramp payload mass range for one or more payload events."""
    del env_ids

    if end_step <= start_step:
        raise ValueError(f"'end_step' ({end_step}) must be greater than 'start_step' ({start_step}).")

    progress = (env.common_step_counter - start_step) / float(end_step - start_step)
    progress = float(torch.clamp(torch.tensor(progress), 0.0, 1.0).item())
    current_mass_range_float = tuple(
        start + (end - start) * progress for start, end in zip(start_mass_range, end_mass_range)
    )
    current_mass_range = (
        float(int(torch.ceil(torch.tensor(current_mass_range_float[0], dtype=torch.float32)).item())),
        float(int(torch.floor(torch.tensor(current_mass_range_float[1], dtype=torch.float32)).item())),
    )
    if current_mass_range[1] < current_mass_range[0]:
        current_mass_range = (current_mass_range[0], current_mass_range[0])

    for term_name in term_names:
        term_cfg = env.event_manager.get_term_cfg(term_name)
        term_cfg.params["payload_mass_range"] = current_mass_range
        env.event_manager.set_term_cfg(term_name, term_cfg)

    return {
        "payload_mass_min": current_mass_range[0],
        "payload_mass_max": current_mass_range[1],
        "progress": progress,
    }
