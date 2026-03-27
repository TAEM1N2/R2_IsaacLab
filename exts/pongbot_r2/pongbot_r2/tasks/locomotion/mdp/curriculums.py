from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


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
        term_cfg = env.reward_manager.get_term_cfg(term_name)

        if term_name not in env._reward_term_base_weights:
            env._reward_term_base_weights[term_name] = term_cfg.weight

        term_cfg.weight = env._reward_term_base_weights[term_name] * current_scale
        env.reward_manager.set_term_cfg(term_name, term_cfg)

    return {"penalty_scale": current_scale}
