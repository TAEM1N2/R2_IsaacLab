"""Lean rollout storage for dual-reward barrier PPO.

@version 0.0.2
@update 2026-07-13: Store rollout-time estimator outputs to keep PPO inputs stationary.
@update 2026-07-13: Add independent GAE streams without observation-history duplication.
"""

import torch


class PaperBarrierStorage:
    def __init__(self, num_envs, num_steps, proprio_dim, command_dim, target_dim, action_dim, device):
        shape = (num_steps, num_envs)
        self.proprio = torch.zeros(*shape, proprio_dim, device=device)
        self.commands = torch.zeros(*shape, command_dim, device=device)
        self.targets = torch.zeros(*shape, target_dim, device=device)
        self.estimated_targets = torch.zeros(*shape, target_dim, device=device)
        self.actions = torch.zeros(*shape, action_dim, device=device)
        self.log_probs = torch.zeros(*shape, device=device)
        self.action_mean = torch.zeros(*shape, action_dim, device=device)
        self.action_std = torch.zeros(*shape, action_dim, device=device)
        self.standard_rewards = torch.zeros(*shape, device=device)
        self.barrier_rewards = torch.zeros(*shape, device=device)
        self.dones = torch.zeros(*shape, device=device)
        self.standard_values = torch.zeros(*shape, device=device)
        self.barrier_values = torch.zeros(*shape, device=device)
        self.standard_returns = torch.zeros(*shape, device=device)
        self.barrier_returns = torch.zeros(*shape, device=device)
        self.standard_advantages = torch.zeros(*shape, device=device)
        self.barrier_advantages = torch.zeros(*shape, device=device)
        self.step = 0

    def add(self, transition: dict[str, torch.Tensor]) -> None:
        if self.step >= self.proprio.shape[0]:
            raise RuntimeError("PaperBarrierStorage overflow")
        for name, value in transition.items():
            getattr(self, name)[self.step].copy_(value)
        self.step += 1

    def compute_returns(self, last_standard, last_barrier, gamma: float, lam: float) -> None:
        gae_standard = torch.zeros_like(last_standard)
        gae_barrier = torch.zeros_like(last_barrier)
        for step in reversed(range(self.step)):
            next_standard = last_standard if step == self.step - 1 else self.standard_values[step + 1]
            next_barrier = last_barrier if step == self.step - 1 else self.barrier_values[step + 1]
            alive = 1.0 - self.dones[step]
            delta_standard = self.standard_rewards[step] + gamma * alive * next_standard - self.standard_values[step]
            delta_barrier = self.barrier_rewards[step] + gamma * alive * next_barrier - self.barrier_values[step]
            gae_standard = delta_standard + gamma * lam * alive * gae_standard
            gae_barrier = delta_barrier + gamma * lam * alive * gae_barrier
            self.standard_returns[step] = gae_standard + self.standard_values[step]
            self.barrier_returns[step] = gae_barrier + self.barrier_values[step]
        self.standard_advantages[: self.step] = self.standard_returns[: self.step] - self.standard_values[: self.step]
        self.barrier_advantages[: self.step] = self.barrier_returns[: self.step] - self.barrier_values[: self.step]
        for advantages in (self.standard_advantages, self.barrier_advantages):
            active = advantages[: self.step]
            active.sub_(active.mean()).div_(active.std(unbiased=False) + 1.0e-8)

    def mini_batches(self, num_mini_batches: int, num_epochs: int):
        total = self.step * self.proprio.shape[1]
        if total % num_mini_batches != 0:
            raise ValueError(f"Batch {total} is not divisible by {num_mini_batches}")
        flat_names = (
            "proprio", "commands", "targets", "estimated_targets", "actions", "log_probs", "action_mean", "action_std",
            "standard_values", "barrier_values", "standard_returns", "barrier_returns",
            "standard_advantages", "barrier_advantages",
        )
        flat = {name: getattr(self, name)[: self.step].flatten(0, 1) for name in flat_names}
        mini_size = total // num_mini_batches
        for _ in range(num_epochs):
            indices = torch.randperm(total, device=self.proprio.device)
            for start in range(0, total, mini_size):
                batch_ids = indices[start : start + mini_size]
                yield {name: tensor[batch_ids] for name, tensor in flat.items()}

    def clear(self) -> None:
        self.step = 0
