"""Dual-advantage PPO used by the barrier-style paper reproduction.

@version 0.0.5
@update 2026-07-13: Bound actor adaptation and transition from tracking discovery to balanced barrier control.
@update 2026-07-13: Separate actor, dual-critic, and estimator optimization with value-scale normalization.
"""

import torch
import torch.nn.functional as F

from rsl_rl.storage.paper_barrier_storage import PaperBarrierStorage


class PaperBarrierPPO:
    def __init__(self, actor_critic, cfg: dict, device: str):
        self.actor_critic = actor_critic
        self.cfg = cfg
        self.device = device
        self.clip_param = cfg.get("clip_param", 0.2)
        self.value_loss_coef = cfg.get("value_loss_coef", 1.0)
        self.entropy_coef = cfg.get("entropy_coef", 0.01)
        self.estimator_loss_coef = cfg.get("estimator_loss_coef", 1.0)
        self.num_learning_epochs = cfg.get("num_learning_epochs", 5)
        self.num_mini_batches = cfg.get("num_mini_batches", 4)
        self.gamma = cfg.get("gamma", 0.99)
        self.lam = cfg.get("lam", 0.95)
        self.max_grad_norm = cfg.get("max_grad_norm", 1.0)
        self.learning_rate = cfg.get("learning_rate", 1.0e-3)
        self.learning_rate_min = cfg.get("learning_rate_min", 1.0e-5)
        self.learning_rate_max = cfg.get("learning_rate_max", 1.0e-3)
        self.standard_critic_learning_rate = cfg.get(
            "standard_critic_learning_rate", self.learning_rate
        )
        self.barrier_critic_learning_rate = cfg.get(
            "barrier_critic_learning_rate", self.learning_rate
        )
        self.estimator_learning_rate = cfg.get("estimator_learning_rate", self.learning_rate)
        self.value_scale_min = cfg.get("value_scale_min", 1.0)
        self.desired_kl = cfg.get("desired_kl", 0.01)
        self.kl_upper_factor = cfg.get("kl_upper_factor", 1.5)
        self.kl_lower_factor = cfg.get("kl_lower_factor", 0.5)
        self.schedule = cfg.get("schedule", "adaptive")
        self.initial_standard_advantage_weight = cfg.get(
            "initial_standard_advantage_weight", 0.6
        )
        self.final_standard_advantage_weight = cfg.get(
            "final_standard_advantage_weight", 0.5
        )
        self.advantage_transition_success = cfg.get("advantage_transition_success", 0.3)
        self.locomotion_success = 0.0

        self.actor_parameters = tuple(actor_critic.actor.parameters()) + (actor_critic.logstd,)
        self.standard_critic_parameters = tuple(actor_critic.standard_critic.parameters())
        self.barrier_critic_parameters = tuple(actor_critic.barrier_critic.parameters())
        self.estimator_parameters = tuple(actor_critic.estimator.parameters())
        self.actor_optimizer = torch.optim.Adam(self.actor_parameters, lr=self.learning_rate)
        self.standard_critic_optimizer = torch.optim.Adam(
            self.standard_critic_parameters, lr=self.standard_critic_learning_rate
        )
        self.barrier_critic_optimizer = torch.optim.Adam(
            self.barrier_critic_parameters, lr=self.barrier_critic_learning_rate
        )
        self.estimator_optimizer = torch.optim.Adam(
            self.estimator_parameters, lr=self.estimator_learning_rate
        )
        # Compatibility alias for utilities that only inspect the actor optimizer.
        self.optimizer = self.actor_optimizer
        self.storage: PaperBarrierStorage | None = None

    def init_storage(self, num_envs, num_steps, proprio_dim, command_dim, target_dim, action_dim):
        self.storage = PaperBarrierStorage(
            num_envs, num_steps, proprio_dim, command_dim, target_dim, action_dim, self.device
        )

    def set_locomotion_success(self, success_rate: float) -> None:
        """Set the latest moving-command success used by the advantage curriculum."""
        self.locomotion_success = min(max(float(success_rate), 0.0), 1.0)

    def _advantage_weights(self) -> tuple[float, float]:
        """Return tracking-first weights that converge to the paper's balanced mixture."""
        fraction = min(
            self.locomotion_success / max(self.advantage_transition_success, 1.0e-6),
            1.0,
        )
        standard_weight = self.initial_standard_advantage_weight + fraction * (
            self.final_standard_advantage_weight - self.initial_standard_advantage_weight
        )
        return standard_weight, 1.0 - standard_weight

    @torch.no_grad()
    def act(self, proprio: torch.Tensor, commands: torch.Tensor, targets: torch.Tensor):
        actor_input, estimate = self.actor_critic.actor_input(proprio, commands, detach_estimator=True)
        critic_input = self.actor_critic.critic_input(proprio, commands, targets)
        actions = self.actor_critic.act(actor_input)
        standard_values, barrier_values = self.actor_critic.evaluate(critic_input)
        return {
            "actions": actions,
            "estimated_targets": estimate,
            "log_probs": self.actor_critic.get_actions_log_prob(actions),
            "action_mean": self.actor_critic.action_mean,
            "action_std": self.actor_critic.action_std,
            "standard_values": standard_values,
            "barrier_values": barrier_values,
        }

    @torch.no_grad()
    def compute_returns(self, proprio, commands, targets):
        critic_input = self.actor_critic.critic_input(proprio, commands, targets)
        last_standard, last_barrier = self.actor_critic.evaluate(critic_input)
        self.storage.compute_returns(last_standard, last_barrier, self.gamma, self.lam)

    def update(self) -> dict[str, float]:
        standard_value_scale = torch.clamp(
            self.storage.standard_returns[: self.storage.step].std(unbiased=False),
            min=self.value_scale_min,
        ).detach()
        barrier_value_scale = torch.clamp(
            self.storage.barrier_returns[: self.storage.step].std(unbiased=False),
            min=self.value_scale_min,
        ).detach()
        totals = {
            "policy_loss": 0.0,
            "standard_value_loss": 0.0,
            "barrier_value_loss": 0.0,
            "estimator_loss": 0.0,
            "velocity_estimator_loss": 0.0,
            "height_estimator_loss": 0.0,
            "contact_estimator_loss": 0.0,
            "entropy": 0.0,
            "mean_kl": 0.0,
            "clip_fraction": 0.0,
            "ratio_mean": 0.0,
            "standard_explained_variance": 0.0,
            "barrier_explained_variance": 0.0,
            "gradient_norm_pre_clip": 0.0,
            "actor_gradient_norm_pre_clip": 0.0,
            "standard_critic_gradient_norm_pre_clip": 0.0,
            "barrier_critic_gradient_norm_pre_clip": 0.0,
            "estimator_gradient_norm_pre_clip": 0.0,
            "standard_value_scale": 0.0,
            "barrier_value_scale": 0.0,
            "standard_advantage_weight": 0.0,
            "barrier_advantage_weight": 0.0,
            "action_std_mean": 0.0,
        }
        updates = 0
        standard_advantage_weight, barrier_advantage_weight = self._advantage_weights()
        for batch in self.storage.mini_batches(self.num_mini_batches, self.num_learning_epochs):
            actor_input = torch.cat(
                (batch["proprio"], batch["commands"], batch["estimated_targets"]), dim=1
            )
            self.actor_critic.update_distribution(actor_input)
            new_log_prob = self.actor_critic.get_actions_log_prob(batch["actions"])
            ratio = torch.exp(new_log_prob - batch["log_probs"])
            old_std = batch["action_std"]
            old_mean = batch["action_mean"]
            new_std = self.actor_critic.action_std
            new_mean = self.actor_critic.action_mean
            with torch.no_grad():
                kl = torch.sum(
                    torch.log(new_std / old_std + 1.0e-5)
                    + (torch.square(old_std) + torch.square(old_mean - new_mean))
                    / (2.0 * torch.square(new_std))
                    - 0.5,
                    dim=-1,
                ).mean()
                clip_fraction = torch.mean((torch.abs(ratio - 1.0) > self.clip_param).float())
            mixed_advantage = (
                standard_advantage_weight * batch["standard_advantages"]
                + barrier_advantage_weight * batch["barrier_advantages"]
            )
            surrogate = -mixed_advantage * ratio
            surrogate_clipped = -mixed_advantage * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            policy_loss = torch.max(surrogate, surrogate_clipped).mean()

            critic_input = self.actor_critic.critic_input(
                batch["proprio"], batch["commands"], batch["targets"]
            )
            standard_value, barrier_value = self.actor_critic.evaluate(critic_input)
            standard_value_loss = self._value_loss(
                standard_value,
                batch["standard_values"],
                batch["standard_returns"],
                standard_value_scale,
            )
            barrier_value_loss = self._value_loss(
                barrier_value,
                batch["barrier_values"],
                batch["barrier_returns"],
                barrier_value_scale,
            )
            standard_explained_variance = self._explained_variance(
                standard_value.detach(), batch["standard_returns"]
            )
            barrier_explained_variance = self._explained_variance(
                barrier_value.detach(), batch["barrier_returns"]
            )

            target = batch["targets"]
            estimate = self.actor_critic.estimator(batch["proprio"])
            velocity_loss = F.mse_loss(estimate[:, :3], target[:, :3])
            height_loss = F.mse_loss(estimate[:, 3:7], target[:, 3:7])
            contact_loss = F.binary_cross_entropy(estimate[:, 7:], target[:, 7:])
            estimator_loss = velocity_loss + height_loss + contact_loss
            entropy = self.actor_critic.entropy.mean()

            actor_loss = policy_loss - self.entropy_coef * entropy
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_parameters, self.max_grad_norm
            )
            self.actor_optimizer.step()
            self.actor_critic.clamp_logstd_()

            self.standard_critic_optimizer.zero_grad(set_to_none=True)
            (self.value_loss_coef * standard_value_loss).backward()
            standard_critic_gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.standard_critic_parameters, self.max_grad_norm
            )
            self.standard_critic_optimizer.step()

            self.barrier_critic_optimizer.zero_grad(set_to_none=True)
            (self.value_loss_coef * barrier_value_loss).backward()
            barrier_critic_gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.barrier_critic_parameters, self.max_grad_norm
            )
            self.barrier_critic_optimizer.step()

            self.estimator_optimizer.zero_grad(set_to_none=True)
            (self.estimator_loss_coef * estimator_loss).backward()
            estimator_gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.estimator_parameters, self.max_grad_norm
            )
            self.estimator_optimizer.step()

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.no_grad():
                    if kl > self.kl_upper_factor * self.desired_kl:
                        self.learning_rate = max(self.learning_rate_min, self.learning_rate / 1.5)
                    elif 0.0 < kl < self.kl_lower_factor * self.desired_kl:
                        self.learning_rate = min(self.learning_rate_max, self.learning_rate * 1.5)
                    for group in self.actor_optimizer.param_groups:
                        group["lr"] = self.learning_rate

            values = (
                policy_loss,
                standard_value_loss,
                barrier_value_loss,
                estimator_loss,
                velocity_loss,
                height_loss,
                contact_loss,
                entropy,
                kl,
                clip_fraction,
                ratio.mean(),
                standard_explained_variance,
                barrier_explained_variance,
                actor_gradient_norm,
                actor_gradient_norm,
                standard_critic_gradient_norm,
                barrier_critic_gradient_norm,
                estimator_gradient_norm,
                standard_value_scale,
                barrier_value_scale,
                torch.as_tensor(standard_advantage_weight, device=self.device),
                torch.as_tensor(barrier_advantage_weight, device=self.device),
                new_std.mean(),
            )
            for key, value in zip(totals, values):
                totals[key] += float(value.detach())
            updates += 1

        self.storage.clear()
        return {key: value / max(1, updates) for key, value in totals.items()} | {
            "learning_rate": self.learning_rate,
            "standard_critic_learning_rate": self.standard_critic_learning_rate,
            "barrier_critic_learning_rate": self.barrier_critic_learning_rate,
            "estimator_learning_rate": self.estimator_learning_rate,
        }

    def optimizer_state_dict(self) -> dict[str, dict]:
        """Return every independent optimizer state for exact continuation."""
        return {
            "actor": self.actor_optimizer.state_dict(),
            "standard_critic": self.standard_critic_optimizer.state_dict(),
            "barrier_critic": self.barrier_critic_optimizer.state_dict(),
            "estimator": self.estimator_optimizer.state_dict(),
        }

    def load_optimizer_state_dict(self, state_dicts: dict[str, dict]) -> None:
        """Restore optimizer states saved by :meth:`optimizer_state_dict`."""
        self.actor_optimizer.load_state_dict(state_dicts["actor"])
        self.standard_critic_optimizer.load_state_dict(state_dicts["standard_critic"])
        self.barrier_critic_optimizer.load_state_dict(state_dicts["barrier_critic"])
        self.estimator_optimizer.load_state_dict(state_dicts["estimator"])

    def _value_loss(self, value, old_value, returns, value_scale):
        """Compute a dimensionless critic loss while retaining raw-value predictions."""
        if self.cfg.get("use_clipped_value_loss", True):
            value_clip = self.clip_param * value_scale
            clipped = old_value + torch.clamp(value - old_value, -value_clip, value_clip)
            squared_error = torch.max(torch.square(value - returns), torch.square(clipped - returns))
        else:
            squared_error = torch.square(value - returns)
        return (squared_error / torch.square(value_scale)).mean()

    @staticmethod
    def _explained_variance(value: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """Return one minus residual variance divided by target variance."""
        target_variance = torch.var(returns, unbiased=False)
        residual_variance = torch.var(returns - value, unbiased=False)
        return 1.0 - residual_variance / torch.clamp(target_variance, min=1.0e-8)
