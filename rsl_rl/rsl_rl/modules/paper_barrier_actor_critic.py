"""Estimator-conditioned actor with two privileged critics.

@version 0.0.3
@update 2026-07-13: Support batched and unbatched estimator export inputs.
@update 2026-07-13: Expose actor input size for the existing ONNX/play utilities.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


def _activation(name: str) -> nn.Module:
    activations = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "selu": nn.SELU}
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]()


def _mlp(input_dim: int, output_dim: int, hidden_dims: list[int], activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for hidden in hidden_dims:
        layers.extend((nn.Linear(current, hidden), _activation(activation)))
        current = hidden
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class PaperStateEstimator(nn.Module):
    """Estimate velocity, terrain-relative foot height, and contact probability."""

    def __init__(self, input_dim: int, hidden_dims: list[int], activation: str = "elu"):
        super().__init__()
        self.num_input_dim = input_dim
        self.network = _mlp(input_dim, 11, hidden_dims, activation)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        unbatched = observations.dim() == 1
        if unbatched:
            observations = observations.unsqueeze(0)
        raw = self.network(observations)
        estimate = torch.cat((raw[:, :7], torch.sigmoid(raw[:, 7:])), dim=1)
        return estimate.squeeze(0) if unbatched else estimate

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        return self(observations)


class PaperBarrierActorCritic(nn.Module):
    """Gaussian actor and independent standard/barrier value functions."""

    is_recurrent = False
    is_sequence = False
    is_vae = False

    def __init__(
        self,
        proprio_dim: int,
        command_dim: int,
        target_dim: int,
        num_actions: int,
        actor_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        estimator_hidden_dims: list[int],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        logstd_min: float = -5.0,
        logstd_max: float = 0.0,
    ):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.command_dim = command_dim
        self.target_dim = target_dim
        self.estimator = PaperStateEstimator(proprio_dim, estimator_hidden_dims, activation)
        actor_input_dim = proprio_dim + command_dim + target_dim
        critic_input_dim = proprio_dim + command_dim + target_dim
        self.num_actor_obs = actor_input_dim
        self.actor = _mlp(actor_input_dim, num_actions, actor_hidden_dims, activation)
        self.standard_critic = _mlp(critic_input_dim, 1, critic_hidden_dims, activation)
        self.barrier_critic = _mlp(critic_input_dim, 1, critic_hidden_dims, activation)
        self.logstd = nn.Parameter(torch.full((num_actions,), float(torch.log(torch.tensor(init_noise_std)))))
        self.logstd_min = logstd_min
        self.logstd_max = logstd_max
        self.distribution: Normal | None = None
        Normal.set_default_validate_args = False

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def actor_input(self, proprio: torch.Tensor, commands: torch.Tensor, detach_estimator: bool = True):
        estimate = self.estimator(proprio)
        if detach_estimator:
            estimate = estimate.detach()
        return torch.cat((proprio, commands, estimate), dim=1), estimate

    @staticmethod
    def critic_input(proprio: torch.Tensor, commands: torch.Tensor, target: torch.Tensor):
        return torch.cat((proprio, commands, target), dim=1)

    def update_distribution(self, actor_input: torch.Tensor) -> None:
        mean = self.actor(actor_input)
        std = torch.exp(torch.clamp(self.logstd, self.logstd_min, self.logstd_max))
        self.distribution = Normal(mean, mean * 0.0 + std)

    def act(self, actor_input: torch.Tensor) -> torch.Tensor:
        self.update_distribution(actor_input)
        return self.distribution.sample()

    def act_inference(self, actor_input: torch.Tensor) -> torch.Tensor:
        return self.actor(actor_input)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.standard_critic(critic_input).squeeze(-1), self.barrier_critic(critic_input).squeeze(-1)

    def clamp_logstd_(self) -> None:
        with torch.no_grad():
            self.logstd.clamp_(self.logstd_min, self.logstd_max)
