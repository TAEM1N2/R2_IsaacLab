import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal

from .actor_critic import get_activation


class AdaptiveActorCritic(nn.Module):
    """Actor-critic used for Phase 2 residual action correction."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=(256, 128),
        critic_hidden_dims=(256, 128),
        activation="elu",
        orthogonal_init=False,
        init_noise_std=0.5,
        action_scale=0.1,
        logstd_min=-5.0,
        logstd_max=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "AdaptiveActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.action_scale = float(action_scale)
        self.logstd_min = float(logstd_min)
        self.logstd_max = float(logstd_max)
        activation = get_activation(activation)

        actor_layers = []
        last_dim = num_actor_obs
        for hidden_dim in actor_hidden_dims:
            actor_layers.append(nn.Linear(last_dim, hidden_dim))
            if orthogonal_init:
                torch.nn.init.orthogonal_(actor_layers[-1].weight, np.sqrt(2))
                torch.nn.init.constant_(actor_layers[-1].bias, 0.0)
            actor_layers.append(activation)
            last_dim = hidden_dim
        actor_layers.append(nn.Linear(last_dim, num_actions))
        if orthogonal_init:
            torch.nn.init.orthogonal_(actor_layers[-1].weight, 0.01)
            torch.nn.init.constant_(actor_layers[-1].bias, 0.0)
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = []
        last_dim = num_critic_obs
        for hidden_dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(last_dim, hidden_dim))
            if orthogonal_init:
                torch.nn.init.orthogonal_(critic_layers[-1].weight, np.sqrt(2))
                torch.nn.init.constant_(critic_layers[-1].bias, 0.0)
            critic_layers.append(activation)
            last_dim = hidden_dim
        critic_layers.append(nn.Linear(last_dim, 1))
        if orthogonal_init:
            torch.nn.init.orthogonal_(critic_layers[-1].weight, 0.01)
            torch.nn.init.constant_(critic_layers[-1].bias, 0.0)
        self.critic = nn.Sequential(*critic_layers)

        self.logstd = nn.Parameter(torch.log(torch.ones(num_actions) * init_noise_std))
        self.distribution = None
        Normal.set_default_validate_args = False

        print(f"Adaptive Actor MLP: {self.actor}")
        print(f"Adaptive Critic MLP: {self.critic}")

    def clamp_logstd_(self):
        with torch.no_grad():
            self.logstd.clamp_(self.logstd_min, self.logstd_max)

    @property
    def clamped_logstd(self):
        return torch.clamp(self.logstd, self.logstd_min, self.logstd_max)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        pass

    def update_distribution(self, observations):
        raw_mean = self.actor(observations)
        self.distribution = Normal(raw_mean, raw_mean * 0.0 + torch.exp(self.clamped_logstd))

    def act(self, observations):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, raw_actions):
        return self.distribution.log_prob(raw_actions).sum(dim=-1)

    def evaluate(self, observations):
        return self.critic(observations)

    def raw_to_delta(self, raw_actions):
        return self.action_scale * torch.tanh(raw_actions)

    def act_inference(self, observations):
        raw_mean = self.actor(observations)
        return self.raw_to_delta(raw_mean)
