import torch
import torch.nn as nn

from .actor_critic import get_activation


class LoadEncoder(nn.Module):
    """Encodes adaptive observation history into a payload-response latent."""

    def __init__(
        self,
        input_dim,
        latent_dim=16,
        hidden_dims=(128, 64),
        activation="elu",
        orthogonal_init=False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        activation = get_activation(activation)

        layers = []
        last_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layer = nn.Linear(last_dim, hidden_dim)
            if orthogonal_init:
                nn.init.orthogonal_(layer.weight, nn.init.calculate_gain("relu"))
                nn.init.constant_(layer.bias, 0.0)
            layers.append(layer)
            layers.append(activation)
            last_dim = hidden_dim
        out = nn.Linear(last_dim, self.latent_dim)
        if orthogonal_init:
            nn.init.orthogonal_(out.weight, 0.01)
            nn.init.constant_(out.bias, 0.0)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, adaptive_history):
        if adaptive_history.dim() > 2:
            adaptive_history = adaptive_history.flatten(start_dim=1)
        return self.net(adaptive_history)


class LoadTransitionModel(nn.Module):
    """Predicts the next load latent from current latent and residual-control context."""

    def __init__(
        self,
        latent_dim,
        action_dim,
        hidden_dims=(128, 64),
        activation="elu",
        orthogonal_init=False,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        activation = get_activation(activation)

        layers = []
        last_dim = self.latent_dim + 2 * self.action_dim
        for hidden_dim in hidden_dims:
            layer = nn.Linear(last_dim, hidden_dim)
            if orthogonal_init:
                nn.init.orthogonal_(layer.weight, nn.init.calculate_gain("relu"))
                nn.init.constant_(layer.bias, 0.0)
            layers.append(layer)
            layers.append(activation)
            last_dim = hidden_dim
        out = nn.Linear(last_dim, self.latent_dim)
        if orthogonal_init:
            nn.init.orthogonal_(out.weight, 0.01)
            nn.init.constant_(out.bias, 0.0)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, z_load, nominal_action, delta_action):
        return self.net(torch.cat((z_load, nominal_action, delta_action), dim=-1))
