"""Context Estimator Network used by the implicit locomotion policy."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_context_policy_input(
    encoder_output: torch.Tensor,
    current_obs: torch.Tensor,
    commands: torch.Tensor,
    obs_history: torch.Tensor | None = None,
    include_history: bool = False,
) -> torch.Tensor:
    """Build nominal policy input concat([encoder_output, current_obs, commands], dim=-1)."""
    assert encoder_output.ndim == 2, (
        f"encoder_output must have shape [batch, encoder_dim], got {tuple(encoder_output.shape)}"
    )
    assert current_obs.ndim == 2, f"current_obs must have shape [batch, obs_dim], got {tuple(current_obs.shape)}"
    assert commands.ndim == 2, f"commands must have shape [batch, command_dim], got {tuple(commands.shape)}"
    assert encoder_output.shape[0] == current_obs.shape[0] == commands.shape[0], (
        "batch size mismatch for policy input concat: "
        f"encoder_output={tuple(encoder_output.shape)}, current_obs={tuple(current_obs.shape)}, "
        f"commands={tuple(commands.shape)}"
    )

    if include_history:
        assert obs_history is not None, "obs_history must be provided when include_history=True"
        assert obs_history.ndim == 2, f"obs_history must be flattened [batch, dim], got {tuple(obs_history.shape)}"
        assert obs_history.shape[0] == current_obs.shape[0], (
            "batch size mismatch for context policy history concat: "
            f"obs_history={tuple(obs_history.shape)}, current_obs={tuple(current_obs.shape)}"
        )
        # policy_input: [batch, encoder_dim + obs_dim + command_dim + history_dim]
        return torch.cat((encoder_output, current_obs, commands, obs_history), dim=-1)

    # policy_input: [batch, encoder_dim + obs_dim + command_dim]
    return torch.cat((encoder_output, current_obs, commands), dim=-1)


class ContextEstimatorNet(nn.Module):
    """DreamWaQ/MULE-style CE Net with explicit velocity and VAE latent heads."""

    is_context_estimator = True
    is_vae = True

    def __init__(
        self,
        history_len: int = 5,
        obs_dim: int | None = None,
        latent_dim: int = 16,
        encoder_hidden_dims: tuple[int, int] | list[int] = (128, 64),
        decoder_hidden_dims: tuple[int, int] | list[int] = (64, 128),
        beta: float = 1.0,
        output_detach: bool = True,
        terrain_conditioned_prior: bool = False,
        z_terrain_dim: int = 0,
        residual_kl_coef: float = 0.0,
        num_terrain_classes: int = 0,
        terrain_prior_radius: float = 1.0,
        terrain_prior_std: float = 2.0,
        terrain_prior_coef: float = 1.0,
        terrain_classification_coef: float = 0.0,
        terrain_property_dim: int = 0,
        terrain_property_coef: float = 0.0,
        **kwargs,
    ) -> None:
        if kwargs:
            print("ContextEstimatorNet got unexpected arguments, which will be ignored: " + str(list(kwargs.keys())))
        super().__init__()

        if history_len <= 0:
            raise ValueError(f"history_len must be positive, got {history_len}")
        if obs_dim is None:
            raise ValueError("obs_dim must be provided from obsHistory.shape[-1]")
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {obs_dim}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")

        self.history_len = history_len
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.output_detach = output_detach
        self.num_output_dim = 3 + latent_dim
        self.terrain_conditioned_prior = bool(terrain_conditioned_prior)
        self.z_terrain_dim = int(z_terrain_dim)
        self.residual_kl_coef = float(residual_kl_coef)
        self.num_terrain_classes = int(num_terrain_classes)
        self.terrain_prior_std = float(terrain_prior_std)
        self.terrain_prior_coef = float(terrain_prior_coef)
        self.terrain_classification_coef = float(terrain_classification_coef)
        self.terrain_property_dim = int(terrain_property_dim)
        self.terrain_property_coef = float(terrain_property_coef)

        if self.terrain_prior_std <= 0.0:
            raise ValueError(f"terrain_prior_std must be positive, got {terrain_prior_std}")
        if (self.terrain_conditioned_prior or self.terrain_classification_coef > 0.0) and self.num_terrain_classes <= 0:
            raise ValueError("num_terrain_classes must be positive when terrain prior/classification is enabled")
        if self.terrain_property_dim < 0:
            raise ValueError(f"terrain_property_dim must be non-negative, got {terrain_property_dim}")
        if self.z_terrain_dim < 0:
            raise ValueError(f"z_terrain_dim must be non-negative, got {z_terrain_dim}")
        if self.z_terrain_dim > latent_dim:
            raise ValueError(f"z_terrain_dim={z_terrain_dim} cannot exceed latent_dim={latent_dim}")
        if self.residual_kl_coef < 0.0:
            raise ValueError(f"residual_kl_coef must be non-negative, got {residual_kl_coef}")
        self.terrain_latent_dim = self.z_terrain_dim if self.z_terrain_dim > 0 else latent_dim

        encoder_hidden_dims = tuple(encoder_hidden_dims)
        decoder_hidden_dims = tuple(decoder_hidden_dims)
        encoder_input_dim = history_len * obs_dim
        self.encoder_feature_dim = encoder_hidden_dims[-1]

        self.encoder = self._build_encoder(encoder_input_dim, encoder_hidden_dims)
        self.velocity_head = nn.Linear(self.encoder_feature_dim, 3)
        self.mu_head = nn.Linear(self.encoder_feature_dim, latent_dim)
        self.logvar_head = nn.Linear(self.encoder_feature_dim, latent_dim)
        self.decoder = self._build_mlp(latent_dim, decoder_hidden_dims, obs_dim)
        self.terrain_classifier = (
            nn.Linear(self.terrain_latent_dim, self.num_terrain_classes)
            if self.terrain_classification_coef > 0.0 and self.num_terrain_classes > 0
            else None
        )
        self.terrain_property_head = (
            nn.Linear(self.terrain_latent_dim, self.terrain_property_dim)
            if self.terrain_property_coef > 0.0 and self.terrain_property_dim > 0
            else None
        )

        if self.num_terrain_classes > 0:
            prior_means = torch.randn(self.num_terrain_classes, self.terrain_latent_dim)
            prior_means = F.normalize(prior_means, p=2, dim=-1) * float(terrain_prior_radius)
        else:
            prior_means = torch.zeros(0, self.terrain_latent_dim)
        self.register_buffer("terrain_prior_means", prior_means)

    @staticmethod
    def _build_encoder(input_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ELU())
            last_dim = hidden_dim
        return nn.Sequential(*layers)

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ELU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        return nn.Sequential(*layers)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # mu/logvar: [batch, latent_dim]
        std = torch.exp(0.5 * logvar)
        # eps: [batch, latent_dim]
        eps = torch.randn_like(std)
        # z: [batch, latent_dim]
        return mu + eps * std

    def forward_train(
        self, obs_history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_obs_history(obs_history)

        batch_size = obs_history.shape[0]
        # obs_history: [batch, H, obs_dim] -> obs_history_flat: [batch, H * obs_dim]
        obs_history_flat = obs_history.reshape(batch_size, self.history_len * self.obs_dim)
        # encoder_features: [batch, encoder_feature_dim]
        encoder_features = self.encoder(obs_history_flat)
        # v_pred: [batch, 3]
        v_pred = self.velocity_head(encoder_features)
        # mu/logvar: [batch, latent_dim]
        mu = self.mu_head(encoder_features)
        logvar = self.logvar_head(encoder_features)
        # z: [batch, latent_dim]
        z = self.reparameterize(mu, logvar)
        # o_next_recon: [batch, obs_dim]
        o_next_recon = self.decoder(z)
        return v_pred, z, mu, logvar, o_next_recon

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        return self.encode(obs_history)

    def encode(self, obs_history: torch.Tensor) -> torch.Tensor:
        v_pred, z, _mu, _logvar, _o_next_recon = self.forward_train(obs_history)
        if self.output_detach:
            v_pred = v_pred.detach()
            z = z.detach()
        # encoder_out: [batch, 3 + latent_dim], built from separate CE heads.
        return torch.cat((v_pred, z), dim=-1)

    def compute_loss(
        self,
        obs_history: torch.Tensor,
        v_target: torch.Tensor,
        o_next_target: torch.Tensor,
        beta: float | None = None,
        terrain_id: torch.Tensor | None = None,
        terrain_property_target: torch.Tensor | None = None,
        terrain_warmup_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # obs_history: [batch, H, obs_dim]
        v_pred, _z, mu, logvar, o_next_recon = self.forward_train(obs_history)
        self._validate_targets(v_pred, v_target, o_next_recon, o_next_target)

        beta_value = self.beta if beta is None else beta
        # L_est: scalar MSE between v_pred [batch, 3] and v_target [batch, 3]
        estimation_loss = F.mse_loss(v_pred, v_target)
        # reconstruction_loss: scalar MSE between next-observation tensors [batch, obs_dim]
        reconstruction_loss = F.mse_loss(o_next_recon, o_next_target)
        warmup_scale = float(max(0.0, min(1.0, terrain_warmup_scale)))
        terrain_mu = self._terrain_latent(mu)
        terrain_logvar = self._terrain_latent(logvar)
        residual_mu = self._residual_latent(mu)
        residual_logvar = self._residual_latent(logvar)

        if self.terrain_conditioned_prior and terrain_id is not None:
            kl_loss = self._terrain_prior_kl(terrain_mu, terrain_logvar, terrain_id)
            kl_loss = kl_loss * self.terrain_prior_coef * warmup_scale
            if self.residual_kl_coef > 0.0 and residual_mu is not None and residual_logvar is not None:
                residual_kl = self._standard_normal_kl(residual_mu, residual_logvar)
                kl_loss = kl_loss + residual_kl * self.residual_kl_coef
        else:
            kl_loss = self._standard_normal_kl(mu, logvar)

        terrain_classification_loss = mu.new_tensor(0.0)
        if self.terrain_classifier is not None and terrain_id is not None:
            terrain_classification_loss = self._terrain_classification_loss(terrain_mu, terrain_id)
            terrain_classification_loss = terrain_classification_loss * self.terrain_classification_coef * warmup_scale

        terrain_property_loss = mu.new_tensor(0.0)
        if self.terrain_property_head is not None and terrain_property_target is not None:
            terrain_property_pred = self.terrain_property_head(terrain_mu)
            terrain_property_loss = F.mse_loss(terrain_property_pred, terrain_property_target)
            terrain_property_loss = terrain_property_loss * self.terrain_property_coef * warmup_scale

        # L_vae: scalar next-observation reconstruction plus beta-weighted KL
        vae_loss = reconstruction_loss + beta_value * kl_loss
        # L_ce: scalar total CE loss
        ce_loss = estimation_loss + vae_loss + terrain_classification_loss + terrain_property_loss
        return ce_loss, {
            "ce_loss": ce_loss,
            "estimation_loss": estimation_loss,
            "vae_loss": vae_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "terrain_classification_loss": terrain_classification_loss,
            "terrain_property_loss": terrain_property_loss,
            "terrain_warmup_scale": mu.new_tensor(warmup_scale),
        }

    def _terrain_latent(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:, : self.terrain_latent_dim]

    def _residual_latent(self, tensor: torch.Tensor) -> torch.Tensor | None:
        if self.z_terrain_dim <= 0 or self.z_terrain_dim >= self.latent_dim:
            return None
        return tensor[:, self.z_terrain_dim :]

    @staticmethod
    def _standard_normal_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl_loss_per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl_loss_per_sample.mean()

    def _valid_terrain_labels(self, terrain_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        labels = terrain_id.view(-1).long()
        valid = (labels >= 0) & (labels < self.num_terrain_classes)
        return labels, valid

    def _terrain_prior_kl(self, mu: torch.Tensor, logvar: torch.Tensor, terrain_id: torch.Tensor) -> torch.Tensor:
        labels, valid = self._valid_terrain_labels(terrain_id)
        prior_mu = torch.zeros_like(mu)
        if valid.any():
            prior_mu[valid] = self.terrain_prior_means[labels[valid]].to(device=mu.device, dtype=mu.dtype)
        prior_var = self.terrain_prior_std * self.terrain_prior_std
        kl_loss_per_sample = 0.5 * torch.sum(
            (logvar.exp() + torch.square(mu - prior_mu)) / prior_var
            - 1.0
            + 2.0 * torch.log(mu.new_tensor(self.terrain_prior_std))
            - logvar,
            dim=-1,
        )
        return kl_loss_per_sample.mean()

    def _terrain_classification_loss(self, mu: torch.Tensor, terrain_id: torch.Tensor) -> torch.Tensor:
        labels, valid = self._valid_terrain_labels(terrain_id)
        if not valid.any():
            return mu.new_tensor(0.0)
        logits = self.terrain_classifier(mu[valid])
        return F.cross_entropy(logits, labels[valid])

    def _validate_obs_history(self, obs_history: torch.Tensor) -> None:
        if obs_history.ndim != 3:
            raise ValueError(
                f"obs_history must have shape [batch, {self.history_len}, {self.obs_dim}], "
                f"got {tuple(obs_history.shape)}"
            )
        if obs_history.shape[1] != self.history_len or obs_history.shape[2] != self.obs_dim:
            raise ValueError(
                f"obs_history must have shape [batch, {self.history_len}, {self.obs_dim}], "
                f"got {tuple(obs_history.shape)}"
            )

    @staticmethod
    def _validate_targets(
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        o_next_recon: torch.Tensor,
        o_next_target: torch.Tensor,
    ) -> None:
        if v_target.shape != v_pred.shape:
            raise ValueError(f"v_target must have shape {tuple(v_pred.shape)}, got {tuple(v_target.shape)}")
        if o_next_target.shape != o_next_recon.shape:
            raise ValueError(
                f"o_next_target must have shape {tuple(o_next_recon.shape)}, got {tuple(o_next_target.shape)}"
            )
