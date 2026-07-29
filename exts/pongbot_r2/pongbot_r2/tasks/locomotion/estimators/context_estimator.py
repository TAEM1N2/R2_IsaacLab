"""Context Estimator Network for history-based locomotion context learning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextEstimatorNet(nn.Module):
    """DreamWaQ/MULE-style CE Net with explicit velocity and VAE latent heads."""

    def __init__(
        self,
        history_len: int = 5,
        obs_dim: int | None = None,
        latent_dim: int = 16,
        encoder_hidden_dims: tuple[int, int] = (128, 64),
        decoder_hidden_dims: tuple[int, int] = (64, 128),
        beta: float = 1.0,
    ) -> None:
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

        encoder_input_dim = history_len * obs_dim
        self.encoder_feature_dim = encoder_hidden_dims[-1]

        self.encoder = self._build_encoder(
            input_dim=encoder_input_dim,
            hidden_dims=encoder_hidden_dims,
        )
        self.velocity_head = nn.Linear(self.encoder_feature_dim, 3)
        self.mu_head = nn.Linear(self.encoder_feature_dim, latent_dim)
        self.logvar_head = nn.Linear(self.encoder_feature_dim, latent_dim)

        self.decoder = self._build_mlp(
            input_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            output_dim=obs_dim,
        )

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
        """Sample z with the beta-VAE reparameterization trick."""
        # mu/logvar: [batch, latent_dim]
        std = torch.exp(0.5 * logvar)
        # eps: [batch, latent_dim]
        eps = torch.randn_like(std)
        # z: [batch, latent_dim]
        return mu + eps * std

    def forward(
        self, obs_history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run CE Net on observation history.

        Args:
            obs_history: Observation history with shape [batch, H, obs_dim].

        Returns:
            v_pred: Body velocity estimate with shape [batch, 3].
            z: Sampled latent variable with shape [batch, latent_dim].
            mu: Latent posterior mean with shape [batch, latent_dim].
            logvar: Latent posterior log-variance with shape [batch, latent_dim].
            o_next_recon: Reconstructed next observation with shape [batch, obs_dim].
        """
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

    def compute_loss(
        self,
        obs_history: torch.Tensor,
        v_target: torch.Tensor,
        o_next_target: torch.Tensor,
        beta: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute L_ce = L_est + L_vae for CE Net training."""
        # obs_history: [batch, H, obs_dim]
        v_pred, _z, mu, logvar, o_next_recon = self.forward(obs_history)
        self._validate_targets(v_pred, v_target, o_next_recon, o_next_target)

        beta_value = self.beta if beta is None else beta

        # L_est: scalar MSE between v_pred [batch, 3] and v_target [batch, 3]
        estimation_loss = F.mse_loss(v_pred, v_target)
        # reconstruction_loss: scalar MSE between next-observation tensors [batch, obs_dim]
        reconstruction_loss = F.mse_loss(o_next_recon, o_next_target)
        # kl_loss_per_sample: [batch]
        kl_loss_per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        # kl_loss: scalar KL(q(z|o_hist) || N(0, I))
        kl_loss = kl_loss_per_sample.mean()
        # L_vae: scalar next-observation reconstruction plus beta-weighted KL
        vae_loss = reconstruction_loss + beta_value * kl_loss
        # L_ce: scalar total CE loss
        ce_loss = estimation_loss + vae_loss

        loss_terms = {
            "ce_loss": ce_loss,
            "estimation_loss": estimation_loss,
            "vae_loss": vae_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }
        return ce_loss, loss_terms

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

    def _validate_targets(
        self,
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
