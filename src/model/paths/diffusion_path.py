"""
Diffusion Path Implementation

Implements DDPM-style diffusion path with various beta schedules.
The model learns to predict the noise added at each timestep.

Reference:
- Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models" (2021)
"""

import math
import torch
from typing import Optional

from ..base.base_path import BasePath, PathSample


class DiffusionPath(BasePath):
    """
    DDPM-style diffusion path.

    Uses the forward process:
        x_t = sqrt(alpha_bar_t) * x_1 + sqrt(1 - alpha_bar_t) * epsilon

    The model learns to predict the noise epsilon.

    Args:
        num_timesteps: Number of discrete diffusion timesteps
        beta_schedule: Schedule for beta values ('linear', 'cosine', 'squaredcos_cap_v2')
        beta_start: Starting beta value (for linear schedule)
        beta_end: Ending beta value (for linear schedule)
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ):
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule

        # Compute beta schedule
        self.betas = self._get_beta_schedule(
            beta_schedule, num_timesteps, beta_start, beta_end
        )

        # Compute alpha values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Precompute useful quantities
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def _get_beta_schedule(
        self,
        schedule: str,
        num_timesteps: int,
        beta_start: float,
        beta_end: float
    ) -> torch.Tensor:
        """Compute beta schedule."""

        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, num_timesteps)

        elif schedule == "cosine":
            # Cosine schedule from Improved DDPM
            steps = torch.linspace(0, num_timesteps, num_timesteps + 1)
            alpha_bar = torch.cos((steps / num_timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            return torch.clamp(betas, 0.0001, 0.9999)

        elif schedule == "squaredcos_cap_v2":
            # Squared cosine schedule (used by diffusers)
            steps = torch.linspace(0, num_timesteps, num_timesteps + 1)
            alpha_bar = torch.cos((steps / num_timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            return torch.clamp(betas, 0.0001, 0.999)

        elif schedule == "sigmoid":
            # Sigmoid schedule
            betas = torch.linspace(-6, 6, num_timesteps)
            betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
            return betas

        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")

    def sample(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> PathSample:
        """
        Sample from the diffusion path at time t.

        Note: In diffusion terminology:
        - x_0 here is the noise (epsilon)
        - x_1 here is the data (clean image)
        This is opposite to the standard diffusion notation but consistent
        with flow matching (x_0 = source/noise, x_1 = target/data).

        Args:
            x_0: Noise samples, shape (B, C, H, W)
            x_1: Data samples, shape (B, C, H, W)
            t: Time values in [0, 1], shape (B,). If None, sample uniformly.

        Returns:
            PathSample with noised data and target noise
        """
        batch_size = x_1.shape[0]
        device = x_1.device

        # Sample time uniformly if not provided
        if t is None:
            t = torch.rand(batch_size, device=device)

        # Convert continuous t to discrete timesteps
        timesteps = (t * self.num_timesteps).long().clamp(0, self.num_timesteps - 1)

        # Move precomputed values to correct device
        sqrt_alpha = self.sqrt_alphas_cumprod.to(device)[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod.to(device)[timesteps].view(-1, 1, 1, 1)

        # x_t = sqrt(alpha_bar) * x_1 + sqrt(1 - alpha_bar) * noise
        noise = x_0
        x_t = sqrt_alpha * x_1 + sqrt_one_minus_alpha * noise

        return PathSample(
            x_t=x_t,
            t=t,
            target=noise,  # Model predicts the noise
            x_0=noise,
            x_1=x_1
        )

    @property
    def prediction_type(self) -> str:
        return "noise"

    def get_x1_from_prediction(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        prediction: torch.Tensor
    ) -> torch.Tensor:
        """
        Recover x_1 (data) from noise prediction.

        From x_t = sqrt(alpha_bar) * x_1 + sqrt(1 - alpha_bar) * noise:
            x_1 = (x_t - sqrt(1 - alpha_bar) * noise) / sqrt(alpha_bar)

        Args:
            x_t: Current noised point
            t: Time values
            prediction: Predicted noise

        Returns:
            Estimated x_1 (data)
        """
        device = x_t.device

        # Convert continuous t to discrete timesteps
        timesteps = (t * self.num_timesteps).long().clamp(0, self.num_timesteps - 1)

        sqrt_alpha = self.sqrt_alphas_cumprod.to(device)[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod.to(device)[timesteps].view(-1, 1, 1, 1)

        # x_1 = (x_t - sqrt(1-alpha) * noise) / sqrt(alpha)
        # Add small epsilon for numerical stability
        x_1 = (x_t - sqrt_one_minus_alpha * prediction) / (sqrt_alpha + 1e-8)

        return x_1

    def get_betas(self) -> torch.Tensor:
        """Return beta schedule (useful for DDPM sampler)."""
        return self.betas

    def get_alphas_cumprod(self) -> torch.Tensor:
        """Return cumulative product of alphas."""
        return self.alphas_cumprod
