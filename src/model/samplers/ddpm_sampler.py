"""
DDPM Sampler for Diffusion Models

Implements the reverse diffusion process for sampling from diffusion models.
Supports both DDPM (stochastic) and DDIM (deterministic) sampling.

Reference:
- Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- Song et al., "Denoising Diffusion Implicit Models" (2021)
"""

import torch
from typing import Callable, Optional, Literal
from tqdm import tqdm

from ..base.base_sampler import BaseSampler


class DDPMSampler(BaseSampler):
    """
    DDPM/DDIM sampler for diffusion models.

    Implements the reverse diffusion process:
        x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * eps_pred) + sigma_t * z

    Args:
        betas: Beta schedule tensor of shape (num_timesteps,)
        ddim_eta: DDIM eta parameter (0 = deterministic DDIM, 1 = stochastic DDPM)
        clip_sample: Whether to clip samples to [-1, 1] during sampling
    """

    def __init__(
        self,
        betas: Optional[torch.Tensor] = None,
        num_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        ddim_eta: float = 1.0,
        clip_sample: bool = False
    ):
        self.ddim_eta = ddim_eta
        self.clip_sample = clip_sample
        self.num_timesteps = num_timesteps

        # Compute or use provided betas
        if betas is not None:
            self.betas = betas
        else:
            self.betas = self._get_beta_schedule(beta_schedule, num_timesteps)

        # Compute alpha values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]),
            self.alphas_cumprod[:-1]
        ])

        # Precompute useful quantities
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # For DDPM posterior
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def _get_beta_schedule(self, schedule: str, num_timesteps: int) -> torch.Tensor:
        """Compute beta schedule."""
        import math

        if schedule == "linear":
            return torch.linspace(1e-4, 0.02, num_timesteps)
        elif schedule == "cosine":
            steps = torch.linspace(0, num_timesteps, num_timesteps + 1)
            alpha_bar = torch.cos((steps / num_timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            return torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def sample(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_init: torch.Tensor,
        num_steps: Optional[int] = None,
        show_progress: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate samples using reverse diffusion process.

        Args:
            model: Noise prediction model eps(x, t)
            x_init: Initial samples (pure noise), shape (B, C, H, W)
            num_steps: Number of sampling steps (if less than num_timesteps, uses DDIM-style skipping)
            show_progress: Whether to show progress bar

        Returns:
            Generated samples, shape (B, C, H, W)
        """
        device = x_init.device
        num_steps = num_steps or self.num_timesteps

        # Move precomputed values to device
        alphas_cumprod = self.alphas_cumprod.to(device)
        alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        betas = self.betas.to(device)
        sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        posterior_variance = self.posterior_variance.to(device)

        x = x_init.clone()

        # Create timestep schedule (for DDIM-style step skipping)
        if num_steps < self.num_timesteps:
            # Uniformly spaced timesteps
            step_ratio = self.num_timesteps // num_steps
            timesteps = torch.arange(0, self.num_timesteps, step_ratio).flip(0)
        else:
            timesteps = torch.arange(self.num_timesteps - 1, -1, -1)

        iterator = timesteps
        if show_progress:
            iterator = tqdm(iterator, desc="DDPM Sampling")

        for i, t_idx in enumerate(iterator):
            t_idx = t_idx.item()

            # Time tensor for model (normalized to [0, 1])
            t = torch.full((x.shape[0],), t_idx / self.num_timesteps, device=device)

            # Predict noise
            with torch.no_grad():
                noise_pred = model(x, t)

            # Get alpha values for current timestep
            alpha = self.alphas[t_idx]
            alpha_cumprod = alphas_cumprod[t_idx]
            alpha_cumprod_prev = alphas_cumprod_prev[t_idx]
            beta = betas[t_idx]

            if self.ddim_eta == 0:
                # DDIM (deterministic)
                # x_{t-1} = sqrt(alpha_{t-1}) * x0_pred + sqrt(1-alpha_{t-1}) * eps_pred
                x0_pred = (x - sqrt_one_minus_alphas_cumprod[t_idx] * noise_pred) / (
                    self.sqrt_alphas_cumprod[t_idx].to(device) + 1e-8
                )

                if self.clip_sample:
                    x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

                dir_xt = torch.sqrt(1.0 - alpha_cumprod_prev) * noise_pred
                x = torch.sqrt(alpha_cumprod_prev) * x0_pred + dir_xt

            else:
                # DDPM (stochastic)
                # x_{t-1} = (1/sqrt(alpha)) * (x_t - beta/sqrt(1-alpha_bar) * eps) + sigma * z
                x = sqrt_recip_alphas[t_idx] * (
                    x - (beta / sqrt_one_minus_alphas_cumprod[t_idx]) * noise_pred
                )

                if t_idx > 0:
                    noise = torch.randn_like(x)
                    sigma = torch.sqrt(posterior_variance[t_idx]) * self.ddim_eta
                    x = x + sigma * noise

                if self.clip_sample:
                    x = torch.clamp(x, -1.0, 1.0)

        return x

    def sample_with_trajectory(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_init: torch.Tensor,
        num_steps: Optional[int] = None,
        save_every: int = 10,
        **kwargs
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Generate samples and save intermediate trajectory.

        Args:
            model: Noise prediction model
            x_init: Initial samples
            num_steps: Number of steps
            save_every: Save trajectory every N steps

        Returns:
            Tuple of (final_samples, trajectory_list)
        """
        device = x_init.device
        num_steps = num_steps or self.num_timesteps

        x = x_init.clone()
        trajectory = [x.clone()]

        # Move values to device
        alphas_cumprod = self.alphas_cumprod.to(device)
        alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        betas = self.betas.to(device)
        sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        posterior_variance = self.posterior_variance.to(device)

        for step, t_idx in enumerate(reversed(range(num_steps))):
            t = torch.full((x.shape[0],), t_idx / num_steps, device=device)

            with torch.no_grad():
                noise_pred = model(x, t)

            alpha_cumprod = alphas_cumprod[t_idx]
            alpha_cumprod_prev = alphas_cumprod_prev[t_idx]
            beta = betas[t_idx]

            x = sqrt_recip_alphas[t_idx] * (
                x - (beta / sqrt_one_minus_alphas_cumprod[t_idx]) * noise_pred
            )

            if t_idx > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(posterior_variance[t_idx]) * self.ddim_eta * noise

            if (step + 1) % save_every == 0:
                trajectory.append(x.clone())

        return x, trajectory
