"""
HuggingFace Diffusers Diffusion Path Implementation

Wraps HuggingFace's diffusers library schedulers for the forward diffusion process.

Supports all beta schedules from diffusers:
- linear: Linear schedule
- scaled_linear: Scaled linear (used by Stable Diffusion)
- squaredcos_cap_v2: Squared cosine with cap (improved DDPM)
- sigmoid: Sigmoid schedule

Install: pip install diffusers

Reference:
- https://huggingface.co/docs/diffusers/api/schedulers/ddpm
- Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
"""

import torch
from typing import Optional, Literal

from ..base.base_path import BasePath, PathSample

# Beta schedule options
BetaSchedule = Literal["linear", "scaled_linear", "squaredcos_cap_v2", "sigmoid"]

# Lazy import to avoid hard dependency
_diffusers_available = None


def _check_diffusers():
    """Check if diffusers library is available."""
    global _diffusers_available
    if _diffusers_available is None:
        try:
            from diffusers import DDPMScheduler
            _diffusers_available = True
        except ImportError:
            _diffusers_available = False
    return _diffusers_available


class HFDiffusionPath(BasePath):
    """
    Diffusion path using HuggingFace's diffusers library.

    Uses DDPMScheduler for the forward noising process during training.
    The model learns to predict the noise added at each timestep.

    Forward process:
        x_t = sqrt(α̅_t) * x_1 + sqrt(1 - α̅_t) * ε

    Args:
        num_train_timesteps: Number of diffusion timesteps
        beta_schedule: Schedule for beta values
            - 'linear': Linear from beta_start to beta_end
            - 'scaled_linear': Scaled linear (Stable Diffusion style)
            - 'squaredcos_cap_v2': Squared cosine with cap (recommended)
            - 'sigmoid': Sigmoid schedule
        beta_start: Starting beta value
        beta_end: Ending beta value
        prediction_type: What the model predicts
            - 'epsilon': Predict noise (default, standard DDPM)
            - 'v_prediction': Predict v = sqrt(α̅)*ε - sqrt(1-α̅)*x
            - 'sample': Predict x_0 directly
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: BetaSchedule = "squaredcos_cap_v2",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        prediction_type: Literal["epsilon", "v_prediction", "sample"] = "epsilon",
    ):
        if not _check_diffusers():
            raise ImportError(
                "HuggingFace diffusers library is required. "
                "Install with: pip install diffusers"
            )

        from diffusers import DDPMScheduler

        self.num_train_timesteps = num_train_timesteps
        self._prediction_type = prediction_type

        # Create the HuggingFace scheduler
        self._scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            prediction_type=prediction_type,
            clip_sample=False,
        )

    def sample(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> PathSample:
        """
        Sample from the diffusion path at time t.

        Uses HuggingFace's scheduler.add_noise() internally.

        Args:
            x_0: Noise samples (ε), shape (B, C, H, W)
            x_1: Data samples (clean), shape (B, C, H, W)
            t: Time values in [0, 1], shape (B,). If None, sample uniformly.

        Returns:
            PathSample with noised data and target
        """
        batch_size = x_1.shape[0]
        device = x_1.device

        # Sample time uniformly if not provided
        if t is None:
            t = torch.rand(batch_size, device=device)

        # Convert continuous t ∈ [0,1] to discrete timesteps
        timesteps = (t * self.num_train_timesteps).long().clamp(0, self.num_train_timesteps - 1)

        # Use HuggingFace's add_noise method
        noise = x_0
        x_t = self._scheduler.add_noise(x_1, noise, timesteps)

        # Compute target based on prediction type
        if self._prediction_type == "epsilon":
            target = noise
        elif self._prediction_type == "v_prediction":
            # v = sqrt(α̅)*ε - sqrt(1-α̅)*x
            alphas_cumprod = self._scheduler.alphas_cumprod.to(device)
            sqrt_alpha = alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)
            sqrt_one_minus_alpha = (1 - alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)
            target = sqrt_alpha * noise - sqrt_one_minus_alpha * x_1
        elif self._prediction_type == "sample":
            target = x_1
        else:
            raise ValueError(f"Unknown prediction_type: {self._prediction_type}")

        return PathSample(
            x_t=x_t,
            t=t,
            target=target,
            x_0=noise,
            x_1=x_1
        )

    @property
    def prediction_type(self) -> str:
        """What the model predicts: 'noise', 'velocity', or 'sample'."""
        if self._prediction_type == "epsilon":
            return "noise"
        elif self._prediction_type == "v_prediction":
            return "velocity"
        else:
            return "sample"

    def get_x1_from_prediction(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        prediction: torch.Tensor
    ) -> torch.Tensor:
        """
        Recover x_1 (data) from model prediction.

        Args:
            x_t: Current noised point
            t: Time values
            prediction: Model's prediction

        Returns:
            Estimated x_1 (data)
        """
        device = x_t.device
        timesteps = (t * self.num_train_timesteps).long().clamp(0, self.num_train_timesteps - 1)

        alphas_cumprod = self._scheduler.alphas_cumprod.to(device)
        sqrt_alpha = alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = (1 - alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)

        if self._prediction_type == "epsilon":
            # x_1 = (x_t - sqrt(1-α̅)*ε) / sqrt(α̅)
            x_1 = (x_t - sqrt_one_minus_alpha * prediction) / (sqrt_alpha + 1e-8)
        elif self._prediction_type == "v_prediction":
            # x_1 = sqrt(α̅)*x_t - sqrt(1-α̅)*v
            x_1 = sqrt_alpha * x_t - sqrt_one_minus_alpha * prediction
        elif self._prediction_type == "sample":
            x_1 = prediction
        else:
            raise ValueError(f"Unknown prediction_type: {self._prediction_type}")

        return x_1

    @property
    def hf_scheduler(self):
        """Access the underlying HuggingFace scheduler."""
        return self._scheduler
