"""
HuggingFace Diffusers Path Implementation

Defines the forward diffusion process using diffusers schedulers.
"""

from __future__ import annotations

from typing import Optional, Literal

import torch

from ..base import BasePath, PathSample

PredictionType = Literal["epsilon", "v_prediction", "sample"]

_diffusers_available = None


def _check_diffusers() -> bool:
    """Check if diffusers library is available."""
    global _diffusers_available
    if _diffusers_available is None:
        try:
            from diffusers import DDPMScheduler  # noqa: F401
            _diffusers_available = True
        except ImportError:
            _diffusers_available = False
    return _diffusers_available


class HFDiffusionPath(BasePath):
    """
    Forward diffusion path using HuggingFace diffusers schedulers.

    Uses DDPMScheduler for consistent noise schedules with DDPM/DDIM samplers.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "squaredcos_cap_v2",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        prediction_type: PredictionType = "epsilon",
        clip_sample: bool = False,
        normalize_t: bool = False,
    ) -> None:
        if not _check_diffusers():
            raise ImportError(
                "HuggingFace diffusers library is required. "
                "Install with: pip install diffusers"
            )

        from diffusers import DDPMScheduler

        self.num_train_timesteps = num_train_timesteps
        self.normalize_t = normalize_t

        self._scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            prediction_type=prediction_type,
            clip_sample=clip_sample,
        )

    @property
    def prediction_type(self) -> str:
        if self._scheduler.prediction_type == "epsilon":
            return "noise"
        if self._scheduler.prediction_type == "v_prediction":
            return "velocity"
        if self._scheduler.prediction_type == "sample":
            return "sample"
        return "unknown"

    def sample(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> PathSample:
        """
        Sample x_t from the forward diffusion process.

        Args:
            x_0: Noise samples (same shape as x_1).
            x_1: Clean data samples.
            t: Optional normalized timesteps in [0, 1]. If None, sample uniformly.
        """
        device = x_1.device
        batch = x_1.shape[0]

        if t is None:
            t_int = torch.randint(0, self.num_train_timesteps, (batch,), device=device)
            if self.normalize_t:
                t = t_int.float() / float(self.num_train_timesteps)
            else:
                t = t_int.float()
        else:
            t = t.to(device)
            if self.normalize_t:
                t_int = torch.clamp(
                    (t * self.num_train_timesteps).round().long(),
                    0,
                    self.num_train_timesteps - 1,
                )
            else:
                t_int = torch.clamp(
                    t.round().long(),
                    0,
                    self.num_train_timesteps - 1,
                )
                t = t_int.float()

        x_t = self._scheduler.add_noise(x_1, x_0, t_int)

        if self._scheduler.prediction_type == "epsilon":
            target = x_0
        elif self._scheduler.prediction_type == "v_prediction":
            target = self._scheduler.get_velocity(x_1, x_0, t_int)
        elif self._scheduler.prediction_type == "sample":
            target = x_1
        else:
            raise ValueError(f"Unsupported prediction_type: {self._scheduler.prediction_type}")

        return PathSample(x_t=x_t, t=t, target=target, x_0=x_0, x_1=x_1)

    def get_x1_from_prediction(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        """Recover x_1 from model prediction."""
        device = x_t.device
        if self.normalize_t:
            t_int = torch.clamp(
                (t * self.num_train_timesteps).round().long(),
                0,
                self.num_train_timesteps - 1,
            )
        else:
            t_int = torch.clamp(
                t.round().long(),
                0,
                self.num_train_timesteps - 1,
            )
        alpha_bar = self._scheduler.alphas_cumprod.to(device)[t_int]
        while alpha_bar.dim() < x_t.dim():
            alpha_bar = alpha_bar.unsqueeze(-1)

        if self._scheduler.prediction_type == "epsilon":
            pred_x1 = (x_t - torch.sqrt(1 - alpha_bar) * prediction) / torch.sqrt(alpha_bar)
        elif self._scheduler.prediction_type == "v_prediction":
            pred_x1 = torch.sqrt(alpha_bar) * x_t - torch.sqrt(1 - alpha_bar) * prediction
        elif self._scheduler.prediction_type == "sample":
            pred_x1 = prediction
        else:
            raise ValueError(f"Unsupported prediction_type: {self._scheduler.prediction_type}")

        return pred_x1

    @property
    def hf_scheduler(self):
        """Access the underlying HuggingFace scheduler."""
        return self._scheduler
