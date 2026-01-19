"""
Vector Diffusion Path using HuggingFace diffusers scheduler.

Provides compute_loss_samples for vector (SE(2)) diffusion training.
"""

from __future__ import annotations

from typing import Literal

import torch

from ..common import TrainingState

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


class VectorHFDiffusionPath:
    """
    Forward diffusion path for vector targets using diffusers DDPMScheduler.

    Returns normalized time t in [0, 1] to align with sinusoidal time embeddings.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "squaredcos_cap_v2",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        prediction_type: PredictionType = "epsilon",
        clip_sample: bool = False,
    ) -> None:
        if not _check_diffusers():
            raise ImportError(
                "HuggingFace diffusers library is required. "
                "Install with: pip install diffusers"
            )

        from diffusers import DDPMScheduler

        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type

        self._scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            prediction_type=prediction_type,
            clip_sample=clip_sample,
        )

    def compute_loss_samples(self, x_0: torch.Tensor, x_1: torch.Tensor) -> TrainingState:
        """
        Sample x_t and target for diffusion training.

        Args:
            x_0: Noise samples (same shape as x_1).
            x_1: Clean data samples (vectors).
        """
        batch = x_1.shape[0]
        device = x_1.device
        t_int = torch.randint(0, self.num_train_timesteps, (batch,), device=device)
        t = t_int.float() / float(self.num_train_timesteps)

        x_t = self._scheduler.add_noise(x_1, x_0, t_int)

        if self.prediction_type == "epsilon":
            target = x_0
        elif self.prediction_type == "v_prediction":
            target = self._scheduler.get_velocity(x_1, x_0, t_int)
        elif self.prediction_type == "sample":
            target = x_1
        else:
            raise ValueError(f"Unsupported prediction_type: {self.prediction_type}")

        return TrainingState(x_t=x_t, t=t, target=target)
