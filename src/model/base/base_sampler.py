"""
Base class for samplers (DDPM, ODE, SDE, etc.)

This abstraction allows switching between different sampling strategies
without changing the model or training code.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, Any
import torch


class BaseSampler(ABC):
    """
    Abstract base class for samplers.

    A sampler defines how to generate samples from a trained model,
    starting from noise and iteratively refining to produce data.

    - DDPM: Iterative denoising with stochastic steps
    - ODE: Deterministic integration of velocity field
    - SDE: Stochastic differential equation sampling
    """

    @abstractmethod
    def sample(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_init: torch.Tensor,
        num_steps: int = 50,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate samples starting from x_init.

        Args:
            model: Neural network that takes (x, t) and returns prediction
                   (noise for diffusion, velocity for flow matching)
            x_init: Initial samples (typically Gaussian noise)
            num_steps: Number of sampling steps
            **kwargs: Additional sampler-specific arguments

        Returns:
            Generated samples (same shape as x_init)
        """
        pass

    def sample_with_trajectory(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_init: torch.Tensor,
        num_steps: int = 50,
        **kwargs
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Generate samples and return intermediate trajectory.

        Useful for visualization and debugging.

        Args:
            model: Neural network
            x_init: Initial samples
            num_steps: Number of steps

        Returns:
            Tuple of (final_samples, trajectory_list)
        """
        # Default implementation - subclasses can override for efficiency
        raise NotImplementedError("Trajectory sampling not implemented for this sampler")
