"""
Base class for generative paths (diffusion, flow matching, etc.)

This abstraction allows switching between different generative methods
by simply changing the path implementation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class PathSample:
    """
    Output of path sampling - unified interface for both diffusion and flow matching.

    Attributes:
        x_t: Interpolated/noised point at time t
        t: Time values (continuous in [0, 1])
        target: What the model should predict (noise for diffusion, velocity for flow matching)
        x_0: Source samples (typically noise)
        x_1: Target samples (data)
    """
    x_t: torch.Tensor
    t: torch.Tensor
    target: torch.Tensor
    x_0: torch.Tensor
    x_1: torch.Tensor


class BasePath(ABC):
    """
    Abstract base class for generative paths.

    A "path" defines how we interpolate between source (noise) and target (data)
    distributions, and what the model should learn to predict.

    - Diffusion: Noisy interpolation, model predicts noise
    - Flow Matching: Linear interpolation, model predicts velocity
    """

    @abstractmethod
    def sample(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> PathSample:
        """
        Sample from the path at time t.

        Args:
            x_0: Source distribution samples (typically Gaussian noise)
            x_1: Target distribution samples (data)
            t: Time values in [0, 1]. If None, sample uniformly.

        Returns:
            PathSample containing interpolated point and prediction target
        """
        pass

    @property
    @abstractmethod
    def prediction_type(self) -> str:
        """
        What the model predicts.

        Returns one of: 'noise', 'velocity', 'x0', 'score'
        """
        pass

    @abstractmethod
    def get_x1_from_prediction(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        prediction: torch.Tensor
    ) -> torch.Tensor:
        """
        Recover x_1 (data) from model prediction.

        This is useful for auxiliary losses (e.g., dice loss on reconstructed images)
        during training.

        Args:
            x_t: Current interpolated point
            t: Time values
            prediction: Model's prediction (noise or velocity)

        Returns:
            Estimated x_1 (data)
        """
        pass
