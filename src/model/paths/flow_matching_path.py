"""
Flow Matching Path Implementation

Implements Optimal Transport Conditional Flow Matching (OT-CFM) path.
The model learns to predict the velocity field that transports noise to data.

Reference:
- Lipman et al., "Flow Matching for Generative Modeling" (2023)
- Tong et al., "Improving and Generalizing Flow-Based Generative Models" (2023)
"""

import torch
from typing import Optional

from ..base.base_path import BasePath, PathSample


class FlowMatchingPath(BasePath):
    """
    Optimal Transport Conditional Flow Matching path.

    Uses linear interpolation between source (noise) and target (data):
        x_t = (1 - t) * x_0 + t * x_1

    The model learns to predict the velocity:
        v = dx/dt = x_1 - x_0

    Args:
        sigma_min: Minimum standard deviation for numerical stability.
                   Set to 0 for exact linear interpolation.
    """

    def __init__(self, sigma_min: float = 0.0):
        self.sigma_min = sigma_min

    def sample(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> PathSample:
        """
        Sample from the flow matching path at time t.

        The path is defined as:
            x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1

        For sigma_min = 0 (default), this simplifies to:
            x_t = (1 - t) * x_0 + t * x_1

        Args:
            x_0: Source samples (noise), shape (B, C, H, W)
            x_1: Target samples (data), shape (B, C, H, W)
            t: Time values in [0, 1], shape (B,). If None, sample uniformly.

        Returns:
            PathSample with interpolated point and target velocity
        """
        batch_size = x_1.shape[0]
        device = x_1.device

        # Sample time uniformly if not provided
        if t is None:
            t = torch.rand(batch_size, device=device)

        # Ensure t is the right shape for broadcasting
        t_view = t.view(-1, 1, 1, 1)

        if self.sigma_min > 0:
            # With sigma_min for numerical stability
            # x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
            sigma_t = 1 - (1 - self.sigma_min) * t_view
            x_t = sigma_t * x_0 + t_view * x_1

            # Target velocity: d(x_t)/dt = x_1 - (1 - sigma_min) * x_0
            velocity = x_1 - (1 - self.sigma_min) * x_0
        else:
            # Standard linear interpolation
            # x_t = (1 - t) * x_0 + t * x_1
            x_t = (1 - t_view) * x_0 + t_view * x_1

            # Target velocity: d(x_t)/dt = x_1 - x_0
            velocity = x_1 - x_0

        return PathSample(
            x_t=x_t,
            t=t,
            target=velocity,
            x_0=x_0,
            x_1=x_1
        )

    @property
    def prediction_type(self) -> str:
        return "velocity"

    def get_x1_from_prediction(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        prediction: torch.Tensor
    ) -> torch.Tensor:
        """
        Recover x_1 from velocity prediction.

        From x_t = (1-t)*x_0 + t*x_1 and v = x_1 - x_0:
            x_1 = x_t + (1-t) * v

        Args:
            x_t: Current point on path
            t: Time values
            prediction: Predicted velocity

        Returns:
            Estimated x_1 (data)
        """
        t_view = t.view(-1, 1, 1, 1)

        if self.sigma_min > 0:
            # More complex formula with sigma_min
            # x_t = sigma_t * x_0 + t * x_1, where sigma_t = 1 - (1-sigma_min)*t
            # v = x_1 - (1-sigma_min) * x_0
            # Solving: x_1 = (x_t + (1-t)*v) / (1 - (1-sigma_min)*(1-t)/sigma_t)
            # Simplified: x_1 ≈ x_t + (1-t) * v for small sigma_min
            return x_t + (1 - t_view) * prediction
        else:
            # x_1 = x_t + (1-t) * v
            return x_t + (1 - t_view) * prediction
