"""
Facebook Flow Matching Path Implementation

Wraps Facebook's flow_matching library path classes for use with our
GenerativeModule training infrastructure.

Supports all schedulers from Facebook's library:
- condot: Conditional Optimal Transport (recommended for most cases)
- cosine: Cosine annealing schedule
- linear_vp: Linear Variance Preserving
- vp: Variance Preserving (similar to diffusion)
- polynomial: Polynomial Convex scheduler

Install: pip install flow-matching

Reference:
- https://github.com/facebookresearch/flow_matching
- https://arxiv.org/abs/2412.06264
"""

import torch
from typing import Optional, Literal

from ..base.base_path import BasePath, PathSample

# Scheduler type options
SchedulerType = Literal["condot", "cosine", "linear_vp", "vp", "polynomial"]

# Lazy import to avoid hard dependency
_flow_matching_available = None


def _check_flow_matching():
    """Check if flow_matching library is available."""
    global _flow_matching_available
    if _flow_matching_available is None:
        try:
            from flow_matching.path import AffineProbPath
            from flow_matching.path.scheduler import CondOTScheduler
            _flow_matching_available = True
        except ImportError:
            _flow_matching_available = False
    return _flow_matching_available


def _get_scheduler(scheduler_type: SchedulerType):
    """
    Get the appropriate scheduler from Facebook's library.

    Args:
        scheduler_type: Type of scheduler to use

    Returns:
        Instantiated scheduler object
    """
    from flow_matching.path.scheduler import (
        CondOTScheduler,
        CosineScheduler,
        LinearVPScheduler,
        VPScheduler,
        PolynomialConvexScheduler,
    )

    schedulers = {
        "condot": CondOTScheduler,
        "cosine": CosineScheduler,
        "linear_vp": LinearVPScheduler,
        "vp": VPScheduler,
        "polynomial": PolynomialConvexScheduler,
    }

    if scheduler_type not in schedulers:
        valid = ", ".join(schedulers.keys())
        raise ValueError(f"Unknown scheduler: {scheduler_type}. Valid options: {valid}")

    return schedulers[scheduler_type]()


class FBFlowMatchingPath(BasePath):
    """
    Flow Matching path using Facebook's flow_matching library.

    Uses AffineProbPath with configurable scheduler for optimal transport
    conditional flow matching.

    For standard conditional OT (CondOTScheduler), the path is:
        x_t = (1 - t) * x_0 + t * x_1

    The model learns to predict the velocity field:
        v(x_t, t) ≈ dx_t/dt

    Args:
        scheduler: Scheduler type for the affine path
            - 'condot': Conditional Optimal Transport (default, recommended)
            - 'cosine': Cosine annealing schedule
            - 'linear_vp': Linear Variance Preserving
            - 'vp': Variance Preserving (similar to diffusion)
            - 'polynomial': Polynomial Convex scheduler
        sigma_min: Minimum noise level (for numerical stability)
    """

    def __init__(
        self,
        scheduler: SchedulerType = "condot",
        sigma_min: float = 0.0,
    ):
        if not _check_flow_matching():
            raise ImportError(
                "Facebook's flow_matching library is required. "
                "Install with: pip install flow-matching"
            )

        from flow_matching.path import AffineProbPath

        self.sigma_min = sigma_min
        self._scheduler_type = scheduler
        self._scheduler = _get_scheduler(scheduler)
        self._path = AffineProbPath(scheduler=self._scheduler)

    def sample(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> PathSample:
        """
        Sample from the flow matching path at time t.

        Uses Facebook's AffineProbPath internally.

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

        # Use Facebook's path to sample
        fb_sample = self._path.sample(t=t, x_0=x_0, x_1=x_1)

        # Convert Facebook's PathSample to our PathSample
        # Facebook uses: x_t, t, dx_t (velocity)
        # We use: x_t, t, target (same as dx_t for flow matching)
        return PathSample(
            x_t=fb_sample.x_t,
            t=fb_sample.t,
            target=fb_sample.dx_t,  # velocity is the prediction target
            x_0=x_0,
            x_1=x_1
        )

    @property
    def prediction_type(self) -> str:
        """Flow matching predicts velocity."""
        return "velocity"

    @property
    def scheduler_type(self) -> str:
        """Return the scheduler type being used."""
        return self._scheduler_type

    def get_x1_from_prediction(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        prediction: torch.Tensor
    ) -> torch.Tensor:
        """
        Recover x_1 from velocity prediction.

        For CondOT path with x_t = (1-t)*x_0 + t*x_1 and v = x_1 - x_0:
            x_1 = x_t + (1-t) * v

        Note: This is an approximation that works well for CondOT scheduler.
        Other schedulers may have slightly different reconstruction formulas.

        Args:
            x_t: Current point on path
            t: Time values
            prediction: Predicted velocity

        Returns:
            Estimated x_1 (data)
        """
        t_view = t.view(-1, 1, 1, 1)
        # x_1 = x_t + (1-t) * v (valid for CondOT, approximate for others)
        return x_t + (1 - t_view) * prediction

    @property
    def fb_path(self):
        """Access the underlying Facebook AffineProbPath object."""
        return self._path

    @property
    def fb_scheduler(self):
        """Access the underlying Facebook scheduler object."""
        return self._scheduler
