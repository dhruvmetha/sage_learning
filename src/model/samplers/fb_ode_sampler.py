"""
Facebook Flow Matching ODE Sampler

Uses Facebook's flow_matching library for ODE integration.
Supports adaptive solvers (dopri5) and various fixed-step methods.

Install: pip install flow-matching

Reference:
- https://github.com/facebookresearch/flow_matching
- https://arxiv.org/abs/2412.06264
"""

import torch
from typing import Callable, Optional, Literal
from tqdm import tqdm

from ..base.base_sampler import BaseSampler

# Lazy import to avoid hard dependency
_flow_matching_available = None


def _check_flow_matching():
    """Check if flow_matching library is available."""
    global _flow_matching_available
    if _flow_matching_available is None:
        try:
            from flow_matching.solver import ODESolver
            from flow_matching.utils import ModelWrapper
            _flow_matching_available = True
        except ImportError:
            _flow_matching_available = False
    return _flow_matching_available


class FBODESampler(BaseSampler):
    """
    ODE sampler using Facebook's flow_matching library.

    Provides access to adaptive solvers (dopri5, tsit5) which can be more
    efficient than fixed-step methods for some problems.

    Args:
        method: ODE solver method
            Fixed-step: 'euler', 'midpoint', 'rk4'
            Adaptive: 'dopri5', 'tsit5' (require torchdyn)
        step_size: Step size for fixed-step methods (ignored for adaptive)
        atol: Absolute tolerance for adaptive solvers
        rtol: Relative tolerance for adaptive solvers
    """

    def __init__(
        self,
        method: Literal["euler", "midpoint", "rk4", "dopri5", "tsit5"] = "midpoint",
        step_size: Optional[float] = None,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ):
        if not _check_flow_matching():
            raise ImportError(
                "Facebook's flow_matching library is required. "
                "Install with: pip install flow-matching"
            )

        self.method = method
        self.step_size = step_size
        self.atol = atol
        self.rtol = rtol

    def sample(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_init: torch.Tensor,
        num_steps: int = 20,
        show_progress: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate samples by integrating the velocity field.

        Args:
            model: Velocity model v(x, t) that takes (x, t) and returns velocity
            x_init: Initial samples (noise at t=0), shape (B, C, H, W)
            num_steps: Number of integration steps (for time grid)
            show_progress: Whether to show progress bar (not used, FB solver has its own)

        Returns:
            Generated samples (data at t=1), shape (B, C, H, W)
        """
        from flow_matching.solver import ODESolver
        from flow_matching.utils import ModelWrapper

        device = x_init.device

        # Wrap the model for Facebook's interface
        wrapped_model = _VelocityModelWrapper(model)

        # Create time grid
        time_grid = torch.linspace(0, 1, num_steps + 1, device=device)

        # Compute step size if not provided
        step_size = self.step_size if self.step_size is not None else 1.0 / num_steps

        # Create solver
        solver = ODESolver(velocity_model=wrapped_model)

        # Sample
        sol = solver.sample(
            time_grid=time_grid,
            x_init=x_init,
            method=self.method,
            step_size=step_size,
            atol=self.atol,
            rtol=self.rtol,
            return_intermediates=False,
        )

        return sol

    def sample_with_trajectory(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_init: torch.Tensor,
        num_steps: int = 20,
        save_every: int = 1,
        **kwargs
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Generate samples and save intermediate trajectory.

        Args:
            model: Velocity model
            x_init: Initial samples
            num_steps: Number of steps
            save_every: Save trajectory every N steps (ignored, returns all intermediates)

        Returns:
            Tuple of (final_samples, trajectory_list)
        """
        from flow_matching.solver import ODESolver

        device = x_init.device
        wrapped_model = _VelocityModelWrapper(model)
        time_grid = torch.linspace(0, 1, num_steps + 1, device=device)
        step_size = self.step_size if self.step_size is not None else 1.0 / num_steps

        solver = ODESolver(velocity_model=wrapped_model)

        sol = solver.sample(
            time_grid=time_grid,
            x_init=x_init,
            method=self.method,
            step_size=step_size,
            atol=self.atol,
            rtol=self.rtol,
            return_intermediates=True,
        )

        # sol shape: (num_steps+1, B, C, H, W) when return_intermediates=True
        trajectory = [sol[i] for i in range(sol.shape[0])]
        final = sol[-1]

        return final, trajectory


class _VelocityModelWrapper:
    """
    Wrapper to adapt our model interface to Facebook's ModelWrapper interface.

    Facebook's ODESolver expects a model with forward(x, t, **extras) signature.
    Our models use forward(x, t) where x is [context, noisy] concatenated.
    """

    def __init__(self, model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        self.model = model

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        return self.model(x, t)

    def __call__(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        return self.forward(x, t, **extras)
