"""
ODE Sampler for Flow Matching

Integrates the learned velocity field using numerical ODE solvers.
Supports multiple integration methods: Euler, Midpoint, RK4, and adaptive solvers.

This sampler is used with FlowMatchingPath where the model predicts velocity.
"""

import torch
from typing import Callable, Optional, Literal
from tqdm import tqdm

from ..base.base_sampler import BaseSampler


class ODESampler(BaseSampler):
    """
    ODE-based sampler for flow matching models.

    Integrates the velocity field v(x, t) from t=0 (noise) to t=1 (data)
    using numerical ODE solvers.

    Args:
        method: ODE solver method
            - 'euler': First-order Euler method (fastest, least accurate)
            - 'midpoint': Second-order midpoint method (good balance)
            - 'rk4': Fourth-order Runge-Kutta (most accurate fixed-step)
            - 'heun': Heun's method (second-order, similar to midpoint)
        atol: Absolute tolerance (for adaptive methods, future use)
        rtol: Relative tolerance (for adaptive methods, future use)
    """

    def __init__(
        self,
        method: Literal["euler", "midpoint", "rk4", "heun"] = "midpoint",
        atol: float = 1e-5,
        rtol: float = 1e-5
    ):
        self.method = method
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
            num_steps: Number of integration steps
            show_progress: Whether to show progress bar

        Returns:
            Generated samples (data at t=1), shape (B, C, H, W)
        """
        device = x_init.device
        dt = 1.0 / num_steps
        x = x_init.clone()

        iterator = range(num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="ODE Sampling")

        for i in iterator:
            t = torch.full((x.shape[0],), i * dt, device=device)

            if self.method == "euler":
                x = self._euler_step(model, x, t, dt)
            elif self.method == "midpoint":
                x = self._midpoint_step(model, x, t, dt)
            elif self.method == "rk4":
                x = self._rk4_step(model, x, t, dt)
            elif self.method == "heun":
                x = self._heun_step(model, x, t, dt)
            else:
                raise ValueError(f"Unknown method: {self.method}")

        return x

    def _euler_step(
        self,
        model: Callable,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """First-order Euler step: x_{n+1} = x_n + dt * v(x_n, t_n)"""
        v = model(x, t)
        return x + dt * v

    def _midpoint_step(
        self,
        model: Callable,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        Second-order midpoint step:
            k1 = v(x_n, t_n)
            x_{n+1} = x_n + dt * v(x_n + 0.5*dt*k1, t_n + 0.5*dt)
        """
        k1 = model(x, t)
        x_mid = x + 0.5 * dt * k1
        t_mid = t + 0.5 * dt
        v_mid = model(x_mid, t_mid)
        return x + dt * v_mid

    def _heun_step(
        self,
        model: Callable,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        Second-order Heun's method (improved Euler):
            k1 = v(x_n, t_n)
            k2 = v(x_n + dt*k1, t_n + dt)
            x_{n+1} = x_n + 0.5*dt*(k1 + k2)
        """
        k1 = model(x, t)
        x_euler = x + dt * k1
        t_next = t + dt
        k2 = model(x_euler, t_next)
        return x + 0.5 * dt * (k1 + k2)

    def _rk4_step(
        self,
        model: Callable,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        Fourth-order Runge-Kutta step:
            k1 = v(x_n, t_n)
            k2 = v(x_n + 0.5*dt*k1, t_n + 0.5*dt)
            k3 = v(x_n + 0.5*dt*k2, t_n + 0.5*dt)
            k4 = v(x_n + dt*k3, t_n + dt)
            x_{n+1} = x_n + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        """
        k1 = model(x, t)
        k2 = model(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = model(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = model(x + dt * k3, t + dt)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

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
            save_every: Save trajectory every N steps

        Returns:
            Tuple of (final_samples, trajectory_list)
        """
        device = x_init.device
        dt = 1.0 / num_steps
        x = x_init.clone()
        trajectory = [x.clone()]

        for i in range(num_steps):
            t = torch.full((x.shape[0],), i * dt, device=device)

            if self.method == "euler":
                x = self._euler_step(model, x, t, dt)
            elif self.method == "midpoint":
                x = self._midpoint_step(model, x, t, dt)
            elif self.method == "rk4":
                x = self._rk4_step(model, x, t, dt)
            elif self.method == "heun":
                x = self._heun_step(model, x, t, dt)

            if (i + 1) % save_every == 0:
                trajectory.append(x.clone())

        return x, trajectory
