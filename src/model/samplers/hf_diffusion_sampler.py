"""
HuggingFace Diffusers Sampler Implementation

Wraps HuggingFace's diffusers library schedulers for the reverse diffusion process.

Supports:
- DDPMScheduler: Stochastic sampling (original DDPM)
- DDIMScheduler: Deterministic sampling (faster, same quality)

Install: pip install diffusers

Reference:
- https://huggingface.co/docs/diffusers/api/schedulers/ddpm
- https://huggingface.co/docs/diffusers/api/schedulers/ddim
"""

import torch
from typing import Callable, Optional, Literal
from tqdm import tqdm

from ..base.base_sampler import BaseSampler

# Sampler type options
SamplerType = Literal["ddpm", "ddim"]
BetaSchedule = Literal["linear", "scaled_linear", "squaredcos_cap_v2", "sigmoid"]

# Lazy import to avoid hard dependency
_diffusers_available = None


def _check_diffusers():
    """Check if diffusers library is available."""
    global _diffusers_available
    if _diffusers_available is None:
        try:
            from diffusers import DDPMScheduler, DDIMScheduler
            _diffusers_available = True
        except ImportError:
            _diffusers_available = False
    return _diffusers_available


class HFDiffusionSampler(BaseSampler):
    """
    Diffusion sampler using HuggingFace's diffusers library.

    Supports both DDPM (stochastic) and DDIM (deterministic) sampling.

    Args:
        sampler_type: Type of sampler
            - 'ddpm': Stochastic sampling (original DDPM)
            - 'ddim': Deterministic sampling (faster)
        num_train_timesteps: Number of training timesteps (must match path)
        beta_schedule: Beta schedule (must match path)
        beta_start: Starting beta (must match path)
        beta_end: Ending beta (must match path)
        prediction_type: What the model predicts (must match path)
            - 'epsilon': Noise prediction
            - 'v_prediction': Velocity prediction
            - 'sample': Direct sample prediction
        clip_sample: Whether to clip samples to [-1, 1]
        eta: DDIM eta parameter (0 = deterministic, only for DDIM)
    """

    def __init__(
        self,
        sampler_type: SamplerType = "ddim",
        num_train_timesteps: int = 1000,
        beta_schedule: BetaSchedule = "squaredcos_cap_v2",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        prediction_type: Literal["epsilon", "v_prediction", "sample"] = "epsilon",
        clip_sample: bool = False,
        eta: float = 0.0,
    ):
        if not _check_diffusers():
            raise ImportError(
                "HuggingFace diffusers library is required. "
                "Install with: pip install diffusers"
            )

        from diffusers import DDPMScheduler, DDIMScheduler

        self.sampler_type = sampler_type
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.eta = eta

        # Create the appropriate scheduler
        scheduler_kwargs = dict(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            prediction_type=prediction_type,
            clip_sample=clip_sample,
        )

        if sampler_type == "ddpm":
            self._scheduler = DDPMScheduler(**scheduler_kwargs)
        elif sampler_type == "ddim":
            self._scheduler = DDIMScheduler(**scheduler_kwargs)
        else:
            raise ValueError(f"Unknown sampler_type: {sampler_type}. Use 'ddpm' or 'ddim'.")

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
            model: Noise/velocity prediction model
            x_init: Initial samples (pure noise), shape (B, C, H, W)
            num_steps: Number of inference steps (can be less than training steps)
            show_progress: Whether to show progress bar

        Returns:
            Generated samples, shape (B, C, H, W)
        """
        device = x_init.device
        num_steps = num_steps or self.num_train_timesteps

        # Set timesteps for inference
        self._scheduler.set_timesteps(num_steps, device=device)
        timesteps = self._scheduler.timesteps

        x = x_init.clone()

        iterator = timesteps
        if show_progress:
            iterator = tqdm(timesteps, desc=f"{self.sampler_type.upper()} Sampling")

        for t in iterator:
            # Normalize timestep to [0, 1] for model input
            t_normalized = torch.full(
                (x.shape[0],),
                t.item() / self.num_train_timesteps,
                device=device
            )

            # Predict noise/velocity
            with torch.no_grad():
                model_output = model(x, t_normalized)

            # Scheduler step
            if self.sampler_type == "ddim":
                scheduler_output = self._scheduler.step(
                    model_output, t, x, eta=self.eta
                )
            else:
                scheduler_output = self._scheduler.step(model_output, t, x)

            x = scheduler_output.prev_sample

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
            model: Noise/velocity prediction model
            x_init: Initial samples
            num_steps: Number of inference steps
            save_every: Save trajectory every N steps

        Returns:
            Tuple of (final_samples, trajectory_list)
        """
        device = x_init.device
        num_steps = num_steps or self.num_train_timesteps

        self._scheduler.set_timesteps(num_steps, device=device)
        timesteps = self._scheduler.timesteps

        x = x_init.clone()
        trajectory = [x.clone()]

        for step_idx, t in enumerate(timesteps):
            t_normalized = torch.full(
                (x.shape[0],),
                t.item() / self.num_train_timesteps,
                device=device
            )

            with torch.no_grad():
                model_output = model(x, t_normalized)

            if self.sampler_type == "ddim":
                scheduler_output = self._scheduler.step(
                    model_output, t, x, eta=self.eta
                )
            else:
                scheduler_output = self._scheduler.step(model_output, t, x)

            x = scheduler_output.prev_sample

            if (step_idx + 1) % save_every == 0:
                trajectory.append(x.clone())

        return x, trajectory

    @property
    def hf_scheduler(self):
        """Access the underlying HuggingFace scheduler."""
        return self._scheduler
