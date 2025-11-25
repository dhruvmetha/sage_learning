"""
Unified Generative Module

A PyTorch Lightning module that supports both diffusion and flow matching
through pluggable path and sampler components.

This design allows switching between generative methods via configuration
without changing the core training/inference code.

Usage:
    # Flow Matching
    module = GenerativeModule(
        network=DiT(...),
        path=FlowMatchingPath(),
        sampler=ODESampler(method='midpoint'),
        optimizer=...,
    )

    # Diffusion
    module = GenerativeModule(
        network=DiT(...),
        path=DiffusionPath(num_timesteps=100),
        sampler=DDPMSampler(num_timesteps=100),
        optimizer=...,
    )
"""

import torch
import torch.nn as nn
import torchvision
import lightning.pytorch as pl
from torchmetrics import MeanMetric, MinMetric
from typing import Optional, Dict, Any, Callable
from tqdm import tqdm

from .base.base_path import BasePath
from .base.base_sampler import BaseSampler


class GenerativeModule(pl.LightningModule):
    """
    Unified generative module supporting diffusion, flow matching, and more.

    The generative method is determined by the path and sampler components:
    - path: Defines how to interpolate between noise and data (training)
    - sampler: Defines how to generate samples from noise (inference)

    Args:
        network: Neural network backbone (DiT, UNet, etc.)
        path: Path implementation (FlowMatchingPath, DiffusionPath, etc.)
        sampler: Sampler implementation (ODESampler, DDPMSampler, etc.)
        optimizer: Optimizer partial function
        aux_loss_weight: Weight for auxiliary losses (e.g., dice loss)
        context_channels: Number of input context channels
        target_channels: Number of output target channels
    """

    def __init__(
        self,
        network: nn.Module,
        path: BasePath,
        sampler: BaseSampler,
        optimizer: Any,
        aux_loss_weight: float = 0.0,
        context_channels: int = 4,
        target_channels: int = 2,
    ):
        super().__init__()

        self.network = network
        self.path = path
        self.sampler = sampler
        self.optimizer_partial = optimizer

        self.aux_loss_weight = aux_loss_weight
        self.context_channels = context_channels
        self.target_channels = target_channels

        # Loss function
        self.criterion = nn.MSELoss()

        # Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_loss_best = MinMetric()

        # Save hyperparameters (excluding non-serializable objects)
        self.save_hyperparameters(ignore=["network", "path", "sampler", "optimizer"])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor (context + noisy/interpolated target)
            t: Time values, shape (B,) or (B, 1)

        Returns:
            Model prediction (noise or velocity depending on path)
        """
        # Ensure t has correct shape for the network
        if t.dim() == 1:
            t_input = t
        else:
            t_input = t.squeeze(-1)

        return self.network(x, t_input)

    def _build_context(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Build context tensor from batch.

        Concatenates: robot, goal, movable_objects, static_objects,
        and optionally target_object and coord_grid.
        """
        parts = [
            batch['robot'],
            batch['goal'],
            batch['movable_objects'],
            batch['static_objects'],
        ]

        # Optional: target object channel
        if 'target_object' in batch and batch['target_object'] is not None:
            target_obj = batch['target_object']
            if target_obj.dim() == 3:
                target_obj = target_obj.unsqueeze(1)
            parts.append(target_obj)

        # Optional: coordinate grid
        if 'coord_grid' in batch:
            parts.append(batch['coord_grid'])

        return torch.cat(parts, dim=1)

    def _build_target(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Build target tensor from batch.

        Concatenates object_mask and goal_mask, clamped to [-1, 1].
        """
        target = torch.cat([batch['object_mask'], batch['goal_mask']], dim=1)
        return torch.clamp(target, -1.0, 1.0)

    def _compute_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_1: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss with optional auxiliary losses.

        Args:
            prediction: Model output (noise or velocity)
            target: Ground truth (noise or velocity)
            x_t: Current interpolated/noised point
            t: Time values
            x_1: Original data (for auxiliary losses)

        Returns:
            Total loss
        """
        # Primary loss: MSE between prediction and target
        primary_loss = self.criterion(prediction, target)

        if self.aux_loss_weight > 0:
            # Auxiliary loss: Dice loss on reconstructed x_1
            x_1_pred = self.path.get_x1_from_prediction(x_t, t, prediction)

            # Clamp and convert to [0, 1] for dice loss
            x_1_pred = torch.sigmoid(torch.clamp(x_1_pred, -10.0, 10.0))
            x_1_target = (x_1 + 1) / 2  # Convert from [-1, 1] to [0, 1]

            aux_loss = self._dice_loss(x_1_pred, x_1_target)

            # Check for NaN
            if torch.isnan(aux_loss):
                return primary_loss

            return primary_loss + self.aux_loss_weight * aux_loss

        return primary_loss

    @staticmethod
    def _dice_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1.0
    ) -> torch.Tensor:
        """
        Compute Dice loss for auxiliary supervision.

        Args:
            pred: Predictions in [0, 1]
            target: Targets in [0, 1]
            smooth: Smoothing factor for numerical stability

        Returns:
            Dice loss (1 - Dice coefficient)
        """
        pred = pred.flatten(1)
        target = target.flatten(1)

        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)

        dice = (2 * intersection + smooth) / (union + smooth + 1e-8)

        return 1 - dice.mean()

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Optional[torch.Tensor]:
        """
        Training step: sample from path and compute loss.
        """
        # Build inputs
        context = self._build_context(batch)
        x_1 = self._build_target(batch)
        x_0 = torch.randn_like(x_1)

        # Sample from path
        path_sample = self.path.sample(x_0=x_0, x_1=x_1)

        # Build model input: [context, x_t]
        model_input = torch.cat([context, path_sample.x_t], dim=1)

        # Get prediction
        prediction = self(model_input, path_sample.t)

        # Compute loss
        loss = self._compute_loss(
            prediction=prediction,
            target=path_sample.target,
            x_t=path_sample.x_t,
            t=path_sample.t,
            x_1=x_1
        )

        # Handle NaN loss
        if loss is None or torch.isnan(loss):
            return None

        # Log metrics
        self.train_loss(loss)
        self.log("train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Optional[torch.Tensor]:
        """
        Validation step: same as training but with logging.
        """
        context = self._build_context(batch)
        x_1 = self._build_target(batch)
        x_0 = torch.randn_like(x_1)

        path_sample = self.path.sample(x_0=x_0, x_1=x_1)
        model_input = torch.cat([context, path_sample.x_t], dim=1)

        prediction = self(model_input, path_sample.t)

        loss = self._compute_loss(
            prediction=prediction,
            target=path_sample.target,
            x_t=path_sample.x_t,
            t=path_sample.t,
            x_1=x_1
        )

        if loss is None or torch.isnan(loss):
            return None

        self.val_loss(loss)
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Save first batch for sample generation
        if batch_idx == 0:
            target_object = batch.get('target_object')
            self._validation_context = context
            self._validation_target = x_1
            self._validation_target_object = target_object

        return loss

    def on_validation_epoch_end(self):
        """Log best validation loss and generate samples."""
        self.val_loss_best(self.val_loss.compute())
        self.log("val_loss_best", self.val_loss_best, prog_bar=True)

        # Generate validation samples
        if hasattr(self, '_validation_context'):
            self._generate_validation_samples()

    def _generate_validation_samples(self):
        """Generate and log samples during validation."""
        context = self._validation_context
        target = self._validation_target

        num_samples = min(16, context.size(0))
        context_samples = context[:num_samples]
        target_samples = target[:num_samples]

        image_size = context_samples.shape[-1]

        # Generate samples
        with torch.no_grad():
            generated_1 = self.sample(context_samples, num_samples=1, num_steps=20)
            generated_2 = self.sample(context_samples, num_samples=1, num_steps=20)
            generated_3 = self.sample(context_samples, num_samples=1, num_steps=20)
            generated_4 = self.sample(context_samples, num_samples=1, num_steps=20)

        # Log to tensorboard if available
        if hasattr(self.logger, 'experiment'):
            def _to_display(x: torch.Tensor) -> torch.Tensor:
                return torch.clamp((x + 1) / 2, 0, 1)

            def _masks_to_rgb(masks: torch.Tensor) -> torch.Tensor:
                rgb = torch.zeros(num_samples, 3, image_size, image_size, device=masks.device)
                channels = min(3, masks.shape[1])
                for ch in range(channels):
                    rgb[:, ch, :, :] = masks[:, ch, :, :]
                return rgb

            # Visualize input scene
            robot = _to_display(context_samples[:, 0:1])
            goal = _to_display(context_samples[:, 1:2])
            movable = _to_display(context_samples[:, 2:3])
            static = _to_display(context_samples[:, 3:4])

            scene_vis = torch.zeros(num_samples, 3, image_size, image_size, device=context_samples.device)
            scene_vis[:, 0, :, :] = robot[:, 0, :, :]
            scene_vis[:, 1, :, :] = goal[:, 0, :, :]
            scene_vis[:, 2, :, :] = movable[:, 0, :, :]

            # Visualize targets
            object_vis = torch.zeros(num_samples, 3, image_size, image_size, device=context_samples.device)
            object_vis[:, 0, :, :] = _to_display(target_samples[:, 0:1])[:, 0, :, :]

            goal_vis = torch.zeros(num_samples, 3, image_size, image_size, device=context_samples.device)
            if target_samples.shape[1] > 1:
                goal_vis[:, 1, :, :] = _to_display(target_samples[:, 1:2])[:, 0, :, :]

            # Build comparison grid
            comparison_parts = [scene_vis, object_vis, goal_vis]

            for generated in [generated_1, generated_2, generated_3, generated_4]:
                comparison_parts.append(_masks_to_rgb(_to_display(generated)))

            comparison = torch.cat(comparison_parts, dim=0)
            grid = torchvision.utils.make_grid(comparison, nrow=num_samples, normalize=True)
            self.logger.experiment.add_image(
                f'Validation_Samples_{self.path.prediction_type}',
                grid,
                self.current_epoch
            )

    def sample(
        self,
        context: torch.Tensor,
        num_samples: int = 1,
        num_steps: int = 20,
        show_progress: bool = False
    ) -> torch.Tensor:
        """
        Generate samples given context.

        Args:
            context: Context tensor, shape (B, C_context, H, W)
            num_samples: Number of samples per context (currently must be 1)
            num_steps: Number of sampling steps
            show_progress: Whether to show progress bar

        Returns:
            Generated samples, shape (B, C_target, H, W)
        """
        B, C, H, W = context.shape

        # Initialize from noise
        x_init = torch.randn(B, self.target_channels, H, W, device=context.device)

        # Create model function that includes context
        def model_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            model_input = torch.cat([context, x], dim=1)
            return self(model_input, t)

        # Sample using the configured sampler
        return self.sampler.sample(
            model=model_fn,
            x_init=x_init,
            num_steps=num_steps,
            show_progress=show_progress
        )

    def sample_from_model(
        self,
        inp: torch.Tensor,
        tgt_size: int = 2,
        samples: int = 32,
        num_steps: int = 20
    ) -> torch.Tensor:
        """
        Legacy interface for backward compatibility with existing inference code.

        Args:
            inp: Input context tensor
            tgt_size: Number of target channels (default 2)
            samples: Number of samples to generate
            num_steps: Number of sampling steps

        Returns:
            Generated samples, shape (samples, tgt_size, H, W)
        """
        # Repeat input for multiple samples
        inp_repeated = inp.repeat(samples, 1, 1, 1)

        # Initialize from same noise for consistency (like original code)
        x_init = torch.randn(1, tgt_size, inp.shape[2], inp.shape[3], device=inp.device)
        x_init = x_init.repeat(samples, 1, 1, 1)

        def model_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            model_input = torch.cat([inp_repeated, x], dim=1)
            return self(model_input, t)

        return self.sampler.sample(
            model=model_fn,
            x_init=x_init,
            num_steps=num_steps,
            show_progress=True
        )

    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = self.optimizer_partial(params=self.parameters())
        return {
            "optimizer": optimizer,
            "gradient_clip_val": 1.0,
        }
