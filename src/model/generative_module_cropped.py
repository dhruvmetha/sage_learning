"""
Generative Module for Cropped Target Prediction

This module handles models that take separate context and noisy input:
- Context: (B, 5, context_size, context_size) - full scene
- Noisy target: (B, 1, crop_size, crop_size) - center crop
- Output: (B, 1, crop_size, crop_size)

The context is passed separately to the network for conditioning via AdaLN.
"""

import torch
import torch.nn as nn
import torchvision
import lightning.pytorch as pl
from torchmetrics import MeanMetric, MinMetric
from typing import Optional, Dict, Any

from .base.base_path import BasePath
from .base.base_sampler import BaseSampler


class GenerativeModuleCropped(pl.LightningModule):
    """
    Generative module for cropped target prediction with separate context.

    Args:
        network: Neural network (DiTCropped)
        path: Path implementation (FlowMatchingPath, etc.)
        sampler: Sampler implementation (ODESampler, etc.)
        optimizer: Optimizer partial function
        context_size: Size of context images (default 64)
        crop_size: Size of cropped target (default 24)
    """

    def __init__(
        self,
        network: nn.Module,
        path: BasePath,
        sampler: BaseSampler,
        optimizer: Any,
        context_size: int = 64,
        crop_size: int = 24,
        warmup_steps: int = 0,
        decay_steps: int = 0,
        end_lr: float = 0.0,
        use_multihorizon: bool = False,
        target_channels: int = 1,
    ):
        super().__init__()

        self.network = network
        self.path = path
        self.sampler = sampler
        self.optimizer_partial = optimizer

        self.context_size = context_size
        self.crop_size = crop_size

        # Multi-horizon settings
        self.use_multihorizon = use_multihorizon
        self.target_channels = target_channels

        # LR schedule parameters
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.end_lr = end_lr

        # Loss
        self.criterion = nn.MSELoss(reduction='none')  # Per-element for masking

        # Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_loss_best = MinMetric()

        self.save_hyperparameters(ignore=["network", "path", "sampler", "optimizer"])

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with separate noisy input and context.

        Args:
            x_noisy: Noisy target (B, 1, crop_size, crop_size)
            t: Timesteps (B,)
            context: Full context (B, 5, context_size, context_size)

        Returns:
            Prediction (B, 1, crop_size, crop_size)
        """
        if t.dim() > 1:
            t = t.squeeze(-1)
        return self.network(x_noisy, t, context)

    def _build_context(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Stack context channels from batch dict.

        Args:
            batch: Contains 'context' dict with static, movable, etc.

        Returns:
            Stacked context tensor (B, 5, context_size, context_size)
        """
        ctx = batch['context']
        return torch.cat([
            ctx['static'],
            ctx['movable'],
            ctx['target_object'],
            ctx['robot_region'],
            ctx['goal_sample_region'],
        ], dim=1)

    def _build_target(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Get cropped target from batch."""
        if self.use_multihorizon:
            # Multi-horizon: target_goals is [B, 2, crop_size, crop_size]
            return torch.clamp(batch['target_goals'], -1, 1)
        else:
            # Single-horizon: target_goal is [B, 1, crop_size, crop_size]
            return torch.clamp(batch['target_goal'], -1, 1)

    def training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> Optional[torch.Tensor]:
        # Build inputs
        context = self._build_context(batch)  # (B, 5, 64, 64)
        x_1 = self._build_target(batch)  # (B, C, 24, 24) where C=1 or 2
        x_0 = torch.randn_like(x_1)  # Noise at crop size

        # Sample from path
        path_sample = self.path.sample(x_0=x_0, x_1=x_1)

        # Get prediction with separate context
        prediction = self(path_sample.x_t, path_sample.t, context)

        # Loss computation
        per_element_loss = self.criterion(prediction, path_sample.target)

        if self.use_multihorizon:
            # Channel 0 (goal_mask_a1): Always supervised
            loss_ch0 = per_element_loss[:, 0].mean()

            # Channel 1 (goal_mask_a2): Only supervised when solution_depth >= 2
            solution_depth = batch['solution_depth']  # (B,)
            if isinstance(solution_depth, torch.Tensor):
                mask = (solution_depth >= 2).float()
            else:
                mask = torch.tensor([1.0 if d >= 2 else 0.0 for d in solution_depth],
                                   device=prediction.device)
            # Expand mask to match spatial dims: (B,) -> (B, 1, 1)
            mask = mask.view(-1, 1, 1)
            loss_ch1_masked = (per_element_loss[:, 1] * mask).sum() / (mask.sum() + 1e-8)

            loss = loss_ch0 + loss_ch1_masked
        else:
            loss = per_element_loss.mean()

        if torch.isnan(loss):
            return None

        self.train_loss(loss)
        self.log("train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> Optional[torch.Tensor]:
        context = self._build_context(batch)
        x_1 = self._build_target(batch)
        x_0 = torch.randn_like(x_1)

        path_sample = self.path.sample(x_0=x_0, x_1=x_1)
        prediction = self(path_sample.x_t, path_sample.t, context)
        per_element_loss = self.criterion(prediction, path_sample.target)

        if self.use_multihorizon:
            loss_ch0 = per_element_loss[:, 0].mean()
            solution_depth = batch['solution_depth']
            if isinstance(solution_depth, torch.Tensor):
                mask = (solution_depth >= 2).float()
            else:
                mask = torch.tensor([1.0 if d >= 2 else 0.0 for d in solution_depth],
                                   device=prediction.device)
            mask = mask.view(-1, 1, 1)
            loss_ch1_masked = (per_element_loss[:, 1] * mask).sum() / (mask.sum() + 1e-8)
            loss = loss_ch0 + loss_ch1_masked
        else:
            loss = per_element_loss.mean()

        if torch.isnan(loss):
            return None

        self.val_loss(loss)
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Save for sample generation
        if batch_idx == 0:
            self._validation_context = context
            self._validation_target = x_1

        return loss

    def on_validation_epoch_end(self):
        self.val_loss_best(self.val_loss.compute())
        self.log("val_loss_best", self.val_loss_best, prog_bar=True)

        if hasattr(self, '_validation_context'):
            self._generate_validation_samples()

        self._validation_context = None
        self._validation_target = None
        torch.cuda.empty_cache()

    def _generate_validation_samples(self):
        """Generate and log samples during validation."""
        context = self._validation_context
        target = self._validation_target

        num_examples = min(8, context.size(0))

        if hasattr(self.logger, 'experiment'):
            import wandb

            def _to_display(x: torch.Tensor) -> torch.Tensor:
                return torch.clamp((x + 1) / 2, 0, 1)

            log_dict = {}
            for i in range(num_examples):
                ctx = context[i:i+1]
                gt = _to_display(target[i:i+1])[0, 0]  # (crop_size, crop_size)

                # Generate predictions
                with torch.no_grad():
                    pred1 = _to_display(self.sample(ctx, num_steps=20)[0, 0])
                    pred2 = _to_display(self.sample(ctx, num_steps=20)[0, 0])
                    pred3 = _to_display(self.sample(ctx, num_steps=20)[0, 0])
                    pred4 = _to_display(self.sample(ctx, num_steps=20)[0, 0])

                # Context visualization (64x64)
                static = _to_display(ctx[0, 0:1])[0]
                movable = _to_display(ctx[0, 1:2])[0]
                target_obj = _to_display(ctx[0, 2:3])[0]
                robot_region = _to_display(ctx[0, 3:4])[0]
                goal_region = _to_display(ctx[0, 4:5])[0]

                # Scene image (64x64)
                img_scene = torch.zeros(1, 3, self.context_size, self.context_size, device=ctx.device)
                img_scene[0, 0] = static
                img_scene[0, 1] = movable
                img_scene[0, 2] = target_obj

                # Reachability image (64x64)
                img_reach = torch.zeros(1, 3, self.context_size, self.context_size, device=ctx.device)
                img_reach[0, 0] = robot_region
                img_reach[0, 1] = goal_region
                img_reach[0, 2] = target_obj

                # GT and predictions: R=static, G=prediction (padded), B=target_obj
                # Pad predictions to context_size if crop is smaller
                def make_pred_img(pred_loc):
                    img = torch.zeros(1, 3, self.context_size, self.context_size, device=ctx.device)
                    img[0, 0] = static      # R = static obstacles
                    img[0, 2] = target_obj  # B = target object
                    # G = prediction (padded to center if needed)
                    if self.crop_size < self.context_size:
                        pad_total = self.context_size - self.crop_size
                        pad_before = pad_total // 2
                        img[0, 1, pad_before:pad_before+self.crop_size, pad_before:pad_before+self.crop_size] = pred_loc
                    else:
                        img[0, 1] = pred_loc
                    return img

                img_gt = make_pred_img(gt)
                img_p1 = make_pred_img(pred1)
                img_p2 = make_pred_img(pred2)
                img_p3 = make_pred_img(pred3)
                img_p4 = make_pred_img(pred4)

                row = torch.cat([img_scene, img_reach, img_gt, img_p1, img_p2, img_p3, img_p4], dim=0)
                grid = torchvision.utils.make_grid(row, nrow=7, normalize=True, padding=2)

                grid_np = grid.cpu().permute(1, 2, 0).numpy()
                caption = f"Epoch {self.current_epoch} | Scene | Reach | GT | Pred1-4 (cropped {self.crop_size}x{self.crop_size})"
                log_dict[f'val_sample_{i+1}'] = wandb.Image(grid_np, caption=caption)

            self.logger.experiment.log(log_dict)

    def sample(
        self,
        context: torch.Tensor,
        num_samples: int = 1,
        num_steps: int = 20,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples given context.

        Args:
            context: Full context (B, 5, context_size, context_size)
            num_samples: Ignored (batch determines samples)
            num_steps: Number of sampling steps
            show_progress: Show progress bar

        Returns:
            Generated samples (B, C, crop_size, crop_size) where C=target_channels
        """
        B = context.size(0)

        # Initialize noise at crop size with appropriate channels
        x_init = torch.randn(B, self.target_channels, self.crop_size, self.crop_size, device=context.device)

        # Model function with context
        def model_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self(x, t, context)

        return self.sampler.sample(
            model=model_fn,
            x_init=x_init,
            num_steps=num_steps,
            show_progress=show_progress,
        )

    def sample_from_model(
        self,
        context: torch.Tensor,
        samples: int = 32,
        num_steps: int = 20,
    ) -> torch.Tensor:
        """
        Generate multiple samples for inference.

        Args:
            context: Context tensor (1, 5, context_size, context_size)
            samples: Number of samples to generate
            num_steps: Sampling steps

        Returns:
            Generated samples (samples, C, context_size, context_size)
            where C=target_channels.
            Note: Output is padded from crop_size to context_size for
            compatibility with existing inference pipeline.
        """
        # Repeat context for multiple samples
        context_repeated = context.repeat(samples, 1, 1, 1)

        x_init = torch.randn(samples, self.target_channels, self.crop_size, self.crop_size, device=context.device)

        def model_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self(x, t, context_repeated)

        samples_out = self.sampler.sample(
            model=model_fn,
            x_init=x_init,
            num_steps=num_steps,
            show_progress=True,
        )

        # Pad to context size for compatibility with inference pipeline
        if self.crop_size < self.context_size:
            import torch.nn.functional as F
            pad = (self.context_size - self.crop_size) // 2
            samples_out = F.pad(samples_out, (pad, pad, pad, pad), value=-1)

        return samples_out

    def configure_optimizers(self):
        optimizer = self.optimizer_partial(params=self.parameters())

        # If no scheduler params, return optimizer only
        if self.warmup_steps == 0 and self.decay_steps == 0:
            return {
                "optimizer": optimizer,
                "gradient_clip_val": 1.0,
            }

        # Get base LR from optimizer
        base_lr = optimizer.param_groups[0]["lr"]

        def lr_lambda(step):
            # Warmup phase
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)

            # Decay phase (cosine decay from base_lr to end_lr)
            if self.decay_steps > 0:
                decay_progress = min(1.0, (step - self.warmup_steps) / self.decay_steps)
                # Cosine decay
                lr_mult = self.end_lr / base_lr + (1 - self.end_lr / base_lr) * 0.5 * (
                    1 + torch.cos(torch.tensor(decay_progress * 3.14159)).item()
                )
                return lr_mult

            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
            "gradient_clip_val": 1.0,
        }
