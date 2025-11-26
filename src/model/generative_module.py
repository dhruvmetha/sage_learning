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
from typing import Optional, Dict, Any

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
        context_channels: Number of input context channels (default 5: robot, goal, movable, static, target_object)
        target_channels: Number of output target channels (default 1: target_goal only)
    """

    def __init__(
        self,
        network: nn.Module,
        path: BasePath,
        sampler: BaseSampler,
        optimizer: Any,
        aux_loss_weight: float = 0.0,
        context_channels: int = 5,
        target_channels: int = 1,
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

        Concatenates: robot, goal, movable_objects, static_objects, target_object, and optionally coord_grid.
        """
        parts = [
            batch['robot'],
            batch['goal'],
            batch['movable'],
            batch['static'],
            batch['target_object']
        ]

        # Optional: coordinate grid
        if 'coord_grid' in batch:
            parts.append(batch['coord_grid'])

        return torch.cat(parts, dim=1)

    def _build_target(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Build target tensor from batch, clamped to [-1, 1].
        """
        return torch.clamp(batch['target_goal'], -1, 1)

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

            aux_loss = self._focal_dice_loss(x_1_pred, x_1_target)

            # Check for NaN
            if torch.isnan(aux_loss):
                return primary_loss

            return primary_loss + self.aux_loss_weight * aux_loss

        return primary_loss

    @staticmethod
    def _focal_dice_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float=0.5,
        gamma: float=2.0,
        smooth: float=1.0
    ) -> torch.Tensor:
        # pred, target: B×1×H×W in [0,1]
        pred = pred.flatten(1)
        target = target.flatten(1)
        
        # Clamp predictions to prevent extreme values
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
        
        focal_weight = alpha * (1 - pred) ** gamma * target + (1 - alpha) * pred ** gamma * (1 - target)
        
        intersection = (pred * target * focal_weight).sum(dim=1)
        denom = (pred * focal_weight).sum(dim=1) + (target * focal_weight).sum(dim=1)
        
        # Add numerical stability
        dice = (2 * intersection + smooth) / (denom + smooth + 1e-8)
        
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
        """Generate and log samples during validation.

        Logs 8 images, each showing a different validation example.
        Each image has 4 panels:
        1. Scene: robot(R) + robot_goal(G) + static(B) + movable(B)
        2. Target object context: robot(R) + robot_goal(G) + static(B) + target_object(cyan)
        3. Ground truth: robot(R) + robot_goal(G) + target_object(B) + ground_truth_location(G)
        4. Prediction: robot(R) + robot_goal(G) + static(B) + target_object(cyan) + prediction(G)
        """
        context = self._validation_context
        target = self._validation_target

        num_examples = min(8, context.size(0))
        image_size = context.shape[-1]

        # Log images if logger is available
        if hasattr(self.logger, 'experiment'):
            import wandb

            def _to_display(x: torch.Tensor) -> torch.Tensor:
                return torch.clamp((x + 1) / 2, 0, 1)

            log_dict = {}
            for i in range(num_examples):
                # Get single example
                # Context channels: 0=robot, 1=robot_goal, 2=movable, 3=static, 4=target_object
                ctx = context[i:i+1]  # (1, 5, H, W)
                gt_target = target[i:i+1]   # (1, 1, H, W) - ground truth target location

                robot = _to_display(ctx[0, 0:1])[0]        # robot position
                robot_goal = _to_display(ctx[0, 1:2])[0]   # robot goal
                movable = _to_display(ctx[0, 2:3])[0]      # movable objects
                static = _to_display(ctx[0, 3:4])[0]       # static obstacles
                target_obj = _to_display(ctx[0, 4:5])[0]   # target/selected object
                gt_location = _to_display(gt_target[0, 0:1])[0]  # ground truth target location

                # Generate 4 predictions
                with torch.no_grad():
                    pred_1 = _to_display(self.sample(ctx, num_steps=20)[0, 0:1])[0]
                    pred_2 = _to_display(self.sample(ctx, num_steps=20)[0, 0:1])[0]
                    pred_3 = _to_display(self.sample(ctx, num_steps=20)[0, 0:1])[0]
                    pred_4 = _to_display(self.sample(ctx, num_steps=20)[0, 0:1])[0]

                # Image 1: Scene (robot=R, robot_goal=G, static+movable=B)
                img1 = torch.zeros(1, 3, image_size, image_size, device=ctx.device)
                img1[0, 0] = robot                          # R
                img1[0, 1] = robot_goal                     # G
                img1[0, 2] = torch.clamp(static + movable, 0, 1)  # B

                # Image 2: Target object context (robot=R, robot_goal=G, static=B, target_obj=cyan)
                img2 = torch.zeros(1, 3, image_size, image_size, device=ctx.device)
                img2[0, 0] = robot                          # R
                img2[0, 1] = torch.clamp(robot_goal + target_obj, 0, 1)  # G (goal + target_obj = cyan with B)
                img2[0, 2] = torch.clamp(static + target_obj, 0, 1)      # B

                # Image 3: Ground truth (robot=R, robot_goal=G, target_obj=B, gt_location=G)
                img3 = torch.zeros(1, 3, image_size, image_size, device=ctx.device)
                img3[0, 0] = robot                          # R
                img3[0, 1] = torch.clamp(robot_goal + gt_location, 0, 1)  # G
                img3[0, 2] = target_obj                     # B

                # Images 4-7: Predictions (robot=R, robot_goal=G, static=B, target_obj=cyan, pred=G)
                def make_pred_img(pred_loc):
                    img = torch.zeros(1, 3, image_size, image_size, device=ctx.device)
                    img[0, 0] = robot                          # R
                    img[0, 1] = torch.clamp(robot_goal + pred_loc, 0, 1)  # G
                    img[0, 2] = torch.clamp(static + target_obj, 0, 1)    # B
                    return img

                img4 = make_pred_img(pred_1)
                img5 = make_pred_img(pred_2)
                img6 = make_pred_img(pred_3)
                img7 = make_pred_img(pred_4)

                # Build row: scene | target_obj | ground_truth | pred1 | pred2 | pred3 | pred4
                row = torch.cat([img1, img2, img3, img4, img5, img6, img7], dim=0)
                grid = torchvision.utils.make_grid(row, nrow=7, normalize=True, padding=2)

                # Convert to numpy and log
                grid_np = grid.cpu().permute(1, 2, 0).numpy()
                log_dict[f'val_sample_{i+1}'] = wandb.Image(
                    grid_np,
                    caption=f"Epoch {self.current_epoch} | Scene | TargetObj | GT | Pred1 | Pred2 | Pred3 | Pred4"
                )

            self.logger.experiment.log(log_dict)

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
        tgt_size: int = 1,
        samples: int = 32,
        num_steps: int = 20
    ) -> torch.Tensor:
        """
        Sample generation interface for inference.

        Args:
            inp: Input context tensor (B, C_context, H, W)
            tgt_size: Number of target channels (default 1 for goal-only)
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
