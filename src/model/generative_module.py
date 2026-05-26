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
        target_channels: Number of output target channels (default 1: target_goal only, 2 for multi-horizon)
        use_local: Use local (object-centered) masks instead of global
        use_multihorizon: Enable multi-horizon prediction (2 output channels with masked loss)
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
        use_local: bool = False,
        use_multihorizon: bool = False,
        warmup_steps: int = 0,
        decay_steps: int = 0,
        end_lr: float = 0.0,
    ):
        super().__init__()

        self.network = network
        self.path = path
        self.sampler = sampler
        self.optimizer_partial = optimizer

        self.aux_loss_weight = aux_loss_weight
        self.context_channels = context_channels
        self.target_channels = target_channels
        self.use_local = use_local
        self.use_multihorizon = use_multihorizon

        # LR schedule parameters
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.end_lr = end_lr

        # Loss function
        self.criterion = nn.MSELoss(reduction='none')  # Use 'none' for masked loss support
        self.criterion_mean = nn.MSELoss()  # For backward compatibility

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

        For global masks: Concatenates robot, goal, movable, static, target_object
        For local masks: Concatenates static, movable, target_object, robot_region, goal_sample_region
        """
        if self.use_local:
            # Local masks: static, movable, target_object, robot_region, goal_sample_region (5 channels)
            parts = [
                batch['static'],
                batch['movable'],
                batch['target_object'],
                batch['robot_region'],
                batch['goal_sample_region'],
            ]
        else:
            # Global masks: robot, goal, movable, static, target_object (5 channels)
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

        For multi-horizon mode, returns [B, 2, H, W] from 'target_goals'.
        For single-horizon mode, returns [B, 1, H, W] from 'target_goal'.
        """
        if self.use_multihorizon:
            # Multi-horizon: target_goals is [B, 2, H, W]
            return torch.clamp(batch['target_goals'], -1, 1)
        else:
            # Single-horizon: target_goal is [B, 1, H, W]
            return torch.clamp(batch['target_goal'], -1, 1)

    def _compute_multihorizon_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        solution_depth: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked loss for multi-horizon prediction.

        Channel 1 (goal_mask_a1) is always supervised.
        Channel 2 (goal_mask_a2) is only supervised when solution_depth >= 2.

        Args:
            prediction: Model output [B, 2, H, W]
            target: Ground truth [B, 2, H, W]
            solution_depth: [B] tensor with solution depth (1 or 2)

        Returns:
            Combined loss for both channels
        """
        B, _, H, W = prediction.shape

        # Channel 1 loss: always computed
        loss_ch1 = self.criterion(prediction[:, 0:1], target[:, 0:1]).mean()

        # Channel 2 loss: only compute where solution_depth >= 2
        # Create mask [B, 1, 1, 1] for broadcasting
        mask = (solution_depth >= 2).float().view(B, 1, 1, 1)

        # Compute per-element loss for channel 2
        loss_ch2_elements = self.criterion(prediction[:, 1:2], target[:, 1:2])  # [B, 1, H, W]

        # Apply mask and compute mean only over valid samples
        num_valid = mask.sum() + 1e-8  # Avoid division by zero
        loss_ch2 = (loss_ch2_elements * mask).sum() / (num_valid * H * W)

        return loss_ch1 + loss_ch2

    def _compute_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_1: torch.Tensor,
        solution_depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute loss with optional auxiliary losses and multi-horizon masking.

        Args:
            prediction: Model output (noise or velocity)
            target: Ground truth (noise or velocity)
            x_t: Current interpolated/noised point
            t: Time values
            x_1: Original data (for auxiliary losses)
            solution_depth: [B] tensor with solution depth (1 or 2) for multi-horizon masking

        Returns:
            Total loss
        """
        if self.use_multihorizon and solution_depth is not None:
            primary_loss = self._compute_multihorizon_loss(prediction, target, solution_depth)
        else:
            # Single-horizon: standard MSE loss
            primary_loss = self.criterion_mean(prediction, target)

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

        # Get solution_depth for multi-horizon masking
        solution_depth = batch.get('solution_depth', None)
        if solution_depth is not None:
            solution_depth = torch.tensor(solution_depth, device=x_1.device) if not isinstance(solution_depth, torch.Tensor) else solution_depth

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
            x_1=x_1,
            solution_depth=solution_depth,
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

        # Get solution_depth for multi-horizon masking
        solution_depth = batch.get('solution_depth', None)
        if solution_depth is not None:
            solution_depth = torch.tensor(solution_depth, device=x_1.device) if not isinstance(solution_depth, torch.Tensor) else solution_depth

        path_sample = self.path.sample(x_0=x_0, x_1=x_1)
        model_input = torch.cat([context, path_sample.x_t], dim=1)

        prediction = self(model_input, path_sample.t)

        loss = self._compute_loss(
            prediction=prediction,
            target=path_sample.target,
            x_t=path_sample.x_t,
            t=path_sample.t,
            x_1=x_1,
            solution_depth=solution_depth,
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
            self._validation_solution_depth = solution_depth

        return loss

    def on_validation_epoch_end(self):
        """Log best validation loss and generate samples."""
        self.val_loss_best(self.val_loss.compute())
        self.log("val_loss_best", self.val_loss_best, prog_bar=True)

        # Generate validation samples
        if hasattr(self, '_validation_context'):
            self._generate_validation_samples()

        # Clear validation tensors to free GPU memory
        self._validation_context = None
        self._validation_target = None
        self._validation_target_object = None
        self._validation_solution_depth = None
        torch.cuda.empty_cache()

    def _generate_validation_samples(self):
        """Generate and log samples during validation.

        Logs 8 images, each showing a different validation example.
        Each image has 7 panels with different visualizations depending on mode.

        Global mode (use_local=False):
        - Context channels: 0=robot, 1=robot_goal, 2=movable, 3=static, 4=target_object
        - Panels: Scene | TargetObj | GT | Pred1-4

        Local mode (use_local=True):
        - Context channels: 0=static, 1=movable, 2=target_object, 3=goal_region
        - Panels: Scene | TargetObj | GT | Pred1-4

        Multi-horizon mode (use_multihorizon=True):
        - Shows both goal_mask_a1 and goal_mask_a2 predictions
        - GT shows both channels
        - Layout: Scene | Reach | GT_a1 | GT_a2 | Pred_a1 | Pred_a2 | Pred_a1 | Pred_a2
        """
        context = self._validation_context
        target = self._validation_target
        solution_depth = self._validation_solution_depth

        num_examples = min(8, context.size(0))
        image_size = context.shape[-1]

        # Log images if logger is available
        if hasattr(self.logger, 'experiment'):
            import wandb

            def _to_display(x: torch.Tensor) -> torch.Tensor:
                return torch.clamp((x + 1) / 2, 0, 1)

            log_dict = {}
            for i in range(num_examples):
                ctx = context[i:i+1]  # (1, C, H, W)

                # Handle multi-horizon vs single-horizon target
                if self.use_multihorizon:
                    # target is [B, 2, H, W], get both channels
                    gt_a1 = _to_display(target[i:i+1, 0:1])[0, 0]  # (H, W)
                    gt_a2 = _to_display(target[i:i+1, 1:2])[0, 0]  # (H, W)
                    # Get solution depth for this sample (for caption)
                    sample_depth = int(solution_depth[i].item()) if solution_depth is not None else 0
                else:
                    # target is [B, 1, H, W]
                    gt_a1 = _to_display(target[i:i+1])[0, 0]  # (H, W)
                    gt_a2 = None
                    sample_depth = 1

                # Generate predictions
                with torch.no_grad():
                    if self.use_multihorizon:
                        # Generate 2 samples, each with both channels
                        sample1 = self.sample(ctx, num_steps=20)[0]  # (2, H, W)
                        sample2 = self.sample(ctx, num_steps=20)[0]  # (2, H, W)
                        pred_a1_1 = _to_display(sample1[0:1])[0]  # (H, W)
                        pred_a2_1 = _to_display(sample1[1:2])[0]  # (H, W)
                        pred_a1_2 = _to_display(sample2[0:1])[0]  # (H, W)
                        pred_a2_2 = _to_display(sample2[1:2])[0]  # (H, W)
                    else:
                        pred_a1_1 = _to_display(self.sample(ctx, num_steps=20)[0, 0:1])[0]
                        pred_a1_2 = _to_display(self.sample(ctx, num_steps=20)[0, 0:1])[0]
                        pred_a1_3 = _to_display(self.sample(ctx, num_steps=20)[0, 0:1])[0]
                        pred_a1_4 = _to_display(self.sample(ctx, num_steps=20)[0, 0:1])[0]

                if self.use_local:
                    # Local mode channels: 0=static, 1=movable, 2=target_object, 3=robot_region, 4=goal_sample_region
                    static = _to_display(ctx[0, 0:1])[0]
                    movable = _to_display(ctx[0, 1:2])[0]
                    target_obj = _to_display(ctx[0, 2:3])[0]
                    robot_region = _to_display(ctx[0, 3:4])[0]
                    goal_sample_region = _to_display(ctx[0, 4:5])[0]

                    # Consistent colors: R=static/context, G=variable/pred, B=target_obj (always blue)

                    # Image 1: Scene (R=static, G=movable, B=target_obj)
                    img1 = torch.zeros(1, 3, image_size, image_size, device=ctx.device)
                    img1[0, 0] = static
                    img1[0, 1] = movable
                    img1[0, 2] = target_obj

                    # Image 2: Reachability (R=robot_region, G=goal_sample_region, B=target_obj)
                    img2 = torch.zeros(1, 3, image_size, image_size, device=ctx.device)
                    img2[0, 0] = robot_region
                    img2[0, 1] = goal_sample_region
                    img2[0, 2] = target_obj

                    # Helper to make prediction/GT images
                    def make_pred_img(pred_loc):
                        img = torch.zeros(1, 3, image_size, image_size, device=ctx.device)
                        img[0, 0] = static
                        img[0, 1] = pred_loc
                        img[0, 2] = target_obj
                        return img

                    if self.use_multihorizon:
                        # Multi-horizon: Show GT_a1, GT_a2, then 2 samples with both channels
                        # Layout: Scene | Reach | GT_a1 | GT_a2 | Pred_a1 | Pred_a2 | Pred_a1 | Pred_a2
                        img3 = make_pred_img(gt_a1)  # GT_a1
                        img4 = make_pred_img(gt_a2)  # GT_a2
                        img5 = make_pred_img(pred_a1_1)  # Sample1 a1
                        img6 = make_pred_img(pred_a2_1)  # Sample1 a2
                        img7 = make_pred_img(pred_a1_2)  # Sample2 a1
                        img8 = make_pred_img(pred_a2_2)  # Sample2 a2

                        row = torch.cat([img1, img2, img3, img4, img5, img6, img7, img8], dim=0)
                        nrow = 8
                        caption = f"Epoch {self.current_epoch} depth={sample_depth} | Scene | Reach | GT_a1 | GT_a2 | S1_a1 | S1_a2 | S2_a1 | S2_a2"
                    else:
                        # Single-horizon: original layout
                        img3 = make_pred_img(gt_a1)
                        img4 = make_pred_img(pred_a1_1)
                        img5 = make_pred_img(pred_a1_2)
                        img6 = make_pred_img(pred_a1_3)
                        img7 = make_pred_img(pred_a1_4)

                        row = torch.cat([img1, img2, img3, img4, img5, img6, img7], dim=0)
                        nrow = 7
                        caption = f"Epoch {self.current_epoch} | Scene | Reach | GT | Pred1-4"

                else:
                    # Global mode channels: 0=robot, 1=robot_goal, 2=movable, 3=static, 4=target_object
                    robot = _to_display(ctx[0, 0:1])[0]
                    robot_goal = _to_display(ctx[0, 1:2])[0]
                    movable = _to_display(ctx[0, 2:3])[0]
                    static = _to_display(ctx[0, 3:4])[0]
                    target_obj = _to_display(ctx[0, 4:5])[0]

                    # Image 1: Scene (robot=R, robot_goal=G, static+movable=B)
                    img1 = torch.zeros(1, 3, image_size, image_size, device=ctx.device)
                    img1[0, 0] = robot
                    img1[0, 1] = robot_goal
                    img1[0, 2] = torch.clamp(static + movable, 0, 1)

                    # Image 2: Target object context (robot=R, robot_goal+target_obj=G, static+target_obj=B)
                    img2 = torch.zeros(1, 3, image_size, image_size, device=ctx.device)
                    img2[0, 0] = robot
                    img2[0, 1] = torch.clamp(robot_goal + target_obj, 0, 1)
                    img2[0, 2] = torch.clamp(static + target_obj, 0, 1)

                    # Image 3: Ground truth (robot=R, robot_goal+gt_location=G, target_obj=B)
                    img3 = torch.zeros(1, 3, image_size, image_size, device=ctx.device)
                    img3[0, 0] = robot
                    img3[0, 1] = torch.clamp(robot_goal + gt_a1, 0, 1)
                    img3[0, 2] = target_obj

                    # Images 4-7: Predictions (robot=R, robot_goal+pred=G, static+target_obj=B)
                    def make_pred_img(pred_loc):
                        img = torch.zeros(1, 3, image_size, image_size, device=ctx.device)
                        img[0, 0] = robot
                        img[0, 1] = torch.clamp(robot_goal + pred_loc, 0, 1)
                        img[0, 2] = torch.clamp(static + target_obj, 0, 1)
                        return img

                    img4 = make_pred_img(pred_a1_1)
                    img5 = make_pred_img(pred_a1_2)
                    img6 = make_pred_img(pred_a1_3)
                    img7 = make_pred_img(pred_a1_4)

                    row = torch.cat([img1, img2, img3, img4, img5, img6, img7], dim=0)
                    nrow = 7
                    caption = f"Epoch {self.current_epoch} | Scene | TargetObj | GT | Pred1 | Pred2 | Pred3 | Pred4"

                grid = torchvision.utils.make_grid(row, nrow=nrow, normalize=True, padding=2)

                # Convert to numpy and log
                grid_np = grid.cpu().permute(1, 2, 0).numpy()
                log_dict[f'val_sample_{i+1}'] = wandb.Image(grid_np, caption=caption)

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
        tgt_size: int = None,
        samples: int = 32,
        num_steps: int = 20,
        seed: int = None
    ) -> torch.Tensor:
        """
        Sample generation interface for inference.

        Args:
            inp: Input context tensor (B, C_context, H, W)
            tgt_size: Number of target channels. If None, uses self.target_channels
                      (supports multi-horizon models with 2 output channels)
            samples: Number of samples to generate
            num_steps: Number of sampling steps
            seed: Random seed for reproducible noise (None for random)

        Returns:
            Generated samples, shape (samples, tgt_size, H, W)
        """
        # Use model's configured target_channels if not specified
        if tgt_size is None:
            tgt_size = self.target_channels

        # Repeat input for multiple samples
        inp_repeated = inp.repeat(samples, 1, 1, 1)

        # Initialize from different noise for each sample (enables diverse outputs)
        if seed is not None:
            generator = torch.Generator(device=inp.device).manual_seed(seed)
            x_init = torch.randn(samples, tgt_size, inp.shape[2], inp.shape[3], device=inp.device, generator=generator)
        else:
            x_init = torch.randn(samples, tgt_size, inp.shape[2], inp.shape[3], device=inp.device)

        def model_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            model_input = torch.cat([inp_repeated, x], dim=1)
            return self(model_input, t)

        return self.sampler.sample(
            model=model_fn,
            x_init=x_init,
            num_steps=num_steps,
            show_progress=False
        )

    def configure_optimizers(self):
        """Configure optimizer and LR scheduler."""
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
