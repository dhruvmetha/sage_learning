from torchmetrics import MeanMetric, MinMetric
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision
import os
import torch.nn as nn
import lightning.pytorch as pl

class DiffusionModule(pl.LightningModule):
    def __init__(self, model, noise_scheduler, optimizer, discrete=True, continuous=False, dice_weight=0.005):
        super().__init__()
        
        # Save hyperparameters for checkpointing FIRST
        self.save_hyperparameters(ignore=["model", "noise_scheduler"])
        
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.optimizer_partial = optimizer
        
        self.discrete = discrete
        self.continuous = continuous
        self.dice_weight = dice_weight
        
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_loss_best = MinMetric()
        
    
    def forward(self, x, timesteps):
        return self.model(x, timesteps)
    
    def loss_fn(self, x, pred):
        return self.criterion(pred, x)
    
    def training_step(self, batch, batch_idx):
        # Build input tensor: 5 channels (robot, goal, movable, static, target_object)
        # Support multiple key naming conventions for backward compatibility
        inp_parts = []
        
        # Channel 1: robot
        inp_parts.append(batch['robot'])
        
        # Channel 2: goal
        inp_parts.append(batch['goal'])
        
        # Channel 3: movable objects
        if 'movable' in batch:
            inp_parts.append(batch['movable'])
        elif 'movable_objects' in batch:
            inp_parts.append(batch['movable_objects'])
        elif 'movable_objects_image' in batch:
            inp_parts.append(batch['movable_objects_image'])
        else:
            raise KeyError('No movable objects key found in batch')
        
        # Channel 4: static objects
        if 'static' in batch:
            inp_parts.append(batch['static'])
        elif 'static_objects' in batch:
            inp_parts.append(batch['static_objects'])
        elif 'static_objects_image' in batch:
            inp_parts.append(batch['static_objects_image'])
        else:
            raise KeyError('No static objects key found in batch')

        # Channel 5: target_object (the object we're predicting a goal position for)
        if 'target_object' in batch:
            inp_parts.append(batch['target_object'])
        elif 'selected_object_mask' in batch:
            inp_parts.append(batch['selected_object_mask'])
        else:
            raise KeyError('No target_object key found in batch')

        inp = torch.cat(inp_parts, dim=1)

        # Target: target_goal (1 channel - the goal position mask for the target object)
        if 'target_goal' in batch:
            tgt = batch['target_goal']
            # Ensure channel dimension exists
            if tgt.ndim == 3:
                tgt = tgt.unsqueeze(1)
        else:
            raise KeyError('No target_goal key found in batch (required for training)')
        
        # Ensure targets are in the correct range [-1, 1] for diffusion
        if tgt.max() > 1.0 or tgt.min() < -1.0:
            tgt = torch.clamp(tgt, -1.0, 1.0)
        
        noise = torch.randn_like(tgt)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (inp.shape[0],), device=inp.device).long()
        noisy_tgt = self.noise_scheduler.add_noise(tgt, noise, timesteps)
        
        noisy_np = torch.cat([inp, noisy_tgt], dim=1)
        
        noise_pred = self(noisy_np, timesteps)
        mse_loss = self.loss_fn(noise_pred, noise)
        
        # Add numerical stability checks for x0 prediction
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(inp.device)
        sqrt_recip_alphas = torch.sqrt(1.0 / (alphas_cumprod[timesteps] + 1e-8)).view(-1,1,1,1)
        sqrt_recipm1 = torch.sqrt(1.0 / (alphas_cumprod[timesteps] + 1e-8) - 1).view(-1,1,1,1)
        x0_pred = noisy_tgt * sqrt_recip_alphas - sqrt_recipm1 * noise_pred
        
        # Clamp x0_pred to prevent extreme values before sigmoid
        x0_pred = torch.clamp(x0_pred, -10.0, 10.0)
        x0_pred = torch.sigmoid(x0_pred)

        # If target is single-channel, compute dice on that channel. Expect target in [0,1]
        dice_loss = self.focal_dice_loss(x0_pred, (tgt + 1) / 2)
        
        # Check for NaN values and handle them
        if torch.isnan(mse_loss) or torch.isnan(dice_loss):
            print(f"NaN detected: mse_loss={mse_loss}, dice_loss={dice_loss}")
            # Use only MSE loss if dice loss is NaN
            loss = mse_loss if not torch.isnan(mse_loss) else torch.tensor(0.0, device=inp.device, requires_grad=True)
        else:
            loss = mse_loss + self.dice_weight * dice_loss
        
        # Final NaN check
        if torch.isnan(loss):
            print("Loss is NaN, skipping this batch")
            return None
        
        self.train_loss(loss)
        self.log("train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        # Build input tensor: 5 channels (robot, goal, movable, static, target_object)
        inp_parts = []
        
        # Channel 1: robot
        inp_parts.append(batch['robot'])
        
        # Channel 2: goal
        inp_parts.append(batch['goal'])
        
        # Channel 3: movable objects
        if 'movable' in batch:
            inp_parts.append(batch['movable'])
        elif 'movable_objects' in batch:
            inp_parts.append(batch['movable_objects'])
        elif 'movable_objects_image' in batch:
            inp_parts.append(batch['movable_objects_image'])
        else:
            raise KeyError('No movable objects key found in batch')
        
        # Channel 4: static objects
        if 'static' in batch:
            inp_parts.append(batch['static'])
        elif 'static_objects' in batch:
            inp_parts.append(batch['static_objects'])
        elif 'static_objects_image' in batch:
            inp_parts.append(batch['static_objects_image'])
        else:
            raise KeyError('No static objects key found in batch')

        # Channel 5: target_object
        if 'target_object' in batch:
            inp_parts.append(batch['target_object'])
        elif 'selected_object_mask' in batch:
            inp_parts.append(batch['selected_object_mask'])
        else:
            raise KeyError('No target_object key found in batch')

        inp = torch.cat(inp_parts, dim=1)

        # Target: target_goal (1 channel)
        if 'target_goal' in batch:
            tgt = batch['target_goal']
            if tgt.ndim == 3:
                tgt = tgt.unsqueeze(1)
        else:
            raise KeyError('No target_goal key found in batch (required for validation)')
        
        # Ensure targets are in the correct range [-1, 1] for diffusion
        if tgt.max() > 1.0 or tgt.min() < -1.0:
            tgt = torch.clamp(tgt, -1.0, 1.0)
        
        noise = torch.randn_like(tgt)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (inp.shape[0],), device=inp.device).long()
        noisy_tgt = self.noise_scheduler.add_noise(tgt, noise, timesteps)
        
        
        noisy_np = torch.cat([inp, noisy_tgt], dim=1)
        
        # Complete the validation step
        noise_pred = self(noisy_np, timesteps)
        mse_loss = self.loss_fn(noise_pred, noise)
        
        # Add numerical stability checks for x0 prediction
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(inp.device)
        sqrt_recip_alphas = torch.sqrt(1.0 / (alphas_cumprod[timesteps] + 1e-8)).view(-1,1,1,1)
        sqrt_recipm1 = torch.sqrt(1.0 / (alphas_cumprod[timesteps] + 1e-8) - 1).view(-1,1,1,1)
        x0_pred = noisy_tgt * sqrt_recip_alphas - sqrt_recipm1 * noise_pred
        
        # Clamp x0_pred to prevent extreme values before sigmoid
        x0_pred = torch.clamp(x0_pred, -10.0, 10.0)
        x0_pred = torch.sigmoid(x0_pred)

        dice_loss = self.focal_dice_loss(x0_pred, (tgt + 1) / 2)
        
        # Check for NaN values and handle them
        if torch.isnan(mse_loss) or torch.isnan(dice_loss):
            print(f"NaN detected in validation: mse_loss={mse_loss}, dice_loss={dice_loss}")
            # Use only MSE loss if dice loss is NaN
            loss = mse_loss if not torch.isnan(mse_loss) else torch.tensor(0.0, device=inp.device, requires_grad=True)
        else:
            loss = mse_loss + self.dice_weight * dice_loss
        
        # Final NaN check
        if torch.isnan(loss):
            print("Validation loss is NaN, skipping this batch")
            return None
        
        # Log validation loss
        self.val_loss(loss)
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Save first batch for potential sample generation
        if batch_idx == 0:
            self.validation_batch = (inp, tgt, noise_pred, noise)
        return loss
    
    def on_validation_epoch_end(self):
        self.val_loss_best(self.val_loss.compute())
        self.log("val_loss_best", self.val_loss_best, prog_bar=True, sync_dist=True)
        
        # Optionally generate samples during validation (only on rank 0 to avoid duplicates)
        if hasattr(self, 'validation_batch') and self.trainer.is_global_zero:
            self._generate_validation_samples()
    
    def _generate_validation_samples(self):
        inp, tgt, noise_pred, noise = self.validation_batch
        
        # Take a few samples for generation
        num_samples = min(8, inp.size(0))
        inp_samples = inp[:num_samples]
        tgt_samples = tgt[:num_samples]
        
        image_size = inp_samples.shape[-1]
        
        # Generate samples using full denoising process
        with torch.no_grad():
            generated_samples_1 = self._full_denoising_process(inp_samples, tgt_samples)
            generated_samples_2 = self._full_denoising_process(inp_samples, tgt_samples)
            generated_samples_3 = self._full_denoising_process(inp_samples, tgt_samples)
            generated_samples_4 = self._full_denoising_process(inp_samples, tgt_samples)
        # Log samples if using tensorboard
        if hasattr(self.logger, 'experiment'):

            def to_display(x: torch.Tensor) -> torch.Tensor:
                """Convert [-1, 1] tensors to [0, 1]."""
                return torch.clamp((x + 1) / 2, 0, 1)

            # 1) Scene visualization (robot + goal + movable + static)
            scene_vis = torch.zeros(num_samples, 3, image_size, image_size, device=inp_samples.device)
            robot = to_display(inp_samples[:, 0:1])
            goal = to_display(inp_samples[:, 1:2]) if inp_samples.shape[1] > 1 else None
            movable = to_display(inp_samples[:, 2:3]) if inp_samples.shape[1] > 2 else None
            static = to_display(inp_samples[:, 3:4]) if inp_samples.shape[1] > 3 else None

            # robot -> blue
            scene_vis[:, 2, :, :] = robot[:, 0, :, :]
            # goal -> green
            if goal is not None:
                scene_vis[:, 1, :, :] = goal[:, 0, :, :]
            # movable -> yellow (red + green)
            if movable is not None:
                scene_vis[:, 0, :, :] = torch.clamp(scene_vis[:, 0, :, :] + movable[:, 0, :, :], 0, 1)
                scene_vis[:, 1, :, :] = torch.clamp(scene_vis[:, 1, :, :] + movable[:, 0, :, :], 0, 1)
            # static -> grey overlay
            if static is not None:
                grey = static[:, 0, :, :]
                scene_vis = torch.clamp(scene_vis + grey.unsqueeze(1).repeat(1, 3, 1, 1) * 0.5, 0, 1)

            # 2) Target object mask (always channel 4 of input)
            target_object_vis = torch.zeros(num_samples, 3, image_size, image_size, device=inp_samples.device)
            if inp_samples.shape[1] > 4:
                target_object = to_display(inp_samples[:, 4:5])
                target_object_vis[:, 0, :, :] = target_object[:, 0, :, :]

            # 3) Ground-truth goal mask (targets are single channel)
            goal_vis = torch.zeros(num_samples, 3, image_size, image_size, device=inp_samples.device)
            tgt_disp = to_display(tgt_samples[:, 0:1])
            goal_vis[:, 1, :, :] = tgt_disp[:, 0, :, :]

            # 4) Generated goal predictions (four independent samples)
            generated_tiles = []
            for generated in (generated_samples_1, generated_samples_2, generated_samples_3, generated_samples_4):
                gen_vis = torch.zeros(num_samples, 3, image_size, image_size, device=inp_samples.device)
                gen_disp = to_display(generated[:, 0:1])
                gen_vis[:, 1, :, :] = gen_disp[:, 0, :, :]
                generated_tiles.append(gen_vis)

            comparison = torch.cat(
                [scene_vis, target_object_vis, goal_vis] + generated_tiles,
                dim=0,
            )
            grid = torchvision.utils.make_grid(comparison, nrow=num_samples, normalize=True)
            self.logger.experiment.add_image('Validation_Samples', grid, self.current_epoch)
            
    def _full_denoising_process(self, inp, tgt):
        """Run full denoising process for sample generation"""
        batch_size = inp.size(0)
        
        # Start with pure noise
        sample = torch.randn((batch_size,) + tgt.shape[1:], device=inp.device)
        
        # Denoise step by step
        for t in reversed(range(self.noise_scheduler.config.num_train_timesteps)):
            timesteps = torch.full((batch_size,), t, device=inp.device, dtype=torch.long)
            
            # Prepare input for UNet
            noisy_input = torch.cat([inp, sample], dim=1)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self(noisy_input, timesteps)
            
            # Remove predicted noise
            sample = self.noise_scheduler.step(noise_pred, t, sample).prev_sample
            
        return sample
    
        
    def sample_from_model(self, inp, tgt_size=1, samples=32):
        """
        Sample from the diffusion model.
        
        Args:
            inp: Input tensor with shape (B, C, H, W) where C=5 (robot, goal, movable, static, target_object)
            tgt_size: Number of channels in target (default=1 for target_goal)
            samples: Number of samples to generate
            
        Returns:
            Generated samples with shape (samples, tgt_size, H, W)
        """
        # Initialize random noise for target_goal (1 channel by default)
        sample = torch.randn((1, tgt_size,) + inp.shape[2:], device=inp.device)
        sample = sample.repeat(samples, 1, 1, 1)
        inp = inp.repeat(samples, 1, 1, 1)
        
        print(f"Input shape: {inp.shape}, Sample shape: {sample.shape}")
        
        for t in tqdm(reversed(range(self.noise_scheduler.config.num_train_timesteps))):
            timesteps = torch.full((samples,), t, device=inp.device, dtype=torch.long)
            noisy_input = torch.cat([inp, sample], dim=1)
            
            with torch.no_grad():
                noise_pred = self(noisy_input, timesteps)
                
            sample = self.noise_scheduler.step(noise_pred, t, sample).prev_sample
            
        return sample
    
    def configure_optimizers(self):
        self.optimizer = self.optimizer_partial(params=self.parameters())
        return self.optimizer
    
    @staticmethod
    def dice_loss(pred, target, smooth: float = 1.0):
        # pred, target: B×1×H×W in [0,1]
        pred = pred.flatten(1)
        target = target.flatten(1)
        
        # Clamp predictions to prevent extreme values
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
        
        intersection = (pred * target).sum(dim=1)
        denom = pred.sum(dim=1) + target.sum(dim=1)
        
        # Add numerical stability
        dice = (2 * intersection + smooth) / (denom + smooth + 1e-8)
        
        # Check for NaN values
        if torch.isnan(dice).any():
            print(f"NaN in dice calculation: pred_range=[{pred.min():.3f}, {pred.max():.3f}], target_range=[{target.min():.3f}, {target.max():.3f}]")
            dice = torch.where(torch.isnan(dice), torch.ones_like(dice), dice)
        
        return 1 - dice.mean()
    
    @staticmethod
    def focal_dice_loss(pred, target, alpha=0.5, gamma=2.0, smooth=1.0):
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
    
        
    