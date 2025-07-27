from torchmetrics import MeanMetric, MinMetric
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision
import os
import torch.nn as nn
import lightning.pytorch as pl

from src.model.diffusion_module import DiffusionModule


class DiffusionMultiGranularityGoalModule(DiffusionModule):
    """Lightning module for multi-granularity goal generation model."""
    
    def __init__(self, model, noise_scheduler, optimizer, loss_type="mse"):
        # Fixed parameters for goal generation
        super().__init__(
            model=model, 
            noise_scheduler=noise_scheduler, 
            optimizer=optimizer, 
            loss_type=loss_type,
            split=True,      # Always use split mode
            discrete=False,  # Not discrete
            continuous=True  # Goal generation is continuous
        )
        
    def training_step(self, batch, batch_idx):
        """Training step for goal generation model."""
        
        # Build input: robot, goal, movable, static, object_mask (5 channels)
        inp = torch.cat([
            batch['robot'], 
            batch['goal'], 
            batch['movable_objects'], 
            batch['static_objects'],
            batch['object_mask']  # Selected object from Stage 1
        ], dim=1)
        
        # Target: goal_mask (where to place the selected object)
        tgt = batch['goal_mask']
        
        # Ensure targets are in the correct range [-1, 1] for diffusion
        if tgt.max() > 1.0 or tgt.min() < -1.0:
            tgt = torch.clamp(tgt, -1.0, 1.0)
        
        # Generate noise and timesteps
        noise = torch.randn_like(tgt)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, 
                                 (inp.shape[0],), device=inp.device).long()
        noisy_tgt = self.noise_scheduler.add_noise(tgt, noise, timesteps)
        
        # Create model input: [robot, goal, movable, static, object_mask, noisy_goal_mask] (6 channels)
        model_input = torch.cat([inp, noisy_tgt], dim=1)
        
        # Forward pass
        noise_pred = self(model_input, timesteps)
        loss = self.loss_fn(noise, noise_pred, timesteps)
        
        # Final NaN check
        if torch.isnan(loss):
            print("Loss is NaN, skipping this batch")
            return None
        
        self.train_loss(loss)
        self.log("train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        """Validation step for goal generation model."""
        
        # Build input: robot, goal, movable, static, object_mask (5 channels)
        inp = torch.cat([
            batch['robot'], 
            batch['goal'], 
            batch['movable_objects'], 
            batch['static_objects'],
            batch['object_mask']  # Selected object from Stage 1
        ], dim=1)
        
        # Target: goal_mask (where to place the selected object)
        tgt = batch['goal_mask']
        
        # Ensure targets are in the correct range [-1, 1] for diffusion
        if tgt.max() > 1.0 or tgt.min() < -1.0:
            tgt = torch.clamp(tgt, -1.0, 1.0)
        
        # Generate noise and timesteps
        noise = torch.randn_like(tgt)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, 
                                 (inp.shape[0],), device=inp.device).long()
        noisy_tgt = self.noise_scheduler.add_noise(tgt, noise, timesteps)
        
        # Create model input: [robot, goal, movable, static, object_mask, noisy_goal_mask] (6 channels)
        model_input = torch.cat([inp, noisy_tgt], dim=1)
        
        # Forward pass
        noise_pred = self(model_input, timesteps)
        loss = self.loss_fn(noise, noise_pred, timesteps)
        
        # Final NaN check
        if torch.isnan(loss):
            print("Loss is NaN, skipping this batch")
            return None
        
        # Log validation loss
        self.val_loss(loss)
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Save first batch for potential sample generation
        if batch_idx == 0:
            self.validation_batch = (inp, tgt, noise_pred, noise)
        return loss
    
    def _full_denoising_process(self, inp, tgt):
        """Run full denoising process for sample generation."""
        batch_size = inp.size(0)
        
        # Start with pure noise
        sample = torch.randn((batch_size,) + tgt.shape[1:], device=inp.device)
        
        # Denoise step by step
        for t in reversed(range(self.noise_scheduler.config.num_train_timesteps)):
            timesteps = torch.full((batch_size,), t, device=inp.device, dtype=torch.long)
            
            # Prepare input: [robot, goal, movable, static, object_mask, sample] (6 channels)
            model_input = torch.cat([inp, sample], dim=1)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self(model_input, timesteps)
            
            # Remove predicted noise
            sample = self.noise_scheduler.step(noise_pred, t, sample).prev_sample
            
        return sample
    
    def sample_from_model(self, inp, tgt_size=1, samples=32):
        """Generate goal placement samples.
        
        Args:
            inp: Input tensor with shape (1, 5, H, W) containing [robot, goal, movable, static, object_mask]
            tgt_size: Number of target channels (always 1 for goal generation)
            samples: Number of samples to generate
            
        Returns:
            Generated goal placement masks with shape (samples, 1, H, W)
        """
        
        # Generate initial noise
        sample = torch.randn((1, tgt_size,) + inp.shape[2:], device=inp.device)
        sample = sample.repeat(samples, 1, 1, 1)
        inp = inp.repeat(samples, 1, 1, 1)
        
        # Denoising loop
        for t in tqdm(reversed(range(self.noise_scheduler.config.num_train_timesteps))):
            timesteps = torch.full((samples,), t, device=inp.device, dtype=torch.long)
            
            # Create model input: [robot, goal, movable, static, object_mask, noisy_sample] (6 channels)
            model_input = torch.cat([inp, sample], dim=1)
            
            with torch.no_grad():
                noise_pred = self(model_input, timesteps)
                
            sample = self.noise_scheduler.step(noise_pred, t, sample).prev_sample
            
        return sample