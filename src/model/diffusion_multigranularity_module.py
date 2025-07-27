from torchmetrics import MeanMetric, MinMetric
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision
import os
import torch.nn as nn
import lightning.pytorch as pl

from src.model.diffusion_module import DiffusionModule


class DiffusionMultiGranularityModule(DiffusionModule):
    """Lightning module for multi-granularity diffusion model."""
    
    def __init__(self, model, noise_scheduler, optimizer, loss_type, split=True, discrete=True, continuous=False):
        super().__init__(model, noise_scheduler, optimizer, loss_type, split, discrete, continuous)
        
    def training_step(self, batch, batch_idx):
        """Modified training step for multi-granularity model following split pattern."""
        
        # Build input array following the same pattern as DiffusionModule
        inp_arr = [batch['robot'], batch['goal'], batch['movable_objects'], batch['static_objects']]
        tgt_arr = []
        
        if self.split:  # for split model
            if self.discrete:
                # Discrete mode: predict object_mask (which object to manipulate)
                tgt_arr.append(batch['object_mask'])
            elif self.continuous:
                # Continuous mode: use object_mask as input, predict goal_mask
                inp_arr.append(batch['object_mask'])
                tgt_arr.append(batch['goal_mask'])
        
        # Optional coordinate grid (not used in multi-granularity but keeping for compatibility)
        if "coord_grid" in batch:
            inp_arr.append(batch['coord_grid'])
        
        inp = torch.cat(inp_arr, dim=1)
        tgt = torch.cat(tgt_arr, dim=1)
        
        # Ensure targets are in the correct range [-1, 1] for diffusion
        if tgt.max() > 1.0 or tgt.min() < -1.0:
            tgt = torch.clamp(tgt, -1.0, 1.0)
        
        # Generate noise and timesteps
        noise = torch.randn_like(tgt)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, 
                                 (inp.shape[0],), device=inp.device).long()
        noisy_tgt = self.noise_scheduler.add_noise(tgt, noise, timesteps)
        
        # Create model input: concatenate input + noisy target
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
        """Modified validation step for multi-granularity model following split pattern."""
        
        # Build input array following the same pattern as DiffusionModule
        inp_arr = [batch['robot'], batch['goal'], batch['movable_objects'], batch['static_objects']]
        tgt_arr = []
        
        if self.split:  # for split model
            if self.discrete:
                # Discrete mode: predict object_mask (which object to manipulate)  
                tgt_arr.append(batch['object_mask'])
            elif self.continuous:
                # Continuous mode: use object_mask as input, predict goal_mask
                inp_arr.append(batch['object_mask'])
                tgt_arr.append(batch['goal_mask'])
        
        # Optional coordinate grid (not used in multi-granularity but keeping for compatibility)
        if "coord_grid" in batch:
            inp_arr.append(batch['coord_grid'])
        
        inp = torch.cat(inp_arr, dim=1)
        tgt = torch.cat(tgt_arr, dim=1)
        
        # Ensure targets are in the correct range [-1, 1] for diffusion
        if tgt.max() > 1.0 or tgt.min() < -1.0:
            tgt = torch.clamp(tgt, -1.0, 1.0)
        
        # Generate noise and timesteps
        noise = torch.randn_like(tgt)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, 
                                 (inp.shape[0],), device=inp.device).long()
        noisy_tgt = self.noise_scheduler.add_noise(tgt, noise, timesteps)
        
        # Create model input: concatenate input + noisy target
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
            
            # Prepare input for multi-granularity model (inp already contains 4 channels)
            model_input = torch.cat([inp, sample], dim=1)  # (B, 5, 64, 64)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self(model_input, timesteps)
            
            # Remove predicted noise
            sample = self.noise_scheduler.step(noise_pred, t, sample).prev_sample
            
        return sample
    
    def sample_from_model(self, inp, tgt_size=1, samples=32):
        """Generate samples from the multi-granularity model.
        
        Args:
            inp: Input tensor with shape (1, 4, H, W) containing [robot, goal, movable, static]
            tgt_size: Number of target channels (default 1)
            samples: Number of samples to generate
            
        Returns:
            Generated samples with shape (samples, tgt_size, H, W)
        """
        
        # Generate initial noise
        sample = torch.randn((1, tgt_size,) + inp.shape[2:], device=inp.device)
        sample = sample.repeat(samples, 1, 1, 1)
        inp = inp.repeat(samples, 1, 1, 1)
        
        # Denoising loop
        for t in tqdm(reversed(range(self.noise_scheduler.config.num_train_timesteps))):
            timesteps = torch.full((samples,), t, device=inp.device, dtype=torch.long)
            
            # Create model input: [robot, goal, movable, static, noisy_sample]
            model_input = torch.cat([inp, sample], dim=1)  # (samples, 5, H, W)
            
            with torch.no_grad():
                noise_pred = self(model_input, timesteps)
                
            sample = self.noise_scheduler.step(noise_pred, t, sample).prev_sample
            
        return sample