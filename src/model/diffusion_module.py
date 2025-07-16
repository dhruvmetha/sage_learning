from torchmetrics import MeanMetric, MinMetric
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision
import os
import torch.nn as nn
import lightning.pytorch as pl

class DiffusionModule(pl.LightningModule):
    def __init__(self, model, noise_scheduler, optimizer, discrete=True, continuous=False):
        super().__init__()
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.optimizer_partial = optimizer
        
        self.discrete = discrete
        self.continuous = continuous
        
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_loss_best = MinMetric()
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=["model", "noise_scheduler"])
    
    def forward(self, x, timesteps):
        return self.model(x, timesteps)
    
    def loss_fn(self, x, pred):
        return self.criterion(pred, x)
    
    def training_step(self, batch, batch_idx):
        
        inp = torch.cat([batch['robot'], batch['goal'], batch['movable_objects'], batch['static_objects'], batch['object_mask']], dim=1)
        tgt = batch['goal_mask']
        
        noise = torch.randn_like(tgt)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (inp.shape[0],), device=inp.device).long()
        noisy_tgt = self.noise_scheduler.add_noise(tgt, noise, timesteps)
        
        noisy_np = torch.cat([inp, noisy_tgt], dim=1)
        
        noise_pred = self(noisy_np, timesteps)
        loss = self.loss_fn(noise_pred, noise)
        
        self.train_loss(loss)
        self.log("train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        # inp, tgt = batch
        
        inp = torch.cat([batch['robot'], batch['goal'], batch['movable_objects'], batch['static_objects'], batch['object_mask']], dim=1)
        tgt = batch['goal_mask']
        
        noise = torch.randn_like(tgt)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (inp.shape[0],), device=inp.device).long()
        noisy_tgt = self.noise_scheduler.add_noise(tgt, noise, timesteps)
        
        
        noisy_np = torch.cat([inp, noisy_tgt], dim=1)
        
        # Complete the validation step
        noise_pred = self(noisy_np, timesteps)
        loss = self.loss_fn(noise_pred, noise)
        
        # Log validation loss
        self.val_loss(loss)
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Save first batch for potential sample generation
        if batch_idx == 0:
            self.validation_batch = (inp, tgt, noise_pred, noise)
        return loss
    
    def on_validation_epoch_end(self):
        self.val_loss_best(self.val_loss.compute())
        self.log("val_loss_best", self.val_loss_best, prog_bar=True)
        
        # Optionally generate samples during validation
        if hasattr(self, 'validation_batch'):
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
            generated_samples = self._full_denoising_process(inp_samples, tgt_samples)
            
        # Log samples if using tensorboard
        if hasattr(self.logger, 'experiment'):
            # Create comparison grid: input, target, generated
            new_tgt_samples, new_generated_samples = torch.zeros(num_samples, 3, image_size, image_size, device=inp_samples.device), torch.zeros(num_samples, 3, image_size, image_size, device=inp_samples.device)
            
            # for any number of channels
            num_channels = min(3, tgt_samples.shape[1])
            new_tgt_samples[:, :num_channels, :, :] = tgt_samples[:, :num_channels, :, :]
            new_generated_samples[:, :num_channels, :, :] = generated_samples[:, :num_channels, :, :]
            
            vis_1 = torch.zeros(num_samples, 3, image_size, image_size, device=inp_samples.device)
            vis_1[:, 0, :, :] = inp_samples[:, 0, :, :] + inp_samples[:, 1, :, :]
            vis_1[:, 1, :, :] = inp_samples[:, 2, :, :]
            vis_1[:, 2, :, :] = inp_samples[:, 3, :, :]
            
            vis_2 = torch.zeros(num_samples, 3, image_size, image_size, device=inp_samples.device)
            vis_2[:, 0, :, :] = inp_samples[:, 4, :, :]
            
            comparison = torch.cat([vis_1, vis_2, new_tgt_samples, new_generated_samples], dim=0)
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
        
        # sample = torch.randn((samples, 2, ) + inp.shape[2:], device=inp.device) # different noise for each sample
        sample = torch.randn((1, tgt_size,) + inp.shape[2:], device=inp.device)
        sample = sample.repeat(samples, 1, 1, 1)
        inp = inp.repeat(samples, 1, 1, 1)
        
        print(inp.shape, sample.shape)
        
        for t in tqdm(reversed(range(self.noise_scheduler.config.num_train_timesteps))):
            timesteps = torch.full((samples,), t, device=inp.device, dtype=torch.long)
            noisy_input = torch.cat([inp, sample], dim=1)
            
            with torch.no_grad():
                noise_pred = self(noisy_input, timesteps)
                
            sample = self.noise_scheduler.step(noise_pred, t, sample).prev_sample
            
        return sample
    
    def configure_optimizers(self):
        self.optimizer = self.optimizer_partial(params=self.parameters())
        return {
            "optimizer": self.optimizer,
        }
    
    
        
    