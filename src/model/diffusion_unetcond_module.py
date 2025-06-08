from torchmetrics import MeanMetric, MinMetric
import torch
import torch.nn.functional as F
import torchvision
import os
import torch.nn as nn
import lightning.pytorch as pl

class DiffusionUNetCondModule(pl.LightningModule):
    def __init__(self, unet, noise_scheduler, optimizer, encoder_model_params):
        super().__init__()
        self.unet = unet
        self.noise_scheduler = noise_scheduler
        self.optimizer_partial = optimizer
        
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_loss_best = MinMetric()
        
        
        
        # Downward encoder to reduce 3x64x64 input to 1x16x16, then flatten.
        self.encoder_model = nn.Sequential(
            # Input: Bx4x64x64
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> Bx16x32x32
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> Bx32x16x16
            
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1), # -> Bx1x16x16
            nn.Flatten(), # -> Bx256
        )
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=["unet", "noise_scheduler"])
    
    
    def forward(self, x, timesteps, cond):
        # print(x.shape, cond.shape)
        cond = self.encoder_model(cond).unsqueeze(1)
        # print(cond.shape)
        return self.unet(x, timesteps, encoder_hidden_states=cond).sample
    
    def loss_fn(self, x, pred):
        return self.criterion(pred, x)
    
    def training_step(self, batch, batch_idx):
        inp, tgt = batch
        noise = torch.randn_like(tgt)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (inp.shape[0],), device=inp.device).long()
        noisy_tgt = self.noise_scheduler.add_noise(tgt, noise, timesteps)
        
        noisy_np = torch.cat([inp, noisy_tgt], dim=1)
        
        noise_pred = self(noisy_np, timesteps, inp)
        loss = self.loss_fn(noise_pred, noise)
        
        self.train_loss(loss)
        self.log("train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        inp, tgt = batch
        
        noise = torch.randn_like(tgt)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (inp.shape[0],), device=inp.device).long()
        noisy_tgt = self.noise_scheduler.add_noise(tgt, noise, timesteps)
        
        
        noisy_np = torch.cat([inp, noisy_tgt], dim=1)
        
        # Complete the validation step
        noise_pred = self(noisy_np, timesteps, inp)
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
            new_tgt_samples[:, :2, :, :] = tgt_samples
            new_generated_samples[:, :2, :, :] = generated_samples
            
            comparison = torch.cat([inp_samples[:, :3, :, :], inp_samples[:, 1:4, :, :], new_tgt_samples, new_generated_samples], dim=0)
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
                noise_pred = self(noisy_input, timesteps, inp)
            
            # Remove predicted noise
            sample = self.noise_scheduler.step(noise_pred, t, sample).prev_sample
            
        return sample
    
    def configure_optimizers(self):
        self.optimizer = self.optimizer_partial(params=self.parameters())
        return {
            "optimizer": self.optimizer,
        }
    
    
        
    