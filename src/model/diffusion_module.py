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
        
        self.dice_weight = getattr(self.hparams, "dice_weight", 0.005)
        
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
        
        if "coord_grid" in batch:
            inp = torch.cat([batch['robot'], batch['goal'], batch['movable_objects'], batch['static_objects'], batch['reachable_objects'], batch['coord_grid']], dim=1)
        else:
            inp = torch.cat([batch['robot'], batch['goal'], batch['movable_objects'], batch['static_objects'], batch['reachable_objects']], dim=1)
        
        tgt = torch.cat([batch['object_mask'], batch['goal_mask']], dim=1)
        
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
        x0_pred = (noisy_tgt - sqrt_recipm1 * noise_pred) * sqrt_recip_alphas
        
        # Clamp x0_pred to prevent extreme values before sigmoid
        x0_pred = torch.clamp(x0_pred, -10.0, 10.0)
        x0_pred = torch.sigmoid(x0_pred)
        
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
        self.log("train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        # inp, tgt = batch
        
        if "coord_grid" in batch:
            inp = torch.cat([batch['robot'], batch['goal'], batch['movable_objects'], batch['static_objects'], batch['reachable_objects'], batch['coord_grid']], dim=1)
        else:
            inp = torch.cat([batch['robot'], batch['goal'], batch['movable_objects'], batch['static_objects'], batch['reachable_objects']], dim=1)
        
        tgt = torch.cat([batch['object_mask'], batch['goal_mask']], dim=1)
        
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
        x0_pred = (noisy_tgt - sqrt_recipm1 * noise_pred) * sqrt_recip_alphas
        
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
            generated_samples_1 = self._full_denoising_process(inp_samples, tgt_samples)
            generated_samples_2 = self._full_denoising_process(inp_samples, tgt_samples)
            generated_samples_3 = self._full_denoising_process(inp_samples, tgt_samples)
            generated_samples_4 = self._full_denoising_process(inp_samples, tgt_samples)
        # Log samples if using tensorboard
        if hasattr(self.logger, 'experiment'):
            # Create comparison grid: input, target, generated
            new_tgt_samples, new_generated_samples_1, new_generated_samples_2, new_generated_samples_3, new_generated_samples_4 = torch.zeros(num_samples, 3, image_size, image_size, device=inp_samples.device), torch.zeros(num_samples, 3, image_size, image_size, device=inp_samples.device), torch.zeros(num_samples, 3, image_size, image_size, device=inp_samples.device), torch.zeros(num_samples, 3, image_size, image_size, device=inp_samples.device), torch.zeros(num_samples, 3, image_size, image_size, device=inp_samples.device)
            
            # for any number of channels
            num_channels = min(3, tgt_samples.shape[1])
            new_tgt_samples[:, :num_channels, :, :] = tgt_samples[:, :num_channels, :, :]
            new_generated_samples_1[:, :num_channels, :, :] = generated_samples_1[:, :num_channels, :, :]
            new_generated_samples_2[:, :num_channels, :, :] = generated_samples_2[:, :num_channels, :, :]
            new_generated_samples_3[:, :num_channels, :, :] = generated_samples_3[:, :num_channels, :, :]
            new_generated_samples_4[:, :num_channels, :, :] = generated_samples_4[:, :num_channels, :, :]
            
            vis_1 = torch.zeros(num_samples, 3, image_size, image_size, device=inp_samples.device)
            vis_1[:, 0, :, :] = inp_samples[:, 0, :, :] + inp_samples[:, 1, :, :]
            vis_1[:, 1, :, :] = inp_samples[:, 2, :, :]
            vis_1[:, 2, :, :] = inp_samples[:, 3, :, :]
            
            vis_2 = torch.zeros(num_samples, 3, image_size, image_size, device=inp_samples.device)
            vis_2[:, 0, :, :] = inp_samples[:, 4, :, :]
            
            comparison = torch.cat([vis_1, vis_2, new_tgt_samples, new_generated_samples_1, new_generated_samples_2, new_generated_samples_3, new_generated_samples_4], dim=0)
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
            "gradient_clip_val": 1.0,  # Add gradient clipping
        }
    
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
    
        
    