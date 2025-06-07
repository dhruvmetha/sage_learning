from torchmetrics import MeanMetric, MinMetric
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torchvision
import os

class VAEModule(pl.LightningModule):
    def __init__(self, vae, optimizer, scheduler, use_kl, beta, num_samples_to_save=8):
        super().__init__()
        self.vae = vae
        self.use_kl = use_kl
        self.kl_weight = beta
        self.num_samples_to_save = num_samples_to_save
        
        # to be used in configure_optimizers
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler
        
        self.criterion = torch.nn.MSELoss()
        
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_loss_best = MinMetric()
        
        # Add KL loss tracking
        self.train_kl_loss = MeanMetric()
        self.val_kl_loss = MeanMetric()
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=["vae"])

    def forward(self, x):
        posterior = self.vae.encode(x).latent_dist
        latent = posterior.sample()
        x_hat = self.vae.decode(latent).sample
        
        return {
            "x_hat": x_hat,
            "latent": latent,
            "posterior": posterior,
        }
    
    def loss_fn(self, x, pred):
        recon_loss = self.criterion(pred['x_hat'], x)
        kl_loss = 0
        
        if self.use_kl:
            kl_loss = pred['posterior'].kl().mean()
            
        loss = recon_loss + self.kl_weight * kl_loss
        
        return loss, kl_loss
    
    def on_train_start(self):
        """Lightning hook that is called when training begins."""
        self.train_loss.reset()
        self.val_loss.reset()
        self.val_loss_best.reset()
        self.train_kl_loss.reset()
        self.val_kl_loss.reset()
    
    def training_step(self, batch, batch_idx):
        x = batch
        pred = self(x)
        loss, kl_loss = self.loss_fn(x, pred)
        
        # Log losses
        self.train_loss(loss)
        self.train_kl_loss(kl_loss)
        
        self.log("train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_kl_loss", self.train_kl_loss, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        pred = self(x)
        loss, kl_loss = self.loss_fn(x, pred)
        
        # Log losses
        self.val_loss(loss)
        self.val_kl_loss(kl_loss)
        
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_kl_loss", self.val_kl_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Save the first batch for visualization
        if batch_idx == 0:
            self.validation_batch = (x, pred)
            
        return loss
    
    def on_validation_epoch_end(self):
        self.val_loss_best(self.val_loss.compute())
        self.log("val_loss_best", self.val_loss_best, prog_bar=True)
        
        # Save sample images
        if hasattr(self, 'validation_batch'):
            x, pred = self.validation_batch
            self._save_sample_images(x, pred['x_hat'])
    
    def _save_sample_images(self, original, reconstructed):
        # Create a grid of original and reconstructed images
        num_samples = min(self.num_samples_to_save, original.size(0))
        originals = original[:num_samples]
        recons = reconstructed[:num_samples]
        
        # Stack originals and reconstructions vertically for comparison
        comparison = torch.cat([originals, recons], dim=0)
        
        # Create image grid
        grid = torchvision.utils.make_grid(comparison, nrow=num_samples, normalize=True, padding=2)
        
        # Save using PyTorch Lightning's logger
        epoch = self.current_epoch
        self.logger.experiment.add_image(f'Original vs Reconstructed', grid, epoch)
        
        # Also save as file if using TensorBoardLogger
        if hasattr(self.logger, 'log_dir'):
            save_dir = os.path.join(self.logger.log_dir, 'sample_images')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'epoch_{epoch:03d}.png')
            torchvision.utils.save_image(grid, save_path)
        
    def configure_optimizers(self):
        self.optimizer = self.optimizer_partial(params=self.parameters())
        wrapper = self.scheduler_partial(optimizer=self.optimizer)
        self.scheduler = wrapper.scheduler
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
        
    def test_step(self, batch, batch_idx):
        # The complete test step.
        x = batch
        pred = self(x)
        loss, kl_loss = self.loss_fn(x, pred)
        return loss

    def predict_step(self, batch, batch_idx):
        # The complete predict step.
        x = batch
        pred = self(x)
        return pred