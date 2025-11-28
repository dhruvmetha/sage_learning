"""
Unified Training Script for Generative Models

Supports both diffusion and flow matching through configuration.

Usage:
    # Train with flow matching
    python src/train_generative.py --config-name=train_flow_matching

    # Train with diffusion
    python src/train_generative.py --config-name=train_diffusion

    # Override parameters
    python src/train_generative.py --config-name=train_flow_matching \
        data.data_dir=/path/to/data \
        batch_size=32 \
        max_epochs=200

    # Resume from checkpoint
    python src/train_generative.py --config-name=train_flow_matching \
        ckpt_path=/path/to/checkpoint.ckpt
"""

import os
import sys
from pathlib import Path

import hydra
import lightning.pytorch as pl
import torch

# Configure default dtype
torch.set_default_dtype(torch.float32)

# Enable TF32 for faster computation on Ampere+ GPUs
torch.set_float32_matmul_precision('medium')

# Use file_system sharing strategy to avoid NFS issues
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@hydra.main(config_path="../config", config_name="train_flow_matching", version_base="1.1")
def main(cfg):
    """Main training function."""

    # Set random seed
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed)

    # Print configuration summary
    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Model type: {cfg.model._target_}")
    if hasattr(cfg.model, 'path'):
        print(f"Path type: {cfg.model.path._target_}")
    if hasattr(cfg.model, 'sampler'):
        print(f"Sampler type: {cfg.model.sampler._target_}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Max epochs: {cfg.max_epochs}")
    print(f"Learning rate: {cfg.base_lr}")
    print(f"Image size: {cfg.image_size}")
    print("=" * 60)

    # Instantiate data module
    print("\nLoading data module...")
    data_module: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Instantiate model
    print("Creating model...")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optionally load from a previous run (for fine-tuning)
    if cfg.get("run_path", None) is not None:
        run_path = Path(cfg.run_path)
        print(f"\nLoading pretrained weights from: {run_path}")

        checkpoint_dir = run_path / "checkpoints"
        checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))

        checkpoint_path = None
        for checkpoint_file in checkpoint_files:
            if "epoch" in checkpoint_file.name:
                checkpoint_path = checkpoint_file
                break

        if checkpoint_path is None:
            raise ValueError(f"No checkpoint found in {checkpoint_dir}")

        ckpt = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded checkpoint: {checkpoint_path}")

    # Instantiate trainer
    print("\nCreating trainer...")
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)

    # Check for resume checkpoint
    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path is not None:
        print(f"Resuming training from: {ckpt_path}")

    # Start training
    print("\nStarting training...")
    print("=" * 60)
    trainer.fit(model=model, datamodule=data_module, ckpt_path=ckpt_path)

    print("\nTraining complete!")
    
    # Use checkpoint callback to get best loss if available to avoid DDP sync issues
    if trainer.checkpoint_callback and getattr(trainer.checkpoint_callback, 'best_model_score', None) is not None:
        print(f"Best validation loss: {trainer.checkpoint_callback.best_model_score:.6f}")
    else:
        print(f"Best validation loss: {model.val_loss_best.compute():.6f}")


if __name__ == "__main__":
    main()
