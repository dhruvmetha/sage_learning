"""
Training script for primitive feasibility classifier.

Usage:
    python src/train_classifier.py --config-name=train_classifier \
        data_dir=/path/to/classifier_npz \
        batch_size=64 \
        max_epochs=200
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import torch

try:
    import wandb
    from lightning.pytorch.loggers import WandbLogger
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


@hydra.main(version_base=None, config_path="../config", config_name="train_classifier")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Seed
    pl.seed_everything(42, workers=True)

    # Data
    data_module = hydra.utils.instantiate(cfg.data)

    # Network
    network = hydra.utils.instantiate(cfg.network)

    # Count parameters
    n_params = sum(p.numel() for p in network.parameters())
    print(f"Network parameters: {n_params:,}")

    # Model
    model = hydra.utils.instantiate(cfg.model, network=network)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
            filename='epoch{epoch:03d}-val_loss{val_loss:.4f}',
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval='step'),
        EarlyStopping(
            monitor='val_loss',
            patience=cfg.get('early_stopping_patience', 20),
            mode='min',
            verbose=True,
        ),
    ]

    # Logger
    logger = None
    if HAS_WANDB:
        logger = WandbLogger(
            project="namo-classifier",
            name=cfg.get('name', 'primitive_classifier'),
            save_dir=cfg.get('output_dir', 'outputs'),
        )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator='auto',
        devices='auto',
        precision=cfg.get('precision', '16-mixed'),
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=cfg.get('check_val_every_n_epoch', 1),
        log_every_n_steps=cfg.get('log_every_n_steps', 50),
        default_root_dir=cfg.get('output_dir', 'outputs'),
    )

    # Train
    trainer.fit(model, data_module)

    print(f"\nTraining complete!")
    print(f"Best model: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best val_loss: {trainer.checkpoint_callback.best_model_score:.4f}")


if __name__ == '__main__':
    main()
