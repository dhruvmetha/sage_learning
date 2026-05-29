#!/usr/bin/env python3
"""Train the SE(2) delta diffusion model.

Self-contained Lightning trainer — no Hydra config required. CLI flags only.

Usage (smoke):
    python scripts/train_se2.py \
        --h5 /scratch/dm1487/h5/car_v2_dualcrop_balanced_1to1.h5 \
        --max-epochs 3 \
        --batch-size 256 \
        --num-workers 8 \
        --wandb-run-name se2_smoke

Usage (full 500 epochs on A100):
    python scripts/train_se2.py \
        --h5 /scratch/dm1487/h5/car_v2_dualcrop_balanced_110k.h5 \
        --max-epochs 500 \
        --batch-size 512 \
        --num-workers 8 \
        --gpus 1 \
        --wandb-run-name se2_v2_dualcrop_110k_500ep
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

# Add project root so `src.*` imports resolve
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.se2_data_cropped import SE2CroppedDataModule  # noqa: E402
from src.model.se2_diffusion_module import SE2DiffusionModule  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument('--h5', type=str, required=True,
                   help='Path to the H5 dataset (e.g. car_v2_dualcrop_balanced_1to1.h5)')
    p.add_argument('--data-dir', type=str, default=None,
                   help='Optional NPZ dir if not using H5 (deprecated for full runs)')
    p.add_argument('--context-size', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--train-split', type=float, default=0.95)
    p.add_argument('--env-family-filter', type=str, default=None,
                   help=('Filter samples by xml_file substring. e.g. "feb_car" '
                         'to exclude aug9_car samples (which have too-small tight crops).'))
    # Model
    p.add_argument('--T', type=int, default=100)
    p.add_argument('--ddim-steps', type=int, default=10)
    p.add_argument('--hidden-dim', type=int, default=256)
    p.add_argument('--feat-dim', type=int, default=256)
    p.add_argument('--n-denoiser-blocks', type=int, default=4)
    # Training
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--warmup-steps', type=int, default=500)
    p.add_argument('--max-epochs', type=int, default=500)
    p.add_argument('--gpus', type=int, default=1)
    p.add_argument('--precision', type=str, default='bf16-mixed',
                   help='bf16-mixed | 32-true | 16-mixed')
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--val-check-interval', type=float, default=1.0,
                   help='Run val every N epochs (1.0) or every fraction of an epoch (<1.0)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ckpt-path', type=str, default=None, help='Resume from this checkpoint')
    # Logging
    p.add_argument('--wandb-project', type=str, default='namo-se2')
    p.add_argument('--wandb-run-name', type=str, default=None)
    p.add_argument('--wandb-mode', type=str, default='online',
                   help='online | offline | disabled')
    p.add_argument('--save-dir', type=str, default='/scratch/dm1487/checkpoints/se2')
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision('medium')

    print("=" * 60)
    print("SE(2) delta diffusion training")
    print(f"H5: {args.h5}")
    print(f"Epochs: {args.max_epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"GPUs: {args.gpus} | Precision: {args.precision}")
    print("=" * 60)

    # Data
    data_dir = args.data_dir or str(Path(args.h5).parent)
    dm = SE2CroppedDataModule(
        data_dir=data_dir,
        h5_path=args.h5,
        context_size=args.context_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=args.train_split,
        use_h5=True,
        env_family_filter=args.env_family_filter,
    )

    # Model
    model = SE2DiffusionModule(
        context_size=args.context_size,
        feat_dim=args.feat_dim,
        hidden_dim=args.hidden_dim,
        n_denoiser_blocks=args.n_denoiser_blocks,
        T=args.T,
        ddim_steps=args.ddim_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f} M")

    # Logger
    run_name = args.wandb_run_name or f"se2_{Path(args.h5).stem}"
    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = WandbLogger(
        project=args.wandb_project,
        name=run_name,
        save_dir=str(save_dir),
        mode=args.wandb_mode,
        config=vars(args),
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=str(save_dir / 'checkpoints'),
            filename='se2-{epoch:03d}-{val/loss:.4f}',
            monitor='val/loss',
            mode='min',
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 1,
        strategy='ddp_find_unused_parameters_true' if args.gpus > 1 else 'auto',
        precision=args.precision,
        gradient_clip_val=args.grad_clip,
        val_check_interval=args.val_check_interval,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
    )

    trainer.fit(model=model, datamodule=dm, ckpt_path=args.ckpt_path)
    print("Training complete.")
    print(f"Checkpoints at: {save_dir / 'checkpoints'}")


if __name__ == '__main__':
    main()
