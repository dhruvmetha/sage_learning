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


def _parse_bool_flag(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean flag, got '{value}'")


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
    p.add_argument('--pin-memory', type=_parse_bool_flag, default=False)
    p.add_argument('--persistent-workers', type=_parse_bool_flag, default=False)
    p.add_argument('--train-split', type=float, default=0.95)
    p.add_argument('--env-family-filter', type=str, default=None,
                   help=('Filter samples by xml_file substring. e.g. "feb_car" '
                         'to exclude aug9_car samples (which have too-small tight crops).'))
    # Model
    p.add_argument('--T', type=int, default=100)
    p.add_argument('--ddim-steps', type=int, default=10)
    p.add_argument('--model-type', type=str, default='diffusion',
                   choices=['diffusion', 'multihyp', 'multihyp_v2'])
    p.add_argument('--hidden-dim', type=int, default=256)
    p.add_argument('--feat-dim', type=int, default=256)
    p.add_argument('--n-denoiser-blocks', type=int, default=4)
    p.add_argument('--num-layers', type=int, default=6)
    p.add_argument('--num-heads', type=int, default=8)
    p.add_argument('--num-hypotheses', type=int, default=4)
    p.add_argument('--assignment-temp', type=float, default=0.35)
    p.add_argument('--cls-loss-weight', type=float, default=0.2)
    p.add_argument('--angle-loss-weight', type=float, default=1.0)
    p.add_argument('--reg-beta', type=float, default=0.5)
    p.add_argument('--hyp-dropout-prob', type=float, default=0.25)
    p.add_argument('--diversity-weight', type=float, default=0.02)
    p.add_argument('--diversity-margin', type=float, default=0.75)
    p.add_argument('--use-hyp-self-attn', type=_parse_bool_flag, default=False)
    p.add_argument('--best-idx-noise-scale', type=float, default=1e-4)
    # Training
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--warmup-steps', type=int, default=500)
    p.add_argument('--max-epochs', type=int, default=500)
    p.add_argument('--gpus', type=int, default=1)
    p.add_argument('--strategy', type=str, default='ddp',
                   help='Lightning strategy, e.g. ddp or auto')
    p.add_argument('--precision', type=str, default='bf16-mixed',
                   help='bf16-mixed | 32-true | 16-mixed')
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--val-check-interval', type=float, default=1.0,
                   help='Run val every N epochs (1.0) or every fraction of an epoch (<1.0)')
    p.add_argument('--check-val-every-n-epoch', type=int, default=5,
                   help='Run validation every N epochs')
    p.add_argument('--num-sanity-val-steps', type=int, default=0,
                   help='Number of sanity validation steps before training')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ckpt-path', type=str, default=None, help='Resume from this checkpoint')
    p.add_argument('--init-ckpt-path', type=str, default=None,
                   help='Load model weights from this checkpoint, but start a fresh optimizer/scheduler state')
    # Logging
    p.add_argument('--wandb-project', type=str, default='namo-se2')
    p.add_argument('--wandb-run-name', type=str, default=None)
    p.add_argument('--wandb-mode', type=str, default='online',
                   help='online | offline | disabled')
    p.add_argument('--save-dir', type=str, default='/scratch/dm1487/checkpoints/se2')
    return p.parse_args()


def main():
    args = parse_args()
    if args.ckpt_path and args.init_ckpt_path:
        raise ValueError("Use only one of --ckpt-path or --init-ckpt-path")

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision('medium')
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
    except (AttributeError, RuntimeError):
        pass

    print("=" * 60)
    print("SE(2) training")
    print(f"H5: {args.h5}")
    print(f"Model: {args.model_type}")
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
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        train_split=args.train_split,
        use_h5=True,
        env_family_filter=args.env_family_filter,
    )

    # Model
    if args.model_type == 'diffusion':
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
    elif args.model_type == 'multihyp':
        from src.model.se2_hypothesis_module import SE2MultiHypothesisModule  # noqa: E402

        model = SE2MultiHypothesisModule(
            context_size=args.context_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_hypotheses=args.num_hypotheses,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            assignment_temp=args.assignment_temp,
            cls_loss_weight=args.cls_loss_weight,
            angle_loss_weight=args.angle_loss_weight,
            reg_beta=args.reg_beta,
        )
    else:
        from src.model.se2_hypothesis_v2_module import SE2MultiHypothesisV2Module  # noqa: E402

        model = SE2MultiHypothesisV2Module(
            context_size=args.context_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_hypotheses=args.num_hypotheses,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            cls_loss_weight=args.cls_loss_weight,
            angle_loss_weight=args.angle_loss_weight,
            reg_beta=args.reg_beta,
            hyp_dropout_prob=args.hyp_dropout_prob,
            diversity_weight=args.diversity_weight,
            diversity_margin=args.diversity_margin,
            use_hyp_self_attn=args.use_hyp_self_attn,
            best_idx_noise_scale=args.best_idx_noise_scale,
        )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f} M")
    if args.init_ckpt_path:
        print(f"Loading weights from: {args.init_ckpt_path}")
        checkpoint = torch.load(args.init_ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights with missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")

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
        strategy=args.strategy if args.gpus > 1 else 'auto',
        precision=args.precision,
        gradient_clip_val=args.grad_clip,
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        num_sanity_val_steps=args.num_sanity_val_steps,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        benchmark=True,
    )

    trainer.fit(model=model, datamodule=dm, ckpt_path=args.ckpt_path)
    print("Training complete.")
    print(f"Checkpoints at: {save_dir / 'checkpoints'}")


if __name__ == '__main__':
    main()
