"""
PyTorch Lightning module for primitive feasibility classifier.

Predicts which of 600 push primitives (60 contact points × 10 depths)
will successfully open a passage, given scene masks as input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torchmetrics import MeanMetric
from typing import Dict, Any, Optional
import numpy as np


class ConvBlock(nn.Module):
    """Conv2d + BatchNorm + ReLU + MaxPool."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class PrimitiveClassifierCNN(nn.Module):
    """Simple CNN baseline for primitive feasibility prediction.

    Input: (B, C, H, W) scene masks
    Output: (B, 60, 10) logits per primitive
    """
    def __init__(self, in_channels: int = 5, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 32),     # 64 → 32
            ConvBlock(32, 64),              # 32 → 16
            ConvBlock(64, 128),             # 16 → 8
            ConvBlock(128, hidden_dim),     # 8 → 4
            nn.AdaptiveAvgPool2d(1),        # 4 → 1
            nn.Flatten(),                   # → hidden_dim
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 600),            # 60 × 10 primitives
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) scene masks
        Returns:
            (B, 60, 10) logits
        """
        features = self.encoder(x)
        logits = self.head(features)
        return logits.view(-1, 60, 10)


class ClassifierModule(pl.LightningModule):
    """Lightning module for training the primitive classifier.

    Handles:
    - Masked BCE loss (only compute on reachable primitives)
    - Top-k accuracy metrics by difficulty
    - Validation logging
    """

    def __init__(
        self,
        network: nn.Module,
        base_lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        decay_steps: int = 100000,
        end_lr: float = 1e-6,
        pos_weight: float = 1.0,
        use_focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dice_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['network'])
        self.network = network

        # Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # Positive weight for class imbalance
        self.pos_weight = pos_weight

        # Focal loss parameters
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # Dice loss weight (set to 0 to disable)
        self.dice_weight = dice_weight

    def forward(self, x: torch.Tensor, contact_px=None) -> torch.Tensor:
        return self.network(x, contact_px)

    def _compute_masked_loss(self, logits: torch.Tensor,
                              labels: torch.Tensor,
                              mask: torch.Tensor) -> torch.Tensor:
        """BCE on all 600 primitives + Dice on reachable only.

        BCE on all 600: teaches the model to output 0 for unreachable primitives.
        Dice on reachable only: forces sharp F boundaries where labels are balanced
        (~67% positive among reachable), avoiding the collapse that happens when
        dice sees 93% zeros on all 600.

        Args:
            logits: (B, 60, 10) raw predictions
            labels: (B, 60, 10) binary labels (0=fail or unreachable, 1=success)
            mask: (B, 60, 10) reachability mask (1=reachable, 0=unreachable)
        """
        B = logits.shape[0]
        logits_flat = logits.reshape(B, -1)   # (B, 600)
        labels_flat = labels.reshape(B, -1)   # (B, 600)
        mask_flat = mask.reshape(B, -1)       # (B, 600)

        # BCE on all 600: learn unreachable=0 + reachable labels
        if self.use_focal_loss:
            bce = self._focal_loss(logits_flat, labels_flat)
        else:
            pw = torch.tensor([self.pos_weight], device=logits.device)
            bce = F.binary_cross_entropy_with_logits(logits_flat, labels_flat, pos_weight=pw)

        # Dice on reachable only: sharp F boundaries where it matters
        if self.dice_weight > 0:
            probs = torch.sigmoid(logits_flat)

            # Mask to reachable primitives per sample
            masked_probs = probs * mask_flat
            masked_labels = labels_flat * mask_flat

            intersection = (masked_probs * masked_labels).sum(dim=1)
            union = masked_probs.sum(dim=1) + masked_labels.sum(dim=1)
            dice = 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)
            return bce + self.dice_weight * dice.mean()

        return bce

    def _focal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Focal loss: downweights easy-to-classify examples, focuses on hard ones.

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

        where p_t = p if y=1, (1-p) if y=0.
        """
        probs = torch.sigmoid(logits)
        # p_t = probability of the true class
        p_t = probs * labels + (1 - probs) * (1 - labels)
        # alpha_t: alpha for positives, (1-alpha) for negatives
        alpha_t = self.focal_alpha * labels + (1 - self.focal_alpha) * (1 - labels)
        # Focal modulation: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.focal_gamma

        # BCE per element
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')

        loss = alpha_t * focal_weight * bce
        return loss.mean()

    @staticmethod
    def _classify_difficulty(ratio: float) -> str:
        if ratio < 0.05: return 'very_hard'
        elif ratio < 0.15: return 'hard'
        elif ratio < 0.40: return 'medium'
        elif ratio < 0.70: return 'easy'
        else: return 'very_easy'

    def _compute_metrics(self, logits: torch.Tensor,
                          labels: torch.Tensor,
                          mask: torch.Tensor,
                          ratios: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Compute top-k accuracy overall and per difficulty category."""
        B = logits.shape[0]
        metrics = {}

        # Overall counters
        top1_correct = 0
        top5_correct = 0
        total = 0

        # Per-difficulty counters
        diff_top1 = {'very_hard': 0, 'hard': 0, 'medium': 0, 'easy': 0, 'very_easy': 0}
        diff_top5 = {'very_hard': 0, 'hard': 0, 'medium': 0, 'easy': 0, 'very_easy': 0}
        diff_total = {'very_hard': 0, 'hard': 0, 'medium': 0, 'easy': 0, 'very_easy': 0}

        with torch.no_grad():
            for i in range(B):
                l = labels[i]             # (60, 10)
                s = torch.sigmoid(logits[i])  # (60, 10)

                if l.sum() == 0:
                    continue

                total += 1

                # Use all 600 primitives (unmasked model predicts F directly)
                scores_flat = s.reshape(-1)
                labels_flat = l.reshape(-1)

                # Top-1
                top1_idx = scores_flat.argmax()
                hit1 = labels_flat[top1_idx] == 1
                if hit1:
                    top1_correct += 1

                # Top-5
                k = min(5, scores_flat.numel())
                top5_indices = scores_flat.topk(k).indices
                hit5 = labels_flat[top5_indices].sum() > 0
                if hit5:
                    top5_correct += 1

                # Per-difficulty
                if ratios is not None:
                    diff = self._classify_difficulty(float(ratios[i]))
                    diff_total[diff] += 1
                    if hit1: diff_top1[diff] += 1
                    if hit5: diff_top5[diff] += 1

        if total > 0:
            metrics['top1_acc'] = top1_correct / total
            metrics['top5_acc'] = top5_correct / total

        # Per-difficulty metrics
        for diff in ['very_hard', 'hard', 'medium', 'easy', 'very_easy']:
            n = diff_total[diff]
            if n > 0:
                metrics[f'top1_{diff}'] = diff_top1[diff] / n
                metrics[f'top5_{diff}'] = diff_top5[diff] / n

        return metrics

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        context = batch['context']      # (B, C, H, W)
        f_labels = batch['f_labels']    # (B, 60, 10)
        r_mask = batch['r_mask']        # (B, 60, 10)

        logits = self(context, batch.get('contact_px'))  # (B, 60, num_depths)
        loss = self._compute_masked_loss(logits, f_labels, r_mask)

        self.train_loss(loss)
        self.log('train_loss', self.train_loss, on_step=True, on_epoch=True,
                 prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        context = batch['context']
        f_labels = batch['f_labels']
        r_mask = batch['r_mask']
        ratios = batch.get('ratio', None)

        logits = self(context, batch.get('contact_px'))
        loss = self._compute_masked_loss(logits, f_labels, r_mask)

        self.val_loss(loss)
        self.log('val_loss', self.val_loss, on_epoch=True, prog_bar=True)

        # Top-k accuracy overall + per difficulty
        metrics = self._compute_metrics(logits, f_labels, r_mask, ratios=ratios)
        for k, v in metrics.items():
            prog = k in ('top1_acc', 'top5_acc')
            self.log(f'val_{k}', v, on_epoch=True, prog_bar=prog, batch_size=context.shape[0])

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.base_lr,
            weight_decay=self.hparams.weight_decay,
        )

        warmup_steps = self.hparams.warmup_steps
        decay_steps = self.hparams.decay_steps
        base_lr = self.hparams.base_lr
        end_lr = self.hparams.end_lr

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = min((step - warmup_steps) / max(decay_steps, 1), 1.0)
            cosine = 0.5 * (1 + np.cos(np.pi * progress))
            return max(end_lr / base_lr, cosine)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
