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

from src.model.hl_gauss import HLGauss


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

    # Precomputed same-face arc-neighbor lookup (edge -> list of (neighbor_edge, arc_distance)).
    # Edge layout: 60 edges, 4 faces × 15 pts, interleaved parity.
    #   even e <30: top face,   positions j = e//2     (e in {0,2,4,...,28})
    #   odd  e <30: bottom face, positions j = e//2     (e in {1,3,5,...,29})
    #   even e>=30: right face, positions j = (e-30)//2 (e in {30,32,...,58})
    #   odd  e>=30: left face,  positions j = (e-30)//2 (e in {31,33,...,59})
    # Same-face neighbors are e±2 (same parity, within [0,59]), arc distance = 1 per step.
    _FACE_NEIGHBORS: list  # set in __init_subclass__ below; built once as a class-level cache

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
        bce_reachable_only: bool = False,
        soft_edge_sigma: float = 0.0,
        soft_depth_sigma: float = 0.0,
        head_mode: str = "sigmoid_bce",
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

        # Ablation: if True, the BCE term is computed ONLY on reachable primitives
        # (no supervision to suppress unreachable cells) — tests whether reachability
        # supervision is a useful auxiliary task or wasted capacity (we mask at inference anyway).
        self.bce_reachable_only = bce_reachable_only

        # Soft Gaussian edge labels (CenterNet-style, sigma=0 → hard one-hot, identical to current).
        # soft_edge_sigma  > 0 → spread credit to same-face arc-neighbors.
        # soft_depth_sigma > 0 → spread credit to adjacent depths.
        # Both must be > 0 to activate (guard against accidentally enabling only one axis).
        self.soft_edge_sigma = soft_edge_sigma
        self.soft_depth_sigma = soft_depth_sigma

        # H1 FRAMING ablation (policy_framework journal): how the (60,D) head is trained.
        #   sigmoid_bce (default) — per-cell independent value (the scorer; BCE+Dice, unchanged).
        #   softmax_ce            — POLICY: one distribution over cells (AlphaZero-style). Target =
        #     f_grid (optionally soft-blurred) normalized over the loss_mask cells; logits masked to
        #     -inf outside loss_mask (legal-move masking); loss = cross-entropy. Dice not applicable.
        #     Samples with no positive cell are skipped (a policy needs a target distribution).
        #   hl_gauss              — BUDGET-Q VALUE (horizon_q_build_journal.md §4): network must emit
        #     (B,60,nd,bins) (EdgeCrossAttn value_bins>0); labels are gamma-discounted targets in [0,1]
        #     (H=1: == f_grid); loss = masked CE to the Gaussian-smoothed histogram (Stop-Regressing
        #     2403.03950). Inference value = E[bin]; rankings unchanged under the monotone map.
        # Ranking metrics are head-agnostic: softmax is monotone in logits, so eval_scorer's
        # argmax/top-k read the same either way.
        assert head_mode in ("sigmoid_bce", "softmax_ce", "hl_gauss"), head_mode
        self.head_mode = head_mode
        self._hl_gauss: Optional[HLGauss] = None   # built lazily from the head's bin count

        # Build neighbor table once (class-level cache pattern)
        ClassifierModule._build_face_neighbors()

    @staticmethod
    def _build_face_neighbors():
        """Build same-face arc-neighbor table if not already built.

        For each edge index e in [0,59], returns a list of (neighbor_e, arc_distance) tuples
        for all in-face neighbors reachable within a generous radius (up to 7 steps covers the
        full 15-point face). Stored as ClassifierModule._FACE_NEIGHBORS.
        """
        if hasattr(ClassifierModule, '_FACE_NEIGHBORS') and isinstance(
                getattr(ClassifierModule, '_FACE_NEIGHBORS', None), list):
            return  # already built

        neighbors = []  # neighbors[e] = list of (neighbor_e, arc_dist)
        for e in range(60):
            nbrs = []
            # Walk along the face in both directions (step = ±2 keeps same parity/face)
            for direction in (+2, -2):
                ne = e + direction
                arc_dist = 1
                while True:
                    # Must stay in [0,59], same parity (same face within top/bottom or right/left)
                    if ne < 0 or ne >= 60:
                        break
                    if ne % 2 != e % 2:
                        break  # crossed face boundary (parity changed — impossible with ±2 steps but guard)
                    # Check same half: {0..29} vs {30..59}
                    if (ne < 30) != (e < 30):
                        break  # crossed the top/bottom ↔ right/left boundary
                    nbrs.append((ne, arc_dist))
                    ne += direction
                    arc_dist += 1
            neighbors.append(nbrs)

        ClassifierModule._FACE_NEIGHBORS = neighbors

    def _build_soft_target(self, f_grid: torch.Tensor) -> torch.Tensor:
        """Build soft Gaussian target from hard binary f_grid.

        When soft_edge_sigma == soft_depth_sigma == 0 this is a no-op (returns f_grid unchanged).

        Args:
            f_grid: (B, 60, D) binary success labels in {0, 1}.

        Returns:
            soft_target: (B, 60, D) float in [0, 1] with same positives and smoothed neighbors.
        """
        if self.soft_edge_sigma <= 0.0 or self.soft_depth_sigma <= 0.0:
            return f_grid  # identity — preserves E9 exact reproducibility

        B, E, D = f_grid.shape  # E=60, D=5
        device = f_grid.device

        # We accumulate the maximum Gaussian weight over all positives.
        # spread[b, e, d] = max over all positive (ep, dp) of gaussian(Δedge) * gaussian(Δdepth)
        spread = torch.zeros_like(f_grid)

        # Precompute depth Gaussian weights for all Δd = 0..D-1
        depth_range = torch.arange(D, device=device, dtype=torch.float32)  # (D,)
        # gaussian(Δd) for each source depth dp and each target depth d: shape (D, D)
        # depth_gauss[dp, d] = exp(-(d - dp)^2 / (2 * sigma_d^2))
        sigma_d = self.soft_depth_sigma
        depth_delta = depth_range.unsqueeze(0) - depth_range.unsqueeze(1)  # (D, D): [dp, d]
        depth_gauss = torch.exp(-(depth_delta ** 2) / (2.0 * sigma_d ** 2))  # (D, D)

        sigma_e = self.soft_edge_sigma

        for ep in range(E):
            # Check if any sample has a positive at this edge (batched)
            has_positive = (f_grid[:, ep, :] > 0.5)  # (B, D)
            if not has_positive.any():
                continue

            # For the source edge ep itself: arc distance 0 → Gaussian weight = 1.0
            # Combine with depth Gaussian: contribution[b, d] = sum over dp of positive[b,dp]*depth_gauss[dp,d]
            # Then take max with existing spread.
            # Arc weight for ep itself = exp(0) = 1.0
            # contribution_ep[b, d] = 1.0 * sum_dp( f_grid[b,ep,dp] * depth_gauss[dp,d] )
            # But we want elementwise max over positives, so weight per (dp) positive separately:
            for dp in range(D):
                pos_mask = f_grid[:, ep, dp] > 0.5  # (B,) bool
                if not pos_mask.any():
                    continue
                # Contribution from positive at (ep, dp) to all (e_target, d_target):
                # edge_weight(ep→e_target) * depth_gauss[dp, d_target]
                # For source edge ep: edge_weight = 1.0 (arc_dist=0)
                depth_contrib = depth_gauss[dp]  # (D,) weights for d=0..D-1
                # Update spread for ep itself
                contrib_ep = depth_contrib.unsqueeze(0)  # (1, D)
                spread[:, ep, :] = torch.max(
                    spread[:, ep, :],
                    pos_mask.float().unsqueeze(1) * contrib_ep
                )
                # Update same-face arc-neighbors
                for (ne, arc_dist) in ClassifierModule._FACE_NEIGHBORS[ep]:
                    edge_weight = np.exp(-(arc_dist ** 2) / (2.0 * sigma_e ** 2))
                    contrib = edge_weight * depth_contrib  # (D,)
                    spread[:, ne, :] = torch.max(
                        spread[:, ne, :],
                        pos_mask.float().unsqueeze(1) * contrib.unsqueeze(0)
                    )

        # Positives always stay exactly 1.0; neighbors get partial credit in [0,1)
        soft_target = torch.max(f_grid, spread)
        return soft_target

    def forward(self, x: torch.Tensor, contact_px=None, x_zoom=None, contact_px_zoom=None, H=None) -> torch.Tensor:
        # dual-crop fields are passed only when present -> single-crop path is unchanged (DiT + EdgeCrossAttn)
        # H (B,) long: remaining push budget, passed only when present (budget-conditioned EdgeCrossAttn)
        kw = {} if H is None else {"H": H}
        if x_zoom is not None:
            return self.network(x, contact_px, x_zoom, contact_px_zoom, **kw)
        return self.network(x, contact_px, **kw)

    def _compute_masked_loss(self, logits: torch.Tensor,
                              labels: torch.Tensor,
                              mask: torch.Tensor) -> torch.Tensor:
        """BCE on all 600 primitives + Dice on reachable only.

        BCE on all 600: teaches the model to output 0 for unreachable primitives.
        Dice on reachable only: forces sharp F boundaries where labels are balanced
        (~67% positive among reachable), avoiding the collapse that happens when
        dice sees 93% zeros on all 600.

        When soft_edge_sigma > 0 and soft_depth_sigma > 0, builds a soft Gaussian
        target (same-face arc-neighbors and adjacent depths get partial credit) before
        computing BCE. The Dice term still uses the soft target. When both sigmas are 0
        (the default) behaviour is identical to the original hard-label BCE+Dice.

        Args:
            logits: (B, 60, D) raw predictions
            labels: (B, 60, D) binary labels (0=fail or unreachable, 1=success)
            mask:   (B, 60, D) reachability mask (1=reachable, 0=unreachable)
        """
        if self.head_mode == "hl_gauss":
            # logits (B,60,nd,bins); labels = gamma targets in [0,1] (NOT soft-blurred — they are
            # already real-valued); masked CE to the Gaussian-smoothed histogram.
            if self._hl_gauss is None or self._hl_gauss.num_bins != logits.shape[-1]:
                self._hl_gauss = HLGauss(num_bins=logits.shape[-1])
            return self._hl_gauss.loss(logits, labels, mask)

        # Build soft target (no-op when sigmas are 0)
        labels = self._build_soft_target(labels)
        B = logits.shape[0]
        logits_flat = logits.reshape(B, -1)   # (B, 600)
        labels_flat = labels.reshape(B, -1)   # (B, 600)
        mask_flat = mask.reshape(B, -1)       # (B, 600)

        if self.head_mode == "softmax_ce":
            # POLICY framing: cross-entropy between the normalized target distribution and a
            # softmax over the masked (legal) cells. Multimodal targets are fine (several 1s →
            # uniform over solutions; soft blur → smeared distribution over the contact manifold).
            masked_logits = logits_flat.masked_fill(mask_flat <= 0, float("-inf"))
            logp = torch.log_softmax(masked_logits, dim=1)
            tgt = labels_flat * mask_flat
            tgt_sum = tgt.sum(dim=1, keepdim=True)
            valid = (tgt_sum.squeeze(1) > 0)              # need >=1 positive to define a distribution
            if not valid.any():
                return logits_flat.sum() * 0.0            # keep graph; no defined target this batch
            p = tgt[valid] / tgt_sum[valid]
            ce = -(p * logp[valid].clamp(min=-30.0)).sum(dim=1)   # clamp guards -inf*0 -> nan
            return ce.mean()

        # BCE: on all 600 (default — also supervises unreachable=0), or reachable-only (ablation).
        if self.use_focal_loss:
            bce = self._focal_loss(logits_flat, labels_flat)
        else:
            pw = torch.tensor([self.pos_weight], device=logits.device)
            if self.bce_reachable_only:
                per = F.binary_cross_entropy_with_logits(
                    logits_flat, labels_flat, pos_weight=pw, reduction='none')
                bce = (per * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)
            else:
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
        # H5-sampling ablation: loss_mask = the cells actually "tried" (== r_mask when exhaustive).
        # Requires bce_reachable_only=true when loss_mask != r_mask (else the all-600 BCE leaks).
        loss_mask = batch.get('loss_mask', r_mask)

        logits = self(context, batch.get('contact_px'), batch.get('context_zoom'), batch.get('contact_px_zoom'),
                      H=batch.get('H'))  # (B, 60, num_depths) — or (B, 60, nd, bins) for hl_gauss
        loss = self._compute_masked_loss(logits, f_labels, loss_mask)

        self.train_loss(loss)
        self.log('train_loss', self.train_loss, on_step=True, on_epoch=True,
                 prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        context = batch['context']
        f_labels = batch['f_labels']
        r_mask = batch['r_mask']
        ratios = batch.get('ratio', None)

        logits = self(context, batch.get('contact_px'), batch.get('context_zoom'), batch.get('contact_px_zoom'),
                      H=batch.get('H'))
        loss = self._compute_masked_loss(logits, f_labels, r_mask)

        self.val_loss(loss)
        self.log('val_loss', self.val_loss, on_epoch=True, prog_bar=True)

        # Top-k accuracy overall + per difficulty. hl_gauss: rank by E[bin] values — _compute_metrics
        # applies sigmoid internally, which is monotone, so top-k over values is unchanged.
        if self.head_mode == "hl_gauss":
            logits = self._hl_gauss.value(logits)
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
