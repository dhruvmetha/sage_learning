from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torchmetrics import MeanMetric, MinMetric

from .networks.multiscale_pose_hypothesis import MultiScaleHypothesisPosePredictor
from .se2_hypothesis_module import _SE2MultiHypothesisBase


class SE2MultiHypothesisV2Module(_SE2MultiHypothesisBase):
    def __init__(
        self,
        context_size: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_hypotheses: int = 4,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        target_mean: Optional[List[float]] = None,
        target_std: Optional[List[float]] = None,
        cls_loss_weight: float = 0.2,
        angle_loss_weight: float = 1.0,
        reg_beta: float = 0.5,
        hyp_dropout_prob: float = 0.25,
        diversity_weight: float = 0.02,
        diversity_margin: float = 0.75,
        use_hyp_self_attn: bool = False,
        best_idx_noise_scale: float = 1e-4,
        use_local: bool = True,
        use_region_masks: bool = True,
        use_coord_grid: bool = False,
        crop_size_meters: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.predictor = MultiScaleHypothesisPosePredictor(
            image_channels=5,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_hypotheses=num_hypotheses,
            use_self_attn=use_hyp_self_attn,
            per_slot_heads=True,
        )
        self._init_shared_state(
            context_size=context_size,
            lr=lr,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            target_mean=target_mean,
            target_std=target_std,
            crop_size_meters=crop_size_meters,
        )

        self.train_loss = MeanMetric()
        self.train_reg_loss = MeanMetric()
        self.train_cls_loss = MeanMetric()
        self.train_div_loss = MeanMetric()
        self.train_top1_prob = MeanMetric()
        self.train_hyp_pair_xy_m = MeanMetric()
        self.train_hyp_pair_th_deg = MeanMetric()

        self.val_loss = MeanMetric()
        self.val_reg_loss = MeanMetric()
        self.val_cls_loss = MeanMetric()
        self.val_div_loss = MeanMetric()
        self.val_loss_best = MinMetric()
        self.val_xy_mae_m = MeanMetric()
        self.val_th_mae_deg = MeanMetric()
        self.val_oracle_xy_mae_m = MeanMetric()
        self.val_oracle_th_mae_deg = MeanMetric()
        self.val_top1_prob = MeanMetric()
        self.val_hyp_pair_xy_m = MeanMetric()
        self.val_hyp_pair_th_deg = MeanMetric()

    def _sample_active_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        keep_prob = 1.0 - float(self.hparams.hyp_dropout_prob)
        if (not self.training) or keep_prob >= 1.0:
            return torch.ones(batch_size, self.hparams.num_hypotheses, dtype=torch.bool, device=device)

        active = torch.rand(batch_size, self.hparams.num_hypotheses, device=device) < keep_prob
        empty_rows = ~active.any(dim=-1)
        if empty_rows.any():
            row_ids = empty_rows.nonzero(as_tuple=False).squeeze(1)
            rescue_idx = torch.randint(
                0,
                self.hparams.num_hypotheses,
                (int(row_ids.numel()),),
                device=device,
            )
            active[row_ids, :] = False
            active[row_ids, rescue_idx] = True
        return active

    def _diversity_loss(self, pred_real: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        if self.hparams.diversity_weight <= 0.0 or pred_real.shape[1] < 2:
            return pred_real.new_zeros(())

        xy_diff = (pred_real[:, :, None, :2] - pred_real[:, None, :, :2]) / self.target_std[:2]
        th_diff = self._wrap_angle(pred_real[:, :, None, 2] - pred_real[:, None, :, 2]) / self.target_std[2]
        dist = torch.sqrt(
            xy_diff.square().sum(dim=-1)
            + self.hparams.angle_loss_weight * th_diff.square()
            + 1e-8
        )
        gap = F.relu(self.hparams.diversity_margin - dist)
        pair_weight = probs[:, :, None] * probs[:, None, :]
        upper = torch.triu(
            torch.ones(
                pred_real.shape[1],
                pred_real.shape[1],
                dtype=torch.bool,
                device=pred_real.device,
            ),
            diagonal=1,
        )
        penalty = gap.square() * pair_weight
        return penalty[:, upper].mean()

    def _pairwise_spread(self, pred_real: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if pred_real.shape[1] < 2:
            zero = pred_real.new_zeros(())
            return zero, zero

        xy_diff = pred_real[:, :, None, :2] - pred_real[:, None, :, :2]
        th_diff = self._wrap_angle(pred_real[:, :, None, 2] - pred_real[:, None, :, 2])
        upper = torch.triu(
            torch.ones(
                pred_real.shape[1],
                pred_real.shape[1],
                dtype=torch.bool,
                device=pred_real.device,
            ),
            diagonal=1,
        )
        pair_xy = torch.sqrt(xy_diff.square().sum(dim=-1) + 1e-8)[:, upper]
        pair_th = th_diff.abs()[:, upper] * (180.0 / math.pi)
        return pair_xy.mean(), pair_th.mean()

    def _compute_losses(
        self,
        pred_real: torch.Tensor,
        logits: torch.Tensor,
        target_real: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pose_cost = self._compute_pose_cost(pred_real, target_real)
        probs = torch.softmax(logits, dim=-1)
        active_mask = self._sample_active_mask(pred_real.shape[0], pred_real.device)

        select_cost = pose_cost.masked_fill(~active_mask, float("inf"))
        if self.training and self.hparams.best_idx_noise_scale > 0:
            select_cost = select_cost + torch.rand_like(select_cost) * self.hparams.best_idx_noise_scale

        best_idx = select_cost.argmin(dim=-1)
        reg_loss = pose_cost.gather(1, best_idx.unsqueeze(1)).squeeze(1).mean()
        cls_loss = F.cross_entropy(logits, best_idx)
        div_loss = self._diversity_loss(pred_real, probs)
        loss = reg_loss + self.hparams.cls_loss_weight * cls_loss + self.hparams.diversity_weight * div_loss

        oracle_idx = pose_cost.argmin(dim=-1)
        top1_idx = probs.argmax(dim=-1)
        return {
            "loss": loss,
            "reg_loss": reg_loss,
            "cls_loss": cls_loss,
            "div_loss": div_loss,
            "pose_cost": pose_cost,
            "probs": probs,
            "best_idx": oracle_idx,
            "top1_idx": top1_idx,
        }

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        pred_real, logits = self._predict(batch["context"])
        losses = self._compute_losses(pred_real, logits, batch["target"])
        pair_xy, pair_th = self._pairwise_spread(pred_real)
        top1_prob = losses["probs"].max(dim=-1).values.mean()

        self.train_loss(losses["loss"])
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_reg_loss(losses["reg_loss"])
        self.log("train/loss_reg", self.train_reg_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.train_cls_loss(losses["cls_loss"])
        self.log("train/loss_cls", self.train_cls_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.train_div_loss(losses["div_loss"])
        self.log("train/loss_div", self.train_div_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.train_top1_prob(top1_prob)
        self.log("train/top1_prob", self.train_top1_prob, on_step=True, on_epoch=True, prog_bar=False)
        self.train_hyp_pair_xy_m(pair_xy)
        self.log("train/hyp_pair_xy_m", self.train_hyp_pair_xy_m, on_step=True, on_epoch=True, prog_bar=False)
        self.train_hyp_pair_th_deg(pair_th)
        self.log("train/hyp_pair_th_deg", self.train_hyp_pair_th_deg, on_step=True, on_epoch=True, prog_bar=False)
        return losses["loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        if batch_idx == 0:
            self._viz_batch = {k: v.detach() for k, v in batch.items()}

        pred_real, logits = self._predict(batch["context"])
        losses = self._compute_losses(pred_real, logits, batch["target"])
        top1_pred = self._gather_hypothesis(pred_real, losses["top1_idx"])
        oracle_pred = self._gather_hypothesis(pred_real, losses["best_idx"])
        gt = batch["target"]

        xy_err = torch.sqrt(((top1_pred[:, :2] - gt[:, :2]) ** 2).sum(dim=-1)).mean()
        th_err = self._wrap_angle(top1_pred[:, 2] - gt[:, 2]).abs().mean() * (180.0 / math.pi)
        oracle_xy_err = torch.sqrt(((oracle_pred[:, :2] - gt[:, :2]) ** 2).sum(dim=-1)).mean()
        oracle_th_err = self._wrap_angle(oracle_pred[:, 2] - gt[:, 2]).abs().mean() * (180.0 / math.pi)
        top1_prob = losses["probs"].max(dim=-1).values.mean()
        pair_xy, pair_th = self._pairwise_spread(pred_real)

        self.val_loss(losses["loss"])
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_reg_loss(losses["reg_loss"])
        self.log("val/loss_reg", self.val_reg_loss, on_epoch=True, prog_bar=False, sync_dist=True)
        self.val_cls_loss(losses["cls_loss"])
        self.log("val/loss_cls", self.val_cls_loss, on_epoch=True, prog_bar=False, sync_dist=True)
        self.val_div_loss(losses["div_loss"])
        self.log("val/loss_div", self.val_div_loss, on_epoch=True, prog_bar=False, sync_dist=True)

        self.val_xy_mae_m(xy_err)
        self.val_th_mae_deg(th_err)
        self.val_oracle_xy_mae_m(oracle_xy_err)
        self.val_oracle_th_mae_deg(oracle_th_err)
        self.val_top1_prob(top1_prob)
        self.val_hyp_pair_xy_m(pair_xy)
        self.val_hyp_pair_th_deg(pair_th)

        self.log("val/xy_mae_m", self.val_xy_mae_m, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/th_mae_deg", self.val_th_mae_deg, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/oracle_xy_mae_m", self.val_oracle_xy_mae_m, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/oracle_th_mae_deg", self.val_oracle_th_mae_deg, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/top1_prob", self.val_top1_prob, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/hyp_pair_xy_m", self.val_hyp_pair_xy_m, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/hyp_pair_th_deg", self.val_hyp_pair_th_deg, on_epoch=True, prog_bar=False, sync_dist=True)
        return losses["loss"]


__all__ = ["SE2MultiHypothesisV2Module"]
