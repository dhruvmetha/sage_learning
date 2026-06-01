from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import cv2
import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import MeanMetric, MinMetric

from .networks.multiscale_pose_hypothesis import MultiScaleHypothesisPosePredictor


class _SE2MultiHypothesisBase(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self._viz_batch: Optional[Dict[str, torch.Tensor]] = None

    def _init_shared_state(
        self,
        context_size: int,
        lr: float,
        weight_decay: float,
        warmup_steps: int,
        target_mean: Optional[List[float]],
        target_std: Optional[List[float]],
        crop_size_meters: float,
    ) -> None:
        mean = torch.tensor(target_mean if target_mean is not None else [0.0, 0.0, 0.0], dtype=torch.float32)
        std = torch.tensor(target_std if target_std is not None else [1.0, 1.0, 1.0], dtype=torch.float32)
        self.register_buffer("target_mean", mean)
        self.register_buffer("target_std", torch.clamp(std, min=1e-6))

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.viz_n_scenes = 8
        self.viz_n_samples = 4
        self.viz_crop_size_meters = crop_size_meters
        self.viz_output_size = context_size

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        state_dict = checkpoint.setdefault("state_dict", {})
        state_dict.setdefault("target_mean", self.target_mean.detach().clone())
        state_dict.setdefault("target_std", self.target_std.detach().clone())

    @staticmethod
    def _wrap_angle(delta: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(delta), torch.cos(delta))

    def _target_to_model_space(self, target_real: torch.Tensor) -> torch.Tensor:
        return (target_real - self.target_mean) / self.target_std

    def _target_from_model_space(self, target_model: torch.Tensor) -> torch.Tensor:
        return target_model * self.target_std + self.target_mean

    def _predict(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred_model, logits = self.predictor(context)
        pred_real = self._target_from_model_space(pred_model)
        return pred_real, logits

    def _compute_pose_cost(self, pred_real: torch.Tensor, target_real: torch.Tensor) -> torch.Tensor:
        target_rep = target_real.unsqueeze(1)
        xy_diff = (pred_real[..., :2] - target_rep[..., :2]) / self.target_std[:2]
        th_diff = self._wrap_angle(pred_real[..., 2] - target_rep[..., 2]) / self.target_std[2]

        xy_cost = F.smooth_l1_loss(
            xy_diff,
            torch.zeros_like(xy_diff),
            beta=self.hparams.reg_beta,
            reduction="none",
        ).sum(dim=-1)
        th_cost = F.smooth_l1_loss(
            th_diff,
            torch.zeros_like(th_diff),
            beta=self.hparams.reg_beta,
            reduction="none",
        )
        return xy_cost + self.hparams.angle_loss_weight * th_cost

    @staticmethod
    def _gather_hypothesis(pred_real: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        gather_idx = indices.view(-1, 1, 1).expand(-1, 1, pred_real.shape[-1])
        return pred_real.gather(1, gather_idx).squeeze(1)

    @staticmethod
    def _render_target_mask(
        delta: np.ndarray,
        obj_size_xy: np.ndarray,
        crop_size_meters: float,
        output_size: int,
    ) -> np.ndarray:
        mask = np.zeros((output_size, output_size), dtype=np.float32)
        dx, dy, dth = float(delta[0]), float(delta[1]), float(delta[2])
        sx, sy = float(obj_size_xy[0]), float(obj_size_xy[1])
        scale = output_size / crop_size_meters

        cx = output_size / 2 + dx * scale
        cy = output_size / 2 + dy * scale
        w_px = sx * scale * 2.0
        h_px = sy * scale * 2.0
        rect = ((cx, cy), (w_px, h_px), np.degrees(dth))
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillPoly(mask, [box], 1.0)
        return mask

    @torch.no_grad()
    def predict_hypotheses(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred_real, logits = self._predict(context)
        return pred_real, torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict_se3_world_delta(self, context: torch.Tensor) -> torch.Tensor:
        pred_real, _ = self.predict_hypotheses(context)
        return pred_real

    def on_validation_epoch_end(self) -> None:
        self.val_loss_best.update(self.val_loss.compute())
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=False, sync_dist=True)

        if self._viz_batch is None:
            return
        if self.trainer is not None and not self.trainer.is_global_zero:
            self._viz_batch = None
            return
        if not hasattr(self.logger, "experiment"):
            self._viz_batch = None
            return

        try:
            self._do_val_viz()
        finally:
            self._viz_batch = None

    def _do_val_viz(self) -> None:
        import torchvision
        import wandb

        batch = self._viz_batch
        if batch is None:
            return

        device = batch["context"].device
        ctx_size = batch["context"].shape[-1]
        n_scenes = min(self.viz_n_scenes, batch["context"].shape[0])

        def to_disp(x: torch.Tensor) -> torch.Tensor:
            return torch.clamp((x + 1) / 2, 0, 1)

        log_dict: Dict[str, Any] = {}
        for i in range(n_scenes):
            ctx_b = batch["context"][i : i + 1]
            pre_np = batch["pre_pose"][i].cpu().numpy()
            obj_np = batch["obj_size"][i].cpu().numpy()
            gt_np = batch["target"][i].cpu().numpy()

            pred_real, probs = self.predict_hypotheses(ctx_b)
            pred_real = pred_real[0]
            probs = probs[0]
            order = torch.argsort(probs, descending=True)
            pred_real = pred_real[order]
            probs = probs[order]
            n_preds = min(self.viz_n_samples, pred_real.shape[0])

            static = to_disp(ctx_b[0, 0])
            movable = to_disp(ctx_b[0, 1])
            target_obj = to_disp(ctx_b[0, 2])
            robot_region = to_disp(ctx_b[0, 3])
            goal_region = to_disp(ctx_b[0, 4])

            img_scene = torch.stack([static, movable, target_obj]).unsqueeze(0)
            img_reach = torch.stack([robot_region, goal_region, target_obj]).unsqueeze(0)

            def render_delta_to_mask(delta: np.ndarray) -> np.ndarray:
                draw = np.array([delta[0], delta[1], float(pre_np[2]) + float(delta[2])], dtype=np.float32)
                return self._render_target_mask(draw, obj_np[:2], self.viz_crop_size_meters, ctx_size)

            gt_mask_np = render_delta_to_mask(gt_np)
            pred_masks_np = [render_delta_to_mask(pred_real[j].cpu().numpy()) for j in range(n_preds)]

            def make_pred_img(mask_np: np.ndarray) -> torch.Tensor:
                mask_t = torch.from_numpy(mask_np).to(device)
                return torch.stack([static, mask_t, target_obj]).unsqueeze(0)

            row = [img_scene, img_reach, make_pred_img(gt_mask_np)]
            row.extend(make_pred_img(mask_np) for mask_np in pred_masks_np)
            grid = torchvision.utils.make_grid(
                torch.cat(row, dim=0),
                nrow=len(row),
                normalize=True,
                padding=2,
            )
            grid_np = grid.cpu().permute(1, 2, 0).numpy()
            prob_text = ", ".join(f"{float(p):.3f}" for p in probs[:n_preds].cpu())
            caption = (
                f"Epoch {self.current_epoch} | Scene | Reach | GT | Pred1-{n_preds} "
                f"(top probs: {prob_text})"
            )
            log_dict[f"val_sample_{i+1}"] = wandb.Image(grid_np, caption=caption)

        self.logger.experiment.log(log_dict)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.warmup_steps > 0:
            def lr_lambda(step: int) -> float:
                if step < self.warmup_steps:
                    return float(step) / max(1, self.warmup_steps)
                return 1.0

            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
        return opt


class SE2MultiHypothesisModule(_SE2MultiHypothesisBase):
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
        assignment_temp: float = 0.35,
        cls_loss_weight: float = 0.2,
        angle_loss_weight: float = 1.0,
        reg_beta: float = 0.5,
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
            use_self_attn=True,
            per_slot_heads=False,
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
        self.train_top1_prob = MeanMetric()

        self.val_loss = MeanMetric()
        self.val_reg_loss = MeanMetric()
        self.val_cls_loss = MeanMetric()
        self.val_loss_best = MinMetric()
        self.val_xy_mae_m = MeanMetric()
        self.val_th_mae_deg = MeanMetric()
        self.val_oracle_xy_mae_m = MeanMetric()
        self.val_oracle_th_mae_deg = MeanMetric()
        self.val_top1_prob = MeanMetric()

    def _compute_losses(
        self,
        pred_real: torch.Tensor,
        logits: torch.Tensor,
        target_real: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pose_cost = self._compute_pose_cost(pred_real, target_real)
        assign = torch.softmax(-pose_cost / self.hparams.assignment_temp, dim=-1).detach()
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        reg_loss = (assign * pose_cost).sum(dim=-1).mean()
        cls_loss = -(assign * log_probs).sum(dim=-1).mean()
        loss = reg_loss + self.hparams.cls_loss_weight * cls_loss
        best_idx = pose_cost.argmin(dim=-1)
        top1_idx = probs.argmax(dim=-1)
        return {
            "loss": loss,
            "reg_loss": reg_loss,
            "cls_loss": cls_loss,
            "pose_cost": pose_cost,
            "assign": assign,
            "probs": probs,
            "best_idx": best_idx,
            "top1_idx": top1_idx,
        }

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        pred_real, logits = self._predict(batch["context"])
        losses = self._compute_losses(pred_real, logits, batch["target"])
        loss = losses["loss"]

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_reg_loss(losses["reg_loss"])
        self.log("train/loss_reg", self.train_reg_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.train_cls_loss(losses["cls_loss"])
        self.log("train/loss_cls", self.train_cls_loss, on_step=True, on_epoch=True, prog_bar=False)
        top1_prob = losses["probs"].max(dim=-1).values.mean()
        self.train_top1_prob(top1_prob)
        self.log("train/top1_prob", self.train_top1_prob, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        if batch_idx == 0:
            self._viz_batch = {k: v.detach() for k, v in batch.items()}

        pred_real, logits = self._predict(batch["context"])
        losses = self._compute_losses(pred_real, logits, batch["target"])
        loss = losses["loss"]

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_reg_loss(losses["reg_loss"])
        self.log("val/loss_reg", self.val_reg_loss, on_epoch=True, prog_bar=False, sync_dist=True)
        self.val_cls_loss(losses["cls_loss"])
        self.log("val/loss_cls", self.val_cls_loss, on_epoch=True, prog_bar=False, sync_dist=True)

        top1_pred = self._gather_hypothesis(pred_real, losses["top1_idx"])
        oracle_pred = self._gather_hypothesis(pred_real, losses["best_idx"])
        gt = batch["target"]

        xy_err = torch.sqrt(((top1_pred[:, :2] - gt[:, :2]) ** 2).sum(dim=-1)).mean()
        th_err = self._wrap_angle(top1_pred[:, 2] - gt[:, 2]).abs().mean() * (180.0 / math.pi)
        oracle_xy_err = torch.sqrt(((oracle_pred[:, :2] - gt[:, :2]) ** 2).sum(dim=-1)).mean()
        oracle_th_err = self._wrap_angle(oracle_pred[:, 2] - gt[:, 2]).abs().mean() * (180.0 / math.pi)
        top1_prob = losses["probs"].max(dim=-1).values.mean()

        self.val_xy_mae_m(xy_err)
        self.val_th_mae_deg(th_err)
        self.val_oracle_xy_mae_m(oracle_xy_err)
        self.val_oracle_th_mae_deg(oracle_th_err)
        self.val_top1_prob(top1_prob)

        self.log("val/xy_mae_m", self.val_xy_mae_m, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/th_mae_deg", self.val_th_mae_deg, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/oracle_xy_mae_m", self.val_oracle_xy_mae_m, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/oracle_th_mae_deg", self.val_oracle_th_mae_deg, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/top1_prob", self.val_top1_prob, on_epoch=True, prog_bar=False, sync_dist=True)
        return losses["loss"]


__all__ = ["SE2MultiHypothesisModule"]
