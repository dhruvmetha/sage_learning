from __future__ import annotations

from typing import Any, Dict, Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torchmetrics import MeanMetric, MinMetric

from .base import BasePath, BaseSampler


class GenerativeModuleCropped(pl.LightningModule):
    """Cropped diffusion module with cross-attention context conditioning."""

    def __init__(
        self,
        network: nn.Module,
        path: BasePath,
        sampler: BaseSampler,
        optimizer: Any,
        context_size: int = 64,
        crop_size: int = 32,
        context_channels: int = 5,
        target_channels: int = 1,
        use_local: bool = True,
        aux_loss_weight: float = 0.0,
        warmup_steps: int = 0,
        decay_steps: int = 0,
        end_lr: float = 0.0,
    ):
        super().__init__()
        self.network = network
        self.path = path
        self.sampler = sampler
        self.optimizer_partial = optimizer
        self.context_size = context_size
        self.crop_size = crop_size
        self.context_channels = context_channels
        self.target_channels = target_channels
        self.use_local = use_local
        self.aux_loss_weight = aux_loss_weight
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.end_lr = end_lr

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_loss_best = MinMetric()

        self.save_hyperparameters(ignore=["network", "path", "sampler", "optimizer"])

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.network(x, t, context)

    def _build_context(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.use_local:
            parts = [
                batch["static"],
                batch["movable"],
                batch["target_object"],
                batch.get("robot_region", torch.zeros_like(batch["static"])),
                batch.get("goal_sample_region", torch.zeros_like(batch["static"])),
            ]
        else:
            parts = [
                batch["robot"],
                batch["goal"],
                batch["movable"],
                batch["static"],
                batch["target_object"],
            ]
        return torch.cat(parts, dim=1)

    def _get_target(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        target = batch["target_goal_mask"]
        if target.dim() == 3:
            target = target.unsqueeze(1)
        return target

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        context = self._build_context(batch)
        x_1 = self._get_target(batch)
        x_0 = torch.randn_like(x_1)

        sample = self.path.sample(x_0=x_0, x_1=x_1)
        prediction = self.network(sample.x_t, sample.t, context)

        loss = torch.nn.functional.mse_loss(prediction, sample.target)

        if self.aux_loss_weight > 0:
            pred_x1 = self.path.get_x1_from_prediction(sample.x_t, sample.t, prediction)
            aux_loss = torch.nn.functional.mse_loss(pred_x1, x_1)
            loss = loss + self.aux_loss_weight * aux_loss

        self.train_loss(loss)
        self.log("train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        context = self._build_context(batch)
        x_1 = self._get_target(batch)
        x_0 = torch.randn_like(x_1)

        sample = self.path.sample(x_0=x_0, x_1=x_1)
        prediction = self.network(sample.x_t, sample.t, context)

        loss = torch.nn.functional.mse_loss(prediction, sample.target)

        if self.aux_loss_weight > 0:
            pred_x1 = self.path.get_x1_from_prediction(sample.x_t, sample.t, prediction)
            aux_loss = torch.nn.functional.mse_loss(pred_x1, x_1)
            loss = loss + self.aux_loss_weight * aux_loss

        self.val_loss(loss)
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.val_loss_best(self.val_loss.compute())
        self.log("val_loss_best", self.val_loss_best, prog_bar=True)

    @torch.no_grad()
    def sample_from_model(
        self,
        context: torch.Tensor,
        samples: int = 1,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        batch = context.shape[0]
        context_rep = context.repeat_interleave(samples, dim=0)
        x_init = torch.randn(
            batch * samples,
            self.target_channels,
            self.crop_size,
            self.crop_size,
            device=context.device,
            dtype=context.dtype,
        )

        def model_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.network(x, t, context_rep)

        return self.sampler.sample(
            model_fn,
            x_init,
            num_steps=num_steps,
            device=context.device,
        )

    def configure_optimizers(self):
        optimizer = self.optimizer_partial(params=self.parameters())
        if self.decay_steps <= 0 and self.warmup_steps <= 0:
            return optimizer

        base_lr = optimizer.param_groups[0].get("lr", 1.0)
        end_ratio = self.end_lr / base_lr if base_lr > 0 else 1.0

        def lr_lambda(step: int) -> float:
            if self.warmup_steps > 0 and step < self.warmup_steps:
                return max(float(step + 1) / float(self.warmup_steps), 1e-6)
            if self.decay_steps <= 0:
                return 1.0
            progress = min(float(step - self.warmup_steps) / float(self.decay_steps), 1.0)
            cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)))
            return float(end_ratio + (1.0 - end_ratio) * cosine)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
