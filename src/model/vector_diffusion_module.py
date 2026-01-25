from __future__ import annotations

import torch
import torch.nn.functional as F

from .generative_module import UnifiedGenerativeModule


class VectorDiffusionModule(UnifiedGenerativeModule):
    """
    Diffusion training for SE(2) vector targets.

    Uses standard MSE on diffusion targets (noise/velocity/sample) without
    flow-matching-specific weighting.
    """

    def training_step(self, batch, batch_idx):
        context = self._build_context(batch)
        if batch["target_goal"].dim() > 2:
            raise ValueError("Target seems to be an image! The model expects a vector.")
        x_1 = self._normalize_pose(batch["target_goal"])
        x_0 = torch.randn_like(x_1)

        sample = self.path.compute_loss_samples(x_0=x_0, x_1=x_1)
        prediction = self.network(sample.x_t, sample.t, context)

        loss = F.mse_loss(prediction, sample.target)

        self.log("train_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.train_loss(loss)
        self.log("train_loss_epoch", self.train_loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        context = self._build_context(batch)
        x_1 = self._normalize_pose(batch["target_goal"])
        x_0 = torch.randn_like(x_1)

        sample = self.path.compute_loss_samples(x_0=x_0, x_1=x_1)
        prediction = self.network(sample.x_t, sample.t, context)

        loss = F.mse_loss(prediction, sample.target)

        self.val_loss(loss)
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            self._validation_context = context
            self._validation_gt_poses = x_1

            if "target_goal_mask" in batch:
                self._validation_gt_images = batch["target_goal_mask"]
            else:
                self._validation_gt_images = torch.zeros_like(batch["static"])

            if "object_theta" in batch:
                self._validation_object_theta = batch["object_theta"]
            else:
                self._validation_object_theta = torch.zeros(
                    batch["target_goal"].shape[0], device=batch["target_goal"].device
                )

        return loss
