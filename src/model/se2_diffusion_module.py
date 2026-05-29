"""SE(2) delta diffusion model.

Predicts (Δx, Δy, Δθ) — the push primitive in world frame relative
to the local_tight crop's center (= the planner's pre-pose for this push step).
Plain Euclidean 3-vec: our data stays well within [-π, π] for Δθ, no wrap.

Architecture:
    local_tight context (5×64×64) ──► CNN encoder ──► feature F ∈ R^{256}
                                                            │
                                                            │  FiLM(γ, β)
                                                            ▼
    noisy x_t (4-d) ─► [concat with t-embed (sinusoidal 128-d)] ─► MLP denoiser
                                                            │
                                                            ▼
                                                     ε_pred (4-d)

Training: standard DDPM with iDDPM cosine β-schedule, T=100, ε-prediction, MSE loss.
Inference: DDIM, 10 steps. To get N candidate pushes per scene, sample N independent
N(0, I) initializations and run inference in parallel — captures multi-modal goals.

References:
  - Chi et al. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (RSS 2023)
  - Nichol & Dhariwal "Improved DDPM" (ICML 2021) — cosine schedule
  - Zhou et al. "On the Continuity of Rotation Representations" (CVPR 2019) — (cos, sin) for SO(2)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import cv2
import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanMetric, MinMetric


# =============================================================================
# Noise schedule
# =============================================================================

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """iDDPM cosine schedule (Nichol & Dhariwal 2021).

    Returns betas of shape (T,) clipped to (1e-6, 0.999).
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bars = f / f[0]
    betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
    return betas.clamp(1e-6, 0.999).to(torch.float32)


class DDPMSchedule:
    """Precomputed schedule tensors. Re-registered as buffers in the LightningModule
    so `.to(device)` moves them with the model.
    """
    def __init__(self, T: int = 100):
        betas = cosine_beta_schedule(T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.T = T
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)


# =============================================================================
# Context encoder (5×64×64 → 256-d feature)
# =============================================================================

class ContextEncoder(nn.Module):
    """Small CNN that pools the binary mask channels into a feature vector.

    Five 3×3 conv blocks with stride-2 downsampling: 64→32→16→8→4 spatial,
    32→64→128→256 channels. AvgPool to a single token, project to feat_dim.
    """

    def __init__(self, in_channels: int = 5, feat_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            self._block(32, 64),    # 64→32
            self._block(64, 128),   # 32→16
            self._block(128, 256),  # 16→8
            self._block(256, 256),  # 8→4
        )
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.blocks(h)
        return self.proj(h)


# =============================================================================
# Denoiser MLP with FiLM conditioning
# =============================================================================

class SinusoidalTimeEmbed(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps (Vaswani-style)."""

    def __init__(self, dim: int = 128):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FiLMBlock(nn.Module):
    """Linear → GELU → FiLM(γ, β) from context features."""

    def __init__(self, dim: int, ctx_dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.film = nn.Linear(ctx_dim, 2 * dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        h = self.act(self.linear(x))
        gamma, beta = self.film(ctx).chunk(2, dim=-1)
        return h * (1 + gamma) + beta


class SE2Denoiser(nn.Module):
    """MLP denoiser that predicts ε given (x_t, t, context_features)."""

    def __init__(self, target_dim: int = 3, hidden_dim: int = 256, ctx_dim: int = 256,
                 t_emb_dim: int = 128, n_blocks: int = 4):
        super().__init__()
        self.t_emb = SinusoidalTimeEmbed(t_emb_dim)
        self.in_proj = nn.Linear(target_dim + t_emb_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [FiLMBlock(hidden_dim, ctx_dim) for _ in range(n_blocks)]
        )
        self.out_proj = nn.Linear(hidden_dim, target_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_emb(t)                              # [B, t_emb_dim]
        h = self.in_proj(torch.cat([x_t, t_emb], dim=-1))  # [B, hidden]
        for block in self.blocks:
            h = block(h, ctx) + h                          # residual + FiLM
        return self.out_proj(h)                            # [B, target_dim]


# =============================================================================
# Lightning module
# =============================================================================

class SE2DiffusionModule(pl.LightningModule):
    """Training + inference for SE(2) delta diffusion.

    Batch contract (from SE2CroppedDataModule):
        context: (B, 5, H, W)  float in [-1, 1]
        target:  (B, 4)        ground-truth (Δx, Δy, cos Δθ, sin Δθ)

    Outputs at sample time: (B, 3) decoded back to (Δx, Δy, Δθ).
    """

    def __init__(
        self,
        context_size: int = 64,
        feat_dim: int = 256,
        hidden_dim: int = 256,
        n_denoiser_blocks: int = 4,
        T: int = 100,
        ddim_steps: int = 10,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = ContextEncoder(in_channels=5, feat_dim=feat_dim)
        self.denoiser = SE2Denoiser(target_dim=3, hidden_dim=hidden_dim,
                                    ctx_dim=feat_dim, n_blocks=n_denoiser_blocks)

        sched = DDPMSchedule(T=T)
        self.register_buffer('betas', sched.betas)
        self.register_buffer('alphas', sched.alphas)
        self.register_buffer('alpha_bars', sched.alpha_bars)
        self.register_buffer('sqrt_alpha_bars', sched.sqrt_alpha_bars)
        self.register_buffer('sqrt_one_minus_alpha_bars', sched.sqrt_one_minus_alpha_bars)
        self.T = T
        self.ddim_steps = ddim_steps

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_loss_best = MinMetric()
        # Decoded-pose validation metrics (in real units)
        self.val_xy_mae_m = MeanMetric()
        self.val_th_mae_deg = MeanMetric()

        # Stash one val batch each epoch for image visualization (matches
        # the cropped mask-DiT val sample layout: 8 scenes × 7-panel row).
        self._viz_batch: Optional[Dict[str, torch.Tensor]] = None
        self.viz_n_scenes = 8
        self.viz_n_samples = 4
        self.viz_crop_size_meters = 0.5
        # output_size = context_size (model's native resolution, same as mask DiT)
        self.viz_output_size = context_size

    # ----- visualization helpers -----

    @staticmethod
    def _render_target_mask(delta: np.ndarray, obj_size_xy: np.ndarray,
                            crop_size_meters: float, output_size: int) -> np.ndarray:
        """Render a binary mask of the object at its predicted target pose,
        in the local_tight crop's pixel coordinates.

        Args:
            delta: (3,) (Δx, Δy, Δθ) world-frame, relative to pre_pose (= crop center)
            obj_size_xy: (2,) (sx, sy) physical object size in meters
            crop_size_meters: side length of the crop (= 0.5 m for local_tight)
            output_size: pixel side length (e.g., 64 or 224)

        Returns:
            (output_size, output_size) float32 binary mask
        """
        mask = np.zeros((output_size, output_size), dtype=np.float32)
        dx, dy, dth = float(delta[0]), float(delta[1]), float(delta[2])
        sx, sy = float(obj_size_xy[0]), float(obj_size_xy[1])
        scale = output_size / crop_size_meters  # pixels per meter

        # Object target center in pixel space (image axes: row = y, col = x).
        # The crop is world-axis-aligned around pre_pose (= crop center).
        cx = output_size / 2 + dx * scale
        cy = output_size / 2 + dy * scale
        w_px = sx * scale
        h_px = sy * scale
        # Δθ is the rotation FROM pre_pose.θ; the object's absolute orientation
        # in the world frame would be pre_pose.θ + Δθ, but the crop's axes are
        # not aligned to the object's body frame — they're world-axis-aligned.
        # So the rectangle's draw angle uses (pre_pose.θ + Δθ). Caller passes
        # delta as (Δx, Δy, target_θ) when they want the absolute angle, or
        # (Δx, Δy, Δθ) plus a separate pre.θ for body-relative.
        # NOTE: this helper expects `delta[2]` to be the angle to draw at.
        rect = ((cx, cy), (w_px, h_px), np.degrees(dth))
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillPoly(mask, [box], 1.0)
        return mask

    # _render_target_mask (above) is the only mask-render helper we need;
    # _build_val_grid below assembles the per-scene 7-panel row to match the
    # existing cropped mask-DiT val sample layout exactly.

    # ----- training -----

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        x_0 = batch['target']                       # [B, 4]
        ctx = self.encoder(batch['context'])         # [B, feat_dim]

        B = x_0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x_0.device)
        eps = torch.randn_like(x_0)
        sab = self.sqrt_alpha_bars[t][:, None]
        somab = self.sqrt_one_minus_alpha_bars[t][:, None]
        x_t = sab * x_0 + somab * eps

        eps_pred = self.denoiser(x_t, t, ctx)
        loss = F.mse_loss(eps_pred, eps)

        self.train_loss(loss)
        self.log('train/loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ----- validation -----

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        # Stash first val batch each epoch for image viz on epoch end
        if batch_idx == 0:
            self._viz_batch = {k: v.detach() for k, v in batch.items()}

        x_0 = batch['target']
        ctx = self.encoder(batch['context'])
        B = x_0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x_0.device)
        eps = torch.randn_like(x_0)
        sab = self.sqrt_alpha_bars[t][:, None]
        somab = self.sqrt_one_minus_alpha_bars[t][:, None]
        x_t = sab * x_0 + somab * eps
        eps_pred = self.denoiser(x_t, t, ctx)
        loss = F.mse_loss(eps_pred, eps)
        self.val_loss(loss)
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Sample one prediction per scene with DDIM, compare to gt in real units.
        if batch_idx < 4:
            with torch.no_grad():
                pred = self.sample(ctx, n_per_scene=1)        # [B, 1, 3]
            pred = pred.squeeze(1)
            gt_dx, gt_dy, gt_dth = x_0[:, 0], x_0[:, 1], x_0[:, 2]
            pr_dx, pr_dy, pr_dth = pred[:, 0], pred[:, 1], pred[:, 2]
            xy_err = torch.sqrt((pr_dx - gt_dx) ** 2 + (pr_dy - gt_dy) ** 2).mean()
            th_diff = pr_dth - gt_dth
            th_diff = torch.atan2(torch.sin(th_diff), torch.cos(th_diff))  # wrap for metric only
            th_err = th_diff.abs().mean() * (180.0 / math.pi)
            self.val_xy_mae_m(xy_err)
            self.val_th_mae_deg(th_err)
            self.log('val/xy_mae_m', self.val_xy_mae_m, on_epoch=True, prog_bar=True)
            self.log('val/th_mae_deg', self.val_th_mae_deg, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.val_loss_best.update(self.val_loss.compute())
        self.log('val/loss_best', self.val_loss_best.compute(), prog_bar=False)

        # Match cropped mask-DiT val sample layout: one wandb image per scene,
        # 7-panel row of [Scene | Reach | GT | Pred1..Pred4] at context_size.
        import sys
        if self._viz_batch is None:
            print(f"[se2 viz] skip — _viz_batch is None (epoch {self.current_epoch})", flush=True, file=sys.stderr)
            return
        if not hasattr(self.logger, 'experiment'):
            print(f"[se2 viz] skip — no logger.experiment", flush=True, file=sys.stderr)
            return
        try:
            import wandb
            import torchvision
        except ImportError:
            print("[se2 viz] skip — wandb/torchvision import failed", flush=True, file=sys.stderr)
            self._viz_batch = None
            return

        try:
            self._do_val_viz()
        except Exception as e:
            import traceback
            print(f"[se2 viz] FAILED at epoch {self.current_epoch}: {type(e).__name__}: {e}", flush=True, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        finally:
            self._viz_batch = None

    def _do_val_viz(self):
        import wandb
        import torchvision
        batch = self._viz_batch
        device = batch['context'].device
        ctx_size = batch['context'].shape[-1]
        n_scenes = min(self.viz_n_scenes, batch['context'].shape[0])
        n_samples = self.viz_n_samples

        def to_disp(x: torch.Tensor) -> torch.Tensor:
            return torch.clamp((x + 1) / 2, 0, 1)

        log_dict: Dict[str, Any] = {}
        for i in range(n_scenes):
            ctx_b = batch['context'][i:i+1]                  # [1, 5, H, W]
            pre_np = batch['pre_pose'][i].cpu().numpy()      # [3]
            obj_np = batch['obj_size'][i].cpu().numpy()      # [3]
            gt_np = batch['target'][i].cpu().numpy()         # [3]

            with torch.no_grad():
                ctx_feat = self.encoder(ctx_b)
                preds = self.sample(ctx_feat, n_per_scene=n_samples)   # [1, N, 3]
            preds_np = preds[0].cpu().numpy()                # [N, 3]

            # Per-channel displays (context_size resolution)
            static = to_disp(ctx_b[0, 0])
            movable = to_disp(ctx_b[0, 1])
            target_obj = to_disp(ctx_b[0, 2])
            robot_region = to_disp(ctx_b[0, 3])
            goal_region = to_disp(ctx_b[0, 4])

            # Scene: R=static, G=movable, B=target_obj
            img_scene = torch.stack([static, movable, target_obj]).unsqueeze(0)
            # Reach: R=robot_region, G=goal_region, B=target_obj
            img_reach = torch.stack([robot_region, goal_region, target_obj]).unsqueeze(0)

            # Mask renderer: the crop is world-axis-aligned around pre_pose,
            # so the rectangle's image-frame angle is pre.θ + Δθ (absolute world).
            def render_delta_to_mask(delta):
                draw = np.array([delta[0], delta[1], float(pre_np[2]) + float(delta[2])],
                                dtype=np.float32)
                return self._render_target_mask(draw, obj_np[:2],
                                                self.viz_crop_size_meters, ctx_size)

            gt_mask_np = render_delta_to_mask(gt_np)
            pred_masks_np = [render_delta_to_mask(p) for p in preds_np]

            def make_pred_img(mask_np):
                mask_t = torch.from_numpy(mask_np).to(device)
                # R=static, G=mask, B=target_obj
                return torch.stack([static, mask_t, target_obj]).unsqueeze(0)

            img_gt = make_pred_img(gt_mask_np)
            pred_imgs = [make_pred_img(m) for m in pred_masks_np]

            # 7-panel row [Scene | Reach | GT | Pred1..Pred4]
            row = torch.cat([img_scene, img_reach, img_gt] + pred_imgs, dim=0)
            grid = torchvision.utils.make_grid(row, nrow=row.shape[0],
                                               normalize=True, padding=2)
            grid_np = grid.cpu().permute(1, 2, 0).numpy()
            caption = (f"Epoch {self.current_epoch} | "
                       f"Scene | Reach | GT | Pred1-{n_samples} "
                       f"(SE(2) Δ rendered as object box at {ctx_size}x{ctx_size}, "
                       f"crop={self.viz_crop_size_meters} m)")
            log_dict[f'val_sample_{i+1}'] = wandb.Image(grid_np, caption=caption)

        import sys
        print(f"[se2 viz] logging {len(log_dict)} val_sample images at epoch {self.current_epoch}", flush=True, file=sys.stderr)
        self.logger.experiment.log(log_dict)
        # _viz_batch reset is done by the caller (on_validation_epoch_end's finally)

    # ----- inference -----

    @torch.no_grad()
    def sample(self, ctx: torch.Tensor, n_per_scene: int = 1) -> torch.Tensor:
        """DDIM sampling. Returns (B, N, 3) — N candidate samples per scene.

        Use n_per_scene > 1 to capture multi-modal goal distribution.
        """
        B, F_dim = ctx.shape
        device = ctx.device
        N = n_per_scene
        # Repeat context for N candidates: [B*N, F]
        ctx_rep = ctx.unsqueeze(1).expand(-1, N, -1).reshape(B * N, F_dim)

        x = torch.randn(B * N, 3, device=device)

        # DDIM timestep subsequence: T-1 → 0 in ddim_steps strides
        timesteps = torch.linspace(self.T - 1, 0, self.ddim_steps + 1, device=device).long()
        for i in range(self.ddim_steps):
            t = timesteps[i]
            t_prev = timesteps[i + 1]
            t_batch = torch.full((B * N,), int(t), device=device, dtype=torch.long)

            eps_pred = self.denoiser(x, t_batch, ctx_rep)

            alpha_bar = self.alpha_bars[t]
            alpha_bar_prev = self.alpha_bars[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

            x0_pred = (x - torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)
            # DDIM η = 0 (deterministic given x_T)
            x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev) * eps_pred

        return x.reshape(B, N, 3)

    @torch.no_grad()
    def predict_se3_world_delta(self, context: torch.Tensor, n_per_scene: int = 1) -> torch.Tensor:
        """Convenience: encode context + sample. Returns (B, N, 3).

        Δx, Δy in meters; Δθ in radians, all world-frame relative to pre_pose_a1.
        """
        ctx = self.encoder(context)
        return self.sample(ctx, n_per_scene=n_per_scene)

    # ----- optimizer -----

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        if self.hparams.warmup_steps > 0:
            def lr_lambda(step):
                if step < self.hparams.warmup_steps:
                    return float(step) / max(1, self.hparams.warmup_steps)
                return 1.0
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
            return {'optimizer': opt, 'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}
        return opt
