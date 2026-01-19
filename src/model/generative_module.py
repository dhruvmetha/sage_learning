import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics import MeanMetric, MinMetric
from typing import Optional, Dict, Any
import math
import json
import numpy as np
import cv2

# UPDATED IMPORTS to match your structure
from .networks.vector_denoiser import VectorDenoiserBackbone
from .common import BasePath, BaseSampler
from .base import BasePath as DiffusionBasePath, BaseSampler as DiffusionBaseSampler

class UnifiedGenerativeModule(pl.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        path: BasePath, 
        sampler: BaseSampler,
        optimizer: Any,
        context_channels: int = 5,
        vector_dim: int = 3,
        use_local: bool = True,
        pose_stats_file: Optional[str] = None,
        norm_mode: str = "mean_std",  # "max_abs" or "mean_std"
        overfit_mode: bool = False, # If True, compute stats per batch
        crop_size_meters: float = 2.0,
    ):
        super().__init__()
        self.network = network
        self.path = path
        self.sampler = sampler
        self.optimizer_partial = optimizer
        self.crop_size_meters = crop_size_meters
        self.context_channels = context_channels
        self.vector_dim = vector_dim
        self.use_local = use_local
        self.norm_mode = norm_mode
        self.overfit_mode = overfit_mode

        self.register_buffer('xy_norm', torch.tensor(1.0))
        self.register_buffer('theta_norm', torch.tensor(math.pi))
        self.register_buffer('mean', torch.zeros(3))
        self.register_buffer('std', torch.ones(3))

        # 2. Load Global Stats (only if not overfitting)
        if pose_stats_file is not None and not overfit_mode:
            stats = self._load_pose_stats(pose_stats_file)
            self.norm_mode = stats.get('mode', self.norm_mode)

            if self.norm_mode == "max_abs":
                self.xy_norm.fill_(stats['xy_norm'])
                self.theta_norm.fill_(stats['theta_norm'])
            else:
                self.mean.copy_(torch.tensor(stats['mean']))
                self.std.copy_(torch.tensor(stats['std']))

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_loss_best = MinMetric()
        self.save_hyperparameters(ignore=["network", "path", "sampler", "optimizer"])
        self._sampler_time_scaling_synced = False

    def _sync_sampler_time_scaling(self) -> None:
        """Ensure sampler timestep scaling matches how the path trained the network.

        Vector diffusion training via `VectorHFDiffusionPath` always feeds normalized
        timesteps `t ∈ [0, 1]` into the network. However, `HFDiffusionSampler` defaults
        to passing integer timesteps unless `normalize_t=True`. If these differ, W&B
        validation samples (and any manual sampling via `sample_pose`) will look wrong.
        """
        if self._sampler_time_scaling_synced:
            return
        self._sampler_time_scaling_synced = True

        desired_normalize_t = None

        # VectorHFDiffusionPath always uses normalized t ∈ [0, 1].
        try:
            from .paths.vector_hf_diffusion_path import VectorHFDiffusionPath

            if isinstance(self.path, VectorHFDiffusionPath):
                desired_normalize_t = True
        except Exception:
            pass

        # Generic HF diffusion path can be configured either way.
        if desired_normalize_t is None and hasattr(self.path, "normalize_t"):
            desired_normalize_t = bool(getattr(self.path, "normalize_t"))

        if desired_normalize_t is None or not hasattr(self.sampler, "normalize_t"):
            return

        current = bool(getattr(self.sampler, "normalize_t"))
        if current == desired_normalize_t:
            return

        setattr(self.sampler, "normalize_t", desired_normalize_t)
        if getattr(getattr(self, "trainer", None), "is_global_zero", True):
            print(
                f"[UnifiedGenerativeModule] Set sampler.normalize_t={desired_normalize_t} "
                "to match training-time path timestep scaling."
            )

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.network(x, t, context)

    @staticmethod
    def _load_pose_stats(stats_file: str) -> Dict[str, float]:
        with open(stats_file, 'r') as f:
            return json.load(f)

    def _build_context(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.use_local:
            parts = [
                batch['static'], batch['movable'], batch['target_object'],
                batch['robot_region'], batch['goal_sample_region']
            ]
        else:
            parts = [
                batch['robot'], batch['goal'], batch['movable'],
                batch['static'], batch['target_object']
            ]
        if 'coord_grid' in batch:
            parts.append(batch['coord_grid'])
        return torch.cat(parts, dim=1)

    def _normalize_pose(self, pose: torch.Tensor) -> torch.Tensor:
        if self.overfit_mode:
            # We use .detach() to ensure we don't try to backprop through the stats themselves
            txm = pose[:, 0:1].mean().detach()
            tym = pose[:, 1:2].mean().detach()
            
            # Robust std computation
            std_val = pose[:, 0:2].std()
            if torch.isnan(std_val) or std_val < 1e-6:
                ts = torch.tensor(1.0, device=pose.device)
            else:
                ts = std_val.detach()

            thm = 0.0
            ths = math.pi
            
            # Store in buffers so _denormalize_pose can find them later
            self.mean[0] = txm
            self.mean[1] = tym
            self.mean[2] = 0.0
            
            self.std[0] = ts
            self.std[1] = ts
            self.std[2] = math.pi
            
            return (pose - self.mean) / self.std

        if self.norm_mode == "max_abs":
            # Scale to [-1, 1] using global max
            norm_scale = torch.tensor([self.xy_norm, self.xy_norm, self.theta_norm], device=pose.device)
            return torch.clamp(pose / norm_scale, -1, 1)
        else:
            # Standardize using global Mean/Std
            return (pose - self.mean) / self.std

    def _denormalize_pose(self, pose: torch.Tensor) -> torch.Tensor:
        """Inverse of normalization. Works for all modes."""
        if self.norm_mode == "max_abs" and not self.overfit_mode:
            norm_scale = torch.tensor([self.xy_norm, self.xy_norm, self.theta_norm], device=pose.device)
            return pose * norm_scale
        else:
            # Works for both global Mean-Std and Overfit Batch-Std
            return (pose * self.std) + self.mean

    def training_step(self, batch, batch_idx):
        context = self._build_context(batch)
        if batch['target_goal'].dim() > 2:
            raise ValueError("Target seems to be an image! The model expects a vector.")
        x_1 = self._normalize_pose(batch['target_goal']) 
        
        if self.overfit_mode:
            # DETERMINISTIC overfitting: Fixed noise, ALL timesteps in EVERY batch
            # We expand the batch to include multiple timesteps per sample
            # This prevents catastrophic forgetting between epochs
            B = x_1.shape[0]
            num_timesteps = 10  # Number of timesteps per sample
            
            # Generate fixed noise (same every iteration)
            generator = torch.Generator(device=x_1.device)
            generator.manual_seed(42)
            x_0 = torch.randn(x_1.shape, generator=generator, device=x_1.device, dtype=x_1.dtype)
            
            # Expand batch: each sample gets num_timesteps copies at different t values
            # x_1: (B, 3) -> (B * num_timesteps, 3)
            # context: (B, C, H, W) -> (B * num_timesteps, C, H, W)
            x_1_expanded = x_1.repeat_interleave(num_timesteps, dim=0)  # (B*T, 3)
            x_0_expanded = x_0.repeat_interleave(num_timesteps, dim=0)  # (B*T, 3)
            context = context.repeat_interleave(num_timesteps, dim=0)   # (B*T, C, H, W)
            
            # Create timesteps: each sample sees [0.05, 0.15, ..., 0.95]
            t_values = torch.linspace(0.05, 0.95, num_timesteps, device=x_1.device)
            t = t_values.repeat(B)  # (B*T,) - pattern: [0.05,0.15,...,0.95, 0.05,0.15,...,0.95, ...]
            
            t_expand = t.view(-1, 1)
            x_t = (1 - t_expand) * x_0_expanded + t_expand * x_1_expanded
            target_v = x_1_expanded - x_0_expanded
            
            # Import TrainingState
            from .common import TrainingState
            sample = TrainingState(x_t=x_t, t=t, target=target_v)
            
            # Save fixed noise for sampling consistency (original batch size)
            if not hasattr(self, '_fixed_x0'):
                self._fixed_x0 = x_0.clone()
                self._fixed_x1 = x_1.clone()
                self._fixed_target_v = (x_1 - x_0).clone()
        else:
            x_0 = torch.randn_like(x_1)
            # Get training state (x_t, t, target) from Path
            sample = self.path.compute_loss_samples(x_0=x_0, x_1=x_1)
        
        # Predict
        prediction = self.network(sample.x_t, sample.t, context)

        pred_xy = prediction[:, 0:2]
        target_xy = sample.target[:, 0:2]
        pred_theta_norm = prediction[:, 2]
        target_theta_norm = sample.target[:, 2]
        loss_xy = torch.nn.functional.mse_loss(pred_xy, target_xy)

        # For Flow Matching, the target is a velocity in Euclidean space.
        # Even for angles, we are transporting in the tangent space (or Euclidean embedding).
        # Cosine loss is incorrect for velocity matching because velocity magnitude matters.
        # We use MSE for theta velocity as well.
        loss_theta = torch.nn.functional.mse_loss(pred_theta_norm, target_theta_norm)

        loss = loss_xy + (2.0 * loss_theta)

        # Log step-level loss for debugging (shows variance in generative training)
        self.log("train_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.train_loss(loss)
        self.log("train_loss_epoch", self.train_loss, on_step=False, on_epoch=True, prog_bar=False)

        # saving to visualize training examples when overfitting
        # Note: In overfit_mode, we save the ORIGINAL batch data (not expanded)
        if self.overfit_mode and (not hasattr(self, "_train_context") or batch_idx == 0):
            # Re-build original context from batch (not the expanded one)
            original_context = self._build_context(batch)
            self._train_context = original_context.detach().clone()
            
            # Save the ORIGINAL normalized poses (before expansion)
            # x_1 here is the non-expanded version from _normalize_pose(batch['target_goal'])
            # We need to use _fixed_x1 which was saved from x_1 before expansion
            self._train_gt_poses = self._fixed_x1.detach().clone()
            
            # Save the normalization stats used for this batch (critical for correct denormalization later)
            self._train_norm_mean = self.mean.detach().clone()
            self._train_norm_std = self.std.detach().clone()
            
            if 'target_goal_mask' in batch:
                self._train_gt_images = batch['target_goal_mask'].detach().clone()
            else:
                self._train_gt_images = torch.zeros_like(batch['static'])
            if 'object_theta' in batch:
                self._train_object_theta = batch['object_theta'].detach().clone()
            else:
                self._train_object_theta = torch.zeros(batch['target_goal'].shape[0], device=batch['target_goal'].device)
        return loss

    def validation_step(self, batch, batch_idx):
        context = self._build_context(batch)
        x_1 = self._normalize_pose(batch['target_goal'])
        x_0 = torch.randn_like(x_1)
        
        sample = self.path.compute_loss_samples(x_0=x_0, x_1=x_1)
        prediction = self.network(sample.x_t, sample.t, context)
        pred_xy = prediction[:, 0:2]
        target_xy = sample.target[:, 0:2]
        pred_theta_norm = prediction[:, 2]
        target_theta_norm = sample.target[:, 2]
        loss_xy = torch.nn.functional.mse_loss(pred_xy, target_xy)

        # For Flow Matching, the target is a velocity in Euclidean space.
        # Even for angles, we are transporting in the tangent space (or Euclidean embedding).
        # Cosine loss is incorrect for velocity matching because velocity magnitude matters.
        # We use MSE for theta velocity as well.
        loss_theta = torch.nn.functional.mse_loss(pred_theta_norm, target_theta_norm)

        loss = loss_xy + (2.0 * loss_theta)
        
        self.val_loss(loss)
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        if batch_idx == 0:
            self._validation_context = context
            self._validation_gt_poses = x_1

            if 'target_goal_mask' in batch:
                self._validation_gt_images = batch['target_goal_mask']
            else:
                self._validation_gt_images = torch.zeros_like(batch['static'])
            
            # Store object theta for visualization (needed to convert object-frame delta to world frame)
            if 'object_theta' in batch:
                self._validation_object_theta = batch['object_theta']
            else:
                self._validation_object_theta = torch.zeros(batch['target_goal'].shape[0], device=batch['target_goal'].device)
        return loss

    def on_validation_epoch_end(self):
        if not self.overfit_mode:
            self.val_loss_best(self.val_loss.compute())
            self.log("val_loss_best", self.val_loss_best, prog_bar=True)
            if hasattr(self, '_validation_context'):
                self._visualize_predictions()

    def on_train_epoch_end(self):
        if self.overfit_mode and hasattr(self, '_train_context'):
            self._visualize_predictions()

    def _get_transformed_mask_cv2(self, current_mask_tensor, pose_norm, object_theta_rad):
            """
            Uses the internal _denormalize_pose to ensure pixel-perfect 
            alignment regardless of normalization mode.
            """
            mask_np = (current_mask_tensor.cpu().numpy() * 255).astype(np.uint8)
            H, W = mask_np.shape
            pixels_per_meter = W / self.crop_size_meters
            
            # 1. DENORMALIZE using the shared model logic
            # Un-normalize back to raw meters/radians
            real_pose = self._denormalize_pose(pose_norm.unsqueeze(0)).squeeze(0)
            
            dx_obj_meters = real_pose[0].item()
            dy_obj_meters = real_pose[1].item()
            dtheta_change = real_pose[2].item()

            # 2. Find current geometry
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return torch.zeros_like(current_mask_tensor)
            cnt = max(contours, key=cv2.contourArea).squeeze()
            M = cv2.moments(cnt)
            if M['m00'] == 0: return torch.zeros_like(current_mask_tensor)
            curr_obj_cx, curr_obj_cy = M['m10'] / M['m00'], M['m01'] / M['m00']
            
            # 3. Rotate delta to world/image frame
            c_theta, s_theta = np.cos(object_theta_rad), np.sin(object_theta_rad)
            dx_world = dx_obj_meters * c_theta - dy_obj_meters * s_theta
            dy_world = dx_obj_meters * s_theta + dy_obj_meters * c_theta
            
            # 4. Translation and Rotation
            new_obj_cx = curr_obj_cx + (dx_world * pixels_per_meter)
            new_obj_cy = curr_obj_cy + (dy_world * pixels_per_meter) # No Y-flip per data gen
            
            pts_centered = (cnt - np.array([curr_obj_cx, curr_obj_cy])).astype(np.float32)
            cr, sr = np.cos(dtheta_change), np.sin(dtheta_change)
            R = np.array(((cr, -sr), (sr, cr)))
            
            pts_final = (R @ pts_centered.T).T + np.array([new_obj_cx, new_obj_cy])
            
            new_mask = np.zeros_like(mask_np)
            cv2.fillPoly(new_mask, [pts_final.astype(np.int32)], 255)
            return torch.from_numpy(new_mask).float().to(self.device) / 255.0

    def _visualize_predictions(self):
        """
        Visualizes the validation samples with specific color encodings.
        """
        if not hasattr(self.logger, 'experiment'): return
        import wandb
        import torchvision
        import cv2
        import numpy as np
        
        # --- HELPER: Display Normalization ---
        def _to_display(x: torch.Tensor) -> torch.Tensor:
            """Convert from [-1, 1] to [0, 1] and ensure 2D (H, W)."""
            return torch.clamp((x + 1) / 2, 0, 1).squeeze()

        if self.overfit_mode:
            context = self._train_context
            gt_images = self._train_gt_images
            gt_poses = self._train_gt_poses
            object_thetas = self._train_object_theta
            # Restore the normalization stats that were used when these poses were normalized
            saved_mean, saved_std = self.mean.clone(), self.std.clone()
            self.mean.copy_(self._train_norm_mean)
            self.std.copy_(self._train_norm_std)
        else:
            context = self._validation_context
            gt_images = self._validation_gt_images
            gt_poses = self._validation_gt_poses
            object_thetas = self._validation_object_theta

        num_examples = min(8, context.size(0))
        image_size = context.shape[-1]
        
        log_dict = {}

        for i in range(num_examples):
            # Extract Data
            ctx = context[i]             # (5, H, W)
            gt_img_mask = gt_images[i]   # (1, H, W) or (H,W)
            gt_pose_delta = gt_poses[i]  # (3,)
            obj_theta = object_thetas[i].item()  # scalar in radians

            # --- CORRECT CHANNEL MAPPING ---
            # Based on _build_context: 
            # [0]=Static, [1]=Movable, [2]=Target, [3]=RobotRegion, [4]=GoalRegion
            static_walls = _to_display(ctx[0:1])
            # movable = _to_display(ctx[1:2]) # Not used in viz
            target_obj_curr = _to_display(ctx[2:3])
            robot_region = _to_display(ctx[3:4])
            goal_sample_region = _to_display(ctx[4:5])
            
            # Ensure GT Image Mask is 2D [0,1]
            gt_img_mask_disp = _to_display(gt_img_mask.unsqueeze(0) if gt_img_mask.dim()==2 else gt_img_mask)

            # Generate "Ghost" Masks using Pose Deltas
            # Pass object_theta to correctly transform object-frame delta to world frame
            gt_pose_mask = self._get_transformed_mask_cv2(target_obj_curr, gt_pose_delta, obj_theta)
            
            # DEBUG: Log denormalized pose values to verify correctness
            if i == 0 and self.current_epoch % 20 == 0:
                denorm_pose = self._denormalize_pose(gt_pose_delta.unsqueeze(0)).squeeze(0)
                print(f"[DEBUG Viz] Epoch {self.current_epoch}, Sample 0:")
                print(f"  Normalized pose: {gt_pose_delta.cpu().numpy()}")
                print(f"  Denormalized pose: {denorm_pose.cpu().numpy()}")
                print(f"  Mean: {self.mean.cpu().numpy()}, Std: {self.std.cpu().numpy()}")
            
            # Generate Predictions
            with torch.no_grad():
                if self.overfit_mode and hasattr(self, '_fixed_x0'):
                    # DIRECT velocity test: x_1 = x_0 + v
                    # This directly tests if the model learned the correct velocity
                    # without relying on ODE integration across unseen timesteps
                    x_0_i = self._fixed_x0[i:i+1]  # (1, 3)
                    # Query model at t=0.5 (middle of training range)
                    t_test = torch.tensor([0.5], device=ctx.device)
                    x_t_test = 0.5 * x_0_i + 0.5 * gt_pose_delta.unsqueeze(0)
                    pred_v = self.network(x_t_test, t_test, ctx.unsqueeze(0))
                    # Reconstruct x_1 from x_0 + predicted velocity
                    pred_x1 = x_0_i + pred_v  # (1, 3)
                    # Repeat for 4 samples (they should all be identical in overfit mode)
                    p_list = pred_x1.repeat(4, 1)
                else:
                    # Normal ODE sampling for non-overfit mode
                    p_list = self.sample_pose(ctx.unsqueeze(0), num_samples=4, denormalize=False, sample_idx=i)

            # --- PANEL 1: LOCAL SCENE ---
            # Req: RobotRegion(Red), GoalRegion(Green), Target(Cyan), Walls(Black)
            img1 = torch.zeros(3, image_size, image_size, device=ctx.device)
            img1[0] = robot_region
            # Green Channel: Goal Region + part of Cyan Target
            img1[1] = torch.clamp(goal_sample_region + target_obj_curr, 0, 1) 
            # Blue Channel: Part of Cyan Target
            img1[2] = target_obj_curr                                 

            # --- PANEL 2: GT ANALYSIS ---
            # Req: Target(Cyan), GT_Image(Orange), GT_Pose(Blue), Walls(Red)
            # Orange = Red(1.0) + Green(0.5)
            img2 = torch.zeros(3, image_size, image_size, device=ctx.device)
            
            # Red Channel: Walls + GT_Image (Full Red)
            img2[0] = torch.clamp(static_walls + gt_img_mask_disp, 0, 1)
            
            # Green Channel: Target (Cyan) + GT_Image (Half Green for Orange)
            img2[1] = torch.clamp(target_obj_curr + (0.5 * gt_img_mask_disp), 0, 1)
            
            # Blue Channel: Target (Cyan) + GT_Pose
            img2[2] = torch.clamp(target_obj_curr + gt_pose_mask, 0, 1)

            # --- PANELS 3-6: PREDICTIONS ---
            # Req: Walls(Red), GT_Pose(Blue), Pred_Pose(Green)
            preds_imgs = []
            for j in range(4):
                pred_pose_mask = self._get_transformed_mask_cv2(target_obj_curr, p_list[j], obj_theta)
                
                p_img = torch.zeros(3, image_size, image_size, device=ctx.device)
                p_img[0] = static_walls     # Red
                p_img[1] = pred_pose_mask   # Green
                p_img[2] = gt_pose_mask     # Blue
                
                preds_imgs.append(p_img)

            # --- STITCH & LOG ---
            row_tensors = [img1, img2] + preds_imgs
            row_stack = torch.stack(row_tensors)
            
            # Normalize=False because we manually constructed [0,1] tensors
            grid = torchvision.utils.make_grid(row_stack, nrow=6, normalize=False, padding=2)

            grid_np = grid.cpu().permute(1, 2, 0).numpy()
            caption = f"Ex {i} | Scene | GT Analysis (Org=Img, Blu=Pose) | Preds (Grn) vs GT (Blu)"
            log_dict[f'val_sample_{i}'] = wandb.Image(grid_np, caption=caption)

        # Restore original stats if we swapped them for overfit visualization
        if self.overfit_mode:
            self.mean.copy_(saved_mean)
            self.std.copy_(saved_std)

        self.logger.experiment.log(log_dict)

    def sample_pose(self, context, num_samples=1, num_steps=20, denormalize=True, show_progress=False, sample_idx=None):
        """
        Sample poses using the learned velocity field.
        
        Args:
            context: (B, C, H, W) context images
            num_samples: number of samples per context
            num_steps: ODE integration steps
            denormalize: whether to denormalize output
            show_progress: show tqdm progress bar
            sample_idx: if in overfit_mode, use fixed noise for this sample index
        """
        self._sync_sampler_time_scaling()
        B = context.shape[0]
        context_repeated = context.repeat_interleave(num_samples, dim=0)
        total_samples = B * num_samples
        
        if self.overfit_mode and hasattr(self, '_fixed_x0') and sample_idx is not None:
            # Use the SAME fixed noise as training for deterministic comparison
            x_init = self._fixed_x0[sample_idx:sample_idx+1].repeat(num_samples, 1)
        elif self.overfit_mode:
            # Use consistent fixed noise even without sample_idx
            generator = torch.Generator(device=context.device)
            generator.manual_seed(42)
            x_init = torch.randn(total_samples, self.vector_dim, generator=generator, 
                                device=context.device, dtype=context.dtype)
        else:
            x_init = torch.randn(total_samples, self.vector_dim, device=context.device)
            
        def model_fn(x, t):
            return self.network(x, t, context_repeated)
        samples = self.sampler.sample(model_fn, x_init, num_steps, show_progress, device=context.device)
        if denormalize: return self._denormalize_pose(samples)
        return samples

    def configure_optimizers(self):
        return self.optimizer_partial(params=self.parameters())


class GenerativeModule(pl.LightningModule):
    """Image-to-image diffusion module (global or local masks)."""

    def __init__(
        self,
        network: nn.Module,
        path: DiffusionBasePath,
        sampler: DiffusionBaseSampler,
        optimizer: Any,
        context_channels: int = 5,
        target_channels: int = 1,
        use_local: bool = True,
        aux_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.network = network
        self.path = path
        self.sampler = sampler
        self.optimizer_partial = optimizer
        self.context_channels = context_channels
        self.target_channels = target_channels
        self.use_local = use_local
        self.aux_loss_weight = aux_loss_weight

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_loss_best = MinMetric()
        self.save_hyperparameters(ignore=["network", "path", "sampler", "optimizer"])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.network(x, t)

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
        if "coord_grid" in batch:
            parts.append(batch["coord_grid"])
        return torch.cat(parts, dim=1)

    def _get_target(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        target = batch["target_goal_mask"]
        if target.dim() == 3:
            target = target.unsqueeze(1)
        return target

    def training_step(self, batch, batch_idx):
        context = self._build_context(batch)
        x_1 = self._get_target(batch)
        x_0 = torch.randn_like(x_1)

        sample = self.path.sample(x_0=x_0, x_1=x_1)
        model_in = torch.cat([context, sample.x_t], dim=1)
        prediction = self.network(model_in, sample.t)

        loss = torch.nn.functional.mse_loss(prediction, sample.target)

        if self.aux_loss_weight > 0:
            pred_x1 = self.path.get_x1_from_prediction(sample.x_t, sample.t, prediction)
            aux_loss = torch.nn.functional.mse_loss(pred_x1, x_1)
            loss = loss + self.aux_loss_weight * aux_loss

        self.train_loss(loss)
        self.log("train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        context = self._build_context(batch)
        x_1 = self._get_target(batch)
        x_0 = torch.randn_like(x_1)

        sample = self.path.sample(x_0=x_0, x_1=x_1)
        model_in = torch.cat([context, sample.x_t], dim=1)
        prediction = self.network(model_in, sample.t)

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
            context.shape[-1],
            context.shape[-1],
            device=context.device,
            dtype=context.dtype,
        )

        def model_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            model_in = torch.cat([context_rep, x], dim=1)
            return self.network(model_in, t)

        return self.sampler.sample(
            model_fn,
            x_init,
            num_steps=num_steps,
            device=context.device,
        )

    def configure_optimizers(self):
        return self.optimizer_partial(params=self.parameters())
