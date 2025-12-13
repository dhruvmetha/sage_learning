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

class UnifiedGenerativeModule(pl.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        path: BasePath, 
        sampler: BaseSampler,
        optimizer: Any,
        context_channels: int = 5,
        vector_dim: int = 3,
        use_local: bool = False,
        pose_stats_file: Optional[str] = None,
        xy_norm: float = 1.0,
        theta_norm: float = math.pi,
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

        if pose_stats_file is not None:
            stats = self._load_pose_stats(pose_stats_file)
            self.xy_norm = stats.get('xy_norm', xy_norm)
            self.theta_norm = stats.get('dtheta_norm', theta_norm)
            print(f"Loaded stats from {pose_stats_file}: xy={self.xy_norm}, theta={self.theta_norm}")
        else:
            self.xy_norm = xy_norm
            self.theta_norm = theta_norm

        self.register_buffer('_xy_norm', torch.tensor(self.xy_norm))
        self.register_buffer('_theta_norm', torch.tensor(self.theta_norm))
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_loss_best = MinMetric()
        self.save_hyperparameters(ignore=["network", "path", "sampler", "optimizer"])

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
        norm_scale = torch.tensor([self._xy_norm, self._xy_norm, self._theta_norm], device=pose.device)
        return torch.clamp(pose / norm_scale, -1, 1)

    def _denormalize_pose(self, pose: torch.Tensor) -> torch.Tensor:
        norm_scale = torch.tensor([self._xy_norm, self._xy_norm, self._theta_norm], device=pose.device)
        return pose * norm_scale

    def training_step(self, batch, batch_idx):
        context = self._build_context(batch)
        if batch['target_goal'].dim() > 2:
            raise ValueError("Target seems to be an image! The model expects a vector.")
        x_1 = self._normalize_pose(batch['target_goal']) 
        x_0 = torch.randn_like(x_1) 
        
        # Get training state (x_t, t, target) from Path
        sample = self.path.compute_loss_samples(x_0=x_0, x_1=x_1)
        
        # Predict
        prediction = self.network(sample.x_t, sample.t, context)
        loss = torch.nn.functional.mse_loss(prediction, sample.target)
        
        self.train_loss(loss)
        self.log("train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        context = self._build_context(batch)
        x_1 = self._normalize_pose(batch['target_goal'])
        x_0 = torch.randn_like(x_1)
        
        sample = self.path.compute_loss_samples(x_0=x_0, x_1=x_1)
        prediction = self.network(sample.x_t, sample.t, context)
        loss = torch.nn.functional.mse_loss(prediction, sample.target)
        
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
        self.val_loss_best(self.val_loss.compute())
        self.log("val_loss_best", self.val_loss_best, prog_bar=True)
        if hasattr(self, '_validation_context'):
            self._visualize_predictions()

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

        def _get_transformed_mask_cv2(current_mask_tensor, pose_norm, object_theta_rad):
            """
            Fixed SE(2) transformation logic.
            
            CRITICAL: The pose delta (dx, dy, dtheta) is in the OBJECT's local frame,
            not world frame. The data generation computes:
                delta_obj = R(-theta) @ delta_world
            where theta is the current object's orientation.
            
            To visualize correctly, we must:
            1. Use the provided object theta
            2. Rotate the object-frame delta back to world frame: delta_world = R(+theta) @ delta_obj
            3. Apply the world-frame delta in image coordinates (no Y-flip since 
               generate_local_episode_masks uses: py = (y - center_y) * scale + img_center)
               
            Args:
                current_mask_tensor: The mask of the current object (H, W), values in [0, 1]
                pose_norm: Normalized pose delta (3,) = (dx_norm, dy_norm, dtheta_norm)
                object_theta_rad: The current object's orientation in radians
            """
            # 1. Prepare Data
            mask_np = (current_mask_tensor.cpu().numpy() * 255).astype(np.uint8)
            H, W = mask_np.shape
            
            # 2. Decode Scaling (Meters to Pixels)
            xy_norm = self._xy_norm.item()
            theta_norm = self._theta_norm.item()
            crop_size = self.crop_size_meters
            pixels_per_meter = W / crop_size
            
            # 3. Denormalize Pose Delta (still in object frame)
            dx_obj_meters = pose_norm[0].item() * xy_norm
            dy_obj_meters = pose_norm[1].item() * xy_norm
            dtheta_rad = pose_norm[2].item() * theta_norm

            # 4. Find the Current Object's Geometry
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return torch.zeros_like(current_mask_tensor)
            
            cnt = max(contours, key=cv2.contourArea).squeeze()
            if cnt.ndim != 2: 
                return torch.zeros_like(current_mask_tensor)

            # 5. Get object centroid from the mask
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                return torch.zeros_like(current_mask_tensor)
            curr_obj_cx = M['m10'] / M['m00']
            curr_obj_cy = M['m01'] / M['m00']
            
            # Use the provided object theta
            curr_theta_rad = object_theta_rad
            
            # 6. Convert object-frame delta to world/image-frame delta
            # The data generation used: delta_obj = R(-theta) @ delta_world
            # So we need: delta_world = R(+theta) @ delta_obj
            c_theta = np.cos(curr_theta_rad)
            s_theta = np.sin(curr_theta_rad)
            
            # Rotate (dx_obj, dy_obj) by +theta to get world frame
            dx_world_meters = dx_obj_meters * c_theta - dy_obj_meters * s_theta
            dy_world_meters = dx_obj_meters * s_theta + dy_obj_meters * c_theta
            
            # Convert to pixels
            dx_px = dx_world_meters * pixels_per_meter
            dy_px = dy_world_meters * pixels_per_meter

            # 7. Apply Transformation
            # Step A: Center the points around the object's CURRENT centroid
            pts_centered = (cnt - np.array([curr_obj_cx, curr_obj_cy])).astype(np.float32)
            
            # Step B: Rotate by dtheta (the change in orientation)
            c, s = np.cos(dtheta_rad), np.sin(dtheta_rad)
            R = np.array(((c, -s), (s, c)))
            pts_rotated = (R @ pts_centered.T).T
            
            # Step C: Translate the centroid to the NEW position
            # In this image coordinate system: +dx_world is right, +dy_world is DOWN
            # (no Y-flip needed since generate_local_episode_masks doesn't flip Y)
            new_obj_cx = curr_obj_cx + dx_px
            new_obj_cy = curr_obj_cy + dy_px 
            
            # Step D: Un-center points to the new centroid
            pts_final = (pts_rotated + np.array([new_obj_cx, new_obj_cy])).astype(np.int32)
            
            # 8. Draw
            new_mask = np.zeros_like(mask_np)
            cv2.fillPoly(new_mask, [pts_final], 255)
            
            return torch.from_numpy(new_mask).float().to(current_mask_tensor.device) / 255.0

        # --- MAIN LOOP ---
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
            gt_pose_mask = _get_transformed_mask_cv2(target_obj_curr, gt_pose_delta, obj_theta)
            
            # Generate Predictions
            with torch.no_grad():
                # sample_pose returns (num_samples, 3) for B=1 input
                # So p_list is (4, 3) - 4 pose samples, each with (dx, dy, dtheta)
                p_list = self.sample_pose(ctx.unsqueeze(0), num_samples=4, denormalize=False)

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
                pred_pose_mask = _get_transformed_mask_cv2(target_obj_curr, p_list[j], obj_theta)
                
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

        self.logger.experiment.log(log_dict)

    def sample_pose(self, context, num_samples=1, num_steps=20, denormalize=True, show_progress=False):
        B = context.shape[0]
        context_repeated = context.repeat_interleave(num_samples, dim=0)
        total_samples = B * num_samples
        x_init = torch.randn(total_samples, self.vector_dim, device=context.device)
        def model_fn(x, t):
            return self.network(x, t, context_repeated)
        samples = self.sampler.sample(model_fn, x_init, num_steps, show_progress, device=context.device)
        if denormalize: return self._denormalize_pose(samples)
        return samples

    def configure_optimizers(self):
        return self.optimizer_partial(params=self.parameters())