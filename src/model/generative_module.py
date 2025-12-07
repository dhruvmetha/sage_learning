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
        crop_size_meters: float = 5.0,
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
            self._validation_gt = x_1
        return loss

    def on_validation_epoch_end(self):
        self.val_loss_best(self.val_loss.compute())
        self.log("val_loss_best", self.val_loss_best, prog_bar=True)
        if hasattr(self, '_validation_context'):
            self._visualize_predictions()

    def _visualize_predictions(self):
        if not hasattr(self.logger, 'experiment'): return
        import wandb
        import torchvision
        
        n_vis = min(8, self._validation_context.shape[0])
        context = self._validation_context[:n_vis]
        gt_norm = self._validation_gt[:n_vis]
        
        # Sample predictions
        preds_flat = self.sample_pose(context, num_samples=4, denormalize=False) 
        preds = preds_flat.view(n_vis, 4, 3)

        log_dict = {}

        for i in range(n_vis):
            ctx = context[i]      # (5, H, W)
            gt = gt_norm[i]       # (3,)
            p_list = preds[i]     # (4, 3)
            
            # --- PREPARE PANELS ---
            # We treat the input as [0, 1] for visualization if it looks good to you.
            # Using .clamp(0, 1) to be safe.
            static = ctx[0]
            movable = ctx[1]
            target_obj = ctx[2]
            robot_reg = ctx[3]
            goal_reg = ctx[4]

            # Panel 1: Scene (R=Static, G=Movable, B=TargetObj)
            panel_scene = torch.stack([static, movable, target_obj], dim=0).clamp(0, 1)

            # Panel 2: Reachability (R=RobotReg, G=GoalReg, B=TargetObj)
            panel_reach = torch.stack([robot_reg, goal_reg, target_obj], dim=0).clamp(0, 1)

            # --- DRAWING HELPER ---
            def draw_pose_on_img(base_tensor, vector, color, alpha=1.0):
                # 1. Convert to Numpy (H, W, 3) uint8
                img_np = (base_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                img_np = np.ascontiguousarray(img_np)
                
                # 2. Draw
                self._draw_pose_cv2(img_np, vector, color, alpha=alpha)
                
                # 3. Convert back and MOVE TO DEVICE (Fixes the crash)
                return torch.from_numpy(img_np).permute(2, 0, 1).float().to(base_tensor.device) / 255.0

            # Colors (RGB)
            COLOR_GT = (0, 180, 0)      # Darker, bold Green
            COLOR_PRED = (255, 0, 0)    # Red

            # --- GENERATE PANELS ---
            
            # Panel 3: Ground Truth Only
            panel_gt = draw_pose_on_img(panel_scene, gt, COLOR_GT)

            # Panels 4-7: Prediction + Ghost GT
            panels_preds = []
            for j in range(4):
                # Start with scene (numpy)
                img_np = (panel_scene.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                img_np = np.ascontiguousarray(img_np)
                
                # Draw Ghost GT (Thinner, No Heading) - Shows context
                self._draw_pose_cv2(img_np, gt, COLOR_GT, thickness=1, show_heading=False)
                
                # Draw Prediction (Bold, With Heading)
                self._draw_pose_cv2(img_np, p_list[j], COLOR_PRED, thickness=2, show_heading=True)
                
                # Convert back AND MOVE TO DEVICE (Fixes the crash)
                p_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().to(panel_scene.device) / 255.0
                panels_preds.append(p_tensor)

            # --- STITCH GRID ---
            # Row: Scene | Reach | GT | Pred1 | Pred2 | Pred3 | Pred4
            row_tensors = [panel_scene, panel_reach, panel_gt] + panels_preds
            row_stack = torch.stack(row_tensors)
            
            grid = torchvision.utils.make_grid(row_stack, nrow=7, padding=2, normalize=False)
            
            grid_np = grid.cpu().permute(1, 2, 0).numpy()
            caption = f"Sample {i} | Scene | Reach | GT | Preds(Red) vs GT(Grn)"
            log_dict[f'val_sample_{i}'] = wandb.Image(grid_np, caption=caption)

        self.logger.experiment.log(log_dict)

    def _draw_pose_cv2(self, img_np, pose_norm, color, thickness=2, alpha=1.0, show_heading=True):
        """
        Draws translation (arrow) and orientation (heading line).
        Includes a WHITE OUTLINE for contrast against blue/black/gray.
        """
        H, W, _ = img_np.shape
        center_x, center_y = W // 2, H // 2
        
        # Scale: Normalized 1.0 -> Edge of image
        # pose_norm is [-1, 1], so we multiply by W/2 to get pixels.
        scale = W / 2.0
        
        # 1. Translation
        dx = int(pose_norm[0].item() * scale)
        dy = int(pose_norm[1].item() * scale)
        end_x = max(0, min(W-1, center_x + dx))
        end_y = max(0, min(H-1, center_y - dy)) # Flip Y for image coords

        # 2. Heading (Whisker)
        # Convert normalized theta back to raw radians for drawing
        raw_theta = pose_norm[2].item() * self.theta_norm
        
        heading_len = 15 # pixels
        head_dx = int(heading_len * math.cos(raw_theta))
        head_dy = int(heading_len * math.sin(raw_theta)) 
        
        head_x = max(0, min(W-1, end_x + head_dx))
        head_y = max(0, min(H-1, end_y - head_dy)) # Flip Y

        # --- DRAWING ---
        
        # A. The Halo (White Outline) - Guarantees visibility
        outline_color = (255, 255, 255)
        outline_thick = thickness + 2
        
        # Arrow Shaft (Outline)
        cv2.arrowedLine(img_np, (center_x, center_y), (end_x, end_y), outline_color, outline_thick, tipLength=0.2)
        
        if show_heading:
            # Heading Whisker (Outline)
            cv2.line(img_np, (end_x, end_y), (head_x, head_y), outline_color, outline_thick)
            cv2.circle(img_np, (end_x, end_y), 4, outline_color, -1)

        # B. The Actual Color Line
        
        # Arrow Shaft (Color)
        cv2.arrowedLine(img_np, (center_x, center_y), (end_x, end_y), color, thickness, tipLength=0.2)
        
        if show_heading:
            # Heading Whisker (Color)
            cv2.line(img_np, (end_x, end_y), (head_x, head_y), color, thickness)
            cv2.circle(img_np, (end_x, end_y), 2, color, -1)

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