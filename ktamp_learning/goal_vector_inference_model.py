"""
Goal Inference Model for SE(2) Vector Predictions

This model predicts goal poses as SE(2) deltas (dx, dy, dtheta) from the object's
local frame, then transforms them back to world coordinates for robot execution.

Unlike the legacy mask-based approach, this model:
1. Predicts a 3D vector (dx_local, dy_local, dtheta) in the object's frame
2. Transforms the local delta to world coordinates using the object's orientation
3. Computes the final world pose by adding the delta to the object's current pose
"""

from omegaconf import OmegaConf, DictConfig, ListConfig
import hydra
import torch
from pathlib import Path
import numpy as np
from torchvision import transforms
import sys
import os
import json

# Add paths for dependencies
NAMO_PYTHON_PATH = "/common/home/dm1487/robotics_research/ktamp/namo/python"
NAMO_VISUALIZATION_PATH = os.path.join(NAMO_PYTHON_PATH, "namo", "visualization")

for extra_path in (NAMO_PYTHON_PATH, NAMO_VISUALIZATION_PATH):
    if extra_path not in sys.path:
        sys.path.append(extra_path)

SAGE_LEARNING_ROOT = Path(__file__).resolve().parents[1]
if SAGE_LEARNING_ROOT.exists():
    sage_root_str = str(SAGE_LEARNING_ROOT)
    if sage_root_str not in sys.path:
        sys.path.insert(0, sage_root_str)

from ml_image_converter_adapter import MLImageConverterAdapter as ImageConverter
from scipy.spatial.transform import Rotation as R

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class GoalVectorInferenceModel:
    """
    Model for performing goal pose inference using a trained vector-output model.
    
    This model predicts SE(2) deltas (dx, dy, dtheta) in the OBJECT's local frame,
    then transforms them to world coordinates.
    
    The local frame is defined as:
    - X-axis: Along the object's current heading
    - Y-axis: Perpendicular to the object's heading (left)
    - Theta: Counter-clockwise rotation
    """

    def __init__(self, model_path, device="cuda", sampler_method=None, num_steps=None):
        """
        Initialize the goal vector inference model.

        Args:
            model_path: Path to model output directory (e.g., outputs/vector_model/2025-12-27_...)
            device: Device to load model on (default: "cuda")
            sampler_method: Override sampler method at inference time.
                For Flow Matching: "euler", "midpoint", "rk4", "dopri5"
                If None, uses the method from training config.
            num_steps: Override number of sampling steps (default: uses training config or 20)
        """
        self.device = device
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available; falling back to CPU.")
            self.device = "cpu"
        self.model_path = Path(model_path)
        self.checkpoint_override = None
        if self.model_path.is_file() and self.model_path.suffix == ".ckpt":
            self.checkpoint_override = self.model_path
            if self.model_path.parent.name == "checkpoints":
                self.model_path = self.model_path.parent.parent
            else:
                self.model_path = self.model_path.parent
        self.sampler_method = sampler_method
        self.num_steps = num_steps

        # Load model
        self.model, self.cfg = self._load_model()

        # Override sampler if specified
        if sampler_method is not None:
            self._override_sampler(sampler_method)

        # NOTE: VectorHFDiffusionPath uses normalized timesteps t∈[0,1] during training,
        # but HFDiffusionSampler defaults to passing integer timesteps to the network.
        # If left mismatched, sampling quality can degrade severely even if val_loss is low.
        self._maybe_enable_sampler_time_normalization()

        print(f"✅ Goal vector model loaded successfully: {type(self.model).__name__}")
        print(f"  Sampler: {type(self.model.sampler).__name__} (method: {self._get_sampler_method()})")
        
        # Get data config
        self.data_cfg = self.cfg.data

        # Check if model uses local (object-centered) masks
        self.use_local = getattr(self.cfg.model, 'use_local', True)
        
        # Get crop size from model config (default 5.0m as in training config)
        self.crop_size_meters = getattr(self.cfg.model, 'crop_size_meters', 5.0)
        print(f"  Using local mode: {self.use_local}, crop_size: {self.crop_size_meters}m")

        # Check if model was trained with coord_grid
        self.use_coord_grid = getattr(self.data_cfg, 'use_coord_grid', False)
        if self.use_coord_grid:
            print(f"  Using coordinate grid (2 extra channels)")

        # Setup image transform (same as training)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.data_cfg.image_size, self.data_cfg.image_size)),
            transforms.Lambda(lambda x: x * 2 - 1),  # Normalize to [-1, 1]
        ])

    def _maybe_enable_sampler_time_normalization(self) -> None:
        """Ensure sampler uses the same timestep scale as the training path."""
        try:
            path_target = str(getattr(self.cfg.model.path, "_target_", ""))
        except Exception:
            path_target = ""

        if "vector_hf_diffusion_path.VectorHFDiffusionPath" not in path_target:
            return

        sampler = getattr(self.model, "sampler", None)
        if sampler is None or not hasattr(sampler, "normalize_t"):
            return

        if getattr(sampler, "normalize_t", False):
            return

        sampler.normalize_t = True
        print("  [GoalVectorInferenceModel] Set sampler.normalize_t=True to match VectorHFDiffusionPath (t in [0,1])")

    def _remap_target(self, target: str) -> str:
        """Normalize Hydra target paths after package migration."""
        remap_prefixes = [
            ("ktamp_learning.src.", "src."),
            ("learning.ktamp_learning.", "ktamp_learning."),
        ]
        for old, new in remap_prefixes:
            if target.startswith(old):
                return target.replace(old, new, 1)
        return target

    def _remap_targets_recursive(self, node):
        """Recursively remap _target_ entries in Dict/List configs."""
        if isinstance(node, DictConfig):
            if "_target_" in node:
                node._target_ = self._remap_target(node._target_)
            for value in node.values():
                self._remap_targets_recursive(value)
        elif isinstance(node, ListConfig):
            for item in node:
                self._remap_targets_recursive(item)

    def _load_model(self):
        """Load a model from the given output directory path."""
        # Load config
        config_path = self.model_path / ".hydra" / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        cfg = OmegaConf.load(config_path)
        if "model" in cfg:
            self._remap_targets_recursive(cfg.model)
        
        # Find checkpoint
        checkpoint_path = None
        if self.checkpoint_override is not None:
            checkpoint_path = self.checkpoint_override
        else:
            checkpoint_dir = self.model_path / "checkpoints"
            if not checkpoint_dir.exists():
                raise FileNotFoundError(f"Checkpoints directory not found at {checkpoint_dir}")

            checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))

            # Look for epoch checkpoint first, then last.ckpt
            for checkpoint_file in checkpoint_files:
                if "epoch" in checkpoint_file.name:
                    checkpoint_path = checkpoint_file
                    break

            if checkpoint_path is None:
                # Fallback to last.ckpt
                last_ckpt = checkpoint_dir / "last.ckpt"
                if last_ckpt.exists():
                    checkpoint_path = last_ckpt
                else:
                    raise FileNotFoundError(f"No suitable checkpoint found in {checkpoint_dir}")
        
        print(f"  Loading checkpoint: {checkpoint_path}")
        
        # Load model
        model = hydra.utils.instantiate(cfg.model)
        map_location = None
        if str(self.device).startswith("cpu") or not torch.cuda.is_available():
            map_location = torch.device("cpu")
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=map_location)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)
        model.eval()
        
        return model, cfg

    def _override_sampler(self, method: str):
        """Override the model's sampler with a new method."""
        flow_matching_methods = {"euler", "midpoint", "rk4", "dopri5"}

        if method in flow_matching_methods:
            from src.model.samplers.fb_ode_sampler import FBODESampler
            self.model.sampler = FBODESampler(method=method)
            print(f"  Overriding sampler to FBODESampler(method='{method}')")
        else:
            raise ValueError(
                f"Unknown sampler method: '{method}'. "
                f"Available: {flow_matching_methods}"
            )

    def _get_sampler_method(self) -> str:
        """Get the current sampler method as a string."""
        sampler = self.model.sampler
        sampler_type = type(sampler).__name__
        if sampler_type == "FBODESampler":
            return getattr(sampler, "method", "unknown")
        return "unknown"

    def _transform_local_to_world(self, dx_local: float, dy_local: float, 
                                   dtheta: float, object_theta: float) -> tuple:
        """
        Transform a local SE(2) delta to world coordinates.
        
        Args:
            dx_local: Delta X in object's local frame (forward)
            dy_local: Delta Y in object's local frame (left)
            dtheta: Delta theta (rotation change)
            object_theta: Object's current orientation in world frame (radians)
            
        Returns:
            (dx_world, dy_world, dtheta_world)
        """
        # Rotation matrix from object frame to world frame
        cos_theta = np.cos(object_theta)
        sin_theta = np.sin(object_theta)
        
        # Transform local delta to world delta
        dx_world = dx_local * cos_theta - dy_local * sin_theta
        dy_world = dx_local * sin_theta + dy_local * cos_theta
        
        # Theta change is the same in both frames
        return dx_world, dy_world, dtheta

    def infer(self, json_message, xml_path, robot_goal, selected_object, 
              samples=32, region_goals_sampled=None):
        """
        Perform goal inference to get goal proposals.
        
        Args:
            json_message: Raw JSON message from planning system
            xml_path: Path to MuJoCo XML file for ImageConverter
            robot_goal: Robot goal position [x, y]
            selected_object: Name of the object to generate goals for
            samples: Number of samples to generate (default: 32)
            region_goals_sampled: Optional list of (x, y, theta) for goal region visualization
            
        Returns:
            List of goal dictionaries, each containing:
            - index: Sample index
            - goal_center: [x, y] goal center in world coordinates
            - final_quat: Quaternion for object rotation [w, x, y, z]
            - x, y, theta: SE(2) pose components in world frame
            - delta_local: (dx, dy, dtheta) in object's local frame
            - delta_world: (dx, dy, dtheta) in world frame
        """
        # Create ImageConverter and process data
        image_converter = ImageConverter(xml_path)
        
        # Get current object state
        if selected_object not in json_message['objects']:
            raise ValueError(f"Object '{selected_object}' not found in json_message")
        
        obj_data = json_message['objects'][selected_object]
        obj_pos = obj_data['position']
        obj_quat = obj_data['quaternion']  # [w, x, y, z] scalar-first
        
        # Get object's current orientation (theta in world frame)
        object_theta = R.from_quat(obj_quat, scalar_first=True).as_euler('xyz')[2]
        
        if self.use_local:
            # Create local masks centered on selected object
            local_masks = image_converter.create_local_masks(
                json_message,
                selected_object,
                robot_goal,
                region_goals_sampled=region_goals_sampled,
                crop_size_meters=self.crop_size_meters,
                output_size=self.data_cfg.image_size
            )
            
            # Build input channels for local mode
            # Order: static, movable, target_object, robot_region, goal_sample_region
            input_channels = [
                local_masks['local_static'],
                local_masks['local_movable'],
                local_masks['local_target_object'],
                local_masks.get('local_robot_region', np.zeros_like(local_masks['local_static'])),
                local_masks.get('local_goal_sample_region', np.zeros_like(local_masks['local_static'])),
            ]
        else:
            # Global mode (legacy fallback)
            inp_data = image_converter.process_datapoint(json_message, robot_goal)
            selected_object_mask = image_converter.create_object_mask(selected_object)
            
            input_channels = [
                inp_data['robot_image'],
                inp_data['goal_image'],
                inp_data['movable_objects_image'],
                inp_data['static_objects_image'],
                selected_object_mask,
            ]

        # Add coordinate grid if model was trained with it
        if self.use_coord_grid:
            orig_size = input_channels[0].shape[0]
            ys, xs = np.meshgrid(np.linspace(0, 1, orig_size),
                                 np.linspace(0, 1, orig_size),
                                 indexing='ij')
            coord_grid = np.stack([xs, ys], axis=-1).astype(np.float32)
            input_channels.append(coord_grid)

        # Stack and transform
        inp_for_model = np.concatenate(input_channels, axis=-1)
        inp_for_model = self.transform(inp_for_model).unsqueeze(0).to(self.device)

        # Generate samples using the model's sample_pose method
        num_steps = self.num_steps if self.num_steps is not None else 20
        
        with torch.no_grad():
            # Sample poses (already denormalized by sample_pose)
            # Returns tensor of shape (samples, 3) with (dx, dy, dtheta) in object frame
            pose_samples = self.model.sample_pose(
                inp_for_model, 
                num_samples=samples, 
                num_steps=num_steps,
                denormalize=True
            ).cpu().numpy()

        # Process samples and convert to world coordinates
        valid_goals = []
        
        for i, delta_local in enumerate(pose_samples):
            dx_local, dy_local, dtheta_local = delta_local
            
            # Transform local delta to world delta
            dx_world, dy_world, dtheta_world = self._transform_local_to_world(
                dx_local, dy_local, dtheta_local, object_theta
            )
            
            # Compute final world position
            goal_x = obj_pos[0] + dx_world
            goal_y = obj_pos[1] + dy_world
            goal_theta = object_theta + dtheta_world
            
            # Normalize theta to [-pi, pi]
            goal_theta = np.arctan2(np.sin(goal_theta), np.cos(goal_theta))
            
            # Compute final quaternion
            final_quat = R.from_euler('xyz', [0, 0, goal_theta]).as_quat(scalar_first=True)
            
            valid_goals.append({
                'index': i,
                'goal_center': [goal_x, goal_y],
                'final_quat': final_quat.tolist(),
                'x': goal_x,
                'y': goal_y,
                'theta': goal_theta,
                'delta_local': (dx_local, dy_local, dtheta_local),
                'delta_world': (dx_world, dy_world, dtheta_world),
                'input_context': inp_for_model.cpu().numpy()  # For visualization
            })

        return valid_goals

    def infer_single(self, json_message, xml_path, robot_goal, selected_object,
                     region_goals_sampled=None):
        """
        Perform single goal inference (mean of samples).
        
        This is a convenience method that returns the mean prediction
        rather than multiple samples.
        
        Returns:
            Dictionary with single goal prediction (same structure as infer())
        """
        goals = self.infer(json_message, xml_path, robot_goal, selected_object,
                          samples=16, region_goals_sampled=region_goals_sampled)
        
        if not goals:
            return None
        
        # Compute mean of predictions
        mean_x = np.mean([g['x'] for g in goals])
        mean_y = np.mean([g['y'] for g in goals])
        mean_theta = np.arctan2(
            np.mean([np.sin(g['theta']) for g in goals]),
            np.mean([np.cos(g['theta']) for g in goals])
        )
        
        final_quat = R.from_euler('xyz', [0, 0, mean_theta]).as_quat(scalar_first=True)
        
        return {
            'goal_center': [mean_x, mean_y],
            'final_quat': final_quat.tolist(),
            'x': mean_x,
            'y': mean_y,
            'theta': mean_theta,
            'num_samples': len(goals)
        }
