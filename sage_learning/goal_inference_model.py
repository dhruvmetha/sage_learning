from omegaconf import OmegaConf, DictConfig, ListConfig
import hydra
import torch
from pathlib import Path
import numpy as np
import cv2
import os
from torchvision import transforms
from sage_learning.utils.image_utils import find_rectangle_corners
from sage_learning.image_converter import MLImageConverterAdapter as ImageConverter
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

class GoalInferenceModel:
    """
    Model for performing goal pose inference using a trained diffusion model.
    Generates goal proposals for a selected object in SE(2) space.

    Supports two model types:
    - Image mask models (GenerativeModule): Output image masks, converted to SE2 via rectangle fitting
    - SE2 pose models (VectorDiffusionModule): Output SE2 deltas directly in object-local frame
    """

    def __init__(self, model_path, device="cuda", sampler_method=None, num_steps=None):
        """
        Initialize the goal inference model.

        Args:
            model_path: Path to model output directory (e.g., outputs/rel_reach_coord_goal_dit/mse/2025-08-10_06-59-27)
            device: Device to load model on (default: "cuda")
            sampler_method: Override sampler method at inference time.
                For Flow Matching: "euler", "midpoint", "rk4", "dopri5"
                For Diffusion: "ddpm", "ddim"
                If None, uses the method from training config.
            num_steps: Override number of sampling steps (default: uses training config or 20)
        """
        self.device = device
        self.model_path = Path(model_path)
        self.sampler_method = sampler_method
        self.num_steps = num_steps

        # Load model
        self.model, self.cfg = self._load_model()

        # Override sampler if specified
        if sampler_method is not None:
            self._override_sampler(sampler_method)

        # Model loaded - sampler: {type(self.model.sampler).__name__} (method: {self._get_sampler_method()})

        # Detect model type: SE2 pose model vs image mask model
        # SE2 models have vector_dim in config and use VectorDiffusionModule
        model_cfg = self.cfg.model
        self.is_se2_model = (
            hasattr(model_cfg, 'vector_dim') or
            'VectorDiffusionModule' in model_cfg.get('_target_', '') or
            'vector_diffusion' in model_cfg.get('_target_', '').lower()
        )

        # For SE2 models, get crop size in meters for coordinate transform
        if self.is_se2_model:
            self.crop_size_meters = getattr(model_cfg, 'crop_size_meters', 5.0)

        # Use data config
        self.data_cfg = self.cfg.data

        # Check if model was trained with coord_grid
        self.use_coord_grid = getattr(self.data_cfg, 'use_coord_grid', False)

        # Check if model was trained with local (object-centered) masks
        # Default to True since all recent models use local masks
        self.use_local = getattr(self.data_cfg, 'use_local', True)

        # Get image/context size - support both naming conventions
        # - image_size: used by older models
        # - context_size: used by newer cropped output models
        self.context_size = getattr(self.data_cfg, 'context_size',
                                    getattr(self.data_cfg, 'image_size', 64))

        # For cropped output models, sample_from_model() pads output to context_size
        # so we always use context_size for coordinate conversion
        self.crop_size = getattr(self.data_cfg, 'crop_size',
                                 getattr(self.cfg, 'crop_size', None))

        # Setup image transform
        # SE2 models use 224x224 input, mask models use context_size
        target_size = 224 if self.is_se2_model else self.context_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((target_size, target_size)),
            transforms.Lambda(lambda x: x * 2 - 1),
        ])
        
    def _remap_target(self, target: str) -> str:
        """Normalize Hydra target paths after package migration."""
        remap_prefixes = [
            ("ktamp_learning.src.", "src."),
            ("learning.ktamp_learning.", "sage_learning."),
            ("ktamp_learning.", "sage_learning."),
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
        checkpoint_dir = self.model_path / "checkpoints"
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoints directory not found at {checkpoint_dir}")
            
        checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
        checkpoint_path = None
        
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
        
        # Load model
        model = hydra.utils.instantiate(cfg.model)
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)
        model.eval()
        
        return model, cfg

    def _override_sampler(self, method: str):
        """Override the model's sampler with a new method.

        Args:
            method: Sampler method to use.
                Flow Matching: "euler", "midpoint", "rk4", "dopri5"
                Diffusion: "ddpm", "ddim"
        """
        flow_matching_methods = {"euler", "midpoint", "rk4", "dopri5"}
        diffusion_methods = {"ddpm", "ddim"}

        if method in flow_matching_methods:
            from src.model.samplers.fb_ode_sampler import FBODESampler
            self.model.sampler = FBODESampler(method=method)

        elif method in diffusion_methods:
            from src.model.samplers.hf_diffusion_sampler import HFDiffusionSampler
            # Get diffusion params from config if available
            sampler_cfg = self.cfg.model.get("sampler", {})
            num_timesteps = sampler_cfg.get("num_train_timesteps", 1000)
            self.model.sampler = HFDiffusionSampler(
                sampler_type=method,
                num_train_timesteps=num_timesteps,
            )

        else:
            raise ValueError(
                f"Unknown sampler method: '{method}'. "
                f"Flow Matching: {flow_matching_methods}, Diffusion: {diffusion_methods}"
            )

    def _get_sampler_method(self) -> str:
        """Get the current sampler method as a string."""
        sampler = self.model.sampler
        sampler_type = type(sampler).__name__

        if sampler_type == "FBODESampler":
            return getattr(sampler, "method", "unknown")
        elif sampler_type == "HFDiffusionSampler":
            return getattr(sampler, "sampler_type", "unknown")
        else:
            return "unknown"

    def infer(self, json_message, xml_path, robot_goal, selected_object, samples=32, seed=None):
        """
        Perform goal inference to get goal proposals.

        Automatically routes to the appropriate inference method based on model type:
        - SE2 models: Direct pose prediction via _infer_se2()
        - Image mask models: Mask prediction via _infer_local() or global inference

        Args:
            json_message: Raw JSON message from planning system
            xml_path: Path to MuJoCo XML file for ImageConverter
            robot_goal: Robot goal position [x, y]
            selected_object: Name of the object to generate goals for
            samples: Number of samples to generate (default: 32)
            seed: Random seed for reproducible noise (None for random)

        Returns:
            List of goal dictionaries, each containing:
            - index: Sample index
            - goal_center: [x, y] goal center in world coordinates (global/mask models only)
            - final_quat: Quaternion for object rotation (global/mask models only)
            - x, y, theta: SE(2) pose components
            - goal_sample: Raw goal sample array (mask models only)
            - input_channels: Input tensor for visualization
        """
        # Route to SE2 inference for direct pose prediction models
        if self.is_se2_model:
            return self._infer_se2(json_message, xml_path, robot_goal, selected_object, samples, seed=seed)

        # Auto-route to local inference if model was trained with use_local=True
        if self.use_local:
            return self._infer_local(json_message, xml_path, robot_goal, selected_object, samples, seed=seed)

        # Global inference (original behavior)
        # Create ImageConverter and process data
        image_converter = ImageConverter(xml_path)
        inp_data = image_converter.process_datapoint(json_message, robot_goal)

        # Create object mask for the selected object
        try:
            selected_object_mask = image_converter.create_object_mask(selected_object)
        except Exception as e:
            raise ValueError(f"Error creating object mask for '{selected_object}': {e}")
        
        # Prepare input for goal model (stack scene context + selected object mask)
        input_channels = [
            inp_data['robot_image'],
            inp_data['goal_image'],
            inp_data['movable_objects_image'],
            inp_data['static_objects_image'],
            selected_object_mask                   # Selected object mask (channel 5)
        ]

        # Add coordinate grid if model was trained with it (matches training exactly)
        if self.use_coord_grid:
            # Use original image size (before transform resizes to data_cfg.image_size)
            orig_size = inp_data['robot_image'].shape[0]
            ys, xs = np.meshgrid(np.linspace(0, 1, orig_size),
                                 np.linspace(0, 1, orig_size),
                                 indexing='ij')
            coord_grid = np.stack([xs, ys], axis=-1).astype(np.float32)
            input_channels.append(coord_grid)

        inp_for_goal = np.concatenate(input_channels, axis=-1)
        inp_for_goal = self.transform(inp_for_goal).unsqueeze(0).to(self.device)

        # Generate goal samples
        num_steps = self.num_steps if self.num_steps is not None else 20
        with torch.no_grad():
            goal_samples = (self.model.sample_from_model(inp_for_goal, samples=samples, num_steps=num_steps, seed=seed)
                          .permute(0, 2, 3, 1).cpu().numpy() + 1) / 2

        inp_for_goal = inp_for_goal.cpu().squeeze(0).numpy()

        # Process goal samples and extract SE(2) poses
        valid_goals = []
        scale = image_converter.IMG_SIZE / self.context_size
        
        # Get object angle for rotation calculation
        _, _, selected_obj_center, obj_angle = find_rectangle_corners(
            (selected_object_mask[:, :, 0] > 0.5).astype(np.uint8))
        
        if selected_obj_center is None:
            # Fallback: use stored angle from obj2angle if available in inp_data
            obj_angle = inp_data.get('obj2angle', {}).get(selected_object, 0)
            
        for i, goal_sample in enumerate(goal_samples):
            goal_mask = (goal_sample[:, :, 0].copy() > 0.5) * 1.0
            goal_mask = goal_mask.astype(np.uint8)
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(goal_mask)
            if num_labels > 2:
                continue
            
            _, _, predicted_goal_center, goal_angle = find_rectangle_corners(goal_mask)
            if predicted_goal_center is None:
                continue
                
            # Scale to original image coordinates
            predicted_goal_center = (int(predicted_goal_center[0] * scale), 
                                   int(predicted_goal_center[1] * scale))
            
            # Convert to world coordinates
            goal_center = list(image_converter.pixel_to_world(
                predicted_goal_center[0], predicted_goal_center[1]))
            
            # Calculate final quaternion
            final_quat = image_converter.rotate_relative_to_world(
                selected_object, goal_angle - obj_angle)
            
            # Convert quaternion to euler angle (θ)
            goal_theta = R.from_quat(final_quat, scalar_first=True).as_euler('xyz')[2]
            
            # print(goal_theta)
            
            valid_goals.append({
                'index': i,
                'goal_center': goal_center,
                'final_quat': final_quat,
                'x': goal_center[0],
                'y': goal_center[1],
                'theta': goal_theta,
                'goal_sample': goal_sample,
                'input_channels': inp_for_goal  # Include input for visualization
            })

        return valid_goals

    def _infer_local(self, json_message, xml_path, robot_goal, selected_object, samples=32, seed=None):
        """
        Perform goal inference using local (object-centered) masks.

        This is an internal method called by infer() when the model was trained
        with use_local=True. It uses the same mask generation as training:
        - 5 input channels: static, movable, target_object, robot_region, goal_sample_region
        - All masks are cropped around the selected object (5m x 5m window)
        - Predictions are converted from local pixel coordinates to world coordinates

        Args:
            json_message: Raw JSON message from planning system
            xml_path: Path to MuJoCo XML file for ImageConverter
            robot_goal: Robot goal position [x, y]
            selected_object: Name of the object to generate goals for
            samples: Number of samples to generate (default: 32)
            seed: Random seed for reproducible noise (None for random)

        Returns:
            List of goal dictionaries, each containing:
            - index: Sample index
            - x, y, theta: SE(2) pose in world coordinates
            - goal_sample: Raw goal sample array
            - input_channels: Input tensor for visualization
        """
        # Create ImageConverter and generate local masks
        image_converter = ImageConverter(xml_path)
        local_data = image_converter.create_local_masks(
            data_point=json_message,
            selected_object=selected_object,
            robot_goal_pos=robot_goal,
            region_goals_sampled=None,  # Will use robot_goal as fallback
            crop_size_meters=5.0,
            highres_size=1024,
            output_size=224
        )

        # Check if local masks were generated successfully
        if 'local_static' not in local_data:
            raise ValueError(f"Failed to generate local masks for object '{selected_object}'")

        # Stack input channels in TRAINING ORDER:
        # static, movable, target_object, robot_region, goal_sample_region
        input_channels = [
            local_data['local_static'],
            local_data['local_movable'],
            local_data['local_target_object'],
            local_data['local_robot_region'],
            local_data['local_goal_sample_region'],
        ]

        # Add coordinate grid if model was trained with it
        if self.use_coord_grid:
            orig_size = local_data['local_static'].shape[0]
            ys, xs = np.meshgrid(np.linspace(0, 1, orig_size),
                                 np.linspace(0, 1, orig_size),
                                 indexing='ij')
            coord_grid = np.stack([xs, ys], axis=-1).astype(np.float32)
            input_channels.append(coord_grid)

        # Concatenate and transform
        inp_for_goal = np.concatenate(input_channels, axis=-1)
        inp_for_goal = self.transform(inp_for_goal).unsqueeze(0).to(self.device)

        # Generate goal samples
        num_steps = self.num_steps if self.num_steps is not None else 20
        with torch.no_grad():
            goal_samples = (self.model.sample_from_model(inp_for_goal, samples=samples, num_steps=num_steps, seed=seed)
                          .permute(0, 2, 3, 1).cpu().numpy() + 1) / 2

        inp_for_goal_np = inp_for_goal.cpu().squeeze(0).numpy()

        # Get metadata for coordinate conversion
        object_center = local_data['object_center']
        object_theta = local_data['object_theta']
        crop_size = local_data['crop_size_meters']

        # Get object angle from input mask for rotation calculation
        # IMPORTANT: local_data masks are 224x224, but model operates on context_size
        # We need to resize the mask to match the model's context size for angle detection
        obj_mask_224 = local_data['local_target_object'][:, :, 0]
        obj_mask_resized = cv2.resize(obj_mask_224, (self.context_size, self.context_size),
                                       interpolation=cv2.INTER_AREA)
        obj_mask = (obj_mask_resized > 0.5).astype(np.uint8)
        _, _, obj_mask_center, obj_angle = find_rectangle_corners(obj_mask)
        if obj_angle is None:
            obj_angle = 0.0

        # Process goal samples and extract SE(2) poses
        # Note: For cropped output models, sample_from_model() already pads the output
        # from crop_size to context_size, so predictions are in context_size coordinates
        valid_goals = []

        for i, goal_sample in enumerate(goal_samples):
            goal_mask = (goal_sample[:, :, 0].copy() > 0.5) * 1.0
            goal_mask = goal_mask.astype(np.uint8)

            # Skip if multiple disconnected regions (invalid prediction)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(goal_mask)
            if num_labels > 2:
                continue

            # Find predicted goal center and angle
            _, _, predicted_goal_center, goal_angle = find_rectangle_corners(goal_mask)
            if predicted_goal_center is None:
                continue

            # Convert LOCAL pixel to world coordinates
            # Note: predicted_goal_center is in context_size space (model pads cropped output)
            world_x, world_y = image_converter.pixel_to_world_local(
                px=predicted_goal_center[0],
                py=predicted_goal_center[1],
                object_center=object_center,
                crop_size_meters=crop_size,
                output_size=self.context_size  # Use context size for world conversion
            )

            # Compute theta from angle difference
            # goal_angle and obj_angle are in degrees from find_rectangle_corners
            angle_diff_deg = goal_angle - obj_angle if goal_angle is not None else 0.0

            # Handle 180° ambiguity: rectangles look identical when rotated 180°
            # Normalize angle_diff to [-90, 90] range
            if angle_diff_deg > 90:
                angle_diff_deg -= 180
            elif angle_diff_deg < -90:
                angle_diff_deg += 180

            goal_theta = object_theta + np.radians(angle_diff_deg)

            # Normalize theta to [-pi, pi]
            while goal_theta > np.pi:
                goal_theta -= 2 * np.pi
            while goal_theta < -np.pi:
                goal_theta += 2 * np.pi

            valid_goals.append({
                'index': i,
                'x': world_x,
                'y': world_y,
                'theta': goal_theta,
                'goal_sample': goal_sample,
                'input_channels': inp_for_goal_np
            })

        return valid_goals

    def _infer_se2(self, json_message, xml_path, robot_goal, selected_object, samples=32, seed=None):
        """
        Perform goal inference using SE2 pose prediction models.

        SE2 models directly output (dx, dy, dtheta) deltas in the object's local frame.
        These deltas are transformed to world coordinates using the object's current pose.

        Args:
            json_message: Raw JSON message from planning system
            xml_path: Path to MuJoCo XML file for ImageConverter
            robot_goal: Robot goal position [x, y]
            selected_object: Name of the object to generate goals for
            samples: Number of samples to generate (default: 32)
            seed: Random seed for reproducible noise (None for random)

        Returns:
            List of goal dictionaries, each containing:
            - index: Sample index
            - x, y, theta: SE(2) pose in world coordinates
            - input_channels: Input tensor for visualization
        """
        # Create ImageConverter and generate local masks (same as mask models)
        image_converter = ImageConverter(xml_path)
        local_data = image_converter.create_local_masks(
            data_point=json_message,
            selected_object=selected_object,
            robot_goal_pos=robot_goal,
            region_goals_sampled=None,
            crop_size_meters=self.crop_size_meters,
            highres_size=1024,
            output_size=224  # SE2 models use 224x224
        )

        if 'local_static' not in local_data:
            raise ValueError(f"Failed to generate local masks for object '{selected_object}'")

        # Stack input channels in TRAINING ORDER:
        # static, movable, target_object, robot_region, goal_sample_region
        input_channels = [
            local_data['local_static'],
            local_data['local_movable'],
            local_data['local_target_object'],
            local_data['local_robot_region'],
            local_data['local_goal_sample_region'],
        ]

        # Add coordinate grid if model was trained with it
        if self.use_coord_grid:
            orig_size = local_data['local_static'].shape[0]
            ys, xs = np.meshgrid(np.linspace(0, 1, orig_size),
                                 np.linspace(0, 1, orig_size),
                                 indexing='ij')
            coord_grid = np.stack([xs, ys], axis=-1).astype(np.float32)
            input_channels.append(coord_grid)

        # Concatenate and transform to tensor
        inp_for_goal = np.concatenate(input_channels, axis=-1)
        inp_for_goal = self.transform(inp_for_goal).unsqueeze(0).to(self.device)

        # Get object's current pose for coordinate transformation
        object_center = local_data['object_center']  # (x, y) in world frame
        object_theta = local_data['object_theta']    # radians

        # Generate SE2 pose samples using model.sample_pose()
        num_steps = self.num_steps if self.num_steps is not None else 20

        with torch.no_grad():
            # Set seed if provided
            if seed is not None:
                torch.manual_seed(seed)

            # sample_pose returns (num_samples, 3) tensor with (dx, dy, dtheta) deltas
            # These are already denormalized (in meters/radians) in object-local frame
            pose_deltas = self.model.sample_pose(
                inp_for_goal,
                num_samples=samples,
                num_steps=num_steps,
                denormalize=True
            )

        inp_for_goal_np = inp_for_goal.cpu().squeeze(0).numpy()

        # Transform deltas from object-local frame to world coordinates
        # This is the same transformation used by primitives:
        # goal = object_pose + R(object_theta) @ delta
        cos_t = np.cos(object_theta)
        sin_t = np.sin(object_theta)

        valid_goals = []
        for i, delta in enumerate(pose_deltas):
            dx, dy, dtheta = delta.cpu().numpy()

            # Transform position delta to world frame
            world_x = object_center[0] + dx * cos_t - dy * sin_t
            world_y = object_center[1] + dx * sin_t + dy * cos_t

            # Add angular delta to get goal theta
            goal_theta = object_theta + dtheta

            # Normalize theta to [-pi, pi]
            while goal_theta > np.pi:
                goal_theta -= 2 * np.pi
            while goal_theta < -np.pi:
                goal_theta += 2 * np.pi

            valid_goals.append({
                'index': i,
                'x': float(world_x),
                'y': float(world_y),
                'theta': float(goal_theta),
                'input_channels': inp_for_goal_np,
                # Include deltas for debugging/visualization
                'delta_x': float(dx),
                'delta_y': float(dy),
                'delta_theta': float(dtheta),
            })

        return valid_goals