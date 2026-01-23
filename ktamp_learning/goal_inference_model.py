from omegaconf import OmegaConf, DictConfig, ListConfig
import hydra
import torch
from pathlib import Path
import numpy as np
import cv2
from torchvision import transforms
from ktamp_learning.utils.image_utils import find_rectangle_corners
# Use unified image converter instead of original json2img
import sys
import os
from pathlib import Path

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
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class GoalInferenceModel:
    """
    Model for performing goal pose inference using a trained diffusion model.
    Generates goal proposals for a selected object in SE(2) space.
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

        print(f"✅ Goal model loaded successfully: {type(self.model).__name__}")
        print(f"  Sampler: {type(self.model.sampler).__name__} (method: {self._get_sampler_method()})")
        
        # Use data config
        self.data_cfg = self.cfg.data

        # Determine local/cropped settings
        self.context_size = getattr(self.data_cfg, "context_size", None)
        self.crop_size = getattr(self.data_cfg, "crop_size", None)
        if self.context_size is None:
            self.context_size = getattr(self.cfg.model, "context_size", None)
        if self.crop_size is None:
            self.crop_size = getattr(self.cfg.model, "crop_size", None)

        self.use_local = getattr(self.cfg.model, "use_local", False)
        if self.context_size is not None or self.crop_size is not None:
            self.use_local = True

        self.image_size = getattr(self.data_cfg, "image_size", None)
        if self.image_size is None:
            self.image_size = self.context_size or 224

        self.crop_size_meters = getattr(self.cfg.model, "crop_size_meters", 5.0)

        # Check if model was trained with coord_grid
        self.use_coord_grid = getattr(self.data_cfg, 'use_coord_grid', False)
        if self.use_coord_grid:
            print(f"  Using coordinate grid (2 extra channels)")

        # Setup image transforms
        context_size = self.context_size or self.image_size
        self.context_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((context_size, context_size)),
            transforms.Lambda(lambda x: x * 2 - 1),
        ])
        
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
            print(f"  Overriding sampler to FBODESampler(method='{method}')")

        elif method in diffusion_methods:
            from src.model.samplers.hf_diffusion_sampler import HFDiffusionSampler
            # Get diffusion params from config if available
            sampler_cfg = self.cfg.model.get("sampler", {})
            path_cfg = self.cfg.model.get("path", {})
            num_timesteps = sampler_cfg.get("num_train_timesteps", path_cfg.get("num_train_timesteps", 1000))
            beta_schedule = sampler_cfg.get("beta_schedule", path_cfg.get("beta_schedule", "squaredcos_cap_v2"))
            beta_start = sampler_cfg.get("beta_start", path_cfg.get("beta_start", 0.0001))
            beta_end = sampler_cfg.get("beta_end", path_cfg.get("beta_end", 0.02))
            prediction_type = sampler_cfg.get("prediction_type", path_cfg.get("prediction_type", "epsilon"))
            clip_sample = sampler_cfg.get(
                "inference_clip_sample",
                sampler_cfg.get("clip_sample", path_cfg.get("clip_sample", False)),
            )
            if not clip_sample:
                clip_sample = True
            eta = sampler_cfg.get("eta", 0.0)
            normalize_t = sampler_cfg.get("normalize_t", path_cfg.get("normalize_t", False))
            self.model.sampler = HFDiffusionSampler(
                sampler_type=method,
                num_train_timesteps=num_timesteps,
                beta_schedule=beta_schedule,
                beta_start=beta_start,
                beta_end=beta_end,
                prediction_type=prediction_type,
                clip_sample=clip_sample,
                eta=eta,
                normalize_t=normalize_t,
            )
            print(f"  Overriding sampler to HFDiffusionSampler(sampler_type='{method}')")

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
            - goal_center: [x, y] goal center in world coordinates
            - final_quat: Quaternion for object rotation
            - x, y, theta: SE(2) pose components
            - goal_sample: Raw goal sample array
        """
        # Create ImageConverter and process data
        image_converter = ImageConverter(xml_path)

        if self.use_local:
            context_size = self.context_size or self.image_size
            local_masks = image_converter.create_local_masks(
                json_message,
                selected_object,
                robot_goal,
                crop_size_meters=self.crop_size_meters,
                output_size=context_size,
            )

            input_channels = [
                local_masks["local_static"],
                local_masks["local_movable"],
                local_masks["local_target_object"],
                local_masks.get("local_robot_region", np.zeros_like(local_masks["local_static"])),
                local_masks.get(
                    "local_goal_sample_region",
                    local_masks.get("local_goal_region", np.zeros_like(local_masks["local_static"])),
                ),
            ]

            if self.use_coord_grid:
                orig_size = input_channels[0].shape[0]
                ys, xs = np.meshgrid(
                    np.linspace(0, 1, orig_size),
                    np.linspace(0, 1, orig_size),
                    indexing="ij",
                )
                coord_grid = np.stack([xs, ys], axis=-1).astype(np.float32)
                input_channels.append(coord_grid)

            inp_for_goal = np.concatenate(input_channels, axis=-1)
            inp_for_goal = self.context_transform(inp_for_goal).unsqueeze(0).to(self.device)

            object_center = local_masks.get("object_center", None)
            object_theta = local_masks.get("object_theta", None)
            crop_size_meters = local_masks.get("crop_size_meters", self.crop_size_meters)
            output_size = self.crop_size or context_size

            if object_theta is not None:
                obj_angle = np.degrees(object_theta)
            else:
                obj_angle = 0.0

            selected_object_mask = local_masks["local_target_object"]
            if object_center is None:
                _, _, _, obj_angle_mask = find_rectangle_corners(
                    (selected_object_mask[:, :, 0] > 0.5).astype(np.uint8)
                )
                if obj_angle_mask is not None:
                    obj_angle = obj_angle_mask
        else:
            inp_data = image_converter.process_datapoint(json_message, robot_goal)

            try:
                selected_object_mask = image_converter.create_object_mask(selected_object)
            except Exception as e:
                raise ValueError(f"Error creating object mask for '{selected_object}': {e}")

            input_channels = [
                inp_data['robot_image'],
                inp_data['goal_image'],
                inp_data['movable_objects_image'],
                inp_data['static_objects_image'],
                selected_object_mask
            ]

            if self.use_coord_grid:
                orig_size = inp_data['robot_image'].shape[0]
                ys, xs = np.meshgrid(np.linspace(0, 1, orig_size),
                                     np.linspace(0, 1, orig_size),
                                     indexing='ij')
                coord_grid = np.stack([xs, ys], axis=-1).astype(np.float32)
                input_channels.append(coord_grid)

            inp_for_goal = np.concatenate(input_channels, axis=-1)
            inp_for_goal = self.context_transform(inp_for_goal).unsqueeze(0).to(self.device)

            scale = image_converter.IMG_SIZE / self.image_size

            _, _, selected_obj_center, obj_angle = find_rectangle_corners(
                (selected_object_mask[:, :, 0] > 0.5).astype(np.uint8))

            if selected_obj_center is None:
                obj_angle = inp_data.get('obj2angle', {}).get(selected_object, 0)

        # Generate goal samples
        num_steps = self.num_steps if self.num_steps is not None else 20
        with torch.no_grad():
            goal_samples = (self.model.sample_from_model(inp_for_goal, samples=samples, num_steps=num_steps, seed=seed)
                          .permute(0, 2, 3, 1).cpu().numpy() + 1) / 2

        inp_for_goal = inp_for_goal.cpu().squeeze(0).numpy()

        # Process goal samples and extract SE(2) poses
        valid_goals = []
            
        for i, goal_sample in enumerate(goal_samples):
            goal_mask = (goal_sample[:, :, 0].copy() > 0.5) * 1.0
            goal_mask = goal_mask.astype(np.uint8)
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(goal_mask)
            if num_labels > 2:
                continue
            
            _, _, predicted_goal_center, goal_angle = find_rectangle_corners(goal_mask)
            if predicted_goal_center is None:
                continue
                
            if self.use_local:
                if object_center is None:
                    continue
                goal_center = list(image_converter.pixel_to_world_local(
                    int(predicted_goal_center[0]),
                    int(predicted_goal_center[1]),
                    object_center,
                    crop_size_meters=crop_size_meters,
                    output_size=output_size,
                ))
                final_quat = image_converter.rotate_relative_to_world(
                    selected_object, goal_angle - obj_angle)
            else:
                predicted_goal_center = (int(predicted_goal_center[0] * scale),
                                       int(predicted_goal_center[1] * scale))

                goal_center = list(image_converter.pixel_to_world(
                    predicted_goal_center[0], predicted_goal_center[1]))

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
