import re
from pathlib import Path

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from scipy.spatial.transform import Rotation as R
from torchvision import transforms

from sage_learning.image_converter import MLImageConverterAdapter as ImageConverter
from sage_learning.utils.image_utils import find_rectangle_corners

class GoalInferenceModel:
    """
    Model for performing goal pose inference using a trained diffusion model.
    Generates goal proposals for a selected object in SE(2) space.
    """

    def __init__(
        self,
        model_path,
        device="cuda",
        sampler_method=None,
        num_steps=None,
        namo_config_path=None,
    ):
        """
        Initialize the goal inference model.

        Args:
            model_path: Path to model output directory (e.g., outputs/rel_reach_coord_goal_dit/mse/2025-08-10_06-59-27)
            device: Device to load model on (default: "cuda")
            sampler_method: Override sampler method at inference time.
                For Flow Matching: "euler", "midpoint", "rk4", "dopri5"
                For diffusion samplers: "ddpm", "ddim"
                Not supported for direct multi-hypothesis SE(2) predictors.
                If None, uses the method from training config.
            num_steps: Override number of sampling steps (default: uses training config or 20)
            namo_config_path: Compatibility placeholder for planner call sites.
        """
        self.device = device
        self.model_path = Path(model_path)
        self.namo_config_path = namo_config_path
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

        # Model loaded - sampler: {type(self.model.sampler).__name__} (method: {self._get_sampler_method()})

        # Use data config
        self.data_cfg = self.cfg.data
        model_name = type(self.model).__name__
        self.is_se2_model = model_name in {
            "SE2DiffusionModule",
            "SE2MultiHypothesisModule",
            "SE2MultiHypothesisV2Module",
        }
        self.is_diffusion_se2_model = model_name == "SE2DiffusionModule"
        self.is_multihyp_model = model_name in {
            "SE2MultiHypothesisModule",
            "SE2MultiHypothesisV2Module",
        }

        # Check if model was trained with coord_grid
        self.use_coord_grid = getattr(self.data_cfg, 'use_coord_grid', False)

        # Check if model was trained with local (object-centered) masks
        # Default to True since all recent models use local masks
        self.use_local = getattr(self.data_cfg, 'use_local', True)

        # Check if model uses BFS region masks or point masks for robot/goal
        # True (default): use local_robot_region (BFS reachability)
        # False: use local_robot (point position)
        # Auto-detect from data_dir if not explicitly set
        use_region_masks = getattr(self.data_cfg, 'use_region_masks', None)
        if use_region_masks is None:
            # Fallback: detect from data_dir path (if contains "non_region", use point masks)
            data_dir = getattr(self.data_cfg, 'data_dir', '') or ''
            use_region_masks = 'non_region' not in data_dir.lower()
        self.use_region_masks = use_region_masks

        # Get image/context size - support both naming conventions
        # - image_size: used by older models
        # - context_size: used by newer cropped output models
        self.context_size = getattr(self.data_cfg, 'context_size',
                                    getattr(self.data_cfg, 'image_size', 64))

        # For cropped output models, sample_from_model() pads output to context_size
        # so we always use context_size for coordinate conversion
        self.crop_size = getattr(self.data_cfg, 'crop_size',
                                 getattr(self.cfg, 'crop_size', None))

        # Current SE(2) models train on local_tight 0.5 m crops, while legacy
        # local mask models use 5.0 m. Default to the SE(2) crop when no config
        # field is available because script-trained SE(2) runs do not emit Hydra.
        self.local_crop_size_meters = float(
            getattr(
                self.data_cfg,
                "crop_size_meters",
                getattr(self.cfg, "crop_size_meters", 0.5 if self.is_se2_model else 5.0),
            )
        )
        self.local_render_size = 224

        # Setup image transform (resize to context_size)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.context_size, self.context_size)),
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

    @staticmethod
    def _find_checkpoint_path(run_dir: Path, checkpoint_override: Path | None) -> Path:
        if checkpoint_override is not None:
            return checkpoint_override

        checkpoint_dir = run_dir / "checkpoints"
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoints directory not found at {checkpoint_dir}")

        last_ckpt = checkpoint_dir / "last.ckpt"
        if last_ckpt.exists():
            return last_ckpt

        checkpoint_files = sorted(checkpoint_dir.glob("*.ckpt"))
        epoch_candidates = [ckpt for ckpt in checkpoint_files if ckpt.name != "last.ckpt"]
        if not epoch_candidates:
            raise FileNotFoundError(f"No suitable checkpoint found in {checkpoint_dir}")

        def sort_key(path: Path):
            name = path.name
            epoch = -1
            val_loss = float("inf")

            match = re.search(r"epoch=(\d+)-val_loss=([0-9]+(?:\.[0-9]+)?)\.ckpt$", name)
            if match:
                epoch = int(match.group(1))
                val_loss = float(match.group(2))
                return (0, val_loss, -epoch, name)

            match = re.search(r"se2-(\d+)-([0-9]+(?:\.[0-9]+)?)\.ckpt$", name)
            if match:
                epoch = int(match.group(1))
                val_loss = float(match.group(2))
                return (0, val_loss, -epoch, name)

            match = re.search(r"epoch[=_-](\d+)", name)
            if match:
                epoch = int(match.group(1))
                return (1, -epoch, name)

            return (2, name)

        return min(epoch_candidates, key=sort_key)

    def _build_checkpoint_only_model(self, checkpoint: dict):
        """Reconstruct script-trained SE(2) models without a Hydra config."""
        hparams = dict(checkpoint.get("hyper_parameters", {}))
        if not hparams:
            raise FileNotFoundError(
                f"Config file not found at {self.model_path / '.hydra' / 'config.yaml'} "
                "and checkpoint lacks hyper_parameters for reconstruction."
            )

        cfg = OmegaConf.create({
            "data": {
                "context_size": int(hparams.get("context_size", 64)),
                "use_local": bool(hparams.get("use_local", True)),
                "use_region_masks": bool(hparams.get("use_region_masks", True)),
                "use_coord_grid": bool(hparams.get("use_coord_grid", False)),
                "crop_size_meters": float(hparams.get("crop_size_meters", 0.5)),
            }
        })

        if "T" in hparams and "ddim_steps" in hparams and "feat_dim" in hparams:
            from src.model.se2_diffusion_module import SE2DiffusionModule
            model = SE2DiffusionModule(**hparams)
            return model, cfg

        if "hyp_dropout_prob" in hparams or "diversity_weight" in hparams or "best_idx_noise_scale" in hparams:
            from src.model.se2_hypothesis_v2_module import SE2MultiHypothesisV2Module

            model = SE2MultiHypothesisV2Module(**hparams)
            return model, cfg

        if "assignment_temp" in hparams:
            from src.model.se2_hypothesis_module import SE2MultiHypothesisModule

            model = SE2MultiHypothesisModule(**hparams)
            return model, cfg

        raise RuntimeError(
            "Unable to reconstruct model without Hydra config. "
            f"Checkpoint hyper_parameters keys: {sorted(hparams.keys())}"
        )

    def _load_model(self):
        """Load a model from a Hydra run directory or checkpoint file."""
        config_path = self.model_path / ".hydra" / "config.yaml"
        checkpoint_path = self._find_checkpoint_path(self.model_path, self.checkpoint_override)
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        if config_path.exists():
            cfg = OmegaConf.load(config_path)
            if "model" in cfg:
                self._remap_targets_recursive(cfg.model)
            model = hydra.utils.instantiate(cfg.model)
        else:
            model, cfg = self._build_checkpoint_only_model(checkpoint)

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
        model_name = type(self.model).__name__

        if model_name == "SE2DiffusionModule":
            if method not in diffusion_methods:
                raise ValueError(
                    f"SE(2) diffusion models support sampler_method in {sorted(diffusion_methods)}, "
                    f"got '{method}'."
                )
            return
        if model_name in {"SE2MultiHypothesisModule", "SE2MultiHypothesisV2Module"}:
            raise ValueError(
                f"sampler_method is not supported for {model_name}; "
                "this model predicts hypotheses directly and does not sample via DDPM/DDIM."
            )

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
        if type(self.model).__name__ == "SE2DiffusionModule":
            return self.sampler_method or "ddim"
        if type(self.model).__name__ in {"SE2MultiHypothesisModule", "SE2MultiHypothesisV2Module"}:
            return "not_applicable"

        sampler = getattr(self.model, "sampler", None)
        if sampler is None:
            return "unknown"
        sampler_type = type(sampler).__name__

        if sampler_type == "FBODESampler":
            return getattr(sampler, "method", "unknown")
        elif sampler_type == "HFDiffusionSampler":
            return getattr(sampler, "sampler_type", "unknown")
        else:
            return "unknown"

    @staticmethod
    def _wrap_angle(theta: float) -> float:
        return float(np.arctan2(np.sin(theta), np.cos(theta)))

    @staticmethod
    def _make_coord_grid(size: int) -> np.ndarray:
        ys, xs = np.meshgrid(
            np.linspace(0, 1, size),
            np.linspace(0, 1, size),
            indexing='ij',
        )
        return np.stack([xs, ys], axis=-1).astype(np.float32)

    def _build_local_context(
        self,
        json_message,
        xml_path,
        robot_goal,
        selected_object,
        region_goals_sampled=None,
    ):
        image_converter = ImageConverter(xml_path)
        local_data = image_converter.create_local_masks(
            data_point=json_message,
            selected_object=selected_object,
            robot_goal_pos=robot_goal,
            region_goals_sampled=region_goals_sampled,
            crop_size_meters=self.local_crop_size_meters,
            highres_size=1024,
            output_size=self.local_render_size,
        )

        if not local_data or 'local_static' not in local_data:
            raise ValueError(f"Failed to generate local masks for object '{selected_object}'")

        if self.use_region_masks:
            robot_channel = local_data.get('local_robot_region')
        else:
            robot_channel = local_data.get('local_robot')
        if robot_channel is None:
            robot_channel = np.zeros_like(local_data['local_static'], dtype=np.float32)

        goal_region = local_data.get('local_goal_sample_region')
        if goal_region is None:
            goal_region = np.zeros_like(local_data['local_static'], dtype=np.float32)

        input_channels = [
            local_data['local_static'],
            local_data['local_movable'],
            local_data['local_target_object'],
            robot_channel,
            goal_region,
        ]

        if self.use_coord_grid:
            input_channels.append(self._make_coord_grid(input_channels[0].shape[0]))

        inp = np.concatenate(input_channels, axis=-1)
        inp_tensor = self.transform(inp).unsqueeze(0).to(self.device)
        inp_np = inp_tensor.detach().cpu().squeeze(0).numpy()
        return image_converter, local_data, inp_tensor, inp_np

    def infer(self, json_message, xml_path, robot_goal, selected_object, samples=32, seed=None,
              region_goals_sampled=None, episode_data=None):
        """
        Perform goal inference to get goal proposals.

        Automatically routes to the appropriate inference method based on whether
        the model was trained with local (object-centered) or global masks.

        Args:
            json_message: Raw JSON message from planning system
            xml_path: Path to MuJoCo XML file for ImageConverter
            robot_goal: Robot goal position [x, y]
            selected_object: Name of the object to generate goals for
            samples: Number of samples to generate (default: 32)
            seed: Random seed for reproducible noise (None for random)
            region_goals_sampled: Optional list of (x, y, theta) tuples representing
                                  goal samples for the target neighbor region.
                                  Used for computing goal_sample_region mask in ML inference.
            episode_data: Compatibility placeholder for older planner call sites.
                The current inference path rebuilds masks from json_message/xml_path.

        Returns:
            List of goal dictionaries, each containing:
            - index: Sample index
            - goal_center: [x, y] goal center in world coordinates (global only)
            - final_quat: Quaternion for object rotation (global only)
            - x, y, theta: SE(2) pose components
            - goal_sample: Raw goal sample array
        """
        if self.is_se2_model:
            return self._infer_se2(
                json_message,
                xml_path,
                robot_goal,
                selected_object,
                samples,
                seed=seed,
                region_goals_sampled=region_goals_sampled,
            )

        # Auto-route to local inference if model was trained with use_local=True
        if self.use_local:
            return self._infer_local(json_message, xml_path, robot_goal, selected_object, samples, seed=seed,
                                     region_goals_sampled=region_goals_sampled)

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

    def _infer_local(self, json_message, xml_path, robot_goal, selected_object, samples=32, seed=None,
                     region_goals_sampled=None):
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
            region_goals_sampled: Optional list of (x, y, theta) tuples representing
                                  goal samples for the target neighbor region.
                                  Used for computing goal_sample_region mask.

        Returns:
            List of goal dictionaries, each containing:
            - index: Sample index
            - x, y, theta: SE(2) pose in world coordinates
            - goal_sample: Raw goal sample array
            - input_channels: Input tensor for visualization
        """
        image_converter, local_data, inp_for_goal, inp_for_goal_np = self._build_local_context(
            json_message,
            xml_path,
            robot_goal,
            selected_object,
            region_goals_sampled=region_goals_sampled,
        )

        # Generate goal samples
        num_steps = self.num_steps if self.num_steps is not None else 20
        with torch.no_grad():
            goal_samples = (self.model.sample_from_model(inp_for_goal, samples=samples, num_steps=num_steps, seed=seed)
                          .permute(0, 2, 3, 1).cpu().numpy() + 1) / 2

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

    def _infer_se2(
        self,
        json_message,
        xml_path,
        robot_goal,
        selected_object,
        samples=32,
        seed=None,
        region_goals_sampled=None,
    ):
        """Infer world-frame SE(2) goals from direct delta-pose models."""
        _, local_data, inp_for_goal, inp_for_goal_np = self._build_local_context(
            json_message,
            xml_path,
            robot_goal,
            selected_object,
            region_goals_sampled=region_goals_sampled,
        )

        pre_x, pre_y = map(float, local_data['object_center'])
        pre_theta = float(local_data['object_theta'])

        if seed is not None:
            torch.manual_seed(seed)

        with torch.no_grad():
            if self.is_multihyp_model:
                pred_real, probs = self.model.predict_hypotheses(inp_for_goal)
                pred_real = pred_real[0]
                probs = probs[0]
                order = torch.argsort(probs, descending=True)
                max_outputs = min(max(1, int(samples)), pred_real.shape[0])
                order = order[:max_outputs]
                selected_deltas = pred_real[order].cpu().numpy()
                selected_weights = probs[order].cpu().numpy()
                selected_indices = order.cpu().tolist()
            else:
                ctx = self.model.encoder(inp_for_goal)
                sampled = self.model.sample(
                    ctx,
                    n_per_scene=max(1, int(samples)),
                    sampler_method=self.sampler_method or "ddim",
                    num_steps=self.num_steps,
                )[0]
                selected_deltas = sampled.cpu().numpy()
                selected_weights = np.ones(len(selected_deltas), dtype=np.float32)
                selected_indices = list(range(len(selected_deltas)))

        valid_goals = []
        for out_idx, (slot_idx, delta, vote_weight) in enumerate(zip(selected_indices, selected_deltas, selected_weights)):
            dx, dy, dtheta = map(float, delta)
            goal_theta = self._wrap_angle(pre_theta + dtheta)
            valid_goals.append({
                'index': int(slot_idx),
                'x': float(pre_x + dx),
                'y': float(pre_y + dy),
                'theta': float(goal_theta),
                'delta_x': dx,
                'delta_y': dy,
                'delta_theta': dtheta,
                'vote_weight': float(vote_weight),
                'input_channels': inp_for_goal_np,
                'anchor_pose': [pre_x, pre_y, pre_theta],
                'prediction_rank': int(out_idx),
            })

        return valid_goals

    def warmup(self, samples: int = 32, num_steps: int | None = None,
               seed: int | None = None, repeats: int = 3) -> None:
        """Run dummy inference passes to compile kernels without env inputs."""
        if repeats <= 0:
            return

        channels = 5 + (2 if self.use_coord_grid else 0)
        dummy_context = torch.zeros(
            1,
            channels,
            self.context_size,
            self.context_size,
            device=self.device,
        )
        warmup_steps = num_steps if num_steps is not None else self.num_steps

        with torch.no_grad():
            for rep in range(int(repeats)):
                if seed is not None:
                    torch.manual_seed(int(seed) + rep)

                if self.is_diffusion_se2_model:
                    ctx = self.model.encoder(dummy_context)
                    _ = self.model.sample(
                        ctx,
                        n_per_scene=max(1, int(samples)),
                        sampler_method=self.sampler_method or "ddim",
                        num_steps=warmup_steps,
                    )
                elif self.is_multihyp_model:
                    _ = self.model.predict_hypotheses(dummy_context)
                else:
                    _ = self.model.sample_from_model(
                        dummy_context,
                        samples=max(1, int(samples)),
                        num_steps=warmup_steps if warmup_steps is not None else 20,
                        seed=None if seed is None else int(seed) + rep,
                    )

        if isinstance(self.device, str) and self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
