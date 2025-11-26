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
    
    def __init__(self, model_path, device="cuda"):
        """
        Initialize the goal inference model.
        
        Args:
            model_path: Path to model output directory (e.g., outputs/rel_reach_coord_goal_dit/mse/2025-08-10_06-59-27)
            device: Device to load model on (default: "cuda")
        """
        self.device = device
        self.model_path = Path(model_path)
        
        # Load model
        self.model, self.cfg = self._load_model()
        print(f"✅ Goal model loaded successfully: {type(self.model).__name__}")
        
        # Use data config
        self.data_cfg = self.cfg.data
        
        # Setup image transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.data_cfg.image_size, self.data_cfg.image_size)),
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
    
    def infer(self, json_message, xml_path, robot_goal, selected_object, samples=32):
        """
        Perform goal inference to get goal proposals.
        
        Args:
            json_message: Raw JSON message from planning system
            xml_path: Path to MuJoCo XML file for ImageConverter
            robot_goal: Robot goal position [x, y]
            selected_object: Name of the object to generate goals for
            samples: Number of samples to generate (default: 32)
            
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
        inp_data = image_converter.process_datapoint(json_message, robot_goal)

        # Create object mask for the selected object
        try:
            selected_object_mask = image_converter.create_object_mask(selected_object)
        except Exception as e:
            raise ValueError(f"Error creating object mask for '{selected_object}': {e}")
        
        # Prepare input for goal model (stack scene context + selected object mask)
        inp_for_goal = np.concatenate([
            inp_data['robot_image'],
            inp_data['goal_image'],
            inp_data['movable_objects_image'],
            inp_data['static_objects_image'],
            selected_object_mask                   # Selected object mask (channel 5)
        ], axis=-1)

        inp_for_goal = self.transform(inp_for_goal).unsqueeze(0).to(self.device)

        # Generate goal samples
        with torch.no_grad():
            goal_samples = (self.model.sample_from_model(inp_for_goal, samples=samples)
                          .permute(0, 2, 3, 1).cpu().numpy() + 1) / 2

        inp_for_goal = inp_for_goal.cpu().squeeze(0).numpy()

        # Process goal samples and extract SE(2) poses
        valid_goals = []
        scale = image_converter.IMG_SIZE / self.data_cfg.image_size
        
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