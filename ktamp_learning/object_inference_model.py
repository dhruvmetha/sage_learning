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
from collections import Counter


class ObjectInferenceModel:
    """
    Model for performing object selection inference using a trained diffusion model.
    Processes environment data and returns object selection results with vote distribution.
    """
    
    def __init__(self, model_path, device="cuda"):
        """
        Initialize the object inference model.
        
        Args:
            model_path: Path to model output directory (e.g., outputs/rel_reach_coord_object_dit/mse/2025-08-10_05-33-43)
            device: Device to load model on (default: "cuda")
        """
        self.device = device
        self.model_path = Path(model_path)
        
        # Load model
        self.model, self.cfg = self._load_model()
        print(f"✅ Object model loaded successfully: {type(self.model).__name__}")
        
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
    
    def infer(self, json_message, xml_path, robot_goal, samples=32):
        """
        Perform object inference to get object distribution.
        
        Args:
            json_message: Raw JSON message from planning system
            xml_path: Path to MuJoCo XML file for ImageConverter
            robot_goal: Robot goal position [x, y]
            samples: Number of samples to generate (default: 32)
            
        Returns:
            Dictionary with:
            - object_votes: Counter with object names and vote counts
            - selected_object: Object with most votes
            - total_valid_samples: Number of valid samples processed
            - reachable_selections: Number of times a reachable object was selected
            - inp_data: Processed image data for later use
            - image_converter: ImageConverter instance for later use
        """
        # Create ImageConverter and process data
        image_converter = ImageConverter(xml_path)
        inp_data = image_converter.process_datapoint(json_message, robot_goal)
        obj2center_px = inp_data['obj2center_px']
        reachable_objects_list = json_message.get('reachable_objects', [])
        
        # Prepare input for obstacle model
        inp_for_obstacle = np.concatenate([
            inp_data['robot_image'], 
            inp_data['goal_image'], 
            inp_data['movable_objects_image'], 
            inp_data['static_objects_image'], 
            inp_data['reachable_objects_image']
        ], axis=-1)
        
        inp_for_obstacle = self.transform(inp_for_obstacle).unsqueeze(0).to(self.device)
        
        # Generate obstacle/object samples
        with torch.no_grad():
            obstacle_samples = (self.model.sample_from_model(inp_for_obstacle, samples=samples)
                              .permute(0, 2, 3, 1).cpu().numpy() + 1) / 2
        
        obj_votes = Counter()
        valid_samples = 0
        reachable_selections = 0
        
        # Process obstacle samples to vote for best object
        for i in range(obstacle_samples.shape[0]):
            # Extract object mask from obstacle model output
            object_mask = (obstacle_samples[i][:, :, 0].copy() > 0.5) * 1.0
            object_mask = object_mask.astype(np.uint8)
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(object_mask)
            
            if num_labels > 2:
                continue
                
            valid_samples += 1
            
            _, _, predicted_obj_center, obj_angle = find_rectangle_corners(object_mask)
            if predicted_obj_center is None:
                continue
                
            # Scale predicted center to match original image size
            scale = image_converter.IMG_SIZE / self.data_cfg.image_size
            predicted_obj_center = (int(predicted_obj_center[0] * scale), 
                                  int(predicted_obj_center[1] * scale))
            
            # Find closest object
            min_dist = float('inf')
            min_obj_name = None
            
            dist = lambda p1, p2: np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            
            for obj_name, obj_center in obj2center_px.items():
                d = dist(obj_center, predicted_obj_center)
                if d < min_dist:
                    min_dist = d
                    min_obj_name = obj_name
            
            if min_obj_name is not None:
                obj_votes[min_obj_name] += 1
                
                if reachable_objects_list and min_obj_name in reachable_objects_list:
                    reachable_selections += 1
        
        # Find the object with the most votes
        selected_object = obj_votes.most_common(1)[0][0] if obj_votes else None
        
        return {
            'object_votes': obj_votes,
            'selected_object': selected_object,
            'total_valid_samples': valid_samples,
            'reachable_selections': reachable_selections,
            'inp_data': inp_data,
            'image_converter': image_converter
        }