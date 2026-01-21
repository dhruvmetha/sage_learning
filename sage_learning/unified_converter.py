#!/usr/bin/env python3
"""
Unified Image Converter for NAMO environments.

This converter provides a standardized way to convert NAMO environment states to images
compatible with ML models. It is based on the approach used in the mask generation visualizer
to ensure consistency across data collection, training, and inference.

The converter handles:
- World-to-pixel coordinate conversion with proper centering and scaling
- Drawing rotated rectangles for objects
- Drawing circles for robots and goals
- Creating multi-channel images for ML models

Key Features:
- Consistent coordinate system across all use cases
- Proper handling of half-extents (no size doubling)
- Support for both mask generation and image creation
- Compatible with existing ML model expectations
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class ObjectInfo:
    """Information about an object in the environment."""
    name: str
    x: float
    y: float
    theta: float
    size_x: float  # Half-extent in X
    size_y: float  # Half-extent in Y
    is_static: bool = False
    is_reachable: bool = False


class UnifiedImageConverter:
    """Unified image converter for NAMO environments.
    
    This converter standardizes image generation across data collection,
    training, and ML inference to ensure consistency.
    """
    
    IMG_SIZE = 224
    
    def __init__(self, world_bounds: Tuple[float, float, float, float]):
        """Initialize the unified image converter.
        
        Args:
            world_bounds: (x_min, x_max, y_min, y_max) world coordinate bounds
        """
        self.world_bounds = world_bounds
        self.x_min, self.x_max, self.y_min, self.y_max = world_bounds
        
        # Calculate world dimensions and scale (following visualizer approach)
        world_width = self.x_max - self.x_min
        world_height = self.y_max - self.y_min
        self.world_size = max(world_width, world_height)  # Use larger dimension for square images
        self.scale = self.IMG_SIZE / self.world_size  # pixels per world unit
        
        # Precompute world center for efficiency
        self.world_center_x = (self.x_min + self.x_max) / 2
        self.world_center_y = (self.y_min + self.y_max) / 2
        self.img_center = self.IMG_SIZE / 2
    
    def world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to pixel coordinates.
        
        This follows the exact approach from the visualizer for consistency.
        
        Args:
            x, y: World coordinates
            
        Returns:
            Tuple of (pixel_x, pixel_y)
        """
        # Translate to center and scale (from visualizer)
        pixel_x = int((x - self.world_center_x) * self.scale + self.img_center)
        pixel_y = int((y - self.world_center_y) * self.scale + self.img_center)
        
        # Clamp to image bounds
        pixel_x = max(0, min(self.IMG_SIZE - 1, pixel_x))
        pixel_y = max(0, min(self.IMG_SIZE - 1, pixel_y))
        
        return pixel_x, pixel_y
    
    def pixel_to_world(self, px: int, py: int) -> Tuple[float, float]:
        """Convert pixel coordinates to world coordinates.
        
        Args:
            px, py: Pixel coordinates
            
        Returns:
            Tuple of (world_x, world_y)
        """
        x = (px - self.img_center) / self.scale + self.world_center_x
        y = (py - self.img_center) / self.scale + self.world_center_y
        return x, y
    
    def draw_circle(self, image: np.ndarray, center_x: float, center_y: float, 
                   radius: float, value: float = 1.0) -> None:
        """Draw a filled circle on the image.
        
        Args:
            image: Image array to draw on (will be modified in-place)
            center_x, center_y: World coordinates of circle center
            radius: Circle radius in world units
            value: Pixel value to set (0.0 to 1.0 for masks, 255 for uint8 images)
        """
        center_px, center_py = self.world_to_pixel(center_x, center_y)
        radius_px = max(1, int(radius * self.scale))
        
        cv2.circle(image, (center_px, center_py), radius_px, value, -1)
    
    def draw_rotated_box(self, image: np.ndarray, center_x: float, center_y: float,
                        half_width: float, half_height: float, angle_rad: float, 
                        value: float = 1.0) -> Tuple[int, int]:
        """Draw a filled rotated rectangle on the image using cv2.
        
        This follows the exact approach from the visualizer.
        
        Args:
            image: Image array to draw on (will be modified in-place)
            center_x, center_y: World coordinates of box center
            half_width, half_height: Half-extents in world units
            angle_rad: Rotation angle in radians
            value: Pixel value to set (0.0 to 1.0 for masks, 255 for uint8 images)
            
        Returns:
            Tuple of (center_px, center_py) in pixel coordinates
        """
        # Convert center to pixel coordinates
        center_px, center_py = self.world_to_pixel(center_x, center_y)
        
        # Convert size to pixel coordinates (multiply by 2 for full width/height)
        size_px = (int(half_width * 2 * self.scale), int(half_height * 2 * self.scale))
        
        # Create rotated rectangle
        angle_deg = np.degrees(angle_rad)
        rect = ((center_px, center_py), size_px, angle_deg)
        
        # Get box points and draw filled polygon
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        # Handle different image types
        if image.dtype == np.float32:
            # For float masks, convert temporarily to uint8
            image_uint8 = (image * 255).astype(np.uint8)
            cv2.fillPoly(image_uint8, [box], int(value * 255))
            image[:] = image_uint8.astype(np.float32) / 255.0
        else:
            # For uint8 images, draw directly
            cv2.fillPoly(image, [box], int(value))
        
        return center_px, center_py
    
    def create_multi_channel_image(self, 
                                  robot_pos: Tuple[float, float, float],
                                  robot_goal: Tuple[float, float, float],
                                  objects: List[ObjectInfo]) -> np.ndarray:
        """Create a multi-channel image compatible with ML models.
        
        This creates the same 5-channel format used in training:
        - Channel 0: Robot position
        - Channel 1: Robot goal position  
        - Channel 2: Movable objects
        - Channel 3: Static objects
        - Channel 4: Reachable objects
        
        Args:
            robot_pos: Robot position (x, y, theta)
            robot_goal: Robot goal position (x, y, theta)
            objects: List of ObjectInfo with object data
            
        Returns:
            Multi-channel image array with shape (5, height, width)
        """
        # Initialize channels
        channels = np.zeros((5, self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32)
        
        # Channel 0: Robot position
        self.draw_circle(channels[0], robot_pos[0], robot_pos[1], 0.2, 1.0)
        
        # Channel 1: Robot goal
        self.draw_circle(channels[1], robot_goal[0], robot_goal[1], 0.25, 1.0)
        
        # Process objects
        for obj in objects:
            if obj.is_static:
                # Channel 3: Static objects
                self.draw_rotated_box(channels[3], obj.x, obj.y, 
                                    obj.size_x, obj.size_y, obj.theta, 1.0)
            else:
                # Channel 2: Movable objects
                self.draw_rotated_box(channels[2], obj.x, obj.y,
                                    obj.size_x, obj.size_y, obj.theta, 1.0)
                
                # Channel 4: Reachable objects (subset of movable)
                if obj.is_reachable:
                    self.draw_rotated_box(channels[4], obj.x, obj.y,
                                        obj.size_x, obj.size_y, obj.theta, 1.0)
        
        return channels
    
    def create_single_mask(self, objects: List[ObjectInfo], 
                          mask_type: str = "all") -> np.ndarray:
        """Create a single mask for specific object types.
        
        Args:
            objects: List of ObjectInfo 
            mask_type: Type of mask - "static", "movable", "reachable", or "all"
            
        Returns:
            Single channel mask with shape (height, width)
        """
        mask = np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32)
        
        for obj in objects:
            should_draw = False
            
            if mask_type == "all":
                should_draw = True
            elif mask_type == "static" and obj.is_static:
                should_draw = True
            elif mask_type == "movable" and not obj.is_static:
                should_draw = True
            elif mask_type == "reachable" and obj.is_reachable:
                should_draw = True
            
            if should_draw:
                self.draw_rotated_box(mask, obj.x, obj.y,
                                    obj.size_x, obj.size_y, obj.theta, 1.0)
        
        return mask
    
    def get_object_center_pixels(self, objects: List[ObjectInfo]) -> Dict[str, Tuple[int, int]]:
        """Get pixel coordinates for object centers.
        
        Args:
            objects: List of ObjectInfo
            
        Returns:
            Dictionary mapping object names to (pixel_x, pixel_y) coordinates
        """
        obj_centers = {}
        for obj in objects:
            center_px, center_py = self.world_to_pixel(obj.x, obj.y)
            obj_centers[obj.name] = (center_px, center_py)
        return obj_centers


def create_converter_from_episode(episode_data: Dict[str, Any]) -> UnifiedImageConverter:
    """Create a UnifiedImageConverter from episode data.
    
    This is a convenience function that extracts world bounds from episode data
    using the same logic as the visualizer.
    
    Args:
        episode_data: Episode data from pickle file
        
    Returns:
        UnifiedImageConverter instance
    """
    static_object_info = episode_data.get('static_object_info') or {}
    state_observations = episode_data.get('state_observations', [])
    robot_goal = episode_data.get('robot_goal', (0.0, 0.0, 0.0))
    
    # Calculate world bounds (following visualizer logic)
    x_coords = []
    y_coords = []
    
    # Add static object bounds
    for obj_name, obj_info in static_object_info.items():
        if 'pos_x' in obj_info and 'pos_y' in obj_info:  # Static object
            x_coords.extend([obj_info['pos_x'] - obj_info['size_x'], 
                           obj_info['pos_x'] + obj_info['size_x']])
            y_coords.extend([obj_info['pos_y'] - obj_info['size_y'], 
                           obj_info['pos_y'] + obj_info['size_y']])
    
    # Add robot positions
    robot_start = (0.0, 0.0, 0.0)
    if state_observations and len(state_observations) > 0:
        first_state = state_observations[0]
        if 'robot_pose' in first_state:
            robot_start = first_state['robot_pose']
    
    x_coords.extend([robot_start[0], robot_goal[0]])
    y_coords.extend([robot_start[1], robot_goal[1]])
    
    # Add movable object positions from state observations
    if state_observations:
        for state in state_observations:
            for obj_name, pose in state.items():
                if obj_name != 'robot_pose' and len(pose) >= 2:
                    # Find object size for bounds
                    obj_base_name = obj_name.replace('_pose', '')
                    obj_info = static_object_info.get(obj_base_name, {})
                    size_x = obj_info.get('size_x', 0.5)
                    size_y = obj_info.get('size_y', 0.5)
                    
                    x_coords.extend([pose[0] - size_x, pose[0] + size_x])
                    y_coords.extend([pose[1] - size_y, pose[1] + size_y])
    
    if x_coords and y_coords:
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding (following visualizer approach)
        x_range = x_max - x_min
        y_range = y_max - y_min
        padding_x = min(x_range * 0.1, 0.2)
        padding_y = min(y_range * 0.1, 0.2)
        
        world_bounds = (x_min - padding_x, x_max + padding_x, 
                       y_min - padding_y, y_max + padding_y)
    else:
        world_bounds = (-5.5, 5.5, -5.5, 5.5)
    
    return UnifiedImageConverter(world_bounds)


def create_converter_from_xml(xml_path: str) -> UnifiedImageConverter:
    """Create a UnifiedImageConverter from MuJoCo XML file.
    
    This extracts world bounds from the XML file geometry.
    
    Args:
        xml_path: Path to MuJoCo XML file
        
    Returns:
        UnifiedImageConverter instance
    """
    import mujoco
    import os
    
    # Handle relative paths by prepending the ML4KP resources path
    if not os.path.isabs(xml_path):
        # Check if the path starts with ../ and we can resolve it properly
        if xml_path.startswith("../ml4kp_ktamp"):
            # It's a relative path from the namo/python dir that has been passed as is.
            # We need to resolve it relative to the current working directory or repo root
            # But wait, `xml_path` here comes from `env.get_xml_path()`, which might already be a relative path that works from the repo root.
            # Let's try to resolve it directly first.
            if os.path.exists(xml_path):
                full_xml_path = os.path.abspath(xml_path)
            else:
                # Fallback: try prepending ML4KP resources path (only if not already starting with ..)
                full_xml_path = os.path.join("/common/home/dm1487/robotics_research/ktamp/ml4kp_ktamp/resources/models", xml_path)
        elif os.path.exists(xml_path):
             full_xml_path = os.path.abspath(xml_path)
        else:
            full_xml_path = os.path.join("/common/home/dm1487/robotics_research/ktamp/ml4kp_ktamp/resources/models", xml_path)
    else:
        full_xml_path = xml_path
    
    model = mujoco.MjModel.from_xml_path(full_xml_path)
    
    # Calculate bounds from geometry (following learning package approach)
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    
    for i in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name is None:
            geom_name = f"geom_{i}"
        
        geom_pos = model.geom_pos[i]
        geom_size = model.geom_size[i]
        
        if geom_name.startswith('wall') or 'static' in geom_name or 'movable' in geom_name:
            geom_x_min = geom_pos[0] - geom_size[0]
            geom_x_max = geom_pos[0] + geom_size[0]
            geom_y_min = geom_pos[1] - geom_size[1]
            geom_y_max = geom_pos[1] + geom_size[1]
            
            x_min = min(x_min, geom_x_min)
            x_max = max(x_max, geom_x_max)
            y_min = min(y_min, geom_y_min)
            y_max = max(y_max, geom_y_max)
    
    # Handle edge case where no bounds were found
    if x_min == float('inf'):
        x_min, x_max, y_min, y_max = -3.0, 3.0, -3.0, 3.0
    
    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding_x = min(x_range * 0.1, 0.2)
    padding_y = min(y_range * 0.1, 0.2)
    
    world_bounds = (x_min - padding_x, x_max + padding_x, 
                   y_min - padding_y, y_max + padding_y)
    
    del model
    return UnifiedImageConverter(world_bounds)