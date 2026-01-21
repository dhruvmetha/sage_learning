#!/usr/bin/env python3
"""
NAMO Data Visualizer - Visualization tool for collected NAMO planning data.

This tool creates visualizations of NAMO environments from collected data files,
extracting static environment information from collected data and overlaying trajectories
and robot states from the planning episodes.

Expected Data Structure:
-----------------------

The tool expects pickle files with the following structure:

{
    "task_id": str,                          # e.g., "ilab1_env_000000"
    "success": bool,                         # Task-level success
    "episodes_collected": int,               # Number of episodes in this task
    "processing_time": float,                # Total processing time
    "episode_results": [                     # List of episode results
        {
            "episode_id": str,               # e.g., "ilab1_env_000000_episode_0"
            "algorithm": str,                # e.g., "idfs"
            "algorithm_version": str,        # Algorithm version
            "success": bool,                 # Episode success
            "solution_found": bool,          # Whether solution was found
            "solution_depth": int | None,    # Solution depth (None if failed)
            "search_time_ms": float | None,  # Search time in milliseconds
            "nodes_expanded": int | None,    # Number of nodes expanded
            "terminal_checks": int | None,   # Number of terminal state checks
            "max_depth_reached": int | None, # Maximum search depth reached
            "error_message": str,            # Error message (if any)
            "xml_file": str,                 # Path to original XML environment
            "robot_goal": tuple,             # Robot goal (x, y, theta)
            
            # Action sequence (if solution found)
            "action_sequence": [
                {
                    "object_id": str,        # e.g., "obstacle_3_movable"
                    "target": tuple          # Target pose (x, y, theta)
                }
            ] | None,
            
            # State observations - SE(2) poses before each action
            "state_observations": [
                {
                    "robot_pose": [float, float, float],                    # [x, y, theta]
                    "obstacle_1_movable_pose": [float, float, float],       # [x, y, theta]
                    "obstacle_2_movable_pose": [float, float, float],       # [x, y, theta]
                    # ... more movable objects
                }
            ] | None,
            
            # Static object information (sizes, positions for walls/static objects)
            "static_object_info": {
                # Movable objects (only size info)
                "obstacle_1_movable": {
                    "size_x": float,         # Half-extent in x direction
                    "size_y": float,         # Half-extent in y direction  
                    "size_z": float          # Half-extent in z direction
                },
                
                # Static objects (walls) - full pose info
                "wall_1": {
                    "pos_x": float, "pos_y": float, "pos_z": float,
                    "quat_w": float, "quat_x": float, "quat_y": float, "quat_z": float,
                    "size_x": float, "size_y": float, "size_z": float
                },
                
                # Robot info
                "robot": {
                    "size_x": float, "size_y": float, "size_z": float
                }
            } | None,
            
            # Algorithm-specific statistics
            "algorithm_stats": dict | None   # Additional algorithm metrics
        }
    ]
}

Notes:
- Coordinates are in world frame (typically meters)
- Rotations are in radians
- size_x/y/z represent half-extents (half-width, half-height, half-depth)
- static_object_info may be None for failed episodes
- state_observations contains poses before each action execution
- Movable object poses come from state_observations, static objects from static_object_info
"""

import os
import sys
import pickle
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.transforms import Affine2D
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import glob
from dataclasses import dataclass


@dataclass
class StaticObject:
    """Static object in the environment."""
    name: str
    x: float
    y: float
    z: float
    size_x: float
    size_y: float
    size_z: float
    quat_w: float = 1.0
    quat_x: float = 0.0
    quat_y: float = 0.0
    quat_z: float = 0.0


@dataclass
class MovableObject:
    """Movable object in the environment."""
    name: str
    size_x: float
    size_y: float
    size_z: float


@dataclass
class EnvironmentInfo:
    """Complete environment information."""
    static_objects: List[StaticObject]
    movable_objects: List[MovableObject]
    robot_start: Tuple[float, float, float]
    robot_goal: Tuple[float, float, float]
    world_bounds: Tuple[float, float, float, float]  # x_min, x_max, y_min, y_max


class NAMOXMLParser:
    """Parser for NAMO environment XML files."""
    
    def __init__(self, xml_file: str):
        self.xml_file = xml_file
        self.tree = ET.parse(xml_file)
        self.root = self.tree.getroot()
    
    def parse_environment(self) -> EnvironmentInfo:
        """Parse XML file and extract environment information."""
        static_objects = []
        movable_objects = []
        robot_start = (0.0, 0.0, 0.0)
        robot_goal = (0.0, 0.0, 0.0)
        
        # Parse static objects (geoms in worldbody)
        worldbody = self.root.find('.//worldbody')
        if worldbody is not None:
            for geom in worldbody.findall('.//geom'):
                name = geom.get('name', 'unnamed')
                if name.startswith('wall') or name.startswith('obstacle'):
                    pos_str = geom.get('pos', '0 0 0')
                    size_str = geom.get('size', '1 1 1')
                    quat_str = geom.get('quat', '1 0 0 0')
                    
                    pos = [float(x) for x in pos_str.split()]
                    size = [float(x) for x in size_str.split()]
                    quat = [float(x) for x in quat_str.split()]
                    
                    static_obj = StaticObject(
                        name=name,
                        x=pos[0], y=pos[1], z=pos[2] if len(pos) > 2 else 0.0,
                        size_x=size[0], size_y=size[1], size_z=size[2] if len(size) > 2 else 1.0,
                        quat_w=quat[0], quat_x=quat[1], quat_y=quat[2], quat_z=quat[3] if len(quat) > 3 else 0.0
                    )
                    static_objects.append(static_obj)
        
        # Parse movable objects (bodies with freejoint)
        for body in self.root.findall('.//body'):
            if body.find('freejoint') is not None:
                name = body.get('name', 'unnamed')
                geom = body.find('geom')
                if geom is not None:
                    size_str = geom.get('size', '0.1 0.1 0.1')
                    size = [float(x) for x in size_str.split()]
                    
                    movable_obj = MovableObject(
                        name=name,
                        size_x=size[0], size_y=size[1], size_z=size[2] if len(size) > 2 else 0.1
                    )
                    movable_objects.append(movable_obj)
        
        # Parse robot start position
        robot_body = self.root.find('.//body[@name="robot"]')
        if robot_body is not None:
            pos_str = robot_body.get('pos', '0 0 0')
            pos = [float(x) for x in pos_str.split()]
            robot_start = (pos[0], pos[1], 0.0)
        
        # Parse goal position from site
        goal_site = self.root.find('.//site[@name="goal"]')
        if goal_site is not None:
            pos_str = goal_site.get('pos', '0 0 0')
            pos = [float(x) for x in pos_str.split()]
            robot_goal = (pos[0], pos[1], 0.0)
        
        # Calculate world bounds based on static objects
        if static_objects:
            x_coords = []
            y_coords = []
            for obj in static_objects:
                x_coords.extend([obj.x - obj.size_x, obj.x + obj.size_x])
                y_coords.extend([obj.y - obj.size_y, obj.y + obj.size_y])
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Use exact bounds without any padding
            world_bounds = (x_min, x_max, y_min, y_max)
        else:
            # Default bounds
            world_bounds = (-3.0, 3.0, -3.0, 3.0)
        
        return EnvironmentInfo(
            static_objects=static_objects,
            movable_objects=movable_objects,
            robot_start=robot_start,
            robot_goal=robot_goal,
            world_bounds=world_bounds
        )


class NAMODataVisualizer:
    """Visualizer for NAMO planning data."""

    IMG_SIZE = 224  # Mask size for image-based representations

    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
    
    def _extract_env_info_from_episode(self, episode_data: Dict[str, Any]) -> EnvironmentInfo:
        """Extract environment information from episode data."""
        static_object_info = episode_data.get('static_object_info') or {}
        state_observations = episode_data.get('state_observations', [])
        
        static_objects = []
        movable_objects = []
        
        # Extract static objects from static_object_info
        for obj_name, info in static_object_info.items():
            if 'pos_x' in info and 'pos_y' in info:  # Static object with position
                static_obj = StaticObject(
                    name=obj_name,
                    x=info['pos_x'],
                    y=info['pos_y'],
                    z=info.get('pos_z', 0.0),
                    size_x=info['size_x'],
                    size_y=info['size_y'], 
                    size_z=info.get('size_z', 0.3),
                    quat_w=info.get('quat_w', 1.0),
                    quat_x=info.get('quat_x', 0.0),
                    quat_y=info.get('quat_y', 0.0),
                    quat_z=info.get('quat_z', 0.0)
                )
                static_objects.append(static_obj)
            elif 'size_x' in info and 'size_y' in info:  # Movable object with just size
                movable_obj = MovableObject(
                    name=obj_name,
                    size_x=info['size_x'],
                    size_y=info['size_y'],
                    size_z=info.get('size_z', 0.3)
                )
                movable_objects.append(movable_obj)
        
        # Get robot start and goal
        robot_start = (0.0, 0.0, 0.0)
        if state_observations and len(state_observations) > 0:
            first_state = state_observations[0]
            if 'robot_pose' in first_state:
                robot_pose = first_state['robot_pose']
                robot_start = (robot_pose[0], robot_pose[1], robot_pose[2])
        
        robot_goal = episode_data.get('robot_goal', (0.0, 0.0, 0.0))
        
        # Calculate world bounds based on static objects and observations
        x_coords = []
        y_coords = []
        
        # Add static object bounds
        for obj in static_objects:
            x_coords.extend([obj.x - obj.size_x, obj.x + obj.size_x])
            y_coords.extend([obj.y - obj.size_y, obj.y + obj.size_y])
        
        # Add robot positions
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
            
            # Add padding (10% of range with minimum 0.2 units, matching reference implementation)
            x_range = x_max - x_min
            y_range = y_max - y_min
            padding_x = min(x_range * 0.1, 0.2)
            padding_y = min(y_range * 0.1, 0.2)
            
            world_bounds = (x_min - padding_x, x_max + padding_x, 
                           y_min - padding_y, y_max + padding_y)
        else:
            world_bounds = (-5.5, 5.5, -5.5, 5.5)
        
        return EnvironmentInfo(
            static_objects=static_objects,
            movable_objects=movable_objects,
            robot_start=robot_start,
            robot_goal=robot_goal,
            world_bounds=world_bounds
        )
    
    def _world_to_pixel(self, x: float, y: float, world_bounds: Tuple[float, float, float, float]) -> Tuple[int, int]:
        """Convert world coordinates to pixel coordinates for masks.
        
        Args:
            x, y: World coordinates
            world_bounds: (x_min, x_max, y_min, y_max)
            
        Returns:
            Tuple of (pixel_x, pixel_y)
        """
        x_min, x_max, y_min, y_max = world_bounds
        
        # Calculate world dimensions (matching reference implementation approach)
        world_width = x_max - x_min
        world_height = y_max - y_min
        # Use the larger dimension to maintain square images (from reference)
        world_size = max(world_width, world_height)
        scale = self.IMG_SIZE / world_size  # pixels per world unit
        
        # Center the world bounds in the image (from reference implementation)
        world_center_x = (x_min + x_max) / 2
        world_center_y = (y_min + y_max) / 2
        img_center = self.IMG_SIZE / 2
        
        # Translate to center and scale (from reference)
        pixel_x = int((x - world_center_x) * scale + img_center)
        pixel_y = int((y - world_center_y) * scale + img_center)
        
        # Clamp to image bounds
        pixel_x = max(0, min(self.IMG_SIZE - 1, pixel_x))
        pixel_y = max(0, min(self.IMG_SIZE - 1, pixel_y))
        
        return pixel_x, pixel_y
    
    def _get_pixel_scale(self, world_bounds: Tuple[float, float, float, float]) -> float:
        """Get the pixel scale for world to pixel conversion."""
        x_min, x_max, y_min, y_max = world_bounds
        world_width = x_max - x_min
        world_height = y_max - y_min
        world_size = max(world_width, world_height)
        return self.IMG_SIZE / world_size
    
    
    def _draw_rotated_box_mask(self, mask: np.ndarray, center_x: float, center_y: float,
                              half_width: float, half_height: float, angle_rad: float,
                              world_bounds: Tuple[float, float, float, float], value: float = 1.0) -> None:
        """Draw a filled rotated rectangle on the mask using cv2.
        
        Args:
            mask: Mask array to draw on (will be modified in-place)
            center_x, center_y: World coordinates of box center
            half_width, half_height: Half-extents in world units
            angle_rad: Rotation angle in radians
            world_bounds: World coordinate bounds
            value: Pixel value to set (0.0 to 1.0)
        """
        # Convert center to pixel coordinates
        center_px, center_py = self._world_to_pixel(center_x, center_y, world_bounds)
        
        # Convert size to pixel coordinates
        scale = self._get_pixel_scale(world_bounds)
        size_px = (int(half_width * 2 * scale), int(half_height * 2 * scale))
        
        # Create rotated rectangle
        angle_deg = np.degrees(angle_rad)
        rect = ((center_px, center_py), size_px, angle_deg)
        
        # Get box points and draw filled polygon
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        # Convert mask to uint8 for cv2, draw, then convert back
        mask_uint8 = (mask * 255).astype(np.uint8)
        cv2.fillPoly(mask_uint8, [box], int(value * 255))
        
        # Convert back to float and update original mask
        mask[:] = mask_uint8.astype(np.float32) / 255.0
    
    def _draw_circle_mask(self, mask: np.ndarray, center_x: float, center_y: float, 
                         radius: float, world_bounds: Tuple[float, float, float, float], value: float = 1.0) -> None:
        """Draw a filled circle on the mask using cv2.
        
        Args:
            mask: Mask array to draw on (will be modified in-place)
            center_x, center_y: World coordinates of circle center
            radius: Circle radius in world units
            world_bounds: World coordinate bounds
            value: Pixel value to set (0.0 to 1.0)
        """
        # Convert center to pixel coordinates
        center_px, center_py = self._world_to_pixel(center_x, center_y, world_bounds)
        
        # Convert radius to pixel coordinates
        scale = self._get_pixel_scale(world_bounds)
        radius_px = int(radius * scale)
        
        # Convert mask to uint8 for cv2, draw, then convert back
        mask_uint8 = (mask * 255).astype(np.uint8)
        cv2.circle(mask_uint8, (center_px, center_py), radius_px, int(value * 255), -1)
        
        # Convert back to float and update original mask
        mask[:] = mask_uint8.astype(np.float32) / 255.0
    
    def _inflate_mask(self, mask: np.ndarray, radius_m: float, world_bounds: Tuple[float, float, float, float]) -> np.ndarray:
        """Inflate/dilate a mask by a given radius to account for robot size.
        
        Args:
            mask: Binary mask to inflate
            radius_m: Radius to inflate by in world units (meters)
            world_bounds: World coordinate bounds for scale calculation
            
        Returns:
            Inflated mask
        """
        # Calculate radius in pixels
        scale = self._get_pixel_scale(world_bounds)
        radius_px = max(1, int(radius_m * scale))
        
        # Create circular kernel for inflation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius_px + 1, 2*radius_px + 1))
        
        # Dilate the mask
        inflated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        
        return inflated_mask.astype(np.float32)
    
    def _compute_distance_field(self, start_x: float, start_y: float, 
                               static_mask: np.ndarray, movable_mask: np.ndarray,
                               world_bounds: Tuple[float, float, float, float]) -> np.ndarray:
        """Compute distance field using wavefront propagation with different costs.
        
        Args:
            start_x, start_y: Starting position in world coordinates
            static_mask: Binary mask of static obstacles (inflated)
            movable_mask: Binary mask of movable objects (inflated)
            world_bounds: World coordinate bounds
            
        Returns:
            Distance field where:
            - Static obstacles: -1
            - Start position: 0
            - Free space: distance with cost 1 per cell
            - Movable objects: distance with cost 2 per cell
            - Normalized to [0, 1] for non-negative values
        """
        import heapq
        
        # Initialize distance field
        dist_field = np.full((self.IMG_SIZE, self.IMG_SIZE), np.inf, dtype=np.float32)
        
        # Convert start position to pixel coordinates
        start_px, start_py = self._world_to_pixel(start_x, start_y, world_bounds)
        
        # Mark static obstacles as impassable (-1)
        dist_field[static_mask > 0.5] = -1
        
        # Set start position to 0
        if 0 <= start_px < self.IMG_SIZE and 0 <= start_py < self.IMG_SIZE:
            dist_field[start_py, start_px] = 0
        else:
            # Start position is outside bounds, return empty field
            dist_field[dist_field == np.inf] = -1
            return dist_field
        
        # Priority queue for Dijkstra-like propagation: (cost, row, col)
        pq = [(0, start_py, start_px)]
        visited = np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=bool)
        
        # 4-connectivity neighbors
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while pq:
            current_cost, row, col = heapq.heappop(pq)
            
            # Skip if already visited
            if visited[row, col]:
                continue
            
            visited[row, col] = True
            
            # Check all neighbors
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                # Check bounds
                if not (0 <= new_row < self.IMG_SIZE and 0 <= new_col < self.IMG_SIZE):
                    continue
                
                # Skip static obstacles
                if dist_field[new_row, new_col] == -1:
                    continue
                
                # Skip if already visited
                if visited[new_row, new_col]:
                    continue
                
                # Calculate movement cost
                if movable_mask[new_row, new_col] > 0.5:
                    # Moving through movable object costs 4
                    movement_cost = 4
                else:
                    # Moving through free space costs 1
                    movement_cost = 1
                
                new_cost = current_cost + movement_cost
                
                # Update if we found a better path
                if new_cost < dist_field[new_row, new_col]:
                    dist_field[new_row, new_col] = new_cost
                    heapq.heappush(pq, (new_cost, new_row, new_col))
        
        # Normalize non-negative values to [0, 1]
        # Find max value among reachable cells (excluding -1 and inf)
        reachable_mask = (dist_field >= 0) & (dist_field != np.inf)
        if np.any(reachable_mask):
            max_dist = np.max(dist_field[reachable_mask])
            if max_dist > 0:
                # Normalize reachable cells to [0, 1]
                normalized_field = dist_field.copy()
                normalized_field[reachable_mask] = dist_field[reachable_mask] / max_dist
                return normalized_field
        
        # If no reachable cells or max_dist is 0, return as-is
        return dist_field
    
    def generate_episode_masks(self, episode_data: Dict[str, Any], 
                              env_info: Optional[EnvironmentInfo] = None) -> Dict[str, np.ndarray]:
        """Generate 224x224 masks for different object types in the episode.
        
        Args:
            episode_data: Episode data from pickle file
            env_info: Environment information (if None, will be extracted from episode data)
            
        Returns:
            Dictionary containing masks with keys:
            - 'robot': Robot position mask
            - 'goal': Goal position mask  
            - 'movable': All movable objects mask
            - 'static': Static objects (walls) mask
            - 'reachable': Reachable objects mask
            - 'target_object': Target object mask (object being manipulated)
            - 'target_goal': Target object at goal position mask
            - 'robot_distance': Distance field from robot position (wavefront)
            - 'goal_distance': Distance field from goal position (wavefront)
            - 'combined_distance': Sum of robot and goal distance fields, normalized
        """
        # Extract environment info if not provided
        if env_info is None:
            env_info = self._extract_env_info_from_episode(episode_data)
        
        # Get data references
        state_observations = episode_data.get('state_observations', [])
        static_object_info = episode_data.get('static_object_info') or {}
        action_sequence = episode_data.get('action_sequence', [])
        robot_goal = episode_data.get('robot_goal', (0.0, 0.0, 0.0))
        world_bounds = env_info.world_bounds
        
        # Initialize masks (224x224, float32, values 0.0-1.0)
        masks = {
            'robot': np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32),
            'goal': np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32),
            'movable': np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32),
            'static': np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32),
            'reachable': np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32),
            'target_object': np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32),
            'target_goal': np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32),
            'robot_distance': np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32),
            'goal_distance': np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32),
            'combined_distance': np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32)
        }
        
        # Draw static objects (walls)
        for obj in env_info.static_objects:
            # Calculate rotation angle from quaternion
            qw, qx, qy, qz = obj.quat_w, obj.quat_x, obj.quat_y, obj.quat_z
            angle = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            
            self._draw_rotated_box_mask(
                masks['static'], obj.x, obj.y, obj.size_x, obj.size_y, 
                angle, world_bounds, 1.0
            )
        
        # Draw robot goal position
        self._draw_circle_mask(masks['goal'], robot_goal[0], robot_goal[1], 0.25, world_bounds, 1.0)
        
        # Get target object info from action sequence
        target_object_id = None
        target_pose = None
        if action_sequence and len(action_sequence) > 0:
            # Use first action as target (could be extended for multi-step plans)
            first_action = action_sequence[0]
            target_object_id = first_action.get('object_id')
            target_pose = first_action.get('target')  # (x, y, theta)
        
        # Process state observations
        if state_observations and len(state_observations) > 0:
            # Use final state for object positions
            final_state = state_observations[-1]
            
            # Draw robot position
            if 'robot_pose' in final_state:
                robot_pose = final_state['robot_pose']
                self._draw_circle_mask(masks['robot'], robot_pose[0], robot_pose[1], 0.2, world_bounds, 1.0)
            
            # Draw movable objects
            for obj_name, pose in final_state.items():
                if obj_name != 'robot_pose':
                    obj_base_name = obj_name.replace('_pose', '')
                    obj_info = static_object_info.get(obj_base_name, {})
                    
                    if 'size_x' in obj_info and 'size_y' in obj_info:
                        x, y, theta = pose[0], pose[1], pose[2]
                        size_x = obj_info['size_x']
                        size_y = obj_info['size_y']
                        
                        # Draw in movable objects mask
                        self._draw_rotated_box_mask(
                            masks['movable'], x, y, size_x, size_y, 
                            theta, world_bounds, 1.0
                        )
                        
                        # Check if this is the target object
                        if target_object_id and target_object_id.startswith(obj_base_name):
                            # Draw target object in current position
                            self._draw_rotated_box_mask(
                                masks['target_object'], x, y, size_x, size_y, 
                                theta, world_bounds, 1.0
                            )
                            
                            # Draw target object at goal position
                            if target_pose:
                                goal_x, goal_y, goal_theta = target_pose[0], target_pose[1], target_pose[2]
                                self._draw_rotated_box_mask(
                                    masks['target_goal'], goal_x, goal_y, size_x, size_y, 
                                    goal_theta, world_bounds, 1.0
                                )
        
        # Generate reachable objects mask
        # Use reachable objects from episode data if available
        reachable_objects_list = episode_data.get('reachable_objects_before_action')
        if reachable_objects_list and len(reachable_objects_list) > 0:
            # Use reachable objects from first state observation
            reachable_objects = set(reachable_objects_list[0])

            # Draw only reachable objects
            if state_observations and len(state_observations) > 0:
                final_state = state_observations[-1]
                for obj_name, pose in final_state.items():
                    if obj_name != 'robot_pose':
                        obj_base_name = obj_name.replace('_pose', '')
                        if obj_base_name in reachable_objects:
                            obj_info = static_object_info.get(obj_base_name, {})
                            if obj_info:
                                self._draw_rotated_box_mask(
                                    masks['reachable'], pose[0], pose[1],
                                    obj_info['size_x'], obj_info['size_y'], pose[2],
                                    world_bounds, 1.0
                                )
        else:
            # Fallback: mark all movable objects as potentially reachable
            masks['reachable'] = masks['movable'].copy()
        
        # Generate distance field masks
        robot_radius = 0.15  # Robot radius in meters
        
        # Inflate static and movable masks by robot radius
        static_inflated = self._inflate_mask(masks['static'], robot_radius, world_bounds)
        movable_inflated = self._inflate_mask(masks['movable'], robot_radius, world_bounds)
        
        # Compute robot distance field
        robot_pos = None
        if state_observations and len(state_observations) > 0:
            final_state = state_observations[-1]
            if 'robot_pose' in final_state:
                robot_pose = final_state['robot_pose']
                robot_pos = (robot_pose[0], robot_pose[1])
        
        if robot_pos is not None:
            masks['robot_distance'] = self._compute_distance_field(
                robot_pos[0], robot_pos[1], static_inflated, movable_inflated, world_bounds
            )
        else:
            # If no robot position, fill with -1 (impassable)
            masks['robot_distance'].fill(-1)
        
        # Compute goal distance field  
        masks['goal_distance'] = self._compute_distance_field(
            robot_goal[0], robot_goal[1], static_inflated, movable_inflated, world_bounds
        )
        
        # Note: Combined distance field removed for batch processing efficiency
        
        return masks
    
    def generate_episode_masks_batch(self, episode_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Generate only the 9 masks needed for batch processing (excludes combined distance).

        Args:
            episode_data: Episode data dictionary

        Returns:
            Dictionary containing 9 masks: robot, goal, movable, static, reachable,
            target_object, target_goal, robot_distance, goal_distance
        """
        masks = self.generate_episode_masks(episode_data)

        # Remove combined distance field if present
        if 'combined_distance' in masks:
            del masks['combined_distance']

        return masks

    def generate_episode_masks_multihorizon(self, episode_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Generate masks with multi-horizon target goal masks.

        For an episode with n remaining actions, generates:
        - Standard masks: robot, movable, static, reachable, target_object, target_goal,
          robot_distance, goal_distance (all based on current state)
        - Multi-horizon target goal masks: goal_mask_a1, goal_mask_a2, ..., goal_mask_a{n}
          Each shows the target object drawn at the corresponding action's target position

        Example for 2-push chain from S0:
          goal_mask_a1: target object at action[0].target (e.g., target_1)
          goal_mask_a2: target object at action[1].target (e.g., target_2)

        Args:
            episode_data: Episode data with 'action_sequence' containing remaining actions

        Returns:
            Dictionary containing base masks + variable number of goal_mask_a{i} channels
        """
        # Generate base masks from current state
        masks = self.generate_episode_masks(episode_data)

        # Remove combined distance (keep goal_distance for compatibility)
        if 'combined_distance' in masks:
            del masks['combined_distance']

        # Extract environment info
        env_info = self._extract_env_info_from_episode(episode_data)
        world_bounds = env_info.world_bounds
        static_object_info = episode_data.get('static_object_info') or {}

        # Get action sequence
        action_sequence = episode_data.get('action_sequence', [])

        if not action_sequence:
            # No actions, no multi-horizon masks
            return masks

        # Get target object info (same for all actions in the sequence)
        target_object_id = action_sequence[0].get('object_id')
        if not target_object_id:
            return masks

        # Get object size
        obj_base_name = target_object_id
        obj_info = static_object_info.get(obj_base_name, {})
        if 'size_x' not in obj_info or 'size_y' not in obj_info:
            return masks

        size_x = obj_info['size_x']
        size_y = obj_info['size_y']

        # Generate goal_mask_a{i} for each action in the sequence
        for action_idx, action in enumerate(action_sequence, start=1):
            target_pose = action.get('target')
            if not target_pose or len(target_pose) < 3:
                continue

            goal_x, goal_y, goal_theta = target_pose[0], target_pose[1], target_pose[2]

            # Create mask for this action's target goal
            goal_mask = np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32)

            # Draw target object at goal position
            self._draw_rotated_box_mask(
                goal_mask, goal_x, goal_y, size_x, size_y,
                goal_theta, world_bounds, 1.0
            )

            # Store as goal_mask_a{action_idx}
            masks[f'goal_mask_a{action_idx}'] = goal_mask

        # Always generate at least 2 goal horizons for consistent dataset schema
        min_goal_horizons = 2
        for action_idx in range(1, min_goal_horizons + 1):
            if f'goal_mask_a{action_idx}' not in masks:
                masks[f'goal_mask_a{action_idx}'] = np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32)

        return masks

    def generate_local_episode_masks(self, episode_data: Dict[str, Any],
                                     crop_size_meters: float = 2.0,
                                     goal_circle_radius: float = 0.2,
                                     highres_size: int = 2048,
                                     output_size: int = 224) -> Optional[Dict[str, np.ndarray]]:
        """Generate local object-centered masks by rendering at high resolution and cropping.

        Approach: Render full scene at high resolution (2048x2048), then crop a window
        centered on the target object and resize to output size (224x224).

        Args:
            episode_data: Episode data dictionary (must have action_sequence with target object)
            crop_size_meters: Size of the square crop region in meters (default: 2.0m)
            goal_circle_radius: Radius of goal region circles in meters (default: 0.1m)
            highres_size: Size of high-resolution render (default: 2048)
            output_size: Size of output masks (default: 224)

        Returns:
            Dictionary containing local masks:
            - 'local_target_object': Target object at center
            - 'local_target_goal': Push target position
            - 'local_goal_region': Circles at region_goals_sampled points
            - 'local_static': Static obstacles (walls)
            - 'local_movable': Other movable objects
            - 'local_metadata': Dict with object_center, local_bounds for inference
            Returns None if episode doesn't have required data (e.g., no action_sequence)
        """
        import cv2

        # Get action sequence to find target object
        action_sequence = episode_data.get('action_sequence', [])
        if not action_sequence or len(action_sequence) == 0:
            return None

        first_action = action_sequence[0]
        target_object_id = first_action.get('object_id')
        target_pose = first_action.get('target')  # (x, y, theta)

        if not target_object_id or not target_pose:
            return None

        # Get state observations to find object positions
        state_observations = episode_data.get('state_observations', [])
        if not state_observations or len(state_observations) == 0:
            return None

        # Use first state observation (before any action)
        first_state = state_observations[0]

        # Find target object pose
        target_obj_pose_key = f"{target_object_id}_pose"
        if target_obj_pose_key not in first_state:
            target_obj_pose_key = target_object_id
            if target_obj_pose_key not in first_state:
                return None

        obj_pose = first_state[target_obj_pose_key]
        obj_x, obj_y, obj_theta = obj_pose[0], obj_pose[1], obj_pose[2]

        # Get environment info and world bounds
        env_info = self._extract_env_info_from_episode(episode_data)
        world_bounds = env_info.world_bounds
        x_min_w, x_max_w, y_min_w, y_max_w = world_bounds

        # Get static object info for sizes
        static_object_info = episode_data.get('static_object_info') or {}

        # Get target object size
        obj_info = static_object_info.get(target_object_id, {})
        if 'size_x' not in obj_info or 'size_y' not in obj_info:
            return None

        target_size_x = obj_info['size_x']
        target_size_y = obj_info['size_y']

        # Helper: world to highres pixel
        world_width = x_max_w - x_min_w
        world_height = y_max_w - y_min_w
        world_size = max(world_width, world_height)
        scale = highres_size / world_size
        world_center_x = (x_min_w + x_max_w) / 2
        world_center_y = (y_min_w + y_max_w) / 2
        img_center = highres_size / 2

        def world_to_highres(x, y):
            px = int((x - world_center_x) * scale + img_center)
            py = int((y - world_center_y) * scale + img_center)
            return px, py

        def size_to_pixels(size):
            return int(size * scale * 2)  # full size, not half

        # Initialize high-res masks
        highres_masks = {
            'target_object': np.zeros((highres_size, highres_size), dtype=np.float32),
            'target_goal': np.zeros((highres_size, highres_size), dtype=np.float32),
            'goal_region': np.zeros((highres_size, highres_size), dtype=np.float32),
            'static': np.zeros((highres_size, highres_size), dtype=np.float32),
            'movable': np.zeros((highres_size, highres_size), dtype=np.float32),
        }

        # Draw rotated box on highres mask
        def draw_rotated_box_highres(mask, cx, cy, size_x, size_y, angle):
            px, py = world_to_highres(cx, cy)
            w, h = size_to_pixels(size_x), size_to_pixels(size_y)
            rect = ((px, py), (w, h), np.degrees(angle))
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.fillPoly(mask, [box], 1.0)

        # Draw circle on highres mask
        def draw_circle_highres(mask, cx, cy, radius):
            px, py = world_to_highres(cx, cy)
            r_px = int(radius * scale)
            cv2.circle(mask, (px, py), max(1, r_px), 1.0, -1)

        # 1. Draw target object
        draw_rotated_box_highres(highres_masks['target_object'], obj_x, obj_y,
                                  target_size_x, target_size_y, obj_theta)

        # 2. Draw target goal
        goal_x, goal_y, goal_theta = target_pose[0], target_pose[1], target_pose[2]
        draw_rotated_box_highres(highres_masks['target_goal'], goal_x, goal_y,
                                  target_size_x, target_size_y, goal_theta)

        # 3. Draw goal region circles
        algorithm_stats = episode_data.get('algorithm_stats') or {}
        region_goals_sampled = episode_data.get('region_goals_sampled')
        if region_goals_sampled is None:
            region_goals_sampled = algorithm_stats.get('region_goals_sampled')
        if not region_goals_sampled:
            region_goal_used = episode_data.get('region_goal_used') or algorithm_stats.get('region_goal_used')
            if region_goal_used:
                region_goals_sampled = [region_goal_used]

        if region_goals_sampled:
            for goal in region_goals_sampled:
                gx, gy = goal[0], goal[1]
                draw_circle_highres(highres_masks['goal_region'], gx, gy, goal_circle_radius)

        # 4. Draw static objects (walls)
        for obj in env_info.static_objects:
            qw, qx, qy, qz = obj.quat_w, obj.quat_x, obj.quat_y, obj.quat_z
            angle = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            draw_rotated_box_highres(highres_masks['static'], obj.x, obj.y,
                                      obj.size_x, obj.size_y, angle)

        # 5. Draw other movable objects
        for obj_name, pose in first_state.items():
            if obj_name == 'robot_pose':
                continue
            obj_base_name = obj_name.replace('_pose', '')
            if obj_base_name == target_object_id:
                continue

            mov_obj_info = static_object_info.get(obj_base_name, {})
            if 'size_x' in mov_obj_info and 'size_y' in mov_obj_info:
                mx, my, mtheta = pose[0], pose[1], pose[2]
                draw_rotated_box_highres(highres_masks['movable'], mx, my,
                                          mov_obj_info['size_x'], mov_obj_info['size_y'], mtheta)

        # Compute crop window in highres pixels
        obj_px, obj_py = world_to_highres(obj_x, obj_y)
        crop_size_px = int(crop_size_meters * scale)
        half_crop = crop_size_px // 2

        # Crop boundaries (with clamping)
        y1 = max(0, obj_py - half_crop)
        y2 = min(highres_size, obj_py + half_crop)
        x1 = max(0, obj_px - half_crop)
        x2 = min(highres_size, obj_px + half_crop)

        # Crop and resize each mask
        masks = {}
        for name, highres_mask in highres_masks.items():
            cropped = highres_mask[y1:y2, x1:x2]
            # Handle edge cases where crop is smaller than expected
            if cropped.shape[0] == 0 or cropped.shape[1] == 0:
                masks[f'local_{name}'] = np.zeros((output_size, output_size), dtype=np.float32)
            else:
                resized = cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_AREA)
                masks[f'local_{name}'] = resized.astype(np.float32)

        # Compute local bounds for metadata
        half_size = crop_size_meters / 2.0
        local_bounds = (
            obj_x - half_size,
            obj_x + half_size,
            obj_y - half_size,
            obj_y + half_size
        )

        # Store metadata for inference coordinate recovery
        masks['local_metadata'] = {
            'object_center': (obj_x, obj_y),
            'object_theta': obj_theta,
            'local_bounds': local_bounds,
            'crop_size_meters': crop_size_meters,
            'resolution': crop_size_meters / output_size  # meters per pixel
        }

        return masks

    # Constants matching C++ WavefrontGrid and WavefrontSnapshotExporter
    INFLATION_EPSILON = 0.005  # Same as WavefrontSnapshotExporter.INFLATION_EPSILON

    # 8-connected neighbor offsets (matching C++ WavefrontGrid and WavefrontSnapshotExporter)
    NEIGHBOR_OFFSETS_8 = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    def generate_all_masks_highres(self, episode_data: Dict[str, Any],
                                   highres_size: int = 1024,
                                   global_output_size: int = 224,
                                   local_output_size: int = 224,
                                   local_crop_size_meters: float = 5.0,
                                   goal_circle_radius: float = 0.25) -> Dict[str, Any]:
        """Generate both global and local masks from a single high-resolution render.

        This is more efficient than calling generate_episode_masks and
        generate_local_episode_masks separately, as it renders once at high resolution
        and then creates both global (full resize) and local (crop + resize) masks.

        Args:
            episode_data: Episode data dictionary
            highres_size: Size of high-resolution render (default: 1024)
            global_output_size: Size of global output masks (default: 224)
            local_output_size: Size of local output masks (default: 224)
            local_crop_size_meters: Size of local crop region in meters (default: 5.0)
            goal_circle_radius: Radius of robot goal circle in meters (default: 0.25)

        Returns:
            Dictionary containing:
            - 'global': Dict of global masks (resized from full highres)
                - robot, goal, movable, static, reachable, target_object, target_goal,
                  robot_region, goal_sample_region
            - 'local': Dict of local masks (cropped around target object) or None
                - local_target_object, local_target_goal, local_static, local_movable,
                  local_robot_region, local_goal_sample_region
            - 'local_metadata': Dict with object_center, local_bounds for inference

            robot_region: Binary mask of cells reachable by robot (computed via BFS on
                inflated obstacle map). 1 = reachable from robot position, 0 = blocked.
            goal_sample_region: Binary mask of cells reachable from first goal sample
                position (computed via BFS on inflated obstacle map).

        Note:
            Wavefront parameters match C++ WavefrontGrid and Python WavefrontSnapshotExporter:
            - 8-connected BFS (not 4-connected)
            - Inflation: robot_half_extent + INFLATION_EPSILON (0.005m) per axis
            - Rotated box inflation (not circular morphological dilation)
            - 3x3 neighborhood clearance around robot/goal positions
        """
        import cv2

        # Extract environment info
        env_info = self._extract_env_info_from_episode(episode_data)
        world_bounds = env_info.world_bounds
        x_min_w, x_max_w, y_min_w, y_max_w = world_bounds

        # Get data
        state_observations = episode_data.get('state_observations', [])
        static_object_info = episode_data.get('static_object_info') or {}
        action_sequence = episode_data.get('action_sequence', [])
        robot_goal = episode_data.get('robot_goal', (0.0, 0.0, 0.0))

        # Helper: world to highres pixel
        world_width = x_max_w - x_min_w
        world_height = y_max_w - y_min_w
        world_size = max(world_width, world_height)
        scale = highres_size / world_size
        world_center_x = (x_min_w + x_max_w) / 2
        world_center_y = (y_min_w + y_max_w) / 2
        img_center = highres_size / 2

        def world_to_highres(x, y):
            px = int((x - world_center_x) * scale + img_center)
            py = int((y - world_center_y) * scale + img_center)
            return px, py

        def size_to_pixels(size):
            return int(size * scale * 2)

        # Draw rotated box on highres mask
        def draw_rotated_box(mask, cx, cy, size_x, size_y, angle):
            px, py = world_to_highres(cx, cy)
            w, h = size_to_pixels(size_x), size_to_pixels(size_y)
            if w <= 0 or h <= 0:
                return
            rect = ((px, py), (w, h), np.degrees(angle))
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.fillPoly(mask, [box], 1.0)

        # Draw circle on highres mask
        def draw_circle(mask, cx, cy, radius):
            px, py = world_to_highres(cx, cy)
            r_px = int(radius * scale)
            cv2.circle(mask, (px, py), max(1, r_px), 1.0, -1)

        # Initialize high-res masks
        highres = {
            'robot': np.zeros((highres_size, highres_size), dtype=np.float32),
            'goal': np.zeros((highres_size, highres_size), dtype=np.float32),
            'movable': np.zeros((highres_size, highres_size), dtype=np.float32),
            'static': np.zeros((highres_size, highres_size), dtype=np.float32),
            'reachable': np.zeros((highres_size, highres_size), dtype=np.float32),
            'target_object': np.zeros((highres_size, highres_size), dtype=np.float32),
            'target_goal': np.zeros((highres_size, highres_size), dtype=np.float32),
            'robot_region': np.zeros((highres_size, highres_size), dtype=np.float32),
            'goal_sample_region': np.zeros((highres_size, highres_size), dtype=np.float32),
        }

        # Initialize multi-horizon goal masks (goal_mask_a1, goal_mask_a2, ...)
        # Always generate at least 2 goal horizons for consistent dataset schema
        min_goal_horizons = 2
        num_goal_horizons = max(min_goal_horizons, len(action_sequence))
        for action_idx in range(1, num_goal_horizons + 1):
            highres[f'goal_mask_a{action_idx}'] = np.zeros((highres_size, highres_size), dtype=np.float32)

        # 1. Draw static objects (walls)
        for obj in env_info.static_objects:
            qw, qx, qy, qz = obj.quat_w, obj.quat_x, obj.quat_y, obj.quat_z
            angle = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            draw_rotated_box(highres['static'], obj.x, obj.y, obj.size_x, obj.size_y, angle)

        # 2. Draw robot goal
        draw_circle(highres['goal'], robot_goal[0], robot_goal[1], goal_circle_radius)

        # Get target object info
        target_object_id = None
        target_pose = None
        target_size_x, target_size_y = None, None
        obj_x, obj_y, obj_theta = None, None, None

        if action_sequence and len(action_sequence) > 0:
            first_action = action_sequence[0]
            target_object_id = first_action.get('object_id')
            target_pose = first_action.get('target')

        # 3. Process state observations - draw movable objects
        if state_observations and len(state_observations) > 0:
            first_state = state_observations[0]

            # Draw robot position
            if 'robot_pose' in first_state:
                robot_pose = first_state['robot_pose']
                draw_circle(highres['robot'], robot_pose[0], robot_pose[1], 0.2)

            # Draw movable objects
            for obj_name, pose in first_state.items():
                if obj_name == 'robot_pose':
                    continue
                obj_base_name = obj_name.replace('_pose', '')
                obj_info = static_object_info.get(obj_base_name, {})

                if 'size_x' in obj_info and 'size_y' in obj_info:
                    x, y, theta = pose[0], pose[1], pose[2]
                    size_x = obj_info['size_x']
                    size_y = obj_info['size_y']

                    # Draw in movable mask
                    draw_rotated_box(highres['movable'], x, y, size_x, size_y, theta)

                    # Check if target object
                    if target_object_id and obj_base_name == target_object_id:
                        obj_x, obj_y, obj_theta = x, y, theta
                        target_size_x, target_size_y = size_x, size_y
                        draw_rotated_box(highres['target_object'], x, y, size_x, size_y, theta)

                        if target_pose:
                            goal_x, goal_y, goal_theta = target_pose[0], target_pose[1], target_pose[2]
                            draw_rotated_box(highres['target_goal'], goal_x, goal_y, size_x, size_y, goal_theta)

                        # Draw multi-horizon goal masks (goal_mask_a1, goal_mask_a2, ...)
                        for action_idx, action in enumerate(action_sequence, start=1):
                            action_target = action.get('target')
                            if action_target and len(action_target) >= 3:
                                ax, ay, atheta = action_target[0], action_target[1], action_target[2]
                                draw_rotated_box(highres[f'goal_mask_a{action_idx}'], ax, ay, size_x, size_y, atheta)

        # 4. Draw reachable objects
        reachable_list = episode_data.get('reachable_objects_before_action')
        if reachable_list and len(reachable_list) > 0 and state_observations:
            first_state = state_observations[0]
            reachable_objects = reachable_list[0] if reachable_list[0] else []
            for obj_name in reachable_objects:
                pose_key = f"{obj_name}_pose"
                if pose_key in first_state:
                    pose = first_state[pose_key]
                    obj_info = static_object_info.get(obj_name, {})
                    if 'size_x' in obj_info and 'size_y' in obj_info:
                        draw_rotated_box(highres['reachable'], pose[0], pose[1],
                                        obj_info['size_x'], obj_info['size_y'], pose[2])

        # 5. Get region_goals_sampled for goal_sample_region computation
        algorithm_stats = episode_data.get('algorithm_stats') or {}
        region_goals_sampled = episode_data.get('region_goals_sampled')
        if region_goals_sampled is None:
            region_goals_sampled = algorithm_stats.get('region_goals_sampled')
        if not region_goals_sampled:
            region_goal_used = episode_data.get('region_goal_used') or algorithm_stats.get('region_goal_used')
            if region_goal_used:
                region_goals_sampled = [region_goal_used]

        # Fallback for inference: use robot_goal if no goal samples available
        # This makes semantic sense - goals should be reachable from robot's target location
        if not region_goals_sampled and robot_goal:
            region_goals_sampled = [(robot_goal[0], robot_goal[1], robot_goal[2] if len(robot_goal) > 2 else 0.0)]

        # 6. Compute robot_region (reachable cells from robot position)
        # Get robot position and robot half-extent for inflation
        robot_px, robot_py = None, None
        robot_half_extent_x = 0.15  # Default fallback
        robot_half_extent_y = 0.15  # Default fallback

        # Get robot half-extent from static_object_info (matching WavefrontSnapshotExporter)
        robot_info = static_object_info.get('robot', {})
        if 'size_x' in robot_info:
            robot_half_extent_x = robot_info['size_x']
        if 'size_y' in robot_info:
            robot_half_extent_y = robot_info['size_y']

        if state_observations and len(state_observations) > 0:
            first_state = state_observations[0]
            if 'robot_pose' in first_state:
                robot_pose = first_state['robot_pose']
                robot_px, robot_py = world_to_highres(robot_pose[0], robot_pose[1])

        if robot_px is not None and robot_py is not None:
            # Compute inflation amounts matching WavefrontSnapshotExporter
            inflate_x = robot_half_extent_x + self.INFLATION_EPSILON
            inflate_y = robot_half_extent_y + self.INFLATION_EPSILON

            # Build inflated obstacle grid by drawing each obstacle with inflated size
            # This matches the WavefrontSnapshotExporter approach (rotated box inflation)
            inflated_obstacles = np.zeros((highres_size, highres_size), dtype=np.uint8)

            # Helper to draw inflated rotated box
            def draw_inflated_box(mask, cx, cy, half_x, half_y, angle):
                """Draw obstacle inflated by robot half-extent + epsilon."""
                inflated_half_x = half_x + inflate_x
                inflated_half_y = half_y + inflate_y
                px, py = world_to_highres(cx, cy)
                w = int(inflated_half_x * 2 * scale)
                h = int(inflated_half_y * 2 * scale)
                if w <= 0 or h <= 0:
                    return
                rect = ((px, py), (w, h), np.degrees(angle))
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cv2.fillPoly(mask, [box], 1)

            # Draw inflated static obstacles
            for obj in env_info.static_objects:
                qw, qx, qy, qz = obj.quat_w, obj.quat_x, obj.quat_y, obj.quat_z
                angle = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                draw_inflated_box(inflated_obstacles, obj.x, obj.y, obj.size_x, obj.size_y, angle)

            # Draw inflated movable obstacles
            if state_observations and len(state_observations) > 0:
                first_state = state_observations[0]
                for obj_name, pose in first_state.items():
                    if obj_name == 'robot_pose':
                        continue
                    obj_base_name = obj_name.replace('_pose', '')
                    obj_info = static_object_info.get(obj_base_name, {})
                    if 'size_x' in obj_info and 'size_y' in obj_info:
                        x, y, theta = pose[0], pose[1], pose[2]
                        draw_inflated_box(inflated_obstacles, x, y, obj_info['size_x'], obj_info['size_y'], theta)

            # Use scipy.ndimage.label for fast connected component analysis
            # This is ~200x faster than Python BFS for 1024x1024 grids
            from scipy import ndimage

            # Fix: If robot position is in inflated obstacle due to discretization,
            # clear the robot cell and its 8-neighborhood (matching C++ wavefront_grid behavior)
            if (0 <= robot_px < highres_size and 0 <= robot_py < highres_size and
                inflated_obstacles[robot_py, robot_px] == 1):
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = robot_py + dy, robot_px + dx
                        if 0 <= ny < highres_size and 0 <= nx < highres_size:
                            inflated_obstacles[ny, nx] = 0

            # Create free space mask and find connected components (8-connected)
            free_space = (inflated_obstacles == 0).astype(np.int32)
            structure_8conn = np.ones((3, 3), dtype=np.int32)  # 8-connected
            labeled_regions, num_regions = ndimage.label(free_space, structure=structure_8conn)

            # Find robot's region and create mask
            if (0 <= robot_px < highres_size and 0 <= robot_py < highres_size and
                labeled_regions[robot_py, robot_px] > 0):
                robot_label = labeled_regions[robot_py, robot_px]
                highres['robot_region'] = (labeled_regions == robot_label).astype(np.float32)
            else:
                highres['robot_region'] = np.zeros((highres_size, highres_size), dtype=np.float32)

            # 7. Compute goal_sample_region (reachable cells from first goal sample)
            # Reuse labeled_regions from robot_region computation
            if region_goals_sampled and len(region_goals_sampled) > 0:
                goal_sample = region_goals_sampled[0]
                goal_sample_px, goal_sample_py = world_to_highres(goal_sample[0], goal_sample[1])

                # Fix: If goal sample position is in inflated obstacle, clear its neighborhood
                # and re-label (only if different from robot case)
                if (0 <= goal_sample_px < highres_size and 0 <= goal_sample_py < highres_size and
                    inflated_obstacles[goal_sample_py, goal_sample_px] == 1):
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = goal_sample_py + dy, goal_sample_px + dx
                            if 0 <= ny < highres_size and 0 <= nx < highres_size:
                                inflated_obstacles[ny, nx] = 0
                    # Re-label with updated free space
                    free_space = (inflated_obstacles == 0).astype(np.int32)
                    labeled_regions, num_regions = ndimage.label(free_space, structure=structure_8conn)

                # Find goal's region and create mask
                if (0 <= goal_sample_px < highres_size and 0 <= goal_sample_py < highres_size and
                    labeled_regions[goal_sample_py, goal_sample_px] > 0):
                    goal_label = labeled_regions[goal_sample_py, goal_sample_px]
                    highres['goal_sample_region'] = (labeled_regions == goal_label).astype(np.float32)
                else:
                    highres['goal_sample_region'] = np.zeros((highres_size, highres_size), dtype=np.float32)

        # === Create global masks (resize full highres to output size) ===
        global_masks = {}
        for name, hr_mask in highres.items():
            resized = cv2.resize(hr_mask, (global_output_size, global_output_size),
                                interpolation=cv2.INTER_AREA)
            global_masks[name] = resized.astype(np.float32)

        # === Create local masks (crop around target object, then resize) ===
        local_masks = None
        local_metadata = None

        if obj_x is not None and obj_y is not None:
            # Compute crop window centered on object
            obj_px, obj_py = world_to_highres(obj_x, obj_y)
            crop_size_px = int(local_crop_size_meters * scale)
            half_crop = crop_size_px // 2

            # Calculate raw crop bounds (may extend outside image)
            y1_raw, y2_raw = obj_py - half_crop, obj_py + half_crop
            x1_raw, x2_raw = obj_px - half_crop, obj_px + half_crop

            # Clamp to image bounds for extraction
            y1 = max(0, y1_raw)
            y2 = min(highres_size, y2_raw)
            x1 = max(0, x1_raw)
            x2 = min(highres_size, x2_raw)

            # Calculate padding needed to keep object at center
            pad_top = y1 - y1_raw      # positive if y1_raw < 0
            pad_bottom = y2_raw - y2   # positive if y2_raw > highres_size
            pad_left = x1 - x1_raw     # positive if x1_raw < 0
            pad_right = x2_raw - x2    # positive if x2_raw > highres_size

            local_masks = {}
            # For local, we want: target_object, target_goal, static, movable, robot_region, goal_sample_region
            local_mask_names = ['target_object', 'target_goal', 'static', 'movable', 'robot_region', 'goal_sample_region']
            # Add multi-horizon goal masks (goal_mask_a1, goal_mask_a2, ...)
            # Always include at least 2 goal horizons for consistent dataset schema
            min_goal_horizons = 2
            num_goal_horizons = max(min_goal_horizons, len(action_sequence))
            for action_idx in range(1, num_goal_horizons + 1):
                local_mask_names.append(f'goal_mask_a{action_idx}')
            for name in local_mask_names:
                hr_mask = highres[name]
                cropped = hr_mask[y1:y2, x1:x2]
                if cropped.shape[0] == 0 or cropped.shape[1] == 0:
                    local_masks[f'local_{name}'] = np.zeros((local_output_size, local_output_size), dtype=np.float32)
                else:
                    # Pad cropped region to maintain object at center (pad with zeros = empty space)
                    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                        cropped = np.pad(cropped,
                                        ((pad_top, pad_bottom), (pad_left, pad_right)),
                                        mode='constant', constant_values=0)
                    resized = cv2.resize(cropped, (local_output_size, local_output_size),
                                        interpolation=cv2.INTER_AREA)
                    local_masks[f'local_{name}'] = resized.astype(np.float32)

            # Local metadata
            # Object is always at center of padded crop, so use object position as crop center
            half_size = local_crop_size_meters / 2.0
            local_bounds = (obj_x - half_size, obj_x + half_size, obj_y - half_size, obj_y + half_size)
            local_metadata = {
                'object_center': (obj_x, obj_y),  # Object is at center of (padded) crop
                'object_theta': obj_theta,
                'local_bounds': local_bounds,
                'crop_size_meters': local_crop_size_meters,
                'resolution': local_crop_size_meters / local_output_size
            }

        return {
            'global': global_masks,
            'local': local_masks,
            'local_metadata': local_metadata
        }

    def save_masks(self, masks: Dict[str, np.ndarray], output_dir: str,
                  episode_id: str) -> None:
        """Save masks as PNG files and create a composite visualization.

        Args:
            masks: Dictionary of masks from generate_episode_masks
            output_dir: Directory to save masks
            episode_id: Episode identifier for filenames
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save individual masks
        for mask_name, mask in masks.items():
            # Handle special values for distance fields
            if mask_name in ['robot_distance', 'goal_distance', 'combined_distance']:
                # Distance fields can have -1 (obstacles) and inf (unreachable)
                # Convert for saving: -1 -> 0 (black), [0,1] -> [64, 255] (gray to white)
                save_mask = np.full_like(mask, 0, dtype=np.uint8)  # Start with black
                
                # Mark obstacles as black (0)
                obstacle_mask = (mask < 0)
                save_mask[obstacle_mask] = 0
                
                # Mark reachable areas in range [64, 255]
                reachable_mask = (mask >= 0) & (mask != np.inf)
                if np.any(reachable_mask):
                    save_mask[reachable_mask] = (64 + mask[reachable_mask] * 191).astype(np.uint8)
            else:
                # Regular masks: just scale to 0-255
                mask_img = np.clip(mask, 0, 1)  # Ensure values are in [0,1]
                save_mask = (mask_img * 255).astype(np.uint8)
            
            filename = f"{episode_id}_{mask_name}_mask.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, save_mask)
        
        # Create composite visualization
        mask_names = ['robot', 'goal', 'movable', 'static', 'target_object', 'target_goal', 'reachable', 'robot_distance', 'goal_distance', 'combined_distance']
        cols = 4
        rows = 3
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, mask_name in enumerate(mask_names):
            if i < len(axes) and mask_name in masks:
                mask = masks[mask_name]
                
                # Use different visualization for distance fields
                if mask_name in ['robot_distance', 'goal_distance', 'combined_distance']:
                    # Distance fields can have negative values (-1 for obstacles)
                    # Use custom colormap: black for -1, gradient for 0 to 1
                    axes[i].imshow(mask, cmap='viridis', vmin=-1, vmax=1)
                else:
                    # Regular binary masks
                    axes[i].imshow(mask, cmap='gray', vmin=0, vmax=1)
                
                axes[i].set_title(f'{mask_name.replace("_", " ").title()} Mask')
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(mask_names), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        composite_path = os.path.join(output_dir, f"{episode_id}_masks_composite.png")
        plt.savefig(composite_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {len(masks)} masks and composite to {output_dir}")
    
    def visualize_episode(self, episode_data: Dict[str, Any], 
                         env_info: Optional[EnvironmentInfo] = None,
                         save_path: Optional[str] = None,
                         show_trajectory: bool = True,
                         show_actions: bool = True) -> None:
        """Visualize a single episode with environment and trajectory.
        
        Args:
            episode_data: Episode data from pickle file
            env_info: Environment information (if None, will be extracted from episode data)
            save_path: Path to save figure (optional)
            show_trajectory: Whether to show robot trajectory
            show_actions: Whether to show action sequence
        """
        # Extract environment info from episode data if not provided
        if env_info is None:
            env_info = self._extract_env_info_from_episode(episode_data)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        # Set world bounds
        x_min, x_max, y_min, y_max = env_info.world_bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        
        # Get state observations for reference throughout the function
        state_observations = episode_data.get('state_observations', [])
        static_object_info = episode_data.get('static_object_info') or {}
        
        # Draw static objects (walls)
        for obj in env_info.static_objects:
            # Calculate rotation angle from quaternion
            qw, qx, qy, qz = obj.quat_w, obj.quat_x, obj.quat_y, obj.quat_z
            angle = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            angle_deg = np.degrees(angle)
            
            # Create rotated rectangle
            rect = Rectangle(
                (obj.x - obj.size_x, obj.y - obj.size_y),
                2 * obj.size_x, 2 * obj.size_y,
                angle=angle_deg,
                facecolor='gray', edgecolor='black', alpha=0.8
            )
            ax.add_patch(rect)
        
        # Draw robot start position (from first state observation if available)
        robot_start_pos = env_info.robot_start
        if state_observations and len(state_observations) > 0:
            first_state = state_observations[0]
            if 'robot_pose' in first_state:
                robot_start_pos = first_state['robot_pose'][:2]
        
        start_circle = Circle(robot_start_pos[:2], 0.15, 
                            facecolor='green', edgecolor='darkgreen', alpha=0.7)
        ax.add_patch(start_circle)
        ax.text(robot_start_pos[0], robot_start_pos[1] + 0.3, 'START', 
                ha='center', va='bottom', fontweight='bold', color='darkgreen')
        
        # Draw robot goal position
        robot_goal = episode_data.get('robot_goal', env_info.robot_goal)
        goal_circle = Circle(robot_goal[:2], 0.1, 
                           facecolor='red', edgecolor='darkred', alpha=0.7)
        ax.add_patch(goal_circle)
        ax.text(robot_goal[0], robot_goal[1] + 0.3, 'GOAL', 
                ha='center', va='bottom', fontweight='bold', color='darkred')
        
        # Draw movable objects in their final positions from state observations
        
        if state_observations and len(state_observations) > 0:
            # Use final state
            final_state = state_observations[-1]
            for obj_name, pose in final_state.items():
                if obj_name != 'robot_pose':
                    obj_base_name = obj_name.replace('_pose', '')
                    
                    # Get object size from static info
                    obj_info = static_object_info.get(obj_base_name, {})
                    if obj_info and 'size_x' in obj_info and 'size_y' in obj_info:
                        x, y, theta = pose[0], pose[1], pose[2]
                        size_x = obj_info['size_x']
                        size_y = obj_info['size_y']
                        
                        # Create rotated rectangle with correct orientation
                        # Note: Rectangle rotation is around bottom-left corner, so we need to adjust
                        rect = Rectangle(
                            (x - size_x, y - size_y),
                            2 * size_x, 2 * size_y,
                            angle=np.degrees(theta),
                            facecolor='lightblue', edgecolor='blue', alpha=0.7
                        )
                        
                        # For proper rotation around center, we need to use patches.FancyBboxPatch or manual rotation
                        # Let's use manual rotation for accuracy
                        cos_theta = np.cos(theta)
                        sin_theta = np.sin(theta)
                        
                        # Define corners relative to center
                        corners = np.array([
                            [-size_x, -size_y],
                            [size_x, -size_y],
                            [size_x, size_y],
                            [-size_x, size_y],
                            [-size_x, -size_y]  # Close the polygon
                        ])
                        
                        # Rotate corners
                        rotated_corners = np.zeros_like(corners)
                        rotated_corners[:, 0] = corners[:, 0] * cos_theta - corners[:, 1] * sin_theta + x
                        rotated_corners[:, 1] = corners[:, 0] * sin_theta + corners[:, 1] * cos_theta + y
                        
                        # Draw as polygon
                        polygon = patches.Polygon(rotated_corners[:-1], closed=True, 
                                                facecolor='lightblue', edgecolor='blue', alpha=0.7)
                        ax.add_patch(polygon)
                        
                        # Add object label
                        ax.text(x, y, obj_base_name, ha='center', va='center', 
                               fontsize=8, fontweight='bold')
        
        # Draw robot trajectory
        if show_trajectory and state_observations:
            robot_positions = []
            for state in state_observations:
                if 'robot_pose' in state:
                    robot_positions.append(state['robot_pose'][:2])
            
            if len(robot_positions) > 1:
                robot_positions = np.array(robot_positions)
                ax.plot(robot_positions[:, 0], robot_positions[:, 1], 
                       'b-', linewidth=2, alpha=0.7, label='Robot Path')
                
                # Mark waypoints
                ax.scatter(robot_positions[:, 0], robot_positions[:, 1], 
                          c='blue', s=30, alpha=0.7, zorder=5)
        
        # Draw action sequence
        if show_actions and episode_data.get('action_sequence'):
            actions = episode_data['action_sequence']
            for i, action in enumerate(actions):
                target = action['target']
                
                # Draw action arrow or marker
                ax.scatter(target[0], target[1], c='orange', s=50, 
                          marker='*', edgecolor='darkorange', zorder=6)
                ax.text(target[0], target[1] + 0.15, f'A{i+1}', 
                       ha='center', va='bottom', fontsize=8, 
                       fontweight='bold', color='darkorange')
        
        # Add title and info
        title = f"Episode: {episode_data.get('episode_id', 'Unknown')}"
        if episode_data.get('solution_found'):
            title += f" ✓ (depth: {episode_data.get('solution_depth', 'N/A')})"
        else:
            title += " ✗ (no solution)"
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markersize=10, label='Start'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=8, label='Goal'),
            patches.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.8, label='Walls'),
            patches.Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.7, label='Objects')
        ]
        
        if show_trajectory and state_observations:
            legend_elements.append(plt.Line2D([0], [0], color='blue', linewidth=2, 
                                            alpha=0.7, label='Robot Path'))
        
        if show_actions and episode_data.get('action_sequence'):
            legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', 
                                            markerfacecolor='orange', markersize=10, 
                                            label='Actions'))
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_batch(self, data_dir: str, output_dir: str, 
                       max_episodes: int = 10,
                       successful_only: bool = False) -> None:
        """Visualize multiple episodes from a data directory.
        
        Args:
            data_dir: Directory containing pickle files
            output_dir: Output directory for visualizations
            max_episodes: Maximum number of episodes to visualize
            successful_only: Only visualize successful episodes
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all pickle files
        pickle_files = glob.glob(os.path.join(data_dir, "*_results.pkl"))
        print(f"Found {len(pickle_files)} data files")
        
        episode_count = 0
        
        for pickle_file in pickle_files:
            if episode_count >= max_episodes:
                break
            
            try:
                with open(pickle_file, 'rb') as f:
                    data = pickle.load(f)
                
                episodes = data.get('episode_results', [])
                for episode in episodes:
                    if episode_count >= max_episodes:
                        break
                    
                    # Filter by success if requested
                    if successful_only and not episode.get('solution_found', False):
                        continue
                    
                    episode_id = episode.get('episode_id', f'episode_{episode_count}')
                    success_suffix = "_success" if episode.get('solution_found') else "_fail"
                    output_path = os.path.join(output_dir, f"{episode_id}{success_suffix}.png")
                    
                    try:
                        self.visualize_episode(episode, save_path=output_path)
                        episode_count += 1
                    except Exception as e:
                        print(f"Warning: Failed to visualize episode {episode_id}: {e}")
                        continue
                
            except Exception as e:
                print(f"Warning: Failed to process {pickle_file}: {e}")
                continue
        
        print(f"Generated {episode_count} visualizations in {output_dir}")
    
    def generate_batch_masks(self, data_dir: str, output_dir: str, 
                           max_episodes: int = 10,
                           successful_only: bool = False) -> None:
        """Generate masks for multiple episodes from a data directory.
        
        Args:
            data_dir: Directory containing pickle files
            output_dir: Output directory for masks
            max_episodes: Maximum number of episodes to process
            successful_only: Only process successful episodes
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all pickle files
        pickle_files = glob.glob(os.path.join(data_dir, "*_results.pkl"))
        print(f"Found {len(pickle_files)} data files")
        
        episode_count = 0
        
        for pickle_file in pickle_files:
            if episode_count >= max_episodes:
                break
            
            try:
                with open(pickle_file, 'rb') as f:
                    data = pickle.load(f)
                
                episodes = data.get('episode_results', [])
                for episode in episodes:
                    if episode_count >= max_episodes:
                        break
                    
                    # Filter by success if requested
                    if successful_only and not episode.get('solution_found', False):
                        continue
                    
                    episode_id = episode.get('episode_id', f'episode_{episode_count}')
                    
                    try:
                        # Generate masks
                        masks = self.generate_episode_masks(episode)
                        self.save_masks(masks, output_dir, episode_id)
                        episode_count += 1
                    except Exception as e:
                        print(f"Warning: Failed to generate masks for episode {episode_id}: {e}")
                        continue
                
            except Exception as e:
                print(f"Warning: Failed to process {pickle_file}: {e}")
                continue
        
        print(f"Generated masks for {episode_count} episodes in {output_dir}")


def main():
    """Main entry point for data visualization."""
    parser = argparse.ArgumentParser(description="NAMO Data Visualizer")
    parser.add_argument("--data-dir", type=str,
                        help="Directory containing NAMO data pickle files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for visualizations")
    parser.add_argument("--max-episodes", type=int, default=10,
                        help="Maximum number of episodes to visualize")
    parser.add_argument("--successful-only", action="store_true",
                        help="Only visualize successful episodes")
    parser.add_argument("--single-file", type=str,
                        help="Visualize a single pickle file instead of batch")
    parser.add_argument("--episode-id", type=str,
                        help="Specific episode ID to visualize (requires --single-file)")
    parser.add_argument("--generate-masks", action="store_true",
                        help="Generate 224x224 masks instead of visualizations")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.single_file and not args.data_dir:
        parser.error("Either --data-dir or --single-file must be specified")
    
    visualizer = NAMODataVisualizer()
    
    if args.single_file:
        # Single file visualization
        if not os.path.exists(args.single_file):
            print(f"Error: File {args.single_file} not found")
            return 1
        
        try:
            with open(args.single_file, 'rb') as f:
                data = pickle.load(f)
            
            episodes = data.get('episode_results', [])
            if not episodes:
                print("No episodes found in file")
                return 1
            
            # Find specific episode or use first one
            target_episode = None
            if args.episode_id:
                for episode in episodes:
                    if episode.get('episode_id') == args.episode_id:
                        target_episode = episode
                        break
                if target_episode is None:
                    print(f"Episode {args.episode_id} not found")
                    return 1
            else:
                target_episode = episodes[0]
            
            # Create output filename
            episode_id = target_episode.get('episode_id', 'episode')
            success_suffix = "_success" if target_episode.get('solution_found') else "_fail"
            output_path = os.path.join(args.output_dir, f"{episode_id}{success_suffix}.png")
            
            os.makedirs(args.output_dir, exist_ok=True)
            
            if args.generate_masks:
                # Generate masks instead of visualization
                masks = visualizer.generate_episode_masks(target_episode)
                episode_id = target_episode.get('episode_id', 'episode')
                visualizer.save_masks(masks, args.output_dir, episode_id)
            else:
                # Standard visualization
                visualizer.visualize_episode(target_episode, save_path=output_path)
            
        except Exception as e:
            print(f"Error processing file: {e}")
            return 1
    
    else:
        # Batch processing
        if not os.path.exists(args.data_dir):
            print(f"Error: Directory {args.data_dir} not found")
            return 1
        
        if args.generate_masks:
            # Batch mask generation
            visualizer.generate_batch_masks(
                args.data_dir, 
                args.output_dir,
                max_episodes=args.max_episodes,
                successful_only=args.successful_only
            )
        else:
            # Batch visualization
            visualizer.visualize_batch(
                args.data_dir, 
                args.output_dir,
                max_episodes=args.max_episodes,
                successful_only=args.successful_only
            )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())