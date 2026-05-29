#!/usr/bin/env python3
"""
ML Image Converter Adapter

This adapter makes the UnifiedImageConverter compatible with the existing ML inference
models (ObjectInferenceModel and GoalInferenceModel) without changing their interface.

The adapter translates between the JSON message format expected by ML models and the
ObjectInfo format used by the UnifiedImageConverter.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sage_learning.unified_converter import UnifiedImageConverter, ObjectInfo, create_converter_from_xml
from scipy.spatial.transform import Rotation as R


class MLImageConverterAdapter:
    """Adapter that makes UnifiedImageConverter compatible with ML inference models.
    
    This class provides the same interface as the original json2img.ImageConverter
    but uses the unified converter internally for consistency.
    """
    
    IMG_SIZE = 224  # Match the original interface
    
    def __init__(self, xml_path: str):
        """Initialize the ML adapter.
        
        Args:
            xml_path: Path to MuJoCo XML file (relative paths will be resolved)
        """
        self.xml_path = xml_path
        self.converter = create_converter_from_xml(xml_path)
        
        # Load object sizes from XML for compatibility
        self.object_sizes = self._load_object_sizes_from_xml(xml_path)
        
        # Store world bounds for compatibility
        self.world_bounds = self.converter.world_bounds
        
        # Expose bounds as individual attributes (for compatibility)
        self.x_min, self.x_max, self.y_min, self.y_max = self.world_bounds
        self.WORLD_WIDTH = self.x_max - self.x_min
        self.WORLD_HEIGHT = self.y_max - self.y_min
        self.WORLD_SIZE = self.converter.world_size
        self.SCALE = self.converter.scale
        
        # Store data_point for compatibility with create_object_mask
        self.data_point = None
    
    def _load_object_sizes_from_xml(self, xml_path: str) -> Dict[str, np.ndarray]:
        """Load object sizes from XML file for compatibility."""
        import mujoco
        import os
        
        # Handle relative paths
        if not os.path.isabs(xml_path):
            # Check if path starts with ../ and can be resolved directly first
            if xml_path.startswith("../ml4kp_ktamp"):
                if os.path.exists(xml_path):
                    full_xml_path = os.path.abspath(xml_path)
                else:
                    # Fallback to hardcoded resource path
                    full_xml_path = os.path.join("/common/home/dm1487/robotics_research/ktamp/ml4kp_ktamp/resources/models", xml_path)
            elif os.path.exists(xml_path):
                full_xml_path = os.path.abspath(xml_path)
            else:
                full_xml_path = os.path.join("/common/home/dm1487/robotics_research/ktamp/ml4kp_ktamp/resources/models", xml_path)
        else:
            full_xml_path = xml_path
        
        model = mujoco.MjModel.from_xml_path(full_xml_path)
        geom_sizes = {}
        
        for i in range(model.ngeom):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name is None:
                geom_name = f"geom_{i}"
            geom_sizes[geom_name] = model.geom_size[i]
        
        del model
        return geom_sizes
    
    def print_bounds_info(self):
        """Print debugging information about bounds (for compatibility)."""
        print(f"Environment bounds: x=[{self.x_min:.2f}, {self.x_max:.2f}], y=[{self.y_min:.2f}, {self.y_max:.2f}]")
        print(f"World size: {self.WORLD_WIDTH:.2f} x {self.WORLD_HEIGHT:.2f}")
        print(f"Using square world size: {self.WORLD_SIZE:.2f}")
        print(f"Scale: {self.SCALE:.2f} pixels per world unit")
        print(f"Image center at pixel: ({self.IMG_SIZE//2}, {self.IMG_SIZE//2})")
        
        # Test corner conversions
        corners = [
            (self.x_min, self.y_min, "bottom-left"),
            (self.x_max, self.y_min, "bottom-right"), 
            (self.x_min, self.y_max, "top-left"),
            (self.x_max, self.y_max, "top-right")
        ]
        print("Corner pixel positions:")
        for x, y, name in corners:
            px, py = self.converter.world_to_pixel(x, y)
            print(f"  {name}: world({x:.2f}, {y:.2f}) -> pixel({px}, {py})")
    
    def _world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world to pixel coordinates (for compatibility)."""
        return self.converter.world_to_pixel(x, y)
    
    def pixel_to_world(self, px: int, py: int) -> Tuple[float, float]:
        """Convert pixel to world coordinates (for compatibility)."""
        return self.converter.pixel_to_world(px, py)
    
    def _json_to_object_info_list(self, data_point: Dict[str, Any]) -> List[ObjectInfo]:
        """Convert JSON message format to ObjectInfo list."""
        objects = []
        
        for obj_name, obj_data in data_point['objects'].items():
            # Get position and rotation
            pos = obj_data['position']
            quat = obj_data['quaternion']  # [w, x, y, z] scalar-first format
            
            # Convert quaternion to Euler angle (Z rotation only)
            rotation = R.from_quat(quat, scalar_first=True).as_euler('xyz', degrees=False)[2]  # radians
            
            # Get object sizes (note: NOT doubled like in original ML converter)
            if obj_name in self.object_sizes:
                size_x = self.object_sizes[obj_name][0]  # Half-extent
                size_y = self.object_sizes[obj_name][1]  # Half-extent
            else:
                size_x = size_y = 0.1  # Default
            
            # Determine if static or movable
            is_static = "movable" not in obj_name
            is_reachable = obj_name in data_point.get('reachable_objects', [])
            
            obj_info = ObjectInfo(
                name=obj_name,
                x=pos[0],
                y=pos[1], 
                theta=rotation,
                size_x=size_x,
                size_y=size_y,
                is_static=is_static,
                is_reachable=is_reachable
            )
            objects.append(obj_info)
        
        return objects
    
    def process_datapoint(self, data_point: Dict[str, Any], robot_goal_pos: Tuple[float, float]) -> Dict[str, Any]:
        """Process data point to create multi-channel images (main ML interface).
        
        This method provides the same interface as the original ImageConverter
        but uses the unified converter internally.
        
        Args:
            data_point: JSON message from planning system
            robot_goal_pos: Robot goal position [x, y]
            
        Returns:
            Dictionary with image channels and object center pixels
        """
        # Store data_point for compatibility with create_object_mask
        self.data_point = data_point
        
        # Extract robot position
        robot_pos = data_point['robot']['position']
        robot_position = (robot_pos[0], robot_pos[1], 0.0)  # Assume theta=0 if not provided
        robot_goal = (robot_goal_pos[0], robot_goal_pos[1], 0.0)
        
        # Convert to ObjectInfo list
        objects = self._json_to_object_info_list(data_point)
        
        # Create multi-channel image
        channels = self.converter.create_multi_channel_image(robot_position, robot_goal, objects)
        
        # Get object center pixels (for ML model compatibility)
        obj2center_px = self.converter.get_object_center_pixels(objects)
        
        # Convert quaternions to angles for compatibility
        obj2angle = {}
        for obj in objects:
            obj2angle[obj.name] = np.degrees(obj.theta)  # Convert to degrees for compatibility
        
        # Return in expected format (channels are already float32 in [0,1] range)
        return {
            'robot_image': channels[0:1].transpose(1, 2, 0),  # (H, W, 1)
            'goal_image': channels[1:2].transpose(1, 2, 0),   # (H, W, 1)
            'movable_objects_image': channels[2:3].transpose(1, 2, 0),  # (H, W, 1)
            'static_objects_image': channels[3:4].transpose(1, 2, 0),   # (H, W, 1)
            'reachable_objects_image': channels[4:5].transpose(1, 2, 0), # (H, W, 1)
            'obj2center_px': obj2center_px,
            'obj2angle': obj2angle
        }
    
    def create_object_mask(self, object_name: str) -> np.ndarray:
        """Create a binary mask for a specific object (for compatibility).
        
        Args:
            object_name: Name of the object to create mask for
            
        Returns:
            Binary mask with shape (IMG_SIZE, IMG_SIZE, 1)
        """
        if self.data_point is None:
            raise ValueError("data_point is not set. Call process_datapoint first.")
            
        if object_name not in self.data_point['objects']:
            raise ValueError(f"Object '{object_name}' not found in data_point")
        
        # Get object data
        obj_data = self.data_point['objects'][object_name]
        pos = obj_data['position']
        quat = obj_data['quaternion']
        
        # Convert quaternion to angle
        rotation = R.from_quat(quat, scalar_first=True).as_euler('xyz', degrees=False)[2]
        
        # Get object sizes (note: NOT doubled)
        if object_name in self.object_sizes:
            size_x = self.object_sizes[object_name][0]
            size_y = self.object_sizes[object_name][1]
        else:
            size_x = size_y = 0.1
        
        # Create mask
        mask = np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32)
        self.converter.draw_rotated_box(mask, pos[0], pos[1], size_x, size_y, rotation, 1.0)
        
        # Return in expected format (H, W, 1)
        return mask[:, :, np.newaxis]
    
    def rotate_relative_to_world(self, obj_name: str, angle: float) -> np.ndarray:
        """Rotate object relative to world (for compatibility).
        
        Args:
            obj_name: Object name
            angle: Angle to add in degrees
            
        Returns:
            New quaternion [w, x, y, z] in scalar-first format
        """
        if self.data_point is None:
            raise ValueError("data_point is not set")
            
        obj_angle = self.data_point['objects'][obj_name]['quaternion']
        current_angle = R.from_quat(obj_angle, scalar_first=True).as_euler('xyz', degrees=True)[2]
        final_quat = R.from_euler('xyz', [0, 0, current_angle + angle], degrees=True).as_quat(scalar_first=True)
        return final_quat

    def _convert_datapoint_to_episode_format(self, data_point: Dict[str, Any],
                                              selected_object: str,
                                              robot_goal_pos: Tuple[float, float],
                                              region_goals_sampled: List[Tuple[float, float, float]] = None) -> Dict[str, Any]:
        """Convert JSON data_point format to episode_data format for visualizer.

        This allows us to use the SAME generate_all_masks_highres() method
        for both training and inference.

        Args:
            data_point: JSON message from planning system
            selected_object: Name of the object to use as target
            robot_goal_pos: Robot goal position [x, y]
            region_goals_sampled: List of (x, y, theta) goal samples

        Returns:
            episode_data dict compatible with visualizer.generate_all_masks_highres()
        """
        # Build state_observations from data_point (movable objects only)
        # Static objects (walls) are handled separately via static_object_info
        state_obs = {}
        for obj_name, obj_info in data_point['objects'].items():
            # Skip static objects - they should only appear in static mask, not movable
            if "movable" not in obj_name:
                continue
            pos = obj_info['position']
            quat = obj_info['quaternion']
            theta = R.from_quat(quat, scalar_first=True).as_euler('xyz', degrees=False)[2]
            state_obs[f'{obj_name}_pose'] = [pos[0], pos[1], theta]

        # Add robot pose
        if 'robot' in data_point:
            robot_pos = data_point['robot']['position']
            state_obs['robot_pose'] = [robot_pos[0], robot_pos[1], 0.0]

        # Build static_object_info from object_sizes
        # Include position info for static objects so they're correctly identified
        static_object_info = {}
        for obj_name, obj_info in data_point['objects'].items():
            if obj_name in self.object_sizes:
                info = {
                    'size_x': self.object_sizes[obj_name][0],
                    'size_y': self.object_sizes[obj_name][1],
                }
                # Add position for static objects (walls, non-movable obstacles)
                is_static = "movable" not in obj_name
                if is_static:
                    pos = obj_info['position']
                    quat = obj_info['quaternion']
                    info['pos_x'] = pos[0]
                    info['pos_y'] = pos[1]
                    info['pos_z'] = pos[2] if len(pos) > 2 else 0.0
                    info['quat_w'] = quat[0]
                    info['quat_x'] = quat[1]
                    info['quat_y'] = quat[2]
                    info['quat_z'] = quat[3] if len(quat) > 3 else 0.0
                static_object_info[obj_name] = info

        # Get selected object's current pose for target (inference doesn't have target_goal)
        # We'll use a dummy target - the local_target_goal won't be used at inference
        selected_pos = data_point['objects'][selected_object]['position']
        selected_quat = data_point['objects'][selected_object]['quaternion']
        selected_theta = R.from_quat(selected_quat, scalar_first=True).as_euler('xyz', degrees=False)[2]

        # Build episode_data dict
        episode_data = {
            'state_observations': [state_obs],
            'static_object_info': static_object_info,
            'action_sequence': [{
                'object_id': selected_object,
                'target': (selected_pos[0], selected_pos[1], selected_theta)  # Dummy target
            }],
            'robot_goal': (robot_goal_pos[0], robot_goal_pos[1], 0.0),
            'world_bounds': self.world_bounds,
            'algorithm_stats': {
                'region_goals_sampled': region_goals_sampled,
                'region_goal_used': (robot_goal_pos[0], robot_goal_pos[1], 0.0) if not region_goals_sampled else None
            }
        }

        return episode_data

    @staticmethod
    def _robot_half_extent_from_config(namo_config_path,
                                       default=(0.035, 0.035)) -> Tuple[float, float]:
        """Parse planning.robot_size [hx, hy] (meters) from the namo config YAML.

        Falls back to the 7 cm car footprint (0.035, 0.035) when the config is
        missing/unreadable. The wavefront exporter treats the config value as
        authoritative; this only needs to populate object_info['robot'] so the
        exporter's presence gate passes.
        """
        if namo_config_path:
            try:
                import yaml
                with open(namo_config_path) as f:
                    cfg = yaml.safe_load(f) or {}
                rs = (cfg.get('planning', {}) or {}).get('robot_size')
                if rs and len(rs) >= 2:
                    return float(rs[0]), float(rs[1])
            except Exception:
                pass
        return default

    def create_local_masks(self, data_point: Dict[str, Any],
                           selected_object: str,
                           robot_goal_pos: Tuple[float, float],
                           region_goals_sampled: List[Tuple[float, float, float]] = None,
                           crop_size_meters: float = 5.0,
                           highres_size: int = 1024,
                           output_size: int = 224,
                           goal_circle_radius: float = 0.1,
                           namo_config_path: str = None,
                           episode_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create local masks centered on selected object using the SAME method as training.

        This method converts the data_point to episode format and calls
        visualizer.generate_all_masks_highres() to ensure identical mask generation.

        Args:
            data_point: JSON message from planning system
            selected_object: Name of the object to center on
            robot_goal_pos: Robot goal position [x, y]
            region_goals_sampled: List of (x, y, theta) goal samples for goal_region mask
            crop_size_meters: Size of local crop in meters (default: 5.0)
            highres_size: High-res render size (default: 2048)
            output_size: Output mask size (default: 224)
            goal_circle_radius: Radius of goal region circles in meters (default: 0.1)

        Returns:
            Dictionary containing:
            - 'local_static': Static obstacles mask (H, W, 1)
            - 'local_movable': Other movable objects mask (H, W, 1)
            - 'local_target_object': Selected object mask (H, W, 1)
            - 'local_goal_region': Goal region circles mask (H, W, 1)
            - 'object_center': (x, y) of selected object
            - 'object_theta': Rotation of selected object
            - 'crop_size_meters': Crop size used
            - 'resolution': Meters per pixel
        """
        from sage_learning.visualizer import NAMODataVisualizer

        # Store data_point for compatibility
        self.data_point = data_point

        # Prefer a caller-supplied episode_data built DIRECTLY from the live env
        # (same source training uses: env.get_object_info()/get_observation()).
        # This bypasses the lossy JSON round-trip entirely. Fall back to building
        # episode_data from the JSON data_point when no env was available.
        if episode_data is None:
            episode_data = self._convert_datapoint_to_episode_format(
                data_point, selected_object, robot_goal_pos, region_goals_sampled
            )
        # The unified wavefront region computation needs the scene XML path
        # (gated on episode_data['xml_file'] in generate_all_masks_highres).
        if 'xml_file' not in episode_data or not episode_data['xml_file']:
            episode_data['xml_file'] = data_point.get('xml_path')
        # WavefrontSnapshotExporter requires a 'robot' entry in object_info
        # (hard gate). Inference builds object_info from movable/static objects
        # only and omits the robot, so the exporter raises and falls back to a
        # legacy BFS with the WRONG default robot size — leaving robot_region /
        # goal_sample_region blank. Inject it (the exporter reads the true
        # footprint from namo_config_path; these sizes are the fallback).
        sinfo = episode_data.get('static_object_info')
        if isinstance(sinfo, dict) and 'robot' not in sinfo:
            hx, hy = self._robot_half_extent_from_config(namo_config_path)
            sinfo['robot'] = {'size_x': hx, 'size_y': hy}

        # Use the SAME visualizer method as training. The v3 cropped models are
        # trained on the TIGHT crop (crop_prefix="local_tight"), so request that
        # crop here. generate_all_masks_highres returns dual crops keyed
        # 'local_wide'/'local_tight'; we consume only tight. (Commit 2993c76
        # moved the visualizer to this dual-crop API; this adapter follows it.)
        visualizer = NAMODataVisualizer(namo_config_path=namo_config_path)
        result = visualizer.generate_all_masks_highres(
            episode_data,
            highres_size=highres_size,
            global_output_size=output_size,
            local_output_size=output_size,
            tight_crop_size_meters=crop_size_meters,
            goal_circle_radius=goal_circle_radius
        )

        # Skip if no region_goals_sampled available
        if result is None:
            return None

        # Extract local masks and convert to (H, W, 1) format for inference
        # compatibility. Visualizer keys are 'local_tight_<name>'; downstream
        # (goal_inference_model) expects 'local_<name>', so strip the crop tag.
        local_masks = {}
        if result['local_tight'] is not None:
            for name, mask in result['local_tight'].items():
                consumer_key = name.replace('local_tight_', 'local_', 1)
                local_masks[consumer_key] = mask[:, :, np.newaxis]

        # Add metadata
        if result['local_tight_metadata'] is not None:
            meta = result['local_tight_metadata']
            local_masks['object_center'] = meta['object_center']
            local_masks['object_theta'] = meta['object_theta']
            local_masks['crop_size_meters'] = meta['crop_size_meters']
            local_masks['resolution'] = meta['resolution']

        return local_masks

    def pixel_to_world_local(self, px: int, py: int,
                             object_center: Tuple[float, float],
                             crop_size_meters: float = 5.0,
                             output_size: int = 224) -> Tuple[float, float]:
        """Convert local pixel coordinates to world coordinates.

        Args:
            px, py: Pixel coordinates in local mask
            object_center: (x, y) of the object the mask is centered on
            crop_size_meters: Size of crop in meters
            output_size: Size of output mask in pixels

        Returns:
            (world_x, world_y) coordinates
        """
        resolution = crop_size_meters / output_size
        center_px = output_size / 2  # 112 for 224x224

        world_x = object_center[0] + (px - center_px) * resolution
        # No Y-flip needed: in this mask generation, world_to_highres maps
        # higher world Y to higher py (row index). So the inverse is the same.
        world_y = object_center[1] + (py - center_px) * resolution

        return world_x, world_y


# For backward compatibility, create an alias
ImageConverter = MLImageConverterAdapter