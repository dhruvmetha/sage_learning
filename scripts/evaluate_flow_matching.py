#!/usr/bin/env python3
"""
Evaluate Flow Matching Model on Region Opening Task

This script evaluates a trained flow matching model by:
1. Loading test samples from the HDF5 dataset (10% held-out split)
2. For each sample, querying the model for N samples (default: 32)
3. Mapping predictions to motion primitive slots using existing ML-primitive alignment
4. Majority voting AFTER mapping (vote-ranked primitives only; no fallback)
5. Measuring success (region opening), pushes-to-success, and time-to-success

Usage:
    python evaluate_flow_matching.py \
        --model-path /path/to/outputs/2025-12-28/max_abs \
        --num-samples 32 \
        --max-test-envs 100

Author: Generated for flow matching evaluation
"""

import sys
import os
import argparse
import json
import time
import math
import random
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter, defaultdict
import pickle
from contextlib import contextmanager

# Add necessary paths
SAGE_LEARNING_PATH = "/common/users/tdn39/Robotics/Mujoco/sage_learning"
NAMO_CPP_PATH = "/common/users/tdn39/Robotics/Mujoco/namo_cpp"
NAMO_PYTHON_PATH = os.path.join(NAMO_CPP_PATH, "python")
NAMO_VIZ_PATH = os.path.join(NAMO_PYTHON_PATH, "namo", "visualization")

for path in [SAGE_LEARNING_PATH, NAMO_PYTHON_PATH, NAMO_VIZ_PATH]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Add build_python for namo_rl
sys.path.append(os.path.join(NAMO_CPP_PATH, "build_python"))

import numpy as np
import h5py
import torch
from tqdm import tqdm

# NAMO imports
try:
    import namo_rl
except ImportError as e:
    print(f"Error importing namo_rl: {e}")
    print("Make sure to add build_python to PYTHONPATH")
    sys.exit(1)


@dataclass
class EvalResult:
    """Result from evaluating on a single environment."""
    env_path: str
    success: bool
    pushes_to_success: int = 0
    time_to_success_ms: float = 0.0  # Time for all push attempts up to success
    total_pushes_attempted: int = 0
    total_time_ms: float = 0.0
    error_message: Optional[str] = None
    primitive_votes: Optional[Dict[int, int]] = None
    chosen_primitive_idx: Optional[int] = None
    # Timing breakdown
    env_init_ms: float = 0.0
    model_inference_ms: float = 0.0
    primitive_mapping_ms: float = 0.0
    simulation_ms: float = 0.0  # Total simulation time across all push attempts
    # Difficulty annotation (if available)
    difficulty_label: Optional[str] = None
    difficulty_score: Optional[float] = None
    # Collision accounting (per-attempt counts)
    collision_total: int = 0
    collision_static_only: int = 0
    collision_movable_only: int = 0
    collision_both: int = 0


@dataclass 
class EvalStats:
    """Aggregate statistics from evaluation."""
    total_envs: int = 0
    successful_envs: int = 0
    failed_envs: int = 0
    success_rate: float = 0.0
    avg_pushes_to_success: float = 0.0
    avg_time_to_success_ms: float = 0.0
    median_pushes_to_success: float = 0.0
    median_time_to_success_ms: float = 0.0
    avg_env_init_ms: float = 0.0
    avg_model_inference_ms: float = 0.0
    avg_primitive_mapping_ms: float = 0.0
    avg_simulation_ms: float = 0.0
    avg_collision_total: float = 0.0
    median_collision_total: float = 0.0
    avg_collision_static_only: float = 0.0
    median_collision_static_only: float = 0.0
    avg_collision_movable_only: float = 0.0
    median_collision_movable_only: float = 0.0
    avg_collision_both: float = 0.0
    median_collision_both: float = 0.0
    total_evaluation_time_sec: float = 0.0
    difficulty_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)
    results: List[EvalResult] = field(default_factory=list)


def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_median(values: List[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _summarize_results(results: List[EvalResult]) -> Dict[str, float]:
    """Summarize metrics for a subset of evaluation results."""
    total = len(results)
    successful = [r for r in results if r.success]
    failed = total - len(successful)

    pushes = [r.pushes_to_success for r in successful]
    times = [r.time_to_success_ms for r in successful]

    collisions_total = [r.collision_total for r in results]
    collisions_static_only = [r.collision_static_only for r in results]
    collisions_movable_only = [r.collision_movable_only for r in results]
    collisions_both = [r.collision_both for r in results]

    return {
        "total_envs": total,
        "successful_envs": len(successful),
        "failed_envs": failed,
        "success_rate": (len(successful) / total) if total else 0.0,
        "avg_pushes_to_success": _safe_mean(pushes),
        "median_pushes_to_success": _safe_median(pushes),
        "avg_time_to_success_ms": _safe_mean(times),
        "median_time_to_success_ms": _safe_median(times),
        "avg_collision_total": _safe_mean(collisions_total),
        "median_collision_total": _safe_median(collisions_total),
        "avg_collision_static_only": _safe_mean(collisions_static_only),
        "median_collision_static_only": _safe_median(collisions_static_only),
        "avg_collision_movable_only": _safe_mean(collisions_movable_only),
        "median_collision_movable_only": _safe_median(collisions_movable_only),
        "avg_collision_both": _safe_mean(collisions_both),
        "median_collision_both": _safe_median(collisions_both),
    }


def get_test_indices(h5_path: str, train_split: float = 0.9) -> List[int]:
    """Get test indices from HDF5 dataset using same logic as training.
    
    For 90/10 split: train=90%, val=10%, test=0%
    But we'll use the val portion as "test" for evaluation.
    """
    with h5py.File(h5_path, 'r') as h5f:
        n_samples = len(h5f['local_static'])
    
    indices = list(range(n_samples))
    random.Random(42).shuffle(indices)  # Same seed as training
    
    # With train_split=0.9, the remaining 10% was used as val
    train_end = int(n_samples * train_split)
    test_indices = indices[train_end:]  # Last 10%
    
    print(f"Dataset: {n_samples} total samples")
    print(f"Test split: {len(test_indices)} samples ({100*(1-train_split):.0f}%)")
    
    return test_indices


def load_test_sample(h5_path: str, idx: int) -> Dict[str, np.ndarray]:
    """Load a single test sample from HDF5."""
    with h5py.File(h5_path, 'r') as h5f:
        sample = {
            'local_static': h5f['local_static'][idx],
            'local_movable': h5f['local_movable'][idx],
            'local_target_object': h5f['local_target_object'][idx],
            'local_robot_region': h5f['local_robot_region'][idx],
            'local_goal_sample_region': h5f['local_goal_sample_region'][idx],
            'target_goal_pose_deltas_obj': h5f['target_goal_pose_deltas_obj'][idx],
            'local_object_theta': h5f['local_object_theta'][idx] if 'local_object_theta' in h5f else np.array([0.0]),
        }
        
        # Also load metadata if available
        if 'xml_path' in h5f:
            sample['xml_path'] = h5f['xml_path'][idx]
        if 'object_id' in h5f:
            sample['object_id'] = h5f['object_id'][idx]
            
    return sample


def _decode_h5_str(value: Any) -> str:
    """Decode HDF5 string/bytes fields into a Python string."""
    if isinstance(value, bytes):
        return value.decode('utf-8')
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        item = value.flat[0]
        if isinstance(item, bytes):
            return item.decode('utf-8')
        if isinstance(item, str):
            return item
        return str(item)
    return str(value)


@contextmanager
def _with_namo_cwd():
    """Temporarily switch CWD to NAMO_CPP_PATH for relative primitive paths."""
    prev_cwd = os.getcwd()
    if prev_cwd == NAMO_CPP_PATH:
        yield
        return
    os.chdir(NAMO_CPP_PATH)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def get_env_metadata_from_h5(h5_path: str, indices: List[int]) -> List[Dict]:
    """Extract environment metadata (xml_path, object_id, robot_goal) from HDF5 for test samples."""
    metadata = []
    
    with h5py.File(h5_path, 'r') as h5f:
        # Check what metadata is available (using actual keys from the HDF5)
        has_xml_file = 'xml_file' in h5f
        has_action_object_ids = 'action_object_ids' in h5f
        has_robot_goal = 'robot_goal' in h5f
        has_local_object_center = 'local_object_center' in h5f
        has_local_object_theta = 'local_object_theta' in h5f
        has_difficulty_label = 'difficulty_label' in h5f
        has_difficulty_score = 'difficulty_score' in h5f
        
        print(
            f"HDF5 metadata available: xml_file={has_xml_file}, action_object_ids={has_action_object_ids}, "
            f"robot_goal={has_robot_goal}, difficulty_label={has_difficulty_label}, difficulty_score={has_difficulty_score}"
        )
        
        for idx in indices:
            meta = {'h5_idx': idx}
            
            if has_xml_file:
                xml_val = h5f['xml_file'][idx]
                # Shape is (1,) with bytes inside
                if isinstance(xml_val, np.ndarray) and len(xml_val) > 0:
                    xml_bytes = xml_val[0]
                    if isinstance(xml_bytes, bytes):
                        meta['xml_path'] = xml_bytes.decode('utf-8')
                    else:
                        meta['xml_path'] = str(xml_bytes)
                elif isinstance(xml_val, bytes):
                    meta['xml_path'] = xml_val.decode('utf-8')
                    
            if has_action_object_ids:
                obj_val = h5f['action_object_ids'][idx]
                # Shape is (1,) with bytes inside
                if isinstance(obj_val, np.ndarray) and len(obj_val) > 0:
                    obj_bytes = obj_val[0]
                    if isinstance(obj_bytes, bytes):
                        meta['object_id'] = obj_bytes.decode('utf-8')
                    else:
                        meta['object_id'] = str(obj_bytes)
                elif isinstance(obj_val, bytes):
                    meta['object_id'] = obj_val.decode('utf-8')
                    
            if has_robot_goal:
                robot_goal = h5f['robot_goal'][idx]
                meta['robot_goal'] = (float(robot_goal[0]), float(robot_goal[1]), float(robot_goal[2]))
                
            if has_local_object_center:
                center = h5f['local_object_center'][idx]
                meta['object_center'] = (float(center[0]), float(center[1]))
                
            if has_local_object_theta:
                theta = h5f['local_object_theta'][idx]
                meta['object_theta'] = float(theta[0]) if isinstance(theta, np.ndarray) else float(theta)

            if has_difficulty_label:
                meta['difficulty_label'] = _decode_h5_str(h5f['difficulty_label'][idx])

            if has_difficulty_score:
                try:
                    score_val = h5f['difficulty_score'][idx]
                    if isinstance(score_val, np.ndarray):
                        score_val = score_val.flat[0]
                    meta['difficulty_score'] = float(score_val)
                except Exception:
                    meta['difficulty_score'] = None
                
            metadata.append(meta)
    
    return metadata


class FlowMatchingEvaluator:
    """Evaluator for flow matching models on region opening task."""
    
    def __init__(
        self,
        model_path: str,
        h5_path: str,
        primitive_data_dir: Optional[str] = None,
        num_samples: int = 32,
        num_steps: int = 10,  # ODE solver steps (fewer = faster)
        sampler_method: Optional[str] = None,
        match_max_per_call: int = 8,
        match_position_tolerance: float = 0.05,
        match_angle_tolerance: float = 0.1,
        match_angle_weight: float = 0.5,
        goals_per_region: int = 5,
        allow_collisions: bool = True,
        device: str = "cuda",
        config_file: str = None,  # Will be set to default if None
        verbose: bool = False
    ):
        self.model_path = Path(model_path)  # Already absolute from main()
        self.h5_path = h5_path
        if primitive_data_dir is None:
            self.primitive_data_dir = Path(NAMO_CPP_PATH) / "data"
        else:
            self.primitive_data_dir = Path(primitive_data_dir)  # Already absolute from main()
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.sampler_method = sampler_method
        self.match_max_per_call = match_max_per_call
        self.match_position_tolerance = match_position_tolerance
        self.match_angle_tolerance = match_angle_tolerance
        self.match_angle_weight = match_angle_weight
        self.goals_per_region = goals_per_region
        self.allow_collisions = allow_collisions
        self.device = device
        self.verbose = verbose
        
        # Set default config file path (absolute)
        if config_file is None:
            self.config_file = str(Path(NAMO_CPP_PATH) / "config" / "namo_config_complete_skill15.yaml")
        else:
            self.config_file = config_file
        
        # Load model
        self._load_model()
        
        # Load primitive database
        self._load_primitives()

        # Initialize goal strategies (reuse existing mapping logic)
        self._init_goal_strategies()
        
    def _load_model(self):
        """Load the trained flow matching model."""
        print(f"Loading model from {self.model_path}...")
        
        from ktamp_learning import GoalVectorInferenceModel
        
        self.model = GoalVectorInferenceModel(
            model_path=str(self.model_path),
            device=self.device,
            sampler_method=self.sampler_method,
            num_steps=self.num_steps  # Use configurable steps
        )
        self.device = getattr(self.model, "device", self.device)
        print(f"✓ Model loaded successfully (num_steps={self.num_steps})")

    def _init_goal_strategies(self):
        """Initialize goal generation/mapping strategies used by existing planners."""
        from namo.strategies.primitive_goal_strategy import PrimitiveGoalStrategy, MLPrimitiveGoalStrategy
        from namo.strategies.ml_strategies import MLGoalSelectionStrategy

        self.primitive_goal_strategy = PrimitiveGoalStrategy(
            data_dir=str(self.primitive_data_dir),
            verbose=self.verbose
        )

        # ML goal inference (uses GoalVectorInferenceModel when preloaded)
        self.ml_goal_strategy = MLGoalSelectionStrategy(
            goal_model_path=str(self.model_path),
            samples=self.num_samples,
            device=self.device,
            min_goals_threshold=1,
            verbose=self.verbose,
            preloaded_model=self.model,
            goals_per_region=self.goals_per_region
        )

        # ML-to-primitive alignment strategy (for shared mapping helpers)
        self.ml_primitive_strategy = MLPrimitiveGoalStrategy(
            goal_model_path=str(self.model_path),
            primitive_data_dir=str(self.primitive_data_dir),
            samples=self.num_samples,
            device=self.device,
            match_position_tolerance=self.match_position_tolerance,
            match_angle_tolerance=self.match_angle_tolerance,
            angle_weight=self.match_angle_weight,
            max_matches=self.match_max_per_call,
            min_goals_threshold=1,
            verbose=self.verbose,
            preloaded_model=self.model,
            goals_per_region=self.goals_per_region
        )

    def _build_json_message(
        self,
        env: namo_rl.RLEnvironment,
        object_id: str,
        xml_path: str,
        robot_goal: Tuple[float, float, float]
    ) -> Dict[str, Any]:
        """Build full JSON message for GoalVectorInferenceModel (includes static + movable objects)."""
        obs = env.get_observation()
        robot_pose = obs.get('robot_pose')
        if robot_pose is None or len(robot_pose) < 3:
            raise ValueError("robot_pose missing from observation")

        from scipy.spatial.transform import Rotation as R

        objects_dict = {}
        for key, value in obs.items():
            if key.endswith('_pose') and key != 'robot_pose':
                obj_name = key[:-5]
                if len(value) >= 3:
                    quat = R.from_euler('xyz', [0, 0, value[2]], degrees=False).as_quat(scalar_first=True)
                    objects_dict[obj_name] = {
                        "position": [float(value[0]), float(value[1]), float(value[2])],
                        "quaternion": [float(q) for q in quat],
                    }

        # Add static objects with positions from cached object info
        try:
            static_info = env.get_object_info()
        except Exception:
            static_info = {}

        for obj_name, info in static_info.items():
            if obj_name == "robot" or obj_name in objects_dict:
                continue
            if "pos_x" not in info or "pos_y" not in info:
                continue
            quat = [
                float(info.get("quat_w", 1.0)),
                float(info.get("quat_x", 0.0)),
                float(info.get("quat_y", 0.0)),
                float(info.get("quat_z", 0.0)),
            ]
            objects_dict[obj_name] = {
                "position": [float(info.get("pos_x", 0.0)), float(info.get("pos_y", 0.0)), float(info.get("pos_z", 0.0))],
                "quaternion": quat,
            }

        # Ensure target object is included
        if object_id not in objects_dict:
            pose_key = f"{object_id}_pose"
            if pose_key in obs and len(obs[pose_key]) >= 3:
                pose = obs[pose_key]
                quat = R.from_euler('xyz', [0, 0, pose[2]], degrees=False).as_quat(scalar_first=True)
                objects_dict[object_id] = {
                    "position": [float(pose[0]), float(pose[1]), float(pose[2])],
                    "quaternion": [float(q) for q in quat],
                }
            else:
                raise ValueError(f"Object {object_id} not found in observation")

        reachable_objects = []
        try:
            reachable_objects = env.get_reachable_objects()
        except Exception:
            reachable_objects = []

        return {
            "xml_path": xml_path,
            "robot_goal": [float(robot_goal[0]), float(robot_goal[1])],
            "reachable_objects": reachable_objects,
            "robot": {
                "position": [float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])]
            },
            "objects": objects_dict,
        }

    def _align_ml_goals_to_primitives(
        self,
        ml_goals: List[Any],
        primitive_goals: List[List[Any]]
    ) -> Tuple[List[List[Any]], Dict[Tuple[int, int], int]]:
        """Align ML goals to primitive slots using existing MLPrimitiveGoalStrategy logic."""
        from namo.strategies.goal_selection_strategy import Goal
        from collections import defaultdict

        if not primitive_goals:
            return [], {}

        max_depth = len(primitive_goals[0]) if primitive_goals and primitive_goals[0] else 0
        aligned_goals: List[List[Optional[Goal]]] = [
            [None for _ in range(max_depth)]
            for _ in range(len(primitive_goals))
        ]

        slot_metadata = self.ml_primitive_strategy._build_slot_metadata(primitive_goals)
        slot_accumulators = defaultdict(lambda: {"count": 0, "goal": None})

        for ml_goal in ml_goals:
            best_slot = None
            best_score = None
            for slot_id, (_, _, primitive_goal) in enumerate(slot_metadata):
                pos_err, ang_err = self.ml_primitive_strategy._goal_error(primitive_goal, ml_goal)
                score = pos_err + self.ml_primitive_strategy.angle_weight * ang_err
                if best_score is None or score < best_score:
                    best_score = score
                    best_slot = slot_id

            if best_slot is None:
                continue

            acc = slot_accumulators[best_slot]
            acc["count"] += 1
            if acc["goal"] is None:
                _, _, primitive_goal = slot_metadata[best_slot]
                acc["goal"] = primitive_goal

        for slot_id, data in slot_accumulators.items():
            edge_idx, depth_idx, primitive_goal = slot_metadata[slot_id]
            aligned_goals[edge_idx][depth_idx] = Goal(
                x=primitive_goal.x,
                y=primitive_goal.y,
                theta=primitive_goal.theta,
                score=data["count"]
            )

        vote_counts = {
            (edge_idx, depth_idx): int(goal.score)
            for edge_idx, edge_goals in enumerate(aligned_goals)
            for depth_idx, goal in enumerate(edge_goals)
            if goal is not None
        }

        return aligned_goals, vote_counts

    def _rank_primitives_from_votes(
        self,
        shape_type: str,
        vote_counts: Dict[Tuple[int, int], int]
    ) -> List[Tuple[int, int]]:
        """Rank primitives by votes only (no fallback to all primitives)."""
        if not vote_counts:
            return []

        ranked = sorted(
            vote_counts.items(),
            key=lambda item: (-item[1], item[0][1], item[0][0])  # Votes desc, depth asc
        )
        return [prim for prim, _ in ranked]

    @staticmethod
    def _quat_to_yaw(w: float, x: float, y: float, z: float) -> float:
        """Convert quaternion to yaw angle (rotation about Z)."""
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _obb_axes(theta: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        c = math.cos(theta)
        s = math.sin(theta)
        return (c, s), (-s, c)

    @classmethod
    def _obb_intersect(
        cls,
        center_a: Tuple[float, float],
        half_a: Tuple[float, float],
        theta_a: float,
        center_b: Tuple[float, float],
        half_b: Tuple[float, float],
        theta_b: float
    ) -> bool:
        axes_a = cls._obb_axes(theta_a)
        axes_b = cls._obb_axes(theta_b)
        t_x = center_b[0] - center_a[0]
        t_y = center_b[1] - center_a[1]

        for axis in (axes_a[0], axes_a[1], axes_b[0], axes_b[1]):
            dist = abs(t_x * axis[0] + t_y * axis[1])
            ra = (
                half_a[0] * abs(axis[0] * axes_a[0][0] + axis[1] * axes_a[0][1])
                + half_a[1] * abs(axis[0] * axes_a[1][0] + axis[1] * axes_a[1][1])
            )
            rb = (
                half_b[0] * abs(axis[0] * axes_b[0][0] + axis[1] * axes_b[0][1])
                + half_b[1] * abs(axis[0] * axes_b[1][0] + axis[1] * axes_b[1][1])
            )
            if dist > ra + rb:
                return False

        return True

    def _build_collision_context(self, env: namo_rl.RLEnvironment) -> Dict[str, Any]:
        """Build cached object geometry for collision checks."""
        object_info = env.get_object_info()
        static_objects = []
        static_names = set()
        object_sizes = {}

        for name, info in object_info.items():
            size_x = info.get("size_x")
            size_y = info.get("size_y")
            if size_x is not None and size_y is not None:
                object_sizes[name] = (float(size_x), float(size_y))

            if "pos_x" in info and "pos_y" in info and "quat_w" in info:
                yaw = self._quat_to_yaw(
                    float(info.get("quat_w", 1.0)),
                    float(info.get("quat_x", 0.0)),
                    float(info.get("quat_y", 0.0)),
                    float(info.get("quat_z", 0.0)),
                )
                static_objects.append(
                    (
                        name,
                        float(info.get("pos_x", 0.0)),
                        float(info.get("pos_y", 0.0)),
                        yaw,
                        float(info.get("size_x", 0.0)),
                        float(info.get("size_y", 0.0)),
                    )
                )
                static_names.add(name)

        return {
            "object_sizes": object_sizes,
            "static_objects": static_objects,
            "static_names": static_names,
        }

    def _detect_collisions(
        self,
        env: namo_rl.RLEnvironment,
        object_id: str,
        step_result: Optional[Any],
        collision_context: Optional[Dict[str, Any]]
    ) -> Tuple[bool, bool]:
        """Detect collisions with static/movable objects (best-effort)."""
        static_hit = False
        movable_hit = False

        if step_result is not None and hasattr(step_result, "info") and step_result.info:
            collision_object = step_result.info.get("collision_object")
            if collision_object:
                static_names = collision_context.get("static_names") if collision_context else set()
                if collision_object in static_names:
                    static_hit = True
                elif collision_object != "robot":
                    movable_hit = True

        if collision_context is None:
            return static_hit, movable_hit

        obs = env.get_observation()
        pose_key = f"{object_id}_pose"
        if pose_key not in obs:
            return static_hit, movable_hit

        target_pose = obs[pose_key]
        size = collision_context["object_sizes"].get(object_id)
        if not size:
            return static_hit, movable_hit

        target_center = (float(target_pose[0]), float(target_pose[1]))
        target_theta = float(target_pose[2])
        target_half = (float(size[0]), float(size[1]))

        for name, x, y, yaw, half_x, half_y in collision_context["static_objects"]:
            if name == object_id:
                continue
            if self._obb_intersect(
                target_center,
                target_half,
                target_theta,
                (x, y),
                (half_x, half_y),
                yaw,
            ):
                static_hit = True
                if movable_hit:
                    return static_hit, movable_hit
                break

        for key, value in obs.items():
            if not key.endswith("_pose") or key == "robot_pose":
                continue
            other_name = key[:-5]
            if other_name == object_id:
                continue
            other_size = collision_context["object_sizes"].get(other_name)
            if not other_size:
                continue
            if self._obb_intersect(
                target_center,
                target_half,
                target_theta,
                (float(value[0]), float(value[1])),
                (float(other_size[0]), float(other_size[1])),
                float(value[2]),
            ):
                movable_hit = True
                if static_hit:
                    return static_hit, movable_hit
                break

        return static_hit, movable_hit
        
    def _infer_from_h5_masks(
        self,
        h5_idx: int,
        fallback_object_pose: Optional[Tuple[float, float, float]] = None
    ) -> Tuple[List[Dict], Tuple[float, float, float]]:
        """Run inference using pre-computed masks from HDF5 (fast path).
        
        This uses the exact same masks the model was trained on.
        
        Args:
            h5_idx: Index in HDF5 dataset
            
        Returns:
            Tuple of (predictions, object_pose) where predictions is list of goal dicts
        """
        if not getattr(self.model, 'use_local', True):
            raise ValueError("use_h5_masks requires a local-mask model (use_local=true).")

        with h5py.File(self.h5_path, 'r') as f:
            # Load pre-computed local masks (shape: H, W)
            local_static = f['local_static'][h5_idx]
            local_movable = f['local_movable'][h5_idx] 
            local_target_object = f['local_target_object'][h5_idx]
            local_robot_region = f['local_robot_region'][h5_idx] if 'local_robot_region' in f else np.zeros_like(local_static)
            local_goal_sample_region = f['local_goal_sample_region'][h5_idx] if 'local_goal_sample_region' in f else np.zeros_like(local_static)
            
            # Load metadata
            obj_center = f['local_object_center'][h5_idx] if 'local_object_center' in f else None
            obj_theta = f['local_object_theta'][h5_idx] if 'local_object_theta' in f else None
        
        # Add channel dimension if needed and stack (H, W) -> (H, W, 1)
        def ensure_channel(arr):
            if arr.ndim == 2:
                return arr[:, :, np.newaxis]
            return arr
        
        input_channels = [
            ensure_channel(local_static),
            ensure_channel(local_movable),
            ensure_channel(local_target_object),
            ensure_channel(local_robot_region),
            ensure_channel(local_goal_sample_region),
        ]
        
        # Add coord grid if model was trained with it
        if getattr(self.model, 'use_coord_grid', False):
            orig_size = input_channels[0].shape[0]
            ys, xs = np.meshgrid(np.linspace(0, 1, orig_size),
                                 np.linspace(0, 1, orig_size),
                                 indexing='ij')
            coord_grid = np.stack([xs, ys], axis=-1).astype(np.float32)
            input_channels.append(coord_grid)
        
        inp_for_model = np.concatenate(input_channels, axis=-1)
        inp_for_model = self.model.transform(inp_for_model).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            pose_samples = self.model.model.sample_pose(
                inp_for_model,
                num_samples=self.num_samples,
                num_steps=self.num_steps,
                denormalize=True
            ).cpu().numpy()
        
        # Convert to goal dictionaries
        from scipy.spatial.transform import Rotation as R
        object_theta = None
        obj_x = None
        obj_y = None
        if obj_theta is not None:
            if isinstance(obj_theta, np.ndarray) and obj_theta.size > 0:
                object_theta = float(obj_theta.flat[0])
            else:
                object_theta = float(obj_theta)
        if obj_center is not None and isinstance(obj_center, np.ndarray) and obj_center.size >= 2:
            obj_x, obj_y = float(obj_center.flat[0]), float(obj_center.flat[1])

        if (obj_x is None or obj_y is None or object_theta is None) and fallback_object_pose is not None:
            if self.verbose:
                print("Warning: local_object_center/theta missing in H5, using environment pose fallback")
            obj_x, obj_y, object_theta = fallback_object_pose

        if obj_x is None or obj_y is None or object_theta is None:
            raise ValueError("Missing local_object_center/local_object_theta in H5 and no fallback pose provided.")
        
        cos_theta = np.cos(object_theta)
        sin_theta = np.sin(object_theta)
        
        valid_goals = []
        for i, delta_local in enumerate(pose_samples):
            dx_local, dy_local, dtheta_local = delta_local
            
            # Transform local delta to world delta
            dx_world = dx_local * cos_theta - dy_local * sin_theta
            dy_world = dx_local * sin_theta + dy_local * cos_theta
            
            goal_x = obj_x + dx_world
            goal_y = obj_y + dy_world
            goal_theta = object_theta + dtheta_local
            goal_theta = np.arctan2(np.sin(goal_theta), np.cos(goal_theta))
            
            valid_goals.append({
                'x': goal_x,
                'y': goal_y,
                'theta': goal_theta,
                'delta_local': (dx_local, dy_local, dtheta_local),
            })
        
        return valid_goals, (obj_x, obj_y, object_theta)
        
    def _load_primitives(self):
        """Load motion primitives for each shape type."""
        from namo.strategies.primitive_goal_strategy import MotionPrimitiveLoader
        
        self.primitives = {}
        for shape in ['square', 'tall', 'wide']:
            filepath = self.primitive_data_dir / f"motion_primitives_15_{shape}.dat"
            if filepath.exists():
                self.primitives[shape] = MotionPrimitiveLoader.load_primitives(str(filepath))
                print(f"✓ Loaded {len(self.primitives[shape])} primitives for {shape}")
            else:
                print(f"⚠ Primitive file not found: {filepath}")
                
    def _get_shape_type(self, object_id: str, env: namo_rl.RLEnvironment) -> str:
        """Determine object shape type (square/tall/wide)."""
        object_info = env.get_object_info()
        
        if object_id not in object_info:
            return 'square'  # Default
            
        info = object_info[object_id]
        
        # Get dimensions
        if 'size_x' in info and 'size_y' in info:
            x, y = info['size_x'], info['size_y']
        elif 'width' in info and 'height' in info:
            x, y = info['width'], info['height']
        else:
            return 'square'
            
        if x <= 0 or y <= 0:
            return 'square'
            
        ratio = max(x, y) / min(x, y)
        
        if ratio < 1.05:
            return 'square'
        elif x > y:
            return 'wide'
        else:
            return 'tall'
            
    def _map_predictions_to_primitives(
        self,
        predictions: List[Dict],
        object_pose: Tuple[float, float, float],
        shape_type: str
    ) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
        """Map model predictions to nearest primitives via majority voting.
        
        Args:
            predictions: List of goal predictions from model
            object_pose: Current object pose (x, y, theta)
            shape_type: Object shape type for primitive selection
            
        Returns:
            Tuple of (ranked_primitives, vote_counts)
            ranked_primitives: List of (edge_idx, depth_idx) sorted by votes
            vote_counts: Dict mapping (edge_idx, depth_idx) to vote count
        """
        primitives = self.primitives.get(shape_type, self.primitives.get('square', []))
        
        if not primitives:
            return [], {}
            
        obj_x, obj_y, obj_theta = object_pose
        cos_theta = np.cos(obj_theta)
        sin_theta = np.sin(obj_theta)
        
        # Build primitive goals in world coordinates
        primitive_goals = []  # List of (edge_idx, depth_idx, goal_x, goal_y, goal_theta)
        
        for prim in primitives:
            dx, dy = prim.delta_x, prim.delta_y
            goal_x = obj_x + dx * cos_theta - dy * sin_theta
            goal_y = obj_y + dx * sin_theta + dy * cos_theta
            goal_theta = obj_theta + prim.delta_theta
            
            primitive_goals.append((
                prim.edge_idx,
                prim.push_steps - 1,  # Convert to 0-indexed depth
                goal_x,
                goal_y,
                goal_theta
            ))
        
        # Map each prediction to nearest primitive
        vote_counts = Counter()
        
        for pred in predictions:
            pred_x, pred_y, pred_theta = pred['x'], pred['y'], pred['theta']
            
            # Find nearest primitive
            best_dist = float('inf')
            best_primitive = None
            
            for edge_idx, depth_idx, goal_x, goal_y, goal_theta in primitive_goals:
                # Position error
                pos_err = np.sqrt((pred_x - goal_x)**2 + (pred_y - goal_y)**2)
                
                # Angle error (wrapped)
                angle_diff = pred_theta - goal_theta
                while angle_diff > np.pi:
                    angle_diff -= 2 * np.pi
                while angle_diff < -np.pi:
                    angle_diff += 2 * np.pi
                ang_err = abs(angle_diff)
                
                # Combined distance (weight angle less)
                dist = pos_err + 0.5 * ang_err
                
                if dist < best_dist:
                    best_dist = dist
                    best_primitive = (edge_idx, depth_idx)
            
            if best_primitive:
                vote_counts[best_primitive] += 1
        
        # Rank primitives by vote count (vote-only, no fallback)
        ranked_primitives = [prim for prim, _ in vote_counts.most_common()]

        return ranked_primitives, dict(vote_counts)
        
    def _execute_primitive(
        self,
        env: namo_rl.RLEnvironment,
        object_id: str,
        edge_idx: int,
        depth_idx: int,
        shape_type: str,
        collision_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, bool, Tuple[bool, bool]]:
        """Execute a single primitive push.
        
        Returns:
            Tuple of (push_success, region_opened, (static_collision, movable_collision))
        """
        primitives = self.primitives.get(shape_type, self.primitives.get('square', []))
        
        # Find the primitive with matching edge_idx and depth_idx
        target_prim = None
        for prim in primitives:
            if prim.edge_idx == edge_idx and prim.push_steps == depth_idx + 1:
                target_prim = prim
                break
                
        if target_prim is None:
            return False, False, (False, False)
            
        # Get current object pose
        obs = env.get_observation()
        pose_key = f"{object_id}_pose"
        if pose_key not in obs:
            return False, False, (False, False)
            
        obj_pose = obs[pose_key]
        obj_x, obj_y, obj_theta = obj_pose[0], obj_pose[1], obj_pose[2]
        
        # Compute goal in world coordinates
        cos_theta = np.cos(obj_theta)
        sin_theta = np.sin(obj_theta)
        
        goal_x = obj_x + target_prim.delta_x * cos_theta - target_prim.delta_y * sin_theta
        goal_y = obj_y + target_prim.delta_x * sin_theta + target_prim.delta_y * cos_theta
        goal_theta = obj_theta + target_prim.delta_theta
        
        step_result = None
        push_success = True

        if hasattr(env, "execute_push_primitive"):
            step_result = env.execute_push_primitive(
                object_id,
                edge_idx,
                target_prim.push_steps,
                goal_x,
                goal_y,
                goal_theta
            )
            push_success = bool(getattr(step_result, "done", True))
        else:
            # Fallback to skill-based execution (slower, re-plans to target)
            action = namo_rl.Action()
            action.object_id = object_id
            action.x = goal_x
            action.y = goal_y
            action.theta = goal_theta
            step_result = env.step(action)
            push_success = bool(getattr(step_result, "done", True))

        # Check if region is now open
        region_opened = env.is_robot_goal_reachable()
        static_hit, movable_hit = self._detect_collisions(env, object_id, step_result, collision_context)

        return push_success, region_opened, (static_hit, movable_hit)
        
    def evaluate_single_env(
        self,
        xml_path: str,
        object_id: str,
        robot_goal: Tuple[float, float, float] = None,
        max_pushes: int = None,  # None = try all voted primitives
        difficulty_label: Optional[str] = None,
        difficulty_score: Optional[float] = None
    ) -> EvalResult:
        """Evaluate model on a single environment.
        
        Args:
            xml_path: Path to environment XML
            object_id: Object to push
            robot_goal: Robot goal position (extracted from XML if None)
            max_pushes: Maximum pushes to attempt (None = try all voted)
            difficulty_label: Difficulty label (easy/medium/hard) if available
            difficulty_score: Difficulty score if available
            
        Returns:
            EvalResult with success/failure info
        """
        total_start = time.time()
        timing = {'env_init': 0, 'model_inference': 0, 'primitive_mapping': 0, 'simulation': 0}
        collision_total = 0
        collision_static_only = 0
        collision_movable_only = 0
        collision_both = 0
        
        try:
            # Initialize environment
            t0 = time.time()
            with _with_namo_cwd():
                env = namo_rl.RLEnvironment(xml_path, self.config_file, visualize=False)
            env.reset()
            
            # Get robot goal from XML if not provided
            if robot_goal is None:
                from namo.core.xml_goal_parser import extract_goal_with_fallback
                robot_goal = extract_goal_with_fallback(xml_path, (-0.5, 1.3, 0.0))
                
            env.set_robot_goal(*robot_goal)
            env.set_collision_checking(not self.allow_collisions)
            timing['env_init'] = (time.time() - t0) * 1000

            try:
                collision_context = self._build_collision_context(env)
            except Exception:
                collision_context = None
            
            # Check if already solved
            if env.is_robot_goal_reachable():
                return EvalResult(
                    env_path=xml_path,
                    success=True,
                    pushes_to_success=0,
                    time_to_success_ms=0.0,
                    total_pushes_attempted=0,
                    total_time_ms=(time.time() - total_start) * 1000,
                    env_init_ms=timing['env_init'],
                    difficulty_label=difficulty_label,
                    difficulty_score=difficulty_score,
                    collision_total=collision_total,
                    collision_static_only=collision_static_only,
                    collision_movable_only=collision_movable_only,
                    collision_both=collision_both
                )
            
            # Get object pose
            obs = env.get_observation()
            pose_key = f"{object_id}_pose"
            if pose_key not in obs:
                return EvalResult(
                    env_path=xml_path,
                    success=False,
                    error_message=f"Object {object_id} not found in observation",
                    total_time_ms=(time.time() - total_start) * 1000,
                    env_init_ms=timing['env_init'],
                    difficulty_label=difficulty_label,
                    difficulty_score=difficulty_score,
                    collision_total=collision_total,
                    collision_static_only=collision_static_only,
                    collision_movable_only=collision_movable_only,
                    collision_both=collision_both
                )
                
            # Save baseline state for strategy calls and per-primitive resets
            baseline_state = env.get_full_state()
            state = baseline_state

            # Query model for predictions using existing ML goal strategy
            t0 = time.time()
            ml_goals = self.ml_goal_strategy.generate_goals(
                object_id=object_id,
                state=state,
                env=env,
                max_goals=self.num_samples
            )
            timing['model_inference'] = (time.time() - t0) * 1000
            
            if not ml_goals:
                return EvalResult(
                    env_path=xml_path,
                    success=False,
                    error_message="No predictions from model",
                    total_time_ms=(time.time() - total_start) * 1000,
                    env_init_ms=timing['env_init'],
                    model_inference_ms=timing['model_inference'],
                    difficulty_label=difficulty_label,
                    difficulty_score=difficulty_score,
                    collision_total=collision_total,
                    collision_static_only=collision_static_only,
                    collision_movable_only=collision_movable_only,
                    collision_both=collision_both
                )
            
            # Map to primitives via existing ML-primitive alignment logic
            t0 = time.time()
            primitive_goals = self.primitive_goal_strategy.generate_goals(
                object_id=object_id,
                state=state,
                env=env,
                max_goals=self.num_samples
            )
            _, vote_counts = self._align_ml_goals_to_primitives(ml_goals, primitive_goals)
            shape_type = self._get_shape_type(object_id, env)
            ranked_primitives = self._rank_primitives_from_votes(shape_type, vote_counts)
            timing['primitive_mapping'] = (time.time() - t0) * 1000
            
            if not ranked_primitives:
                return EvalResult(
                    env_path=xml_path,
                    success=False,
                    error_message="No primitives matched",
                    total_time_ms=(time.time() - total_start) * 1000,
                    env_init_ms=timing['env_init'],
                    model_inference_ms=timing['model_inference'],
                    primitive_mapping_ms=timing['primitive_mapping'],
                    difficulty_label=difficulty_label,
                    difficulty_score=difficulty_score,
                    collision_total=collision_total,
                    collision_static_only=collision_static_only,
                    collision_movable_only=collision_movable_only,
                    collision_both=collision_both
                )
            
            if self.verbose:
                print(f"  Top 5 primitives: {ranked_primitives[:5]}")
                print(f"  Vote counts: {dict(list(vote_counts.items())[:5])}")
                print(f"  Timing - Init: {timing['env_init']:.1f}ms, Inference: {timing['model_inference']:.1f}ms")
            
            # Try ranked primitives (vote-only)
            primitives_to_try = ranked_primitives if max_pushes is None else ranked_primitives[:max_pushes]
            pushes_attempted = 0
            for edge_idx, depth_idx in primitives_to_try:
                # Reset to baseline state for independent primitive evaluation
                t0 = time.time()
                env.set_full_state(baseline_state)
                timing['simulation'] += (time.time() - t0) * 1000

                pushes_attempted += 1
                
                t0 = time.time()
                push_success, region_opened, collision_flags = self._execute_primitive(
                    env,
                    object_id,
                    edge_idx,
                    depth_idx,
                    shape_type,
                    collision_context=collision_context
                )
                push_time = (time.time() - t0) * 1000
                timing['simulation'] += push_time

                static_hit, movable_hit = collision_flags
                if static_hit or movable_hit:
                    collision_total += 1
                    if static_hit and movable_hit:
                        collision_both += 1
                    elif static_hit:
                        collision_static_only += 1
                    else:
                        collision_movable_only += 1
                
                if region_opened:
                    return EvalResult(
                        env_path=xml_path,
                        success=True,
                        pushes_to_success=pushes_attempted,
                        time_to_success_ms=timing['simulation'],  # All push attempts up to success
                        total_pushes_attempted=pushes_attempted,
                        total_time_ms=(time.time() - total_start) * 1000,
                        primitive_votes={str(k): v for k, v in vote_counts.items()},
                        chosen_primitive_idx=edge_idx * 10 + depth_idx,
                        env_init_ms=timing['env_init'],
                        model_inference_ms=timing['model_inference'],
                        primitive_mapping_ms=timing['primitive_mapping'],
                        simulation_ms=timing['simulation'],
                        difficulty_label=difficulty_label,
                        difficulty_score=difficulty_score,
                        collision_total=collision_total,
                        collision_static_only=collision_static_only,
                        collision_movable_only=collision_movable_only,
                        collision_both=collision_both
                    )
                    
                # Baseline reset handled at start of next iteration
            
            # No primitive succeeded
            return EvalResult(
                env_path=xml_path,
                success=False,
                total_pushes_attempted=pushes_attempted,
                total_time_ms=(time.time() - total_start) * 1000,
                error_message=f"No primitive succeeded after {pushes_attempted} attempts",
                primitive_votes={str(k): v for k, v in vote_counts.items()},
                env_init_ms=timing['env_init'],
                model_inference_ms=timing['model_inference'],
                primitive_mapping_ms=timing['primitive_mapping'],
                simulation_ms=timing['simulation'],
                difficulty_label=difficulty_label,
                difficulty_score=difficulty_score,
                collision_total=collision_total,
                collision_static_only=collision_static_only,
                collision_movable_only=collision_movable_only,
                collision_both=collision_both
            )
            
        except Exception as e:
            import traceback
            return EvalResult(
                env_path=xml_path,
                success=False,
                error_message=f"Exception: {str(e)}\n{traceback.format_exc()}",
                total_time_ms=(time.time() - total_start) * 1000,
                difficulty_label=difficulty_label,
                difficulty_score=difficulty_score,
                collision_total=collision_total,
                collision_static_only=collision_static_only,
                collision_movable_only=collision_movable_only,
                collision_both=collision_both
            )
    
    def evaluate_single_from_h5(
        self,
        h5_idx: int,
        max_pushes: int = None
    ) -> EvalResult:
        """Evaluate model using pre-computed masks from HDF5 (fast path).
        
        This uses the exact same masks the model was trained on, avoiding
        the expensive mask regeneration step (~3s → ~0.2s).
        
        Args:
            h5_idx: Index in HDF5 dataset
            max_pushes: Maximum pushes to attempt (None = try all voted)
            
        Returns:
            EvalResult with success/failure info
        """
        total_start = time.time()
        timing = {'env_init': 0, 'model_inference': 0, 'primitive_mapping': 0, 'simulation': 0}
        collision_total = 0
        collision_static_only = 0
        collision_movable_only = 0
        collision_both = 0
        
        # Get metadata from H5
        with h5py.File(self.h5_path, 'r') as f:
            if 'xml_file' not in f or 'action_object_ids' not in f:
                raise ValueError("H5 missing required keys: xml_file and/or action_object_ids")
            xml_path = _decode_h5_str(f['xml_file'][h5_idx])
            object_id = _decode_h5_str(f['action_object_ids'][h5_idx])
            if not xml_path or not object_id:
                raise ValueError("H5 sample missing xml_path or object_id")
            difficulty_label = None
            difficulty_score = None
            if 'difficulty_label' in f:
                difficulty_label = _decode_h5_str(f['difficulty_label'][h5_idx])
            if 'difficulty_score' in f:
                try:
                    score_val = f['difficulty_score'][h5_idx]
                    if isinstance(score_val, np.ndarray):
                        score_val = score_val.flat[0]
                    difficulty_score = float(score_val)
                except Exception:
                    difficulty_score = None
            robot_goal = None
            if 'robot_goal' in f:
                robot_goal_arr = np.array(f['robot_goal'][h5_idx]).astype(float).flatten()
                if robot_goal_arr.size >= 2:
                    robot_goal = (
                        float(robot_goal_arr[0]),
                        float(robot_goal_arr[1]),
                        float(robot_goal_arr[2]) if robot_goal_arr.size > 2 else 0.0
                    )
        
        try:
            # Initialize environment
            t0 = time.time()
            with _with_namo_cwd():
                env = namo_rl.RLEnvironment(xml_path, self.config_file, visualize=False)
            env.reset()

            if robot_goal is None:
                from namo.core.xml_goal_parser import extract_goal_with_fallback
                robot_goal = extract_goal_with_fallback(xml_path, (-0.5, 1.3, 0.0))

            env.set_robot_goal(*robot_goal)
            env.set_collision_checking(not self.allow_collisions)
            timing['env_init'] = (time.time() - t0) * 1000
            
            # Check if already solved
            if env.is_robot_goal_reachable():
                return EvalResult(
                    env_path=xml_path,
                    success=True,
                    pushes_to_success=0,
                    time_to_success_ms=0.0,
                    total_pushes_attempted=0,
                    total_time_ms=(time.time() - total_start) * 1000,
                    env_init_ms=timing['env_init'],
                    difficulty_label=difficulty_label,
                    difficulty_score=difficulty_score,
                    collision_total=collision_total,
                    collision_static_only=collision_static_only,
                    collision_movable_only=collision_movable_only,
                    collision_both=collision_both
                )

            try:
                collision_context = self._build_collision_context(env)
            except Exception:
                collision_context = None
            
            # Get object pose for fallback if H5 metadata is missing
            obs = env.get_observation()
            pose_key = f"{object_id}_pose"
            if pose_key not in obs:
                return EvalResult(
                    env_path=xml_path,
                    success=False,
                    error_message=f"Object {object_id} not found in observation",
                    total_time_ms=(time.time() - total_start) * 1000,
                    env_init_ms=timing['env_init'],
                    difficulty_label=difficulty_label,
                    difficulty_score=difficulty_score,
                    collision_total=collision_total,
                    collision_static_only=collision_static_only,
                    collision_movable_only=collision_movable_only,
                    collision_both=collision_both
                )

            fallback_pose = (obs[pose_key][0], obs[pose_key][1], obs[pose_key][2])

            baseline_state = env.get_full_state()

            # Run inference using pre-computed masks (fast!)
            t0 = time.time()
            predictions, _ = self._infer_from_h5_masks(
                h5_idx,
                fallback_object_pose=fallback_pose
            )
            timing['model_inference'] = (time.time() - t0) * 1000
            
            if not predictions:
                return EvalResult(
                    env_path=xml_path,
                    success=False,
                    error_message="No predictions from model",
                    total_time_ms=(time.time() - total_start) * 1000,
                    env_init_ms=timing['env_init'],
                    model_inference_ms=timing['model_inference'],
                    difficulty_label=difficulty_label,
                    difficulty_score=difficulty_score,
                    collision_total=collision_total,
                    collision_static_only=collision_static_only,
                    collision_movable_only=collision_movable_only,
                    collision_both=collision_both
                )
            
            # Map to primitives via existing ML-primitive alignment logic
            t0 = time.time()
            from namo.strategies.goal_selection_strategy import Goal
            state = baseline_state
            ml_goals = [Goal(x=pred['x'], y=pred['y'], theta=pred['theta']) for pred in predictions]
            primitive_goals = self.primitive_goal_strategy.generate_goals(
                object_id=object_id,
                state=state,
                env=env,
                max_goals=self.num_samples
            )
            _, vote_counts = self._align_ml_goals_to_primitives(ml_goals, primitive_goals)
            shape_type = self._get_shape_type(object_id, env)
            ranked_primitives = self._rank_primitives_from_votes(shape_type, vote_counts)
            timing['primitive_mapping'] = (time.time() - t0) * 1000
            
            if not ranked_primitives:
                return EvalResult(
                    env_path=xml_path,
                    success=False,
                    error_message="No primitives matched",
                    total_time_ms=(time.time() - total_start) * 1000,
                    env_init_ms=timing['env_init'],
                    model_inference_ms=timing['model_inference'],
                    primitive_mapping_ms=timing['primitive_mapping'],
                    difficulty_label=difficulty_label,
                    difficulty_score=difficulty_score,
                    collision_total=collision_total,
                    collision_static_only=collision_static_only,
                    collision_movable_only=collision_movable_only,
                    collision_both=collision_both
                )
            
            if self.verbose:
                print(f"  Top 5 primitives: {ranked_primitives[:5]}")
                print(f"  Vote counts: {dict(list(vote_counts.items())[:5])}")
                print(f"  Timing - Init: {timing['env_init']:.1f}ms, Inference: {timing['model_inference']:.1f}ms")
            
            # Try ranked primitives (vote-only)
            primitives_to_try = ranked_primitives if max_pushes is None else ranked_primitives[:max_pushes]
            pushes_attempted = 0
            for edge_idx, depth_idx in primitives_to_try:
                # Reset to baseline state for independent primitive evaluation
                t0 = time.time()
                env.set_full_state(baseline_state)
                timing['simulation'] += (time.time() - t0) * 1000

                pushes_attempted += 1
                
                t0 = time.time()
                push_success, region_opened, collision_flags = self._execute_primitive(
                    env,
                    object_id,
                    edge_idx,
                    depth_idx,
                    shape_type,
                    collision_context=collision_context
                )
                push_time = (time.time() - t0) * 1000
                timing['simulation'] += push_time

                static_hit, movable_hit = collision_flags
                if static_hit or movable_hit:
                    collision_total += 1
                    if static_hit and movable_hit:
                        collision_both += 1
                    elif static_hit:
                        collision_static_only += 1
                    else:
                        collision_movable_only += 1
                
                if region_opened:
                    return EvalResult(
                        env_path=xml_path,
                        success=True,
                        pushes_to_success=pushes_attempted,
                        time_to_success_ms=timing['simulation'],
                        total_pushes_attempted=pushes_attempted,
                        total_time_ms=(time.time() - total_start) * 1000,
                        primitive_votes={str(k): v for k, v in vote_counts.items()},
                        chosen_primitive_idx=edge_idx * 10 + depth_idx,
                        env_init_ms=timing['env_init'],
                        model_inference_ms=timing['model_inference'],
                        primitive_mapping_ms=timing['primitive_mapping'],
                        simulation_ms=timing['simulation'],
                        difficulty_label=difficulty_label,
                        difficulty_score=difficulty_score,
                        collision_total=collision_total,
                        collision_static_only=collision_static_only,
                        collision_movable_only=collision_movable_only,
                        collision_both=collision_both
                    )
                
                # Baseline reset handled at start of next iteration
            
            return EvalResult(
                env_path=xml_path,
                success=False,
                total_pushes_attempted=pushes_attempted,
                total_time_ms=(time.time() - total_start) * 1000,
                error_message=f"No primitive succeeded after {pushes_attempted} attempts",
                primitive_votes={str(k): v for k, v in vote_counts.items()},
                env_init_ms=timing['env_init'],
                model_inference_ms=timing['model_inference'],
                primitive_mapping_ms=timing['primitive_mapping'],
                simulation_ms=timing['simulation'],
                difficulty_label=difficulty_label,
                difficulty_score=difficulty_score,
                collision_total=collision_total,
                collision_static_only=collision_static_only,
                collision_movable_only=collision_movable_only,
                collision_both=collision_both
            )
            
        except Exception as e:
            import traceback
            return EvalResult(
                env_path=xml_path,
                success=False,
                error_message=f"Exception: {str(e)}\n{traceback.format_exc()}",
                total_time_ms=(time.time() - total_start) * 1000,
                difficulty_label=difficulty_label,
                difficulty_score=difficulty_score,
                collision_total=collision_total,
                collision_static_only=collision_static_only,
                collision_movable_only=collision_movable_only,
                collision_both=collision_both
            )
            
    def evaluate_test_set(
        self,
        test_indices: List[int],
        max_envs: Optional[int] = None,
        max_pushes: int = None,
        use_h5_masks: bool = False
    ) -> EvalStats:
        """Evaluate model on test set.
        
        Args:
            test_indices: Indices of test samples in HDF5
            max_envs: Maximum environments to evaluate (None = all)
            max_pushes: Maximum pushes per environment (None = try all voted)
            use_h5_masks: If True, use pre-computed masks from H5 (fast mode)
            
        Returns:
            EvalStats with aggregate statistics
        """
        if max_envs is not None:
            test_indices = test_indices[:max_envs]
            
        mode_str = "H5 masks (fast)" if use_h5_masks else "regenerated masks"
        print(f"\nEvaluating on {len(test_indices)} test samples using {mode_str}...")
        
        results = []
        eval_start = time.time()
        
        if use_h5_masks:
            # Fast mode: evaluate each H5 sample directly using its pre-computed masks
            for h5_idx in tqdm(test_indices, desc="Evaluating"):
                if self.verbose:
                    with h5py.File(self.h5_path, 'r') as f:
                        xml_path = _decode_h5_str(f['xml_file'][h5_idx])
                        object_id = _decode_h5_str(f['action_object_ids'][h5_idx])
                    print(f"\n{'='*60}")
                    print(f"Evaluating: {xml_path}")
                    print(f"Object: {object_id}")
                
                result = self.evaluate_single_from_h5(
                    h5_idx=h5_idx,
                    max_pushes=max_pushes
                )
                results.append(result)
                
                if self.verbose:
                    status = "✓ SUCCESS" if result.success else "✗ FAILED"
                    print(f"Result: {status}")
                    if result.success:
                        print(f"  Pushes: {result.pushes_to_success}, Time to success: {result.time_to_success_ms:.1f}ms")
                    print(f"  Timing breakdown: Init={result.env_init_ms:.1f}ms, Inference={result.model_inference_ms:.1f}ms, Map={result.primitive_mapping_ms:.1f}ms, Sim={result.simulation_ms:.1f}ms")
        else:
            # Original mode: get metadata and evaluate unique (env, object) pairs
            metadata = get_env_metadata_from_h5(self.h5_path, test_indices)
            
            unique_envs = {}
            for meta in metadata:
                if 'xml_path' in meta and 'object_id' in meta:
                    key = (meta['xml_path'], meta['object_id'])
                    if key not in unique_envs:
                        unique_envs[key] = meta
                        
            print(f"Found {len(unique_envs)} unique (env, object) pairs")
            
            for (xml_path, object_id), meta in tqdm(unique_envs.items(), desc="Evaluating"):
                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"Evaluating: {xml_path}")
                    print(f"Object: {object_id}")
                    
                robot_goal = meta.get('robot_goal', None)
                    
                result = self.evaluate_single_env(
                    xml_path=xml_path,
                    object_id=object_id,
                    robot_goal=robot_goal,
                    max_pushes=max_pushes,
                    difficulty_label=meta.get('difficulty_label'),
                    difficulty_score=meta.get('difficulty_score')
                )
                results.append(result)
                
                if self.verbose:
                    status = "✓ SUCCESS" if result.success else "✗ FAILED"
                    print(f"Result: {status}")
                    if result.success:
                        print(f"  Pushes: {result.pushes_to_success}, Time to success: {result.time_to_success_ms:.1f}ms")
                    print(f"  Timing breakdown: Init={result.env_init_ms:.1f}ms, Inference={result.model_inference_ms:.1f}ms, Map={result.primitive_mapping_ms:.1f}ms, Sim={result.simulation_ms:.1f}ms")
        
        # Compute statistics
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        stats = EvalStats(
            total_envs=len(results),
            successful_envs=len(successful),
            failed_envs=len(failed),
            success_rate=len(successful) / len(results) if results else 0.0,
            total_evaluation_time_sec=time.time() - eval_start,
            results=results
        )
        
        if successful:
            pushes = [r.pushes_to_success for r in successful]
            times = [r.time_to_success_ms for r in successful]
            
            stats.avg_pushes_to_success = np.mean(pushes)
            stats.avg_time_to_success_ms = np.mean(times)
            stats.median_pushes_to_success = np.median(pushes)
            stats.median_time_to_success_ms = np.median(times)

        collisions_total = [r.collision_total for r in results]
        collisions_static_only = [r.collision_static_only for r in results]
        collisions_movable_only = [r.collision_movable_only for r in results]
        collisions_both = [r.collision_both for r in results]

        if collisions_total:
            stats.avg_collision_total = float(np.mean(collisions_total))
            stats.median_collision_total = float(np.median(collisions_total))
            stats.avg_collision_static_only = float(np.mean(collisions_static_only))
            stats.median_collision_static_only = float(np.median(collisions_static_only))
            stats.avg_collision_movable_only = float(np.mean(collisions_movable_only))
            stats.median_collision_movable_only = float(np.median(collisions_movable_only))
            stats.avg_collision_both = float(np.mean(collisions_both))
            stats.median_collision_both = float(np.median(collisions_both))

        difficulty_groups = defaultdict(list)
        for result in results:
            label = (result.difficulty_label or "unknown").strip().lower()
            if not label:
                label = "unknown"
            difficulty_groups[label].append(result)

        stats.difficulty_breakdown = {
            label: _summarize_results(group_results)
            for label, group_results in sorted(difficulty_groups.items())
        }

        # Average timing breakdowns across all evaluated envs
        env_inits = [r.env_init_ms for r in results if r.env_init_ms > 0]
        model_times = [r.model_inference_ms for r in results if r.model_inference_ms > 0]
        map_times = [r.primitive_mapping_ms for r in results if r.primitive_mapping_ms > 0]
        sim_times = [r.simulation_ms for r in results if r.simulation_ms > 0]

        if env_inits:
            stats.avg_env_init_ms = float(np.mean(env_inits))
        if model_times:
            stats.avg_model_inference_ms = float(np.mean(model_times))
        if map_times:
            stats.avg_primitive_mapping_ms = float(np.mean(map_times))
        if sim_times:
            stats.avg_simulation_ms = float(np.mean(sim_times))
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate Flow Matching Model on Region Opening")
    
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model directory")
    parser.add_argument("--h5-path", type=str,
                        default="/common/users/shared/robot_learning/dm1487/namo/datasets/images/dec2/aug9_envs/1_push_train/h5/training_data.h5",
                        help="Path to HDF5 dataset")
    parser.add_argument("--primitive-dir", type=str, default=str(Path(NAMO_CPP_PATH) / "data"),
                        help="Directory containing motion primitive files")
    parser.add_argument("--num-samples", type=int, default=32,
                        help="Number of model samples per inference")
    parser.add_argument("--num-steps", type=int, default=10,
                        help="Number of ODE solver steps (fewer = faster, try 5-20)")
    parser.add_argument("--sampler-method", type=str, default=None,
                        help="Override sampler method (euler, midpoint, rk4, dopri5). Default uses training config")
    parser.add_argument("--ml-match-max-per-call", type=int, default=8,
                        help="Maximum ML goals to align per call (for parity with region_opening)")
    parser.add_argument("--ml-match-position-tolerance", type=float, default=0.05,
                        help="Position tolerance for ML-primitive alignment (meters)")
    parser.add_argument("--ml-match-angle-tolerance", type=float, default=0.1,
                        help="Angle tolerance for ML-primitive alignment (radians)")
    parser.add_argument("--ml-match-angle-weight", type=float, default=0.5,
                        help="Angle weight for ML-primitive alignment scoring")
    parser.add_argument("--goals-per-region", type=int, default=5,
                        help="Number of region goal samples to include (vector models only)")
    parser.add_argument("--allow-collisions", action="store_true",
                        help="Allow collisions during pushes (matches region_opening_collection defaults)")
    parser.add_argument("--disallow-collisions", action="store_true",
                        help="Terminate pushes on collisions (overrides --allow-collisions)")
    parser.add_argument("--max-test-envs", type=int, default=None,
                        help="Maximum test environments to evaluate")
    parser.add_argument("--max-pushes", type=int, default=None,
                        help="Maximum pushes per environment (None = try all voted)")
    parser.add_argument("--train-split", type=float, default=0.9,
                        help="Training split ratio (to compute test set)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for model inference")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results JSON")
    parser.add_argument("--use-h5-masks", action="store_true",
                        help="Use pre-computed masks from H5 (fast mode, ~0.2s vs ~3s per sample)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Convert paths to absolute BEFORE changing directory
    model_path_abs = Path(args.model_path).resolve()
    primitive_dir_abs = Path(args.primitive_dir).resolve()
    output_path_abs = Path(args.output).resolve() if args.output else None
    
    print("="*60)
    print("Flow Matching Model Evaluation")
    print("="*60)
    print(f"Model: {model_path_abs}")
    print(f"Dataset: {args.h5_path}")
    print(f"Samples per inference: {args.num_samples}")
    print(f"ODE solver steps: {args.num_steps}")
    print(f"Sampler method override: {args.sampler_method or 'training config'}")
    print(f"ML-primitive alignment: max_matches={args.ml_match_max_per_call}, pos_tol={args.ml_match_position_tolerance}, ang_tol={args.ml_match_angle_tolerance}, ang_weight={args.ml_match_angle_weight}")
    print(f"Region goal samples per region: {args.goals_per_region}")
    if args.disallow_collisions:
        allow_collisions = False
    elif args.allow_collisions:
        allow_collisions = True
    else:
        allow_collisions = True

    print(f"Allow collisions: {allow_collisions}")
    print(f"Train split: {args.train_split} (test = {1-args.train_split:.0%})")
    print(f"Use H5 masks: {args.use_h5_masks} {'(fast mode)' if args.use_h5_masks else '(regenerate masks)'}")
    print("Primitive ranking: vote-only (mapped from model samples)")
    print("="*60)
    
    # Get test indices
    test_indices = get_test_indices(args.h5_path, args.train_split)
    
    # Initialize evaluator
    evaluator = FlowMatchingEvaluator(
        model_path=str(model_path_abs),
        h5_path=args.h5_path,
        primitive_data_dir=str(primitive_dir_abs),
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        sampler_method=args.sampler_method,
        match_max_per_call=args.ml_match_max_per_call,
        match_position_tolerance=args.ml_match_position_tolerance,
        match_angle_tolerance=args.ml_match_angle_tolerance,
        match_angle_weight=args.ml_match_angle_weight,
        goals_per_region=args.goals_per_region,
        allow_collisions=allow_collisions,
        device=args.device,
        verbose=args.verbose
    )
    
    # Run evaluation
    stats = evaluator.evaluate_test_set(
        test_indices=test_indices,
        max_envs=args.max_test_envs,
        max_pushes=args.max_pushes,
        use_h5_masks=args.use_h5_masks
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total environments: {stats.total_envs}")
    print(f"Successful: {stats.successful_envs} ({stats.success_rate:.1%})")
    print(f"Failed: {stats.failed_envs}")
    print(f"\nOn successful environments:")
    print(f"  Avg pushes to success: {stats.avg_pushes_to_success:.2f}")
    print(f"  Median pushes to success: {stats.median_pushes_to_success:.2f}")
    print(f"  Avg time to success: {stats.avg_time_to_success_ms:.1f}ms")
    print(f"  Median time to success: {stats.median_time_to_success_ms:.1f}ms")
    print(f"\nCollision counts per environment (all attempts):")
    print(f"  Avg collisions: {stats.avg_collision_total:.2f} (median {stats.median_collision_total:.2f})")
    print(f"  Avg static-only: {stats.avg_collision_static_only:.2f} (median {stats.median_collision_static_only:.2f})")
    print(f"  Avg movable-only: {stats.avg_collision_movable_only:.2f} (median {stats.median_collision_movable_only:.2f})")
    print(f"  Avg both: {stats.avg_collision_both:.2f} (median {stats.median_collision_both:.2f})")

    if stats.difficulty_breakdown:
        print(f"\nDifficulty breakdown:")
        preferred_order = ["easy", "medium", "hard", "unknown"]
        for label in preferred_order:
            if label not in stats.difficulty_breakdown:
                continue
            summary = stats.difficulty_breakdown[label]
            print(
                f"  {label}: n={summary['total_envs']}, success={summary['success_rate']:.1%}, "
                f"pushes(avg/med)={summary['avg_pushes_to_success']:.2f}/{summary['median_pushes_to_success']:.2f}, "
                f"time(avg/med)={summary['avg_time_to_success_ms']:.1f}/{summary['median_time_to_success_ms']:.1f}ms, "
                f"collisions(avg/med)={summary['avg_collision_total']:.2f}/{summary['median_collision_total']:.2f}"
            )
        for label, summary in stats.difficulty_breakdown.items():
            if label in preferred_order:
                continue
            print(
                f"  {label}: n={summary['total_envs']}, success={summary['success_rate']:.1%}, "
                f"pushes(avg/med)={summary['avg_pushes_to_success']:.2f}/{summary['median_pushes_to_success']:.2f}, "
                f"time(avg/med)={summary['avg_time_to_success_ms']:.1f}/{summary['median_time_to_success_ms']:.1f}ms, "
                f"collisions(avg/med)={summary['avg_collision_total']:.2f}/{summary['median_collision_total']:.2f}"
            )
    print(f"\nAverage timing breakdown (all envs):")
    print(f"  Env init: {stats.avg_env_init_ms:.1f}ms")
    print(f"  Model inference: {stats.avg_model_inference_ms:.1f}ms")
    print(f"  Primitive mapping: {stats.avg_primitive_mapping_ms:.1f}ms")
    print(f"  Simulation: {stats.avg_simulation_ms:.1f}ms")
    print(f"\nTotal evaluation time: {stats.total_evaluation_time_sec:.1f}s")
    print("="*60)
    
    # Save results
    if output_path_abs:
        output_path = output_path_abs
    else:
        output_path = model_path_abs / "eval_results.json"
        
    # Convert results to JSON-serializable format
    results_dict = {
        "model_path": str(model_path_abs),
        "h5_path": args.h5_path,
        "num_samples": args.num_samples,
        "num_steps": args.num_steps,
        "sampler_method": args.sampler_method,
        "train_split": args.train_split,
        "ml_match_max_per_call": args.ml_match_max_per_call,
        "ml_match_position_tolerance": args.ml_match_position_tolerance,
        "ml_match_angle_tolerance": args.ml_match_angle_tolerance,
        "ml_match_angle_weight": args.ml_match_angle_weight,
        "goals_per_region": args.goals_per_region,
        "allow_collisions": allow_collisions,
        "total_envs": stats.total_envs,
        "successful_envs": stats.successful_envs,
        "failed_envs": stats.failed_envs,
        "success_rate": stats.success_rate,
        "avg_pushes_to_success": stats.avg_pushes_to_success,
        "median_pushes_to_success": stats.median_pushes_to_success,
        "avg_time_to_success_ms": stats.avg_time_to_success_ms,
        "median_time_to_success_ms": stats.median_time_to_success_ms,
        "avg_collision_total": stats.avg_collision_total,
        "median_collision_total": stats.median_collision_total,
        "avg_collision_static_only": stats.avg_collision_static_only,
        "median_collision_static_only": stats.median_collision_static_only,
        "avg_collision_movable_only": stats.avg_collision_movable_only,
        "median_collision_movable_only": stats.median_collision_movable_only,
        "avg_collision_both": stats.avg_collision_both,
        "median_collision_both": stats.median_collision_both,
        "avg_env_init_ms": stats.avg_env_init_ms,
        "avg_model_inference_ms": stats.avg_model_inference_ms,
        "avg_primitive_mapping_ms": stats.avg_primitive_mapping_ms,
        "avg_simulation_ms": stats.avg_simulation_ms,
        "total_evaluation_time_sec": stats.total_evaluation_time_sec,
        "difficulty_breakdown": stats.difficulty_breakdown,
        "individual_results": [
            {
                "env_path": r.env_path,
                "success": r.success,
                "pushes_to_success": r.pushes_to_success,
                "time_to_success_ms": r.time_to_success_ms,
                "total_pushes_attempted": r.total_pushes_attempted,
                "env_init_ms": r.env_init_ms,
                "model_inference_ms": r.model_inference_ms,
                "primitive_mapping_ms": r.primitive_mapping_ms,
                "simulation_ms": r.simulation_ms,
                "difficulty_label": r.difficulty_label,
                "difficulty_score": r.difficulty_score,
                "collision_total": r.collision_total,
                "collision_static_only": r.collision_static_only,
                "collision_movable_only": r.collision_movable_only,
                "collision_both": r.collision_both,
                "error_message": r.error_message
            }
            for r in stats.results
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
