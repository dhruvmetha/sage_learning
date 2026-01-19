#!/usr/bin/env python3
"""
Visualize flow matching evaluation for a few HDF5 samples.

This script reuses FlowMatchingEvaluator to:
1) Load an H5 sample and environment
2) Run ML inference, align to primitives, and rank by votes
3) Execute ranked primitives with visualization enabled
"""

import argparse
import time
import random
import math
from pathlib import Path

import h5py
import numpy as np

import namo_rl
from namo.core.xml_goal_parser import extract_goal_with_fallback
from namo.strategies.goal_selection_strategy import Goal

from evaluate_flow_matching import (
    FlowMatchingEvaluator,
    get_test_indices,
    _decode_h5_str,
    _with_namo_cwd,
    NAMO_CPP_PATH,
)


def _parse_indices(indices_str):
    if not indices_str:
        return []
    return [int(x.strip()) for x in indices_str.split(",") if x.strip()]


def _load_h5_metadata(h5_path, h5_idx):
    with h5py.File(h5_path, "r") as f:
        xml_path = _decode_h5_str(f["xml_file"][h5_idx])
        object_id = _decode_h5_str(f["action_object_ids"][h5_idx])
        robot_goal = None
        if "robot_goal" in f:
            robot_goal_arr = np.array(f["robot_goal"][h5_idx]).astype(float).flatten()
            if robot_goal_arr.size >= 2:
                robot_goal = (
                    float(robot_goal_arr[0]),
                    float(robot_goal_arr[1]),
                    float(robot_goal_arr[2]) if robot_goal_arr.size > 2 else 0.0,
                )
    return xml_path, object_id, robot_goal


def _wrap_angle(theta):
    return math.atan2(math.sin(theta), math.cos(theta))


def _delta_local_to_world(dx_local, dy_local, dtheta, object_theta):
    c_theta = math.cos(object_theta)
    s_theta = math.sin(object_theta)
    dx_world = dx_local * c_theta - dy_local * s_theta
    dy_world = dx_local * s_theta + dy_local * c_theta
    return dx_world, dy_world, dtheta


def _load_h5_target_delta(h5_path, h5_idx, object_id):
    with h5py.File(h5_path, "r") as f:
        if "target_goal_pose_deltas_obj" not in f:
            return None
        deltas = np.array(f["target_goal_pose_deltas_obj"][h5_idx]).astype(float)

        if deltas.ndim == 1:
            deltas = deltas.reshape(1, -1)
        if deltas.shape[1] < 3:
            return None

        delta_idx = 0
        if "action_object_ids" in f:
            obj_ids = f["action_object_ids"][h5_idx]
            if isinstance(obj_ids, np.ndarray):
                obj_ids_decoded = [_decode_h5_str(obj_id) for obj_id in obj_ids]
                if object_id in obj_ids_decoded:
                    delta_idx = obj_ids_decoded.index(object_id)

        return deltas[min(delta_idx, deltas.shape[0] - 1)][:3]


def main():
    parser = argparse.ArgumentParser(description="Visualize flow matching evaluation")
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
                        help="Number of ODE solver steps")
    parser.add_argument("--sampler-method", type=str, default=None,
                        help="Override sampler method (euler, midpoint, rk4, dopri5)")
    parser.add_argument("--ml-match-max-per-call", type=int, default=8,
                        help="Maximum ML goals to align per call")
    parser.add_argument("--ml-match-position-tolerance", type=float, default=0.05,
                        help="Position tolerance for ML-primitive alignment (meters)")
    parser.add_argument("--ml-match-angle-tolerance", type=float, default=0.1,
                        help="Angle tolerance for ML-primitive alignment (radians)")
    parser.add_argument("--ml-match-angle-weight", type=float, default=0.5,
                        help="Angle weight for ML-primitive alignment scoring")
    parser.add_argument("--goals-per-region", type=int, default=5,
                        help="Number of region goal samples to include (vector models only)")
    parser.add_argument("--num-examples", type=int, default=3,
                        help="Number of examples to visualize")
    parser.add_argument("--indices", type=str, default=None,
                        help="Comma-separated H5 indices to visualize (overrides --num-examples)")
    parser.add_argument("--max-pushes", type=int, default=5,
                        help="Maximum pushes to visualize per environment")
    parser.add_argument("--train-split", type=float, default=0.9,
                        help="Training split ratio (to compute test set)")
    parser.add_argument("--use-h5-masks", action="store_true",
                        help="Use pre-computed masks from H5 (fast mode)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for model inference")
    parser.add_argument("--render-delay", type=float, default=1.0,
                        help="Seconds to wait after each push")
    parser.add_argument("--step-mode", action="store_true",
                        help="Wait for Enter between pushes")
    parser.add_argument("--compare-target", action="store_true",
                        help="Compare ML predictions to target_goal_pose_deltas_obj from H5")
    parser.add_argument("--compare-top-k", type=int, default=5,
                        help="How many predictions to print for target comparison")
    parser.add_argument("--show-target-marker", action="store_true",
                        help="Temporarily show GT/pred positions using the robot goal marker")
    parser.add_argument("--allow-collisions", action="store_true",
                        help="Allow collisions during pushes (matches region_opening_collection defaults)")
    parser.add_argument("--disallow-collisions", action="store_true",
                        help="Terminate pushes on collisions (overrides --allow-collisions)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")

    args = parser.parse_args()

    indices = _parse_indices(args.indices)
    if not indices:
        test_indices = get_test_indices(args.h5_path, args.train_split)
        indices = test_indices[:args.num_examples]

    if args.disallow_collisions:
        allow_collisions = False
    elif args.allow_collisions:
        allow_collisions = True
    else:
        allow_collisions = True

    evaluator = FlowMatchingEvaluator(
        model_path=str(Path(args.model_path).resolve()),
        h5_path=args.h5_path,
        primitive_data_dir=str(Path(args.primitive_dir).resolve()),
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
        verbose=args.verbose,
    )

    for h5_idx in indices:
        xml_path, object_id, robot_goal = _load_h5_metadata(args.h5_path, h5_idx)
        if not xml_path or not object_id:
            print(f"Skipping H5 idx {h5_idx}: missing xml_path or object_id")
            continue

        print("\n" + "=" * 60)
        print(f"H5 idx: {h5_idx}")
        print(f"XML: {xml_path}")
        print(f"Object: {object_id}")

        with _with_namo_cwd():
            env = namo_rl.RLEnvironment(xml_path, evaluator.config_file, visualize=True)

        env.reset()
        if robot_goal is None:
            robot_goal = extract_goal_with_fallback(xml_path, (-0.5, 1.3, 0.0))
        env.set_robot_goal(*robot_goal)
        env.set_collision_checking(not allow_collisions)

        if env.is_robot_goal_reachable():
            print("Robot goal already reachable; skipping pushes.")
            continue

        obs = env.get_observation()
        pose_key = f"{object_id}_pose"
        if pose_key not in obs:
            print(f"Object {object_id} not found in observation; skipping.")
            continue

        baseline_state = env.get_full_state()
        target_world = None
        if args.compare_target:
            delta_local = _load_h5_target_delta(args.h5_path, h5_idx, object_id)
            if delta_local is None:
                print("No target_goal_pose_deltas_obj in H5 for this sample.")
            else:
                obj_pose = obs[pose_key]
                dx_w, dy_w, dtheta = _delta_local_to_world(
                    float(delta_local[0]),
                    float(delta_local[1]),
                    float(delta_local[2]),
                    float(obj_pose[2]),
                )
                target_world = (
                    float(obj_pose[0]) + dx_w,
                    float(obj_pose[1]) + dy_w,
                    _wrap_angle(float(obj_pose[2]) + dtheta),
                )
                print(
                    "GT target (world): "
                    f"x={target_world[0]:.3f}, y={target_world[1]:.3f}, theta={target_world[2]:.3f}"
                )

        pred_goals = []
        if args.use_h5_masks:
            fallback_pose = (obs[pose_key][0], obs[pose_key][1], obs[pose_key][2])
            predictions, _ = evaluator._infer_from_h5_masks(h5_idx, fallback_object_pose=fallback_pose)
            ml_goals = [Goal(x=pred["x"], y=pred["y"], theta=pred["theta"]) for pred in predictions]
            pred_goals = [(pred["x"], pred["y"], pred["theta"]) for pred in predictions]
        else:
            ml_goals = evaluator.ml_goal_strategy.generate_goals(
                object_id=object_id,
                state=baseline_state,
                env=env,
                max_goals=args.num_samples,
            )
            pred_goals = [(goal.x, goal.y, goal.theta) for goal in ml_goals]

        if not ml_goals:
            print("No ML goals generated; skipping.")
            continue

        primitive_goals = evaluator.primitive_goal_strategy.generate_goals(
            object_id=object_id,
            state=baseline_state,
            env=env,
            max_goals=args.num_samples,
        )
        _, vote_counts = evaluator._align_ml_goals_to_primitives(ml_goals, primitive_goals)
        shape_type = evaluator._get_shape_type(object_id, env)
        ranked_primitives = evaluator._rank_primitives_from_votes(shape_type, vote_counts)

        if args.max_pushes is not None:
            ranked_primitives = ranked_primitives[:args.max_pushes]

        print(f"Top primitives: {ranked_primitives[:5]}")
        print(f"Top votes: {dict(list(vote_counts.items())[:5])}")

        if args.compare_target and target_world is not None:
            print(f"Comparing top {min(args.compare_top_k, len(pred_goals))} predictions:")
            for i, (px, py, ptheta) in enumerate(pred_goals[:args.compare_top_k]):
                pos_err = math.hypot(px - target_world[0], py - target_world[1])
                ang_err = abs(_wrap_angle(ptheta - target_world[2]))
                print(
                    f"  Pred {i}: x={px:.3f}, y={py:.3f}, theta={ptheta:.3f} "
                    f"(pos_err={pos_err:.3f}m, ang_err={ang_err:.3f}rad)"
                )

        if args.show_target_marker and target_world is not None and pred_goals:
            # Show GT marker
            env.set_robot_goal(*target_world)
            env.render()
            if args.step_mode:
                input("Showing GT target marker. Press Enter to continue...")
            else:
                time.sleep(max(0.0, args.render_delay))

            # Show first prediction marker
            env.set_robot_goal(*pred_goals[0])
            env.render()
            if args.step_mode:
                input("Showing pred[0] marker. Press Enter to continue...")
            else:
                time.sleep(max(0.0, args.render_delay))

            # Restore original robot goal
            env.set_robot_goal(*robot_goal)

        for edge_idx, depth_idx in ranked_primitives:
            env.set_full_state(baseline_state)
            push_success, region_opened, collision_flags = evaluator._execute_primitive(
                env, object_id, edge_idx, depth_idx, shape_type
            )

            env.render()
            static_hit, movable_hit = collision_flags
            print(
                f"Edge {edge_idx} depth {depth_idx + 1}: "
                f"push_success={push_success}, region_opened={region_opened}, "
                f"collision_static={static_hit}, collision_movable={movable_hit}"
            )

            if region_opened:
                print("✅ Success: robot goal reachable.")
                break

            if args.step_mode:
                input("Press Enter to continue...")
            else:
                time.sleep(max(0.0, args.render_delay))


if __name__ == "__main__":
    main()
