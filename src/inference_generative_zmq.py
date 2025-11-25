"""
Inference Server for Generative Models (Flow Matching / Diffusion)

ZMQ-based inference server that supports both flow matching and diffusion models
through the unified GenerativeModule interface.

Usage:
    python src/inference_generative_zmq.py \
        --config-path=/path/to/run/.hydra \
        --config-name=config \
        run_path=/path/to/run \
        +zmq.host=localhost \
        +zmq.port=5556

The server expects requests in JSON format with the following structure:
    - msg_type: 'decision_req' or 'result_info'
    - For 'decision_req': scene information for inference
    - For 'result_info': results feedback
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import random
from pathlib import Path

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import zmq
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
from torchvision import transforms

from utils.image_utils import find_rectangle_corners
from utils.json2img import ImageConverter


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg):
    """Main inference server function."""

    print("=" * 60)
    print("Generative Model Inference Server")
    print("=" * 60)

    # Load model from run path
    run_path = Path(cfg.run_path)
    print(f"Loading model from: {run_path}")

    # Load the training config
    config_path = run_path / ".hydra" / "config.yaml"
    if config_path.exists():
        train_cfg = OmegaConf.load(config_path)
    else:
        train_cfg = cfg

    # Setup output directory
    custom_output_dir = run_path / "inference_results"
    custom_output_dir.mkdir(parents=True, exist_ok=True)

    results_file = custom_output_dir / "results.txt"
    images_dir = custom_output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Initialize counters
    if results_file.exists():
        with open(results_file, "r") as f:
            lines = f.readlines()
        idx = len(lines) + 1
    else:
        idx = 1

    decision_idx = 0
    reachable_selection = 0
    reachable_checks = 0

    # Load checkpoint
    checkpoint_dir = run_path / "checkpoints"
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))

    checkpoint_path = None
    for checkpoint_file in checkpoint_files:
        if "epoch" in checkpoint_file.name:
            checkpoint_path = checkpoint_file
            break

    if checkpoint_path is None:
        raise ValueError(f"No checkpoint found in {checkpoint_dir}")

    # Instantiate model
    model = hydra.utils.instantiate(train_cfg.model)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to("cuda")
    model.eval()

    print(f"Model loaded: {type(model).__name__}")
    print(f"Path type: {model.path.prediction_type}")

    # Get sampling steps from config or default
    sampling_steps = cfg.get('sampling_steps', 20 if model.path.prediction_type == 'velocity' else 100)
    print(f"Sampling steps: {sampling_steps}")

    # Setup ZMQ server
    zmq_host = cfg.get('zmq', {}).get('host', 'localhost')
    zmq_port = cfg.get('zmq', {}).get('port', 5556)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{zmq_host}:{zmq_port}")
    print(f"ZMQ server started on {zmq_host}:{zmq_port}")

    # Setup data transform
    image_size = train_cfg.data.get('image_size', 64)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.Lambda(lambda x: x * 2 - 1),
    ])

    # Mujoco model configuration (adjust paths as needed)
    mujoco_model_dir = cfg.get('mujoco_model_dir',
        "/common/home/dm1487/robotics_research/ktamp/ml4kp_ktamp/resources/models/custom_walled_envs/jun22")
    mujoco_model_type = cfg.get('mujoco_model_type',
        "random_start_random_goal_single_obstacle_room_2_200k_halfrad")

    print("=" * 60)
    print("Server ready, waiting for requests...")

    # Main inference loop
    while True:
        message = socket.recv_string()
        json_message = json.loads(message)

        if json_message['msg_type'] == 'result_info':
            # Handle result feedback
            config_name = json_message['config_name']
            success = json_message['success']
            action_steps = json_message['action_steps']

            if action_steps == 0:
                idx += 1
                decision_idx = 0
            else:
                with open(results_file, "a") as f:
                    f.write(f"{config_name},{success},{action_steps},{decision_idx},"
                            f"{reachable_selection},{reachable_checks}\n")

            reachable_selection = 0
            reachable_checks = 0
            idx += 1
            decision_idx = 0
            socket.send_string(json.dumps(json_message))

        elif json_message['msg_type'] == 'decision_req':
            # Handle inference request
            reachable_objects_list = json_message.get('reachable_objects', [])

            # Create image converter
            image_converter = ImageConverter(
                json_message['config_name'],
                mujoco_model_dir,
                mujoco_model_type
            )

            # Process input
            inp = image_converter.process_datapoint(json_message, json_message['robot_goal'])
            obj2center_px = inp['obj2center_px']
            inp_img = inp.copy()

            # Build input tensor
            inp_tensor = np.concatenate([
                inp['robot_image'],
                inp['goal_image'],
                inp['movable_objects_image'],
                inp['static_objects_image'],
                inp['reachable_objects_image']
            ], axis=-1)
            inp_tensor = transform(inp_tensor).unsqueeze(0).to("cuda")

            # Generate samples
            with torch.no_grad():
                samples = model.sample_from_model(
                    inp_tensor,
                    tgt_size=2,
                    samples=4,
                    num_steps=sampling_steps
                )
                samples = (samples.permute(0, 2, 3, 1).cpu().numpy() + 1) / 2

            # Process samples and vote for object
            obj_goal = {}
            obj_votes = {}

            for i in range(samples.shape[0]):
                image_idx_dir = images_dir / str(idx)
                image_idx_dir.mkdir(parents=True, exist_ok=True)

                # Extract object mask
                object_mask = (samples[i][:, :, 0].copy() > 0.5) * 1.0
                object_mask = object_mask.astype(np.uint8)

                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(object_mask)

                if num_labels > 2:
                    continue

                reachable_checks += 1

                _, _, predicted_obj_center, obj_angle = find_rectangle_corners(object_mask)
                if predicted_obj_center is None:
                    continue

                dist = lambda p1, p2: np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                scale = image_converter.IMG_SIZE / image_size
                predicted_obj_center = (
                    int(predicted_obj_center[0] * scale),
                    int(predicted_obj_center[1] * scale)
                )

                # Find closest object
                min_dist = float('inf')
                min_obj_name = None
                for obj_name, obj_center in obj2center_px.items():
                    d = dist(obj_center, predicted_obj_center)
                    if d < min_dist:
                        min_dist = d
                        min_obj_name = obj_name

                if min_obj_name is None:
                    continue

                if min_obj_name not in obj_goal:
                    obj_goal[min_obj_name] = []
                    obj_votes[min_obj_name] = 0

                obj_votes[min_obj_name] += 1

                if min_obj_name in reachable_objects_list:
                    reachable_selection += 1

                # Extract goal mask
                goal_mask = (samples[i][:, :, 1].copy() > 0.5) * 1.0
                goal_mask = goal_mask.astype(np.uint8)

                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(goal_mask)
                if num_labels > 2:
                    continue

                _, _, predicted_goal_center, goal_angle = find_rectangle_corners(goal_mask)
                if predicted_goal_center is None:
                    continue

                predicted_goal_center = (
                    int(predicted_goal_center[0] * scale),
                    int(predicted_goal_center[1] * scale)
                )
                goal_center = list(image_converter.pixel_to_world(
                    predicted_goal_center[0],
                    predicted_goal_center[1]
                ))
                final_quat = image_converter.rotate_relative_to_world(
                    min_obj_name,
                    goal_angle - obj_angle
                )

                prediction_img = np.zeros((image_size, image_size, 3))
                prediction_img[:, :, :2] = samples[i]

                obj_goal[min_obj_name].append((goal_center, final_quat, i, prediction_img))

            # Check if any object was found
            if not obj_goal:
                print("No object found")
                response = {
                    'object': None,
                    'goal_center': None,
                    'final_quat': None,
                    'error': True,
                    'error_message': "No object found"
                }
                socket.send_string(json.dumps(response))
                continue

            # Find object with most votes
            max_votes = 0
            selected_object = None
            for obj_name, votes in obj_votes.items():
                if votes > max_votes:
                    max_votes = votes
                    selected_object = obj_name

            print(f"Selected object: {selected_object}")

            if selected_object not in obj_goal or len(obj_goal[selected_object]) == 0:
                print("No goal found for selected object")
                response = {
                    'object': None,
                    'goal_center': None,
                    'final_quat': None,
                    'error': True,
                    'error_message': "No goal found"
                }
                socket.send_string(json.dumps(response))
                continue

            # Select goal (random from voted object's goals)
            chosen_goal = random.choice(obj_goal[selected_object])
            goal_center, final_quat, selected_sample_idx, prediction_img = chosen_goal

            # Save visualization
            reachable_vis = np.zeros((image_size, image_size, 3))
            reachable_vis[:, :, 0] = cv2.resize(
                inp_img['reachable_objects_image'],
                (image_size, image_size)
            )

            scene_vis = np.concatenate([
                inp_img['robot_image'] + inp_img['goal_image'],
                inp_img['movable_objects_image'],
                inp_img['static_objects_image']
            ], axis=-1)

            final_img = np.concatenate([
                cv2.resize(scene_vis, (image_size, image_size)),
                reachable_vis,
                prediction_img
            ], axis=1)

            try:
                plt.imsave(image_idx_dir / f"sample_{decision_idx}.png", final_img)
                plt.close()
            except Exception as e:
                print(f"Error saving image: {e}")

            # Send response
            response = {
                'object': selected_object,
                'goal_center': goal_center,
                'final_quat': final_quat.tolist(),
                'error': False,
                'error_message': ""
            }
            socket.send_string(json.dumps(response))
            decision_idx += 1


if __name__ == "__main__":
    main()
