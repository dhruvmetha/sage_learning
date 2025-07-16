from omegaconf import DictConfig, OmegaConf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import hydra
import lightning.pytorch as pl
import torch
from pathlib import Path
import torchvision
import zmq
import json
from utils.json2img import ImageConverter
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import cv2
from utils.image_utils import find_rectangle_corners
import random
from hydra.core.hydra_config import HydraConfig

@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg):
    print("Config loaded from:", cfg)
    
    # Get run paths for obstacle and goal models from config or use defaults
    obstacle_run_path = Path(cfg.get('obstacle_run_path', 'outputs/2025-07-14/01-35-54'))
    goal_run_path = Path(cfg.get('goal_run_path', 'outputs/2025-07-14/01-59-56'))
    
    print(f"Loading obstacle model from: {obstacle_run_path}")
    print(f"Loading goal model from: {goal_run_path}")
    
    # Load obstacle model config
    obstacle_config_path = obstacle_run_path / ".hydra" / "config.yaml"
    obstacle_cfg = OmegaConf.load(obstacle_config_path)
    
    # Load goal model config  
    goal_config_path = goal_run_path / ".hydra" / "config.yaml"
    goal_cfg = OmegaConf.load(goal_config_path)
    
    # Use the main config's run_path if provided, otherwise use obstacle_run_path for outputs
    run_path = Path(cfg.get('run_path', obstacle_run_path))
    
    custom_output_dir = Path(f"{run_path}/ood_results_hard_50")
    custom_output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = custom_output_dir / f"results.txt"
    images_dir = custom_output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    if results_file.exists():
        with open(results_file, "r") as f:
            lines = f.readlines()
        idx = len(lines) + 1
    else:
        idx = 1
    decision_idx = 0
    num_diffusion = 0
    reachable_selection = 0
    reachable_checks = 0
    
    # Load obstacle selection model
    obstacle_checkpoint_dir = obstacle_run_path / "checkpoints"
    obstacle_checkpoint_files = list(obstacle_checkpoint_dir.glob("*.ckpt"))
    for checkpoint_file in obstacle_checkpoint_files:
        if "epoch" in checkpoint_file.name:
            obstacle_checkpoint_path = checkpoint_file
            break
    
    obstacle_model = hydra.utils.instantiate(obstacle_cfg.model)
    obstacle_checkpoint = torch.load(obstacle_checkpoint_path, weights_only=False)
    obstacle_model.load_state_dict(obstacle_checkpoint["state_dict"])
    print(f"✅ Obstacle model loaded successfully: {type(obstacle_model).__name__}")
    obstacle_model.to("cuda")
    
    # Load goal selection model
    goal_checkpoint_dir = goal_run_path / "checkpoints"
    goal_checkpoint_files = list(goal_checkpoint_dir.glob("*.ckpt"))
    for checkpoint_file in goal_checkpoint_files:
        if "epoch" in checkpoint_file.name:
            goal_checkpoint_path = checkpoint_file
            break
    
    goal_model = hydra.utils.instantiate(goal_cfg.model)
    goal_checkpoint = torch.load(goal_checkpoint_path, weights_only=False)
    goal_model.load_state_dict(goal_checkpoint["state_dict"])
    print(f"✅ Goal model loaded successfully: {type(goal_model).__name__}")
    goal_model.to("cuda")
    
    # Use data config from obstacle model (assuming both have same data structure)
    data_cfg = obstacle_cfg.data
    
    # start zmq server
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{cfg.zmq.host}:{cfg.zmq.port}")
    print(f"ZMQ server started on port {cfg.zmq.port}")
    
    mujoco_model_dir = "/common/home/dm1487/robotics_research/ktamp/ml4kp_ktamp/resources/models/custom_walled_envs/jun22"
    mujoco_model_type = "random_start_random_goal_single_obstacle_room_2_200k_halfrad"
    
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((data_cfg.image_size, data_cfg.image_size)),
            transforms.Lambda(lambda x: x * 2 - 1),
    ])
    
    # TODO:inference with zmq communication (no other option like simple inference)
    while True:
        message = socket.recv_string()
        json_message = json.loads(message) # recieve json message to construct the image
        
        # message type
        if json_message['msg_type'] == 'result_info':
            config_name = json_message['config_name']
            success = json_message['success']
            action_steps = json_message['action_steps']
            
            if action_steps == 0:
                idx += 1
                decision_idx = 0
            else:
                with open(custom_output_dir / f"results.txt", "a") as f:
                    f.write(f"{config_name},{success},{action_steps},{decision_idx},{reachable_selection},{reachable_checks}\n")
            reachable_selection = 0
            reachable_checks = 0
            idx += 1
            decision_idx = 0
            socket.send_string(json.dumps(json_message))
        
        elif json_message['msg_type'] == 'decision_req':
            reachable_objects_list = json_message['reachable_objects']
            image_converter = ImageConverter(json_message['config_name'], mujoco_model_dir, mujoco_model_type)
            inp = image_converter.process_datapoint(json_message, json_message['robot_goal'])
            obj2center_px = inp['obj2center_px']
            
            inp_img = inp.copy()
            
            # Step 1: First inference with obstacle model to select object
            inp_for_obstacle = np.concatenate([inp['robot_image'], inp['goal_image'], inp['movable_objects_image'], inp['static_objects_image'], inp['reachable_objects_image']], axis=-1)
            inp_for_obstacle = transform(inp_for_obstacle).unsqueeze(0).to("cuda")
            
            # Generate obstacle/object samples using obstacle model
            obstacle_samples = (obstacle_model.sample_from_model(inp_for_obstacle, samples=4).permute(0, 2, 3, 1).cpu().numpy() + 1)/ 2
            
            obj_votes = {}
            
            # Process obstacle samples to vote for best object
            for i in range(obstacle_samples.shape[0]):
                # Extract object mask from obstacle model output
                object_mask = (obstacle_samples[i][:, :, 0].copy() > 0.5) * 1.0
                object_mask = object_mask.astype(np.uint8)
                
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(object_mask)
                
                if num_labels > 2:
                    continue
                
                reachable_checks += 1
                
                _, _, predicted_obj_center, obj_angle = find_rectangle_corners(object_mask)
                if predicted_obj_center is None:
                    continue
                dist = lambda p1, p2: np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                scale = image_converter.IMG_SIZE / data_cfg.image_size
                predicted_obj_center = (int(predicted_obj_center[0] * scale), int(predicted_obj_center[1] * scale))
                # find object
                min_dist = float('inf')
                for obj_name, obj_center in obj2center_px.items():
                    d = dist(obj_center, predicted_obj_center)
                    if d < min_dist:
                        min_dist = d
                        min_obj_name = obj_name

                if min_obj_name not in obj_votes:
                    obj_votes[min_obj_name] = 0
                    
                obj_votes[min_obj_name] += 1
                
                if min_obj_name in reachable_objects_list:
                    reachable_selection += 1
            
            # Find the object with the most votes
            if not obj_votes:
                print("no object found")
                new_json_message = {
                    'object': None,
                    'goal_center': None,
                    'final_quat': None,
                    'error': True,
                    'error_message': "No object found"
                }
                socket.send_string(json.dumps(new_json_message))
                continue
            
            max_votes = 0
            selected_object = None
            for obj_name, votes in obj_votes.items():
                if votes > max_votes:
                    max_votes = votes
                    selected_object = obj_name
                    
            print("selected object", selected_object)
            
            # Step 2: Create object mask for the selected object
            try:
                selected_object_mask = image_converter.create_object_mask(selected_object)
            except Exception as e:
                print(f"Error creating object mask: {e}")
                new_json_message = {
                    'object': None,
                    'goal_center': None,
                    'final_quat': None,
                    'error': True,
                    'error_message': f"Error creating object mask: {e}"
                }
                socket.send_string(json.dumps(new_json_message))
                continue
            
            # Step 3: Create modified input for goal model (replace reachable_objects with object mask)
            inp_for_goal = np.concatenate([inp['robot_image'], inp['goal_image'], inp['movable_objects_image'], inp['static_objects_image'], selected_object_mask], axis=-1)
            inp_for_goal = transform(inp_for_goal).unsqueeze(0).to("cuda")
            
            # Step 4: Generate goal samples using goal model (8 samples)
            goal_samples = (goal_model.sample_from_model(inp_for_goal, samples=16).permute(0, 2, 3, 1).cpu().numpy() + 1)/ 2
            
            # Step 5: Process all goal samples and extract SE(2) poses
            valid_goals = []
            scale = image_converter.IMG_SIZE / data_cfg.image_size
            
            # Get object angle for rotation calculation (used for all goals)
            _, _, selected_obj_center, obj_angle = find_rectangle_corners((selected_object_mask[:, :, 0] > 0.5).astype(np.uint8))
            if selected_obj_center is None:
                # Fallback: use stored angle from obj2angle
                obj_angle = inp['obj2angle'].get(selected_object, 0)
            
            for i, goal_sample in enumerate(goal_samples):
                goal_mask = (goal_sample[:, :, 0].copy() > 0.5) * 1.0
                goal_mask = goal_mask.astype(np.uint8)
                
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(goal_mask)
                if num_labels > 2:
                    continue
                
                _, _, predicted_goal_center, goal_angle = find_rectangle_corners(goal_mask)
                if predicted_goal_center is None:
                    continue
                    
                predicted_goal_center = (int(predicted_goal_center[0] * scale), int(predicted_goal_center[1] * scale))
                goal_center = list(image_converter.pixel_to_world(predicted_goal_center[0], predicted_goal_center[1]))
                final_quat = image_converter.rotate_relative_to_world(selected_object, goal_angle - obj_angle)
                
                # Convert quaternion to euler angle (θ)
                from scipy.spatial.transform import Rotation as R
                goal_theta = R.from_quat(final_quat, scalar_first=True).as_euler('xyz')[2]  # Z-axis rotation
                
                valid_goals.append({
                    'index': i,
                    'goal_center': goal_center,
                    'final_quat': final_quat,
                    'x': goal_center[0],
                    'y': goal_center[1], 
                    'theta': goal_theta,
                    'goal_sample': goal_sample
                })
            
            # Check if we have any valid goals
            if not valid_goals:
                print("no valid goals found")
                new_json_message = {
                    'object': None,
                    'goal_center': None,
                    'final_quat': None,
                    'error': True,
                    'error_message': "No valid goals found"
                }
                socket.send_string(json.dumps(new_json_message))
                continue
            
            # Step 6: SE(2) density-based voting
            def se2_distance(goal1, goal2, w_pos=1.0, w_rot=0.3):
                """Calculate SE(2) distance between two goals"""
                # Position distance (meters)
                pos_dist = np.sqrt((goal1['x'] - goal2['x'])**2 + (goal1['y'] - goal2['y'])**2)
                
                # Angular distance (radians) - shortest path on circle  
                angle_diff = abs(goal1['theta'] - goal2['theta'])
                rot_dist = min(angle_diff, 2*np.pi - angle_diff)
                
                return w_pos * pos_dist + w_rot * rot_dist
            
            # Calculate density for each goal (count neighbors within threshold)
            threshold = 0.5 # SE(2) distance threshold
            densities = []
            
            for i, goal in enumerate(valid_goals):
                density = 0
                for j, other_goal in enumerate(valid_goals):
                    if i != j and se2_distance(goal, other_goal) < threshold:
                        density += 1
                densities.append(density)
            
            # Select goal with highest density
            max_density = max(densities)
            best_goals = [goal for goal, density in zip(valid_goals, densities) if density == max_density]
            
            # If no consensus (all densities are 0), pick random goal
            if max_density == 0:
                print(f"No consensus found (max density: {max_density}), picking random goal")
                selected_goal = random.choice(valid_goals)
            else:
                print(f"Found consensus with density: {max_density}, selecting from {len(best_goals)} candidates")
                selected_goal = random.choice(best_goals)  # Random choice among tied goals
            
            # Extract final results
            goal_center = selected_goal['goal_center']
            final_quat = selected_goal['final_quat']
            selected_goal_sample = selected_goal['goal_sample']
            
            print(f"Selected goal at ({goal_center[0]:.3f}, {goal_center[1]:.3f}, θ={selected_goal['theta']:.3f})")
            
            # Create visualization combining obstacle selection and goal prediction
            prediction_img = np.zeros((data_cfg.image_size, data_cfg.image_size, 3))
            prediction_img[:, :, 0] = cv2.resize(selected_object_mask[:, :, 0], (data_cfg.image_size, data_cfg.image_size))  # Selected object mask
            prediction_img[:, :, 1] = selected_goal_sample[:, :, 0]  # Selected goal mask
                
            # Save visualization image
            image_idx_dir = images_dir / str(idx)
            image_idx_dir.mkdir(parents=True, exist_ok=True)
            
            # Create visualization showing the pipeline: input -> selected object -> goal prediction
            # Original input (robot + goal + objects)
            new_inp = np.concatenate([inp_img['robot_image'] + inp_img['goal_image'], inp_img['movable_objects_image'], inp_img['static_objects_image']], axis=-1)
            new_inp_resized = cv2.resize(new_inp, (data_cfg.image_size, data_cfg.image_size))
            
            # Selected object mask (resized for visualization)
            selected_object_vis = np.zeros((data_cfg.image_size, data_cfg.image_size, 3))
            selected_object_vis[:, :, 0] = cv2.resize(selected_object_mask[:, :, 0], (data_cfg.image_size, data_cfg.image_size))
            
            # Combine: [original_input | selected_object | prediction]
            final_img = np.concatenate([new_inp_resized, selected_object_vis, prediction_img], axis=1)
            
            try:    
                plt.imsave(image_idx_dir / f"sample_{decision_idx}.png", final_img)
                plt.close()
            except Exception as e:
                print(e)
                print(f"Error saving image")
            
            new_json_message = {
                'object': selected_object,
                'goal_center': goal_center,
                'final_quat': final_quat.tolist(),
                'error': False,
                'error_message': ""
            }
            socket.send_string(json.dumps(new_json_message))
            decision_idx += 1
            
if __name__ == "__main__":
    main()