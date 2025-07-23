from omegaconf import DictConfig, OmegaConf
import omegaconf
import os
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
from scipy.spatial.transform import Rotation as R

@hydra.main(config_path="../config", config_name="inference.yaml", version_base=None)
def main(cfg):
    print("Inference config loaded from:", cfg)
    run_path = Path(cfg.run_path)
    
    # Load the original model config from the checkpoint directory
    model_config_path = run_path / ".hydra" / "config.yaml"
    if not model_config_path.exists():
        raise ValueError(f"Model config not found at {model_config_path}")
    
    model_cfg = OmegaConf.load(model_config_path)
    print(f"Model config loaded from: {model_config_path}")
    
    # Get results suffix from config or use default
    results_suffix = cfg.get('results_suffix', 'ood_results_25')
    custom_output_dir = Path(f"{run_path}/{results_suffix}")
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
    current_trial_id = None  # Track current trial ID
    
    checkpoint_dir = run_path / "checkpoints"
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    for checkpoint_file in checkpoint_files:
        if "epoch" in checkpoint_file.name:
            checkpoint_path = checkpoint_file
            break
    
    # Instantiate model using the loaded model config
    if not hasattr(model_cfg, 'model'):
        raise ValueError(f"Model config does not contain 'model' key. Available keys: {list(model_cfg.keys())}")
    
    print(f"Model config structure: {model_cfg.model}")
    model = hydra.utils.instantiate(model_cfg.model)
    print(f"✅ Model loaded successfully: {type(model).__name__}")
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"✅ Model state dict loaded successfully")
    
    model.to("cuda")
    
    # start zmq server
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    
    # Get ZMQ configuration from config or use defaults
    zmq_host = cfg.get('zmq', {}).get('host', 'arrakis.cs.rutgers.edu')
    zmq_port = cfg.get('zmq', {}).get('port', 5556)
    
    socket.bind(f"tcp://{zmq_host}:{zmq_port}")
    print(f"ZMQ server started on {zmq_host}:{zmq_port}")
    
    mujoco_model_dir = "/common/home/dm1487/robotics_research/ktamp/ml4kp_ktamp/resources/models/custom_walled_envs/jun22"
    mujoco_model_type = "random_start_random_goal_single_obstacle_room_2_200k_halfrad"
    
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((model_cfg.data.image_size, model_cfg.data.image_size)),
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
            trial_id = json_message.get('trial_id', current_trial_id)
            
            print(f"DEBUG: Trial {trial_id} completed. Success: {success}, Action steps: {action_steps}, Decisions: {decision_idx}")
            print(f"DEBUG: Message flow - result_info received for trial {trial_id}")
            
            # Write results to file (except for failed trials with 0 action steps)
            if action_steps > 0:
                with open(custom_output_dir / f"results.txt", "a") as f:
                    f.write(f"{config_name},{success},{action_steps},{decision_idx},{reachable_selection},{reachable_checks}\n")
            
            # Reset counters for next trial
            reachable_selection = 0
            reachable_checks = 0
            idx += 1  # Only increment once per trial
            decision_idx = 0
            current_trial_id = None  # Reset trial tracking
            socket.send_string(json.dumps(json_message))
        
        elif json_message['msg_type'] == 'decision_req':
            # Track trial ID and detect new trials
            trial_id = json_message.get('trial_id', 'unknown')
            if current_trial_id != trial_id:
                if current_trial_id is not None:
                    print(f"DEBUG: New trial detected. Previous: {current_trial_id}, Current: {trial_id}")
                else:
                    print(f"DEBUG: First trial started. Trial ID: {trial_id}")
                current_trial_id = trial_id
                decision_idx = 0  # Reset decision counter for new trial
            
            print(f"DEBUG: Message flow - decision_req #{decision_idx + 1} for trial {trial_id}")
            
            reachable_objects_list = json_message['reachable_objects']
            image_converter = ImageConverter(json_message['config_name'], mujoco_model_dir, mujoco_model_type)
            inp = image_converter.process_datapoint(json_message, json_message['robot_goal'])
            obj2center_px = inp['obj2center_px']
            
            inp_img = inp.copy()
            
            # Check if model uses coordinate grid
            use_coord_grid = getattr(model_cfg.data, 'use_coord_grid', False)
            
            # Create coordinate grid if needed
            coord_grid = None
            if use_coord_grid:
                image_size = inp['robot_image'].shape[0]
                ys, xs = np.meshgrid(np.linspace(0, 1, image_size), 
                                     np.linspace(0, 1, image_size), 
                                     indexing='ij')
                coord_grid = np.stack([xs, ys], axis=-1)
                coord_grid = coord_grid.reshape(image_size, image_size, 2).astype(np.float32)
            
            # Concatenate input channels
            input_channels = [inp['robot_image'], inp['goal_image'], inp['movable_objects_image'], inp['static_objects_image'], inp['reachable_objects_image']]
            if coord_grid is not None:
                input_channels.append(coord_grid)
                
            # print(input_channels[0].shape, input_channels[1].shape, input_channels[2].shape, input_channels[3].shape, input_channels[4].shape, input_channels[5].shape if len(input_channels) > 5 else None)
            
            inp = np.concatenate(input_channels, axis=-1)
            inp = transform(inp).unsqueeze(0).to("cuda")
            # print(inp.shape)
            
            samples = (model.sample_from_model(inp, tgt_size=model_cfg.model.model.out_ch, samples=16).permute(0, 2, 3, 1).cpu().numpy() + 1)/ 2
            obj_goal = {}
            obj_votes = {}
            
            for i in range(samples.shape[0]):
                
                image_idx_dir = images_dir / str(idx)
                image_idx_dir.mkdir(parents=True, exist_ok=True)
                
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
                scale = image_converter.IMG_SIZE / model_cfg.data.image_size
                predicted_obj_center = (int(predicted_obj_center[0] * scale), int(predicted_obj_center[1] * scale))
                # find object
                min_dist = float('inf')
                for obj_name, obj_center in obj2center_px.items():
                    d = dist(obj_center, predicted_obj_center)
                    if d < min_dist:
                        min_dist = d
                        min_obj_name = obj_name

                if min_obj_name not in obj_goal:
                    obj_goal[min_obj_name] = []
                    obj_votes[min_obj_name] = 0
                    
                # print(min_obj_name, obj_goal.keys())
                
                obj_votes[min_obj_name] += 1
                goal_mask = (samples[i][:, :, 1].copy() > 0.5) * 1.0
                goal_mask = goal_mask.astype(np.uint8)
                
                if min_obj_name in reachable_objects_list:
                    reachable_selection += 1
                
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(goal_mask)
                if num_labels > 2:
                    continue
                
                _, _, predicted_goal_center, goal_angle = find_rectangle_corners(goal_mask)
                if predicted_goal_center is None:
                    continue
                predicted_goal_center = (int(predicted_goal_center[0] * scale), int(predicted_goal_center[1] * scale))
                goal_center = list(image_converter.pixel_to_world(predicted_goal_center[0], predicted_goal_center[1]))
                final_quat = image_converter.rotate_relative_to_world(min_obj_name, goal_angle - obj_angle)
                prediction_img = np.zeros((model_cfg.data.image_size, model_cfg.data.image_size, 3))
                prediction_img[:, :, :2] = samples[i]
                obj_goal[min_obj_name].append((goal_center, final_quat, i, prediction_img))
                
                
            if obj_goal.keys() == []:
                print(f"DEBUG: ERROR - No object found for trial {trial_id}, decision #{decision_idx + 1}")
                new_json_message = {
                    'object': None,
                    'goal_center': None,
                    'final_quat': None,
                    'error': True,
                    'error_message': "No object found"
                }
                socket.send_string(json.dumps(new_json_message))
                continue
            
            # find the object with the most votes
            max_votes = 0
            for obj_name, votes in obj_votes.items():
                if votes > max_votes:
                    max_votes = votes
                    min_obj_name = obj_name
                    
            # print("selected object", min_obj_name)
            if min_obj_name not in obj_goal:
                print("something is wrong")
                
            if min_obj_name not in obj_goal or len(obj_goal[min_obj_name]) == 0:
                print("no goal found")
                new_json_message = {
                    'object': None,
                    'goal_center': None,
                    'final_quat': None,
                    'error': True,
                    'error_message': "No goal found"
                }
                socket.send_string(json.dumps(new_json_message))
                continue
            
            # SE(2) density-based goal selection instead of random choice
            valid_goals = []
            scale = image_converter.IMG_SIZE / model_cfg.data.image_size
            
            # Get object angle for rotation calculation
            for goal_center, final_quat, sample_idx, prediction_img in obj_goal[min_obj_name]:
                # Convert quaternion to euler angle (θ)
                goal_theta = R.from_quat(final_quat, scalar_first=True).as_euler('xyz')[2]  # Z-axis rotation
                
                valid_goals.append({
                    'index': sample_idx,
                    'goal_center': goal_center,
                    'final_quat': final_quat,
                    'x': goal_center[0],
                    'y': goal_center[1], 
                    'theta': goal_theta,
                    'prediction_img': prediction_img
                })
            
            # SE(2) density-based voting
            def se2_distance(goal1, goal2, w_pos=1.0, w_rot=0.3):
                """Calculate SE(2) distance between two goals"""
                # Position distance (meters)
                pos_dist = np.sqrt((goal1['x'] - goal2['x'])**2 + (goal1['y'] - goal2['y'])**2)
                
                # Angular distance (radians) - shortest path on circle  
                angle_diff = abs(goal1['theta'] - goal2['theta'])
                rot_dist = min(angle_diff, 2*np.pi - angle_diff)
                
                return w_pos * pos_dist + w_rot * rot_dist
            
            # Calculate density for each goal (count neighbors within threshold)
            threshold = 0.5  # SE(2) distance threshold
            densities = []
            
            for i, goal in enumerate(valid_goals):
                density = 0
                for j, other_goal in enumerate(valid_goals):
                    if i != j and se2_distance(goal, other_goal) < threshold:
                        density += 1
                densities.append(density)
            
            # Select all goals with highest density
            max_density = max(densities) if densities else 0
            best_goals = [goal for goal, density in zip(valid_goals, densities) if density == max_density]
            
            # If no consensus (all densities are 0), use all goals
            if max_density == 0:
                # print(f"No consensus found (max density: {max_density}), using all goals")
                cluster_goals = valid_goals
            else:
                # print(f"Found consensus with density: {max_density}, using {len(best_goals)} candidates")
                cluster_goals = best_goals
            
            # Prepare all goals in the cluster for sending
            goals_list = []
            for goal in cluster_goals:
                goals_list.append({
                    'goal_center': goal['goal_center'],
                    'final_quat': goal['final_quat'].tolist(),  # Convert numpy array to list
                    'index': goal['index']
                })
            
            # For backward compatibility, also select one goal as primary
            selected_goal = random.choice(cluster_goals)
            goal_center = selected_goal['goal_center']
            final_quat = selected_goal['final_quat'] 
            selected_sample_idx = selected_goal['index']
            prediction_img = selected_goal['prediction_img']
            
            # print(f"Selected goal at ({goal_center[0]:.3f}, {goal_center[1]:.3f}, θ={selected_goal['theta']:.3f})")
            
            
            # print(inp_img['robot_image'].shape, inp_img['goal_image'].shape, inp_img['movable_objects_image'].shape, inp_img['static_objects_image'].shape, inp_img['reachable_objects_image'].shape)
            reachable_objects_image =  np.zeros((model_cfg.data.image_size, model_cfg.data.image_size, 3))
            reachable_objects_image[:, :, 0] =  cv2.resize(inp_img['reachable_objects_image'], (model_cfg.data.image_size, model_cfg.data.image_size))
            
            new_inp = np.concatenate([inp_img['robot_image'] + inp_img['goal_image'], inp_img['movable_objects_image'], inp_img['static_objects_image']], axis=-1)
            
            
            final_img = np.concatenate([cv2.resize(new_inp, (model_cfg.data.image_size, model_cfg.data.image_size)), reachable_objects_image, prediction_img], axis=1)
            # print(decision_idx)
            
            try:    
                plt.imsave(image_idx_dir / f"sample_{decision_idx}.png", final_img)
                plt.close()
            except Exception as e:
                print(e)
                print(f"Error saving image")
            
            new_json_message = {
                'object': min_obj_name,
                'goal_center': goal_center,
                'final_quat': final_quat.tolist(),
                'goals_cluster': goals_list,  # All goals in the cluster
                'error': False,
                'error_message': ""
            }
            
            print(f"DEBUG: SUCCESS - Sending object '{min_obj_name}' for trial {trial_id}, decision #{decision_idx + 1}")
            socket.send_string(json.dumps(new_json_message))
            decision_idx += 1
            
if __name__ == "__main__":
    main()