from omegaconf import DictConfig, OmegaConf
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
import time

def generate_comprehensive_goal_image(decision_data, selected_goal_index, data_cfg):
    """Generate comprehensive image with Option 2 layout: Top row + Grid"""
    
    inp_img = decision_data['inp_img']
    selected_object_mask = decision_data['selected_object_mask']
    valid_goals = decision_data['valid_goals']
    
    # Create base visualization components
    new_inp = np.concatenate([inp_img['robot_image'] + inp_img['goal_image'], 
                             inp_img['movable_objects_image'], 
                             inp_img['static_objects_image']], axis=-1)
    new_inp_resized = cv2.resize(new_inp, (data_cfg.image_size, data_cfg.image_size))
    
    # Selected object mask
    selected_object_vis = np.zeros((data_cfg.image_size, data_cfg.image_size, 3))
    selected_object_vis[:, :, 0] = cv2.resize(selected_object_mask[:, :, 0], (data_cfg.image_size, data_cfg.image_size))
    
    # Find the selected goal for top row highlight
    selected_goal_vis = None
    for goal in valid_goals:
        if goal['index'] == selected_goal_index:
            selected_goal_sample = goal['goal_sample']
            selected_goal_vis = np.zeros((data_cfg.image_size, data_cfg.image_size, 3))
            selected_goal_vis[:, :, 0] = cv2.resize(selected_object_mask[:, :, 0], (data_cfg.image_size, data_cfg.image_size))
            selected_goal_vis[:, :, 1] = selected_goal_sample[:, :, 0]  # Goal mask in green
            selected_goal_vis[:, :, 2] = 0.7  # Strong red highlight for selected goal
            break
    
    # If no selected goal found, create empty panel
    if selected_goal_vis is None:
        selected_goal_vis = np.zeros((data_cfg.image_size, data_cfg.image_size, 3))
    
    # GRID: Create grid of all goal candidates
    goal_panels = []
    for i, goal in enumerate(valid_goals):
        goal_sample = goal['goal_sample']
        
        # Create prediction image for this goal
        prediction_img = np.zeros((data_cfg.image_size, data_cfg.image_size, 3))
        prediction_img[:, :, 0] = cv2.resize(selected_object_mask[:, :, 0], (data_cfg.image_size, data_cfg.image_size))
        prediction_img[:, :, 1] = goal_sample[:, :, 0]  # Goal mask in green channel
        
        # Subtle highlight if this is the selected goal
        if goal['index'] == selected_goal_index:
            prediction_img[:, :, 2] = 0.3  # Subtle red highlight for consistency
            
        goal_panels.append(prediction_img)
    
    # Determine grid dimensions (6 columns max for readability)
    cols = min(6, max(3, len(goal_panels))) if goal_panels else 6  # At least 3 columns to match top row
    rows = (len(goal_panels) + cols - 1) // cols if goal_panels else 1
    
    # Pad top row to match grid width if needed
    top_row_panels = [new_inp_resized, selected_object_vis, selected_goal_vis]
    while len(top_row_panels) < cols:
        empty_panel = np.zeros((data_cfg.image_size, data_cfg.image_size, 3))
        top_row_panels.append(empty_panel)
    
    # Create top row with same width as grid
    top_row = np.concatenate(top_row_panels[:cols], axis=1)
    
    # Arrange goal panels in grid
    if goal_panels:
        # Pad with empty panels if needed to complete the grid to full rows
        total_panels_needed = rows * cols
        while len(goal_panels) < total_panels_needed:
            empty_panel = np.zeros((data_cfg.image_size, data_cfg.image_size, 3))
            goal_panels.append(empty_panel)
        
        # Create grid rows - ensure each row has exactly 'cols' panels
        grid_rows = []
        for row in range(rows):
            start_idx = row * cols
            end_idx = start_idx + cols  # Always take exactly 'cols' panels
            row_panels = goal_panels[start_idx:end_idx]
            
            # Double check we have the right number of panels
            assert len(row_panels) == cols, f"Row {row} has {len(row_panels)} panels, expected {cols}"
            
            grid_row = np.concatenate(row_panels, axis=1)
            grid_rows.append(grid_row)
        
        # Combine all grid rows vertically - now all rows have same width
        goals_grid = np.concatenate(grid_rows, axis=0)
        
        # Combine top row and goals grid vertically
        final_img = np.concatenate([top_row, goals_grid], axis=0)
    else:
        # No goals, just use top row
        final_img = top_row
    
    return final_img

def periodic_cleanup_decision_history(decision_history, current_trial_id, max_age_seconds=300):
    """Clean up old decision history entries to prevent memory leaks"""
    current_time = time.time()
    keys_to_remove = []
    
    for key, data in decision_history.items():
        trial_id, decision_idx = key
        # Remove entries older than max_age_seconds or from trials that are far behind current
        entry_age = current_time - data.get('timestamp', current_time)
        is_old_trial = current_trial_id is not None and trial_id < (current_trial_id - 2)
        
        if entry_age > max_age_seconds or is_old_trial:
            keys_to_remove.append(key)
    
    if keys_to_remove:
        print(f"DEBUG: Periodic cleanup removing {len(keys_to_remove)} old decision history entries")
        for key in keys_to_remove:
            del decision_history[key]

@hydra.main(config_path="../config", config_name="split_inference.yaml", version_base=None)
def main(cfg):
    print("Split inference config loaded from:", cfg)
    
    # Get run paths for obstacle and goal models from config - no defaults
    if 'obstacle_run_path' not in cfg or 'goal_run_path' not in cfg:
        raise ValueError("Both obstacle_run_path and goal_run_path must be specified in config")
    
    obstacle_run_path = Path(cfg.obstacle_run_path)
    goal_run_path = Path(cfg.goal_run_path)
    
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
    
    # Get results suffix from config or use default
    results_suffix = cfg.get('results_suffix', 'split_results')
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
    
    # Decision history storage for deferred image generation
    decision_history = {}  # Key: (trial_id, decision_idx), Value: decision data
    request_count = 0  # Counter for periodic cleanup
    
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
    
    # Get ZMQ configuration from config or use defaults
    zmq_host = cfg.get('zmq', {}).get('host', 'arrakis.cs.rutgers.edu')
    zmq_port = cfg.get('zmq', {}).get('port', 5556)
    
    socket.bind(f"tcp://{zmq_host}:{zmq_port}")
    print(f"ZMQ server started on {zmq_host}:{zmq_port}")
    
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
            trial_id = json_message.get('trial_id', current_trial_id)
            
            print(f"DEBUG: Trial {trial_id} completed. Success: {success}, Action steps: {action_steps}, Decisions: {decision_idx}")
            print(f"DEBUG: Message flow - result_info received for trial {trial_id}")
            
            # Handle final decision feedback if available
            if 'final_selected_goal_index' in json_message and decision_idx > 0:
                final_selected_goal_index = json_message['final_selected_goal_index']
                final_decision_idx = decision_idx - 1  # Last decision index
                
                print(f"DEBUG: Final decision feedback - trial {trial_id}, decision {final_decision_idx}, selected goal {final_selected_goal_index}")
                
                # Generate corrected image for final decision
                final_decision_key = (trial_id, final_decision_idx)
                if final_decision_key in decision_history:
                    try:
                        decision_data = decision_history[final_decision_key]
                        comprehensive_img = generate_comprehensive_goal_image(
                            decision_data, final_selected_goal_index, data_cfg)
                        
                        # Save corrected image
                        image_idx_dir = images_dir / str(decision_data['idx'])
                        image_idx_dir.mkdir(parents=True, exist_ok=True)
                        
                        filename = f"sample_{final_decision_idx}_FINAL_CORRECTED.png"
                        plt.imsave(image_idx_dir / filename, comprehensive_img)
                        plt.close()
                        
                        print(f"DEBUG: Saved final corrected image for trial {trial_id}, decision {final_decision_idx}")
                        
                        # Clean up old decision data
                        del decision_history[final_decision_key]
                        
                    except Exception as e:
                        print(f"ERROR: Failed to generate final corrected image: {e}")
                else:
                    print(f"WARNING: No stored data found for final decision trial {trial_id}, decision {final_decision_idx}")
            
            # Clean up any remaining decision history for this trial
            keys_to_remove = [key for key in decision_history.keys() if key[0] == trial_id]
            for key in keys_to_remove:
                del decision_history[key]
                print(f"DEBUG: Cleaned up decision history for trial {trial_id}, decision {key[1]}")
            
            # Write results to file (including failed trials with 0 action steps)
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
            # Periodic cleanup every 50 requests to prevent memory leaks
            request_count += 1
            if request_count % 50 == 0:
                periodic_cleanup_decision_history(decision_history, current_trial_id)
            
            # Track trial ID and detect new trials
            trial_id = json_message.get('trial_id', 'unknown')
            if current_trial_id != trial_id:
                if current_trial_id is not None:
                    print(f"DEBUG: New trial detected. Previous: {current_trial_id}, Current: {trial_id}")
                    
                    # Clean up any leftover decision history from previous trial
                    # This handles cases where result_info cleanup didn't happen
                    prev_keys_to_remove = [key for key in decision_history.keys() if key[0] == current_trial_id]
                    if prev_keys_to_remove:
                        print(f"WARNING: Cleaning up {len(prev_keys_to_remove)} leftover decision entries from previous trial {current_trial_id}")
                        for key in prev_keys_to_remove:
                            del decision_history[key]
                            print(f"DEBUG: Emergency cleanup - removed decision data for trial {key[0]}, decision {key[1]}")
                else:
                    print(f"DEBUG: First trial started. Trial ID: {trial_id}")
                current_trial_id = trial_id
                decision_idx = 0  # Reset decision counter for new trial
            
            print(f"DEBUG: Message flow - decision_req #{decision_idx + 1} for trial {trial_id}")
            
            # Handle previous decision feedback if available
            if 'previous_decision' in json_message:
                prev_info = json_message['previous_decision']
                prev_trial_id = trial_id  # Same trial
                prev_decision_idx = prev_info['decision_step']
                selected_goal_index = prev_info['selected_goal_index']
                
                print(f"DEBUG: Received feedback - trial {prev_trial_id}, decision {prev_decision_idx}, selected goal {selected_goal_index}")
                
                # Validate decision indices are in sync
                if prev_decision_idx >= decision_idx:
                    print(f"ERROR: Received feedback for future/current decision {prev_decision_idx}, current is {decision_idx}. Indices may be out of sync!")
                    # Still try to process if data exists, but log the mismatch
                
                # Generate corrected image for previous decision
                prev_decision_key = (prev_trial_id, prev_decision_idx)
                if prev_decision_key in decision_history:
                    try:
                        decision_data = decision_history[prev_decision_key]
                        comprehensive_img = generate_comprehensive_goal_image(
                            decision_data, selected_goal_index, data_cfg)
                        
                        # Save corrected image
                        image_idx_dir = images_dir / str(decision_data['idx'])
                        image_idx_dir.mkdir(parents=True, exist_ok=True)
                        
                        filename = f"sample_{prev_decision_idx}_CORRECTED.png"
                        plt.imsave(image_idx_dir / filename, comprehensive_img)
                        plt.close()
                        
                        print(f"DEBUG: Saved corrected image for trial {prev_trial_id}, decision {prev_decision_idx}")
                        
                        # Clean up old decision data
                        del decision_history[prev_decision_key]
                        
                    except Exception as e:
                        print(f"ERROR: Failed to generate corrected image: {e}")
                else:
                    print(f"WARNING: No stored data found for trial {prev_trial_id}, decision {prev_decision_idx}")
            
            reachable_objects_list = json_message['reachable_objects']
            image_converter = ImageConverter(json_message['config_name'], mujoco_model_dir, mujoco_model_type)
            inp = image_converter.process_datapoint(json_message, json_message['robot_goal'])
            obj2center_px = inp['obj2center_px']
            
            inp_img = inp.copy()
            
            # Step 1: First inference with obstacle model to select object
            inp_for_obstacle = np.concatenate([inp['robot_image'], inp['goal_image'], inp['movable_objects_image'], inp['static_objects_image'], inp['reachable_objects_image']], axis=-1)
            inp_for_obstacle = transform(inp_for_obstacle).unsqueeze(0).to("cuda")
            
            # Generate obstacle/object samples using obstacle model
            obstacle_samples = (obstacle_model.sample_from_model(inp_for_obstacle, samples=32).permute(0, 2, 3, 1).cpu().numpy() + 1)/ 2
            
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
                print(f"DEBUG: ERROR - No object found for trial {trial_id}, decision #{decision_idx + 1}")
                new_json_message = {
                    'object': None,
                    'goal_center': None,
                    'final_quat': None,
                    'error': True,
                    'error_message': "No object found"
                }
                socket.send_string(json.dumps(new_json_message))
                decision_idx += 1  # Increment to stay in sync with C++ planning step
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
                decision_idx += 1  # Increment to stay in sync with C++ planning step
                continue
            
            # Step 3: Create modified input for goal model (replace reachable_objects with object mask)
            inp_for_goal = np.concatenate([inp['robot_image'], inp['goal_image'], inp['movable_objects_image'], inp['static_objects_image'], selected_object_mask], axis=-1)
            inp_for_goal = transform(inp_for_goal).unsqueeze(0).to("cuda")
            
            # Step 4: Generate goal samples using goal model (8 samples)
            goal_samples = (goal_model.sample_from_model(inp_for_goal, samples=32).permute(0, 2, 3, 1).cpu().numpy() + 1)/ 2
            
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
                print(f"DEBUG: ERROR - No valid goals found for trial {trial_id}, decision #{decision_idx + 1}")
                new_json_message = {
                    'object': None,
                    'goal_center': None,
                    'final_quat': None,
                    'error': True,
                    'error_message': "No valid goals found"
                }
                socket.send_string(json.dumps(new_json_message))
                decision_idx += 1  # Increment to stay in sync with C++ planning step
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
            
            # Select all goals with highest density
            max_density = max(densities) if densities else 0
            best_goals = [goal for goal, density in zip(valid_goals, densities) if density == max_density]
            
            # If no consensus (all densities are 0), use all goals
            if max_density == 0:
                # print(f"No consensus found (max density: {max_density}), using all goals")
                cluster_goals = valid_goals
            else:
                # print(f"Found consensus with density: {max_density}, using {len(best_goals)} candidates")
                cluster_goals = valid_goals
            
            # print the number of goals in the cluster
            print(f"DEBUG: Number of goals in the cluster: {len(cluster_goals)}")
            
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
            
            # Extract final results for backward compatibility
            goal_center = selected_goal['goal_center']
            final_quat = selected_goal['final_quat']
            selected_goal_sample = selected_goal['goal_sample']
            
            # Store decision data for deferred image generation
            decision_key = (trial_id, decision_idx)
            decision_history[decision_key] = {
                'inp_img': inp_img.copy(),
                'selected_object_mask': selected_object_mask.copy(),
                'valid_goals': [goal.copy() for goal in valid_goals],  # Deep copy
                'goals_sent': goals_list.copy(),
                'timestamp': time.time(),
                'trial_id': trial_id,
                'decision_idx': decision_idx,
                'idx': idx  # For image directory naming
            }
            
            print(f"DEBUG: Stored decision data for trial {trial_id}, decision {decision_idx} ({len(valid_goals)} goals)")
            
            new_json_message = {
                'object': selected_object,
                'goal_center': goal_center,
                'final_quat': final_quat.tolist(),
                'goals_cluster': goals_list,  # All goals in the cluster
                'error': False,
                'error_message': ""
            }
            
            print(f"DEBUG: SUCCESS - Sending object '{selected_object}' for trial {trial_id}, decision #{decision_idx + 1}")
            socket.send_string(json.dumps(new_json_message))
            decision_idx += 1
            
if __name__ == "__main__":
    main()