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
    run_path = Path(cfg.run_path)
    
    custom_output_dir = Path(f"{run_path}/ood_results_25")
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
    
    checkpoint_dir = run_path / "checkpoints"
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    for checkpoint_file in checkpoint_files:
        if "epoch" in checkpoint_file.name:
            checkpoint_path = checkpoint_file
            break
    
    model = hydra.utils.instantiate(cfg.model)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"✅ Model loaded successfully: {type(model).__name__}")
    
    model.to("cuda")
    
    # start zmq server
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://arrakis.cs.rutgers.edu:5556")
    print("ZMQ server started on port 5556")
    
    mujoco_model_dir = "/common/home/dm1487/robotics_research/ktamp/ml4kp_ktamp/resources/models/custom_walled_envs/jun22"
    mujoco_model_type = "random_start_random_goal_single_obstacle_room_2_200k_halfrad"
    
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((cfg.data.image_size, cfg.data.image_size)),
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
            
            inp = np.concatenate([inp['robot_image'], inp['goal_image'], inp['movable_objects_image'], inp['static_objects_image'], inp['reachable_objects_image']], axis=-1)
            inp = transform(inp).unsqueeze(0).to("cuda")
            
            samples = (model.sample_from_model(inp, samples=4).permute(0, 2, 3, 1).cpu().numpy() + 1)/ 2
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
                scale = image_converter.IMG_SIZE / cfg.data.image_size
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
                    
                print(min_obj_name, obj_goal.keys())
                
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
                prediction_img = np.zeros((cfg.data.image_size, cfg.data.image_size, 3))
                prediction_img[:, :, :2] = samples[i]
                obj_goal[min_obj_name].append((goal_center, final_quat, i, prediction_img))
                
                
            if obj_goal.keys() == []:
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
            
            # find the object with the most votes
            max_votes = 0
            for obj_name, votes in obj_votes.items():
                if votes > max_votes:
                    max_votes = votes
                    min_obj_name = obj_name
                    
            print("selected object", min_obj_name)
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
            
            chosen_goal = random.choice(obj_goal[min_obj_name])
            goal_center, final_quat, selected_sample_idx, prediction_img = chosen_goal
            
            
            # print(inp_img['robot_image'].shape, inp_img['goal_image'].shape, inp_img['movable_objects_image'].shape, inp_img['static_objects_image'].shape, inp_img['reachable_objects_image'].shape)
            reachable_objects_image =  np.zeros((cfg.data.image_size, cfg.data.image_size, 3))
            reachable_objects_image[:, :, 0] =  cv2.resize(inp_img['reachable_objects_image'], (cfg.data.image_size, cfg.data.image_size))
            
            new_inp = np.concatenate([inp_img['robot_image'] + inp_img['goal_image'], inp_img['movable_objects_image'], inp_img['static_objects_image']], axis=-1)
            
            
            final_img = np.concatenate([cv2.resize(new_inp, (cfg.data.image_size, cfg.data.image_size)), reachable_objects_image, prediction_img], axis=1)
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
                'error': False,
                'error_message': ""
            }
            socket.send_string(json.dumps(new_json_message))
            decision_idx += 1
            
if __name__ == "__main__":
    main()