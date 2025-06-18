from omegaconf import DictConfig, OmegaConf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4"
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

@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg):
    print("Config loaded from:", cfg)
    run_path = Path(cfg.run_path)
    
    checkpoint_dir = run_path / "logs" / cfg.trainer.logger.name / "version_0" / "checkpoints"
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    checkpoint_path = sorted(checkpoint_files)[-1]
    
    model = hydra.utils.instantiate(cfg.model)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"✅ Model loaded successfully: {type(model).__name__}")
    
    model.to("cuda")
    
    # start zmq server
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://arrakis.cs.rutgers.edu:5555")
    print("ZMQ server started on port 5555")
    
    mujoco_model_dir = "/common/home/dm1487/robotics_research/ktamp/ml4kp_ktamp/resources/models/custom_walled_envs/may5"
    mujoco_model_type = "random_start_random_goal_many_env"
    
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((cfg.data.image_size, cfg.data.image_size)),
            transforms.Lambda(lambda x: x * 2 - 1),
    ])
    
    while True:
        message = socket.recv_string()
        json_message = json.loads(message) # recieve json message to construct the image
        
        
        # message type
        
        print(json_message.keys())
        image_converter = ImageConverter(json_message['config_name'], mujoco_model_dir, mujoco_model_type)
        inp = image_converter.process_datapoint(json_message, json_message['robot_goal'])
        print(json_message['reachable_objects'])
        obj2center_px = inp['obj2center_px']
        # obj2angle = inp['obj2angle']
        
        plt.figure()
        plt.imshow(inp['scene'])
        plt.savefig("test/scene.png")
        
        plt.figure()
        plt.imshow(inp['reachable_objects'])
        plt.savefig("test/reachable_objects.png")
        
        inp = np.concatenate([inp['scene'], inp['reachable_objects'][:, :, :1]], axis=-1)
        inp = transform(inp).unsqueeze(0).to("cuda")
        
        samples = (model.sample_from_model(inp, samples=8).permute(0, 2, 3, 1).cpu().numpy() + 1)/ 2
        
        obj_goal = {}
        obj_votes = {}
        
        for i in range(samples.shape[0]):
            img = np.zeros((64, 64, 3))
            img[:, :, :2] = samples[i]
            
            object_mask = (samples[i][:, :, 0].copy() > 0.5) * 1.0
            object_mask = object_mask.astype(np.uint8)
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(object_mask)
            if num_labels > 2:
                continue
            
            _, _, predicted_obj_center, obj_angle = find_rectangle_corners(object_mask)
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
            print(min_obj_name)

            if min_obj_name not in obj_goal:
                obj_goal[min_obj_name] = []
                obj_votes[min_obj_name] = 0
            
            obj_votes[min_obj_name] += 1
            goal_mask = (samples[i][:, :, 1].copy() > 0.5) * 1.0
            goal_mask = goal_mask.astype(np.uint8)
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(goal_mask)
            if num_labels > 2:
                continue
            
            _, _, predicted_goal_center, goal_angle = find_rectangle_corners(goal_mask) 
            predicted_goal_center = (int(predicted_goal_center[0] * scale), int(predicted_goal_center[1] * scale))
            goal_center = list(image_converter.pixel_to_world(predicted_goal_center[0], predicted_goal_center[1]))
            final_quat = image_converter.rotate_relative_to_world(min_obj_name, goal_angle - obj_angle)
            
            obj_goal[min_obj_name].append((goal_center, final_quat, i))
           
            plt.imsave(f"test/sample_{i}.png", np.concatenate([object_mask, goal_mask], axis=-1), cmap='gray')
            plt.close()
            
        # find the object with the most votes
        max_votes = 0
        for obj_name, votes in obj_votes.items():
            if votes > max_votes:
                max_votes = votes
                min_obj_name = obj_name
                
        # print(obj_votes)
                
        chosen_goal = random.choice(obj_goal[min_obj_name])
        goal_center, final_quat, selected_sample_idx = chosen_goal
        
        print(min_obj_name, goal_center, final_quat, selected_sample_idx)
        
        
        new_json_message = {
            'object': min_obj_name,
            'goal_center': goal_center,
            'final_quat': final_quat.tolist(),
            'error': False,
            'error_message': ""
        }
        
        input("enter to send")
        socket.send_string(json.dumps(new_json_message))
            

if __name__ == "__main__":
    main()