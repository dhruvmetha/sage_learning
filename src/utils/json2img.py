import mujoco
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

class ImageConverter:
    IMG_SIZE = 224
    WORLD_SIZE = 6.0  # -3 to 3
    SCALE = IMG_SIZE / WORLD_SIZE  # pixels per world unit
    
    def __init__(self, env_config_name, model_folder, model_type):
        self.object_sizes = self.get_geom_sizes(env_config_name, model_folder, model_type)
    
    @staticmethod
    def get_geom_sizes(config_name, model_dir, model_type):
        model = mujoco.MjModel.from_xml_path(f"{model_dir}/{model_type}/{config_name}.xml")
        # Get all geom information
        geom_sizes = {}
        for i in range(model.ngeom):
            # Get geom name
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name is None:
                geom_name = f"geom_{i}"
            geom_sizes[geom_name] = model.geom_size[i]
        del model
        return geom_sizes
    
    def pixel_to_world(self, px, py):
        x = (px / self.SCALE) - self.WORLD_SIZE/2
        y = (py / self.SCALE) - self.WORLD_SIZE/2
        return x, y
    
    def rotate_relative_to_world(self, obj_name, angle):
        if self.data_point is None:
            raise ValueError("data_point is not set")
        # obj_center = self.data_point['objects'][obj_name]['position']
        obj_angle = self.data_point['objects'][obj_name]['quaternion']
        obj_angle = R.from_quat(obj_angle, scalar_first=True).as_euler('xyz', degrees=True)[2]
        # print("converting obj_angle, angle", obj_angle, angle, "to", obj_angle + angle)
        final_quat  = R.from_euler('xyz', [0, 0, obj_angle + angle], degrees=True).as_quat(scalar_first=True)
        return final_quat
    
    def _world_to_pixel(self, x, y):
        px = int((x + self.WORLD_SIZE/2) * self.SCALE)
        py = int((y + self.WORLD_SIZE/2) * self.SCALE)
        return px, py
    
    def _pixel_to_world(self, px, py):
        x = (px / self.SCALE) - self.WORLD_SIZE/2
        y = (py / self.SCALE) - self.WORLD_SIZE/2
        return x, y
    
    def _draw_rotated_rectangle(self, img, center, size, angle, color):
        center_px = self._world_to_pixel(center[0], center[1])
        size_px = (int(size[0] * self.SCALE), int(size[1] * self.SCALE))
        rect = ((center_px[0], center_px[1]), size_px, angle)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.fillPoly(img, [box], color)
        return img, center_px
    
    def _draw_circle(self, img, center, radius, color):
        center_px = self._world_to_pixel(center[0], center[1])
        radius_px = int(radius * self.SCALE)
        cv2.circle(img, center_px, radius_px, color, -1)
        return img
    
    def process_datapoint(self, data_point, robot_goal_pos):
        # Create base images
        self.data_point = data_point
        
        scene_image = np.zeros((self.IMG_SIZE, self.IMG_SIZE, 3), dtype=np.uint8)
        reachable_objects_image = np.zeros((self.IMG_SIZE, self.IMG_SIZE, 3), dtype=np.uint8)
        object_mask = np.zeros((self.IMG_SIZE, self.IMG_SIZE, 1), dtype=np.uint8)
        goal_mask = np.zeros((self.IMG_SIZE, self.IMG_SIZE, 1), dtype=np.uint8)
        
        obj2center_px = {}
        obj2angle = {}
        
        # Draw robot
        robot_pos = data_point['robot']['position']
        self._draw_circle(scene_image, (robot_pos[0], robot_pos[1]), 0.2, (255, 0, 0))
        
        # Draw goal indicator
        self._draw_circle(scene_image, (robot_goal_pos[0], robot_goal_pos[1]), 0.25, (0, 255, 0))
        
        # Draw objects
        for obj_name, obj_data in data_point['objects'].items():
            # TODO: for now, remove in next iteration of diffusion training
            if "movable" not in obj_name:
                continue
            size_x, size_y = self.object_sizes[obj_name][0], self.object_sizes[obj_name][1]
            size_x *= 2
            size_y *= 2
            rotation = R.from_quat(obj_data['quaternion'], scalar_first=True).as_euler('xyz', degrees=True)[2]
            _, center_px = self._draw_rotated_rectangle(scene_image,
                                      (obj_data['position'][0], obj_data['position'][1]),
                                      (size_x, size_y),
                                      rotation,
                                      (255, 255, 0))
            
            obj2center_px[obj_name] = center_px
            obj2angle[obj_name] = rotation
            
            if obj_name in data_point['reachable_objects']:
                self._draw_rotated_rectangle(reachable_objects_image,
                                      (obj_data['position'][0], obj_data['position'][1]),
                                      (size_x, size_y),
                                      rotation,
                                      (255, 255, 0))
                
            
        # if 'reachable_objects' in data_point['state']:
        #     for obj_name in data_point['state']['reachable_objects']:
        #         print(obj_name)
        #         size_x, size_y = self.object_sizes[obj_name]
        #         size_x *= 2
        #         size_y *= 2
        #         rotation = R.from_quat(obj_data['quaternion'], scalar_first=True).as_euler('xyz', degrees=True)[2]
        #         self._draw_rotated_rectangle(reachable_objects_image,
        #                               (obj_data['position'][0], obj_data['position'][1]),
        #                               (size_x, size_y),
        #                               rotation,
        #                               (255, 255, 0))
        
        # Draw masks if action exists
        # if 'action' in data_point:
        #     goal_obj = data_point['action']['object_name']
        #     obj_data = data_point['state']['objects'][goal_obj]
        #     size_x, size_y = self.object_sizes[goal_obj][0], self.object_sizes[goal_obj][1]
        #     size_x *= 2
        #     size_y *= 2
            
        #     # Object mask
        #     rotation = R.from_quat(obj_data['quaternion'], scalar_first=True).as_euler('xyz', degrees=True)[2]
        #     self._draw_rotated_rectangle(object_mask,
        #                               (obj_data['position'][0], obj_data['position'][1]),
        #                               (size_x, size_y),
        #                               rotation,
        #                               255)
            
        #     # Goal mask
        #     goal_state = data_point['action']['goal_state']
        #     rotation = R.from_quat(goal_state['quaternion'], scalar_first=True).as_euler('xyz', degrees=True)[2]
        #     self._draw_rotated_rectangle(goal_mask,
        #                               (goal_state['position'][0], goal_state['position'][1]),
        #                               (size_x, size_y),
        #                               rotation,
        #                               255)
        
        # Normalize arrays
        return {
            'scene': scene_image.astype(np.float32) / 255.0,
            # 'object_mask': object_mask.astype(np.float32) / 255.0,
            # 'goal_mask': goal_mask.astype(np.float32) / 255.0,
            'reachable_objects': reachable_objects_image.astype(np.float32) / 255.0,
            'obj2center_px': obj2center_px,
            'obj2angle': obj2angle
        }