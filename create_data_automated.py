import os
import magnum as mn
import numpy as np
import random
import json
import pickle
from pathlib import Path
import colorsys

# interpolation:
from scipy.interpolate import splprep, splev

from PIL import Image
import habitat_sim

import attr
import magnum as mn
import numpy as np
import quaternion
from habitat_sim import registry
from habitat_sim.agent import SceneNodeControl

from habitat_sim.utils.common import d3_40_colors_rgb, quat_from_coeffs
from generate_prompts import HabitatSceneDataReader

import cv2
from create_data import DataRecorder

### TODO: implement automated prompt generator

# Only works for Matterport dataset
class AutomatedDataRecorder(DataRecorder):
    def __init__(self, scene_id: int = 0, rec_type = "automated"):
        super().__init__(scene_id=scene_id, rec_type = rec_type)
        self.dataset = 'hm3d'
        if self.dataset == 'hm3d':
            hm3d_dir = "data/scene_datasets/hm3d/"
            self.scene_config = os.path.join(hm3d_dir, "hm3d_annotated_basis.scene_dataset_config.json")
            all_scenes = self.get_all_scenes()
            self.test_scene = os.path.join(hm3d_dir, [i for i in all_scenes if "train" in i][scene_id])

            self.scene_name = self.test_scene.split("/")[-1].split(".")[0]

        with open("/home/nico/semesterproject/data/re-id_benchmark_ycb/object_groups.json") as f:
            self.object_groups = json.load(f)
        self.object_categories = [["kitchen_items"], ["toys"], ["tools"]]

        trajectory_path = f"/home/nico/semesterproject/data/re-id_benchmark_ycb/trajectories/train/{self.scene_name}.pickle"
        self.trajectory_dict = self.load_trajectory_dict(trajectory_path)
        self.trajectory_list = self.trajectory_dict["trajectory"]
    
        self.selected_coordinates = (0,0)

        self.dynamic_scene = True
        self.selected = False
        
    def create_data_auto(self):
        self.save = True
        print(self.object_categories)
        random.shuffle(self.object_categories, random.random)
        print(self.object_categories)
        for idx, trajectory in enumerate(self.trajectory_list):
            if idx <= 1: continue
            self.create_folder_structure(idx)
            self.initialize_habitat_sim()
            self.spawn_objects_along_trajectory(trajectory, traj_id=idx)
            self.sim.step_physics(1.0/60.0)
            self.play_trajectory(trajectory)
            self.remove_all_objects()
            self.save_count = 0
            self.postprocess()

    def create_folder_structure(self, id):
        self.datadir = os.path.join(self.out_dir, self.scene_name + "_" + str(id))
        os.makedirs(os.path.join(self.datadir, "color/"), exist_ok=True)
        os.makedirs(os.path.join(self.datadir, "depth/"), exist_ok=True)
        os.makedirs(os.path.join(self.datadir, "semantic/"), exist_ok=True)
        os.makedirs(os.path.join(self.datadir, "semantic_raw/"), exist_ok=True)

        #create file if not exist:
        if not os.path.exists(os.path.join(self.datadir, "camera_poses.json")):
            with open(os.path.join(self.datadir, "camera_poses.json"),'a+') as f:
                json.dump({}, f)
        with open(os.path.join(self.datadir, "camera_poses.json"),'r') as f:
            self.camera_poses = json.load(f)
        
        if not os.path.exists(os.path.join(self.datadir, "color_map.json")):
            with open(os.path.join(self.datadir, "color_map.json"),'a+') as f:
                json.dump({}, f)

    def spawn_objects_along_trajectory(self, trajectory, traj_id):

        all_objects = []
        for category in self.object_categories:
            all_objects += [self.object_groups[category[0]][random.randint(0, len(self.object_groups[category[0]])-1)]]

        for i in range(10):
            random_category = self.object_categories[random.randint(0, len(self.object_categories)-1)][0]
            random_object = random.randint(0, len(self.object_groups[random_category])-1)
            object_add = self.object_groups[random_category][random_object]
            if object_add not in all_objects:
                all_objects += [object_add]
            
            all_objects = list(np.unique(all_objects))

        for idx, obj in enumerate(all_objects):
            state = trajectory[20*idx]
            agent_state = habitat_sim.AgentState()
            agent_state.position = state[0]
            agent_state.rotation = state[1]
            agent_state.sensor_states['color_sensor'] = habitat_sim.SixDOFPose(state[2], state[3])
            agent_state.sensor_states['depth_sensor'] = habitat_sim.SixDOFPose(state[2], state[3])
            agent_state.sensor_states['semantic_sensor'] = habitat_sim.SixDOFPose(state[2], state[3])
            self.sim.agents[0].set_state(agent_state, infer_sensor_states=False)
            obj_pos = self.get_object_position(obj, agent_state.sensor_states['color_sensor'])
            if obj_pos is None:
                continue
            self.spawn_object_at_pos(obj, obj_pos)
            self.sim.step_physics(1.0/120.0)
            #test:
            observation = self.sim.get_sensor_observations()
            cv2.imshow("object_placement", cv2.cvtColor(np.asarray(Image.fromarray(observation["color_sensor"], mode="RGBA")), cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
            print("spawned object: ", obj)

    
    def get_object_position(self, object: str, sensor_state: habitat_sim.SixDOFPose):
        pos = sensor_state.position
        rot = sensor_state.rotation

        observation = self.sim.get_sensor_observations()
        cv2.imshow("object_placement", cv2.cvtColor(np.asarray(Image.fromarray(observation["color_sensor"], mode="RGBA")), cv2.COLOR_BGR2RGB))

        print("Object: ", object)
        cv2.setMouseCallback("object_placement", self.select)
        while True:
            print("select surface to spawn object on. press 's' to skip")
            k = cv2.waitKey(0)
            if k == ord("s"):
                return None
            elif self.selected:
                break

        u,v = self.selected_coordinates
        depth = observation["depth_sensor"][v,u]
        # projective transformation:
        P_cam = - depth * np.linalg.inv(self.camera_matrix).dot(np.array([[1280-u],[v],[1]]))
        P_world = quaternion.as_rotation_matrix(rot).dot(P_cam).T + pos + habitat_sim.geo.UP * 0.3

        self.selected = False

        return P_world[0]

    def select(self, event, u,v,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(u,v)
            self.selected_coordinates = (u,v)
            self.selected = True

    def spawn_object_at_pos(self, object_name, spawn_position):
        template_handles = self.obj_templates_mgr.get_template_handles()
        object_template_handle = os.path.join(self.object_config_dir, object_name + ".object_config.json")
        if not Path.exists(Path(object_template_handle)):
            print("invalid object name")
            return
        obj = self.rigid_obj_mgr.add_object_by_template_handle(object_template_handle)
        obj.translation = spawn_position

        obj.velocity_control.controlling_lin_vel = True
        obj.velocity_control.controlling_ang_vel = True
        obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
        self.object_list.append(obj)

    def load_trajectory_dict(self, trajectory_path):
        with open(trajectory_path, "rb") as f:
            trajectory_dict = pickle.load(f)
        return trajectory_dict

    def remove_all_objects(self):
        for obj in self.object_list:
            self.rigid_obj_mgr.remove_object_by_handle(obj.handle)
        self.object_list = []

    def postprocess(self):
        print("postprocessing...")
        cv2.destroyAllWindows()
        self.save_camera_poses()
        self.postprocess_semantic_obs()
        self.save_color_map()

        pr = "n" # input("generate prompts (y/n)?")

        if pr == "y":
            datareader = HabitatSceneDataReader(self.datadir)

            image_names = [n.split('.')[0] for n in sorted(os.listdir(datareader.rgb_path))][::20]

            for image in image_names:
                datareader(image)

            datareader.save_prompts()

def main():
    scene_id = 10
    rec = AutomatedDataRecorder(scene_id)
    rec.create_data_auto()





if __name__ == "__main__":
    main()