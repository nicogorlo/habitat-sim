import os
import magnum as mn
import numpy as np
import random
import json
import pickle
from pathlib import Path
import colorsys
import math
from datetime import datetime

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

class DataRecorder():

    def __init__(self, scene_id: int = 3, rec_type = "manual") -> None:

        self.dataset = 'hm3d' # Replica, hm3d or Replica_CAD (Replica CAD no_semantics)

        if self.dataset == 'Replica':
            self.scene_config = "/home/nico/semesterproject/habitat-sim/data/Replica-Dataset/dataset/replica.scene_dataset_config.json"
            scenes = [i for i in sorted(os.listdir('data/Replica-Dataset/dataset/')) if '.' not in i]  # Replica: 18 scenes (scene_id 0 to 17)
            scene = scenes[scene_id]
            print("chosen scene: ", scene)
            self.test_scene = f"data/Replica-Dataset/dataset/{scene}/habitat/mesh_semantic.ply"
            self.scene_name = self.test_scene.split('/')[-3]
        elif self.dataset == 'hm3d':
            hm3d_dir = "data/scene_datasets/hm3d/"
            self.scene_config = os.path.join(hm3d_dir, "hm3d_annotated_basis.scene_dataset_config.json")
            all_scenes = self.get_all_scenes()
            self.test_scene = os.path.join(hm3d_dir, sorted([i for i in all_scenes if "train" in i])[10])
            self.scene_name = self.test_scene.split('/')[-1].split('.')[0]
            print("scene: ", self.test_scene)
        elif self.dataset == 'Replica_CAD':
            self.scene_config = "/home/nico/semesterproject/habitat-sim/data/replica_cad/replicaCAD.scene_dataset_config.json"
            self.test_scene = "apt_1"
            self.scene_name = self.test_scene + "_CAD"
        else:
            print("dataset not supported")
            exit()
        
        self.trajectory_folder = "/home/nico/semesterproject/data/re-id_benchmark_ycb/trajectories/train/"        
        self.out_dir = '/home/nico/semesterproject/data/re-id_benchmark_ycb'
        self.initial_state_dict_path = 'data/init_state_dict.pkl'
        self.object_config_dir = "data/replica_cad/configs/objects_new_label"

        self.save = False
        self.rec_type = rec_type
        
        width = 1280
        height = 720
        fov = math.pi/2
        self.camera_matrix = np.array(
            [[width / (2 * math.tan(fov / 2)), 0, width / 2],
            [0, width / (2 * math.tan(fov / 2)), height / 2],
            [0, 0, 1]], dtype=np.float32)

        self.agent_poses = []
        self.object_list = []

        self.count = 0
        self.save_count = 0

        self.translation_step = 0.1
        self.rotation_step = 4

        self.list_of_actions = []
        self.object_dynamics = {}
        self.dynamic_scene = False


        self.semantic_palette = np.array(self.generate_color_palette(256),dtype=np.uint8)
        
        self.color_map = {}

    def initialize_habitat_sim(self):

        self.sim_settings = {
            "scene": self.test_scene,  # Scene path
            "scene_config": self.scene_config,  # Scene config path
            "default_agent": 0,  # Index of the default agent
            "sensor_height": 1.5,  # Height of sensors in meters
            "width":  1280,  # Spatial resolution of the observations
            "height": 720,
            "color_sensor": True,  # RGB sensor
            "semantic_sensor": True,  # Semantic sensor
            "depth_sensor": True,  # Depth sensor
            "enable_physics": True,
        }

        # generate simple simulator config:

        self.cfg = self.make_simple_cfg(self.sim_settings)


        # # create simulator
        self.sim = habitat_sim.Simulator(self.cfg)

        # initialize an agent
        agent = self.sim.initialize_agent(self.sim_settings["default_agent"])

        # Set agent state
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array([-0.6, 1.0, 0.0])  # in world space
        agent_state.rotation = quat_from_coeffs([0, 0, 0, 1]) # x y z w
        agent.set_state(agent_state)

        # Load initial state dict if exists. The initial state Dict contains the initial state of the agent for each scene.
        try:
            Path(self.initial_state_dict_path).touch(exist_ok=False)
            with open(self.initial_state_dict_path, 'wb') as f:
                pickle.dump({self.scene_name: agent_state}, f)
        except FileExistsError:
            pass
        with open(self.initial_state_dict_path, 'rb') as f:
            self.agent_initial_state_dict = pickle.load(f)
        
        if self.scene_name in self.agent_initial_state_dict.keys():
            agent_state = self.agent_initial_state_dict[self.scene_name]
            agent.set_state(agent_state)

        # Get agent state
        agent_state = agent.get_state()
        print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

        # obtain the default, discrete actions that an agent can perform
        # default action space contains 3 actions: move_forward, turn_left, and turn_right
        self.action_names = list(self.cfg.agents[self.sim_settings["default_agent"]].action_space.keys())
        print("Discrete action space: ", self.action_names)

        self.key_action_dict = {
            106: "turn_left", 97: "move_left", 108:"turn_right", 
            100:"move_right", 110:"move_up",109:"move_down",
            105:"look_down", 107:"look_up", 119:"move_forward", 
            115:"move_backward"}
        # objects:
        self.obj_templates_mgr = self.sim.get_object_template_manager()

        # get the rigid object manager, which provides direct access to objects
        self.rigid_obj_mgr = self.sim.get_rigid_object_manager()

        self.obj_templates_mgr.load_configs(self.object_config_dir)

    def create_folder_structure(self):
        self.datadir = os.path.join(self.out_dir, self.scene_name + "_" + self.rec_type)
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

    def get_all_scenes(self):
        with open(self.scene_config, 'r') as f:
            scene_conf = json.load(f)
        
        all_scene_paths_short = scene_conf["stages"]["paths"][".glb"]
        scene_names = [i.split("-")[-1].split("/")[0] for i in all_scene_paths_short]
        all_scene_paths_full = [val.replace("*", scene_names[idx]) for idx, val in enumerate(all_scene_paths_short)]
        return sorted(all_scene_paths_full)


    def __call__(self, key):
        if key != 8 and key != 27 and key != 111 and key != 32:
            self.list_of_actions.append(key)

        print(self.count, self.save_count, key)
        if key == 81: # arrow left
            self.observations = self.sim.step("turn_left")
            if self.save:
                self.save_images()

        elif key == 97: # a
            self.observations = self.sim.step("move_left")
            if self.save:
                self.save_images()

        elif key == 83: # arrow right
            self.observations  = self.sim.step("turn_right")
            if self.save:
                self.save_images()

        elif key == 100: # d
            self.observations  = self.sim.step("move_right")
            if self.save:
                self.save_images()

        elif key == 110: # n
            self.observations  = self.sim.step("move_up")
            if self.save:
                self.save_images()

        elif key == 109: # m
            self.observations  = self.sim.step("move_down")
            if self.save:
                self.save_images()

        elif key == 82: # arrow up
            self.observations  = self.sim.step("look_up")
            if self.save:
                self.save_images()

        elif key == 84: # arrow down
            self.observations  = self.sim.step("look_down")
            if self.save:
                self.save_images()

        elif key == 119: # w
            self.observations = self.sim.step("move_forward")
            if self.save:
                self.save_images()

        elif key == 115: # s
            self.observations = self.sim.step("move_backward")
            if self.save:
                self.save_images()

        elif key == 113: # q
            self.observations = self.sim.step("loiter_left")
            if self.save:
                self.save_images()
        
        elif key == 101: # e
            self.observations = self.sim.step("loiter_right")
            if self.save:
                self.save_images()
        
        elif key == 85: #Page up
            self.observations = self.sim.step("tilt_left")
            if self.save:
                self.save_images()
        
        elif key == 86: #Page down
            self.observations = self.sim.step("tilt_right")
            if self.save:
                self.save_images()

        elif key == 114: # r
            self.record_pose()
            return

        elif key == 116: # t
            self.agent_poses.pop(-1)
            return

        elif key == 111: # o
            self.spawn_random_object_infront_of_agent()
            self.sim.step_physics(1.0)
            if self.save:
                self.save_images()
        
        elif key == 118: # v
            obj = input("Enter object name: ")
            self.spawn_object_infront_of_agent_by_string(obj, dst= 1.7)
            self.sim.step_physics(1.0/60.0)
            if self.save:
                self.save_images()

        elif key == 105: # i
            print("postprocessing previous scene")
            self.save_camera_poses()
            self.postprocess_semantic_obs()
            self.save_color_map()

            pr = input("generate prompts (y/n)?")

            if pr == "y":
                datareader = HabitatSceneDataReader(self.datadir)

                image_names = [n.split('.')[0] for n in sorted(os.listdir(datareader.rgb_path))][::20]

                for image in image_names:
                    datareader(image)

                datareader.save_prompts()

                cv2.destroyWindow('color')
                cv2.destroyWindow('seg')

            print("postprocessing done, initializing new scene")
            self.save = False
            self.save_count = 0
            self.agent_poses = []
            self.scene_name = input("Enter new scene name: ")
            self.datadir = os.path.join(self.out_dir, self.scene_name + "_" + self.rec_type)
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
            
        elif key == 32: # space
            self.save = True

        elif key == 102: # f
            self.save_images()
            self.save_count += 1
            return
        
        elif key == 108: # l
            self.save = True
            n = 400
            trajectory = self.interpolate_poses(self.agent_poses, n)
            if trajectory is None:
                print("faulty agent poses")
                self.agent_poses = []
                return
            self.play_trajectory(trajectory)
            self.save_trajectory(trajectory)
            return
        
        elif key == 98: # b
            n = 400
            trajectory = self.interpolate_poses(self.agent_poses, n)
            if trajectory is None:
                print("faulty agent poses")
                self.agent_poses = []
                return
            self.save_trajectory(trajectory)
            return
        
        elif key == 107: # k
            # obj = self.object_list[0]
            # obj.velocity_control.linear_velocity = [0.0, 0.0, 0.0]
            # obj.velocity_control.angular_velocity = [0.0, 0.0, 0.0]
            return

        
        elif key == 112: # p
            self.set_agent_state_as_initial_state()

        elif key == 8: # backspace

            if len(self.list_of_actions) == 0:
                return
            
            take_reverse_action = self.reverse_action(self.key_action_dict[self.list_of_actions[-1]])
            print(take_reverse_action)
            self.observations = self.sim.step(take_reverse_action)
            self.count -= 1
            
            if self.save:
                self.save_count -= 1
                try:
                    os.remove(os.path.join(self.out_dir, self.scene_name, f"color/{str(self.save_count).zfill(7)}.jpg"))
                    os.remove(os.path.join(self.out_dir, self.scene_name, f"depth/{str(self.save_count).zfill(7)}.jpg"))
                    os.remove(os.path.join(self.out_dir, self.scene_name, f"semantic/{str(self.save_count).zfill(7)}.jpg"))

                except:
                    name = os.path.join(self.out_dir, self.scene_name, f"color/{str(self.save_count).zfill(7)}.jpg")
                    print(f"there are no files by the name of {name}")
                    pass

            self.list_of_actions.pop(-1)

        self.sim.step_physics(1.0/30.0)

        if self.save:
            self.save_count += 1
        
        self.count+=1

    def make_simple_cfg(self, settings):
        # simulator backend
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = settings["scene"]
        sim_cfg.scene_dataset_config_file = settings["scene_config"]

        sim_cfg.enable_physics = settings["enable_physics"]

        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
        rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]

        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [settings['height'], settings['width']]
        depth_sensor_spec.position   = [0.0, settings['sensor_height'], 0.0]

        semantic_sensor_spec = habitat_sim.CameraSensorSpec()
        semantic_sensor_spec.uuid = "semantic_sensor"
        semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_sensor_spec.resolution = [settings['height'], settings['width']]
        semantic_sensor_spec.position = [0.0, settings['sensor_height'], 0.0]
        
        # Custom action spaces
        # We can also re-register this function such that it effects just the sensors
        habitat_sim.registry.register_move_fn(
            LookUp, name="look_up", body_action=False
        )

        habitat_sim.registry.register_move_fn(
            LookDown, name="look_down", body_action=False
        )

        habitat_sim.registry.register_move_fn(
            MoveBackward, name="move_backward", body_action=True
        )

        habitat_sim.registry.register_move_fn(
            MoveLeft, name="move_left", body_action=True
        )

        habitat_sim.registry.register_move_fn(
            MoveRight, name="move_right", body_action=True
        )

        habitat_sim.registry.register_move_fn(
            TurnLeft, name="turn_left", body_action=True
        )

        habitat_sim.registry.register_move_fn(
            TurnRight, name="turn_right", body_action=True
        )

        habitat_sim.registry.register_move_fn(
            TiltLeft, name="tilt_left", body_action=False
        )

        habitat_sim.registry.register_move_fn(
            TiltRight, name="tilt_right", body_action=False
        )

        habitat_sim.registry.register_move_fn(
            MoveForward, name="move_forward", body_action=True
        )

        habitat_sim.registry.register_move_fn(
            MoveUp, name="move_up", body_action=True
        )

        habitat_sim.registry.register_move_fn(
            MoveDown, name="move_down", body_action=True
        )

        habitat_sim.registry.register_move_fn(
            LoiterLeft, name="loiter_left", body_action=True
        )

        habitat_sim.registry.register_move_fn(
            LoiterRight, name="loiter_right", body_action=True
        )

        # agent
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_sensor_spec, semantic_sensor_spec, depth_sensor_spec]
        agent_cfg.action_space = {
            "move_forward" : habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=self.translation_step)),
            "move_backward": habitat_sim.ActionSpec("move_backward",      habitat_sim.agent.ActuationSpec(amount=self.translation_step)),
            "move_left"    : habitat_sim.ActionSpec("move_left",          habitat_sim.agent.ActuationSpec(amount=self.translation_step)),
            "move_right"   : habitat_sim.ActionSpec("move_right",         habitat_sim.agent.ActuationSpec(amount=self.translation_step)),
            "look_down"    : habitat_sim.ActionSpec("look_down",          habitat_sim.agent.ActuationSpec(amount=self.rotation_step)),
            "look_up"      : habitat_sim.ActionSpec("look_up",            habitat_sim.agent.ActuationSpec(amount=self.rotation_step)),
            "turn_left"    : habitat_sim.ActionSpec("turn_left",          habitat_sim.agent.ActuationSpec(amount=self.rotation_step)),
            "turn_right"   : habitat_sim.ActionSpec("turn_right",         habitat_sim.agent.ActuationSpec(amount=self.rotation_step)),
            "move_up"      : habitat_sim.ActionSpec("move_up",            habitat_sim.agent.ActuationSpec(amount=self.translation_step)),
            "move_down"    : habitat_sim.ActionSpec("move_down",          habitat_sim.agent.ActuationSpec(amount=self.translation_step)),
            "loiter_left"  : habitat_sim.ActionSpec("loiter_left",       LoiterActuationSpec(amount = 1.0 , amount_rot=self.rotation_step, amount_trans=self.translation_step)),
            "loiter_right" : habitat_sim.ActionSpec("loiter_right",       LoiterActuationSpec(amount = 1.0 , amount_rot=self.rotation_step, amount_trans=self.translation_step)),
            "tilt_left"    : habitat_sim.ActionSpec("tilt_left",          habitat_sim.agent.ActuationSpec(amount=self.rotation_step)),
            "tilt_right"   : habitat_sim.ActionSpec("tilt_right",         habitat_sim.agent.ActuationSpec(amount=self.rotation_step)),
        }

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])
    
    def reverse_action(self, action):
        if action=='move_forward':
            return 'move_backward'

        elif action =='move_backward':
            return 'move_forward'

        elif action == 'move_up':
            return 'move_down'

        elif action == 'move_down':
            return 'move_up'

        elif action=='move_left':
            return 'move_right'

        elif action=='move_right':
            return 'move_left'

        elif action =='turn_right':
            return 'turn_left'

        elif action == 'turn_left':
            return 'turn_right'
        
        elif action == 'tilt_right':
            return 'tilt_left'
        
        elif action == 'tilt_left':
            return 'tilt_right'

        elif action == 'look_down':
            return 'look_up'

        elif action == 'look_up':
            return 'look_down'
        
        elif action == 'loiter_left':
            return 'loiter_right'
        
        elif action == 'loiter_right':
            return 'loiter_left'
        
    def get_camera_pose(self):
        position = self.sim.agents[0].get_state().sensor_states['color_sensor'].position
        rotation_quat = self.sim.agents[0].get_state().sensor_states['color_sensor'].rotation
        rotation = quaternion.as_float_array(rotation_quat)
        return position, rotation
        
    def save_images(self):

        rgb_img     = cv2.cvtColor(np.asarray(Image.fromarray(self.observations["color_sensor"], mode="RGBA")), cv2.COLOR_BGR2RGB)
        depth_image = (self.observations["depth_sensor"]*1000).astype(np.uint16)
        semantic_obs = self.observations["semantic_sensor"]
        # np.save(os.path.join(self.datadir, f"semantic_raw/{str(self.save_count).zfill(7)}.npy"), semantic_obs)
        semantic_obs = semantic_obs.astype(np.uint16)
        cv2.imwrite(os.path.join(self.datadir, f"semantic_raw/{str(self.save_count).zfill(7)}.png"), semantic_obs)

        sensor_pos, sensor_rot = self.get_camera_pose()
        self.camera_poses[str(self.save_count).zfill(7)] ={"position": sensor_pos.tolist(), "orientation": sensor_rot.tolist()}

        cv2.imwrite(os.path.join(self.datadir, f"color/{str(self.save_count).zfill(7)}.jpg"), rgb_img)
        cv2.imwrite(os.path.join(self.datadir, f"depth/{str(self.save_count).zfill(7)}.png"), depth_image)

    def semantic_obs_to_img(self, semantic_obs):
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(self.semantic_palette.flatten())
        semantic_img.putdata((semantic_obs.flatten()).astype(np.uint8))
        semantic_img = semantic_img.convert('RGB')
        semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_BGR2RGB)
        return semantic_img
    
    def spawn_random_object_infront_of_agent(self, dst=2.0):
        """
        Spawn a random object in front of the agent.
        """
        template_handles = self.obj_templates_mgr.get_template_handles()
        object_template_handle = template_handles[random.randint(0,len(template_handles))]

        agent_position = self.sim.agents[0].scene_node.translation

        agent_rotation = self.sim.agents[0].scene_node.rotation

        agent_forward = self.sim.agents[0].scene_node.rotation.transform_vector(
            habitat_sim.geo.FRONT
        )

        spawn_position = agent_position + agent_forward * dst + 0.5 * habitat_sim.geo.UP

        obj = self.rigid_obj_mgr.add_object_by_template_handle(object_template_handle)
        obj.translation = spawn_position

    def spawn_object_infront_of_agent_by_string(self, object_name, dst=2.0):
        """
        Spawn an object in front of the agent.
        """
        template_handles = self.obj_templates_mgr.get_template_handles()
        object_template_handle = os.path.join(self.object_config_dir, object_name + ".object_config.json")

        if not Path.exists(Path(object_template_handle)):
            print("invalid object name")
            return
        
        agent_position = self.sim.agents[0].get_state().sensor_states['color_sensor'].position
        agent_rotation = habitat_sim.utils.common.quat_to_magnum(self.sim.agents[0].get_state().sensor_states['color_sensor'].rotation)
        agent_forward = agent_rotation.transform_vector(
            habitat_sim.geo.FRONT
        )

        spawn_position = agent_position + agent_forward * dst #+ 0.5 * habitat_sim.geo.UP

        obj = self.rigid_obj_mgr.add_object_by_template_handle(object_template_handle)
        obj.translation = spawn_position

        obj.velocity_control.controlling_lin_vel = True
        obj.velocity_control.controlling_ang_vel = True
        obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
        # obj.velocity_control.lin_vel_is_local = True
        # obj.velocity_control.ang_vel_is_local = True
        
        self.object_list.append(obj)

    def remove_object(self, id):
        obj = self.object_list[id]
        self.rigid_obj_mgr.remove_object_by_handle(obj.handle)
        self.object_list.pop(id)

    def get_random_force(self, obj):
        force = (mn.Vector3([random.uniform(0, 10) for i in range(3)])+self.sim.get_gravity()*obj.mass)
        # object.apply_force(force, np.array([0,0,0]))
        return force

    def get_random_torque(self, obj):
        torque = mn.Vector3([random.uniform(0, 1) for i in range(3)])
        # object.apply_torque(torque)
        return torque

    def set_forces_torques(self):
        for obj in self.object_list:
            force = self.get_random_force(obj)
            torque = self.get_random_torque(obj)
            self.object_dynamics[obj.handle] = {"force": force, "torque": torque}

    def apply_forces_torques(self):
        for obj in self.object_list:
            obj.apply_force(self.object_dynamics[obj.handle]["force"], np.array([0.0,0.0,0.0]))
            obj.apply_torque(self.object_dynamics[obj.handle]["torque"])

    def set_agent_state_as_initial_state(self):
        """
        Set the agent's initial state.
        """
        state = self.sim.agents[0].get_state()

        self.agent_initial_state_dict[self.scene_name] = state

        with open(self.initial_state_dict_path, 'wb') as f:
            pickle.dump(self.agent_initial_state_dict, f)

    def raw_to_object_seg(self, semantic_obs, semantic_palette, labels):
        semantic_obs = np.where(semantic_obs < 1660, 0, semantic_obs)
            
        for i, value in enumerate(labels):
            value = int(value)
            if value == 0:
                self.color_map[value] = (0, 0, 0)
            else:
                self.color_map[value] = semantic_palette[i % len(semantic_palette)]
        semantic_img = np.zeros((semantic_obs.shape[0], semantic_obs.shape[1], 3))
        for val in self.color_map.keys():
            semantic_img[semantic_obs == val] = self.color_map[val]
        semantic_img = semantic_img.astype(np.uint8)
        return semantic_img

    def postprocess_semantic_obs(self):
        datadir = os.path.join(self.datadir, "semantic_raw")
        data_names = sorted(os.listdir(datadir))
        labels = []
        for img_name in data_names:
            img = cv2.imread(os.path.join(datadir, img_name), -1)
            img = np.where(img < 1000, 0, img)
            labels += [i for i in np.unique(img) if i not in labels]
        labels = np.array(sorted(labels))
        semantic_palette = self.generate_color_palette(np.unique(labels).shape[0]-1)

        for data_name in data_names:
            data_raw = cv2.imread(os.path.join(datadir, data_name), -1)
            semantics = self.raw_to_object_seg(data_raw, semantic_palette, labels)
            cv2.imwrite(os.path.join(self.datadir,"semantic", data_name.replace("npy", "png")), semantics)
    
    def generate_color_palette(self, num_colors):
        random.seed(42)
        hsv_tuples = [(x / num_colors, 1., 1.) for x in range(num_colors)]
        random.shuffle(hsv_tuples)
        random.seed(datetime.now().timestamp())
        rgb_tuples = map(lambda x: tuple(int(255 * i) for i in colorsys.hsv_to_rgb(*x)), hsv_tuples)
        bgr_tuples = map(lambda x: (x[2], x[1], x[0]), rgb_tuples)
        
        return list(bgr_tuples)
    
    def record_pose(self):
        position = self.sim.agents[0].get_state().position
        orientation = self.sim.agents[0].get_state().rotation

        camera_position = self.sim.agents[0].get_state().sensor_states['color_sensor'].position
        camera_orientation = self.sim.agents[0].get_state().sensor_states['color_sensor'].rotation

        self.agent_poses.append((position, orientation, camera_position, camera_orientation))

    def interpolate_poses(self, agent_poses, n):
        positions = []
        quaternions = []
        camera_positions = []
        camera_quaternions = []
        for pose in agent_poses:
            if positions and np.all(pose[0] == positions[-1]):
                continue
            positions.append(pose[0])
            quaternions.append(pose[1])
            camera_positions.append(pose[2])
            camera_quaternions.append(pose[3])
        # positions = [pose[0] for pose in agent_poses]
        # quaternions = [pose[1] for pose in agent_poses]
        # camera_positions = [pose[2] for pose in agent_poses]
        # camera_quaternions = [pose[3] for pose in agent_poses]
        positions_array = np.array(positions)
        camera_positions_array = np.array(camera_positions)
        try:
            tck_pos, u_pos = splprep(positions_array.T, s=0)
        except Exception as e:
            print("Trajectory could not be interpolated")
            return None
        tck_cam, u_cam = splprep(camera_positions_array.T, s=0)

        u_pos_new = np.linspace(u_pos.min(), u_pos.max(), n)
        u_cam_new = np.linspace(u_cam.min(), u_cam.max(), n)

        new_positions = np.array(splev(u_pos_new, tck_pos)).T
        new_positions = list(new_positions)

        new_camera_positions = np.array(splev(u_cam_new, tck_cam)).T
        new_camera_positions = list(new_camera_positions)

        t_in = np.linspace(0, 1, len(quaternions))
        t_out = np.linspace(0, 1, n)
        # Interpolate the quaternions using squad
        new_quaternions = quaternion.squad(np.array(quaternions), t_in, t_out, unflip_input_rotors=True)
        new_camera_quaternions = quaternion.squad(np.array(camera_quaternions), t_in, t_out, unflip_input_rotors=True)

        return list(zip(new_positions, new_quaternions, new_camera_positions, new_camera_quaternions))
    
    def save_trajectory(self, trajectory):
        trajectory_name = self.test_scene.split("/")[-1].split(".")[0]
        trajectory_info = {"scene": self.test_scene, "trajectory": []}
        if os.path.exists(os.path.join(self.trajectory_folder, trajectory_name + ".pickle")):
            with open(os.path.join(self.trajectory_folder, trajectory_name + ".pickle"), "rb") as f:
                trajectory_info = pickle.load(f)
        trajectory_info["trajectory"].append(trajectory)

        with open(os.path.join(self.trajectory_folder, trajectory_name + ".pickle"), "wb") as f:
            pickle.dump(trajectory_info, f)

        self.agent_poses = []
        print("trajectory saved.")
    
    def play_trajectory(self, trajectory):
        self.set_forces_torques()
        for state in trajectory:
            agent_state = habitat_sim.AgentState()
            agent_state.position = state[0]
            agent_state.rotation = state[1]
            sensor_rot_modified = state[3] * quaternion.from_rotation_vector(habitat_sim.geo.LEFT * np.pi / 25)
            agent_state.sensor_states['color_sensor'] = habitat_sim.SixDOFPose(state[2], sensor_rot_modified)
            agent_state.sensor_states['depth_sensor'] = habitat_sim.SixDOFPose(state[2], sensor_rot_modified)
            agent_state.sensor_states['semantic_sensor'] = habitat_sim.SixDOFPose(state[2], sensor_rot_modified)
            self.sim.agents[0].set_state(agent_state, infer_sensor_states=False)
            if self.dynamic_scene:
                self.apply_forces_torques()
            self.sim.step_physics(1.0/30.0)
            self.observations = self.sim.get_sensor_observations()

            if self.save:
                self.save_images()
                self.save_count += 1
            rgb_img = cv2.cvtColor(np.asarray(Image.fromarray(self.observations["color_sensor"], mode="RGBA")), cv2.COLOR_BGR2RGB)
            # cv2.imshow('RGB', rgb_img)
            # k = cv2.waitKey(1.0)
    
    def save_camera_poses(self):
        """
        Save the camera poses.
        """
        with open(os.path.join(self.datadir, "camera_poses.json"), 'w') as f:
            json.dump(self.camera_poses, f)
    
    def save_color_map(self):
        """
        Save the color map.
        """
        with open(os.path.join(self.datadir, "color_map.json"), 'w') as f:
            json.dump(self.color_map, f)          

@registry.register_move_fn(body_action=False)
class LookDown(SceneNodeControl):
    def __call__(
            self, scene_node: habitat_sim.SceneNode, actuation_spec:  habitat_sim.agent.ActuationSpec
            ) -> None:
        rotation_ax = habitat_sim.geo.LEFT
        scene_node.rotate_local(mn.Deg(actuation_spec.amount), rotation_ax)
        scene_node.rotation = scene_node.rotation.normalized()

@registry.register_move_fn(body_action=False)
class LookUp(SceneNodeControl):
    def __call__(
            self, scene_node: habitat_sim.SceneNode, actuation_spec:  habitat_sim.agent.ActuationSpec
            ) -> None:

        rotation_ax = habitat_sim.geo.RIGHT
        scene_node.rotate_local(mn.Deg(actuation_spec.amount), rotation_ax)
        scene_node.rotation = scene_node.rotation.normalized()

@registry.register_move_fn(body_action=True)
class MoveForward(SceneNodeControl):
    def __call__(
            self, scene_node: habitat_sim.SceneNode, actuation_spec: habitat_sim.agent.ActuationSpec
            ) -> None:
    
        ax = (
                np.array(scene_node.absolute_transformation().rotation_scaling())
                @ habitat_sim.geo.FRONT
            )
       
        scene_node.translate_local(ax * actuation_spec.amount)

@registry.register_move_fn(body_action=True)
class MoveUp(SceneNodeControl):
    def __call__(
            self, scene_node: habitat_sim.SceneNode, actuation_spec: habitat_sim.agent.ActuationSpec
            ) -> None:
    
        ax = (
                np.array(scene_node.absolute_transformation().rotation_scaling())
                @ habitat_sim.geo.UP
            )
       
        scene_node.translate_local(ax * actuation_spec.amount)

@registry.register_move_fn(body_action=True)
class MoveDown(SceneNodeControl):
    def __call__(
            self, scene_node: habitat_sim.SceneNode, actuation_spec: habitat_sim.agent.ActuationSpec
            ) -> None:
    
        ax = (
                np.array(scene_node.absolute_transformation().rotation_scaling())
                @ habitat_sim.geo.UP
            )
       
        scene_node.translate_local(-ax * actuation_spec.amount)

@registry.register_move_fn(body_action=True)
class MoveBackward(SceneNodeControl):
    def __call__(
            self, scene_node: habitat_sim.SceneNode, actuation_spec: habitat_sim.agent.ActuationSpec
            ) -> None:
    
        ax = (
                np.array(scene_node.absolute_transformation().rotation_scaling())
                @ habitat_sim.geo.FRONT
            )
        
        scene_node.translate_local(-ax * actuation_spec.amount)

@registry.register_move_fn(body_action=True)
class MoveLeft(SceneNodeControl):
    def __call__(
            self, scene_node: habitat_sim.SceneNode, actuation_spec: habitat_sim.agent.ActuationSpec
            ) -> None:

        ax = (
                np.array(scene_node.absolute_transformation().rotation_scaling())
                @ habitat_sim.geo.LEFT
            )
    
        scene_node.translate_local(ax * actuation_spec.amount)

@registry.register_move_fn(body_action=True)
class MoveRight(SceneNodeControl):
    def __call__(
            self, scene_node: habitat_sim.SceneNode, actuation_spec: habitat_sim.agent.ActuationSpec
            ) -> None:

        ax = (
                np.array(scene_node.absolute_transformation().rotation_scaling())
                @ habitat_sim.geo.RIGHT
            )
      
        scene_node.translate_local(ax * actuation_spec.amount)

@registry.register_move_fn(body_action=True)
class TurnLeft(SceneNodeControl):
    def __call__(
            self, scene_node: habitat_sim.SceneNode, actuation_spec: habitat_sim.agent.ActuationSpec
            ) -> None:
        
        ax = (
                np.array(scene_node.absolute_transformation().rotation_scaling())
                @ habitat_sim.geo.UP
            )
        scene_node.rotate_local(mn.Deg(actuation_spec.amount), ax)
        scene_node.rotation = scene_node.rotation.normalized()

@registry.register_move_fn(body_action=True)
class TurnRight(SceneNodeControl):
    def __call__(
            self, scene_node: habitat_sim.SceneNode, actuation_spec: habitat_sim.agent.ActuationSpec
            ) -> None:
        
        ax = (
                np.array(scene_node.absolute_transformation().rotation_scaling())
                @ habitat_sim.geo.UP
            )
        scene_node.rotate_local(mn.Deg(-actuation_spec.amount), ax)
        scene_node.rotation = scene_node.rotation.normalized()

@registry.register_move_fn(body_action=True)
class TiltLeft(SceneNodeControl):
    def __call__(
            self, scene_node: habitat_sim.SceneNode, actuation_spec: habitat_sim.agent.ActuationSpec
            ) -> None:
        
        ax = (
                np.array(scene_node.absolute_transformation().rotation_scaling())
                @ habitat_sim.geo.FRONT
            )
        scene_node.rotate_local(mn.Deg(actuation_spec.amount), ax)
        scene_node.rotation = scene_node.rotation.normalized()

@registry.register_move_fn(body_action=True)
class TiltRight(SceneNodeControl):
    def __call__(
            self, scene_node: habitat_sim.SceneNode, actuation_spec: habitat_sim.agent.ActuationSpec
            ) -> None:
        
        ax = (
                np.array(scene_node.absolute_transformation().rotation_scaling())
                @ habitat_sim.geo.FRONT
            )
        scene_node.rotate_local(mn.Deg(-actuation_spec.amount), ax)
        scene_node.rotation = scene_node.rotation.normalized()


@attr.s(auto_attribs=True, slots=True)
class LoiterActuationSpec(habitat_sim.agent.ActuationSpec):
    amount_rot: int = 2
    amount_trans: float = 0.1

@registry.register_move_fn(body_action=True)
class LoiterRight(SceneNodeControl):
    def __call__(
            self, scene_node: habitat_sim.SceneNode, acutation_spec: LoiterActuationSpec
            ) -> None:
        
        rotation_ax = (
                np.array(scene_node.absolute_transformation().rotation_scaling())
                @ habitat_sim.geo.UP
            )
        
        translation_ax = (
                np.array(scene_node.absolute_transformation().rotation_scaling())
                @ habitat_sim.geo.RIGHT
            )
        
        scene_node.rotate_local(mn.Deg(acutation_spec.amount_rot * acutation_spec.amount), rotation_ax)
        scene_node.translate_local(translation_ax * acutation_spec.amount_trans * acutation_spec.amount)

@registry.register_move_fn(body_action=True)
class LoiterLeft(SceneNodeControl):
    def __call__(
            self, scene_node: habitat_sim.SceneNode, acutation_spec: LoiterActuationSpec
            ) -> None:
        
        rotation_ax = (
                np.array(scene_node.absolute_transformation().rotation_scaling())
                @ habitat_sim.geo.UP
            )
        
        translation_ax = (
                np.array(scene_node.absolute_transformation().rotation_scaling())
                @ habitat_sim.geo.RIGHT
            )
        
        scene_node.rotate_local(mn.Deg(-acutation_spec.amount_rot * acutation_spec.amount), rotation_ax)
        scene_node.translate_local(-translation_ax * acutation_spec.amount_trans * acutation_spec.amount)




def main():

    rec = DataRecorder()
    rec.create_folder_structure()
    rec.initialize_habitat_sim()

    cv2.namedWindow('Semantics' , cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('RGB', cv2.WINDOW_AUTOSIZE)

    cv2.moveWindow('RGB', 0, 0)
    cv2.moveWindow('Semantics', 1280, 0)

    while True:

        key = cv2.waitKey(0)
        rec(key)

        rgb_img = cv2.cvtColor(np.asarray(Image.fromarray(rec.observations["color_sensor"], mode="RGBA")), cv2.COLOR_BGR2RGB)
        semantic_img = rec.semantic_obs_to_img(rec.observations["semantic_sensor"])

        cv2.imshow('RGB', rgb_img)
        cv2.imshow('Semantics', semantic_img)

        if key == 27: # ESC
            break
    
    print("postprocessing...")
    cv2.destroyAllWindows()
    rec.save_camera_poses()
    rec.postprocess_semantic_obs()
    rec.save_color_map()

    pr = input("generate prompts (y/n)?")

    if pr == "y":
        datareader = HabitatSceneDataReader(rec.datadir)

        image_names = [n.split('.')[0] for n in sorted(os.listdir(datareader.rgb_path))][::20]

        for image in image_names:
            datareader(image)

        datareader.save_prompts()

        cv2.destroyWindow('color')
        # cv2.destroyWindow('seg')


if __name__ == "__main__":
    main()