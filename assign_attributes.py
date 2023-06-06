from typing import Any
import numpy as np
import json
import os
import cv2
from scipy.spatial.distance import cosine
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class AttributeWriter():
    def __init__(self):

        self.clutter_threshold = 6
        self.small_object_threshold = 0.02
        self.small_object_in_individual_frame_threshold = 0.01
        self.total_spatial_distance_threshold = 20
        self.total_rotation_distance_threshold = 8
        self.linear_velocity_threshold = 1.5
        self.angular_velocity_threshold = 1.0


        self.default_attributes = {
            "DYN": False,
            "CLT": False,
            "CLA": False,
            "SML": False,
            "SMF": False,
            "FST": False,
        }

        self.manual_attributes = ["DYN","CLA"]

        self.attribute_descriptions = {
            "DYN": "Dynamic scene",
            "CLT": "Cluttered scene",
            "CLA": "Equal class objects",
            "SML": "Small objects",
            "SMF": "Object small in individual frames",
            "FST": "Camera is moving fast"
        }

        self.datadir = "/home/nico/semesterproject/data/re-id_benchmark_ycb"

    def __call__(self):
        for setting in sorted(os.listdir(self.datadir)):
            print("Setting: {}".format(setting))
            for task in sorted(os.listdir(os.path.join(self.datadir, setting))):
                print("Task: {}".format(task))
                for train_test in ["train", "test"]:
                    for sequence in sorted(os.listdir(os.path.join(self.datadir, setting, task, train_test))):
                        seq_dir = os.path.join(self.datadir, setting, task, train_test, sequence)
                        # success = False
                        success = True
                        while not success:
                            img_dir = os.path.join(self.datadir, setting, task, train_test, sequence, "color")
                            for img_name in sorted(os.listdir(img_dir)):
                                img = cv2.imread(os.path.join(img_dir, img_name))
                                cv2.imshow("img", img)
                                key = cv2.waitKey(int(1000/30))
                                if key == 27: # ESC
                                    success = True
                                    break
                                elif key == 114: # r
                                    print("replay")
                                    break
                            else:
                                success = True
                                continue

                        cv2.destroyAllWindows()
                        
                        if os.path.exists(os.path.join(self.datadir, setting, task, train_test, sequence, "attributes.json")):
                            scene_attributes = json.load(open(os.path.join(self.datadir, setting, task, train_test, sequence, "attributes.json")))
                        else:
                            scene_attributes = self.default_attributes
                            for attribute in self.manual_attributes:
                                scene_attributes[attribute] = self.evaluate_manual(attribute)

                        scene_attributes["FST"] = self.evaluate_fast(seq_dir)
                        scene_attributes["CLT"] = self.evaluate_clutter(seq_dir)
                        scene_attributes["SML"], scene_attributes["SMF"] = self.evaluate_small_objects(seq_dir)
                        

                        with open(os.path.join(self.datadir, setting, task, train_test, sequence, "attributes.json"), "w") as f:
                            json.dump(scene_attributes, f, indent=4)

    def evaluate_fast(self, seq_dir):
        def quaternion_distance(q1, q2):
            q1 /= np.linalg.norm(q1)
            q2 /= np.linalg.norm(q2)
            dot_product = np.clip(np.dot(q1, q2), -1.0, 1.0)
            angle = np.arccos(dot_product)
            return angle
        
        with open(os.path.join(seq_dir, "camera_poses.json")) as f:
            camera_poses = json.load(f)
        linear_velocities = []
        angular_velocities = []
        total_spatial_distance = 0
        total_rotation_distance = 0
        position = list(camera_poses.values())[0]["position"]
        orientation = list(camera_poses.values())[0]["orientation"]
        for img_name, pose in camera_poses.items():
            new_position = pose["position"]
            dist_lin = np.linalg.norm(np.array(new_position) - np.array(position))
            total_spatial_distance += dist_lin
            linear_velocities.append(dist_lin / (1/30))

            new_orientation = pose["orientation"]
            dist_ang = quaternion_distance(np.array(new_orientation), np.array(orientation))
            total_rotation_distance += dist_ang
            angular_velocities.append(dist_ang / (1/30))

            position = new_position
            orientation = new_orientation

        # print("Total spatial distance: {}".format(total_spatial_distance))
        # print("Total rotation distance: {}".format(total_rotation_distance))
        max_lin_velocity = max(linear_velocities)
        max_ang_velocity = max(angular_velocities)
        print("Max linear velocity: {}".format(max_lin_velocity))
        print("Max angular velocity: {}".format(max_ang_velocity))
        if max_lin_velocity > self.linear_velocity_threshold or max_ang_velocity > self.angular_velocity_threshold:
            return True
        else:
            return False
        
    def evaluate_manual(self, attribute):
        print("Attribute: {}".format(attribute))
        print("Description: {}".format(self.attribute_descriptions[attribute]))
        successful = False
        while not successful:
            att_str = input("y/n: ")
            if att_str == "y":
                att_bool = True
                successful = True
            elif att_str == "n":
                att_bool = False
                successful = True
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

        return att_bool
    
    def evaluate_clutter(self, seq_dir):
        if not os.path.exists(os.path.join(seq_dir, "color_map.json")):
            return 
        with open(os.path.join(seq_dir, "color_map.json")) as f:
            color_map = json.load(f)

        if len(list(color_map.keys())) - 1 >= self.clutter_threshold:
            return True
        else:
            return False
        
    def evaluate_small_objects(self, seq_dir: str):

        small_objects_bool = False
        small_objects_individual_frames_bool = False

        task_dir = seq_dir.split("/")[:-2]
        task_dir = "/".join(task_dir)
        with open(os.path.join(task_dir, "info.json")) as f:
            task_info = json.load(f)
        tracked_semantic_ids = task_info["semantic_ids"]
        areas = {}
        for id in tracked_semantic_ids:
            areas[id] = []
        for img_name in sorted(os.listdir(os.path.join(seq_dir, "semantic_raw"))):
            sem = np.load(os.path.join(seq_dir, "semantic_raw", img_name))
            for id in tracked_semantic_ids:
                bbox = self.get_bbox_from_mask(sem == id)
                if bbox is None:
                    continue
                if bbox[0] <= 1 or bbox[1] <= 1 or bbox[2] >= sem.shape[1]-1 or bbox[3] >= sem.shape[0]-1:
                    continue
                bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                img_area = sem.shape[0] * sem.shape[1]
                areas[id].append(bbox_area / img_area)
        
        for id in tracked_semantic_ids:
            if np.mean(areas[id]) < self.small_object_threshold:
                small_objects_bool = True
            
            if np.min(areas[id]) < self.small_object_in_individual_frame_threshold:
                small_objects_individual_frames_bool = True
        
        return small_objects_bool, small_objects_individual_frames_bool

    def get_bbox_from_mask(self, mask: np.ndarray):
        mask = mask.squeeze().astype(np.uint8)
        if np.sum(mask) == 0:
            return None
        
        row_indices, col_indices = np.where(mask)

        # Calculate the min and max row and column indices
        row_min, row_max = np.min(row_indices), np.max(row_indices)
        col_min, col_max = np.min(col_indices), np.max(col_indices)

        # Return the bounding box coordinates as a tuple
        return (col_min, row_min, col_max, row_max)



if __name__ == "__main__":
    writer = AttributeWriter()
    writer()