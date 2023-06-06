import numpy as np
import os
import json
import cv2
import argparse


class HabitatSceneDataReader():
    def __init__(self, datadir) -> None:
        self.rgb_path = os.path.join(datadir, "color")
        self.semantic_path = os.path.join(datadir, "semantic_raw")
        self.out_path_multi = os.path.join(datadir, "prompts_multi.json")
        self.out_path_single = os.path.join(datadir, "prompts_single.json")
        self.object_path = "/home/nico/semesterproject/habitat-sim/data/replica_cad/configs/objects"
        self.prompt_dict_multi = {}
        self.prompt_dict_single = {}

        self.object_map = {}
        objects = [i for i in sorted(os.listdir(self.object_path)) if i.endswith(".json")]

        for object in objects:
            with open(os.path.join(self.object_path, object), 'r') as f:
                data = json.load(f)
                self.object_map.update({data["semantic_id"]: object.split(".")[0]})

        self.classes = set()
        for img in os.listdir(self.semantic_path):
            semantic_annot = np.load(os.path.join(self.semantic_path, img))
            self.classes = self.classes.union(set([i for i in np.unique(semantic_annot).tolist() if i >= 1000]))

        self.initial_coordinates = None

        cv2.namedWindow("color")
        cv2.setMouseCallback("color",self.select)

    def __call__(self, image_name):

        img, all_semantic_annotations = self.load_image(image_name)
        cv2.imshow("color", img)

        print("select objects to track, press ENTER to confirm, press ESC to quit")
        for i in self.classes:
            single_prompt = False
            print("selecting class: ", i, "; name: ", self.object_map[i])
            flag = False
            while True:
                key = cv2.waitKey(0)
                if key == 27: # ESC
                    cv2.destroyAllWindows()
                    exit()
                if key == 110: # n
                    print("skipping object")
                    flag = True
                    break
                if key == 13: # ENTER
                    if self.initial_coordinates is None:
                        print ("please select an object to track before pressing ENTER")
                        continue
                    else:
                        print("confirmed, initial coordinates: ", self.initial_coordinates)
                        break
                if key == 32: # SPACE
                    print("selecting single prompt")
                    if self.initial_coordinates is None:
                        print ("please select an object to track before pressing ENTER")
                        continue
                    else:
                        print("confirmed, initial coordinates: ", self.initial_coordinates)
                        single_prompt = True
                        break
            if flag:
                continue

            selected_class_id = all_semantic_annotations[self.initial_coordinates[1], self.initial_coordinates[0]]
            if selected_class_id == i:
                print("selected class ID: ", selected_class_id)
            else:
                print(f"wrong object selected. Switching to class {selected_class_id}")
            selected_class_mask = (all_semantic_annotations == selected_class_id)
            selected_class_id = int(selected_class_id)

            bbox = self.get_bbox_from_mask(selected_class_mask)
            if self.prompt_dict_multi.get(image_name) == None:
                self.prompt_dict_multi[image_name] = {}
            self.prompt_dict_multi[image_name].update({selected_class_id: {"point_prompt": self.initial_coordinates, "bbox": bbox}}) # add bounding box from mask

            if single_prompt:
                if self.prompt_dict_single.get(image_name) == None:
                    self.prompt_dict_single[image_name] = {}
                self.prompt_dict_single[image_name].update({selected_class_id: {"point_prompt": self.initial_coordinates, "bbox": bbox}}) # add bounding box from mask


        if flag:
            return
        
        cv2.imshow("seg", self.show_mask(selected_class_mask, random_color=False))
        cv2.waitKey(int(1000/30))

        self.initial_coordinates = None


    def load_image(self, image_name):
    
        img = cv2.imread(os.path.join(self.rgb_path, image_name + ".jpg"))
        all_semantic_annotations = np.load(os.path.join(self.semantic_path, image_name + ".npy"))

        return img, all_semantic_annotations

    def select(self, event, x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x,y)
            self.initial_coordinates = (x,y)


    def show_mask(self, mask: np.ndarray, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([0, 0, 255, 1.0])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        
        return mask_image


    def get_mask(self, image):
        img, semantic_annot = self.load_image(image)

        mask = (semantic_annot == self.selected_class_id)

        show_mask = self.show_mask(mask, random_color=False)

        return img, show_mask


    def get_prompt(self):
        return self.initial_coordinates
    

    def get_bbox_from_mask(self, mask: np.ndarray):
        mask = mask.squeeze().astype(np.uint8)
        if np.sum(mask) == 0:
            return None
        
        row_indices, col_indices = np.where(mask)

        # Calculate the min and max row and column indices
        row_min, row_max = np.min(row_indices), np.max(row_indices)
        col_min, col_max = np.min(col_indices), np.max(col_indices)

        row_min, row_max, col_min, col_max = int(row_min), int(row_max), int(col_min), int(col_max)

        # Return the bounding box coordinates as a tuple
        return (col_min, row_min, col_max, row_max)
    
    def save_prompts(self):
        with open(self.out_path_multi, 'w') as f:
            json.dump(self.prompt_dict_multi, f)

        with open(self.out_path_single, 'w') as f:
            json.dump(self.prompt_dict_single, f)


def main(datadir):

    datareader = HabitatSceneDataReader(datadir)

    image_names = [n.split('.')[0] for n in sorted(os.listdir(datareader.rgb_path))][::20]

    for image in image_names:
        datareader(image)

    datareader.save_prompts()


if __name__== "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '-d', '--datadir', default='/home/nico/semesterproject/data/re-id_benchmark_ycb/single_object/toys/train/toys_on_ground', help='path to dataset'
        )

    args = argparser.parse_args()

    main(args.datadir)