import numpy as np
import os
import json
import cv2
import argparse


class HabitatSceneDataReader():
    def __init__(self, datadir) -> None:
        self.rgb_path = os.path.join(datadir, "color")
        self.semantic_path = os.path.join(datadir, "semantic_raw")
        self.out_path = os.path.join(datadir, "prompts.json")
        self.prompt_dict = {}
        # structure of prompt_dict needs to account for multiple objects

        self.initial_coordinates = None

        cv2.namedWindow("RGB")
        cv2.setMouseCallback("RGB",self.select)

    def __call__(self, image_name):

        img, all_semantic_annotations = self.load_image(image_name)
        cv2.imshow("RGB", img)

        print("select object to track, press ENTER to confirm, press ESC to quit")
        while True:
            key = cv2.waitKey(0)
            if key == 27: # ESC
                cv2.destroyAllWindows()
                exit()
            if key == 110: # n
                print("skipping image")
                return
            if key == 13: # ENTER
                if self.initial_coordinates is None:
                    print ("please select an object to track before pressing ENTER")
                    continue
                else:
                    print("confirmed")
                    print("initial coordinates: ", self.initial_coordinates)
                    break

        selected_class_id = all_semantic_annotations[self.initial_coordinates[1], self.initial_coordinates[0]]
        print("selected class ID: ", selected_class_id)
        selected_class_mask = (all_semantic_annotations == selected_class_id)
        selected_class_id = int(selected_class_id)

        bbox = self.get_bbox_from_mask(selected_class_mask)
        self.prompt_dict[image_name] = {selected_class_id: {"point_prompt": self.initial_coordinates, "bbox": bbox}} # add bounding box from mask

        cv2.imshow("Semantics", self.show_mask(selected_class_mask, random_color=False))
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
        img = cv2.imread(os.path.join(self.rgb_path, image + ".jpg"))
        semantic_annot = np.load(os.path.join(self.semantic_path, image + ".npy"))

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
        with open(self.out_path, 'w') as f:
            json.dump(self.prompt_dict, f)


def main(datadir):

    datareader = HabitatSceneDataReader(datadir)

    image_names = [n.split('.')[0] for n in sorted(os.listdir(datareader.rgb_path))][::10]

    for image in image_names:
        datareader(image)

    datareader.save_prompts()


if __name__== "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '-d', '--datadir', default='/home/nico/semesterproject/data/re-id_benchmark/single_object/trashcan/train/trashcan_loiter', help='path to dataset'
        )

    args = argparser.parse_args()

    main(args.datadir)