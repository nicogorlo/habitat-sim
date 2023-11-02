import os
import json
import numpy as np
import cv2

from tqdm import tqdm

# if __name__ == "__main__":
    
#     number_of_objects = []
#     number_of_scenes = []
#     for test_case in os.listdir(datadir):
#         test_case_path = os.path.join(datadir,test_case)
#         with open(os.path.join(test_case_path, "info.json")) as f:
#             info = json.load(f)

#         scene_count=1
#         for scene in os.listdir(os.path.join(test_case_path, "test")):
#             scene_count += 1

#         number_of_scenes.append(scene_count)

#         number_of_objects.append(len(info["semantic_ids"]))

#     print("number of scenes: ", number_of_scenes)
#     print("number of objects: ",number_of_objects)

#     print("total number of scenes: ", sum(number_of_scenes))
#     print("number of annotations: ", sum([number_of_objects[i]*(number_of_scenes[i]-1)*400 for i in range(len(number_of_objects))]))
#     print("number of annotations per sequence: ", np.mean(number_of_objects))

if __name__ == "__main__":
    datadir = "/home/nico/semesterproject/data/re_id_benchmark_ycb/multi_object"
    number_of_annotations = []
    for test_case in tqdm(sorted(os.listdir(datadir))):
        test_case_path = os.path.join(datadir,test_case)

        #test scenes
        for scene in os.listdir(os.path.join(test_case_path, "test")):
            scene_dir = os.path.join(test_case_path, "test", scene)
            number_of_annotations_scene = []
            img_names = sorted(os.listdir(os.path.join(scene_dir, "semantic_raw")))
            for img_name in img_names:
                if img_name.endswith("png"):
                    img = cv2.imread(os.path.join(scene_dir, "semantic_raw", img_name), -1)
                    number_of_annotations_scene.append(len([i for i in np.unique(img) if i >= 1100]))
                elif img_name.endswith("npy") and img_name.replace("npy", "png") not in img_names:
                    img = np.load(os.path.join(scene_dir, "semantic_raw", img_name))
                    number_of_annotations_scene.append(len([i for i in np.unique(img) if i >= 1100]))

            number_of_annotations.append(np.sum(number_of_annotations_scene))

        for scene in os.listdir(os.path.join(test_case_path, "train")):
            scene_dir = os.path.join(test_case_path, "train", scene)
            number_of_annotations_scene = []
            img_names = sorted(os.listdir(os.path.join(scene_dir, "semantic_raw")))
            for img_name in img_names:
                if img_name.endswith("png"):
                    img = cv2.imread(os.path.join(scene_dir, "semantic_raw", img_name), -1)
                    number_of_annotations_scene.append(len([i for i in np.unique(img) if i >= 1100]))
                elif img_name.endswith("npy") and img_name.replace("npy", "png") not in img_names:
                    img = np.load(os.path.join(scene_dir, "semantic_raw", img_name))
                    number_of_annotations_scene.append(len([i for i in np.unique(img) if i >= 1100]))

            number_of_annotations.append(np.sum(number_of_annotations_scene))

    print(number_of_annotations)
    print(np.sum(number_of_annotations))