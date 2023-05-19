import cv2
import os
import numpy as np
from PIL import Image
import random
import colorsys


def generate_color_palette(num_colors):
    random.seed(123)  # to get the same colors every time
    hsv_tuples = [(x / num_colors, 1., 1.) for x in range(num_colors)]
    random.shuffle(hsv_tuples)  # to decorrelate neighboring classes
    rgb_tuples = map(lambda x: tuple(int(255 * i) for i in colorsys.hsv_to_rgb(*x)), hsv_tuples)
    bgr_tuples = map(lambda x: (x[2], x[1], x[0]), rgb_tuples)
    return list(bgr_tuples)

def semantic_obs_to_img(semantic_obs, semantic_palette, labels):
        semantic_obs = np.where(semantic_obs < 1000, 0, semantic_obs)
        color_map = {}
        for i, value in enumerate(labels):
            if value == 0:
                color_map[value] = (0, 0, 0)
            else:
                color_map[value] = semantic_palette[i % len(semantic_palette)]
        semantic_img = np.zeros((semantic_obs.shape[0], semantic_obs.shape[1], 3))
        for val in color_map.keys():
            semantic_img[semantic_obs == val] = color_map[val]
        semantic_img = semantic_img.astype(np.uint8)
        return semantic_img


if __name__ == "__main__":
    datadir = "/home/nico/semesterproject/data/re-id_benchmark/single_object/test/plastic_drum_in_scene/semantic_raw"
    outdir = "/home/nico/semesterproject/data/re-id_benchmark/single_object/test/plastic_drum_in_scene/semantic"
    n_objects = 1
    data_names = sorted(os.listdir(datadir))
    first_data = np.load(os.path.join(datadir, data_names[0]))
    first_data = np.where(first_data < 1000, 0, first_data)
    labels = np.unique(first_data)
    semantic_palette = generate_color_palette(1)

    for data_name in data_names:
        data_raw = np.load(os.path.join(datadir, data_name))
        labels = np.unique(data_raw)
        semantics = semantic_obs_to_img(data_raw, semantic_palette, labels)

        cv2.imshow("test", semantics)
        cv2.imwrite(os.path.join(outdir, data_name.replace("npy", "jpg")), semantics)
        cv2.waitKey(int(1000/60))

# if __name__ == "__main__":
#     datadir = "/home/nico/semesterproject/data/re-id_benchmark/single_object/test/smeg_kettle_in_context/semantic_raw"
#     outdir = "/home/nico/semesterproject/data/re-id_benchmark/single_object/test/smeg_kettle_in_context/semantic"
#     data_names = sorted(os.listdir(datadir))
#     first_data = np.load(os.path.join(datadir, data_names[0]))
#     first_data = np.where(first_data < 1000, 0, first_data)
#     labels = np.unique(first_data)
#     semantic_palette = generate_color_palette(np.unique(first_data).shape[0]-1)

#     for data_name in data_names:
#         data_raw = np.load(os.path.join(datadir, data_name))
#         semantics = semantic_obs_to_img(data_raw, semantic_palette, labels)

#         cv2.imshow("test", semantics)
#         cv2.imwrite(os.path.join(outdir, data_name.replace("npy", "jpg")), semantics)
#         cv2.waitKey(int(1000/60))

        