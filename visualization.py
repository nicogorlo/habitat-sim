import cv2
import os
import numpy as np
import json
import random
import colorsys

def generate_video(data_path, video_name, fps):
    image_folder = os.path.join(data_path, "color")
    semantic_folder = os.path.join(data_path, "semantic")
    if os.path.exists(os.path.join(data_path, "prompts_multi.json")):
        with open(os.path.join(data_path, "prompts_multi.json")) as f:
            prompts = json.load(f)
    else:
        prompts = None
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
    
    # Read the first image to get the size
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Define the codec and create a VideoWriter object
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        img = cv2.imread(os.path.join(image_folder, image))

        if prompts is not None:
            if image.split('.')[0] in list(prompts.keys()):
                for i in prompts[image.split('.')[0]]:
                    cv2.circle(img, (prompts[image.split('.')[0]][i]["point_prompt"][0], prompts[image.split('.')[0]][i]["point_prompt"][1]), 6, (0, 0, 255), -1)
                    cv2.rectangle(  
                                img, 
                                (prompts[image.split('.')[0]][i]["bbox"][0], prompts[image.split('.')[0]][i]["bbox"][1]), 
                                (prompts[image.split('.')[0]][i]["bbox"][2], prompts[image.split('.')[0]][i]["bbox"][3]), 
                                (0, 255, 0), 
                                3)
                for i in range(15):
                    video.write(img)
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

    
# from moviepy.editor import VideoFileClip, AudioFileClip

# def combine_video_audio(video_path, audio_path, output_path):
#     video = VideoFileClip(video_path)
#     audio = AudioFileClip(audio_path)

#     # Cut the audio to match the duration of the video
#     audio = audio.subclip(0, video.duration)

#     # Set the audio of the video clip
#     video = video.set_audio(audio)

#     # Save the combined video with audio
#     video.write_videofile(output_path, codec='libx264', audio_codec='aac')

#     # Close the video and audio clips
#     video.close()
#     audio.close()

def visualize_annotated_frames(data_path, outpath):
    image_folder = os.path.join(data_path, "color")
    semantic_folder = os.path.join(data_path, "semantic")
    if os.path.exists(os.path.join(data_path, "prompts_multi.json")):
        with open(os.path.join(data_path, "prompts_multi.json")) as f:
            prompts = json.load(f)
    else:
        prompts = None
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])

    # Define the codec and create a VideoWriter object
    if not os.path.exists(os.path.join(outpath, data_path.split('/')[-1])):
        os.makedirs(os.path.join(outpath, data_path.split('/')[-1]))

    for image in images:
        img = cv2.imread(os.path.join(image_folder, image))

        if prompts is not None:
            if image.split('.')[0] in list(prompts.keys()):
                for i in prompts[image.split('.')[0]]:
                    cv2.circle(img, (prompts[image.split('.')[0]][i]["point_prompt"][0], prompts[image.split('.')[0]][i]["point_prompt"][1]), 5, (0, 0, 255), -1)
                    cv2.rectangle(img, (prompts[image.split('.')[0]][i]["bbox"][0], prompts[image.split('.')[0]][i]["bbox"][1]), (prompts[image.split('.')[0]][i]["bbox"][2], prompts[image.split('.')[0]][i]["bbox"][3]), (0, 255, 0), 2)

        cv2.imwrite(os.path.join(outpath, data_path.split('/')[-1], image), img)
    cv2.destroyAllWindows()


def visualize_gt_semantic_frames(data_path, outpath, video_name, fps = 30):
    image_folder = os.path.join(data_path, "color")
    semantic_folder = os.path.join(data_path, "semantic")
    semantic_raw_folder = os.path.join(data_path, "semantic_raw")

    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    if not os.path.exists(os.path.join(outpath, data_path.split('/')[-1])):
        os.makedirs(os.path.join(outpath, data_path.split('/')[-1]))

    with open(os.path.join("/".join(data_path.split("/")[:-2]), "info.json")) as f:
        semantic_ids = json.load(f)["semantic_ids"]
    
    with open(os.path.join("/".join(data_path.split("/")[:-2]), "info.json")) as f:
        color_map = json.load(f)["color_map"]

    for image in images:
        img = cv2.imread(os.path.join(image_folder, image))
        sem_raw = np.load(os.path.join(semantic_raw_folder, image.replace(".jpg", ".npy")))
        sem = raw_to_object_seg(sem_raw, semantic_ids, color_map)
        contours = find_contours(img, sem_raw, semantic_ids)
        semantic_contoured = cv2.addWeighted(sem , 0.5, contours, 1.0, 0)
        combined = cv2.addWeighted(img, 0.5, semantic_contoured, 1.0, 30)

        video.write(combined)

        # cv2.imshow("combined", combined)
        # cv2.waitKey(int(1000/30))
        cv2.imwrite(os.path.join(outpath, data_path.split('/')[-1], image), combined)
    cv2.destroyAllWindows()
    video.release()

def combine_modalities_debug(data_path, semantic_path, outpath):
    # image_folder = os.path.join(data_path, "color")
    image_folder = data_path
    semantic_folder = semantic_path

    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])

    if not os.path.exists(os.path.join(outpath, data_path.split('/')[-1])):
        os.makedirs(os.path.join(outpath, data_path.split('/')[-1]))

    with open(os.path.join("/".join(data_path.split("/")[:-2]), "info.json")) as f:
        semantic_ids = json.load(f)["semantic_ids"]
    
    with open(os.path.join("/".join(data_path.split("/")[:-2]), "info.json")) as f:
        color_map = json.load(f)["color_map"]

    for image in images:
        img = cv2.imread(os.path.join(image_folder, image))
        sem_raw = cv2.imread(os.path.join(semantic_folder, image.replace(".jpg", ".png")), -1)
        sem = raw_to_object_seg(sem_raw, semantic_ids, color_map)
        contours = find_contours(img, sem_raw, semantic_ids)
        semantic_contoured = cv2.addWeighted(sem , 0.5, contours, 1.0, 0)
        combined = cv2.addWeighted(img, 0.5, semantic_contoured, 1.0, 30)

        # cv2.imshow("combined", combined)
        # cv2.waitKey(int(1000/30))
        cv2.imwrite(os.path.join(outpath, data_path.split('/')[-1], image), combined)
    cv2.destroyAllWindows()

def combine_modalities_davis(data_path, semantic_path, outfile, fps = 30):
    image_folder = os.path.join(data_path, "color")
    image_folder = data_path
    semantic_folder = semantic_path

    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])

    img0 = cv2.imread(os.path.join(image_folder, images[0]))

    height, width, layers = img0.shape

    # Define the codec and create a VideoWriter object
    video = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    video_sem = cv2.VideoWriter(outfile.replace(".mp4", "_sem.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for idx, image in enumerate(images):
        img = cv2.imread(os.path.join(image_folder, image))
        sem = cv2.imread(os.path.join(semantic_folder, image.replace(".jpg", ".png")))
        contours, _ = cv2.findContours(cv2.cvtColor(sem, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = np.zeros_like(img)
        for contour in contours:
            for point in contour:
                x, y = point[0]

                color = sem[y, x].tolist()
                darker_color = darken_color(color)
                contour_img[y, x] = darker_color
        semantic_contoured = cv2.addWeighted(sem , 1.0, contour_img, 1.0, 0)
        combined = cv2.addWeighted(img, 0.5, semantic_contoured, 0.5, 30)

        # cv2.imshow("combined", combined)
        # cv2.waitKey(int(1000/30))
        # if idx == 0:
        #     for i in range(36):
        #         video.write(combined)
        video_sem.write(combined)
        video.write(img)
    cv2.destroyAllWindows()
    video.release()


def mot_demo(data_path, semantic_path, outfile, fps = 30):
    def create_binary_masks(mask_path):
        mask = cv2.imread(mask_path)
        colors = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)
        binary_masks = []
        for color in colors:
            temp_mask = mask.copy()
            temp_mask[(temp_mask == color).all(axis=2)] = [255, 255, 255]
            temp_mask[(temp_mask != [255, 255, 255]).any(axis=2)] = [0, 0, 0]
            temp_mask = cv2.cvtColor(temp_mask, cv2.COLOR_BGR2GRAY)
            binary_masks.append(temp_mask)
        return binary_masks, colors
    
    image_folder = os.path.join(data_path, "color")
    image_folder = data_path
    semantic_folder = semantic_path
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
    img0 = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = img0.shape

    video = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    video_ann = cv2.VideoWriter(outfile.replace(".mp4", "_ann.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for idx, image in enumerate(images):
        img = cv2.imread(os.path.join(image_folder, image))
        masks, colors = create_binary_masks(os.path.join(semantic_folder, image.replace(".jpg", ".png")))
        video.write(img)
        for mask, color in zip(masks, colors):
            bbox = get_bbox_from_mask(mask)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color.tolist(), 4)

        cv2.imshow("image", img)
        cv2.waitKey(int(1000/30))
        if idx == 0:
            for i in range(36):
                video.write(img)
        video_ann.write(img)
    cv2.destroyAllWindows()
    video.release()
    video_ann.release()

def panoptic_demo(data_path, semantic_path, outfile, fps = 30):
    images = sorted([img for img in os.listdir(data_path) if img.endswith(".jpg")])
    imgs = [cv2.imread(os.path.join(data_path, image)) for image in images]
    img0 = imgs[0]

    semantics = [np.load(os.path.join(semantic_path, image.replace(".jpg", ".npy"))) for image in images]
    height, width, layers = img0.shape

    video = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    video_ann = cv2.VideoWriter(outfile.replace(".mp4", "_panoptic.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    sem_ids = set()

    for sem in semantics:
        unique_ids = np.unique(sem)
        sem_ids.update(unique_ids)

    sem_ids = list(sem_ids)
    color_palette = generate_color_palette(len(sem_ids))
    color_map = {sem_ids[id]: color_palette[id] for id in range(len(sem_ids))}

    for idx, image in enumerate(images):
        img = cv2.imread(os.path.join(data_path, image))
        sem_raw = np.load(os.path.join(semantic_path, image.replace(".jpg", ".npy")))
        sem_img = np.zeros_like(img)
        for val in sem_ids:
            sem_img[sem_raw == val] = color_map[val]

        combined = cv2.addWeighted(img, 0.6, sem_img, 0.4, 0.0)
        video.write(img)

        cv2.imshow("image", combined)
        cv2.waitKey(int(1000/24))

        video_ann.write(combined)
    cv2.destroyAllWindows()
    video.release()
    video_ann.release()


def darken_color(color, factor=0.9):
    return tuple(int(c * factor) for c in color)

def find_contours(img, semantic_image, semantic_ids):

    contour_image = np.zeros_like(img)
    for i in semantic_ids:
        mask = semantic_image == i

        contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)
    
    return contour_image

def raw_to_object_seg(semantic_obs, semantic_ids, color_map):
        semantic_obs = np.where(semantic_obs < 1000, 0, semantic_obs)
            
        semantic_img = np.zeros((semantic_obs.shape[0], semantic_obs.shape[1], 3))
        for val in semantic_ids:
            semantic_img[semantic_obs == val] = color_map[str(val)]
        semantic_img = semantic_img.astype(np.uint8)
        return semantic_img

def get_bbox_from_mask(mask: np.ndarray):
    mask = mask.squeeze().astype(np.uint8)
    if np.sum(mask) == 0:
        return None
    
    row_indices, col_indices = np.where(mask)

    row_min, row_max = np.min(row_indices), np.max(row_indices)
    col_min, col_max = np.min(col_indices), np.max(col_indices)

    return (col_min, row_min, col_max, row_max)

def generate_color_palette(num_colors):
    random.seed(42) 
    hsv_tuples = [(x / num_colors, 1., 1.) for x in range(num_colors)]
    random.shuffle(hsv_tuples)
    rgb_tuples = map(lambda x: tuple(int(255 * i) for i in colorsys.hsv_to_rgb(*x)), hsv_tuples)
    bgr_tuples = map(lambda x: (x[2], x[1], x[0]), rgb_tuples)
    return list(bgr_tuples)

if __name__ == "__main__":
    # generate_video("/home/nico/semesterproject/data/re-id_benchmark_ycb/multi_object/toys/train/toys_on_ground",
    #                "/home/nico/semesterproject/vis_final_pres/toys_on_ground.mp4", 30)
    # visualize_annotated_frames("/home/nico/semesterproject/data/re-id_benchmark_ycb/multi_object/toys/train/toys_on_ground",
    #                "/home/nico/semesterproject/paper/visualizations/annotated_frames/")
    visualize_gt_semantic_frames("/home/nico/semesterproject/data/re-id_benchmark_ycb/multi_object/tools/train/tools_on_table",
                   "/home/nico/semesterproject/paper/visualizations/gt_segmentation/", "/home/nico/semesterproject/videos/cheezit_in_context.mp4")
    # combine_modalities_davis("/home/nico/semesterproject/data/DAVIS-2017-Unsupervised-trainval-480p/DAVIS/JPEGImages/480p/boxing-fisheye",
    #                    "/home/nico/semesterproject/data/DAVIS-2017-Unsupervised-trainval-480p/DAVIS/Annotations_unsupervised/480p/boxing-fisheye", 
    #                    "/home/nico/semesterproject/vis_final_pres/videos_background/drift_turn.mp4", fps=24)
    # mot_demo("/home/nico/semesterproject/data/DAVIS-2017-Unsupervised-trainval-480p/DAVIS/JPEGImages/480p/boxing-fisheye",
    #                    "/home/nico/semesterproject/data/DAVIS-2017-Unsupervised-trainval-480p/DAVIS/Annotations_unsupervised/480p/boxing-fisheye", 
    #                    "/home/nico/semesterproject/vis_final_pres/videos_background/boxing-fisheye_mot.mp4", fps=24)
    # panoptic_demo("/home/nico/semesterproject/data/re-id_benchmark_ycb/multi_object/cans/test/falling_cans/color",
    #               "/home/nico/semesterproject/data/re-id_benchmark_ycb/multi_object/cans/test/falling_cans/semantic_raw",
    #               "/home/nico/semesterproject/vis_final_pres/videos_background/panoptic_cans.mp4", fps=24)
    # combine_video_audio("/home/nico/Videos/giovanni_giorgio.mp4", "/home/nico/Downloads/giovanni_gorgio_audio.mp3", "/home/nico/Videos/giovanni_giorgio_audio.mp4")