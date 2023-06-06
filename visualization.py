import cv2
import os
import numpy as np
import json

def generate_video(data_path, video_name, fps):
    image_folder = os.path.join(data_path, "color")
    semantic_folder = os.path.join(data_path, "semantic")
    if os.path.exists(os.path.join(data_path, "prompts.json")):
        with open(os.path.join(data_path, "prompts.json")) as f:
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
        combined = cv2.addWeighted(cv2.imread(os.path.join(image_folder, image)), 0.5, cv2.imread(os.path.join(semantic_folder, image)), 0.5, 0)

        if prompts is not None:
            if image.split('.')[0] in list(prompts.keys()):
                for i in prompts[image.split('.')[0]]:
                    cv2.circle(combined, (prompts[image.split('.')[0]][i]["point_prompt"][0], prompts[image.split('.')[0]][i]["point_prompt"][1]), 5, (0, 0, 255), -1)
                    cv2.rectangle(combined, (prompts[image.split('.')[0]][i]["bbox"][0], prompts[image.split('.')[0]][i]["bbox"][1]), (prompts[image.split('.')[0]][i]["bbox"][2], prompts[image.split('.')[0]][i]["bbox"][3]), (0, 255, 0), 2)
        video.write(combined)

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


def visualize_gt_semantic_frames(data_path, outpath):
    image_folder = os.path.join(data_path, "color")
    semantic_folder = os.path.join(data_path, "semantic")
    semantic_raw_folder = os.path.join(data_path, "semantic_raw")

    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])

    # Define the codec and create a VideoWriter object
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

        # cv2.imshow("combined", combined)
        # cv2.waitKey(int(1000/30))
        cv2.imwrite(os.path.join(outpath, data_path.split('/')[-1], image), combined)
    cv2.destroyAllWindows()

def find_contours(img, semantic_image, semantic_ids):


    contour_image = np.zeros_like(img)
    for i in semantic_ids:
        mask = semantic_image == i
        # Use OpenCV to find contours
        contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Create an empty image to draw the contours
        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)
    
    return contour_image

def raw_to_object_seg(semantic_obs, semantic_ids, color_map):
        semantic_obs = np.where(semantic_obs < 1000, 0, semantic_obs)
            
        semantic_img = np.zeros((semantic_obs.shape[0], semantic_obs.shape[1], 3))
        for val in semantic_ids:
            semantic_img[semantic_obs == val] = color_map[str(val)]
        semantic_img = semantic_img.astype(np.uint8)
        return semantic_img

if __name__ == "__main__":
    # generate_video("/home/nico/semesterproject/data/re-id_benchmark_ycb/multi_object/tools/train/tools_on_table",
    #                "/home/nico/semesterproject/videos/tools_on_table.mp4", 30)
    visualize_annotated_frames("/home/nico/semesterproject/data/re-id_benchmark_ycb/multi_object/toys/train/toys_on_ground",
                   "/home/nico/semesterproject/paper/visualizations/annotated_frames/")
    visualize_gt_semantic_frames("/home/nico/semesterproject/data/re-id_benchmark_ycb/multi_object/toys/test/toys_moving_manual",
                   "/home/nico/semesterproject/paper/visualizations/gt_segmentation/")
    # combine_video_audio("/home/nico/Videos/giovanni_giorgio.mp4", "/home/nico/Downloads/giovanni_gorgio_audio.mp3", "/home/nico/Videos/giovanni_giorgio_audio.mp4")