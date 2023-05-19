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

if __name__ == "__main__":
    generate_video("/home/nico/semesterproject/data/re-id_benchmark/multi_object/sliding_trashcan_plastic_drum/",
                   "/home/nico/semesterproject/videos/sliding_trashcan_plastic_drum.mp4", 30)
    # combine_video_audio("/home/nico/Videos/giovanni_giorgio.mp4", "/home/nico/Downloads/giovanni_gorgio_audio.mp3", "/home/nico/Videos/giovanni_giorgio_audio.mp4")