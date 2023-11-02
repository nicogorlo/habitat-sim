import pickle
import os
import numpy as np
import random
import json
from pathlib import Path
import matplotlib.pyplot as plt

import cv2
from PIL import Image

import numpy as np
import quaternion


def load_trajectory_dict(trajectory_path):
    with open(trajectory_path, "rb") as f:
        trajectory_dict = pickle.load(f)
    return trajectory_dict

def visualize_trajectory(trajectory):
    # creates plot of trajectory (given as list of positions) in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    positions = [i[2] for i in trajectory]
    orientations = [i[3] for i in trajectory]
    norm = lambda x: x / np.linalg.norm(x)
    orientations_vec = [norm(quaternion.as_rotation_matrix(i).dot(np.array([1.0,0,0]))) for i in orientations]
    ax.quiver([i[0] for i in positions[::10]], 
              [i[1] for i in positions[::10]],
              [i[2] for i in positions[::10]],
              [i[0] for i in orientations_vec[::10]],
              [i[1] for i in orientations_vec[::10]],
              [i[2] for i in orientations_vec[::10]],
              length=0.4, 
              normalize=True, 
              label='orientation',
              color='r')
    plt.xlabel('x')
    plt.ylabel('y')
    # ax.scatter([i[0] for i in positions],
    #         [i[1] for i in positions],
    #         [i[2] for i in positions],
    #         label='position')
    ax.legend()
    plt.show()


def main():
    trajectory_path = "/home/nico/semesterproject/data/re-id_benchmark_ycb/trajectories/TEEsavR23oF.pickle"
    trajectory_dict = load_trajectory_dict(trajectory_path)
    trajectory_list = trajectory_dict["trajectory"]
    visualize_trajectory(trajectory_list[2])




if __name__ == "__main__":
    main()