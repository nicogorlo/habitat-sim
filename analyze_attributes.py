import numpy as np
import json
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class AttributeEvaluator:
    def __init__(self):

        self.attribute_counter = {
            "DYN": 0,
            "CLT": 0,
            "CLA": 0,
            "SML": 0,
            "SMF": 0,
            "FST": 0,
        }

        self.datadir = "/home/nico/semesterproject/data/re_id_benchmark_ycb"

        self.setting = "multi_object"

        self.scene_counter = 0

    def load_attributes(self):
        dir = os.path.join(self.datadir, self.setting)
        for task in sorted([i for i in os.listdir(dir) if os.path.isdir(os.path.join(dir, i))]):
            for train_test in ["train", "test"]:
                for sequence in sorted(os.listdir(os.path.join(dir, task, train_test))):
                    self.scene_counter += 1
                    scene_attributes = json.load(open(os.path.join(dir, task, train_test, sequence, "attributes.json")))

                    for attribute in scene_attributes:
                        self.attribute_counter[attribute] += scene_attributes[attribute]
                    
        print(self.scene_counter)

    def visualize_attributes(self):
        font = {'fontname': "Times New Roman", "fontsize": 14}
        plt.rc('ytick', labelsize=14)
        plt.bar(range(len(self.attribute_counter)),list(self.attribute_counter.values()), align='center')
        plt.xticks(range(len(self.attribute_counter)),list(self.attribute_counter.keys()), **font)
        plt.show()


if __name__ == "__main__":
    eval = AttributeEvaluator()
    eval.load_attributes()
    eval.visualize_attributes()
