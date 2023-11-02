import numpy as np
import os
from itertools import combinations
import json
import cv2
from tqdm import tqdm

def find_best_sets(datadir):
    samples = {}
    for sequence in tqdm(sorted(os.listdir(datadir))):
        indices = []
        sequence_path = os.path.join(datadir, sequence)
        sem_raw_path = os.path.join(sequence_path, "semantic_raw")
        for img_name in os.listdir(sem_raw_path):
            img = cv2.imread(os.path.join(sem_raw_path, img_name), -1)
            indices += [i for i in np.unique(img) if i >= 1660]
        
        indices = np.unique(indices)

        samples[sequence] = indices
        
    return samples

def common_attributes(sample_ids, samples):
    if not sample_ids:
        return set()
    common = set(samples[sample_ids[0]])
    for sample_id in sample_ids[1:]:
        common = common.intersection(set(samples[sample_id]))
    return common

# Function to add a sample to a group that maximizes common attributes
def add_sample_to_best_group(sample_id, samples, groups):
    max_common = -1
    best_group = None
    
    for group in groups:
        new_group = group + [sample_id]
        common = common_attributes(new_group, samples)
        
        if len(common) > max_common:
            max_common = len(common)
            best_group = group
            
    best_group.append(sample_id)



if __name__ == "__main__":
    datadir = "/home/nico/semesterproject/data/re_id_benchmark_ycb/new_scenes_test"
    
    samples = find_best_sets(datadir)

    groups = [[] for _ in range(7)]
    
    for sample_id in samples.keys():
        add_sample_to_best_group(sample_id, samples, groups)

    # Print the groups
    for i, group in enumerate(groups):
        common = common_attributes(group, samples)
        print(f"Group {i+1}: Sample IDs {group} with common attributes {common}")
