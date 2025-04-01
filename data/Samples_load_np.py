import json
from torch import tensor
import torch 
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd


file = open("/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/slovo_mediapipe.json", "r")
data = json.load(file)
annotation_file = "/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/annotations.csv" 
df = pd.read_csv(annotation_file, sep='\t')

window_size = 15

c_map = []
c_keys = dict()
landmarks = []
test_lm = []
test_c_map = []

has_2_only = False
has_both = False

index = 0
i = 0
for video in data:
    matching_row = df[df['attachment_id'] == video]
    
    if not matching_row.empty:
        gloss = matching_row.iloc[0]['text']
        
    if gloss not in c_keys:
        c_keys[gloss] = i
        i += 1
    
    video_lm = []
    index += 1
    for frame in data[video]:
        if 'hand 1' in frame:
            landmark_t = np.concatenate([np.array([[point['x']], [point['z']], [point['z']]]) for point in frame['hand 1']], axis=1)
            video_lm.append(landmark_t)
        
        if 'hand 2' in frame:
            landmark_t_2 = np.concatenate([np.array([[point['x']], [point['z']], [point['z']]]) for point in frame['hand 2']], axis=1)
            video_lm[-1] += landmark_t_2

    for j in range(5, len(video_lm) - window_size, 2):
        if index % 5 == 0:
            test_lm.append(np.concatenate(video_lm[j: j + window_size], axis=1))
            test_c_map.append(c_keys[gloss])
        else:
            landmarks.append(np.concatenate(video_lm[j: j + window_size], axis=1))
            c_map.append(c_keys[gloss])

    
print("Tensored")
landmarks = np.array(landmarks)
c_map = np.array(c_map)

test_lm = np.array(test_lm)
test_c_map = np.array(test_c_map)

print(landmarks.shape)
print(test_lm.shape)

np.save('/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/landmarks__window_onehand.npy', landmarks)
np.save('/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/c_map_window_onehand.npy', c_map)
np.save('/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/landmarks_window_test.npy', test_lm)
np.save('/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/c_map_window_test.npy', test_c_map)