import numpy as np
from torch import tensor
import torch
import os
import uuid

landmarks = np.load('/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/landmarks_window_train.npy')
y = np.load('/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/c_map_window_train.npy')

basic_path = "/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/landmarks_dataset/"

def save_data(landmarks, labels):
    for i in range(landmarks.shape[0]):
        path = f"{basic_path}{labels[i]}/{uuid.uuid4()}.pt"
        torch.save(tensor(landmarks[i]), path)

save_data(landmarks, y)

n = 10

for _ in range(n - 1):
    r_m = np.random.sample((3, 3))
    Q, R = np.linalg.qr(r_m)
    noise = np.random.uniform(0, 1e-5, (3, 3))
    Q += noise
    save_data(Q @ landmarks, y)