import json
from torch import tensor
import torch 
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

landmarks = np.load('/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/landmarks_window_train.npy')
test = np.load('/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/landmarks_window_test.npy')

average = np.mean(landmarks, axis=0)
landmarks -= np.array([average] * landmarks.shape[0])
test -= np.array([average] * test.shape[0])

landmarks = landmarks.T
test = test.T

U, S, VT = np.linalg.svd(landmarks, full_matrices=False)
new = U[:, :2] @ np.diag(S[:2])

test_new = new @ (new.T @ test)

np.save("/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/svd_train.npy", new)
np.save("/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/svd_test.npy", test)