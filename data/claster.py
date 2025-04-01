import json
from torch import tensor
import torch 
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

data_2d = np.load('/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/data_svd_add_two_full_r_1000.npy')
centers_2d = np.load('/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/centers_svd_add_two_full_r_1000.npy')
c_map = np.load("/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/c_map_window_onehand.npy")

print("drawing")

plt.figure(figsize=(10, 8))
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=c_map, s=25, cmap='viridis', label='Data Points')
for i in range(0, data_2d.shape[0]):
    plt.text(data_2d[i, 0], data_2d[i, 1], s=c_map[i])
plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', s=60, marker='X', label='Centers')

plt.title(f'K-Means Clustering of {data_2d.shape[0]} vectors')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid()
plt.show()