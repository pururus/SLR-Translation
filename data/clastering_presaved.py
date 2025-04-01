import json
from torch import tensor
import torch 
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

landmarks = np.load('/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/landmarks__window_onehand.npy')

c_map = np.load("/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/c_map_window_onehand.npy")

print(c_map.size)
landmarks = landmarks.reshape((273469, 3 * 315))
n_clusters = 1000

kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(landmarks)

# Получение меток кластеров и центров кластеров
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Понижение размерности для визуализации (если нужно)
pca = PCA(n_components=2)
data_2d = pca.fit_transform(landmarks)
centers_2d = pca.transform(centers)

np.save(f'/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/data_svd_add_two_full_r_{n_clusters}.npy', data_2d)
np.save(f'/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/centers_svd_add_two_full_r_{n_clusters}.npy', centers_2d)