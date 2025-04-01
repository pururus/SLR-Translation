from sklearn.neighbors import KNeighborsClassifier
import json
from torch import tensor
import torch 
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix

data = np.load('/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/landmarks__window_onehand.npy')
data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))

data /= (np.max(data, axis=0) + 1e-9)

y = np.load("/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/c_map_window_onehand.npy")

print(data.shape)
print(y.shape)

test_data = np.load('/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/landmarks_window_test.npy')
test_data = test_data.reshape((test_data.shape[0], test_data.shape[1] * test_data.shape[2]))

test_data /= (np.max(test_data, axis=0) + 1e-9)
print(data.shape)

test_y = np.load("/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/c_map_window_test.npy")

# x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(data, y, test_size = 0.1, shuffle=True)

print("Loaded")

model = KNeighborsClassifier(n_neighbors = 1)
model.fit(data, y)

print("Trained")

# predictions = model.predict(test_data)
# print(accuracy_score(test_y, predictions))

error_rates = []
for i in np.arange(1, 11):
    new_model = KNeighborsClassifier(n_neighbors = i)
    new_model.fit(data, y)
    new_predictions = new_model.predict(test_data)
    error_rates.append(accuracy_score(test_y, new_predictions))
    
    print("Done", i)

print(error_rates)
plt.plot(error_rates)
plt.title("Accuracy of KNN classifier")
plt.xlabel("Neighbour number")
plt.ylabel("Accuracy")
plt.show()