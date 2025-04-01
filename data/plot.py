import matplotlib.pyplot as plt

import numpy as np

ar = np.load("/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/ac.npy")

plt.plot(ar)
plt.title("Valid Accuracy history per batch utill 10 epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
