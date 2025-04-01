import json
from torch import tensor
import torch 
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

annotation_file = "/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/annotations.csv"  # Путь к вашему файлу аннотаций
df = pd.read_csv(annotation_file, sep='\t')

c_map = []
c_keys = dict()
rev = dict()

i = 0
for index, row in df.iterrows():
    if row["text"] not in c_keys:
        c_keys[row["text"]] = i
        rev[i] = row["text"]
        i += 1

inp = int(input())
while inp >= 0:
    print(rev[inp])
    inp = int(input())
    
inp = input()
while inp != "-1":
    print(c_keys[inp])
    inp = input()