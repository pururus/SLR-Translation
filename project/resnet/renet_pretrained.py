import torch
from torchvision import transforms, models
import mediapipe as mp
import numpy as np
import torch.nn as nn

def pretrained_resnet(num_classes):
  model = models.resnet50(pretrained=True)

  for param in model.parameters():
    param.requires_grad = False

  model.fc = nn.Linear(model.fc.in_features, num_classes)
  for param in list(model.children())[-1].parameters():
    param.requires_grad = True

