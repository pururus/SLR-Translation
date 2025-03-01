import torch
from torchvision import transforms, models
import mediapipe as mp
import numpy as np
import torch.nn as nn

def pretrained_resnet(num_classes):
  '''
  Pretrained resnet 50
  '''
  
  model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)

  for param in model.parameters():
    param.requires_grad = False

  model.fc = nn.Linear(model.fc.in_features, num_classes)
  for param in list(model.children())[-1].parameters():
    param.requires_grad = True
  for param in list(model.children())[-2].parameters():
    param.requires_grad = True
  
  return model

