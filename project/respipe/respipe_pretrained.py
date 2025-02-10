import torch
from torchvision import transforms, models
import mediapipe as mp
import numpy as np
import torch.nn as nn
from mediapipe_transformer.mediapipe_transformer import MediapipeTransformer

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class id_layer(nn.Module):
  def __init__(self):
    super(id_layer, self).__init__()
  
  def forward(self, x):
    return x

class PreResNetMediapipe(nn.Module):
  def __init__(self, num_classes):
    super(PreResNetMediapipe, self).__init__()

    self.model = models.resnet50(pretrained=True)

    for param in self.model.parameters():
      param.requires_grad = False
    
    in_features = self.model.fc.in_features

    self.model.fc = id_layer()
    for param in list(self.model.children())[-1].parameters():
      param.requires_grad = True
    
    self.mp_transformer = MediapipeTransformer()

    self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features + 126, num_classes * 100),
            nn.ReLU())
    self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_classes * 100, num_classes * 100),
            nn.ReLU())
    self.fc2= nn.Sequential(
            nn.Linear(num_classes * 100, num_classes))

  def forward(self, x):
    landmarks = self.mp_transformer.process_img(x)

    x = self.model(x)
    x = torch.concatenate(x, landmarks)
    x = self.fc(x)
    x = self.fc1(x)
    x = self.fc2(x)
        
    return x