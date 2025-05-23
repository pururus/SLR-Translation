import cv2
import mediapipe as mp
import torch
import torch.nn as nn
from typing import List

from mediapipe_transformer.mediapipe_transformer import MediapipeTransformer

class ToVect(nn.Module):
  def forward(self, img):
    return img.view(img.size(0), -1)

class MediapipeF2G(nn.Module):
    '''
    Frame2Gloss module taking into account only mediapipe landmark
    '''
    def __init__(self, num_classes):
        super(MediapipeF2G, self).__init__()
        
        self.mp_transformer = MediapipeTransformer()
        
        self.fc1 = nn.Sequential(
            nn.Linear(126, num_classes * 10),
            nn.ReLU())
        
        self.fc2= nn.Sequential(
            nn.Linear(num_classes * 10, num_classes))
    
    def forward(self, x):
        x = self.mp_transformer.process_img(x)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x