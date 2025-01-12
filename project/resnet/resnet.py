import torch
import torch.nn as nn
from typing import List
from basicblock import BasicBlock
from bottleneck import BottleneckBlock

class ToVect(nn.Module):
  def forward(self, img):
    return img.view(img.size(0), -1)

class ResNetF2G(nn.Module):
    '''
    Frame2Gloss module based on ResNet50
    '''
    def __init__(self, frame_shape: List[int], num_classes):
        super(ResNetF2G, self).__init__()
        
        self.conv1= nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Sequential(BasicBlock(64, 64), BottleneckBlock(64, 64), BottleneckBlock(256, 64))
        
        self.conv3 = nn.Sequential(BasicBlock(256, 128), BottleneckBlock(128, 128), BottleneckBlock(512, 128), BottleneckBlock(512, 128))
        
        self.conv4 = nn.Sequential(BasicBlock(512, 256), BottleneckBlock(256, 256), BottleneckBlock(1024, 256), BottleneckBlock(1024, 256), BottleneckBlock(1024, 256), BottleneckBlock(1024, 256))
        
        self.conv5 = nn.Sequential(BasicBlock(1024, 512), BottleneckBlock(512, 512), BottleneckBlock(2048, 512))
        
        self.tovect = ToVect()
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(frame_shape[0] * frame_shape[1] * 2048, num_classes * 100),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_classes * 100, num_classes * 100),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(num_classes * 100, num_classes))
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        x = self.tovect(x)
        x = self.fc(x)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x