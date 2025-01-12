import torch.nn as nn
from abc import abstractmethod

class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
    
    @abstractmethod
    def forward(self, x):
        pass