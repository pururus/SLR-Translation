from torch import nn, tensor

from basic_blocks.tovect import ToVect
from basic_blocks.linear_function import LinearFunction

class Landmark2G(nn.Module):
    def __init__(self, num_classes, batch_size=8):
        super(Landmark2G, self).__init__()
        
        self.lin_func = LinearFunction(batch_size)
        self.tovect = ToVect()
        
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(121, num_classes * 100),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(num_classes * 100, num_classes))
    
    def forward(self, x):
        x = self.lin_func(x)
        x = self.tovect(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x        