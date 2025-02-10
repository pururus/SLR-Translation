import torch
import torch.nn as nn
if __name__ == "__main__":
    from block import Block
else:
    from resnet.block import Block

class BasicBlock(Block):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU()
    
    def forward(self, x: torch):
        identity = x.clone()
        
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)

        x += identity
        x = self.relu(x)
        
        return x
        