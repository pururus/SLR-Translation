import torch
import torch.nn as nn
if __name__ == "__main__":
    from block import Block
else:
    from resnet.block import Block

class BottleneckBlock(Block):    
    def __init__(self, in_channels, out_channels, expansion=4):
        super(BottleneckBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * expansion)

        self.downsample = nn.Conv2d(in_channels=in_channels, out_channels=out_channels * expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm4 = nn.BatchNorm2d(out_channels * expansion)
        
        self.relu = nn.ReLU()
    
    def forward(self, x: torch):
        identity = x.clone()
        
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
            
        x += self.batch_norm4(self.downsample(identity))
        x = self.relu(x)
        
        return x
        