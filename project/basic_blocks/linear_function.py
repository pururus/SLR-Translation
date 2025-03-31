from torch import nn
import torch

class LinearFunction(nn.Module):
    '''
    Apply Linear function R^n -> R^n with parametric matrix
    '''
    def __init__(self, batch_size=8, n=3):
        super(LinearFunction, self).__init__()
        
        self.matrix = nn.Parameter(torch.stack([torch.eye(n) for _ in range(batch_size)]))
    
    def forward(self, x):
        try:
            return self.matrix @ x
        except:
            return self.matrix[:x.shape[0]] @ x

