from torch import nn

class ToVect(nn.Module):
    def forward(self, img):
        return img.view(img.size(0), -1)