import torch
from torch import nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self,
                 z_dim=100,
                 img_dim=(1, 28, 28)):
        super().__init__()
        self.img_dim = img_dim
        self.li1 = nn.Linear(100, 256)
        self.li2 = nn.Linear(256, self.img_dim[0]*self.img_dim[1]*self.img_dim[2])

        self.acti = nn.ReLU()

    def forward(self, z):
        z = self.acti(self.li1(z))
        x = F.sigmoid(self.li2(z))
        x = x.view(-1, self.img_dim[0], self.img_dim[1], self.img_dim[2])
        return x