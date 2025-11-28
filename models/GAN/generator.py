import torch
from torch import nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self,
                 z_dim=100,
                 img_dim=(1, 28, 28)):
        super().__init__()
        self.img_dim = img_dim
        self.li1 = nn.Linear(100, 128)
        self.bn1 = nn.BatchNorm1d(128, 0.8)
        
        self.li2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256, 0.8)

        self.li3 = nn.Linear(256, 512)
        self.bn3 = nn.BatchNorm1d(512, 0.8)

        self.li4 = nn.Linear(512, 1024)
        self.bn4 = nn.BatchNorm1d(1024, 0.8)

        self.li5 = nn.Linear(1024, self.img_dim[0]*self.img_dim[1]*self.img_dim[2])

        self.acti = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, z):
        z = self.acti(self.bn1(self.li1(z)))
        z = self.acti(self.bn2(self.li2(z)))
        z = self.acti(self.bn3(self.li3(z)))
        z = self.acti(self.bn4(self.li4(z)))
        z = self.li5(z)
        x = F.sigmoid(z) # different with referece repository, they used tanh().
        x = x.view(-1, self.img_dim[0], self.img_dim[1], self.img_dim[2])
        return x