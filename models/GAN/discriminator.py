import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self,
                 img_dim=(1, 28, 28)):
        super().__init__()
        self.img_dim = img_dim
        self.li1 = nn.Linear(self.img_dim[0]*self.img_dim[1]*self.img_dim[2], 512)
        self.li2 = nn.Linear(512, 256)
        self.li3 = nn.Linear(256, 1)

        self.acti = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, self.img_dim[0]*self.img_dim[1]*self.img_dim[2])
        x = self.acti(self.li1(x))
        x = self.acti(self.li2(x))
        x = self.li3(x)
        y = self.sigmoid(x)
        return y