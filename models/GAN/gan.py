import torch
from torch import nn

from .generator import Generator
from .discriminator import Discriminator

class GAN(nn.Module):
    def __init__(self,
                 z_dim=100,
                 img_dim=(1, 28, 28)):
        super().__init__()
        self.z_dim = z_dim
        self.generator = Generator(z_dim=z_dim,
                                   img_dim=img_dim)
        self.discriminator = Discriminator(img_dim=img_dim)
    
    def forward(self, x):
        pass
    
    @classmethod
    def from_config(cls, cfg):
        return cls(
            z_dim=cfg.get('z_dim', 100),
            img_dim=cfg.get('img_dim', (1, 28, 28))
        )