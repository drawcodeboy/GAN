import torch
from torch import nn
import torch.nn.functional as F

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

    def adv_loss_update(self, x, d_optim, g_optim):
        '''
        Adversarial learning k-steps(k=1)
        '''
        bce_loss = nn.BCELoss(reduction='mean')
        bz = x.size(0)

        # Train Discriminator
        d_optim.zero_grad()
        z = torch.randn(bz, self.z_dim).to(x.device)
        x_prime = self.generator(z).detach() # Don't train Generator when train Discriminator
    
    @classmethod
    def from_config(cls, cfg):
        return cls(
            z_dim=cfg.get('z_dim', 100),
            img_dim=cfg.get('img_dim', (1, 28, 28))
        )
    
if __name__ == '__main__':
    exit()

    gan = GAN()
    x = torch.randn((32, 1, 28, 28))
    z = torch.randn((32, 100))
    from torch.optim import SGD
    d_optim = SGD(gan.discriminator.parameters())
    g_optim = SGD(gan.generator.parameters())


    gan.adv_loss_update(x, d_optim, g_optim)