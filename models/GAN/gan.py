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
        bce_loss = nn.BCELoss(reduction='mean') # In paper, they used mean
        bz = x.size(0)

        # Train Discriminator
        d_optim.zero_grad()
        z = torch.randn(bz, self.z_dim).to(x.device)
        x_prime = self.generator(z).detach() # Don't need to calculate Generator's gradient

        # Discrimnator wants to maximize minimax loss
        # Maximize log(D(x)) + log(1-D(G(z)))
        # So, minimize -log(D(x)) -log(1-D(G(z)))
        ones = torch.ones(bz, 1).to(x.device)
        zeros = torch.ones(bz, 1).to(x.device)
        d_loss_1 = bce_loss(self.discriminator(x), ones) # -log(D(x))
        d_loss_2 = bce_loss(self.discriminator(x_prime), zeros) # -log(1-D(G(z)))
        d_loss = d_loss_1 + d_loss_2

        d_loss.backward()
        d_optim.step()

        # Train Generator
        g_optim.zero_grad()
        z = torch.randn(bz, self.z_dim).to(x.device)
        x_prime = self.generator(z)

        # Generator wants to minimize log(1-D(G(z)))
        # But, It suffers from gradient saturation problem.
        # So, Change the loss function "Maximize log(D(G(z)))"
        # Finally, If I want to maximize this term, "Minimize -(D(G(z)))"
        ones = torch.ones(bz, 1).to(x.device)
        g_loss = bce_loss(self.discriminator(x_prime), ones)

        # I don't worried that "Is Discriminator trained?"
        # -> Because, training is only depend on "optimizer(model.parameters())"
        # -> I declared g_optim = Optim(model.generator.parameters())
        g_loss.backward()
        g_optim.step() 

        return d_loss.item(), g_loss.item()

    @classmethod
    def from_config(cls, cfg):
        return cls(
            z_dim=cfg.get('z_dim', 100),
            img_dim=cfg.get('img_dim', (1, 28, 28))
        )
    
if __name__ == '__main__':
    gan = GAN()
    x = torch.randn((32, 1, 28, 28))
    z = torch.randn((32, 100))
    from torch.optim import SGD
    d_optim = SGD(gan.discriminator.parameters())
    g_optim = SGD(gan.generator.parameters())


    gan.adv_loss_update(x, d_optim, g_optim)