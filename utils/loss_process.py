import torch
from torch import nn

def adversarial_process(model, data, loss_fn, optimizer, device):
    x = data
    x = x.to(device)

    batch_size = x.size(0)

    real = torch.ones(batch_size, 1).to(device)
    fake = torch.zeros(batch_size, 1).to(device)

    ### Train Generator ###
    optimizer['gen_optim'].zero_grad()
    z = torch.randn(batch_size, model.z_dim).to(device)

    x_prime = model.generator(z)

    # Non-saturating loss (Generator)
    g_loss = loss_fn(model.discriminator(x_prime), real) # -E[log D(G(z))]

    g_loss.backward()
    optimizer['gen_optim'].step()

    # ==================================================

    ### Train Discriminator ###
    optimizer['disc_optim'].zero_grad()
    real_loss = loss_fn(model.discriminator(x), real) # -E[log D(x)]
    fake_loss = loss_fn(model.discriminator(x_prime.detach()), fake) # -E[log(1 - D(G(z)))]
    d_loss = (real_loss + fake_loss) / 2 

    d_loss.backward()
    optimizer['disc_optim'].step()

    return g_loss.item(), d_loss.item()