from .GAN.gan import GAN

def load_model(cfg):
    if cfg['name'] == 'GAN':
        return GAN.from_config(cfg)