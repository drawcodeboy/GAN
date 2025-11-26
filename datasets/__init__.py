from .mnist_dataset import MNIST_Dataset

def load_dataset(cfg):
    if cfg['name'] == 'MNIST':
        return MNIST_Dataset.from_config(cfg)