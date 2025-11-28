import torch
import os, sys
sys.path.append(os.getcwd())

from models import load_model
import cv2
import numpy as np
import argparse
import torchvision.utils as vutils

def main():
    device = 'cuda:0'
    model = load_model({
        'name': 'GAN',
        'img_dim': (1, 28, 28),
        'z_dim': 100}).to(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--optim', '-o', default='sgd')
    
    args = parser.parse_args()
    optim = args.optim

    ckpt = torch.load(f"saved/weights/gan.mnist.{optim}.epochs_200.pth", 
                      map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt['model'])

    # don't need to call torch.no_grad(), and model.eval()
    # model.generate() includes these functions.
    x_prime = model.generate(n_samples=49, device=device)

    vutils.save_image(x_prime,
                      f'assets/generation_results_mnist_grid/generated_samples_{optim}.jpg',
                      nrow=7,
                      normalize=True)

if __name__ == '__main__':
    main()