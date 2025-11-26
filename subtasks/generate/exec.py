import torch
from torch import nn
import os, sys
sys.path.append(os.getcwd())

from models import load_model
import cv2
import numpy as np
from einops import rearrange

def main():
    device = 'cuda:0'
    model = load_model({
        'name': 'GAN',
        'img_dim': (1, 28, 28),
        'z_dim': 100}).to(device)

    ckpt = torch.load("saved/weights/gan.mnist.epochs_060.pth", 
                      map_location=device,
                      weights_only=False)
    z = torch.randn(1, model.z_dim).to(device)

    model.eval()
    with torch.no_grad():
        x_prime = model.generator(z)
    x_prime = rearrange(x_prime, '1 1 h w -> h w 1')
    x_prime = x_prime.cpu().numpy() * 255.0
    print(x_prime.shape, x_prime.dtype)
    x_prime = x_prime.astype(np.uint8)

    cv2.imwrite("subtasks/generate/sample.png", x_prime)

if __name__ == '__main__':
    main()