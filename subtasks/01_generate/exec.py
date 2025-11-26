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

    ckpt = torch.load("saved/weights/gan.mnist.epochs_100.pth", 
                      map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt['model'])

    # don't need to call torch.no_grad(), and model.eval()
    # model.generate() includes these functions.
    x_prime = model.generate(n_samples=3, device=device)

    for idx in range(1, x_prime.size(0)+1):
        generated_img = (x_prime[0].cpu().numpy() * 255.0).astype(np.uint8)
        generated_img = np.transpose(generated_img, (1, 2, 0)) # (C, H, W) -> (H, W, C)

        cv2.imwrite(f"subtasks/01_generate/sample_{idx:02d}.png", generated_img)

if __name__ == '__main__':
    main()