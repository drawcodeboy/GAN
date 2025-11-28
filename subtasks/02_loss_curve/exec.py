import matplotlib.pyplot as plt
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optim', '-o', default='sgd')

    args = parser.parse_args()
    optim = args.optim

    plt.figure(figsize=(8, 4))

    d_loss = np.load(f"saved/loss/train_loss_discriminator_gan.mnist.{optim}.npy")
    g_loss = np.load(f"saved/loss/train_loss_generator_gan.mnist.{optim}.npy")

    epochs = np.array([i + 1 for i in range(0, len(d_loss))])

    plt.plot(epochs, d_loss, label='Discriminator', color='blue')
    plt.plot(epochs, g_loss, label='Generator', color='red')

    plt.title(f"Loss Curve (optim: {optim})")
    plt.xlabel("Epochs")
    plt.ylabel("Minimax Loss")

    plt.ylim(-0.2, 7)

    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"assets/loss_curve/loss_curve_{optim}.jpg", dpi=500)

if __name__ == '__main__':
    main()