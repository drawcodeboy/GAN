import matplotlib.pyplot as plt
import numpy as np

def main():
    plt.figure(figsize=(10, 5))

    d_loss = np.load("saved/loss/train_loss_discriminator_gan.mnist.npy")
    g_loss = np.load("saved/loss/train_loss_generator_gan.mnist.npy")

    epochs = np.array([i + 1 for i in range(0, len(d_loss))])

    plt.plot(epochs, d_loss, label='Discriminator')
    plt.plot(epochs, g_loss, label='Generator')

    plt.xlabel("Epochs")
    plt.ylabel("Minimax Loss")

    plt.legend()
    plt.tight_layout()
    
    plt.savefig("assets/loss_curve/loss_curve.jpg", dpi=500)

if __name__ == '__main__':
    main()