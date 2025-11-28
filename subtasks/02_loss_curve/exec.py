import matplotlib.pyplot as plt
import numpy as np

def main():
    plt.figure(figsize=(8, 4))

    optim = 'sgd'
    
    d_loss = np.load("saved/loss/train_loss_discriminator_gan.mnist.{optim}.npy")
    g_loss = np.load("saved/loss/train_loss_generator_gan.mnist.{optim}.npy")

    epochs = np.array([i + 1 for i in range(0, len(d_loss))])

    plt.plot(epochs, d_loss, label='Discriminator', color='blue')
    plt.plot(epochs, g_loss, label='Generator', color='red')

    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Minimax Loss")

    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"assets/loss_curve/loss_curve_{optim}.jpg", dpi=500)

if __name__ == '__main__':
    main()