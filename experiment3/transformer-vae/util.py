import matplotlib.pyplot as plt
import numpy as np
import random
import torch

def set_fixed_randomness(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def plot_losses(epochs, recon_losses, kl_divergences, bce_losses):
    def plot_loss(epochs, loss, description, color):
        plt.plot(epochs, loss, label=description, color=color)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(description)
        plt.legend()

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plot_loss(epochs, recon_losses, 'Reconstruction Binary Cross-Entropy Loss', 'blue')
    
    plt.subplot(2, 2, 3)
    plot_loss(epochs, kl_divergences, 'KL Divergence Loss', 'red')
    
    plt.subplot(2, 2, 4)
    plot_loss(epochs, bce_losses, 'Classification Binary Cross-Entropy Loss', 'green')

    plt.tight_layout()
    plt.savefig(f'losses.png')
    plt.close()
