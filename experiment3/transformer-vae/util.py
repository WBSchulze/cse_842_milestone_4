import matplotlib.pyplot as plt
import numpy as np
import random
import torch

def set_fixed_randomness(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plot_losses(epochs, recon_losses, kl_divergences, bce_losses):
    def plot_loss(ax, epochs, loss, description, color, linewidth=2):
        ax.plot(epochs, loss, label=description, color=color, linewidth=linewidth)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title(description)
        ax.legend()

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # Define a 2x2 subplot grid

    # Plot the Reconstruction Binary Cross-Entropy Loss
    plot_loss(axs[0, 0], epochs, recon_losses, 'Reconstruction Binary Cross-Entropy Loss', 'blue')

    # Plot the KL Divergence Loss
    plot_loss(axs[1, 0], epochs, kl_divergences, 'KL Divergence Loss', 'red')

    # Plot the Classification Binary Cross-Entropy Loss
    plot_loss(axs[1, 1], epochs, bce_losses, 'Classification Binary Cross-Entropy Loss', 'green')

    # Combined plot for the last 10 epochs in the top right position
    if len(epochs) > 10:
        last_10_epochs = epochs[-10:]
        recon_last_10 = recon_losses[-10:]
        kl_last_10 = kl_divergences[-10:]
        bce_last_10 = bce_losses[-10:]

        axs[0, 1].clear()  # Clear any previous plot
        # Plot lines, making sure the one with the largest values is plotted last
        plot_loss(axs[0, 1], last_10_epochs, recon_last_10, 'Reconstruction Loss (Last 10 Epochs)', 'blue')
        plot_loss(axs[0, 1], last_10_epochs, bce_last_10, 'Classification Loss (Last 10 Epochs)', 'green')
        plot_loss(axs[0, 1], last_10_epochs, kl_last_10, 'KL Divergence (Last 10 Epochs)', 'red')

        # Ensure the limits are set to include all data
        axs[0, 1].set_xlim(left=min(last_10_epochs), right=max(last_10_epochs))
        axs[0, 1].set_ylim(bottom=0, top=max(max(recon_last_10), max(kl_last_10), max(bce_last_10)) * 1.1) # 10% more than max

        axs[0, 1].set_title('Combined Loss (Last 10 Epochs)')
        axs[0, 1].legend()

    plt.tight_layout()
    plt.savefig(f'losses.png')
    plt.close()
    #plt.show()  # Show the plot

# Example usage:
# Assuming you have your epochs and loss values as lists
# plot_losses(epochs, recon_losses, kl_divergences, bce_losses)
