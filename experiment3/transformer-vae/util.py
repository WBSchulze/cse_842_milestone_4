import matplotlib.pyplot as plt
import numpy as np
import random
import torch

def set_fixed_randomness(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def plot_losses(epochs, recon_losses, kl_divergences, bce_losses, epoch_window=10):
    def plot_loss(ax, epochs, loss, description, color, linewidth=2):
        ax.plot(epochs, loss, label=description, color=color, linewidth=linewidth)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title(description)
        ax.legend()

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # Define a 2x2 subplot grid

    plot_loss(axs[0, 1], epochs, recon_losses, 'Reconstruction Binary Cross-Entropy Loss', 'blue')
    plot_loss(axs[1, 0], epochs, kl_divergences, 'KL Divergence Loss', 'red')
    plot_loss(axs[1, 1], epochs, bce_losses, 'Classification Binary Cross-Entropy Loss', 'green')

    # Combined plot for the last epoch_window epochs in the top right position
    if len(epochs) > epoch_window:
        last_n_epochs = epochs[-epoch_window:]
        recon_last_n = recon_losses[-epoch_window:]
        kl_last_n = kl_divergences[-epoch_window:]
        bce_last_n = bce_losses[-epoch_window:]

        axs[0, 0].clear()  # Clear any previous plot
        # Plot lines, making sure the one with the largest values is plotted last
        plot_loss(axs[0, 0], last_n_epochs, recon_last_n, f'Reconstruction Loss (Last {epoch_window} Epochs)', 'blue')
        plot_loss(axs[0, 0], last_n_epochs, bce_last_n, f'Classification Loss (Last {epoch_window} Epochs)', 'green')
        plot_loss(axs[0, 0], last_n_epochs, kl_last_n, f'KL Divergence (Last {epoch_window} Epochs)', 'red')

        # Ensure the limits are set to include all data
        axs[0, 0].set_xlim(left=min(last_n_epochs), right=max(last_n_epochs))
        axs[0, 0].set_ylim(bottom=0, top=max(max(recon_last_n), max(kl_last_n), max(bce_last_n)) * 1.1) # 10% more than max

        axs[0, 0].set_title('Combined Loss (Last 10 Epochs)')
        axs[0, 0].legend()

    plt.tight_layout()
    plt.savefig(f'losses.png')
    plt.close()

def log_epoch(epoch, recon_losses, kl_divergences, bce_losses, classification, rejected, num_batches, epoch_recon_loss, epoch_kl_div, epoch_bce_loss, dataloader, device, vae):
    # Forward pass example (dataloader gives us a batch but we just want one sample for now)
    input_ids, attention_mask, rejected = next(iter(dataloader))
    input_ids, attention_mask, rejected = (input_ids[:1].to(device), attention_mask[:1].to(device), rejected[:1].to(device))
    reconstructed, mu, logvar, classification = vae(input_ids, attention_mask)
    correct_tokens =    [vae.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
    predicted_tokens =  [vae.tokenizer.convert_ids_to_tokens(ids) for ids in vae.logits_to_tokens(reconstructed)]
    
    print(f"Correct text:       { ' '.join( correct_tokens[0]   ) }") 
    print(f"Predicted text:     { ' '.join( predicted_tokens[0] ) }")
    
    # Extract values from tensors for clarity
    classification_value = classification.item()
    rejected_value = rejected.item()

    # Determine classification status
    if ((classification_value > 0.5 and rejected_value == 1) or 
        (classification_value < 0.5 and rejected_value == 0)):
        classification_status = 'CORRECT'
    else:
        classification_status = 'INCORRECT'

    # Construct and print the message
    print(f"Classification was: {classification_status} "
        f"(Expected class {rejected_value}, got class {classification_value:.3f}).\n")

    # Store losses for plotting
    recon_losses.append(epoch_recon_loss / num_batches)
    kl_divergences.append(epoch_kl_div / num_batches)
    bce_losses.append(epoch_bce_loss / num_batches)
    plot_losses(range(1, epoch + 2), recon_losses, kl_divergences, bce_losses)

    if epoch > 0:
        print(f"-------- Epoch {epoch + 1} -------- \n")
        print
        (       
            f"Reconstruction Loss: {    recon_losses[-1]   / num_batches:10.4f},\n"
            f"KL Divergence: {          kl_divergences[-1] / num_batches:10.4f},\n"
            f"Classification Loss: {    bce_losses[-1]     / num_batches:10.4f}"
        )

def get_device():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}\n")

    return device
