import torch
from torch.utils.data import DataLoader
from torch.nn import BCELoss

from data_processing import CustomDataset, preprocess_data, split_data
from TransformerVAE import TransformerVAE
from util import set_fixed_randomness, plot_losses

# General constants
seed = 42

# Dataset constants
max_tokenized_length = 16

# Model constants
model_name = 'bert-base-uncased'
latent_dim = 512            # Dimension of latent space the VAE encodes to

# Training constants
num_epochs = 2000
learning_rate = 1e-5
alpha = 0.5                 # Weight for reconstruction loss
beta = 13.5                 # Weight for KL divergence
annealing_interval = 10     # Increment beta every 10 epochs
beta_increment = 0.675      # Increment beta by this amount every annealing_interval epochs
gamma = 207.5               # Weight for classification loss

# Dev/debug constants
training_dataset_size = 3
text_start_index = 25       # Start at the 25th token to get past the "As an AI language model, I cannot..." part
text_length = 64

# Set randomness to be fixed for reproducibility
set_fixed_randomness(seed=seed)

# Define model and optimizer
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
vae = TransformerVAE(model_name, latent_dim).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

# Load and preprocess data, then setup the dataloader.
X, y = preprocess_data('all_hand_labeled.json', 'response')
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
texts =   [X_train[i][text_start_index:text_length] for i in range(len(X_train))][:training_dataset_size]
print(f"{len(texts)} texts, example: {texts[0]}")
classes = torch.tensor( [ 1. ] * 50 ).float().reshape( ( -1, 1 ) )
dataset = CustomDataset(texts, classes, vae.tokenizer, max_length=max_tokenized_length)
dataloader = DataLoader(dataset, batch_size=min(2, len(dataset)), shuffle=True, drop_last=False)

# Initialize lists to store losses for plotting
recon_losses = []
kl_divergences = []
bce_losses = []
bceLoss = BCELoss()

# Training loop
vae.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    epoch_recon_loss = 0.0
    epoch_kl_div = 0.0
    epoch_bce_loss = 0.0
    num_batches = 0

    for input_ids, attention_mask, rejected in dataloader:
        num_batches += 1
        input_ids, attention_mask, rejected = input_ids.to(device), attention_mask.to(device), rejected.to(device)
        optimizer.zero_grad()

        reconstructed, mu, logvar, classification = vae(input_ids, attention_mask)
        recon_loss, kl_div = vae.vae_loss(reconstructed, input_ids, mu, logvar)
        classification_loss = bceLoss(classification, rejected)
        total_loss = alpha * recon_loss + beta * kl_div + gamma * classification_loss

        total_loss.backward()
        optimizer.step()

        epoch_recon_loss += recon_loss.item() * alpha
        epoch_kl_div += kl_div.item() * beta
        epoch_bce_loss += classification_loss.item() * gamma

    if epoch > 0:
        print(f"Epoch {epoch + 1:3},          Recon Loss: {recon_losses[-1]/num_batches:10.4f}, KL Div: {kl_divergences[-1]/num_batches:10.4f}, BCE Loss: {bce_losses[-1]/num_batches:10.4f}")

    # Forward pass example
    input_ids, attention_mask, rejected = next(iter(dataloader))
    # Dataloader gives us a batch but we just want one sample for now.
    input_ids, attention_mask, rejected = ( input_ids[:1].to(device), 
                                            attention_mask[:1].to(device), 
                                            rejected[:1].to(device) )
    reconstructed, mu, logvar, classification = vae(input_ids, attention_mask)

    # Convert logits to probabilities and tokens
    tokens = vae.logits_to_tokens( reconstructed )
    predicted_tokens = [vae.tokenizer.convert_ids_to_tokens(ids) for ids in tokens]
    correct_tokens = [vae.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]

    print(f"Correct text:       { ' '.join( correct_tokens[0] ) }") 
    print(f"Predicted text:     { ' '.join( predicted_tokens[0] ) }" )
    print(f"Classification was: {'CORRECT' if (classification.item() > 0.5 and rejected.item() == 1) or (classification.item() < 0.5 and rejected.item() == 0) else 'INCORRECT'} (Expected class:   {rejected.item()}, Got class: {classification.item()})")
    print()

    # Store losses for plotting
    recon_losses.append(epoch_recon_loss / num_batches)
    kl_divergences.append(epoch_kl_div / num_batches)
    bce_losses.append(epoch_bce_loss / num_batches)
    plot_losses(range(1, epoch + 2), recon_losses, kl_divergences, bce_losses)

    # Update beta at the specified interval
    if (epoch + 1) % annealing_interval == 0:
        beta = min(beta + beta_increment, beta * 2)  # Ensure beta does not exceed 1
