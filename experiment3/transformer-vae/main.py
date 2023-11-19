import torch
from torch.nn import BCELoss

from data_processing import load_data
from TransformerVAE import TransformerVAE
from util import set_fixed_randomness, log_epoch, get_device

set_fixed_randomness(seed=42)

# Model constants
model_name = 'bert-base-uncased'
latent_dim = 512

# Training constants
num_epochs              = 5000
learning_rate           = 1e-6
recon_weight            = 0.5
kl_weight, kl_increment = 0.1, 0.01
classification_weight   = 15

# Define model and optimizer
device = get_device()
vae = TransformerVAE(model_name, latent_dim).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate, weight_decay=1e-5)

dataloader = load_data(
    text_start_index        = 0,
    text_length             = 64,
    training_dataset_size   = 16,
    max_tokenized_length    = 16,
    tokenizer               = vae.tokenizer
)

# Training loop
recon_losses, kl_divergences, bce_losses, bceLoss = [], [], [], BCELoss()
vae.train()
for epoch in range(num_epochs):
    epoch_recon_loss, epoch_kl_div, epoch_bce_loss, num_batches = 0.0, 0.0, 0.0, 0
    for input_ids, attention_mask, rejected in dataloader:
        num_batches += 1
        input_ids, attention_mask, rejected = input_ids.to(device), attention_mask.to(device), rejected.to(device)

        reconstructed, mu, logvar, classification = vae(input_ids, attention_mask)
        recon_loss, kl_div =                        vae.vae_loss(reconstructed, input_ids, mu, logvar)
        classification_loss =                       bceLoss(classification, rejected)
        epoch_recon_loss +=                         recon_loss.item()           * recon_weight
        epoch_kl_div +=                             kl_div.item()               * kl_weight
        epoch_bce_loss +=                           classification_loss.item()  * classification_weight
        total_loss = recon_loss * recon_weight + kl_div * kl_weight + classification_loss * classification_weight

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    kl_weight = min(kl_weight + kl_increment, 1.0)  # Annealing

    log_epoch(epoch, recon_losses, kl_divergences, bce_losses, classification, rejected, num_batches, epoch_recon_loss, epoch_kl_div, epoch_bce_loss, dataloader, device, vae)
