import datetime
import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_processing import CustomDataset, preprocess_data, split_data
from GruVae import GruVae
from util import ( set_fixed_randomness, plot_losses, engineered_factor, 
                  engineered_factor_switching, save, load )

# General constants
seed = 42

# Model constants
latent_dim = 512            # Dimension of latent space the VAE encodes to
gru_layers = 3              # Layers per GRU

# Training constants
num_epochs = 10000
# training_dataset_size = 500     # 517 rejected texts in current dataset
learning_rate = 1e-5
lw_recon = 1                      # Weight for reconstruction loss
lw_KL_max = 1e-1                  # Weight for KL divergence
lw_classification_max = 1e-2      # Weight for classification loss
annealing_epoch_period = 15       # How long the annealing cycle is.  Ideally a multiple of 3.
batch_size = 10                   # How many samples the GRU processes between backprops.

# Set randomness to be fixed for reproducibility
set_fixed_randomness(seed=seed)

# Define model and optimizer
# Check if MPS or CUDA is available and use them; otherwise, use CPU
device = torch.device("cpu" )
if torch.backends.mps.is_available():
    device = torch.device( "mps" )
elif torch.cuda.is_available():
    device = torch.device( "cuda" )
print(f"Using device: {device}")

model = GruVae( latent_dim, gru_layers ).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.1, patience = 2, verbose = True, min_lr = 1e-7 )

# Load and preprocess labeled refusals into train/val/test.
X, y, c = preprocess_data('quora_insincere_hand_labeled.json', 'prompt', 1.0 )
X_train, X_val, X_test, y_train, y_val, y_test, c_train, c_val, c_test = split_data(X, y, c)
X_train_01, y_train_01, c_train_01 = preprocess_data('quora_64k_bootstrapped.json', 'prompt', 0.1 )

datasets = [ CustomDataset(X_d, y_d, c_d, model.tokenizer)
             for X_d, y_d, c_d 
             in zip( ( X_train_01, 
                      X_train, ),
                     ( y_train_01, 
                      y_train, ),
                     ( c_train_01, 
                      c_train, ) ) ]
dataloaders = [ DataLoader(dataset, batch_size = 1, shuffle=True, drop_last=False) 
                for dataset in datasets ]

# Initialize lists to store losses for plotting
recon_losses = []
kl_divergences = []
bce_losses = []
lws_overall = []
lws_KL = []
lws_classification = []

# Training loop
epochStartTime = datetime.datetime.now()
batch_loss = None
initial_epoch = 0
try: 
     initial_epoch = load( model, optimizer )
     print( f"Loaded saved model at epoch {initial_epoch}.")
except FileNotFoundError:
     print( f"No saved model.  Starting from epoch {initial_epoch}.")

for epoch in range(initial_epoch, initial_epoch + num_epochs):
    annealing_factor = engineered_factor( epoch, annealing_epoch_period )
    lw_KL = lw_KL_max * annealing_factor
    lw_classification = lw_classification_max * annealing_factor
    lws_overall.append( optimizer.param_groups[0]['lr'] )
    lws_KL.append( lw_KL )
    lws_classification.append( lw_classification )
    print( f"LR: {optimizer.param_groups[0]['lr']:.3e}, Annealing: {annealing_factor:.3e}, LW_KL: {lw_KL:.3e}, LW_classification: {lw_classification:.3e}.")

    # Train model
    #==================================================================
    model.train()

    epoch_loss = 0.0
    epoch_recon_loss = 0.0
    epoch_kl_div = 0.0
    epoch_bce_loss = 0.0
    num_batches = 0
    
    for dataloader in dataloaders:
        for input_ids, rejected, confidence in dataloader:
            input_ids, rejected = input_ids[0].to(device), rejected[0].to(device)
            logits, mu, logvar, classification = model(input_ids)
            recon_loss = model.recon_loss(logits, input_ids)
            kl_div = model.kl_div_loss( mu, logvar)
            classification_loss = model.classification_loss( rejected, classification )
            if batch_loss is None:
                batch_loss = lw_recon * recon_loss + lw_KL * kl_div + lw_classification * classification_loss
            else:
                batch_loss += lw_recon * recon_loss + lw_KL * kl_div + lw_classification * classification_loss

            epoch_recon_loss += recon_loss.item()
            epoch_kl_div += kl_div.item()
            epoch_bce_loss += classification_loss.item() * confidence.item()

            num_batches += 1

            if num_batches % batch_size == 0:
                epoch_loss += batch_loss.item()
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_loss = None

    #==================================================================
    # Adjust hyperparameters for next epoch
    #------------------------------------------------------------------
    # Reset "best value" if this is coming off a KL/classification cycle.
    if engineered_factor_switching( epoch, annealing_epoch_period ):
            print( f"Resetting learning rate scheduler's best loss value at epoch {epoch}.")
            lr_scheduler.best = float( 'inf' )
    if ( annealing_factor == 0 ) or ( annealing_factor == 1 ):
        lr_scheduler.step( epoch_recon_loss )
    epochTime = datetime.datetime.now() - epochStartTime
    epochStartTime = datetime.datetime.now()
    print(f"\nEpoch {epoch + 1} ({num_batches} batches), Time: {epochTime}, Recon Loss: {epoch_recon_loss/num_batches:.3e}, KL Div: {epoch_kl_div/num_batches:.3e}, Classification Loss: {epoch_bce_loss/num_batches:.3e}")

    # Forward pass example, plotting
    #==================================================================
    input_ids, rejected, _ = next(iter(dataloader))
    # Dataloader gives us a batch but we just want one sample for now.
    input_ids, rejected = ( input_ids[0].to(device), 
                             rejected[0].to(device) )
    correct_tokens = model.tokenizer.convert_ids_to_tokens(input_ids)
    correct_text = " ".join( correct_tokens )

    reconstructed, _, _, rejected_pred = model(input_ids)

    # Convert logits to probabilities and tokens
    tokens = model.logits_to_tokens( reconstructed )
    predicted_tokens = model.tokenizer.convert_ids_to_tokens(tokens)
    predicted_text = " ".join( predicted_tokens )
    print(f"Correct text:   {correct_text}") 
    print(f"Predicted text: {predicted_text}" )
    print(f"Classification was: {'CORRECT' if (rejected_pred.item() > 0.5 and rejected.item() == 1) or (rejected_pred.item() < 0.5 and rejected.item() == 0) else 'INCORRECT'} (Expected class: {rejected.item()}, Got class: {rejected_pred.item():.4f})")

    # Store losses for plotting
    recon_losses.append(epoch_recon_loss / num_batches)
    kl_divergences.append(epoch_kl_div / num_batches)
    bce_losses.append(epoch_bce_loss / num_batches)

    plot_losses(range(initial_epoch, epoch + 1), recon_losses, kl_divergences, bce_losses,
                lws_overall, lws_KL, lws_classification )
    
    save( epoch, model, optimizer, path = f'model_{epoch % 10}.pt' )
