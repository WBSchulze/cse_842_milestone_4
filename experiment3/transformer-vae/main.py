import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torch.nn import BCELoss

from data_processing import preprocess_data, split_data

import numpy as np
import random
# Set a fixed randomness for reproducibility.
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

class TransformerVAE(nn.Module):
    def __init__(self, model_name, latent_dim):
        super(TransformerVAE, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Define the mean and log-variance layers for the latent space
        self.fc_mu = nn.Linear(self.encoder.config.hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder.config.hidden_size, latent_dim)

        # Add a linear layer to map hidden states to the vocabulary size
        self.linear_layer = nn.Linear(self.encoder.config.hidden_size, self.tokenizer.vocab_size)

        # 1 dimension: rejected or not
        self.rejection_classifier = nn.Linear( latent_dim, 1 )
        self.sigmoid = nn.Sigmoid()

        # Decoder can be another transformer or a different architecture
        self.decoder = AutoModel.from_pretrained(model_name)

    def encode(self, input_ids, attention_mask):
        encoded = self.encoder(input_ids, attention_mask=attention_mask)[0]
        cls_encoded = encoded[:,0,:]
        mu = self.fc_mu(cls_encoded)
        logvar = self.fc_logvar(cls_encoded)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, input_seq_len, temperature=1.0):
        generated = torch.full((z.size(0), 1), self.tokenizer.cls_token_id, dtype=torch.long, device=z.device)
        for _ in range(input_seq_len - 1):
            outputs = self.decoder(input_ids=generated, encoder_hidden_states=z)
            hidden_states = outputs.last_hidden_state
            next_token_logits = self.linear_layer(hidden_states[:, -1, :])
            next_token_logits = next_token_logits / temperature  # Apply temperature
            probabilities = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1)  # Sample from the probability distribution
            generated = torch.cat((generated, next_token), dim=-1)
        logits = self.linear_layer(hidden_states[:, :input_seq_len, :])
        return logits


    # def decode(self, z, input_seq_len):
    #     generated = torch.full((z.size(0), 1), self.tokenizer.cls_token_id, dtype=torch.long, device=z.device)

    #     # Adjust loop to generate tokens up to input_seq_len
    #     for _ in range(input_seq_len - 1):  # Adjust for the [CLS] token
    #         if generated.size(1) == input_seq_len:  # Stop if sequence length is reached
    #             break
    #         outputs = self.decoder(input_ids=generated, encoder_hidden_states=z)
    #         hidden_states = outputs.last_hidden_state
    #         next_token_logits = self.linear_layer(hidden_states[:, -1, :])
    #         next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
    #         generated = torch.cat((generated, next_token), dim=-1)

    #     logits = self.linear_layer(hidden_states[:, :input_seq_len, :])
    #     return logits


    # def decode(self, z, input_seq_len):
    #     generated = torch.full((z.size(0), 1), self.tokenizer.cls_token_id, dtype=torch.long, device=z.device)

    #     logits = None
    #     # Adjust loop to generate tokens up to input_seq_len
    #     for _ in range(input_seq_len - 1):  # Adjust for the [CLS] token
    #         if generated.size(1) == input_seq_len:  # Stop if sequence length is reached
    #             break
    #         outputs = self.decoder(input_ids=generated, encoder_hidden_states=z)
    #         hidden_states = outputs.last_hidden_state
    #         next_token_logits = self.linear_layer(hidden_states[:, -1, :]).unsqueeze( 1 )
    #         next_token = torch.argmax(next_token_logits, dim=-1)
    #         generated = torch.cat((generated, next_token), dim=-1)
    #         if logits is None:
    #             logits = next_token_logits # Dimensions: batch, token, vocabulary
    #         else:
    #             logits = torch.cat( ( logits, next_token_logits ), dim = 1 ) 

    #     if self.training:
    #         logits_prob = torch.softmax(logits, dim=2)
    #     return logits_prob


    def logits_to_tokens( self, logits ):
        return torch.argmax( logits, dim=2)
        

    def forward(self, input_ids, attention_mask):
        #print("Shape of input_ids in forward:", input_ids.shape)  # Add this line
        mu, logvar = self.encode(input_ids, attention_mask)
        z = self.reparameterize(mu, logvar)
        rejected = self.sigmoid(self.rejection_classifier( z ))
        input_seq_len = input_ids.size(1)
        #reconstructed = self.decode(z, input_seq_len)
        # During training, when calling the decode method
        reconstructed = self.decode(z, input_seq_len, temperature=0.7)  # Example temperature

        return reconstructed, mu, logvar, rejected

    
    def to(self, device):
        super().to(device)
        self.encoder.to(device)
        self.decoder.to(device)
        self.fc_mu.to(device)
        self.fc_logvar.to(device)
        self.linear_layer.to(device)  # Don't forget to move this layer as well
        self.rejection_classifier.to(device)
        return self

    def vae_loss(self, reconstructed, input_ids, mu, logvar):
        # Align sequence lengths: pad or truncate `reconstructed` to match `input_ids`
        seq_len = input_ids.size(1)
        current_len = reconstructed.size(1)

        if current_len < seq_len:
            # Pad reconstructed if it's shorter
            padding = torch.zeros((reconstructed.size(0), seq_len - current_len, reconstructed.size(2)), device=reconstructed.device)
            reconstructed = torch.cat([reconstructed, padding], dim=1)
        elif current_len > seq_len:
            # Truncate reconstructed if it's longer
            reconstructed = reconstructed[:, :seq_len, :]

        # Flatten tensors for loss computation
        recon_loss = F.cross_entropy(
            reconstructed.reshape(-1, self.tokenizer.vocab_size),  # [batch_size * seq_len, vocab_size]
            input_ids.reshape(-1),  # [batch_size * seq_len]
            reduction='sum'
        )

        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss, kl_div


class MyDataset(Dataset):
    def __init__(self, texts, rejected, tokenizer, max_length):
        self.texts = texts
        self.rejected = rejected
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        return inputs.input_ids.squeeze(0), inputs.attention_mask.squeeze(0), self.rejected[idx]

# Check if MPS is available and use it; otherwise, use CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model_name = 'bert-base-uncased'
latent_dim = 512  # Example latent dimension size
vae = TransformerVAE(model_name, latent_dim).to(device)

optimizer = torch.optim.Adam(vae.parameters(), lr=1e-5)

# Load and preprocess labeled refusals into train/val/test.
X, y = preprocess_data('all_hand_labeled.json', 'response')
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# Prepare the dataset and dataloader
# texts = [
#     "Bob is silly",
#     "I like apples"
# ] * 20  # Ensure you have enough texts

num_epochs = 2000

texts =   [X_train[i][25:89] for i in range(len(X_train))][:3] #if y_train[i] == 1][:3]
classes = torch.tensor( [ 1. ] * 50 ).float().reshape( ( -1, 1 ) )
#------------------------------------------------
# Use this for thorough training with both classes (hard)
#------------------------------------------------
# texts = X_train[:]
# classes = torch.tensor( y_train ).float().reshape( ( -1, 1 ) )
#------------------------------------------------

# filter to short refusals
print(f"{len(texts)} texts, example: {texts[0]}")

dataset = MyDataset(texts, classes, vae.tokenizer, max_length=16)

# Adjust batch size if necessary
batch_size = min(2, len(dataset))  # Ensure batch size is not larger than dataset
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False) 

bceLoss = BCELoss()

# Initialize lists to store losses for plotting
recon_losses = []
kl_divergences = []
bce_losses = []

# Define scaling weights
alpha = 0.5  # Weight for reconstruction loss
beta = 27/2   # Weight for KL divergence
gamma = 415/2  # Weight for classification loss

# Increment beta after a certain number of epochs
beta_increment = 27/2 * 0.05
annealing_interval = 10  # Increment beta every 10 epochs

# Training loop
vae.train()
for epoch in range(num_epochs):
    try:
        epoch_loss = 0.0  # Initialize epoch loss
        epoch_recon_loss = 0.0
        epoch_kl_div = 0.0
        epoch_bce_loss = 0.0
        num_batches = 0  # Initialize numberof batches processed

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

        # if epoch % 10 == 0:
        # Avoid division by zero
        if num_batches > 0:
            average_loss = epoch_loss / num_batches
            #print(f"Epoch {epoch}, Average Loss: {average_loss}")
        else:
            pass
            #print(f"Epoch {epoch}, No batches processed")

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

        # Reconstructed text
        print(f"Correct text:       { ' '.join( correct_tokens[0] ) }") 
        print(f"Predicted text:     { ' '.join( predicted_tokens[0] ) }" )

        # Predicted classification
        print(f"Classification was: {'CORRECT' if (classification.item() > 0.5 and rejected.item() == 1) or (classification.item() < 0.5 and rejected.item() == 0) else 'INCORRECT'} (Expected class:   {rejected.item()}, Got class: {classification.item()})")

        # Store losses for plotting
        recon_losses.append(epoch_recon_loss / num_batches)
        kl_divergences.append(epoch_kl_div / num_batches)
        bce_losses.append(epoch_bce_loss / num_batches)

        # Plotting and saving graphs
        epochs = range(1, epoch + 2)
        
        # After each epoch, plot all three losses in one image
        plt.figure(figsize=(10, 8))

        # First subplot for Binary Cross-Entropy Loss
        plt.subplot(2, 1, 1)
        plt.plot(epochs, recon_losses, label='Reconstruction Binary Cross-Entropy Loss', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Reconstruction Loss during Training')
        plt.legend()

        # Second subplot for KL Divergence Loss
        plt.subplot(2, 2, 3)
        plt.plot(epochs, kl_divergences, label='KL Divergence Loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('KL Divergence Loss during Training')
        plt.legend()

        # Third subplot for classification BCE Loss
        plt.subplot(2, 2, 4)
        plt.plot(epochs, bce_losses, label='Classification Binary Cross-Entropy', color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Classification BCE Loss during Training')
        plt.legend()

        # Save the figure
        plt.tight_layout()
        plt.savefig(f'all_losses.png')
        plt.close()

        print()

        # Update beta at the specified interval
        if (epoch + 1) % annealing_interval == 0:
            beta = min(beta + beta_increment, 27/2 * 2)  # Ensure beta does not exceed 1


    except KeyboardInterrupt as e:
        print( "WBS: Program interrupted, dropping to debug console.  Type 'c' to resume.")
        import pdb; pdb.set_trace()
