import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

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

        # Decoder can be another transformer or a different architecture
        self.decoder = AutoModel.from_pretrained(model_name)

    def encode(self, input_ids, attention_mask):
        encoded = self.encoder(input_ids, attention_mask=attention_mask)[0]
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, input_seq_len):
        generated = torch.full((z.size(0), 1), self.tokenizer.cls_token_id, dtype=torch.long, device=z.device)

        # Adjust loop to generate tokens up to input_seq_len
        for _ in range(input_seq_len - 1):  # Adjust for the [CLS] token
            if generated.size(1) == input_seq_len:  # Stop if sequence length is reached
                break
            outputs = self.decoder(input_ids=generated, encoder_hidden_states=z)
            hidden_states = outputs.last_hidden_state
            next_token_logits = self.linear_layer(hidden_states[:, -1, :])
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated = torch.cat((generated, next_token), dim=-1)

        logits = self.linear_layer(hidden_states[:, :input_seq_len, :])
        return logits


    def forward(self, input_ids, attention_mask):
        #print("Shape of input_ids in forward:", input_ids.shape)  # Add this line
        mu, logvar = self.encode(input_ids, attention_mask)
        z = self.reparameterize(mu, logvar)
        input_seq_len = input_ids.size(1)
        reconstructed = self.decode(z, input_seq_len)
        return reconstructed, mu, logvar

    
    def to(self, device):
        super().to(device)
        self.encoder.to(device)
        self.decoder.to(device)
        self.fc_mu.to(device)
        self.fc_logvar.to(device)
        self.linear_layer.to(device)  # Don't forget to move this layer as well
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
        return recon_loss + kl_div



class MyDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        return inputs.input_ids.squeeze(0), inputs.attention_mask.squeeze(0)

# Check if MPS is available and use it; otherwise, use CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model_name = 'bert-base-uncased'
latent_dim = 256  # Example latent dimension size
vae = TransformerVAE(model_name, latent_dim).to(device)

optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)

# Load and preprocess labeled refusals into train/val/test.
X, y = preprocess_data('all_hand_labeled.json', 'response')
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# Prepare the dataset and dataloader
# texts = [
#     "Bob is silly",
#     "I like apples"
# ] * 20  # Ensure you have enough texts

scale = 2
num_epochs = 400

i = 8
texts = X_train[i:i+scale]
print(texts)

dataset = MyDataset(texts, vae.tokenizer, max_length=32)

# Adjust batch size if necessary
batch_size = min(scale, len(dataset))  # Ensure batch size is not larger than dataset
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False) 

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0  # Initialize epoch loss
    num_batches = 0  # Initialize number of batches processed

    for input_ids, attention_mask in dataloader:
        num_batches += 1
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        optimizer.zero_grad()
        reconstructed, mu, logvar = vae(input_ids, attention_mask)
        loss = vae.vae_loss(reconstructed, input_ids, mu, logvar)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Avoid division by zero
    if num_batches > 0:
        average_loss = epoch_loss / num_batches
        print(f"Epoch {epoch}, Average Loss: {average_loss}")
    else:
        print(f"Epoch {epoch}, No batches processed")

    if epoch % 10 == 0:
        # Forward pass example
        input_ids, attention_mask = next(iter(dataloader))
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        reconstructed, mu, logvar = vae(input_ids, attention_mask)

        # Convert logits to probabilities and tokens
        probabilities = torch.softmax(reconstructed, dim=-1)
        predicted_token_ids = torch.argmax(probabilities, dim=-1)
        predicted_tokens = [vae.tokenizer.convert_ids_to_tokens(ids) for ids in predicted_token_ids]

        print(predicted_tokens)
